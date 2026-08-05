"""
Tests of the inference engine: tiling, overlay, frame saving.

Tiling is where a one-pixel error is invisible to the eye but destroys the
annotation accuracy: a window shifted by a pixel is no longer the crop the model
was trained on. Exact pixel agreement after the "mosaic -> tiling -> windows"
round trip is therefore checked.
"""

import struct
import zlib

import numpy as np
import pytest

import config
import vision


# ── Tiling ────────────────────────────────────────────────────────────────────

def test_tile_positions_exact_fit():
    positions = vision.tile_positions(128, 192, tile=64, stride=64)
    assert positions == [(0, 0), (0, 64), (0, 128),
                         (64, 0), (64, 64), (64, 128)]


def test_tile_positions_drops_remainder():
    """Incomplete windows are discarded: padding them with invented pixels is not
    allowed."""
    assert vision.tile_positions(100, 100, tile=64, stride=64) == [(0, 0)]


@pytest.mark.parametrize('height,width', [(63, 64), (64, 63), (10, 10), (0, 0),
                                          (63, 1000), (1000, 63)])
def test_tile_positions_small_frame_gives_nothing(height, width):
    """
    A frame smaller than the window — not a single position.

    Previously one position (0, 0) was returned here, after which extract_tiles
    failed on a shape mismatch: a 64x64 window cannot be cut out of a 40x30
    frame. Video with a small frame killed the worker thread.
    """
    assert vision.tile_positions(height, width, tile=64) == []


def test_tile_positions_stride_overlap():
    positions = vision.tile_positions(64, 128, tile=64, stride=32)
    assert positions == [(0, 0), (0, 32), (0, 64)]


def test_small_frame_survives_full_path():
    """End-to-end check: a small frame must not break the annotation pipeline."""
    frame = np.zeros((40, 30, 3), dtype=np.uint8)
    positions = vision.tile_positions(frame.shape[0], frame.shape[1])
    assert positions == []
    tiles = vision.extract_tiles(frame, positions)
    assert tiles.shape == (0, config.IMG_SIZE, config.IMG_SIZE, 3)


# ── Mosaic and cropping ───────────────────────────────────────────────────────

def test_mosaic_tiling_roundtrip():
    """From a mosaic, tiling must return exactly the same crops, without shifts."""
    rng = np.random.default_rng(0)
    rows, cols = 3, 4
    crops = rng.integers(0, 256, (rows * cols, config.IMG_SIZE,
                                  config.IMG_SIZE, 3), dtype=np.uint8)
    frame = vision.mosaic(crops, rows, cols)
    assert frame.shape == (rows * config.IMG_SIZE, cols * config.IMG_SIZE, 3)

    positions = vision.tile_positions(frame.shape[0], frame.shape[1])
    assert len(positions) == rows * cols
    assert np.array_equal(vision.extract_tiles(frame, positions), crops)


def test_center_crop_takes_center():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[32:96, 32:96] = 255
    assert np.all(vision.center_crop(frame, 64) == 255)


def test_center_crop_upscales_small_frame():
    """A frame smaller than the window is upscaled rather than failing."""
    frame = np.full((10, 20, 3), 7, dtype=np.uint8)
    crop = vision.center_crop(frame, 64)
    assert crop.shape == (64, 64, 3)
    assert np.all(crop == 7)


def test_resize_nearest_keeps_values():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    big = vision.resize_nearest(frame, 32, 32)
    assert big.shape == (32, 32, 3)
    assert set(np.unique(big).tolist()) <= set(np.unique(frame).tolist())


# ── Overlay ───────────────────────────────────────────────────────────────────

def test_overlay_does_not_touch_source():
    """The overlay returns a new array: otherwise a repeated render washes the
    frame out."""
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    original = frame.copy()
    positions = vision.tile_positions(128, 128)
    preds = np.zeros(len(positions), dtype=int)
    out = vision.draw_overlay(frame, positions, preds)
    assert np.array_equal(frame, original)
    assert not np.array_equal(out, frame)
    assert out.dtype == np.uint8


def test_overlay_respects_confidence_threshold():
    """Windows below the confidence threshold are not filled."""
    frame = np.full((64, 128, 3), 100, dtype=np.uint8)
    positions = vision.tile_positions(64, 128)
    preds = np.zeros(len(positions), dtype=int)
    conf = np.array([0.9, 0.1])
    out = vision.draw_overlay(frame, positions, preds, conf, conf_threshold=0.5)
    assert not np.array_equal(out[:, :64], frame[:, :64])
    assert np.array_equal(out[:, 64:], frame[:, 64:])


# ── Softmax ───────────────────────────────────────────────────────────────────

def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    probs = vision.softmax(rng.normal(scale=30, size=(10, config.N_CLASSES)))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs >= 0)


def test_softmax_survives_huge_logits():
    """Subtracting the maximum saves from exp() overflow."""
    probs = vision.softmax(np.array([[1e6, 0.0, -1e6, 5.0]]))
    assert np.all(np.isfinite(probs))
    assert probs.argmax() == 0


def test_softmax_floor_for_four_classes():
    """
    The softmax maximum over 4 classes is never below 0.25 — so a confidence
    threshold below 0.25 filters nothing. This is stated in the interface.
    """
    probs = vision.softmax(np.zeros((1, config.N_CLASSES)))
    assert probs.max() == pytest.approx(0.25)


# ── Frame saving ──────────────────────────────────────────────────────────────

def decode_png(blob: bytes) -> np.ndarray:
    """Minimal PNG decoder for checking the write (8 bit, truecolor)."""
    assert blob[:8] == b'\x89PNG\r\n\x1a\x0a'
    pos, chunks, size = 8, {}, None
    idat = b''
    while pos < len(blob):
        length, = struct.unpack('>I', blob[pos:pos + 4])
        tag = blob[pos + 4:pos + 8]
        payload = blob[pos + 8:pos + 8 + length]
        if tag == b'IHDR':
            width, height, depth, color = struct.unpack('>IIBB', payload[:10])
            assert (depth, color) == (8, 2), "expected 8-bit truecolor RGB"
            size = (height, width)
        elif tag == b'IDAT':
            idat += payload
        chunks[tag] = payload
        pos += 12 + length
    assert b'IEND' in chunks

    height, width = size
    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    rows = []
    for y in range(height):
        row = raw[y * stride:(y + 1) * stride]
        assert row[0] == 0, "expected filter 0 (no filtering)"
        rows.append(np.frombuffer(row[1:], dtype=np.uint8))
    return np.stack(rows).reshape(height, width, 3)


def test_write_image_keeps_exact_pixels(tmp_path):
    """
    The frame is saved pixel for pixel.

    Previously it went through matplotlib with dpi=100 and was saved with
    dpi=150, i.e. a resampled image one and a half times larger landed on disk —
    not the pixels the user judged the annotation by.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    path = str(tmp_path / 'frame.png')
    vision.write_image(path, frame)

    # Which of the two wrote the file — cv2 or the own encoder — is unknown and
    # unimportant; what matters is that the pixels are the same. We read with
    # whatever is available: cv2 understands both variants, the own decoder only
    # its own (it does not handle the PNG filters cv2 uses).
    try:
        import cv2
        assert np.array_equal(cv2.imread(path)[:, :, ::-1], frame)
    except ImportError:
        assert np.array_equal(decode_png(open(path, 'rb').read()), frame)


def test_png_encoder_roundtrip():
    """The own PNG encoder is checked independently of OpenCV's presence."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (12, 9, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(vision._png_bytes(frame)), frame)


def test_write_image_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        vision.write_image(str(tmp_path / 'x.png'),
                           np.zeros((8, 8), dtype=np.uint8))


# ── Class colors ──────────────────────────────────────────────────────────────

def test_every_class_has_a_colour():
    seen = {vision.class_color(i) for i in range(config.N_CLASSES)}
    assert len(seen) == config.N_CLASSES
    for colour in seen:
        assert len(colour) == 3 and all(0 <= v <= 255 for v in colour)


def test_class_colors_stay_distinct_for_any_dataset(monkeypatch):
    """
    Classes of a foreign folder get DIFFERENT colors, not a shared gray.

    The colors are hand-picked only for the original four names. Previously
    everything else fell back to one gray, and the overlay on a new dataset
    turned into a uniform fill showing nothing — while looking like working
    annotation.
    """
    monkeypatch.setattr(config, 'CLASS_NAMES',
                        ['cat', 'dog', 'fox', 'hedgehog', 'snake'])
    monkeypatch.setattr(config, 'N_CLASSES', 5)
    colors = [vision.class_color(i) for i in range(5)]
    assert len(set(colors)) == 5
    for rgb in colors:
        assert len(rgb) == 3 and all(0 <= v <= 255 for v in rgb)


def test_known_class_colors_are_unchanged():
    """The original four classes keep their colors: old reports are read by them."""
    for i, name in enumerate(config.CLASS_NAMES):
        assert vision.class_color(i) == vision.CLASS_COLORS[name]
