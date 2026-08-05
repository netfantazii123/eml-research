"""
Tests of image reading and conversion to the model input.

They catch exactly the breakages that do NOT fail on their own: a PNG decoder
with a confused row filter returns an image of the right shape and plausible
colors, and decimation-based scaling returns an image of the right size. Both
would flow on into the features and would be noticed, at best, as "the model
somehow trains worse".

The PNGs here are assembled byte by byte, in all five filter types. The project's
own encoder (vision.py) writes only filter 0, so on real project files the
Sub/Up/Average/Paeth branches never execute — while in a foreign folder they
appear immediately.
"""

import struct
import zlib

import numpy as np
import pytest

import imaging


# ── Assembling a PNG in the test ──────────────────────────────────────────────

def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _filter_row(raw: bytes, prev: bytes, ftype: int, bpp: int) -> bytes:
    if ftype == 0:
        return raw
    out = bytearray(len(raw))
    for i, value in enumerate(raw):
        a = raw[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ftype == 1:
            pred = a
        elif ftype == 2:
            pred = b
        elif ftype == 3:
            pred = (a + b) >> 1
        else:
            pred = _paeth(a, b, c)
        out[i] = (value - pred) & 0xFF
    return bytes(out)


def make_png(pixels: np.ndarray, color_type: int = 2, ftype: int = 0) -> bytes:
    """Assemble a PNG (8 bit, non-interlaced) with the given row filter."""
    height, width = pixels.shape[:2]
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    stride = width * channels

    body, prev = bytearray(), b'\x00' * stride
    for y in range(height):
        raw = pixels[y].tobytes()
        body.append(ftype)
        body += _filter_row(raw, prev, ftype, channels)
        prev = raw

    def chunk(tag: bytes, payload: bytes) -> bytes:
        blob = tag + payload
        return (struct.pack('>I', len(payload)) + blob
                + struct.pack('>I', zlib.crc32(blob) & 0xFFFFFFFF))

    ihdr = struct.pack('>IIBBBBB', width, height, 8, color_type, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr)
            + chunk(b'IDAT', zlib.compress(bytes(body), 6))
            + chunk(b'IEND', b''))


def write(tmp_path, name: str, blob: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(blob)
    return str(path)


# ── PNG ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('ftype', [0, 1, 2, 3, 4])
def test_read_png_all_filters(tmp_path, ftype):
    """
    All five row filters are reconstructed exactly.

    A filter is a difference with the neighbor to the left/above; an error in
    the predictor yields not garbage but pixels shifted in brightness that still
    look quite "image-like". The comparison here is therefore bitwise.
    """
    rng = np.random.default_rng(ftype)
    rgb = rng.integers(0, 256, (9, 7, 3), dtype=np.uint8)
    got = imaging.read_png(write(tmp_path, f'f{ftype}.png',
                                 make_png(rgb, ftype=ftype)))
    assert got.shape == rgb.shape
    assert np.array_equal(got, rgb)


def test_read_png_matches_own_encoder(tmp_path):
    """The decoder here and the encoder in vision.py must be mutual inverses."""
    import vision

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (12, 5, 3), dtype=np.uint8)
    got = imaging.read_png(write(tmp_path, 'v.png', vision._png_bytes(rgb)))
    assert np.array_equal(got, rgb)


def test_read_png_grayscale_becomes_three_channels(tmp_path):
    """A grayscale image spreads over three channels instead of breaking the shape."""
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    got = imaging.read_png(write(tmp_path, 'g.png',
                                 make_png(gray[:, :, None], color_type=0)))
    assert got.shape == (4, 4, 3)
    assert np.array_equal(got[:, :, 0], gray)
    assert np.array_equal(got[:, :, 0], got[:, :, 2])


def test_read_png_rejects_interlaced(tmp_path):
    """Adam7 is unsupported — a clear failure, not half an image."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    blob = bytearray(make_png(rgb))
    blob[8 + 8 + 12] = 1                     # the interlace byte in IHDR
    with pytest.raises(ValueError, match='nterlacing'):
        imaging.read_png(write(tmp_path, 'i.png', bytes(blob)))


def test_read_image_dispatches_by_extension(tmp_path):
    """BMP goes through its parser, PNG through its own; both give equal pixels."""
    from test_data import make_bmp

    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    from_bmp = imaging.read_image(write(tmp_path, 'a.bmp', make_bmp(rgb)))
    from_png = imaging.read_image(write(tmp_path, 'a.png', make_png(rgb)))
    assert np.array_equal(from_bmp, rgb)
    assert np.array_equal(from_png, rgb)


# ── Conversion to the model input ────────────────────────────────────────────

def test_center_crop_takes_the_middle():
    """The crop is taken from the center, not a corner: the subject is usually there."""
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    img[:, 5:15] = 255
    assert (imaging.center_crop_square(img) == 255).all()


def test_resize_preserves_constant_image():
    """A uniform image stays uniform after scaling."""
    img = np.full((30, 30, 3), 137, dtype=np.uint8)
    for size in (8, 30, 64):
        out = imaging.resize(img, size)
        assert out.shape == (size, size, 3)
        assert (out == 137).all()


def test_downscale_averages_instead_of_dropping_pixels():
    """
    Downscaling averages over area rather than taking every n-th pixel.

    On a 1x1 checkerboard, decimation would return solid black or solid white
    (depending on parity), while averaging returns an even gray. The difference
    matters: on a decimated image the texture features (block_std, coherence)
    measure a scaling artifact rather than the texture.
    """
    board = np.indices((32, 32)).sum(axis=0) % 2
    img = (board * 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)
    out = imaging.resize(img, 8)
    assert out.min() >= 120 and out.max() <= 135


def test_upscale_is_smooth():
    """Upscaling is bilinear: intermediate values appear between two levels."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, 2:] = 255
    out = imaging.resize(img, 16)
    assert out.shape == (16, 16, 3)
    assert ((out > 20) & (out < 235)).any()


def test_fit_to_size_normalizes_anything():
    """Any size and any proportions -> exactly (IMG_SIZE, IMG_SIZE, 3) uint8."""
    rng = np.random.default_rng(3)
    for shape in ((17, 200, 3), (200, 17, 3), (64, 64, 3), (5, 5, 3)):
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        out = imaging.fit_to_size(img, 64)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.uint8


def test_fit_to_size_does_not_stretch():
    """
    Proportions are not distorted: first a square crop, then scaling.

    Stretching an elongated frame would change the texture scale along one axis
    — exactly the quantity anisotropy and coherence measure.
    """
    img = np.zeros((20, 60, 3), dtype=np.uint8)
    img[:, 20:40] = 255                       # the central square is white
    assert (imaging.fit_to_size(img, 8) == 255).all()
