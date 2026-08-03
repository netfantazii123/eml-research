"""
Тесты движка инференса: тайлинг, оверлей, сохранение кадра.

Тайлинг — то место, где ошибка на один пиксель не видна глазом, но роняет
точность разметки: окно, съехавшее на пиксель, это уже не та вырезка, на
которой обучалась модель. Поэтому проверяется точное совпадение пикселей
после круга «мозаика -> тайлинг -> окна».
"""

import struct
import zlib

import numpy as np
import pytest

import config
import vision


# ── Тайлинг ───────────────────────────────────────────────────────────────────

def test_tile_positions_exact_fit():
    positions = vision.tile_positions(128, 192, tile=64, stride=64)
    assert positions == [(0, 0), (0, 64), (0, 128),
                         (64, 0), (64, 64), (64, 128)]


def test_tile_positions_drops_remainder():
    """Неполные окна отбрасываются: дополнять их выдуманными пикселями нельзя."""
    assert vision.tile_positions(100, 100, tile=64, stride=64) == [(0, 0)]


@pytest.mark.parametrize('height,width', [(63, 64), (64, 63), (10, 10), (0, 0),
                                          (63, 1000), (1000, 63)])
def test_tile_positions_small_frame_gives_nothing(height, width):
    """
    Кадр меньше окна — ни одной позиции.

    Раньше здесь возвращалась одна позиция (0, 0), после чего extract_tiles
    падал на несовпадении форм: вырезать 64x64 из кадра 40x30 нельзя. Видео с
    маленьким кадром роняло рабочий поток.
    """
    assert vision.tile_positions(height, width, tile=64) == []


def test_tile_positions_stride_overlap():
    positions = vision.tile_positions(64, 128, tile=64, stride=32)
    assert positions == [(0, 0), (0, 32), (0, 64)]


def test_small_frame_survives_full_path():
    """Сквозная проверка: маленький кадр не должен ронять конвейер разметки."""
    frame = np.zeros((40, 30, 3), dtype=np.uint8)
    positions = vision.tile_positions(frame.shape[0], frame.shape[1])
    assert positions == []
    tiles = vision.extract_tiles(frame, positions)
    assert tiles.shape == (0, config.IMG_SIZE, config.IMG_SIZE, 3)


# ── Мозаика и вырезка ─────────────────────────────────────────────────────────

def test_mosaic_tiling_roundtrip():
    """Из мозаики тайлинг обязан вернуть ровно те же вырезки, без сдвигов."""
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
    """Кадр меньше окна растягивается, а не падает."""
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


# ── Оверлей ───────────────────────────────────────────────────────────────────

def test_overlay_does_not_touch_source():
    """Оверлей возвращает новый массив: иначе повторный рендер «выжигает» кадр."""
    frame = np.full((128, 128, 3), 100, dtype=np.uint8)
    original = frame.copy()
    positions = vision.tile_positions(128, 128)
    preds = np.zeros(len(positions), dtype=int)
    out = vision.draw_overlay(frame, positions, preds)
    assert np.array_equal(frame, original)
    assert not np.array_equal(out, frame)
    assert out.dtype == np.uint8


def test_overlay_respects_confidence_threshold():
    """Окна ниже порога уверенности не закрашиваются."""
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
    """Вычитание максимума спасает от переполнения exp()."""
    probs = vision.softmax(np.array([[1e6, 0.0, -1e6, 5.0]]))
    assert np.all(np.isfinite(probs))
    assert probs.argmax() == 0


def test_softmax_floor_for_four_classes():
    """
    Максимум softmax по 4 классам не бывает меньше 0.25 — значит и порог
    уверенности ниже 0.25 ничего не фильтрует. Это оговорено в интерфейсе.
    """
    probs = vision.softmax(np.zeros((1, config.N_CLASSES)))
    assert probs.max() == pytest.approx(0.25)


# ── Сохранение кадра ──────────────────────────────────────────────────────────

def decode_png(blob: bytes) -> np.ndarray:
    """Минимальный декодер PNG для проверки записи (8 бит, truecolor)."""
    assert blob[:8] == b'\x89PNG\r\n\x1a\x0a'
    pos, chunks, size = 8, {}, None
    idat = b''
    while pos < len(blob):
        length, = struct.unpack('>I', blob[pos:pos + 4])
        tag = blob[pos + 4:pos + 8]
        payload = blob[pos + 8:pos + 8 + length]
        if tag == b'IHDR':
            width, height, depth, color = struct.unpack('>IIBB', payload[:10])
            assert (depth, color) == (8, 2), "ожидался 8-битный truecolor RGB"
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
        assert row[0] == 0, "ожидался фильтр 0 (без фильтрации)"
        rows.append(np.frombuffer(row[1:], dtype=np.uint8))
    return np.stack(rows).reshape(height, width, 3)


def test_write_image_keeps_exact_pixels(tmp_path):
    """
    Кадр сохраняется пиксель в пиксель.

    Раньше он прогонялся через matplotlib с dpi=100 и сохранялся с dpi=150,
    то есть на диск ложилась пересэмплированная картинка в полтора раза
    крупнее — не те пиксели, по которым пользователь оценивал разметку.
    """
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    path = str(tmp_path / 'frame.png')
    vision.write_image(path, frame)

    # Кто именно писал файл — cv2 или собственный кодировщик — неизвестно и
    # неважно; важно, что пиксели те же. Читаем тем, что есть: cv2 понимает
    # оба варианта, собственный декодер — только свой (он не умеет фильтры
    # PNG, которыми пользуется cv2).
    try:
        import cv2
        assert np.array_equal(cv2.imread(path)[:, :, ::-1], frame)
    except ImportError:
        assert np.array_equal(decode_png(open(path, 'rb').read()), frame)


def test_png_encoder_roundtrip():
    """Собственный кодировщик PNG проверяется независимо от наличия OpenCV."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (12, 9, 3), dtype=np.uint8)
    assert np.array_equal(decode_png(vision._png_bytes(frame)), frame)


def test_write_image_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError):
        vision.write_image(str(tmp_path / 'x.png'),
                           np.zeros((8, 8), dtype=np.uint8))


# ── Цвета классов ─────────────────────────────────────────────────────────────

def test_every_class_has_a_colour():
    seen = {vision.class_color(i) for i in range(config.N_CLASSES)}
    assert len(seen) == config.N_CLASSES
    for colour in seen:
        assert len(colour) == 3 and all(0 <= v <= 255 for v in colour)


def test_class_colors_stay_distinct_for_any_dataset(monkeypatch):
    """
    Классы чужой папки получают РАЗНЫЕ цвета, а не общий серый.

    Цвета подобраны вручную только для исходных четырёх имён. Раньше всё
    остальное падало в один запасной серый, и оверлей на новом датасете
    превращался в равномерную заливку, по которой ничего не разобрать —
    причём выглядело это как работающая разметка.
    """
    monkeypatch.setattr(config, 'CLASS_NAMES', ['кот', 'пёс', 'лиса', 'ёж', 'уж'])
    monkeypatch.setattr(config, 'N_CLASSES', 5)
    colors = [vision.class_color(i) for i in range(5)]
    assert len(set(colors)) == 5
    for rgb in colors:
        assert len(rgb) == 3 and all(0 <= v <= 255 for v in rgb)


def test_known_class_colors_are_unchanged():
    """Исходные четыре класса сохраняют свои цвета: по ним читаются старые отчёты."""
    for i, name in enumerate(config.CLASS_NAMES):
        assert vision.class_color(i) == vision.CLASS_COLORS[name]
