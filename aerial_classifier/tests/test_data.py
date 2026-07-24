"""
Тесты парсера BMP и сплитов.

Парсер BMP написан вручную (data.read_bmp), и это самое хрупкое место
загрузки: ошибка в отступе строки или в порядке строк не падает, а молча
портит картинки — обучение потом идёт на мусоре с виду нормальной формы.
Поэтому BMP здесь собираются в самом тесте, байт за байтом, во всех
подформатах, которые парсер обещает понимать.
"""

import struct

import numpy as np
import pytest

import config
import data


# ── Сборка BMP в тесте ────────────────────────────────────────────────────────

def make_bmp(rgb: np.ndarray, top_down: bool = False, bpp: int = 24,
             compression: int = 0) -> bytes:
    """
    Собрать 24-битный BMP из массива (H, W, 3) uint8 RGB.

    top_down=False — строки снизу вверх (как в датасете), True — сверху вниз
    (отрицательная высота в заголовке).
    """
    height, width = rgb.shape[:2]
    row_bytes = width * 3
    padded = (row_bytes + 3) & ~3
    pad = b'\x00' * (padded - row_bytes)

    rows = [rgb[y][:, ::-1].tobytes() + pad          # RGB -> BGR
            for y in range(height)]
    if not top_down:
        rows = rows[::-1]
    pixels = b''.join(rows)

    offset = 14 + 40
    file_header = b'BM' + struct.pack('<IHHI', offset + len(pixels), 0, 0, offset)
    info_header = struct.pack('<IiiHHIIiiII', 40, width,
                              -height if top_down else height,
                              1, bpp, compression, len(pixels), 2835, 2835, 0, 0)
    return file_header + info_header + pixels


def write_bmp(tmp_path, name: str, rgb: np.ndarray, **kwargs) -> str:
    path = tmp_path / name
    path.write_bytes(make_bmp(rgb, **kwargs))
    return str(path)


# ── Парсер ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('top_down', [False, True])
def test_read_bmp_roundtrip(tmp_path, top_down):
    """Пиксели читаются ровно теми же и в обоих порядках строк."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (7, 12, 3), dtype=np.uint8)
    got = data.read_bmp(write_bmp(tmp_path, 'a.bmp', rgb, top_down=top_down))
    assert got.shape == rgb.shape
    assert np.array_equal(got, rgb)


@pytest.mark.parametrize('width', [1, 2, 3, 4, 5, 6, 7, 8])
def test_read_bmp_row_padding(tmp_path, width):
    """
    Строка BMP выровнена до кратности 4 байтам.

    При ширине, не кратной 4, между строками есть байты-заполнители. Если их
    не отбросить, картинка «поедет» вбок нарастающим сдвигом — самая коварная
    поломка парсера, потому что первая строка при ней выглядит правильно.
    """
    rng = np.random.default_rng(width)
    rgb = rng.integers(0, 256, (5, width, 3), dtype=np.uint8)
    got = data.read_bmp(write_bmp(tmp_path, f'w{width}.bmp', rgb))
    assert np.array_equal(got, rgb)


def test_read_bmp_channel_order(tmp_path):
    """BMP хранит BGR; парсер обязан вернуть RGB, а не перепутать каналы."""
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 200        # красный
    rgb[..., 1] = 100        # зелёный
    rgb[..., 2] = 50         # синий
    got = data.read_bmp(write_bmp(tmp_path, 'rgb.bmp', rgb))
    assert got[0, 0].tolist() == [200, 100, 50]


def test_read_bmp_rejects_foreign_format(tmp_path):
    """Чужой подформат — падение, а не молча испорченные данные."""
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        data.read_bmp(write_bmp(tmp_path, 'p8.bmp', rgb, bpp=8))
    with pytest.raises(ValueError):
        data.read_bmp(write_bmp(tmp_path, 'rle.bmp', rgb, compression=1))

    not_bmp = tmp_path / 'x.bmp'
    not_bmp.write_bytes(b'PNG\x00' + b'\x00' * 60)
    with pytest.raises(ValueError):
        data.read_bmp(str(not_bmp))


# ── Сплиты ────────────────────────────────────────────────────────────────────

def make_labels() -> np.ndarray:
    """Несбалансированные метки, как в настоящем датасете (примерно 3.4:1)."""
    counts = [61, 92, 130, 206]
    return np.concatenate([np.full(n, c, dtype=np.int64)
                           for c, n in enumerate(counts)])


def test_stratified_split_is_a_partition():
    """Три части не пересекаются и вместе покрывают датасет целиком."""
    labels = make_labels()
    train, val, test = data.stratified_split(labels, seed=1337)

    joined = np.concatenate([train, val, test])
    assert len(joined) == len(labels)
    assert len(np.unique(joined)) == len(labels)
    assert np.array_equal(np.sort(joined), np.arange(len(labels)))
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(a.tolist()) & set(b.tolist())


def test_stratified_split_is_deterministic():
    """Один сид — один и тот же сплит; иначе метрики между прогонами несравнимы."""
    labels = make_labels()
    first = data.stratified_split(labels, seed=7)
    second = data.stratified_split(labels, seed=7)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)

    other = data.stratified_split(labels, seed=8)
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_stratified_split_keeps_class_proportions():
    """
    Доли классов сохраняются в каждой части.

    Ради этого стратификация и нужна: при случайном сплите редкий класс может
    почти исчезнуть из валидации, и метрика поедет.
    """
    labels = make_labels()
    train, val, test = data.stratified_split(labels, seed=1337)
    overall = np.bincount(labels, minlength=config.N_CLASSES) / len(labels)
    for part in (train, val, test):
        share = (np.bincount(labels[part], minlength=config.N_CLASSES)
                 / len(part))
        assert np.allclose(share, overall, atol=0.02)


def test_stratified_split_sizes_match_config():
    labels = make_labels()
    train, val, _ = data.stratified_split(labels, seed=1337)
    assert len(train) / len(labels) == pytest.approx(config.SPLIT_TRAIN, abs=0.01)
    assert len(val) / len(labels) == pytest.approx(config.SPLIT_VAL, abs=0.01)


def test_class_weights_compensate_imbalance():
    """Вес обратно пропорционален частоте, среднее веса = 1."""
    labels = make_labels()
    w = data.class_weights(labels)
    assert len(w) == config.N_CLASSES
    assert w.mean() == pytest.approx(1.0, abs=1e-6)
    counts = np.bincount(labels, minlength=config.N_CLASSES)
    assert w[np.argmin(counts)] > w[np.argmax(counts)]
    # Вес * частота одинаков для всех классов — в этом и смысл 1/freq.
    products = w * counts
    assert np.allclose(products, products[0], rtol=1e-6)
