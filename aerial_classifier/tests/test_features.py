"""
Тесты экстрактора признаков.

Проверяются три вещи, каждая из которых при поломке НЕ падает сама по себе:

  1. Все признаки лежат в [-1, 1]. EML-оператор содержит exp(), и признак,
     вылетевший за диапазон, даёт формуле inf — не ошибку, а тихо испорченный
     результат.
  2. Поворотно-инвариантные признаки действительно инвариантны. На этом
     построен отбор признаков в дипломе (grad_vert/grad_horz выброшены именно
     потому, что зависели от ориентации кадра) — утверждение надо проверять,
     а не декларировать.
  3. Отпечаток кэша меняется при правке констант. Без этого правка масштаба
     молча не доезжает до обучения.
"""

import numpy as np
import pytest

import config
import features


# ── Наборы тестовых картинок ──────────────────────────────────────────────────

def sample_images() -> dict:
    """Картинки разного характера — от вырожденных до похожих на классы."""
    rng = np.random.default_rng(0)
    size = config.IMG_SIZE
    out = {}

    out['шум'] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    out['чёрная'] = np.zeros((size, size, 3), dtype=np.uint8)
    out['белая'] = np.full((size, size, 3), 255, dtype=np.uint8)
    out['серая'] = np.full((size, size, 3), 128, dtype=np.uint8)

    # Полосы: грубая имитация дороги — одно доминирующее направление.
    stripes = np.zeros((size, size, 3), dtype=np.uint8)
    stripes[:, ::4] = 255
    out['полосы'] = stripes

    # Плавный градиент — гладкий фон без текстуры.
    ramp = np.linspace(0, 255, size, dtype=np.uint8)
    out['градиент'] = np.repeat(np.tile(ramp, (size, 1))[:, :, None], 3, axis=2)

    # Яркое пятно в центре — «объект в кадре».
    spot = np.full((size, size, 3), 40, dtype=np.uint8)
    spot[24:40, 24:40] = 220
    out['пятно'] = spot
    return out


def transforms():
    """Восемь ориентаций кадра: у аэроснимка нет верха и низа."""
    return {
        'rot90': lambda a: np.rot90(a, 1, axes=(0, 1)),
        'rot180': lambda a: np.rot90(a, 2, axes=(0, 1)),
        'rot270': lambda a: np.rot90(a, 3, axes=(0, 1)),
        'fliplr': lambda a: a[:, ::-1],
        'flipud': lambda a: a[::-1],
    }


# ── Диапазон и форма ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(sample_images()))
def test_features_stay_in_range(name):
    """Каждый признак в [-1, 1] и конечен — на любой картинке, включая вырожденные."""
    feats = features.extract(sample_images()[name])
    assert feats.shape == (config.N_FEATURES,)
    assert np.all(np.isfinite(feats)), f"нефинитные признаки на «{name}»"
    assert feats.min() >= -1.0 - 1e-6, f"признак ниже -1 на «{name}»"
    assert feats.max() <= 1.0 + 1e-6, f"признак выше +1 на «{name}»"


def test_batch_matches_single():
    """extract_batch и extract считают одно и то же."""
    images = np.stack(list(sample_images().values()))
    batch = features.extract_batch(images)
    for i, image in enumerate(images):
        assert np.allclose(batch[i], features.extract(image), atol=1e-6)


def test_batch_chunking_is_transparent():
    """Разбиение на куски не влияет на результат."""
    images = np.stack(list(sample_images().values()))
    assert np.allclose(features.extract_batch(images, chunk=2),
                       features.extract_batch(images, chunk=512), atol=1e-6)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        features.extract_batch(np.zeros((2, 64, 64), dtype=np.uint8))


# ── Поворотная инвариантность ─────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(sample_images()))
def test_radial_slope_is_rotation_invariant(name):
    """
    radial_slope не зависит от ориентации кадра.

    Он считается через расстояние до центра, а карта расстояний симметрична
    относительно поворотов на 90° и отражений, поэтому совпадение здесь
    точное — с точностью до порядка суммирования float32.
    """
    idx = config.FEATURE_NAMES.index('radial_slope')
    image = sample_images()[name]
    base = features.extract(image)[idx]
    for label, transform in transforms().items():
        got = features.extract(np.ascontiguousarray(transform(image)))[idx]
        assert got == pytest.approx(base, abs=1e-5), f"{name}/{label}"


@pytest.mark.parametrize('name', list(sample_images()))
def test_coherence_is_rotation_invariant(name):
    """
    coherence не зависит от ориентации кадра.

    Точного равенства тут не бывает и не может быть: градиенты считаются на
    сетке 63x63, то есть последние строка и столбец в неё не попадают, и при
    повороте выпадают ДРУГИЕ пиксели. На структурных картинках расхождение
    нулевое, на чистом шуме доходит до ~0.03 — это разница выборки, а не
    зависимость от ориентации, поэтому допуск 0.05.
    """
    idx = config.FEATURE_NAMES.index('coherence')
    image = sample_images()[name]
    base = features.extract(image)[idx]
    for label, transform in transforms().items():
        got = features.extract(np.ascontiguousarray(transform(image)))[idx]
        assert got == pytest.approx(base, abs=0.05), f"{name}/{label}"


def test_coherence_separates_lines_from_noise():
    """
    Главное утверждение диплома про признаки: у дорог одно доминирующее
    направление, у песка и толпы — нет. Полосы должны давать когерентность
    заметно выше, чем шум.
    """
    idx = config.FEATURE_NAMES.index('coherence')
    images = sample_images()
    lines = features.extract(images['полосы'])[idx]
    noise = features.extract(images['шум'])[idx]
    assert lines > noise + 0.5


def test_center_features_react_to_center():
    """Признаки центр-периферия должны отличать кадр с объектом от однородного."""
    images = sample_images()
    spot = features.extract(images['пятно'])
    flat = features.extract(images['серая'])
    for name in ('center_gray_diff', 'center_edge_diff'):
        idx = config.FEATURE_NAMES.index(name)
        assert abs(spot[idx]) > abs(flat[idx]) + 0.05, name


# ── Масштабы и отпечаток кэша ─────────────────────────────────────────────────

def test_center_edge_has_its_own_scale():
    """
    center_edge_diff нормируется своей константой, а не _SCALE_GRAD_MAG.

    Разность градиентов центр-периферия вчетверо мельче самого градиента; на
    общем масштабе признак занимал четверть диапазона [-1, 1]. Тест держит
    исправление на месте.
    """
    assert features._SCALE_CENTER_EDGE != features._SCALE_GRAD_MAG
    assert 'center_edge_diff' in features.EMPIRICALLY_SCALED


def test_fingerprint_reacts_to_scale_change(monkeypatch):
    """
    Отпечаток обязан меняться при правке любой константы масштаба.

    Иначе кэш признаков (форма которого от масштаба не зависит) остаётся
    «валидным», обучение идёт на старых числах, и правка выглядит как
    не давшая эффекта.
    """
    before = features.fingerprint()
    monkeypatch.setattr(features, '_SCALE_CENTER_EDGE', 33.0)
    assert features.fingerprint() != before


def test_fingerprint_reacts_to_threshold_change(monkeypatch):
    before = features.fingerprint()
    monkeypatch.setattr(features, 'EDGE_THRESHOLD', 42.0)
    assert features.fingerprint() != before


def test_fingerprint_is_stable():
    """Без правок отпечаток не меняется — иначе кэш пересчитывался бы всегда."""
    assert features.fingerprint() == features.fingerprint()


# ── F-статистика ──────────────────────────────────────────────────────────────

def test_anova_f_is_scale_invariant():
    """
    F-статистика не меняется при аффинном масштабировании признака.

    Из этого следует важное для диплома: правка констант _SCALE_* не может
    изменить таблицу F-статистик — она влияет только на EML-формулы.
    """
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(config.N_CLASSES), 30)
    X = rng.normal(size=(len(y), 5)) + y[:, None] * 0.7

    base = features.anova_f(X, y)
    scaled = features.anova_f(X * 4.0 - 1.5, y)
    assert np.allclose(base, scaled, rtol=1e-6)


def test_anova_f_ranks_informative_feature_first():
    rng = np.random.default_rng(1)
    y = np.repeat(np.arange(config.N_CLASSES), 50)
    informative = y + rng.normal(scale=0.1, size=len(y))
    noise = rng.normal(size=len(y))
    scores = features.anova_f(np.stack([noise, informative], axis=1), y)
    assert scores[1] > scores[0]
