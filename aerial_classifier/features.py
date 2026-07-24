"""
features.py — экстрактор инженерных признаков из картинки 64x64 RGB.

Это ключевой модуль ветки A. Идея та же, что в tetris/features.py: оракул и
EML-формула едят ОДИН И ТОТ ЖЕ вектор признаков. Тогда после дистилляции от
нейросети не остаётся ничего — на микроконтроллере работают только этот
экстрактор и 4 формулы.

Поэтому каждый признак выбирался по двум критериям сразу:
  1. различает классы (BEACH / MOTO / PEOPLE / ROADS);
  2. считается за один-два прохода по картинке простой арифметикой.

Ничего вроде HOG, SIFT или БПФ здесь нет сознательно — они дают лучшую
разделимость, но на STM32 это уже не «дешёвый препроцессинг».

32 признака, порядок = config.FEATURE_NAMES:

    Цвет (14)      — моменты каналов RGB, яркости, оппонентных осей rg/yb
                     и насыщенности. Ловят «песок жёлтый и светлый, асфальт
                     серый и тёмный».
    Текстура (8)   — модули градиентов, плотность краёв, анизотропия,
                     когерентность структурного тензора, энтропия яркости.
                     Когерентность — главный признак ДОРОГ: у них выражено
                     одно доминирующее направление, у пляжа и толпы — нет.
    Блоки 8x8 (2)  — шероховатость внутри блоков и разброс между блоками:
                     разделяет «гладкий крупный фон» и «мелкий частый детрит».
    Раскладка (8)  — центр против периферии, верх против низа, квадранты,
                     доли тёмных/ярких пикселей. Картинки — вырезки вокруг
                     объекта, поэтому «объект в центре» сам по себе сигнал.

Все признаки нормализованы в [-1, 1]: EML-оператор содержит exp(), и подавать
в него сырые величины масштаба 0..255 нельзя — вылетит в inf (см. update.md,
правило про clamp перед exp).
"""

import hashlib
import json

import numpy as np

import config
from config import (BLOCK_SIZE, BRIGHT_THRESHOLD, DARK_THRESHOLD,
                    EDGE_THRESHOLD, ENTROPY_BINS, IMG_SIZE, N_FEATURES)

_EPS = 1e-6

# Эмпирические максимумы для нормализации. Не «магические числа с потолка»:
# посчитаны как ~99-й перцентиль соответствующей величины на train-сплите
# (скрипт main.py calibrate печатает актуальные значения).
_SCALE_STD_CH = 70.0        # std канала (0..255)
_SCALE_GRAD = 40.0          # средний |градиент|
_SCALE_GRAD_MAG = 80.0      # |gx| + |gy|
_SCALE_CHROMA_STD = 0.30    # std оппонентных осей rg / yb
_SCALE_SAT_STD = 0.30       # std насыщенности
_SCALE_BLOCK = 60.0         # блочные статистики яркости
_SCALE_CENTER = 64.0        # разница средних яркостей (знаковая)
_SCALE_CENTER_SAT = 0.30    # разница средних насыщенностей (знаковая)
# center_edge_diff — это РАЗНОСТЬ двух средних градиентов, а не сам градиент,
# и она вчетверо мельче: делить её на _SCALE_GRAD_MAG (масштаб самой величины)
# значило зажать признак в четверть диапазона. Для оракула это было безразлично
# (BatchNorm и обучаемые веса съедают любое аффинное преобразование входа), но
# EML-оператор содержит exp(), и сжатый в узкую полосу признак ведёт себя
# принципиально иначе. Отдельная константа подтверждена `main.py calibrate`.
_SCALE_CENTER_EDGE = 20.0   # разница средних |grad| центр-периферия (знаковая)

# Признаки, чей масштаб ПОДОБРАН эмпирически (константы _SCALE_* выше) — только
# у них имеет смысл проверять, попадает ли распределение в [-1, 1]. Остальные
# ограничены по построению: доли лежат в [0, 1], корреляции и нормированные
# отношения — в [-1, 1], средние каналов делятся на 255. Для них упор в границу
# диапазона это правда о картинке, а не просчёт нормировки, и команда
# `main.py calibrate` не должна на них ругаться.
EMPIRICALLY_SCALED = {
    'std_r', 'std_g', 'std_b', 'std_gray', 'std_rg', 'std_yb', 'std_sat',
    'grad_x', 'grad_y', 'grad_mag', 'grad_std',
    'block_std', 'block_spread', 'block16_spread', 'quad_spread',
    'center_gray_diff', 'center_sat_diff', 'center_edge_diff',
}


# ── Нормализация ──────────────────────────────────────────────────────────────

def _norm(values, max_value: float):
    """[0, max_value] -> [-1, 1] с клиппингом."""
    return np.clip(values / max_value, 0.0, 1.0) * 2.0 - 1.0


def _norm_signed(values, max_value: float):
    """[-max_value, +max_value] -> [-1, 1] с клиппингом."""
    return np.clip(values / max_value, -1.0, 1.0)


# ── Основной экстрактор ───────────────────────────────────────────────────────

def extract_batch(images: np.ndarray, chunk: int = 512) -> np.ndarray:
    """
    Посчитать признаки для пачки картинок.

    Args:
        images: (N, 64, 64, 3) uint8, RGB.
        chunk:  сколько картинок обрабатывать за раз (память: весь датасет в
                float32 — это ~250 МБ, поэтому режем).

    Returns:
        (N, 32) float32, все значения в [-1, 1].
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"ожидался массив (N, H, W, 3), получен {images.shape}")

    out = np.empty((len(images), N_FEATURES), dtype=np.float32)
    for start in range(0, len(images), chunk):
        end = min(start + chunk, len(images))
        out[start:end] = _extract_chunk(images[start:end].astype(np.float32))
    return out


def extract(image: np.ndarray) -> np.ndarray:
    """Признаки одной картинки (64, 64, 3) uint8 -> (32,) float32."""
    return extract_batch(image[None, ...])[0]


def _extract_chunk(img: np.ndarray) -> np.ndarray:
    """Ядро экстрактора. img: (N, 64, 64, 3) float32 в шкале 0..255."""
    n = len(img)
    feats = np.empty((n, N_FEATURES), dtype=np.float32)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b          # (N, 64, 64)
    px = (1, 2)                                        # оси пикселей

    # ── Цвет: моменты каналов (0..7) ─────────────────────────────────────────
    feats[:, 0] = _norm(r.mean(axis=px), 255.0)
    feats[:, 1] = _norm(g.mean(axis=px), 255.0)
    feats[:, 2] = _norm(b.mean(axis=px), 255.0)
    feats[:, 3] = _norm(r.std(axis=px), _SCALE_STD_CH)
    feats[:, 4] = _norm(g.std(axis=px), _SCALE_STD_CH)
    feats[:, 5] = _norm(b.std(axis=px), _SCALE_STD_CH)
    feats[:, 6] = _norm(gray.mean(axis=px), 255.0)
    feats[:, 7] = _norm(gray.std(axis=px), _SCALE_STD_CH)

    # ── Цвет: оппонентные оси (8..11) ────────────────────────────────────────
    # rg и yb — дешёвая замена переводу в HSV/Lab: две вычитания и деление,
    # но они уже инвариантны к общей яркости, в отличие от сырых R, G, B.
    rg = (r - g) / (r + g + _EPS)                      # [-1, 1]
    yb = (r + g - 2.0 * b) / (r + g + 2.0 * b + _EPS)  # [-1, 1]
    feats[:, 8] = np.clip(rg.mean(axis=px), -1.0, 1.0)
    feats[:, 9] = np.clip(yb.mean(axis=px), -1.0, 1.0)
    feats[:, 10] = _norm(rg.std(axis=px), _SCALE_CHROMA_STD)
    feats[:, 11] = _norm(yb.std(axis=px), _SCALE_CHROMA_STD)

    # ── Цвет: насыщенность (12..13) ──────────────────────────────────────────
    cmax = img.max(axis=3)
    cmin = img.min(axis=3)
    sat = (cmax - cmin) / (cmax + _EPS)                # [0, 1]
    feats[:, 12] = _norm(sat.mean(axis=px), 1.0)
    feats[:, 13] = _norm(sat.std(axis=px), _SCALE_SAT_STD)

    # ── Текстура: градиенты (14..19) ─────────────────────────────────────────
    # Прямые разности, а не Собель: на MCU это два вычитания на пиксель.
    gx = gray[:, :-1, 1:] - gray[:, :-1, :-1]          # (N, 63, 63)
    gy = gray[:, 1:, :-1] - gray[:, :-1, :-1]
    agx, agy = np.abs(gx), np.abs(gy)
    mean_gx = agx.mean(axis=px)
    mean_gy = agy.mean(axis=px)
    grad_mag = agx + agy                               # L1 вместо sqrt: дешевле

    feats[:, 14] = _norm(mean_gx, _SCALE_GRAD)
    feats[:, 15] = _norm(mean_gy, _SCALE_GRAD)
    feats[:, 16] = _norm(grad_mag.mean(axis=px), _SCALE_GRAD_MAG)
    feats[:, 17] = _norm(grad_mag.std(axis=px), _SCALE_GRAD_MAG)
    feats[:, 18] = _norm((grad_mag > EDGE_THRESHOLD).mean(axis=px), 1.0)
    # Анизотропия: >0 — текстура тянется вертикально, <0 — горизонтально.
    feats[:, 19] = (mean_gx - mean_gy) / (mean_gx + mean_gy + _EPS)

    # ── Текстура: когерентность структурного тензора (20) ────────────────────
    # J = [[<gx^2>, <gx*gy>], [<gx*gy>, <gy^2>]]; когерентность — нормированная
    # разность его собственных чисел. 1 = все градиенты смотрят в одну сторону
    # (прямая линия, дорога), 0 = хаос (песок, толпа).
    jxx = (gx * gx).mean(axis=px)
    jyy = (gy * gy).mean(axis=px)
    jxy = (gx * gy).mean(axis=px)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + _EPS)
    feats[:, 20] = _norm(coherence, 1.0)

    # ── Текстура: энтропия гистограммы яркости (21) ──────────────────────────
    feats[:, 21] = _norm(_entropy(gray), np.log(ENTROPY_BINS))

    # ── Блочные статистики 8x8 (22..23) ──────────────────────────────────────
    nb = IMG_SIZE // BLOCK_SIZE
    blocks = gray.reshape(n, nb, BLOCK_SIZE, nb, BLOCK_SIZE).transpose(0, 1, 3, 2, 4)
    blocks = blocks.reshape(n, nb * nb, BLOCK_SIZE * BLOCK_SIZE)
    feats[:, 22] = _norm(blocks.std(axis=2).mean(axis=1), _SCALE_BLOCK)
    feats[:, 23] = _norm(blocks.mean(axis=2).std(axis=1), _SCALE_BLOCK)

    # ── Раскладка: центр против периферии (24..26) ───────────────────────────
    # Картинки — вырезки вокруг найденного объекта, поэтому «в центре не то же,
    # что по краям» отличает MOTO/PEOPLE (объект в кадре) от BEACH/ROADS (фон).
    c0, c1 = IMG_SIZE // 4, IMG_SIZE - IMG_SIZE // 4    # 16..48
    feats[:, 24] = _norm_signed(_center_minus_border(gray, c0, c1), _SCALE_CENTER)
    feats[:, 25] = _norm_signed(_center_minus_border(sat, c0, c1), _SCALE_CENTER_SAT)
    # Карта краёв на сетке 63x63 — берём её центральную четверть.
    edge_c = grad_mag[:, c0:c1 - 1, c0:c1 - 1].mean(axis=px)
    feats[:, 26] = _norm_signed(edge_c - grad_mag.mean(axis=px), _SCALE_CENTER_EDGE)

    # ── Раскладка: радиальный профиль и крупный масштаб (27..28) ─────────────
    # Первая версия признаков считала здесь «верх минус низ» и «лево минус
    # право». На аэроснимке у кадра нет канонической ориентации, и оба признака
    # оказались мусором (F-статистика 2.5 и 1.2 против 330 у coherence).
    # Замена — поворотно-инвариантные величины того же смысла:
    #   radial_slope   — темнеет ли картинка к центру (объект) или к краям;
    #   block16_spread — та же неоднородность, что block_spread, но на вдвое
    #                    более крупном масштабе (пара признаков даёт грубое
    #                    представление о том, на каком масштабе живёт текстура).
    feats[:, 27] = _radial_slope(gray)
    nb16 = IMG_SIZE // 16
    blocks16 = gray.reshape(n, nb16, 16, nb16, 16).transpose(0, 1, 3, 2, 4)
    blocks16 = blocks16.reshape(n, nb16 * nb16, 16 * 16)
    feats[:, 28] = _norm(blocks16.mean(axis=2).std(axis=1), _SCALE_BLOCK)

    # ── Раскладка: квадранты (29) ────────────────────────────────────────────
    half = IMG_SIZE // 2
    quads = np.stack([
        gray[:, :half, :half].mean(axis=px), gray[:, :half, half:].mean(axis=px),
        gray[:, half:, :half].mean(axis=px), gray[:, half:, half:].mean(axis=px),
    ], axis=1)
    feats[:, 29] = _norm(quads.std(axis=1), _SCALE_BLOCK)

    # ── Раскладка: доли тёмного и яркого (30..31) ────────────────────────────
    feats[:, 30] = _norm((gray < DARK_THRESHOLD).mean(axis=px), 1.0)
    feats[:, 31] = _norm((gray > BRIGHT_THRESHOLD).mean(axis=px), 1.0)

    return feats


def _center_minus_border(field: np.ndarray, c0: int, c1: int) -> np.ndarray:
    """
    Среднее по центральному окну минус среднее по рамке вокруг него.

    Периферия считается вычитанием, а не маской: сумма по рамке = полная сумма
    минус сумма по центру. Один проход вместо двух.
    """
    total_sum = field.sum(axis=(1, 2))
    center = field[:, c0:c1, c0:c1]
    center_sum = center.sum(axis=(1, 2))
    n_center = center.shape[1] * center.shape[2]
    n_total = field.shape[1] * field.shape[2]
    center_mean = center_sum / n_center
    border_mean = (total_sum - center_sum) / (n_total - n_center)
    return center_mean - border_mean


def _radial_distance_map() -> np.ndarray:
    """Центрированная карта расстояний от центра кадра (считается один раз)."""
    axis = np.arange(IMG_SIZE, dtype=np.float32) - (IMG_SIZE - 1) / 2.0
    dist = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    return dist - dist.mean()


_DIST_CENTERED = _radial_distance_map()
_DIST_STD = float(np.sqrt((_DIST_CENTERED ** 2).mean()))


def _radial_slope(gray: np.ndarray) -> np.ndarray:
    """
    Корреляция яркости с расстоянием от центра кадра, (N,) в [-1, 1].

    Отрицательное значение = центр темнее краёв (характерно для кадра с
    объектом посередине), положительное = центр светлее.

    В отличие от «верх минус низ», величина не зависит от поворота кадра —
    именно поэтому она заменила два выброшенных признака.
    """
    gray_c = gray - gray.mean(axis=(1, 2), keepdims=True)
    cov = (gray_c * _DIST_CENTERED).mean(axis=(1, 2))
    gray_std = np.sqrt((gray_c ** 2).mean(axis=(1, 2)))
    return np.clip(cov / (gray_std * _DIST_STD + _EPS), -1.0, 1.0)


def _entropy(gray: np.ndarray) -> np.ndarray:
    """
    Энтропия гистограммы яркости (ENTROPY_BINS бинов), (N,) float.

    Гистограммы всех картинок пачки считаются одним bincount: индексы бинов
    сдвигаются на i * BINS, поэтому один вызов вместо цикла по N.
    """
    n = len(gray)
    idx = np.clip((gray * (ENTROPY_BINS / 256.0)).astype(np.int32),
                  0, ENTROPY_BINS - 1)
    offset = np.arange(n, dtype=np.int32)[:, None] * ENTROPY_BINS
    flat = (idx.reshape(n, -1) + offset).ravel()
    counts = np.bincount(flat, minlength=n * ENTROPY_BINS).reshape(n, ENTROPY_BINS)
    p = counts / counts.sum(axis=1, keepdims=True)
    return -(p * np.log(p + _EPS)).sum(axis=1)


# ── Разделяющая сила признаков ────────────────────────────────────────────────

def anova_f(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Однофакторная F-статистика по каждому признаку: (N, F), (N,) -> (F,).

    Отношение межклассовой дисперсии к внутриклассовой — «насколько признак
    разделяет классы». Живёт здесь, а не в main.py: это свойство признаков, и
    считать его хотят и CLI (`main.py info`), и вкладка «Признаки».
    Инвариантна к аффинному масштабированию признака, поэтому правка
    констант _SCALE_* её не меняет.
    """
    X = np.asarray(X, dtype=np.float64)
    grand_mean = X.mean(axis=0)
    between, within = [], []
    for c in range(config.N_CLASSES):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        between.append(len(Xc) * (Xc.mean(axis=0) - grand_mean) ** 2)
        within.append(((Xc - Xc.mean(axis=0)) ** 2).sum(axis=0))
    df_b = config.N_CLASSES - 1
    df_w = len(X) - config.N_CLASSES
    return (np.sum(between, axis=0) / df_b) / (np.sum(within, axis=0) / df_w + 1e-12)


# ── Кэш признаков ─────────────────────────────────────────────────────────────

def fingerprint() -> str:
    """
    Отпечаток ВСЕХ величин, влияющих на значения признаков.

    Нужен, потому что проверять кэш по одной только форме (N, 32) — ловушка:
    правка масштаба (_SCALE_*) или порога форму не меняет, кэш считается
    валидным, обучение молча идёт на старых числах, и правка «не работает».
    Ровно так чуть не потерялось исправление _SCALE_CENTER_EDGE.

    Сюда входит всё, от чего зависит хоть один столбец: имена и порядок
    признаков, эмпирические масштабы, пороги и геометрические константы.
    """
    payload = {
        'feature_names': list(config.FEATURE_NAMES),
        'scales': {name: float(value) for name, value in sorted(globals().items())
                   if name.startswith('_SCALE_')},
        'thresholds': {
            'edge': float(EDGE_THRESHOLD),
            'dark': float(DARK_THRESHOLD),
            'bright': float(BRIGHT_THRESHOLD),
            'entropy_bins': int(ENTROPY_BINS),
            'block_size': int(BLOCK_SIZE),
            'img_size': int(IMG_SIZE),
        },
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()


def load_or_compute(images: np.ndarray, use_cache: bool = True,
                    verbose: bool = True) -> np.ndarray:
    """Признаки всего датасета с кэшированием в dataset/features.npz."""
    import os

    path = config.FEATURES_CACHE_PATH
    current = fingerprint()
    if use_cache and os.path.exists(path):
        d = np.load(path)
        cached = str(d['fingerprint']) if 'fingerprint' in d.files else None
        shape_ok = (len(d['features']) == len(images)
                    and d['features'].shape[1] == N_FEATURES)
        if shape_ok and cached == current:
            if verbose:
                print(f"[features] кэш: {path}")
            return d['features']
        if verbose:
            reason = (f"форма {d['features'].shape}" if not shape_ok
                      else f"другой отпечаток параметров ({cached} != {current})")
            print(f"[features] кэш устарел ({reason}) — пересчёт")

    if verbose:
        print(f"[features] считаю {N_FEATURES} признаков для {len(images)} картинок...")
    feats = extract_batch(images)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, features=feats, fingerprint=current)
    if verbose:
        print(f"[features] готово -> {path}")
    return feats
