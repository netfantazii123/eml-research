"""
features.py — engineered feature extractor for a 64x64 RGB image.

This is the key module of branch A. The idea is the same as in
tetris/features.py: the oracle and the EML formula consume THE SAME feature
vector. Then after distillation nothing of the neural network remains — the
microcontroller runs only this extractor and 4 formulas.

Every feature was therefore chosen against two criteria at once:
  1. it separates the classes (BEACH / MOTO / PEOPLE / ROADS);
  2. it is computed in one or two passes over the image with plain arithmetic.

HOG, SIFT and FFT are deliberately absent — they give better separability, but
on an STM32 they no longer qualify as cheap preprocessing.

32 features, order = config.FEATURE_NAMES:

    Color (14)     — moments of the RGB channels, luminance, the opponent axes
                     rg/yb and saturation. Capture "sand is yellow and bright,
                     asphalt is gray and dark".
    Texture (8)    — gradient magnitudes, edge density, anisotropy, structure
                     tensor coherence, luminance entropy. Coherence is the main
                     ROADS feature: they have one pronounced dominant direction,
                     a beach or a crowd does not.
    8x8 blocks (2) — roughness within blocks and spread between blocks:
                     separates "smooth coarse background" from "fine dense
                     debris".
    Layout (8)     — center vs periphery, top vs bottom, quadrants, fractions of
                     dark/bright pixels. The images are crops around an object,
                     so "object in the center" is itself a signal.

All features are normalized to [-1, 1]: the EML operator contains exp(), and raw
values on a 0..255 scale must not be fed into it — it would overflow to inf (see
update.md, the rule about clamping before exp).
"""

import hashlib
import json

import numpy as np

import config
from config import (BLOCK_SIZE, BRIGHT_THRESHOLD, DARK_THRESHOLD,
                    EDGE_THRESHOLD, ENTROPY_BINS, IMG_SIZE, N_FEATURES)

_EPS = 1e-6

# Empirical maxima for normalization. Not arbitrary magic numbers: computed as
# the ~99th percentile of the corresponding quantity on the train split (the
# `main.py calibrate` command prints the current values).
_SCALE_STD_CH = 70.0        # channel std (0..255)
_SCALE_GRAD = 40.0          # mean |gradient|
_SCALE_GRAD_MAG = 80.0      # |gx| + |gy|
_SCALE_CHROMA_STD = 0.30    # std of the opponent axes rg / yb
_SCALE_SAT_STD = 0.30       # saturation std
_SCALE_BLOCK = 60.0         # block luminance statistics
_SCALE_CENTER = 64.0        # difference of mean luminances (signed)
_SCALE_CENTER_SAT = 0.30    # difference of mean saturations (signed)
# center_edge_diff is a DIFFERENCE of two mean gradients rather than the
# gradient itself, and it is four times smaller: dividing it by _SCALE_GRAD_MAG
# (the scale of the quantity itself) squeezed the feature into a quarter of the
# range. This was immaterial for the oracle (BatchNorm and trainable weights
# absorb any affine transform of the input), but the EML operator contains
# exp(), and a feature compressed into a narrow band behaves differently. The
# separate constant is confirmed by `main.py calibrate`.
_SCALE_CENTER_EDGE = 20.0   # difference of mean |grad| center-periphery (signed)

# Features whose scale is FITTED empirically (the _SCALE_* constants above) —
# only for these does it make sense to check whether the distribution falls
# inside [-1, 1]. The rest are bounded by construction: fractions lie in [0, 1],
# correlations and normalized ratios in [-1, 1], channel means are divided by
# 255. For them, hitting the range boundary is a fact about the image rather
# than a normalization mistake, and `main.py calibrate` must not flag them.
EMPIRICALLY_SCALED = {
    'std_r', 'std_g', 'std_b', 'std_gray', 'std_rg', 'std_yb', 'std_sat',
    'grad_x', 'grad_y', 'grad_mag', 'grad_std',
    'block_std', 'block_spread', 'block16_spread', 'quad_spread',
    'center_gray_diff', 'center_sat_diff', 'center_edge_diff',
}


# ── Normalization ─────────────────────────────────────────────────────────────

def _norm(values, max_value: float):
    """[0, max_value] -> [-1, 1] with clipping."""
    return np.clip(values / max_value, 0.0, 1.0) * 2.0 - 1.0


def _norm_signed(values, max_value: float):
    """[-max_value, +max_value] -> [-1, 1] with clipping."""
    return np.clip(values / max_value, -1.0, 1.0)


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_batch(images: np.ndarray, chunk: int = 512) -> np.ndarray:
    """
    Compute features for a batch of images.

    Args:
        images: (N, 64, 64, 3) uint8, RGB.
        chunk:  how many images to process at a time (memory: the whole dataset
                in float32 is ~250 MB, hence the chunking).

    Returns:
        (N, 32) float32, all values in [-1, 1].
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[3] != 3:
        raise ValueError(f"expected an (N, H, W, 3) array, got {images.shape}")

    out = np.empty((len(images), N_FEATURES), dtype=np.float32)
    for start in range(0, len(images), chunk):
        end = min(start + chunk, len(images))
        out[start:end] = _extract_chunk(images[start:end].astype(np.float32))
    return out


def extract(image: np.ndarray) -> np.ndarray:
    """Features of a single (64, 64, 3) uint8 image -> (32,) float32."""
    return extract_batch(image[None, ...])[0]


def _extract_chunk(img: np.ndarray) -> np.ndarray:
    """Extractor core. img: (N, 64, 64, 3) float32 on a 0..255 scale."""
    n = len(img)
    feats = np.empty((n, N_FEATURES), dtype=np.float32)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b          # (N, 64, 64)
    px = (1, 2)                                        # pixel axes

    # ── Color: channel moments (0..7) ────────────────────────────────────────
    feats[:, 0] = _norm(r.mean(axis=px), 255.0)
    feats[:, 1] = _norm(g.mean(axis=px), 255.0)
    feats[:, 2] = _norm(b.mean(axis=px), 255.0)
    feats[:, 3] = _norm(r.std(axis=px), _SCALE_STD_CH)
    feats[:, 4] = _norm(g.std(axis=px), _SCALE_STD_CH)
    feats[:, 5] = _norm(b.std(axis=px), _SCALE_STD_CH)
    feats[:, 6] = _norm(gray.mean(axis=px), 255.0)
    feats[:, 7] = _norm(gray.std(axis=px), _SCALE_STD_CH)

    # ── Color: opponent axes (8..11) ─────────────────────────────────────────
    # rg and yb are a cheap replacement for a conversion to HSV/Lab: two
    # subtractions and a division, but already invariant to overall brightness,
    # unlike raw R, G, B.
    rg = (r - g) / (r + g + _EPS)                      # [-1, 1]
    yb = (r + g - 2.0 * b) / (r + g + 2.0 * b + _EPS)  # [-1, 1]
    feats[:, 8] = np.clip(rg.mean(axis=px), -1.0, 1.0)
    feats[:, 9] = np.clip(yb.mean(axis=px), -1.0, 1.0)
    feats[:, 10] = _norm(rg.std(axis=px), _SCALE_CHROMA_STD)
    feats[:, 11] = _norm(yb.std(axis=px), _SCALE_CHROMA_STD)

    # ── Color: saturation (12..13) ───────────────────────────────────────────
    cmax = img.max(axis=3)
    cmin = img.min(axis=3)
    sat = (cmax - cmin) / (cmax + _EPS)                # [0, 1]
    feats[:, 12] = _norm(sat.mean(axis=px), 1.0)
    feats[:, 13] = _norm(sat.std(axis=px), _SCALE_SAT_STD)

    # ── Texture: gradients (14..19) ──────────────────────────────────────────
    # Plain differences rather than Sobel: on an MCU that is two subtractions
    # per pixel.
    gx = gray[:, :-1, 1:] - gray[:, :-1, :-1]          # (N, 63, 63)
    gy = gray[:, 1:, :-1] - gray[:, :-1, :-1]
    agx, agy = np.abs(gx), np.abs(gy)
    mean_gx = agx.mean(axis=px)
    mean_gy = agy.mean(axis=px)
    grad_mag = agx + agy                               # L1 instead of sqrt: cheaper

    feats[:, 14] = _norm(mean_gx, _SCALE_GRAD)
    feats[:, 15] = _norm(mean_gy, _SCALE_GRAD)
    feats[:, 16] = _norm(grad_mag.mean(axis=px), _SCALE_GRAD_MAG)
    feats[:, 17] = _norm(grad_mag.std(axis=px), _SCALE_GRAD_MAG)
    feats[:, 18] = _norm((grad_mag > EDGE_THRESHOLD).mean(axis=px), 1.0)
    # Anisotropy: >0 — the texture is elongated vertically, <0 — horizontally.
    feats[:, 19] = (mean_gx - mean_gy) / (mean_gx + mean_gy + _EPS)

    # ── Texture: structure tensor coherence (20) ─────────────────────────────
    # J = [[<gx^2>, <gx*gy>], [<gx*gy>, <gy^2>]]; coherence is the normalized
    # difference of its eigenvalues. 1 = all gradients point the same way (a
    # straight line, a road), 0 = chaos (sand, a crowd).
    jxx = (gx * gx).mean(axis=px)
    jyy = (gy * gy).mean(axis=px)
    jxy = (gx * gy).mean(axis=px)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + _EPS)
    feats[:, 20] = _norm(coherence, 1.0)

    # ── Texture: luminance histogram entropy (21) ────────────────────────────
    feats[:, 21] = _norm(_entropy(gray), np.log(ENTROPY_BINS))

    # ── 8x8 block statistics (22..23) ────────────────────────────────────────
    nb = IMG_SIZE // BLOCK_SIZE
    blocks = gray.reshape(n, nb, BLOCK_SIZE, nb, BLOCK_SIZE).transpose(0, 1, 3, 2, 4)
    blocks = blocks.reshape(n, nb * nb, BLOCK_SIZE * BLOCK_SIZE)
    feats[:, 22] = _norm(blocks.std(axis=2).mean(axis=1), _SCALE_BLOCK)
    feats[:, 23] = _norm(blocks.mean(axis=2).std(axis=1), _SCALE_BLOCK)

    # ── Layout: center vs periphery (24..26) ─────────────────────────────────
    # The images are crops around a detected object, so "the center differs from
    # the edges" separates MOTO/PEOPLE (object in frame) from BEACH/ROADS
    # (background).
    c0, c1 = IMG_SIZE // 4, IMG_SIZE - IMG_SIZE // 4    # 16..48
    feats[:, 24] = _norm_signed(_center_minus_border(gray, c0, c1), _SCALE_CENTER)
    feats[:, 25] = _norm_signed(_center_minus_border(sat, c0, c1), _SCALE_CENTER_SAT)
    # The edge map lives on a 63x63 grid — take its central quarter.
    edge_c = grad_mag[:, c0:c1 - 1, c0:c1 - 1].mean(axis=px)
    feats[:, 26] = _norm_signed(edge_c - grad_mag.mean(axis=px), _SCALE_CENTER_EDGE)

    # ── Layout: radial profile and coarse scale (27..28) ─────────────────────
    # The first version of the features computed "top minus bottom" and "left
    # minus right" here. An aerial frame has no canonical orientation, and both
    # features turned out to be noise (F-statistics of 2.5 and 1.2 against 330
    # for coherence). The replacements are rotation-invariant quantities of the
    # same meaning:
    #   radial_slope   — whether the image darkens toward the center (object) or
    #                    toward the edges;
    #   block16_spread — the same inhomogeneity as block_spread but at twice the
    #                    scale (the pair gives a coarse indication of the scale
    #                    at which the texture lives).
    feats[:, 27] = _radial_slope(gray)
    nb16 = IMG_SIZE // 16
    blocks16 = gray.reshape(n, nb16, 16, nb16, 16).transpose(0, 1, 3, 2, 4)
    blocks16 = blocks16.reshape(n, nb16 * nb16, 16 * 16)
    feats[:, 28] = _norm(blocks16.mean(axis=2).std(axis=1), _SCALE_BLOCK)

    # ── Layout: quadrants (29) ───────────────────────────────────────────────
    half = IMG_SIZE // 2
    quads = np.stack([
        gray[:, :half, :half].mean(axis=px), gray[:, :half, half:].mean(axis=px),
        gray[:, half:, :half].mean(axis=px), gray[:, half:, half:].mean(axis=px),
    ], axis=1)
    feats[:, 29] = _norm(quads.std(axis=1), _SCALE_BLOCK)

    # ── Layout: dark and bright fractions (30..31) ───────────────────────────
    feats[:, 30] = _norm((gray < DARK_THRESHOLD).mean(axis=px), 1.0)
    feats[:, 31] = _norm((gray > BRIGHT_THRESHOLD).mean(axis=px), 1.0)

    return feats


def _center_minus_border(field: np.ndarray, c0: int, c1: int) -> np.ndarray:
    """
    Mean over the central window minus the mean over the frame around it.

    The periphery is obtained by subtraction rather than by a mask: the sum over
    the frame = total sum minus the sum over the center. One pass instead of two.
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
    """Centered map of distances from the frame center (computed once)."""
    axis = np.arange(IMG_SIZE, dtype=np.float32) - (IMG_SIZE - 1) / 2.0
    dist = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)
    return dist - dist.mean()


_DIST_CENTERED = _radial_distance_map()
_DIST_STD = float(np.sqrt((_DIST_CENTERED ** 2).mean()))


def _radial_slope(gray: np.ndarray) -> np.ndarray:
    """
    Correlation of luminance with distance from the frame center, (N,) in
    [-1, 1].

    A negative value = the center is darker than the edges (typical of a frame
    with an object in the middle), positive = the center is brighter.

    Unlike "top minus bottom", the quantity does not depend on frame rotation —
    which is exactly why it replaced the two discarded features.
    """
    gray_c = gray - gray.mean(axis=(1, 2), keepdims=True)
    cov = (gray_c * _DIST_CENTERED).mean(axis=(1, 2))
    gray_std = np.sqrt((gray_c ** 2).mean(axis=(1, 2)))
    return np.clip(cov / (gray_std * _DIST_STD + _EPS), -1.0, 1.0)


def _entropy(gray: np.ndarray) -> np.ndarray:
    """
    Entropy of the luminance histogram (ENTROPY_BINS bins), (N,) float.

    The histograms of all images in the batch are computed with a single
    bincount: bin indices are shifted by i * BINS, so one call replaces a loop
    over N.
    """
    n = len(gray)
    idx = np.clip((gray * (ENTROPY_BINS / 256.0)).astype(np.int32),
                  0, ENTROPY_BINS - 1)
    offset = np.arange(n, dtype=np.int32)[:, None] * ENTROPY_BINS
    flat = (idx.reshape(n, -1) + offset).ravel()
    counts = np.bincount(flat, minlength=n * ENTROPY_BINS).reshape(n, ENTROPY_BINS)
    p = counts / counts.sum(axis=1, keepdims=True)
    return -(p * np.log(p + _EPS)).sum(axis=1)


# ── Feature discriminative power ──────────────────────────────────────────────

def anova_f(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    One-way F-statistic per feature: (N, F), (N,) -> (F,).

    The ratio of between-class to within-class variance — "how well the feature
    separates the classes". It lives here rather than in main.py: it is a
    property of the features, and both the CLI (`main.py info`) and the
    "Features" tab want to compute it. It is invariant to affine scaling of a
    feature, so editing the _SCALE_* constants does not change it.
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


# ── Feature cache ─────────────────────────────────────────────────────────────

def fingerprint() -> str:
    """
    Fingerprint of ALL quantities that affect the feature values.

    Needed because validating the cache by shape (N, 32) alone is a trap: an
    edit to a scale (_SCALE_*) or a threshold does not change the shape, the
    cache is considered valid, training silently proceeds on the old numbers,
    and the edit appears to have no effect. This is exactly how the
    _SCALE_CENTER_EDGE fix was nearly lost.

    It covers everything at least one column depends on: feature names and
    order, empirical scales, thresholds, and geometric constants.
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
    """Features of the whole dataset, cached in dataset/features.npz."""
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
                print(f"[features] cache: {path}")
            return d['features']
        if verbose:
            reason = (f"shape {d['features'].shape}" if not shape_ok
                      else f"different parameter fingerprint ({cached} != {current})")
            print(f"[features] cache is stale ({reason}) — recomputing")

    if verbose:
        print(f"[features] computing {N_FEATURES} features for {len(images)} images...")
    feats = extract_batch(images)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, features=feats, fingerprint=current)
    if verbose:
        print(f"[features] done -> {path}")
    return feats
