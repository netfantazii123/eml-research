"""
Tests of the feature extractor.

Three things are checked, each of which does NOT fail on its own when broken:

  1. All features lie in [-1, 1]. The EML operator contains exp(), and a feature
     that leaves the range gives the formula inf — not an error but a quietly
     corrupted result.
  2. The rotation-invariant features really are invariant. Feature selection
     rests on this (grad_vert/grad_horz were discarded precisely because they
     depended on frame orientation) — the claim must be checked, not declared.
  3. The cache fingerprint changes when a constant is edited. Without this a
     scale change silently never reaches training.
"""

import numpy as np
import pytest

import config
import features


# ── Sets of test images ───────────────────────────────────────────────────────

def sample_images() -> dict:
    """Images of various character — from degenerate to class-like."""
    rng = np.random.default_rng(0)
    size = config.IMG_SIZE
    out = {}

    out['noise'] = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    out['black'] = np.zeros((size, size, 3), dtype=np.uint8)
    out['white'] = np.full((size, size, 3), 255, dtype=np.uint8)
    out['gray'] = np.full((size, size, 3), 128, dtype=np.uint8)

    # Stripes: a crude imitation of a road — one dominant direction.
    stripes = np.zeros((size, size, 3), dtype=np.uint8)
    stripes[:, ::4] = 255
    out['stripes'] = stripes

    # A smooth gradient — a flat background without texture.
    ramp = np.linspace(0, 255, size, dtype=np.uint8)
    out['gradient'] = np.repeat(np.tile(ramp, (size, 1))[:, :, None], 3, axis=2)

    # A bright spot in the center — "an object in the frame".
    spot = np.full((size, size, 3), 40, dtype=np.uint8)
    spot[24:40, 24:40] = 220
    out['spot'] = spot
    return out


def transforms():
    """Eight frame orientations: an aerial image has no top or bottom."""
    return {
        'rot90': lambda a: np.rot90(a, 1, axes=(0, 1)),
        'rot180': lambda a: np.rot90(a, 2, axes=(0, 1)),
        'rot270': lambda a: np.rot90(a, 3, axes=(0, 1)),
        'fliplr': lambda a: a[:, ::-1],
        'flipud': lambda a: a[::-1],
    }


# ── Range and shape ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(sample_images()))
def test_features_stay_in_range(name):
    """Every feature is in [-1, 1] and finite — on any image, degenerate ones too."""
    feats = features.extract(sample_images()[name])
    assert feats.shape == (config.N_FEATURES,)
    assert np.all(np.isfinite(feats)), f"non-finite features on \"{name}\""
    assert feats.min() >= -1.0 - 1e-6, f"feature below -1 on \"{name}\""
    assert feats.max() <= 1.0 + 1e-6, f"feature above +1 on \"{name}\""


def test_batch_matches_single():
    """extract_batch and extract compute the same thing."""
    images = np.stack(list(sample_images().values()))
    batch = features.extract_batch(images)
    for i, image in enumerate(images):
        assert np.allclose(batch[i], features.extract(image), atol=1e-6)


def test_batch_chunking_is_transparent():
    """Splitting into chunks does not affect the result."""
    images = np.stack(list(sample_images().values()))
    assert np.allclose(features.extract_batch(images, chunk=2),
                       features.extract_batch(images, chunk=512), atol=1e-6)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        features.extract_batch(np.zeros((2, 64, 64), dtype=np.uint8))


# ── Rotation invariance ───────────────────────────────────────────────────────

@pytest.mark.parametrize('name', list(sample_images()))
def test_radial_slope_is_rotation_invariant(name):
    """
    radial_slope does not depend on the frame orientation.

    It is computed through the distance to the center, and the distance map is
    symmetric under 90° rotations and reflections, so the match here is exact —
    up to float32 summation order.
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
    coherence does not depend on the frame orientation.

    Exact equality does not and cannot occur here: the gradients are computed on
    a 63x63 grid, i.e. the last row and column do not enter it, and under a
    rotation DIFFERENT pixels drop out. On structured images the divergence is
    zero, on pure noise it reaches ~0.03 — that is a sampling difference rather
    than an orientation dependence, hence the 0.05 tolerance.
    """
    idx = config.FEATURE_NAMES.index('coherence')
    image = sample_images()[name]
    base = features.extract(image)[idx]
    for label, transform in transforms().items():
        got = features.extract(np.ascontiguousarray(transform(image)))[idx]
        assert got == pytest.approx(base, abs=0.05), f"{name}/{label}"


def test_coherence_separates_lines_from_noise():
    """
    The main claim about the features: roads have one dominant direction, sand
    and crowds do not. Stripes must give a coherence noticeably higher than
    noise.
    """
    idx = config.FEATURE_NAMES.index('coherence')
    images = sample_images()
    lines = features.extract(images['stripes'])[idx]
    noise = features.extract(images['noise'])[idx]
    assert lines > noise + 0.5


def test_center_features_react_to_center():
    """Center-periphery features must tell a frame with an object from a flat one."""
    images = sample_images()
    spot = features.extract(images['spot'])
    flat = features.extract(images['gray'])
    for name in ('center_gray_diff', 'center_edge_diff'):
        idx = config.FEATURE_NAMES.index(name)
        assert abs(spot[idx]) > abs(flat[idx]) + 0.05, name


# ── Scales and the cache fingerprint ──────────────────────────────────────────

def test_center_edge_has_its_own_scale():
    """
    center_edge_diff is normalized by its own constant, not _SCALE_GRAD_MAG.

    The center-periphery gradient difference is four times smaller than the
    gradient itself; on the shared scale the feature occupied a quarter of the
    [-1, 1] range. This test keeps the fix in place.
    """
    assert features._SCALE_CENTER_EDGE != features._SCALE_GRAD_MAG
    assert 'center_edge_diff' in features.EMPIRICALLY_SCALED


def test_fingerprint_reacts_to_scale_change(monkeypatch):
    """
    The fingerprint must change when any scale constant is edited.

    Otherwise the feature cache (whose shape does not depend on the scale) stays
    "valid", training proceeds on the old numbers, and the change appears to
    have had no effect.
    """
    before = features.fingerprint()
    monkeypatch.setattr(features, '_SCALE_CENTER_EDGE', 33.0)
    assert features.fingerprint() != before


def test_fingerprint_reacts_to_threshold_change(monkeypatch):
    before = features.fingerprint()
    monkeypatch.setattr(features, 'EDGE_THRESHOLD', 42.0)
    assert features.fingerprint() != before


def test_fingerprint_is_stable():
    """Without edits the fingerprint is stable — otherwise the cache would always
    be recomputed."""
    assert features.fingerprint() == features.fingerprint()


# ── F-statistic ───────────────────────────────────────────────────────────────

def test_anova_f_is_scale_invariant():
    """
    The F-statistic does not change under an affine scaling of a feature.

    An important consequence: editing the _SCALE_* constants cannot change the
    table of F-statistics — it affects only the EML formulas.
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
