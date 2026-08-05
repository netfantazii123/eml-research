"""
Tests of the C code generator.

The generated header is the final artifact of the project, and a desync between
features.py and the template in export.py is the most likely breakage after
edits: Python computes one thing, the microcontroller another, and it is only
noticeable through `main.py verify-c` — if it is run at all. What can be checked
without gcc and without trained models is checked here.
"""

import re

import numpy as np
import pytest

import config
import eml_distiller as dist
import export
import features


@pytest.fixture(autouse=True)
def inputs_configured():
    dist.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))


# ── Feature extractor ─────────────────────────────────────────────────────────

def test_features_header_has_no_unfilled_placeholders():
    """
    Not a single unclosed format brace.

    A forgotten key in generate_features_header would fail on its own, but a
    stray placeholder in the template would travel into the C as is and break
    compilation for whoever took the header to a device.
    """
    header = export.generate_features_header()
    assert not re.search(r'(?<!\{)\{[a-z_][a-z0-9_]*\}', header)


def test_features_header_uses_dedicated_center_edge_scale():
    """
    f[26] is normalized by its own constant, not the shared _SCALE_GRAD_MAG.

    The test keeps the center_edge_diff scale fix in place: if they diverged,
    Python and C would give different numbers for this feature — silently,
    because the shapes and types match.
    """
    header = export.generate_features_header()
    line = next(ln for ln in header.splitlines() if 'f[26]' in ln)
    assert f"{features._SCALE_CENTER_EDGE}f" in line
    assert f"{features._SCALE_GRAD_MAG}f" not in line


def test_features_header_carries_config_constants():
    """Thresholds are written in from config rather than duplicated by hand."""
    header = export.generate_features_header()
    for value in (config.EDGE_THRESHOLD, config.DARK_THRESHOLD,
                  config.BRIGHT_THRESHOLD):
        assert str(value) in header
    assert f"#define AERIAL_N_FEATURES   {config.N_FEATURES}" in header
    assert f"#define AERIAL_IMG_SIZE     {config.IMG_SIZE}" in header
    assert f"#define AERIAL_ENTROPY_BINS {config.ENTROPY_BINS}" in header


def test_features_header_assigns_every_feature():
    """Each of the 32 features must receive a value in the C code."""
    header = export.generate_features_header()
    for i in range(config.N_FEATURES):
        assert re.search(rf'f\[{i}\]\s*=', header), f"feature {i} is not filled"


def test_radial_constants_match_python():
    """
    The radial profile constants in C match those computed in features.py.

    They do not come from config but are derived from the frame geometry — so
    they can diverge when IMG_SIZE changes.
    """
    header = export.generate_features_header()
    assert f"{features._DIST_STD:.10f}" in header
    assert f"{export._dist_mean():.10f}" in header
    expected_mean = float(np.sqrt(
        (np.arange(config.IMG_SIZE) - (config.IMG_SIZE - 1) / 2.0) ** 2
        + ((np.arange(config.IMG_SIZE) - (config.IMG_SIZE - 1) / 2.0)[:, None]) ** 2
    ).mean())
    assert export._dist_mean() == pytest.approx(expected_mean, rel=1e-12)


# ── Formulas ──────────────────────────────────────────────────────────────────

def toy_classifier():
    trees = [
        dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=20),
                     right=dist.EMLNode('const', value=-0.367)),
        dist.EMLNode('var', var_idx=0),
        dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=1),
                     right=dist.EMLNode('var', var_idx=2)),
        dist.EMLNode('const', value=1.5),
    ]
    return dist.EMLClassifier(trees, offsets=[0.1, 0.2, 0.3, 0.4],
                              scales=[1.0, 2.0, 3.0, 4.0])


def test_formulas_header_is_named_after_branch():
    """The name in the file header must match the file name itself."""
    header = export.generate_formulas_header(
        toy_classifier(), list(config.FEATURE_NAMES), {}, branch='A')
    assert 'aerial_eml_A.h' in header
    assert 'aerial_eml.h —' not in header


def test_formulas_header_contains_calibration():
    """
    The calibration is part of the artifact rather than debug information:
    without it the argmax between four independent formulas is meaningless.
    """
    clf = toy_classifier()
    header = export.generate_formulas_header(
        clf, list(config.FEATURE_NAMES), {}, branch='A')
    for value in clf.offsets:
        assert f"{value:.6f}f" in header
    for value in clf.scales:
        assert f"{value:.6f}f" in header
    assert 'aerial_classify' in header
    assert f"#define AERIAL_N_CLASSES {config.N_CLASSES}" in header


def test_formulas_header_translates_ast():
    """A variable becomes x[i], a constant a literal, eml a call to eml_op."""
    header = export.generate_formulas_header(
        toy_classifier(), list(config.FEATURE_NAMES), {}, branch='A')
    assert 'eml_op(x[20], -0.367000f)' in header
    assert 'return x[0];' in header
    assert 'eml_op(x[1], x[2])' in header


def test_export_writes_expected_files(tmp_path):
    written = export.export_all(toy_classifier(), list(config.FEATURE_NAMES),
                                {}, branch='A', out_dir=str(tmp_path),
                                verbose=False)
    names = sorted(p.rsplit('\\', 1)[-1].rsplit('/', 1)[-1] for p in written)
    assert names == ['aerial_eml_A.h', 'aerial_features.h']


def test_branch_b_export_has_no_feature_extractor(tmp_path):
    """
    In branch B the formula input is the convolution output, so the feature
    extractor does not apply to it. Placing it alongside would imply that branch
    B also does without the network.
    """
    written = export.export_all(toy_classifier(),
                                [f'e{i}' for i in range(config.CNN_EMBED_DIM)],
                                {}, branch='B', out_dir=str(tmp_path),
                                verbose=False)
    assert len(written) == 1
    assert written[0].endswith('aerial_eml_B.h')


# ── Branch C: one formula and a threshold table ───────────────────────────────

def toy_single(thresholds=(-2.5, 1.25, 22021.0156), labels=(0, 3, 2, 1)):
    import eml_single as single

    tree = dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=20),
                        right=dist.EMLNode('var', var_idx=3))
    return single.EMLSingle(
        tree, single.IntervalDecoder(list(thresholds), list(labels), 4))


def test_single_header_has_thresholds_and_labels():
    """The threshold table and interval labels reach the C in full."""
    model = toy_single()
    header = export.generate_single_header(model, list(config.FEATURE_NAMES),
                                           {}, branch='C')
    assert '#define AERIAL_N_INTERVALS  4' in header
    assert '#define AERIAL_N_THRESHOLDS 3' in header
    assert 'aerial_interval_class[AERIAL_N_INTERVALS] = { 0, 3, 2, 1 }' in header
    assert 'aerial_classify' in header
    # Branch C has no calibration by construction: there is nothing to compare.
    assert 'aerial_offset' not in header


def test_single_header_thresholds_round_trip_through_float32():
    """
    A threshold is read back from C as exactly the same float32.

    The threshold is the decision boundary. Written with insufficient precision
    it shifts, and some frames on the device get a different class: the verify-c
    comparison fails on an otherwise correct export.
    """
    values = [-2.5, 1.2345678e-05, 22021.015625, 22024.258]
    model = toy_single(thresholds=values[:3], labels=(0, 1, 2, 3))
    header = export.generate_single_header(model, list(config.FEATURE_NAMES),
                                           {}, branch='C')
    line = next(ln for ln in header.splitlines() if 'aerial_thresholds' in ln)
    written = [float(v.strip().rstrip('f'))
               for v in line.split('{')[1].split('}')[0].split(',')]
    assert np.array_equal(np.float32(written), np.float32(values[:3]))


def test_single_header_survives_a_degenerate_table():
    """
    A single interval is valid C too.

    A degenerate formula (a constant) yields a table without thresholds, and a
    zero-length array is not valid C: the header would simply fail to compile,
    and only in the rare run where evolution found nothing.
    """
    header = export.generate_single_header(
        toy_single(thresholds=(), labels=(2,)),
        list(config.FEATURE_NAMES), {}, branch='C')
    assert '#define AERIAL_N_THRESHOLDS 0' in header
    assert 'aerial_thresholds[] = { 0.0f' in header


def test_branch_c_export_keeps_feature_extractor(tmp_path):
    """Branch C runs on the same features as A — it needs the extractor."""
    written = export.export_all(toy_single(), list(config.FEATURE_NAMES), {},
                                branch='C', out_dir=str(tmp_path), verbose=False)
    names = sorted(p.rsplit('\\', 1)[-1].rsplit('/', 1)[-1] for p in written)
    assert names == ['aerial_eml_C.h', 'aerial_features.h']
