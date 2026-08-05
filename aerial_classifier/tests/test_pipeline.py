"""
Tests protecting the saved formulas from a silent regression.

This logic fires at the very end of a multi-hour run, so an error in it costs
the most: the fact that the best formula set was overwritten by a worse one can
only be noticed from the metrics in the report, and only if the previous ones
are remembered.

pipeline pulls in torch (through oracle.py), so without it the test is skipped —
the rest of the suite does not depend on torch and runs as usual.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip('torch', reason="pipeline imports oracle -> torch")

import config
import eml_distiller as dist
import pipeline
import storage


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Private models/ and runs/, so the test does not touch real artifacts."""
    monkeypatch.setattr(config, 'MODELS_DIR', str(tmp_path / 'models'))
    monkeypatch.setattr(config, 'RUNS_DIR', str(tmp_path / 'runs'))
    dist.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))
    return tmp_path


def trees():
    return [dist.EMLNode('var', var_idx=i) for i in range(config.N_CLASSES)]


def meta_with(val: float, stopped: bool = False, test: float | None = None) -> dict:
    return {
        'branch': 'A',
        'stopped': stopped,
        'metrics': {
            'val': {'eml_bal_acc': val},
            'test': {'eml_bal_acc': test if test is not None else val},
        },
    }


def names():
    return list(config.FEATURE_NAMES)


def stored_val(branch: str = 'A') -> float:
    return storage.formulas_meta(branch)['metrics']['val']['eml_bal_acc']


# ── Promoting the best ────────────────────────────────────────────────────────

def test_first_run_becomes_best(sandbox):
    result = pipeline.save_best(trees(), 'A', meta_with(0.5), names(),
                                verbose=False)
    assert result['promoted'] is True
    assert os.path.exists(storage.formulas_path('A'))
    assert stored_val() == pytest.approx(0.5)


def test_better_run_replaces_best(sandbox):
    pipeline.save_best(trees(), 'A', meta_with(0.50), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.61), names(),
                                verbose=False)
    assert result['promoted'] is True
    assert stored_val() == pytest.approx(0.61)


def test_worse_run_does_not_replace_best(sandbox):
    """
    The key property: an unlucky run must not erase a good one.

    Previously any run with save=True (the default in both the GUI and the CLI)
    overwrote models/best_eml_A.json unconditionally.
    """
    pipeline.save_best(trees(), 'A', meta_with(0.61), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.44), names(),
                                verbose=False)
    assert result['promoted'] is False
    assert result['best_path'] is None
    assert stored_val() == pytest.approx(0.61), \
        "the best set was overwritten by a worse one"


def test_equal_run_replaces_best(sandbox):
    """A run equal by val passes: all else equal, the fresher result wins."""
    pipeline.save_best(trees(), 'A', meta_with(0.55), names(), verbose=False)
    assert pipeline.save_best(trees(), 'A', meta_with(0.55), names(),
                              verbose=False)['promoted'] is True


def test_stopped_run_never_becomes_best(sandbox):
    """A manually interrupted run does not claim best even with a high val."""
    pipeline.save_best(trees(), 'A', meta_with(0.50), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.99, stopped=True),
                                names(), verbose=False)
    assert result['promoted'] is False
    assert 'interrupted' in result['reason']
    assert stored_val() == pytest.approx(0.50)


def test_stopped_run_is_not_best_even_when_none_stored(sandbox):
    result = pipeline.save_best(trees(), 'A', meta_with(0.9, stopped=True),
                                names(), verbose=False)
    assert result['promoted'] is False
    assert not os.path.exists(storage.formulas_path('A'))


# ── Nothing is lost ───────────────────────────────────────────────────────────

def test_every_run_is_saved_to_runs(sandbox):
    """Even a rejected run sits complete in runs/ — nothing disappears."""
    pipeline.save_best(trees(), 'A', meta_with(0.61), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.30), names(),
                                verbose=False)

    assert os.path.exists(result['run_path'])
    payload = json.load(open(result['run_path'], encoding='utf-8'))
    assert payload['meta']['metrics']['val']['eml_bal_acc'] == pytest.approx(0.30)
    assert len(payload['trees']) == config.N_CLASSES


def test_previous_best_is_backed_up(sandbox):
    """Before replacement the previous file is kept alongside as .prev.json."""
    pipeline.save_best(trees(), 'A', meta_with(0.50), names(), verbose=False)
    pipeline.save_best(trees(), 'A', meta_with(0.70), names(), verbose=False)

    backup = storage.formulas_path('A').replace('.json', '.prev.json')
    assert os.path.exists(backup)
    assert (json.load(open(backup, encoding='utf-8'))
            ['meta']['metrics']['val']['eml_bal_acc']) == pytest.approx(0.50)


def test_branches_do_not_interfere(sandbox):
    """Branch B must not disturb branch A: separate files and separate comparisons."""
    pipeline.save_best(trees(), 'A', meta_with(0.60), names(), verbose=False)
    result = pipeline.save_best(trees(), 'B', meta_with(0.40), names(),
                                verbose=False)
    assert result['promoted'] is True
    assert stored_val('A') == pytest.approx(0.60)
    assert stored_val('B') == pytest.approx(0.40)


def test_corrupt_best_file_is_replaced(sandbox):
    """Corrupted JSON must not block saving — treat it as absent."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(storage.formulas_path('A'), 'w', encoding='utf-8') as f:
        f.write('{not json')

    result = pipeline.save_best(trees(), 'A', meta_with(0.33), names(),
                                verbose=False)
    assert result['promoted'] is True
    assert stored_val() == pytest.approx(0.33)


def test_saved_formulas_reload_into_working_classifier(sandbox):
    """
    The saved artifact must load back and compute the same thing — otherwise
    what goes to the microcontroller is not what the report showed.
    """
    original = [
        dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=20),
                     right=dist.EMLNode('const', value=-0.367)),
        dist.EMLNode('var', var_idx=0),
        dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=1),
                     right=dist.EMLNode('var', var_idx=2)),
        dist.EMLNode('const', value=1.5),
    ]
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (64, config.N_FEATURES))
    source = dist.EMLClassifier(original).fit_calibration(X)

    meta = meta_with(0.5)
    meta['calibration'] = source.calibration()
    pipeline.save_best(original, 'A', meta, names(), verbose=False)

    loaded, payload = storage.load_classifier(branch='A')
    assert payload['input_names'] == names()
    assert np.allclose(loaded.offsets, source.offsets)
    assert np.allclose(loaded.scales, source.scales)
    assert np.array_equal(loaded.predict(X), source.predict(X))


# ── Branch C ──────────────────────────────────────────────────────────────────

def single_model():
    import eml_single as single

    tree = dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=20),
                        right=dist.EMLNode('var', var_idx=3))
    decoder = single.IntervalDecoder([-1.0, 0.5, 3.0], [0, 2, 1, 2], 4,
                                     score=0.61)
    return single.EMLSingle(tree, decoder)


def test_single_formula_round_trip(sandbox):
    """
    A branch C formula returns from JSON together with its threshold table.

    Forgetting the table is the same as forgetting the calibration in branch A:
    the formula runs and the class silently turns out to be garbage. The table is
    therefore part of the artifact.
    """
    import eml_single as single

    source = single_model()
    meta = meta_with(0.55)
    meta['branch'] = 'C'
    meta['decoder'] = source.decoder.to_dict()
    pipeline.save_best([source.tree], 'C', meta, names(), verbose=False,
                       decoder=source.decoder.to_dict())

    loaded, payload = storage.load_classifier(branch='C')
    assert isinstance(loaded, single.EMLSingle)
    assert payload['class_names'] == list(config.CLASS_NAMES)

    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (64, config.N_FEATURES))
    assert np.array_equal(loaded.predict(X), source.predict(X))


def test_branch_c_does_not_touch_other_branches(sandbox):
    """The active slots of branches A, B and C are independent."""
    pipeline.save_best(trees(), 'A', meta_with(0.7), names(), verbose=False)
    source = single_model()
    meta = meta_with(0.4)
    meta['branch'] = 'C'
    pipeline.save_best([source.tree], 'C', meta, names(), verbose=False,
                       decoder=source.decoder.to_dict())

    assert stored_val('A') == pytest.approx(0.7)
    assert stored_val('C') == pytest.approx(0.4)


def test_foreign_class_set_is_refused(sandbox, monkeypatch):
    """
    A model trained on different classes must not be applied.

    After a dataset folder change the class set differs, but the formulas would
    run as if nothing happened: the metrics would be computed, the report
    printed, and all of it would mean something else. A refusal with a recipe is
    cheaper than silent nonsense.
    """
    source = single_model()
    meta = meta_with(0.5)
    meta['branch'] = 'C'
    pipeline.save_best([source.tree], 'C', meta, names(), verbose=False,
                       decoder=source.decoder.to_dict())

    monkeypatch.setattr(config, 'CLASS_NAMES', ['OTHER', 'CLASSES'])
    monkeypatch.setattr(config, 'N_CLASSES', 2)
    with pytest.raises(ValueError, match='was trained on classes'):
        storage.load_classifier(branch='C')


def test_library_name_reaches_the_caller(sandbox):
    """
    The library name lands in the passed meta as well, not only in the file.

    The run panel in the GUI reads it from exactly there. Previously archive_run
    worked with a copy, and after every successful distillation the window said
    "in the library: —" even though the name was written into the artifact.
    """
    meta = meta_with(0.5)
    entry, _ = pipeline.archive_run(trees(), 'A', meta, names(),
                                    name='visible-name')
    assert meta['library_name'] == entry.name == 'visible-name'
