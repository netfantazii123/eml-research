"""
Tests of the named model library.

The library is where work is easy to lose: names come from the user and end up
in file paths, the active model is determined by a record inside the artifact,
and the runs it is chosen from cost from seconds to an hour. What is checked
first is preservation: that a name does not break the path, that a name
collision does not overwrite someone else's result, that renaming does not lose
the active mark, and that deleting from the library does not touch the active
slot.

storage imports torch — without it the suite is skipped; the other tests do not
depend on torch.
"""

import json
import os

import pytest

pytest.importorskip('torch', reason="storage imports torch")

import torch

import config
import storage


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'MODELS_DIR', str(tmp_path / 'models'))
    return tmp_path


def oracle_meta(val: float, test: float = 0.5, params: int = 13124) -> dict:
    return {'val_bal_acc': val, 'test_bal_acc': test, 'params': params,
            'epochs_run': 42, 'seconds': 14.0}


def eml_meta(val: float, test: float = 0.5, nodes: int = 30) -> dict:
    return {'total_nodes': nodes, 'max_depth': 4, 'n_inputs_used': 10,
            'n_inputs_available': 32,
            'metrics': {'val': {'eml_bal_acc': val},
                        'test': {'eml_bal_acc': test, 'fidelity': 0.7}}}


def write_active(kind: str, branch: str, meta: dict) -> str:
    """Place an artifact into the active slot, as train / distill do."""
    path = storage.active_path(kind, branch)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if kind == 'oracle':
        torch.save({'state_dict': {}, 'meta': meta}, path)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'branch': branch, 'trees': [], 'formulas': [],
                       'input_names': [], 'meta': meta}, f, ensure_ascii=False)
    return path


def publish(kind: str, branch: str, name: str | None, meta: dict):
    """The full save path: the name is reserved before the artifact is written."""
    reserved = storage.reserve_name(kind, branch, name, meta)
    meta = dict(meta, library_name=reserved)
    write_active(kind, branch, meta)
    return storage.add_to_library(kind, branch, reserved, meta, overwrite=True)


# ── Names ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('raw,expected_ok', [
    ('ordinary name', True),
    ('a/b\\c', True),
    ('../../escape', True),
    ('name:with*star?', True),
    ('   ', True),
])
def test_slugify_produces_safe_filename(raw, expected_ok):
    """
    The name comes from the user and ends up in a file path.

    Slashes and colons either break the path or create nested directories, and
    "../.." is an escape outside the library. After slugify the name must be
    exactly one safe path component.
    """
    slug = storage.slugify(raw)
    assert slug, "an empty name is not acceptable"
    assert os.sep not in slug and '/' not in slug
    assert os.path.basename(slug) == slug
    assert '..' not in slug
    assert expected_ok


def test_slugify_keeps_unicode_letters():
    """Non-ASCII letters must not turn into a mush of hyphens."""
    assert storage.slugify('bätch-sëed4') == 'bätch-sëed4'


def test_auto_name_carries_metric():
    name = storage.auto_name('oracle', 'A', oracle_meta(0.908))
    assert 'bal0.908' in name


def test_library_filename_roundtrip():
    path = storage.library_path('eml', 'B', 'my-model')
    parsed = storage._parse_library_filename(os.path.basename(path))
    assert parsed == ('eml', 'B', 'my-model')


# ── Listing and marks ─────────────────────────────────────────────────────────

def test_empty_library(sandbox):
    assert storage.list_models() == []


def test_entries_are_sorted_by_val(sandbox):
    publish('oracle', 'A', 'middle', oracle_meta(0.80))
    publish('oracle', 'A', 'best', oracle_meta(0.91))
    publish('oracle', 'A', 'worst', oracle_meta(0.55))

    names = [e.name for e in storage.list_models('oracle', 'A')]
    assert names == ['best', 'middle', 'worst']


def test_best_is_marked_per_group(sandbox):
    """
    The best is computed within a (kind, branch) pair.

    The branch A oracle and the branch B formula set are different entities with
    different metric ranges and nothing to compete over.
    """
    publish('oracle', 'A', 'oracle-good', oracle_meta(0.91))
    publish('oracle', 'A', 'oracle-bad', oracle_meta(0.70))
    publish('oracle', 'B', 'oracle-B', oracle_meta(0.92))
    publish('eml', 'A', 'formulas-A', eml_meta(0.58))
    publish('eml', 'A', 'formulas-A-worse', eml_meta(0.52))

    best = {(e.kind, e.branch): e.name
            for e in storage.list_models() if e.is_best}
    assert best == {('oracle', 'A'): 'oracle-good',
                    ('oracle', 'B'): 'oracle-B',
                    ('eml', 'A'): 'formulas-A'}


def test_active_is_marked_and_independent_of_best(sandbox):
    """Active and best are different marks: the active one need not be the best."""
    publish('oracle', 'A', 'good', oracle_meta(0.91))
    publish('oracle', 'A', 'fast-and-bad', oracle_meta(0.60))

    entries = {e.name: e for e in storage.list_models('oracle', 'A')}
    assert entries['fast-and-bad'].is_active is True
    assert entries['fast-and-bad'].is_best is False
    assert entries['good'].is_best is True
    assert entries['good'].is_active is False
    assert storage.active_name('oracle', 'A') == 'fast-and-bad'


def test_models_without_metrics_go_last(sandbox):
    publish('oracle', 'A', 'with-metric', oracle_meta(0.5))
    publish('oracle', 'A', 'without-metric', {})
    assert [e.name for e in storage.list_models('oracle', 'A')][-1] == 'without-metric'


def test_corrupt_file_does_not_break_listing(sandbox):
    """A corrupted artifact is shown without characteristics rather than
    breaking the whole list."""
    publish('eml', 'A', 'normal', eml_meta(0.6))
    broken = storage.library_path('eml', 'A', 'broken')
    with open(broken, 'w', encoding='utf-8') as f:
        f.write('{not json')

    entries = storage.list_models('eml', 'A')
    assert {e.name for e in entries} == {'normal', 'broken'}
    assert next(e for e in entries if e.name == 'broken').val_score is None


def test_filters(sandbox):
    publish('oracle', 'A', 'o-A', oracle_meta(0.9))
    publish('eml', 'B', 'f-B', eml_meta(0.9))
    assert [e.name for e in storage.list_models(kind='eml')] == ['f-B']
    assert [e.name for e in storage.list_models(branch='A')] == ['o-A']
    assert storage.find_model('oracle', 'A', 'o-A') is not None
    assert storage.find_model('oracle', 'A', 'no-such') is None


# ── Preservation ──────────────────────────────────────────────────────────────

def test_name_collision_does_not_overwrite(sandbox):
    """
    A name collision must not cost someone else's run.

    Runs take from seconds to an hour, so a taken name is qualified rather than
    overwritten silently.
    """
    first = publish('eml', 'A', 'my-run', eml_meta(0.60))
    second = publish('eml', 'A', 'my-run', eml_meta(0.42))

    assert first.name == 'my-run'
    assert second.name == 'my-run-2'
    names = {e.name for e in storage.list_models('eml', 'A')}
    assert names == {'my-run', 'my-run-2'}
    survived = storage.find_model('eml', 'A', 'my-run')
    assert survived.val_score == pytest.approx(0.60)


def test_library_copy_matches_active_byte_for_byte(sandbox):
    """
    The library copy is the same file that was measured.

    It is copied rather than re-serialized, so a divergence between "the model
    that was evaluated" and "the model that was saved" is impossible.
    """
    entry = publish('eml', 'A', 'check', eml_meta(0.6))
    active = storage.active_path('eml', 'A')
    assert open(entry.path, 'rb').read() == open(active, 'rb').read()


def test_activate_replaces_active_slot(sandbox):
    publish('oracle', 'A', 'first', oracle_meta(0.91))
    publish('oracle', 'A', 'second', oracle_meta(0.60))
    assert storage.active_name('oracle', 'A') == 'second'

    first = storage.find_model('oracle', 'A', 'first')
    storage.activate(first)

    assert storage.active_name('oracle', 'A') == 'first'
    assert storage.oracle_meta('A')['val_bal_acc'] == pytest.approx(0.91)


def test_activate_keeps_library_copy(sandbox):
    """Activation copies rather than moves: the model stays in the library."""
    publish('oracle', 'A', 'first', oracle_meta(0.91))
    publish('oracle', 'A', 'second', oracle_meta(0.60))
    storage.activate(storage.find_model('oracle', 'A', 'first'))
    assert storage.find_model('oracle', 'A', 'first') is not None
    assert storage.find_model('oracle', 'A', 'second') is not None


def test_rename_keeps_active_mark(sandbox):
    """
    The name lives both in the path and inside the artifact.

    Updating only the path would make a renamed model stop being recognized as
    active even though it sits in the active slot.
    """
    publish('eml', 'A', 'old-name', eml_meta(0.6))
    entry = storage.find_model('eml', 'A', 'old-name')
    assert entry.is_active

    renamed = storage.rename_model(entry, 'new name')

    assert renamed.name == 'new-name'
    assert not os.path.exists(entry.path)
    assert storage.active_name('eml', 'A') == 'new-name'
    assert storage.find_model('eml', 'A', 'new-name').is_active is True


def test_rename_of_inactive_does_not_touch_active(sandbox):
    publish('eml', 'A', 'active-one', eml_meta(0.7))
    publish('eml', 'A', 'spare', eml_meta(0.3))
    storage.activate(storage.find_model('eml', 'A', 'active-one'))

    storage.rename_model(storage.find_model('eml', 'A', 'spare'), 'other')
    assert storage.active_name('eml', 'A') == 'active-one'


def test_delete_removes_only_library_copy(sandbox):
    """Deleting from the library must not deprive the project of an active model."""
    entry = publish('oracle', 'A', 'the-only-one', oracle_meta(0.9))
    active = storage.active_path('oracle', 'A')

    storage.delete_model(entry)

    assert not os.path.exists(entry.path)
    assert os.path.exists(active), "the active slot must not be affected"
    assert storage.list_models('oracle', 'A') == []


def test_reserve_name_before_write(sandbox):
    """
    reserve_name picks a free name BEFORE the artifact is written.

    Otherwise the name would have to be adjusted after it was already written
    inside the file, and the active model would stop being recognized.
    """
    publish('eml', 'A', 'taken', eml_meta(0.5))
    assert storage.reserve_name('eml', 'A', 'taken', eml_meta(0.5)) == 'taken-2'
    assert storage.reserve_name('eml', 'A', 'free', eml_meta(0.5)) == 'free'


def test_main_score_uses_validation(sandbox):
    """
    The primary metric is val, not test.

    Selecting by test would mean looking into the held-out sample, and it is
    exactly this metric by which the library marks the best model.
    """
    meta = oracle_meta(val=0.60, test=0.99)
    assert storage.main_score('oracle', meta) == pytest.approx(0.60)
    entry = publish('oracle', 'A', 'check', meta)
    assert entry.val_score == pytest.approx(0.60)
    assert entry.test_score == pytest.approx(0.99)


def test_entry_label_and_details(sandbox):
    entry = publish('eml', 'A', 'mine', eml_meta(0.58, 0.61, nodes=30))
    entries = storage.list_models('eml', 'A')
    label = entries[0].label()
    assert 'mine' in label and '0.5800' in label and '30 nodes' in label
    details = entries[0].details()
    assert 'ACTIVE' in details and 'best by val' in details


# ── Compatibility with the current dataset ────────────────────────────────────

def publish_formulas(name: str, val: float = 0.5):
    """
    Save formulas through the full path — via storage.save_formulas.

    Not via write_active: the class set is written to the top level of the
    artifact there, and without it there is nothing to check dataset
    compatibility against.
    """
    meta = eml_meta(val)
    reserved = storage.reserve_name('eml', 'A', name, meta)
    meta['library_name'] = reserved
    storage.save_formulas([], branch='A', meta=meta, input_names=[])
    return storage.add_to_library('eml', 'A', reserved, meta, overwrite=True)


def test_library_marks_models_from_another_dataset(sandbox, monkeypatch):
    """
    A model with a foreign class set is marked in the list rather than silent.

    With folder selection, models for different tasks sit side by side in the
    library. Without the mark the difference surfaces only on an attempt to
    load, and it looks like a failure rather than "the model is simply from
    another dataset".
    """
    monkeypatch.setattr(config, 'CLASS_NAMES', ['A', 'B', 'C', 'D'])
    publish_formulas('four-class')
    monkeypatch.setattr(config, 'CLASS_NAMES', ['CAT', 'DOG'])
    publish_formulas('two-class')

    entries = {e.name: e for e in storage.list_models('eml', 'A')}
    assert entries['two-class'].fits_dataset is True
    assert entries['four-class'].fits_dataset is False
    assert entries['four-class'].classes == ['A', 'B', 'C', 'D']
    assert '[other dataset]' in entries['four-class'].label()
    assert 'NOT FROM THE CURRENT DATASET' in entries['four-class'].details()


def test_best_ignores_models_from_another_dataset(sandbox, monkeypatch):
    """
    The star does not go to a model of a foreign dataset, even with a higher
    metric.

    The metrics are not comparable: on two classes balanced accuracy starts at
    0.5, and such a model would automatically be "the best" whatever sat nearby.
    """
    monkeypatch.setattr(config, 'CLASS_NAMES', ['CAT', 'DOG'])
    publish_formulas('two-class', val=0.95)
    monkeypatch.setattr(config, 'CLASS_NAMES', ['A', 'B', 'C', 'D'])
    publish_formulas('own-weak', val=0.60)

    entries = {e.name: e for e in storage.list_models('eml', 'A')}
    assert entries['own-weak'].is_best is True
    assert entries['two-class'].is_best is False


def test_old_artifact_without_class_list_is_not_rejected(sandbox):
    """
    An artifact saved before dataset selection existed counts as applicable.

    It holds no class list, and there is nothing to decide about it — declaring
    such a file foreign would devalue the entire previous library with one
    update.
    """
    publish('eml', 'A', 'old-model', eml_meta(0.7))
    entry = storage.find_model('eml', 'A', 'old-model')
    assert entry.classes is None
    assert entry.fits_dataset is True
    assert '[other dataset]' not in entry.label()
