"""
Тесты защиты сохранённых формул от молчаливого регресса.

Эта логика срабатывает в самом конце многочасового прогона, поэтому ошибка в
ней обходится дороже всего: заметить, что лучший набор формул затёрли худшим,
можно только по метрикам в отчёте, и то если помнить прежние.

pipeline тянет за собой torch (через oracle.py), поэтому без него тест
пропускается — остальной набор от torch не зависит и идёт как обычно.
"""

import json
import os

import numpy as np
import pytest

pytest.importorskip('torch', reason="pipeline импортирует oracle -> torch")

import config
import eml_distiller as dist
import pipeline
import storage


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Свои models/ и runs/, чтобы тест не трогал настоящие артефакты."""
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


# ── Продвижение лучшего ───────────────────────────────────────────────────────

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
    Главное свойство: неудачный прогон не должен стирать хороший.

    Раньше любой прогон с save=True (умолчание и в GUI, и в CLI) перезаписывал
    models/best_eml_A.json безусловно.
    """
    pipeline.save_best(trees(), 'A', meta_with(0.61), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.44), names(),
                                verbose=False)
    assert result['promoted'] is False
    assert result['best_path'] is None
    assert stored_val() == pytest.approx(0.61), "лучший набор затёрт худшим"


def test_equal_run_replaces_best(sandbox):
    """Равный по val прогон проходит: свежий результат при прочих равных лучше."""
    pipeline.save_best(trees(), 'A', meta_with(0.55), names(), verbose=False)
    assert pipeline.save_best(trees(), 'A', meta_with(0.55), names(),
                              verbose=False)['promoted'] is True


def test_stopped_run_never_becomes_best(sandbox):
    """Прерванный вручную прогон не претендует на лучший даже с высоким val."""
    pipeline.save_best(trees(), 'A', meta_with(0.50), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.99, stopped=True),
                                names(), verbose=False)
    assert result['promoted'] is False
    assert 'прерван' in result['reason']
    assert stored_val() == pytest.approx(0.50)


def test_stopped_run_is_not_best_even_when_none_stored(sandbox):
    result = pipeline.save_best(trees(), 'A', meta_with(0.9, stopped=True),
                                names(), verbose=False)
    assert result['promoted'] is False
    assert not os.path.exists(storage.formulas_path('A'))


# ── Ничего не теряется ────────────────────────────────────────────────────────

def test_every_run_is_saved_to_runs(sandbox):
    """Даже отвергнутый прогон полностью лежит в runs/ — ничего не пропадает."""
    pipeline.save_best(trees(), 'A', meta_with(0.61), names(), verbose=False)
    result = pipeline.save_best(trees(), 'A', meta_with(0.30), names(),
                                verbose=False)

    assert os.path.exists(result['run_path'])
    payload = json.load(open(result['run_path'], encoding='utf-8'))
    assert payload['meta']['metrics']['val']['eml_bal_acc'] == pytest.approx(0.30)
    assert len(payload['trees']) == config.N_CLASSES


def test_previous_best_is_backed_up(sandbox):
    """Перед заменой прежний файл остаётся рядом как .prev.json."""
    pipeline.save_best(trees(), 'A', meta_with(0.50), names(), verbose=False)
    pipeline.save_best(trees(), 'A', meta_with(0.70), names(), verbose=False)

    backup = storage.formulas_path('A').replace('.json', '.prev.json')
    assert os.path.exists(backup)
    assert (json.load(open(backup, encoding='utf-8'))
            ['meta']['metrics']['val']['eml_bal_acc']) == pytest.approx(0.50)


def test_branches_do_not_interfere(sandbox):
    """Ветка B не должна мешать ветке A: у них разные файлы и свои сравнения."""
    pipeline.save_best(trees(), 'A', meta_with(0.60), names(), verbose=False)
    result = pipeline.save_best(trees(), 'B', meta_with(0.40), names(),
                                verbose=False)
    assert result['promoted'] is True
    assert stored_val('A') == pytest.approx(0.60)
    assert stored_val('B') == pytest.approx(0.40)


def test_corrupt_best_file_is_replaced(sandbox):
    """Битый JSON не должен блокировать сохранение — считаем, что его нет."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(storage.formulas_path('A'), 'w', encoding='utf-8') as f:
        f.write('{не json')

    result = pipeline.save_best(trees(), 'A', meta_with(0.33), names(),
                                verbose=False)
    assert result['promoted'] is True
    assert stored_val() == pytest.approx(0.33)


def test_saved_formulas_reload_into_working_classifier(sandbox):
    """
    Сохранённый артефакт должен загружаться обратно и считать то же самое —
    иначе на микроконтроллер уедет не то, что показывал отчёт.
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


# ── Ветка C ───────────────────────────────────────────────────────────────────

def single_model():
    import eml_single as single

    tree = dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=20),
                        right=dist.EMLNode('var', var_idx=3))
    decoder = single.IntervalDecoder([-1.0, 0.5, 3.0], [0, 2, 1, 2], 4,
                                     score=0.61)
    return single.EMLSingle(tree, decoder)


def test_single_formula_round_trip(sandbox):
    """
    Формула ветки C возвращается из JSON вместе со своей таблицей порогов.

    Забыть таблицу — то же самое, что забыть калибровку в ветке A: формула
    отработает, а класс молча окажется мусором. Поэтому она часть артефакта.
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
    """Активные слоты веток A, B и C независимы."""
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
    Модель, обученную на других классах, применять нельзя.

    После смены папки датасета набор классов другой, а формулы отработают как
    ни в чём не бывало: метрики посчитаются, отчёт напечатается, и всё это
    будет означать не то. Отказ с рецептом дешевле молчаливой бессмыслицы.
    """
    source = single_model()
    meta = meta_with(0.5)
    meta['branch'] = 'C'
    pipeline.save_best([source.tree], 'C', meta, names(), verbose=False,
                       decoder=source.decoder.to_dict())

    monkeypatch.setattr(config, 'CLASS_NAMES', ['ДРУГИЕ', 'КЛАССЫ'])
    monkeypatch.setattr(config, 'N_CLASSES', 2)
    with pytest.raises(ValueError, match='обучены на классах'):
        storage.load_classifier(branch='C')


def test_library_name_reaches_the_caller(sandbox):
    """
    Имя в библиотеке попадает и в переданный meta, а не только в файл.

    Панель прогона в интерфейсе читает его именно оттуда. Раньше archive_run
    работал с копией, и после каждой успешной дистилляции в окне значилось
    «в библиотеке: —» — при том, что имя в самом артефакте было записано.
    """
    meta = meta_with(0.5)
    entry, _ = pipeline.archive_run(trees(), 'A', meta, names(),
                                    name='видимое-имя')
    assert meta['library_name'] == entry.name == 'видимое-имя'
