"""
Тесты библиотеки именованных моделей.

Библиотека — это то место, где легко потерять работу: имена приходят от
пользователя и уезжают в пути файлов, активная модель определяется по записи
внутри артефакта, а прогоны, из которых её выбирают, стоят от секунд до часа.
Поэтому проверяется в первую очередь сохранность: что имя не разрушит путь,
что совпадение имён не затрёт чужой результат, что переименование не потеряет
пометку активной и что удаление из библиотеки не трогает активный слот.

storage импортирует torch — без него набор пропускается; остальные тесты от
torch не зависят.
"""

import json
import os

import pytest

pytest.importorskip('torch', reason="storage импортирует torch")

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
    """Положить артефакт в активный слот, как это делают train / distill."""
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
    """Полный путь сохранения: имя резервируется до записи артефакта."""
    reserved = storage.reserve_name(kind, branch, name, meta)
    meta = dict(meta, library_name=reserved)
    write_active(kind, branch, meta)
    return storage.add_to_library(kind, branch, reserved, meta, overwrite=True)


# ── Имена ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('raw,expected_ok', [
    ('обычное имя', True),
    ('a/b\\c', True),
    ('../../побег', True),
    ('имя:с*звёздочкой?', True),
    ('   ', True),
])
def test_slugify_produces_safe_filename(raw, expected_ok):
    """
    Имя приходит от пользователя и попадает в путь файла.

    Слэши и двоеточия либо ломают путь, либо создают вложенные папки, а
    «../..» — это выход за пределы библиотеки. После slugify имя обязано быть
    ровно одним безопасным компонентом пути.
    """
    slug = storage.slugify(raw)
    assert slug, "пустое имя недопустимо"
    assert os.sep not in slug and '/' not in slug
    assert os.path.basename(slug) == slug
    assert '..' not in slug
    assert expected_ok


def test_slugify_keeps_cyrillic():
    """Имена на русском не должны превращаться в кашу из дефисов."""
    assert storage.slugify('батч-сид4') == 'батч-сид4'


def test_auto_name_carries_metric():
    name = storage.auto_name('oracle', 'A', oracle_meta(0.908))
    assert 'bal0.908' in name


def test_library_filename_roundtrip():
    path = storage.library_path('eml', 'B', 'моя-модель')
    parsed = storage._parse_library_filename(os.path.basename(path))
    assert parsed == ('eml', 'B', 'моя-модель')


# ── Список и пометки ──────────────────────────────────────────────────────────

def test_empty_library(sandbox):
    assert storage.list_models() == []


def test_entries_are_sorted_by_val(sandbox):
    publish('oracle', 'A', 'середина', oracle_meta(0.80))
    publish('oracle', 'A', 'лучшая', oracle_meta(0.91))
    publish('oracle', 'A', 'худшая', oracle_meta(0.55))

    names = [e.name for e in storage.list_models('oracle', 'A')]
    assert names == ['лучшая', 'середина', 'худшая']


def test_best_is_marked_per_group(sandbox):
    """
    Лучшая считается внутри пары (вид, ветка).

    Оракул ветки A и набор формул ветки B — разные сущности с разными
    диапазонами метрик, соревноваться им не в чем.
    """
    publish('oracle', 'A', 'оракул-хороший', oracle_meta(0.91))
    publish('oracle', 'A', 'оракул-плохой', oracle_meta(0.70))
    publish('oracle', 'B', 'оракул-B', oracle_meta(0.92))
    publish('eml', 'A', 'формулы-A', eml_meta(0.58))
    publish('eml', 'A', 'формулы-A-хуже', eml_meta(0.52))

    best = {(e.kind, e.branch): e.name
            for e in storage.list_models() if e.is_best}
    assert best == {('oracle', 'A'): 'оракул-хороший',
                    ('oracle', 'B'): 'оракул-B',
                    ('eml', 'A'): 'формулы-A'}


def test_active_is_marked_and_independent_of_best(sandbox):
    """Активная и лучшая — разные пометки: активной может быть и не лучшая."""
    publish('oracle', 'A', 'хорошая', oracle_meta(0.91))
    publish('oracle', 'A', 'быстрая-и-плохая', oracle_meta(0.60))

    entries = {e.name: e for e in storage.list_models('oracle', 'A')}
    assert entries['быстрая-и-плохая'].is_active is True
    assert entries['быстрая-и-плохая'].is_best is False
    assert entries['хорошая'].is_best is True
    assert entries['хорошая'].is_active is False
    assert storage.active_name('oracle', 'A') == 'быстрая-и-плохая'


def test_models_without_metrics_go_last(sandbox):
    publish('oracle', 'A', 'с-метрикой', oracle_meta(0.5))
    publish('oracle', 'A', 'без-метрики', {})
    assert [e.name for e in storage.list_models('oracle', 'A')][-1] == 'без-метрики'


def test_corrupt_file_does_not_break_listing(sandbox):
    """Битый артефакт показывается без характеристик, а не роняет весь список."""
    publish('eml', 'A', 'нормальная', eml_meta(0.6))
    broken = storage.library_path('eml', 'A', 'битая')
    with open(broken, 'w', encoding='utf-8') as f:
        f.write('{не json')

    entries = storage.list_models('eml', 'A')
    assert {e.name for e in entries} == {'нормальная', 'битая'}
    assert next(e for e in entries if e.name == 'битая').val_score is None


def test_filters(sandbox):
    publish('oracle', 'A', 'о-A', oracle_meta(0.9))
    publish('eml', 'B', 'ф-B', eml_meta(0.9))
    assert [e.name for e in storage.list_models(kind='eml')] == ['ф-B']
    assert [e.name for e in storage.list_models(branch='A')] == ['о-A']
    assert storage.find_model('oracle', 'A', 'о-A') is not None
    assert storage.find_model('oracle', 'A', 'нет-такой') is None


# ── Сохранность ───────────────────────────────────────────────────────────────

def test_name_collision_does_not_overwrite(sandbox):
    """
    Совпадение имён не должно стоить чужого прогона.

    Прогоны идут от секунд до часа, поэтому занятое имя уточняется, а не
    затирается молча.
    """
    first = publish('eml', 'A', 'мой-прогон', eml_meta(0.60))
    second = publish('eml', 'A', 'мой-прогон', eml_meta(0.42))

    assert first.name == 'мой-прогон'
    assert second.name == 'мой-прогон-2'
    names = {e.name for e in storage.list_models('eml', 'A')}
    assert names == {'мой-прогон', 'мой-прогон-2'}
    survived = storage.find_model('eml', 'A', 'мой-прогон')
    assert survived.val_score == pytest.approx(0.60)


def test_library_copy_matches_active_byte_for_byte(sandbox):
    """
    Копия в библиотеке — тот же файл, что померили.

    Она копируется, а не сериализуется заново, поэтому расхождение между
    «моделью, которую оценили» и «моделью, которую сохранили» невозможно.
    """
    entry = publish('eml', 'A', 'проверка', eml_meta(0.6))
    active = storage.active_path('eml', 'A')
    assert open(entry.path, 'rb').read() == open(active, 'rb').read()


def test_activate_replaces_active_slot(sandbox):
    publish('oracle', 'A', 'первая', oracle_meta(0.91))
    publish('oracle', 'A', 'вторая', oracle_meta(0.60))
    assert storage.active_name('oracle', 'A') == 'вторая'

    first = storage.find_model('oracle', 'A', 'первая')
    storage.activate(first)

    assert storage.active_name('oracle', 'A') == 'первая'
    assert storage.oracle_meta('A')['val_bal_acc'] == pytest.approx(0.91)


def test_activate_keeps_library_copy(sandbox):
    """Активация копирует, а не переносит: модель остаётся в библиотеке."""
    publish('oracle', 'A', 'первая', oracle_meta(0.91))
    publish('oracle', 'A', 'вторая', oracle_meta(0.60))
    storage.activate(storage.find_model('oracle', 'A', 'первая'))
    assert storage.find_model('oracle', 'A', 'первая') is not None
    assert storage.find_model('oracle', 'A', 'вторая') is not None


def test_rename_keeps_active_mark(sandbox):
    """
    Имя живёт и в пути, и внутри артефакта.

    Обновить только путь — и переименованная модель перестанет опознаваться
    как активная, хотя лежит в активном слоте.
    """
    publish('eml', 'A', 'старое-имя', eml_meta(0.6))
    entry = storage.find_model('eml', 'A', 'старое-имя')
    assert entry.is_active

    renamed = storage.rename_model(entry, 'новое имя')

    assert renamed.name == 'новое-имя'
    assert not os.path.exists(entry.path)
    assert storage.active_name('eml', 'A') == 'новое-имя'
    assert storage.find_model('eml', 'A', 'новое-имя').is_active is True


def test_rename_of_inactive_does_not_touch_active(sandbox):
    publish('eml', 'A', 'активная', eml_meta(0.7))
    publish('eml', 'A', 'запасная', eml_meta(0.3))
    storage.activate(storage.find_model('eml', 'A', 'активная'))

    storage.rename_model(storage.find_model('eml', 'A', 'запасная'), 'другая')
    assert storage.active_name('eml', 'A') == 'активная'


def test_delete_removes_only_library_copy(sandbox):
    """Удаление из библиотеки не должно лишать проект активной модели."""
    entry = publish('oracle', 'A', 'единственная', oracle_meta(0.9))
    active = storage.active_path('oracle', 'A')

    storage.delete_model(entry)

    assert not os.path.exists(entry.path)
    assert os.path.exists(active), "активный слот не должен пострадать"
    assert storage.list_models('oracle', 'A') == []


def test_reserve_name_before_write(sandbox):
    """
    reserve_name подбирает свободное имя ДО записи артефакта.

    Иначе имя пришлось бы уточнять после того, как оно уже записано внутрь
    файла, и активная модель перестала бы опознаваться.
    """
    publish('eml', 'A', 'занято', eml_meta(0.5))
    assert storage.reserve_name('eml', 'A', 'занято', eml_meta(0.5)) == 'занято-2'
    assert storage.reserve_name('eml', 'A', 'свободно', eml_meta(0.5)) == 'свободно'


def test_main_score_uses_validation(sandbox):
    """
    Главная метрика — val, не test.

    Отбор по test означал бы подглядывание в отложенную выборку, а именно по
    этой метрике библиотека помечает лучшую модель.
    """
    meta = oracle_meta(val=0.60, test=0.99)
    assert storage.main_score('oracle', meta) == pytest.approx(0.60)
    entry = publish('oracle', 'A', 'проверка', meta)
    assert entry.val_score == pytest.approx(0.60)
    assert entry.test_score == pytest.approx(0.99)


def test_entry_label_and_details(sandbox):
    entry = publish('eml', 'A', 'моя', eml_meta(0.58, 0.61, nodes=30))
    entries = storage.list_models('eml', 'A')
    label = entries[0].label()
    assert 'моя' in label and '0.5800' in label and '30 узл.' in label
    details = entries[0].details()
    assert 'АКТИВНАЯ' in details and 'лучшая по val' in details


# ── Совместимость с текущим датасетом ─────────────────────────────────────────

def publish_formulas(name: str, val: float = 0.5):
    """
    Сохранить формулы полным путём — через storage.save_formulas.

    Не через write_active: набор классов пишется на верхний уровень артефакта
    именно там, а без него проверять совместимость с датасетом нечем.
    """
    meta = eml_meta(val)
    reserved = storage.reserve_name('eml', 'A', name, meta)
    meta['library_name'] = reserved
    storage.save_formulas([], branch='A', meta=meta, input_names=[])
    return storage.add_to_library('eml', 'A', reserved, meta, overwrite=True)


def test_library_marks_models_from_another_dataset(sandbox, monkeypatch):
    """
    Модель с чужим набором классов помечена в списке, а не молчит.

    С выбором папки в библиотеке соседствуют модели разных задач. Без пометки
    разница обнаружится только при попытке загрузить, и выглядеть это будет
    как поломка, а не как «модель просто от другого датасета».
    """
    monkeypatch.setattr(config, 'CLASS_NAMES', ['A', 'B', 'C', 'D'])
    publish_formulas('четырёхклассовая')
    monkeypatch.setattr(config, 'CLASS_NAMES', ['КОТ', 'ПЁС'])
    publish_formulas('двухклассовая')

    entries = {e.name: e for e in storage.list_models('eml', 'A')}
    assert entries['двухклассовая'].fits_dataset is True
    assert entries['четырёхклассовая'].fits_dataset is False
    assert entries['четырёхклассовая'].classes == ['A', 'B', 'C', 'D']
    assert '[другой датасет]' in entries['четырёхклассовая'].label()
    assert 'НЕ ОТ ТЕКУЩЕГО ДАТАСЕТА' in entries['четырёхклассовая'].details()


def test_best_ignores_models_from_another_dataset(sandbox, monkeypatch):
    """
    Звёздочка не достаётся модели чужого датасета, даже если её метрика выше.

    Метрики несравнимы: на двух классах сбалансированная accuracy стартует с
    0.5, и «лучшей» такая модель оказывалась бы автоматически, что бы ни
    лежало рядом.
    """
    monkeypatch.setattr(config, 'CLASS_NAMES', ['КОТ', 'ПЁС'])
    publish_formulas('двухклассовая', val=0.95)
    monkeypatch.setattr(config, 'CLASS_NAMES', ['A', 'B', 'C', 'D'])
    publish_formulas('своя-слабая', val=0.60)

    entries = {e.name: e for e in storage.list_models('eml', 'A')}
    assert entries['своя-слабая'].is_best is True
    assert entries['двухклассовая'].is_best is False


def test_old_artifact_without_class_list_is_not_rejected(sandbox):
    """
    Артефакт, сохранённый до появления выбора датасета, считается подходящим.

    Списка классов в нём нет, и решить о нём нечего — объявить такой файл
    чужим значило бы обесценить всю прежнюю библиотеку одним обновлением.
    """
    publish('eml', 'A', 'старая-модель', eml_meta(0.7))
    entry = storage.find_model('eml', 'A', 'старая-модель')
    assert entry.classes is None
    assert entry.fits_dataset is True
    assert '[другой датасет]' not in entry.label()
