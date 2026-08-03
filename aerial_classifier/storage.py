"""
storage.py — сохранение/загрузка оракулов (PyTorch) и EML-формул (JSON).

АКТИВНЫЕ артефакты (их берут export, verify-c, bench и вкладки теста):
    models/oracle_A.pt        FeatureMLP
    models/oracle_B.pt        PixelCNN
    models/best_eml_A.json    4 формулы ветки A
    models/best_eml_B.json    4 формулы ветки B

БИБЛИОТЕКА — models/library/: именованные копии каждого прогона.

    models/library/oracle_A__20260724-1230_bal0.908.pt
    models/library/eml_A__medium-seed4.json

Зачем библиотека. Раньше каждое обучение затирало единственный слот, и
сравнить две модели можно было только по памяти: «вроде вчера было лучше».
Прогоны при этом стоят от 14 секунд до часа, а разброс GA по сидам больше
разницы между правками — то есть выбор из нескольких сохранённых наборов это
не удобство, а необходимое условие осмысленного сравнения.

Библиотека не заводит индексный файл: список собирается сканированием папки, а
все характеристики читаются из meta самих артефактов. Индекс пришлось бы
чинить каждый раз, когда файл удалили мимо программы.

Лучшая модель определяется ПО VAL — той же метрикой, по которой идёт отбор
внутри прогона. Test для выбора не трогаем нигде: его смотрят один раз, в
самом конце.

Формулы кладутся в JSON, а не в pickle, сознательно: это конечный артефакт
проекта, его должно быть можно прочитать глазами, продиффать в git и
разобрать любым другим языком при переносе на микроконтроллер.
"""

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import torch

import config


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def oracle_path(branch: str) -> str:
    return os.path.join(config.MODELS_DIR, f'oracle_{branch.upper()}.pt')


def formulas_path(branch: str) -> str:
    return os.path.join(config.MODELS_DIR, f'best_eml_{branch.upper()}.json')


# ── Оракул ───────────────────────────────────────────────────────────────────

def save_oracle(model, branch: str = 'A', path: str | None = None,
                meta: dict | None = None) -> str:
    """Сохранить state_dict оракула + метаданные обучения."""
    path = path or oracle_path(branch)
    _ensure_dir(path)
    meta = dict(meta or {})
    meta.setdefault('class_names', list(config.CLASS_NAMES))
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, path)
    return path


def load_oracle(model, branch: str = 'A', path: str | None = None,
                device=None, check_classes: bool = True) -> dict:
    """
    Загрузить веса в существующий экземпляр модели. Возвращает meta.

    Несовпадение набора классов ловится ДО load_state_dict: при другом числе
    классов torch и так упадёт, но сообщением про размерность тензора, из
    которого причина («выбран другой датасет») не читается вовсе. А при том же
    числе классов, но других именах, не упадёт ничего.
    """
    path = path or oracle_path(branch)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"нет чекпоинта оракула {path}. Сначала: python main.py train "
            f"--branch {branch.upper()}")
    ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
    meta = ckpt.get('meta', {})
    if check_classes:
        check_class_names(meta.get('class_names'),
                          f"оракул ветки {branch.upper()}")
    model.load_state_dict(ckpt['state_dict'])
    return meta


def oracle_meta(branch: str = 'A') -> dict:
    """Прочитать только метаданные оракула, не создавая модель."""
    path = oracle_path(branch)
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location='cpu',
                      weights_only=False).get('meta', {})


# ── EML-формулы ──────────────────────────────────────────────────────────────

def save_formulas(trees: list, branch: str = 'A', path: str | None = None,
                  meta: dict | None = None,
                  input_names: list[str] | None = None,
                  calibration: dict | None = None,
                  decoder: dict | None = None) -> str:
    """
    Сохранить EML-деревья как JSON.

    Args:
        trees: ветки A и B — список EMLNode длиной N_CLASSES в порядке
            config.CLASS_NAMES; ветка C — список из одного дерева.
        input_names: имена входов формулы — признаки (ветки A и C) или
            имена координат эмбеддинга (ветка B).
        calibration: {'offsets': [...], 'scales': [...]} — константы выравнивания
            выходов формул (ветки A и B). Без них argmax между формулами не
            имеет смысла, поэтому они часть артефакта, а не отладочная
            информация.
        decoder: таблица порогов ветки C. Ровно та же роль: без неё выход
            формулы это просто число.

    class_names пишутся в артефакт обязательно. Набор классов теперь зависит
    от выбранной папки, и формулы, обученные на четырёх классах, при пяти
    означают не то же самое — молча применить их было бы хуже, чем отказаться.
    """
    path = path or formulas_path(branch)
    _ensure_dir(path)
    if input_names is None:
        input_names = (config.FEATURE_NAMES if branch.upper() != 'B'
                       else [f'e{i}' for i in range(config.CNN_EMBED_DIM)])
    if calibration is None and meta:
        calibration = meta.get('calibration')
    if decoder is None and meta:
        decoder = meta.get('decoder')
    payload = {
        'branch': branch.upper(),
        'class_names': list(config.CLASS_NAMES),
        'input_names': input_names,
        'calibration': calibration,
        'decoder': decoder,
        'trees': [t.to_dict() for t in trees],
        'formulas': [t.to_string() for t in trees],   # человекочитаемая копия
        'meta': meta or {},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def check_class_names(payload_classes, what: str):
    """
    Сверить набор классов артефакта с текущим датасетом.

    Молчаливое несовпадение — самая дорогая ошибка после смены папки: модель
    отработает, метрики посчитаются, и всё будет означать не то. Поэтому не
    предупреждение, а отказ с готовым рецептом.
    """
    if not payload_classes:
        return                                   # старый артефакт без списка
    if list(payload_classes) == list(config.CLASS_NAMES):
        return
    raise ValueError(
        f"{what} обучены на классах {list(payload_classes)}, а сейчас выбран "
        f"датасет с классами {list(config.CLASS_NAMES)}. Либо верните прежний "
        f"датасет (python main.py --dataset <папка> ...), либо обучите заново.")


def formulas_meta(branch: str = 'A', path: str | None = None) -> dict:
    """
    Метаданные сохранённых формул без разбора деревьев ({} если файла нет).

    Нужна, чтобы сравнить новый прогон с уже лежащим и не затереть лучший
    результат худшим (см. pipeline.save_best).
    """
    path = path or formulas_path(branch)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get('meta', {})
    except (json.JSONDecodeError, OSError):
        # Битый файл — как будто его нет: пусть новый прогон его заменит.
        return {}


def load_formulas(node_cls, branch: str = 'A', path: str | None = None):
    """
    Загрузить EML-деревья. node_cls — класс EMLNode (с .from_dict).

    Returns: (trees, payload) — деревья и весь прочитанный JSON.
    """
    path = path or formulas_path(branch)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"нет файла формул {path}. Сначала: python main.py distill "
            f"--branch {branch.upper()}")
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    trees = [node_cls.from_dict(d) for d in payload['trees']]
    return trees, payload


def load_classifier(branch: str = 'A', path: str | None = None,
                    check_classes: bool = True):
    """
    Загрузить готовую модель вместе с её константами.

    Ветки A и B — EMLClassifier с калибровкой, ветка C — EMLSingle с таблицей
    порогов. Отдельная функция, потому что забыть применить константы после
    загрузки — самая вероятная ошибка при использовании артефакта: формулы
    отработают, а ответ молча окажется мусором.
    """
    from eml_distiller import EMLClassifier, EMLNode

    trees, payload = load_formulas(EMLNode, branch=branch, path=path)
    if check_classes:
        check_class_names(payload.get('class_names'),
                          f"формулы ветки {branch.upper()}")

    if payload.get('decoder'):
        from eml_single import EMLSingle, IntervalDecoder
        return EMLSingle(trees[0],
                         IntervalDecoder.from_dict(payload['decoder'])), payload

    calib = payload.get('calibration') or {}
    return EMLClassifier(trees, offsets=calib.get('offsets'),
                         scales=calib.get('scales')), payload


# ── Библиотека именованных моделей ───────────────────────────────────────────

KINDS = ('oracle', 'eml')
_EXT = {'oracle': '.pt', 'eml': '.json'}


def library_dir() -> str:
    return os.path.join(config.MODELS_DIR, 'library')


def slugify(name: str) -> str:
    """
    Имя модели -> безопасное имя файла.

    Пользователь вводит имя руками («ветка A, medium, сид 4»), а оно попадает
    в путь. Пропускать его в файловую систему как есть нельзя: пробелы,
    двоеточия и слэши либо ломают путь, либо создают вложенные папки.
    """
    slug = re.sub(r'[^\w.-]+', '-', name.strip(), flags=re.UNICODE)
    slug = re.sub(r'-{2,}', '-', slug).strip('-.')
    return slug[:80] or 'model'


def auto_name(kind: str, branch: str, meta: dict) -> str:
    """
    Имя по умолчанию: время плюс главная метрика.

    Время — чтобы прогоны не сливались, метрика — чтобы список читался без
    раскрытия каждого пункта.
    """
    score = main_score(kind, meta)
    stamp = time.strftime('%Y%m%d-%H%M')
    return f"{stamp}_bal{score:.3f}" if score is not None else stamp


def main_score(kind: str, meta: dict) -> float | None:
    """
    Главная характеристика для сравнения моделей — сбалансированная accuracy
    на ВАЛИДАЦИИ. Та же метрика, по которой идёт отбор внутри прогона; test
    для выбора не используется нигде.
    """
    if kind == 'oracle':
        value = meta.get('val_bal_acc')
    else:
        value = meta.get('metrics', {}).get('val', {}).get('eml_bal_acc')
    return float(value) if value is not None else None


def _test_score(kind: str, meta: dict) -> float | None:
    if kind == 'oracle':
        value = meta.get('test_bal_acc')
    else:
        value = meta.get('metrics', {}).get('test', {}).get('eml_bal_acc')
    return float(value) if value is not None else None


@dataclass
class ModelEntry:
    """Одна модель в библиотеке вместе с её характеристиками."""

    kind: str                    # 'oracle' | 'eml'
    branch: str
    name: str
    path: str
    meta: dict = field(repr=False, default_factory=dict)
    saved_at: float = 0.0
    is_active: bool = False
    is_best: bool = False

    @property
    def val_score(self) -> float | None:
        return main_score(self.kind, self.meta)

    @property
    def test_score(self) -> float | None:
        return _test_score(self.kind, self.meta)

    @property
    def size(self) -> int | None:
        """Сложность: параметров у оракула, узлов AST у формул."""
        return (self.meta.get('params') if self.kind == 'oracle'
                else self.meta.get('total_nodes'))

    def label(self) -> str:
        """Строка для выпадающего списка: имя, метрика, сложность, пометки."""
        marks = ('★' if self.is_best else ' ') + ('•' if self.is_active else ' ')
        val = f"{self.val_score:.4f}" if self.val_score is not None else "  ?   "
        unit = 'пар.' if self.kind == 'oracle' else 'узл.'
        size = f"{self.size} {unit}" if self.size is not None else ''
        return f"{marks} {self.name:<28} val {val}  {size}"

    def details(self) -> str:
        """Многострочная сводка для панели."""
        lines = [f"имя: {self.name}",
                 f"ветка {self.branch}, "
                 f"{'оракул' if self.kind == 'oracle' else 'формулы'}"]
        if self.is_active:
            lines.append("АКТИВНАЯ")
        if self.is_best:
            lines.append("лучшая по val")
        if self.val_score is not None:
            lines.append(f"val bal:  {self.val_score:.4f}")
        if self.test_score is not None:
            lines.append(f"test bal: {self.test_score:.4f}")
        if self.kind == 'oracle':
            if self.meta.get('params'):
                lines.append(f"параметров: {self.meta['params']:,}")
            if self.meta.get('epochs_run'):
                lines.append(f"эпох: {self.meta['epochs_run']}")
            if self.meta.get('seconds'):
                lines.append(f"обучение: {self.meta['seconds']:.0f} c")
        else:
            if self.meta.get('total_nodes'):
                lines.append(f"узлов AST: {self.meta['total_nodes']}, "
                             f"глубина {self.meta.get('max_depth', '?')}")
            if self.meta.get('n_inputs_used') is not None:
                lines.append(f"входов: {self.meta['n_inputs_used']}"
                             f"/{self.meta.get('n_inputs_available', '?')}")
            fidelity = self.meta.get('metrics', {}).get('test', {}).get('fidelity')
            if fidelity is not None:
                lines.append(f"fidelity: {fidelity:.4f}")
            if self.meta.get('depth_penalty'):
                lines.append(f"штраф: {self.meta['depth_penalty']}, "
                             f"сид {self.meta.get('seed', '?')}")
        lines.append(time.strftime('%Y-%m-%d %H:%M',
                                   time.localtime(self.saved_at)))
        return '\n'.join(lines)


def library_path(kind: str, branch: str, name: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind должен быть 'oracle' или 'eml', получено {kind!r}")
    return os.path.join(library_dir(),
                        f"{kind}_{branch.upper()}__{slugify(name)}{_EXT[kind]}")


def _parse_library_filename(filename: str):
    """'oracle_A__имя.pt' -> ('oracle', 'A', 'имя'); None, если не наше."""
    stem, ext = os.path.splitext(filename)
    for kind in KINDS:
        if ext == _EXT[kind] and stem.startswith(f"{kind}_") and '__' in stem:
            head, name = stem.split('__', 1)
            branch = head[len(kind) + 1:]
            if branch and name:
                return kind, branch.upper(), name
    return None


def active_path(kind: str, branch: str) -> str:
    return oracle_path(branch) if kind == 'oracle' else formulas_path(branch)


def active_name(kind: str, branch: str) -> str | None:
    """
    Имя библиотечной модели, лежащей сейчас в активном слоте.

    Определяется по meta самого активного файла, а не по сравнению содержимого:
    имя записывается туда при сохранении, поэтому активную модель видно даже
    после ручного копирования файла.
    """
    meta = (oracle_meta(branch) if kind == 'oracle' else formulas_meta(branch))
    name = meta.get('library_name')
    return str(name) if name else None


def _read_meta(kind: str, path: str) -> dict:
    try:
        if kind == 'oracle':
            return torch.load(path, map_location='cpu',
                              weights_only=False).get('meta', {})
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get('meta', {})
    except Exception:                                  # noqa: BLE001
        # Битый или чужой файл не должен ронять весь список: показываем его
        # без характеристик, пользователь сам решит удалить.
        return {}


def list_models(kind: str | None = None, branch: str | None = None
                ) -> list[ModelEntry]:
    """
    Всё, что лежит в библиотеке, отсортированное по главной метрике (убывание).

    Модели без метрики уезжают в конец: сравнивать их не с чем.
    """
    directory = library_dir()
    if not os.path.isdir(directory):
        return []

    entries: list[ModelEntry] = []
    for filename in sorted(os.listdir(directory)):
        parsed = _parse_library_filename(filename)
        if parsed is None:
            continue
        entry_kind, entry_branch, name = parsed
        if kind is not None and entry_kind != kind:
            continue
        if branch is not None and entry_branch != branch.upper():
            continue
        path = os.path.join(directory, filename)
        entries.append(ModelEntry(
            kind=entry_kind, branch=entry_branch, name=name, path=path,
            meta=_read_meta(entry_kind, path),
            saved_at=os.path.getmtime(path)))

    entries.sort(key=lambda e: (e.val_score is None,
                                -(e.val_score or 0.0), e.name))

    # Имя активной модели читается по одному разу на пару (вид, ветка), а не
    # на каждую запись: это чтение чекпоинта с диска, а активный файл для всех
    # записей группы один и тот же.
    active: dict[tuple[str, str], str | None] = {}
    for entry in entries:
        key = (entry.kind, entry.branch)
        if key not in active:
            active[key] = active_name(*key)
        entry.is_active = (entry.name == active[key])

    # Пометки считаются внутри каждой пары (вид, ветка): лучший оракул ветки A
    # и лучший набор формул ветки B — разные сущности, соревноваться им не в чем.
    for entry_kind in KINDS:
        for entry_branch in {e.branch for e in entries if e.kind == entry_kind}:
            group = [e for e in entries
                     if e.kind == entry_kind and e.branch == entry_branch
                     and e.val_score is not None]
            if group:
                max(group, key=lambda e: e.val_score).is_best = True
    return entries


def find_model(kind: str, branch: str, name: str) -> ModelEntry | None:
    for entry in list_models(kind, branch):
        if entry.name == name:
            return entry
    return None


def _unique_name(kind: str, branch: str, name: str) -> str:
    """Свободное имя: занятое дополняется -2, -3 и так далее."""
    if not os.path.exists(library_path(kind, branch, name)):
        return name
    for suffix in range(2, 1000):
        candidate = f"{name}-{suffix}"
        if not os.path.exists(library_path(kind, branch, candidate)):
            return candidate
    raise RuntimeError(f"не удалось подобрать свободное имя для {name!r}")


def reserve_name(kind: str, branch: str, name: str | None, meta: dict) -> str:
    """
    Подобрать свободное имя в библиотеке ДО сохранения.

    Порядок важен: имя записывается внутрь артефакта (по нему опознаётся
    активная модель), поэтому уточнять его после записи нельзя — активный файл
    и библиотечная копия разъехались бы по имени, и активная модель перестала
    бы опознаваться.
    """
    return _unique_name(kind, branch.upper(),
                        slugify(name) if name else auto_name(kind, branch, meta))


def add_to_library(kind: str, branch: str, name: str | None, meta: dict,
                   source_path: str | None = None,
                   overwrite: bool = False) -> ModelEntry:
    """
    Положить в библиотеку копию артефакта под именем.

    Копируется уже сохранённый файл, а не сериализуется заново: так
    библиотечная копия гарантированно побитово совпадает с тем, что померили,
    и расхождение между «моделью, которую оценили» и «моделью, которую
    сохранили» невозможно в принципе.

    Занятое имя не затирается молча (кроме overwrite=True): прогоны стоят от
    минут до часа, и потерять чужой результат из-за совпадения имени обиднее,
    чем увидеть в списке «имя-2».
    """
    branch = branch.upper()
    name = slugify(name) if name else auto_name(kind, branch, meta)
    if not overwrite:
        name = _unique_name(kind, branch, name)
    source_path = source_path or active_path(kind, branch)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"нечего копировать в библиотеку: {source_path}")

    target = library_path(kind, branch, name)
    _ensure_dir(target)
    shutil.copyfile(source_path, target)
    return ModelEntry(kind=kind, branch=branch, name=name, path=target,
                      meta=meta, saved_at=os.path.getmtime(target))


def rename_model(entry: ModelEntry, new_name: str) -> ModelEntry:
    """
    Переименовать модель в библиотеке.

    Имя живёт и в пути файла, и в meta (по нему определяется активная модель),
    поэтому меняется в обоих местах — иначе переименованная модель перестала бы
    опознаваться как активная.
    """
    name = _unique_name(entry.kind, entry.branch, slugify(new_name))
    target = library_path(entry.kind, entry.branch, name)
    os.rename(entry.path, target)

    meta = dict(entry.meta)
    meta['library_name'] = name
    _write_library_name(entry.kind, target, name)
    if entry.is_active:
        _write_library_name(entry.kind, active_path(entry.kind, entry.branch),
                            name)
    return ModelEntry(kind=entry.kind, branch=entry.branch, name=name,
                      path=target, meta=meta,
                      saved_at=os.path.getmtime(target), is_best=entry.is_best,
                      is_active=entry.is_active)


def _write_library_name(kind: str, path: str, name: str):
    """Переписать library_name внутри уже сохранённого артефакта."""
    if not os.path.exists(path):
        return
    if kind == 'oracle':
        payload = torch.load(path, map_location='cpu', weights_only=False)
        payload.setdefault('meta', {})['library_name'] = name
        torch.save(payload, path)
        return
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    payload.setdefault('meta', {})['library_name'] = name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def activate(entry: ModelEntry) -> str:
    """
    Сделать библиотечную модель активной: копия уезжает в основной слот.

    Копия, а не символическая ссылка: артефакт должен оставаться обычным
    файлом, который можно унести на другую машину, и не зависеть от прав на
    создание ссылок в Windows.
    """
    target = active_path(entry.kind, entry.branch)
    _ensure_dir(target)
    shutil.copyfile(entry.path, target)
    return target


def delete_model(entry: ModelEntry) -> str:
    """Удалить модель из библиотеки. Активный слот при этом не трогается."""
    os.remove(entry.path)
    return entry.path
