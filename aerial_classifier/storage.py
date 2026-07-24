"""
storage.py — сохранение/загрузка оракулов (PyTorch) и EML-формул (JSON).

Оракулы:  models/oracle_A.pt (FeatureMLP), models/oracle_B.pt (PixelCNN)
Формулы:  models/best_eml_A.json, models/best_eml_B.json

Формулы кладутся в JSON, а не в pickle, сознательно: это конечный артефакт
проекта, его должно быть можно прочитать глазами, продиффать в git и
разобрать любым другим языком при переносе на микроконтроллер.
"""

import json
import os

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
    torch.save({'state_dict': model.state_dict(), 'meta': meta or {}}, path)
    return path


def load_oracle(model, branch: str = 'A', path: str | None = None,
                device=None) -> dict:
    """Загрузить веса в существующий экземпляр модели. Возвращает meta."""
    path = path or oracle_path(branch)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"нет чекпоинта оракула {path}. Сначала: python main.py train "
            f"--branch {branch.upper()}")
    ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    return ckpt.get('meta', {})


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
                  calibration: dict | None = None) -> str:
    """
    Сохранить набор EML-деревьев (по одному на класс) как JSON.

    Args:
        trees: список EMLNode длиной N_CLASSES; порядок = config.CLASS_NAMES.
        input_names: имена входов формулы — признаки (ветка A) или
            имена координат эмбеддинга (ветка B).
        calibration: {'offsets': [...], 'scales': [...]} — константы выравнивания
            выходов формул. Без них argmax между формулами не имеет смысла,
            поэтому они часть артефакта, а не отладочная информация.
    """
    path = path or formulas_path(branch)
    _ensure_dir(path)
    if input_names is None:
        input_names = (config.FEATURE_NAMES if branch.upper() == 'A'
                       else [f'e{i}' for i in range(config.CNN_EMBED_DIM)])
    if calibration is None and meta:
        calibration = meta.get('calibration')
    payload = {
        'branch': branch.upper(),
        'class_names': config.CLASS_NAMES,
        'input_names': input_names,
        'calibration': calibration,
        'trees': [t.to_dict() for t in trees],
        'formulas': [t.to_string() for t in trees],   # человекочитаемая копия
        'meta': meta or {},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


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


def load_classifier(branch: str = 'A', path: str | None = None):
    """
    Загрузить готовый EMLClassifier вместе с его калибровкой.

    Отдельная функция, потому что забыть применить калибровку после загрузки —
    самая вероятная ошибка при использовании артефакта: формулы отработают,
    argmax молча выдаст мусор.
    """
    from eml_distiller import EMLClassifier, EMLNode

    trees, payload = load_formulas(EMLNode, branch=branch, path=path)
    calib = payload.get('calibration') or {}
    return EMLClassifier(trees, offsets=calib.get('offsets'),
                         scales=calib.get('scales')), payload
