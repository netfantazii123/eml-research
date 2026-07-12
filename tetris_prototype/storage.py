"""
storage.py — сохранение/загрузка оракула (PyTorch) и EML-формул (JSON).

Оракул:   models/best_ppo.pt    (state_dict + метаданные)
Формулы:  models/best_eml.json  (6 AST-деревьев + метрики)
"""

import os
import json

import torch

import config


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


# ── Оракул (CNN) ─────────────────────────────────────────────────────────────

def save_oracle(model, path: str | None = None, meta: dict | None = None):
    """Сохранить state_dict оракула + метаданные обучения."""
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_ppo.pt')
    _ensure_dir(path)
    torch.save({
        'state_dict': model.state_dict(),
        'meta': meta or {},
    }, path)
    return path


def load_oracle(model, path: str | None = None, device=None):
    """Загрузить веса в существующий экземпляр модели. Возвращает meta."""
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_ppo.pt')
    ckpt = torch.load(path, map_location=device or 'cpu')
    model.load_state_dict(ckpt['state_dict'])
    return ckpt.get('meta', {})


# ── EML-формулы ──────────────────────────────────────────────────────────────

def save_formulas(trees: list, path: str | None = None,
                  meta: dict | None = None):
    """
    Сохранить 6 EML-деревьев (по одному на действие) как JSON.

    Args:
        trees: список EMLNode (длиной N_ACTIONS), каждый имеет .to_dict().
    """
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_eml.json')
    _ensure_dir(path)
    data = {
        'trees': [t.to_dict() for t in trees],
        'feature_names': config.FEATURE_NAMES,
        'meta': meta or {},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_formulas(node_cls, path: str | None = None):
    """
    Загрузить EML-деревья. node_cls — класс EMLNode (с .from_dict).
    Возвращает (trees, meta).
    """
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_eml.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trees = [node_cls.from_dict(d) for d in data['trees']]
    return trees, data.get('meta', {})
