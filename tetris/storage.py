"""
storage.py — saving/loading the oracle (PyTorch) and the EML formulas (JSON).

Oracle:   models/best_ppo.pt    (state_dict + metadata)
Formulas: models/best_eml.json  (6 AST trees + metrics)
"""

import os
import json

import torch

import config


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


# ── Oracle (CNN) ─────────────────────────────────────────────────────────────

def save_oracle(model, path: str | None = None, meta: dict | None = None):
    """Save the oracle state_dict + training metadata."""
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_ppo.pt')
    _ensure_dir(path)
    torch.save({
        'state_dict': model.state_dict(),
        'meta': meta or {},
    }, path)
    return path


def load_oracle(model, path: str | None = None, device=None):
    """Load weights into an existing model instance. Returns meta."""
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_ppo.pt')
    ckpt = torch.load(path, map_location=device or 'cpu')
    model.load_state_dict(ckpt['state_dict'])
    return ckpt.get('meta', {})


# ── EML formulas ─────────────────────────────────────────────────────────────

def save_formulas(trees: list, path: str | None = None,
                  meta: dict | None = None):
    """
    Save the EML trees as JSON (v2: a single afterstate formula in the list).

    Args:
        trees: list of EMLNode, each with .to_dict().
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
    Load the EML trees. node_cls is the EMLNode class (with .from_dict).
    Returns (trees, meta).
    """
    if path is None:
        path = os.path.join(config.MODELS_DIR, 'best_eml.json')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trees = [node_cls.from_dict(d) for d in data['trees']]
    return trees, data.get('meta', {})
