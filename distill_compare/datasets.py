"""
datasets.py — tabular benchmarks for the classical distillation methods.

Three are bundled with scikit-learn; four are the standard UCI sets, fetched
from a plain-CSV mirror (github.com/jbrownlee/Datasets) and pinned by SHA-256
so a changed upstream file fails loudly instead of silently changing results.
Downloads land in distill_compare/data/ (ignored by git).

Every dataset is returned as (X float64, y int64, feature_names). The labels y
are used ONLY to train the teacher and to report the teacher's own accuracy;
students see the teacher's logits, never y.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_MIRROR = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/'

# name -> (file, sha256, label column handling)
_REMOTE = {
    'banknote': ('banknote_authentication.csv',
                 'd0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9'),
    'pima': ('pima-indians-diabetes.csv',
             '6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af'),
    'ionosphere': ('ionosphere.csv',
                   'fd6dd7864b55d56dac0a1e6e24af9ccc35bf2555ac79af8ab9f3d1daa065ab83'),
    'wine_quality': ('winequality-white.csv',
                     '659d419fff887f225bf977d20520bb64a64cae203e460087f809721d4430ba27'),
}

TABULAR = ('banknote', 'pima', 'ionosphere', 'breast_cancer', 'wine',
           'wine_quality', 'digits')


def _fetch(fname: str, sha: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        urllib.request.urlretrieve(_MIRROR + fname, path)
    h = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if h != sha:
        raise RuntimeError(f"{fname}: sha256 {h} != pinned {sha}; "
                           f"delete {path} and check the mirror")
    return path


def load(name: str):
    """(X, y, feature_names) for a benchmark name from TABULAR."""
    if name in ('breast_cancer', 'wine', 'digits'):
        from sklearn import datasets as skd
        d = {'breast_cancer': skd.load_breast_cancer, 'wine': skd.load_wine,
             'digits': skd.load_digits}[name]()
        names = list(getattr(d, 'feature_names', [])) or \
            [f'x{i}' for i in range(d.data.shape[1])]
        return d.data.astype(np.float64), d.target.astype(np.int64), names
    if name not in _REMOTE:
        raise ValueError(f"unknown dataset {name!r} (have {TABULAR})")
    fname, sha = _REMOTE[name]
    path = _fetch(fname, sha)
    rows = [ln.strip().split(',') for ln in open(path) if ln.strip()]
    if name == 'wine_quality' and not _is_number(rows[0][0]):
        rows = rows[1:]
    X = np.array([[float(v) for v in r[:-1]] for r in rows])
    raw = [r[-1] for r in rows]
    if name == 'wine_quality':
        # Quality 3..9 is heavily imbalanced (5 and 6 are 75 % of the rows);
        # binned into the usual low (<=5) / mid (6) / high (>=7) classes.
        q = np.array([int(float(v)) for v in raw])
        y = np.where(q <= 5, 0, np.where(q == 6, 1, 2))
    else:
        classes = sorted(set(raw))
        y = np.array([classes.index(v) for v in raw])
    return X, y.astype(np.int64), [f'x{i}' for i in range(X.shape[1])]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def split(X, y, seed: int, fractions=(0.6, 0.2, 0.2)):
    """Stratified train / val / test index arrays."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=fractions[0], stratify=y,
                                random_state=seed)
    va, te = train_test_split(rest, train_size=fractions[1] / (1 - fractions[0]),
                              stratify=y[rest], random_state=seed)
    return tr, va, te
