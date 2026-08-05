"""
runs.py — dated run directories: a guaranteed full snapshot of an experiment.

Every training/distillation run creates runs/<date_time>_<kind>/ and places
there EVERYTHING needed to reproduce it:
    config_snapshot.json   — all config constants at run time
    <model/formulas>       — copies of the final artifacts
    *_history.json / *.png — history and plots
    summary.json           — final run metrics

The canonical paths (models/best_ppo.pt, models/best_eml.json) keep being
updated — the pipeline and the GUI look at them; the run directory is an
immutable archive.
"""

import os
import json
import time
import shutil

import config

RUNS_DIR = 'runs'


def create_run_dir(kind: str) -> str:
    """Create runs/<YYYY-MM-DD_HH-MM-SS>_<kind>/ and return the path."""
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(RUNS_DIR, f"{stamp}_{kind}")
    os.makedirs(path, exist_ok=True)
    snapshot_config(path)
    return path


def snapshot_config(run_dir: str) -> str:
    """All UPPERCASE config constants -> config_snapshot.json."""
    snap = {}
    for name in dir(config):
        if name.isupper():
            val = getattr(config, name)
            if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                snap[name] = list(val) if isinstance(val, tuple) else val
    path = os.path.join(run_dir, 'config_snapshot.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    return path


def save_summary(run_dir: str, summary: dict) -> str:
    """Final run metrics -> summary.json."""
    path = os.path.join(run_dir, 'summary.json')
    payload = {'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), **summary}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def copy_into(run_dir: str, *paths: str) -> list[str]:
    """Copy files into the run directory (missing ones are skipped silently)."""
    copied = []
    for p in paths:
        if p and os.path.exists(p):
            dst = os.path.join(run_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            copied.append(dst)
    return copied
