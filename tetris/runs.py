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
import hashlib
import subprocess

import config

RUNS_DIR = 'runs'


def create_run_dir(kind: str) -> str:
    """Create runs/<YYYY-MM-DD_HH-MM-SS>_<kind>/ and return the path."""
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(RUNS_DIR, f"{stamp}_{kind}")
    os.makedirs(path, exist_ok=True)
    snapshot_config(path)
    save_provenance(path)
    return path


def config_dict() -> dict:
    """All UPPERCASE config constants, JSON-serializable."""
    snap = {}
    for name in dir(config):
        if name.isupper():
            val = getattr(config, name)
            if isinstance(val, (int, float, str, bool, list, dict, tuple)):
                snap[name] = list(val) if isinstance(val, tuple) else val
    return snap


def config_hash() -> str:
    """12 hex chars over the whole config — the run's parameter fingerprint."""
    blob = json.dumps(config_dict(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()[:12]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(('git', *args), capture_output=True, text=True,
                             timeout=10, cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def provenance() -> dict:
    """
    What it takes to find this result again: commit, working-tree state, config
    fingerprint. `dirty` matters more than the commit — a run made on uncommitted
    edits is not reproducible from the hash alone, and that has to be visible in
    the result rather than remembered.
    """
    commit = _git('rev-parse', '--short', 'HEAD')
    status = _git('status', '--porcelain')
    return {
        'commit': commit,
        'dirty': bool(status) if status is not None else None,
        'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'config_hash': config_hash(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def save_provenance(run_dir: str) -> str:
    path = os.path.join(run_dir, 'provenance.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(provenance(), f, indent=2, ensure_ascii=False)
    return path


def snapshot_config(run_dir: str) -> str:
    """All UPPERCASE config constants -> config_snapshot.json."""
    path = os.path.join(run_dir, 'config_snapshot.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict(), f, indent=2, ensure_ascii=False)
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
