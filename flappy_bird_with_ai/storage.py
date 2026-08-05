"""
Storage — a single point for saving/loading models, formulas and logs.

Naming convention:
    {method}_{YYYY-MM-DD}_{HH-MM-SS}_score-{N}.{ext}

    Examples:
        ga_2026-05-24_14-30-15_score-42.pt
        eml-medium_2026-05-24_14-31-02_score-37.json
        benchmark_2026-05-24_14-30-15.json
        train_2026-05-24_14-30-15.log

Directory layout:
    models/   — *.pt (GANet) and *.json (EML formulas)
    results/  — benchmark_*.json (metrics) and benchmark_*.png (plots)
    logs/     — train_*.log (text logs of the runs)
"""

import os
import re
import json
import glob
import time
from datetime import datetime
from typing import Optional

import torch

from config import MODELS_DIR, RESULTS_DIR
from eml_distiller import EMLNode, EMLAgent
from ga_net import GANet


LOGS_DIR = 'logs'


# ── File names ──────────────────────────────────────────────────────────────

def make_timestamp(dt: Optional[datetime] = None) -> str:
    """The current time as '2026-05-24_14-30-15'."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y-%m-%d_%H-%M-%S')


def make_run_name(method: str, score: Optional[int] = None,
                  timestamp: Optional[str] = None) -> str:
    """
    File name without an extension: 'ga_2026-05-24_14-30-15_score-42'.
    If score=None, 'noscore' is appended instead.
    """
    ts = timestamp if timestamp is not None else make_timestamp()
    score_part = f"score-{int(score)}" if score is not None else "noscore"
    return f"{method}_{ts}_{score_part}"


_RUN_NAME_RE = re.compile(
    r'^(?P<method>[a-zA-Z0-9-]+)_'
    r'(?P<date>\d{4}-\d{2}-\d{2})_'
    r'(?P<time>\d{2}-\d{2}-\d{2})_'
    r'(?P<score>score-\d+|noscore)$'
)


def parse_run_name(stem: str) -> Optional[dict]:
    """Parse a file name. Returns a dict or None."""
    m = _RUN_NAME_RE.match(stem)
    if not m:
        return None
    sp = m.group('score')
    score = None if sp == 'noscore' else int(sp.split('-')[1])
    return {
        'method': m.group('method'),
        'date': m.group('date'),
        'time': m.group('time').replace('-', ':'),
        'timestamp': f"{m.group('date')}_{m.group('time')}",
        'score': score,
    }


# ── GA models ───────────────────────────────────────────────────────────────

def save_ga_model(net: GANet, score: int,
                  metadata: Optional[dict] = None,
                  timestamp: Optional[str] = None) -> str:
    """
    Save a GA network as .pt plus a sidecar .meta.json with the metadata.
    Returns the path to the .pt file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = timestamp if timestamp is not None else make_timestamp()
    name = make_run_name('ga', score, ts)
    pt_path = os.path.join(MODELS_DIR, name + '.pt')
    meta_path = os.path.join(MODELS_DIR, name + '.meta.json')

    torch.save(net.state_dict(), pt_path)

    meta = {
        'method': 'GA',
        'timestamp': ts,
        'score': int(score),
        'arch': '4-16-1',
        'params': sum(p.numel() for p in net.parameters()),
    }
    if metadata:
        meta.update(metadata)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return pt_path


def load_ga_model(path: str) -> GANet:
    """Load a GA network from a .pt file."""
    net = GANet()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def load_ga_meta(pt_path: str) -> dict:
    """Load the GA model metadata. If there is none, return an empty dict."""
    meta_path = pt_path.replace('.pt', '.meta.json')
    if not os.path.exists(meta_path):
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        parsed = parse_run_name(stem)
        return parsed or {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── EML formulas ────────────────────────────────────────────────────────────

def save_eml_formula(
    tree: EMLNode,
    score: int,
    *,
    mode: str,
    depth_penalty: str,
    test_scores: Optional[list] = None,
    history: Optional[list] = None,
    elapsed: Optional[float] = None,
    dataset_size: Optional[int] = None,
    oracle_ref: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Save an EML formula as JSON. Returns the path to the file.

    Contents:
        method        — 'eml-{depth_penalty}'
        timestamp, score
        mode          — 'sigmoid' | 'binary'
        depth_penalty — 'weak' | 'medium' | 'strong'
        formula       — a human-readable string
        tree          — the AST as a dict
        depth, size, n_vars
        test_scores   — list of scores over the test games
        history       — evolution history (optional)
        elapsed       — seconds
        dataset_size, oracle_ref — for traceability
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = timestamp if timestamp is not None else make_timestamp()
    method = f"eml-{depth_penalty}"
    name = make_run_name(method, score, ts)
    path = os.path.join(MODELS_DIR, name + '.json')

    payload = {
        'method': method,
        'timestamp': ts,
        'score': int(score),
        'mode': mode,
        'depth_penalty': depth_penalty,
        'formula': tree.to_string(),
        'depth': tree.depth(),
        'size': tree.size(),
        'n_vars': tree.n_unique_vars(),
        'tree': tree.to_dict(),
        'test_scores': list(test_scores) if test_scores is not None else None,
        'elapsed': elapsed,
        'dataset_size': dataset_size,
        'oracle_ref': oracle_ref,
        'history': history,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_eml_formula(path: str) -> dict:
    """Load an EML formula JSON and restore the AST into the 'tree_node' key."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['tree_node'] = EMLNode.from_dict(data['tree'])
    data['agent'] = EMLAgent(data['tree_node'], name=data.get('method', 'EML'))
    return data


# ── Listing of saved runs ───────────────────────────────────────────────────

def list_saved_runs() -> list[dict]:
    """
    All saved models and formulas, sorted by time (newest first).

    Every entry:
        {
            'kind': 'ga' | 'eml',
            'path': str,
            'method': str,            # 'GA' | 'eml-weak' | ...
            'timestamp': str,         # '2026-05-24_14-30-15'
            'date': str, 'time': str,
            'score': int | None,
            'extra': dict,            # method-specific fields
        }
    """
    runs = []
    if not os.path.isdir(MODELS_DIR):
        return runs

    for path in glob.glob(os.path.join(MODELS_DIR, '*.pt')):
        stem = os.path.splitext(os.path.basename(path))[0]
        parsed = parse_run_name(stem) or {}
        meta = load_ga_meta(path)
        runs.append({
            'kind': 'ga',
            'path': path,
            'method': meta.get('method', 'GA'),
            'timestamp': parsed.get('timestamp', meta.get('timestamp', '')),
            'date': parsed.get('date', ''),
            'time': parsed.get('time', ''),
            'score': parsed.get('score', meta.get('score')),
            'extra': meta,
        })

    for path in glob.glob(os.path.join(MODELS_DIR, '*.json')):
        if path.endswith('.meta.json'):
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        parsed = parse_run_name(stem) or {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if 'tree' not in data:
            continue
        runs.append({
            'kind': 'eml',
            'path': path,
            'method': data.get('method', parsed.get('method', 'eml')),
            'timestamp': parsed.get('timestamp', data.get('timestamp', '')),
            'date': parsed.get('date', ''),
            'time': parsed.get('time', ''),
            'score': parsed.get('score', data.get('score')),
            'extra': {
                'formula': data.get('formula', ''),
                'depth': data.get('depth'),
                'size': data.get('size'),
                'n_vars': data.get('n_vars'),
                'mode': data.get('mode'),
                'depth_penalty': data.get('depth_penalty'),
                'test_scores': data.get('test_scores'),
                'elapsed': data.get('elapsed'),
            },
        })

    runs.sort(key=lambda r: r['timestamp'], reverse=True)
    return runs


def latest_ga_path() -> Optional[str]:
    """Path to the latest GA .pt file (None if there is none)."""
    for r in list_saved_runs():
        if r['kind'] == 'ga':
            return r['path']
    return None


# ── Benchmark (results + plot) ───────────────────────────────────────────────

def benchmark_path(timestamp: Optional[str] = None) -> tuple[str, str]:
    """
    Returns a pair of paths: (json_results_path, png_plot_path) with one ts.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = timestamp if timestamp is not None else make_timestamp()
    name = f"benchmark_{ts}"
    return (
        os.path.join(RESULTS_DIR, name + '.json'),
        os.path.join(RESULTS_DIR, name + '.png'),
    )


def save_benchmark_results(results: dict, path: str) -> str:
    """
    Save the full benchmark result as JSON.
    The agents are dropped (not serializable); formulas and metadata are kept.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialisable = {}
    for name, r in results.items():
        entry = {
            'method': name,
            'elapsed': r.get('elapsed'),
            'total_frames': r.get('total_frames'),
            'test_scores': list(r.get('test_scores', [])),
            'test_avg': r.get('test_avg'),
            'test_max': r.get('test_max'),
            'history': r.get('history'),
        }
        if 'formula' in r:
            entry['formula'] = r['formula']
            entry['depth'] = r.get('depth')
        if 'saved_path' in r:
            entry['saved_path'] = r['saved_path']
        serialisable[name] = entry

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serialisable, f, indent=2, ensure_ascii=False)
    return path


# ── Logs ────────────────────────────────────────────────────────────────────

class RunLogger:
    """
    A simple logger: writes to logs/train_<ts>.log and to stdout in parallel.

    Usage:
        log = RunLogger(timestamp='...')
        log.write("hello")
        log.close()

    Supports a callback (for live viewing in the GUI).
    """

    def __init__(self, timestamp: Optional[str] = None,
                 prefix: str = 'train',
                 callback=None, also_stdout: bool = True):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.timestamp = timestamp if timestamp is not None else make_timestamp()
        self.path = os.path.join(LOGS_DIR, f"{prefix}_{self.timestamp}.log")
        self._fh = open(self.path, 'w', encoding='utf-8', buffering=1)
        self.callback = callback
        self.also_stdout = also_stdout
        self.write(f"=== Run started at {self.timestamp} ===")

    def write(self, msg: str = ''):
        line = str(msg)
        self._fh.write(line + '\n')
        if self.also_stdout:
            print(line)
        if self.callback is not None:
            try:
                self.callback(line)
            except Exception:
                pass

    def close(self):
        if self._fh and not self._fh.closed:
            self.write(f"=== Run finished at {make_timestamp()} ===")
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
