"""
Storage — единая точка для сохранения/загрузки моделей, формул и логов.

Соглашение об именах:
    {method}_{YYYY-MM-DD}_{HH-MM-SS}_score-{N}.{ext}

    Примеры:
        ga_2026-05-24_14-30-15_score-42.pt
        eml-medium_2026-05-24_14-31-02_score-37.json
        benchmark_2026-05-24_14-30-15.json
        train_2026-05-24_14-30-15.log

Структура каталогов:
    models/   — *.pt (GANet) и *.json (EML формулы)
    results/  — benchmark_*.json (метрики) и benchmark_*.png (графики)
    logs/     — train_*.log (текстовые логи запусков)
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


# ── Имена файлов ────────────────────────────────────────────────────────────

def make_timestamp(dt: Optional[datetime] = None) -> str:
    """Текущее время как '2026-05-24_14-30-15'."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%Y-%m-%d_%H-%M-%S')


def make_run_name(method: str, score: Optional[int] = None,
                  timestamp: Optional[str] = None) -> str:
    """
    Имя файла без расширения: 'ga_2026-05-24_14-30-15_score-42'.
    Если score=None, добавляется 'noscore'.
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
    """Распарсить имя файла. Возвращает dict или None."""
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


# ── GA модели ───────────────────────────────────────────────────────────────

def save_ga_model(net: GANet, score: int,
                  metadata: Optional[dict] = None,
                  timestamp: Optional[str] = None) -> str:
    """
    Сохранить GA сеть как .pt + sidecar .meta.json с метаданными.
    Возвращает путь к .pt файлу.
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
    """Загрузить GA сеть из .pt файла."""
    net = GANet()
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def load_ga_meta(pt_path: str) -> dict:
    """Загрузить метаданные GA модели. Если их нет — вернуть пустой dict."""
    meta_path = pt_path.replace('.pt', '.meta.json')
    if not os.path.exists(meta_path):
        stem = os.path.splitext(os.path.basename(pt_path))[0]
        parsed = parse_run_name(stem)
        return parsed or {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── EML формулы ─────────────────────────────────────────────────────────────

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
    Сохранить EML-формулу как JSON. Возвращает путь к файлу.

    Содержимое:
        method        — 'eml-{depth_penalty}'
        timestamp, score
        mode          — 'sigmoid' | 'binary'
        depth_penalty — 'weak' | 'medium' | 'strong'
        formula       — человекочитаемая строка
        tree          — AST как dict
        depth, size, n_vars
        test_scores   — список очков на тестовых играх
        history       — история эволюции (опционально)
        elapsed       — секунды
        dataset_size, oracle_ref — для трассируемости
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
    """Загрузить JSON с EML-формулой и восстановить AST в ключе 'tree_node'."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data['tree_node'] = EMLNode.from_dict(data['tree'])
    data['agent'] = EMLAgent(data['tree_node'], name=data.get('method', 'EML'))
    return data


# ── Список сохранёнок ───────────────────────────────────────────────────────

def list_saved_runs() -> list[dict]:
    """
    Все сохранённые модели и формулы, отсортированные по времени (новые сверху).

    Каждая запись:
        {
            'kind': 'ga' | 'eml',
            'path': str,
            'method': str,            # 'GA' | 'eml-weak' | ...
            'timestamp': str,         # '2026-05-24_14-30-15'
            'date': str, 'time': str,
            'score': int | None,
            'extra': dict,            # method-специфичные поля
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
    """Путь к последнему GA .pt файлу (None если нет)."""
    for r in list_saved_runs():
        if r['kind'] == 'ga':
            return r['path']
    return None


# ── Бенчмарк (результаты + плот) ─────────────────────────────────────────────

def benchmark_path(timestamp: Optional[str] = None) -> tuple[str, str]:
    """
    Возвращает пару путей: (json_results_path, png_plot_path) с одним ts.
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
    Сохранить полный результат бенчмарка как JSON.
    Агенты убираются (несериализуемы); формулы и метаданные сохраняются.
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


# ── Логи ────────────────────────────────────────────────────────────────────

class RunLogger:
    """
    Простой логер: пишет в файл logs/train_<ts>.log и параллельно в stdout.

    Использование:
        log = RunLogger(timestamp='...')
        log.write("hello")
        log.close()

    Поддерживает callback (для GUI live-просмотра).
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
