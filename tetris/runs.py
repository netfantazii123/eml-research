"""
runs.py — папки запусков с датой: гарантированный полный слепок эксперимента.

Каждый запуск обучения/дистилляции создаёт runs/<дата_время>_<вид>/ и кладёт
туда ВСЁ, что нужно для воспроизведения и диплома:
    config_snapshot.json   — все константы config на момент запуска
    <модель/формулы>       — копии итоговых артефактов
    *_history.json / *.png — история и графики
    summary.json           — итоговые метрики запуска

Канонические пути (models/best_ppo.pt, models/best_eml.json) продолжают
обновляться — пайплайн и GUI смотрят на них; run-папка — неизменяемый архив.
"""

import os
import json
import time
import shutil

import config

RUNS_DIR = 'runs'


def create_run_dir(kind: str) -> str:
    """Создать runs/<YYYY-MM-DD_HH-MM-SS>_<kind>/ и вернуть путь."""
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join(RUNS_DIR, f"{stamp}_{kind}")
    os.makedirs(path, exist_ok=True)
    snapshot_config(path)
    return path


def snapshot_config(run_dir: str) -> str:
    """Все UPPERCASE-константы config → config_snapshot.json."""
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
    """Итоговые метрики запуска → summary.json."""
    path = os.path.join(run_dir, 'summary.json')
    payload = {'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'), **summary}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def copy_into(run_dir: str, *paths: str) -> list[str]:
    """Скопировать файлы в run-папку (отсутствующие тихо пропускаются)."""
    copied = []
    for p in paths:
        if p and os.path.exists(p):
            dst = os.path.join(run_dir, os.path.basename(p))
            shutil.copy2(p, dst)
            copied.append(dst)
    return copied
