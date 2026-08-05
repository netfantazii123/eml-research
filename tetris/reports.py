"""
reports.py — saving training/distillation results: plots + history.

Everything is written into results/:
    ppo_history.json          — full PPO training history (per update)
    ppo_learning_curve.png    — lines/episode length + losses
    distill_history.json      — history of the DATA/GAME distillation phases
    eml_fitness_curve.png     — DATA phase fitness + GAME phase lines

Used by both the CLI (main.py train/distill) and the GUI (Save Charts buttons).
Matplotlib in headless mode (Agg) — safe from background threads.
"""

import os
import json
import time

import matplotlib
from matplotlib.figure import Figure

import config


def _ensure_results() -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return config.RESULTS_DIR


def _stamp(meta: dict | None) -> dict:
    out = {'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    if meta:
        out.update(meta)
    return out


# ── PPO ───────────────────────────────────────────────────────────────────────

def save_ppo_history(history: list[dict], meta: dict | None = None,
                     path: str | None = None) -> str:
    """PPO training history -> JSON (the learning curve point by point)."""
    if path is None:
        path = os.path.join(_ensure_results(), 'ppo_history.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'meta': _stamp(meta), 'history': history}, f)
    return path


def plot_ppo_history(history: list[dict], path: str | None = None) -> str:
    """PPO learning curve -> PNG (2 panels: lines/length + losses)."""
    if path is None:
        path = os.path.join(_ensure_results(), 'ppo_learning_curve.png')

    steps = [h['global_step'] for h in history]
    fig = Figure(figsize=(10, 7), dpi=120)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(steps, [h['avg_lines'] for h in history],
             color='tab:green', lw=1.6, label='avg lines (100 ep)')
    ax1.plot(steps, [h['max_lines'] for h in history],
             color='tab:blue', lw=0.9, alpha=0.5, label='max lines')
    if history and 'avg_len' in history[0]:
        ax1b = ax1.twinx()
        ax1b.plot(steps, [h.get('avg_len', 0) for h in history],
                  color='tab:orange', lw=1.0, alpha=0.6, label='avg episode len')
        ax1b.set_ylabel('placements/episode', color='tab:orange')
    ax1.set_xlabel('placements')
    ax1.set_ylabel('lines/episode')
    ax1.set_title('PPO learning curve (placement-based oracle)')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(steps, [h['pg_loss'] for h in history],
             color='tab:red', lw=0.9, label='pg loss')
    ax2.plot(steps, [h['vf_loss'] for h in history],
             color='tab:olive', lw=0.9, alpha=0.8, label='vf loss')
    ax2b = ax2.twinx()
    ax2b.plot(steps, [h['entropy'] for h in history],
              color='tab:purple', lw=1.1, alpha=0.8, label='entropy')
    ax2b.set_ylabel('entropy', color='tab:purple')
    ax2.set_xlabel('placements')
    ax2.set_ylabel('loss')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    return path


# ── Distillation ─────────────────────────────────────────────────────────────

def save_distill_history(result: dict, meta: dict | None = None,
                         path: str | None = None) -> str:
    """Distillation history (DATA fitness + GAME lines) -> JSON."""
    if path is None:
        path = os.path.join(_ensure_results(), 'distill_history.json')
    data_hist = []
    if result.get('data_results'):
        data_hist = result['data_results'][0].get('history', [])
    joint_hist = []
    if result.get('joint_result'):
        joint_hist = result['joint_result'].get('history', [])
    payload = {
        'meta': _stamp(meta),
        'data_fitness': data_hist,
        'game_fitness': joint_hist,
        'base_lines': result.get('base_lines'),
        'final_lines': result.get('final_lines'),
        'dataset_size': result.get('dataset_size'),
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
    return path


def plot_distill_history(data_hist: list[float], joint_hist: list[float],
                         path: str | None = None) -> str:
    """Distillation curves -> PNG (DATA fitness + GAME lines)."""
    if path is None:
        path = os.path.join(_ensure_results(), 'eml_fitness_curve.png')

    fig = Figure(figsize=(10, 6), dpi=120)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(range(len(data_hist)), data_hist, color='tab:green', lw=1.4)
    ax1.set_title('DATA phase — regression fitness (per generation)')
    ax1.set_xlabel('generation')
    ax1.set_ylabel('fitness')
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    lines = [f / 100.0 for f in joint_hist]
    ax2.plot(range(len(lines)), lines, color='tab:purple', lw=1.4)
    ax2.set_title('GAME phase — in-game score (lines + placements/100)')
    ax2.set_xlabel('generation')
    ax2.set_ylabel('~lines/game')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    return path
