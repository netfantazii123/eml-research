"""
Benchmark — запуск GA + EML-дистилляции, сбор метрик, графики.

Все артефакты получают имена с датой/временем/скором:
    models/ga_2026-05-24_14-30-15_score-42.pt
    models/eml-medium_2026-05-24_14-31-02_score-37.json
    results/benchmark_2026-05-24_14-30-15.{json,png}
    logs/train_2026-05-24_14-30-15.log
"""

import os
import time
import numpy as np

from config import RESULTS_DIR, GA_MAX_FRAMES
from env import FlappyEnv
from ga_net import GANet, train_ga
from eml_distiller import distill
from storage import (
    RunLogger, make_timestamp,
    save_ga_model, save_eml_formula,
    benchmark_path, save_benchmark_results,
    load_ga_model as _storage_load_ga, latest_ga_path,
)


# ── Тестирование агента ──────────────────────────────────────────────────────

def test_agent_score(agent, env: FlappyEnv, n_games: int = 10,
                     max_frames: int = 5000) -> list[int]:
    """Прогнать агента n_games раз и вернуть список очков."""
    scores = []
    for _ in range(n_games):
        env.reset()
        f = 0
        while not env.done and f < max_frames:
            action = agent.get_action(env.get_state())
            env.step(action)
            f += 1
        scores.append(env.score)
    return scores


# ── Save / Load (обёртки для обратной совместимости) ─────────────────────────

def load_ga_model() -> GANet | None:
    """Загрузить последнюю GA модель из models/. None если нет."""
    path = latest_ga_path()
    if path is None:
        print("  No GA model found in models/.")
        return None
    net = _storage_load_ga(path)
    print(f"  Loaded GA <- {path}")
    return net


# ── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(verbose: bool = True, log_callback=None) -> dict:
    """
    Полный бенчмарк: GA → дистилляция (3 варианта penalty) → сравнение.

    Args:
        verbose: писать ли в stdout.
        log_callback: опц. callable(line) для live-логирования (GUI).
    """
    ts = make_timestamp()
    log = RunLogger(timestamp=ts, prefix='train',
                    callback=log_callback, also_stdout=verbose)

    env = FlappyEnv()
    results = {}

    try:
        # ── 1. GA ────────────────────────────────────────────────────────
        log.write("\n" + "=" * 60)
        log.write("  [1/4] Training GA Oracle...")
        log.write("=" * 60)

        ga = train_ga(verbose=verbose)
        ga_scores = test_agent_score(ga['best_agent'], env)
        ga_avg = sum(ga_scores) / len(ga_scores)
        ga_max = max(ga_scores)
        ga_path = save_ga_model(
            ga['best_agent'], score=ga_max,
            metadata={
                'test_scores': ga_scores,
                'test_avg': ga_avg,
                'test_max': ga_max,
                'elapsed': ga['elapsed'],
                'total_frames': ga['total_frames'],
            },
            timestamp=ts,
        )
        results['GA'] = {
            'agent': ga['best_agent'],
            'history': ga['history'],
            'total_frames': ga['total_frames'],
            'elapsed': ga['elapsed'],
            'test_scores': ga_scores,
            'test_avg': ga_avg,
            'test_max': ga_max,
            'saved_path': ga_path,
        }
        log.write(f"\n  GA Test: avg={ga_avg:.1f}, max={ga_max}")
        log.write(f"  Saved GA -> {ga_path}")

        # ── 2-4. EML дистилляция ─────────────────────────────────────────
        oracle = ga['best_agent']
        for i, penalty_name in enumerate(['weak', 'medium', 'strong']):
            label = f"EML-{penalty_name}"
            log.write("\n" + "=" * 60)
            log.write(f"  [{i+2}/4] Distilling {label}...")
            log.write("=" * 60)

            t0 = time.perf_counter()
            eml_result = distill(oracle, mode='sigmoid',
                                 depth_penalty_name=penalty_name,
                                 verbose=verbose)
            elapsed = time.perf_counter() - t0

            eml_scores = test_agent_score(eml_result['best_agent'], env)
            eml_avg = sum(eml_scores) / len(eml_scores)
            eml_max = max(eml_scores)

            eml_path = save_eml_formula(
                eml_result['best_tree'],
                score=eml_max,
                mode='sigmoid',
                depth_penalty=penalty_name,
                test_scores=eml_scores,
                history=eml_result.get('history'),
                elapsed=elapsed,
                dataset_size=eml_result.get('dataset_size'),
                oracle_ref=os.path.basename(ga_path),
                timestamp=ts,
            )

            results[label] = {
                'agent': eml_result['best_agent'],
                'history': eml_result['history'],
                'total_frames': eml_result.get('dataset_size', 0),
                'elapsed': elapsed,
                'test_scores': eml_scores,
                'test_avg': eml_avg,
                'test_max': eml_max,
                'formula': eml_result['best_formula'],
                'depth': eml_result['best_tree'].depth(),
                'saved_path': eml_path,
            }
            log.write(f"\n  {label} Test: avg={eml_avg:.1f}, max={eml_max}")
            log.write(f"  Formula: {eml_result['best_formula']}")
            log.write(f"  Saved   -> {eml_path}")

        # ── Таблица ──────────────────────────────────────────────────────
        log.write("\n" + "=" * 60)
        log.write("  BENCHMARK RESULTS")
        log.write("=" * 60)
        header = f"  {'Method':<14} {'Time(s)':>8} {'Avg':>6} {'Max':>6}"
        log.write(header)
        log.write("  " + "-" * (len(header) - 2))
        for name, r in results.items():
            log.write(f"  {name:<14} {r['elapsed']:>8.1f} "
                      f"{r['test_avg']:>6.1f} {r['test_max']:>6d}")
        log.write("=" * 60)

        # ── Графики + JSON метрик ────────────────────────────────────────
        json_path, png_path = benchmark_path(timestamp=ts)
        plot_benchmark(results, out_path=png_path)
        log.write(f"  Plot saved:    {png_path}")
        save_benchmark_results(results, json_path)
        log.write(f"  Results saved: {json_path}")

    finally:
        log.close()

    return results


# ── Графики ──────────────────────────────────────────────────────────────────

def plot_benchmark(results: dict, out_path: str | None = None):
    """Построить и сохранить графики."""
    import matplotlib
    matplotlib.use('Agg')  # без GUI-бэкенда — безопасно из любого потока
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if out_path is None:
        _, out_path = benchmark_path()

    colors = {
        'GA': '#e74c3c',
        'EML-weak': '#f39c12',
        'EML-medium': '#2ecc71',
        'EML-strong': '#3498db',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, r in results.items():
        color = colors.get(name, '#999')
        hist = r['history']
        if name == 'GA':
            x = [h['generation'] for h in hist]
            y = [h['best_score'] for h in hist]
            ax.plot(x, y, label=f'{name} (best score)',
                    color=color, linewidth=2)
        else:
            x = [h['generation'] for h in hist]
            y = [h['best_fitness'] for h in hist]
            ax.plot(x, y, label=f'{name} (fitness)',
                    color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Score / Fitness')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    names = list(results.keys())
    avgs = [results[n]['test_avg'] for n in names]
    maxs = [results[n]['test_max'] for n in names]
    x_pos = np.arange(len(names))
    w = 0.35
    bars1 = ax2.bar(x_pos - w/2, avgs, w, label='Avg score',
                     color=[colors.get(n, '#999') for n in names], alpha=0.7)
    bars2 = ax2.bar(x_pos + w/2, maxs, w, label='Max score',
                     color=[colors.get(n, '#999') for n in names], alpha=1.0)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15)
    ax2.set_ylabel('Score (pipes)')
    ax2.set_title('Test Results (10 games)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}',
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h, f'{int(h)}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
