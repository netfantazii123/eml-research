"""
Benchmark — runs GA + EML distillation, collects metrics, draws plots.

Every artifact is named with the date/time/score:
    models/ga_2026-05-24_14-30-15_score-42.pt
    models/eml-medium_2026-05-24_14-31-02_score-37.json
    results/benchmark_2026-05-24_14-30-15.{json,png}
    logs/train_2026-05-24_14-30-15.log
"""

import os
import time
import numpy as np

import config
from config import RESULTS_DIR, GA_MAX_FRAMES
from env import FlappyEnv
from ga_net import GANet, train_ga
from eml_distiller import distill, pool_slice
from storage import (
    RunLogger, make_timestamp,
    save_ga_model, save_eml_formula,
    benchmark_path, save_benchmark_results,
    load_ga_model as _storage_load_ga, latest_ga_path,
)


# ── Agent testing ────────────────────────────────────────────────────────────

def test_agent_score(agent, env: FlappyEnv | None = None,
                     n_games: int | None = None,
                     max_frames: int | None = None, seeds=None) -> list[int]:
    """
    Run the agent once per HOLDOUT seed and return the list of scores.

    The `env` argument is ignored and kept only so existing callers keep working:
    every game now gets its own env seeded from the TEST pool. Reusing one env
    made every agent play different pipes AND put the comparison on whatever
    layouts came up, which is not a measurement of anything.
    """
    if max_frames is None:
        max_frames = config.EML_TEST_MAX_FRAMES
    if n_games is None:
        n_games = config.EML_TEST_GAMES
    if seeds is None:
        seeds = pool_slice(config.EML_SEED_POOL_TEST, n_games)
    scores = []
    for s in seeds:
        game = FlappyEnv(seed=int(s))
        game.reset()
        f = 0
        while not game.done and f < max_frames:
            action = agent.get_action(game.get_state())
            game.step(action)
            f += 1
        scores.append(game.score)
    return scores


# ── Save / Load (wrappers for backward compatibility) ────────────────────────

def load_ga_model() -> GANet | None:
    """Load the latest GA model from models/. None if there is none."""
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
    Full benchmark: GA -> distillation (3 penalty variants) -> comparison.

    Args:
        verbose: whether to write to stdout.
        log_callback: optional callable(line) for live logging (GUI).
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

        # ── 2-4. EML distillation ────────────────────────────────────────
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

        # ── Table ────────────────────────────────────────────────────────
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

        # ── Plots + metrics JSON ─────────────────────────────────────────
        json_path, png_path = benchmark_path(timestamp=ts)
        plot_benchmark(results, out_path=png_path)
        log.write(f"  Plot saved:    {png_path}")
        save_benchmark_results(results, json_path)
        log.write(f"  Results saved: {json_path}")

    finally:
        log.close()

    return results


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_benchmark(results: dict, out_path: str | None = None):
    """Build and save the plots."""
    import matplotlib
    matplotlib.use('Agg')  # no GUI backend — safe from any thread
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
