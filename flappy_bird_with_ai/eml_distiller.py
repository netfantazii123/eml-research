"""
EML Distiller — symbolic distillation of a neural network into a formula.

Pipeline:
    1. Collect a dataset (state -> oracle_output) from the GA oracle.
    2. Evolve EML trees (symbolic regression).
    3. Obtain a compact formula jump = eml(...).

EML operator: eml(x, y) = exp(x) - ln(|y| + eps)
"""

import os
import sys
import math
import time
import random

import numpy as np

# emlcore lives at the repository root, one level above this task directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import DEPTH_PENALTIES
from env import FlappyEnv
from ga_net import GANet

from emlcore import (Node, eml_basis, search, static_metrics, pool_slice,
                     FitnessCache)


# ── Basis ────────────────────────────────────────────────────────────────────

VAR_NAMES = ('y', 'vel', 'dx', 'gap')


def make_basis(max_depth: int | None = None,
               max_nodes: int | None = None):
    """The basis this task searches in, built from config at call time."""
    return eml_basis(
        config.STATE_SIZE, VAR_NAMES,
        max_depth=config.EML_MAX_DEPTH if max_depth is None else max_depth,
        epsilon=config.EML_EPSILON,
        clamp=config.EML_CLAMP,
        max_nodes=max_nodes,
    )


TASK_BASIS = make_basis()


class EMLNode(Node):
    """Node bound to this task's basis (keeps one-argument from_dict working)."""

    @classmethod
    def from_dict(cls, d: dict, basis=None) -> Node:
        return Node.from_dict(d, basis or TASK_BASIS)


def eml_op(x: float, y: float) -> float:
    """eml(x, y) for a single pair — kept for tests and quick checks."""
    return TASK_BASIS.default_op.scalar(x, y)



# ── Dataset collection ───────────────────────────────────────────────────────

def collect_dataset(
    oracle: GANet,
    n_episodes: int | None = None,
    mode: str = 'sigmoid',
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect a dataset (states, targets) from the GA oracle's play.

    Args:
        oracle: a trained GANet.
        n_episodes: number of episodes (None -> config.EML_DATASET_EPISODES).
        mode: 'sigmoid' (float 0..1) or 'binary' (0/1).
        max_frames: frame limit per episode (None -> config.GA_MAX_FRAMES).

    Returns:
        (states, targets): np.ndarray of shape (N, 4) and (N,).
    """
    if n_episodes is None:
        n_episodes = config.EML_DATASET_EPISODES
    if max_frames is None:
        max_frames = config.GA_MAX_FRAMES
    states_list = []
    targets_list = []

    # One env per episode with its own seed: reproducible, and the layouts stay
    # clear of the TRAIN/VAL/TEST pools, so no evaluation ever runs on pipes the
    # formula was distilled from.
    for ep in range(n_episodes):
        env = FlappyEnv(seed=config.EML_DATA_SEED * 100_000 + ep)
        state = env.reset()
        frames = 0
        while not env.done and frames < max_frames:
            states_list.append(state.copy())
            if mode == 'sigmoid':
                targets_list.append(oracle.get_sigmoid(state))
            else:
                targets_list.append(float(oracle.get_action(state)))
            action = oracle.get_action(state)
            state, _, _ = env.step(action)
            frames += 1

    return np.array(states_list), np.array(targets_list)


# ── Fitness ──────────────────────────────────────────────────────────────────

def _compute_fitness(
    tree: EMLNode,
    states: np.ndarray,
    targets: np.ndarray,
    depth_penalty: float,
    mode: str,
) -> float:
    """
    Fitness of a tree on the dataset.

    Rules:
    - A tree WITHOUT variables (a pure constant) -> nearly zero.
    - For sigmoid: fitness = 1 / (MSE + depth_penalty * depth + eps)
    - For binary:  fitness = balanced_accuracy - depth_penalty * depth
      (balanced = (TPR + TNR) / 2, so that the class imbalance does not make
       the trivial "always 0" solution win)
    - A bonus for using >= 2 variables.
    """
    n = len(states)
    if n == 0:
        return 0.0

    # Penalty for having no variables — a constant tree is useless
    n_vars = tree.n_unique_vars()
    if n_vars == 0:
        return 1e-6

    # One vectorised pass over the dataset instead of a Python loop calling
    # tree.evaluate() per row: same numbers (the scalar and batch forms of every
    # operator carry the same guards, and a non-finite output anywhere sinks
    # the tree in both), roughly two orders of magnitude faster on 5000 rows.
    try:
        pred = tree.evaluate_batch(np.asarray(states, dtype=np.float64))
    except (OverflowError, ValueError, ZeroDivisionError, FloatingPointError):
        return 1e-6
    if not np.all(np.isfinite(pred)):
        return 1e-6

    depth = tree.depth()

    # Bonus for variable diversity (max +20% with 4 variables)
    var_bonus = 1.0 + 0.05 * n_vars

    if mode == 'sigmoid':
        mse = float(np.mean((pred - targets) ** 2))
        base = 1.0 / (mse + depth_penalty * depth + 1e-8)
        return base * var_bonus

    # Balanced accuracy = (TPR + TNR) / 2
    act = pred > 0.5
    t = np.asarray(targets).astype(int) == 1
    tp = int(np.sum(act & t))
    tn = int(np.sum(~act & ~t))
    fp = int(np.sum(act & ~t))
    fn = int(np.sum(~act & t))
    tpr = tp / max(1, tp + fn)  # sensitivity (recall class 1)
    tnr = tn / max(1, tn + fp)  # specificity (recall class 0)
    balanced_acc = (tpr + tnr) / 2.0
    base = max(0.0, balanced_acc - depth_penalty * depth)
    return base * var_bonus


# ── EML agent ────────────────────────────────────────────────────────────────

class EMLAgent:
    """A wrapper over EMLNode for playing in FlappyEnv."""

    def __init__(self, tree: EMLNode, name: str = "EML"):
        self.tree = tree
        self.name = name

    def get_action(self, state: np.ndarray) -> int:
        """Binary action: 1 (flap) or 0 (no-flap)."""
        try:
            val = self.tree.evaluate(state)
            if math.isnan(val) or math.isinf(val):
                return 0
            return 1 if val > 0.5 else 0
        except (OverflowError, ValueError, ZeroDivisionError):
            return 0

    def to_string(self) -> str:
        return self.tree.to_string()


def play_episodes(tree: EMLNode, seeds,
                  max_frames: int | None = None) -> tuple[list[int], list[int]]:
    """
    Play one game per seed. Returns (score_per_game, frames_per_game).

    A fresh FlappyEnv per seed: every formula meets exactly the same pipe
    layouts, which is what makes two fitness values comparable.
    """
    if max_frames is None:
        max_frames = config.GA_MAX_FRAMES
    agent = EMLAgent(tree)
    scores: list[int] = []
    frames: list[int] = []
    for s in seeds:
        env = FlappyEnv(seed=int(s))
        state = env.reset()
        f = 0
        while not env.done and f < max_frames:
            action = agent.get_action(state)
            state, _, _ = env.step(action)
            f += 1
        scores.append(env.score)
        frames.append(f)
    return scores, frames


def score_episodes(tree: EMLNode, seeds, max_frames: int | None = None) -> dict:
    """play_episodes + the aggregates every caller needs (mean/std/fitness)."""
    scores, frames = play_episodes(tree, seeds, max_frames)
    sc = np.asarray(scores, dtype=np.float64)
    fr = np.asarray(frames, dtype=np.float64)
    return {
        'scores': scores, 'frames': frames, 'seeds': [int(s) for s in seeds],
        'avg_score': float(sc.mean()),
        'std_score': float(sc.std(ddof=1)) if len(sc) > 1 else 0.0,
        'avg_frames': float(fr.mean()),
        'fitness': float(sc.mean() * 100.0 + fr.mean()),
    }


def _ingame_fitness(tree: EMLNode, seeds=None,
                    max_frames: int | None = None) -> float:
    """
    In-game fitness: run the formula in FlappyEnv and compute the score.
    fitness = avg_score * 100 + avg_frames (as for the GA).

    seeds: the layouts to play. Defaults to the head of the TRAIN pool; callers
    that evolve should rotate the offset per generation.
    """
    if seeds is None:
        seeds = pool_slice(config.EML_SEED_POOL_TRAIN, config.EML_INGAME_GAMES)
    return score_episodes(tree, seeds, max_frames)['fitness']


# ── Main evolution loop ─────────────────────────────────────────────────────

def evolve_eml(
    states: np.ndarray,
    targets: np.ndarray,
    mode: str = 'sigmoid',
    depth_penalty_name: str = 'medium',
    generations: int | None = None,
    population_size: int | None = None,
    elitism: int | None = None,
    patience: int | None = None,
    verbose: bool = True,
    should_stop=None,
) -> dict:
    """
    GA evolution of EML trees.

    Hyperparameters: taken from the current `config` when not passed.

    Two-phase fitness:
    - Phase 1 (the first 30% of generations): dataset fitness (fast screening).
    - Phase 2 (the rest): in-game fitness (the real evaluation).

    Key mechanisms:
    - Adaptive mutation intensity.
    - Tournament selection.
    - Saving the best of every generation.
    - patience > 0: early stop on a plateau.

    Returns:
        dict: 'best_tree', 'best_agent', 'history', 'elapsed',
              'stopped_early', 'stop_reason'.
    """
    if generations is None:     generations = config.EML_GENERATIONS
    if population_size is None: population_size = config.EML_POPULATION
    if elitism is None:         elitism = config.EML_ELITISM
    if patience is None:        patience = config.EML_PATIENCE

    dp = DEPTH_PENALTIES[depth_penalty_name]
    # One basis for the whole run: both phases search the same language.
    basis = make_basis()
    # 10% of the generations go to the DATA phase (a quick warm-up), the rest to
    # GAME. With narrow oracle targets the DATA fitness is noise and drives the
    # population toward degenerate trees, so the warm-up is kept short and the
    # real evaluation is done by playing.
    phase_switch = max(1, int(generations * 0.1))

    # Initialization
    population = [search.random_tree(basis, max_depth=3, force_op=True)
                  for _ in range(population_size)]
    history = []
    best_fitness_ever = -1.0
    best_tree_ever = None
    patience_counter = 0
    stop_reason = ''

    # Hall of fame (GAME phase): fixed layouts from the VAL pool, never used as
    # fitness, so they can decide which generation's champion actually wins.
    max_frames = config.GA_MAX_FRAMES
    val_seeds = pool_slice(config.EML_SEED_POOL_VAL, config.EML_VAL_GAMES)
    best_val_fitness = -float('inf')
    best_val_result = None

    subsample_size = min(5000, len(states))
    # Exact memoisation (see emlcore/cache.py): the champion's VAL re-scoring
    # is usually a replay of last generation's champion, and duplicates within
    # a generation would replay the same pipes.
    cache = FitnessCache()
    val_ctx = ('val', tuple(int(s) for s in val_seeds), max_frames)

    t0 = time.perf_counter()

    for gen in range(generations):
        if should_stop is not None and should_stop():
            stop_reason = 'cancelled'
            if verbose:
                print(f"    -> Cancelled at gen {gen}.")
            break

        intensity = max(0.1, 1.0 - gen / (generations * 0.7))
        use_ingame = gen >= phase_switch

        # ── DATA -> GAME switch: different fitness scales, reset the baseline ──
        # DATA fitness = 1/MSE (tens), GAME fitness = score*100+frames (hundreds
        # to thousands). Without a reset either the GAME fitness "cannot beat"
        # the DATA record, or the other way round.
        if use_ingame and gen == phase_switch and phase_switch > 0:
            best_fitness_ever = -float('inf')
            patience_counter = 0
            if verbose:
                print(f"    -- Phase switch DATA -> GAME at gen {gen}, baseline reset.")

        # ── Fitness ──────────────────────────────────────────────────────
        if use_ingame:
            # In-game: every tree plays the SAME layouts this generation
            # (common random numbers), rotated across generations so the
            # population cannot settle into one lucky set of pipes.
            n_games = config.EML_INGAME_GAMES
            gen_seeds = pool_slice(config.EML_SEED_POOL_TRAIN, n_games,
                                   offset=gen * n_games)
            def _game_fit(tree):
                if tree.n_unique_vars() == 0:
                    return 0.0
                f = _ingame_fitness(tree, gen_seeds)
                # Depth penalty (milder than in dataset mode)
                return max(0.0, f - dp * tree.depth() * 50.0)
            fitness_scores = cache.map(
                population, ('game', tuple(int(s) for s in gen_seeds)),
                _game_fit)
        else:
            # Dataset: fast screening
            if len(states) > subsample_size:
                idx = np.random.choice(len(states), subsample_size, replace=False)
                fit_states = states[idx]
                fit_targets = targets[idx]
            else:
                fit_states = states
                fit_targets = targets

            fitness_scores = []
            for tree in population:
                f = _compute_fitness(tree, fit_states, fit_targets, dp, mode)
                fitness_scores.append(f)

        # Sort
        ranked = sorted(zip(fitness_scores, population),
                        key=lambda x: x[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [t for _, t in ranked]

        best_fitness = fitness_scores[0]
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        best_tree = population[0]

        phase_str = "GAME" if use_ingame else "DATA"
        gen_stats = {
            'generation': gen,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'best_depth': best_tree.depth(),
            'best_size': best_tree.size(),
            'best_formula': best_tree.to_string(),
            'intensity': intensity,
            'phase': phase_str,
        }
        history.append(gen_stats)

        # Save the best. In the GAME phase the champion is re-scored on the
        # disjoint VAL pool first: training fitness now comes from a different
        # set of layouts every generation, so a running max over it would keep
        # whichever generation drew the kindest pipes, not the best formula.
        if use_ingame:
            val_res = cache.get(best_tree, val_ctx, lambda t: score_episodes(
                t, val_seeds, max_frames))
            gen_stats['val_score'] = val_res['avg_score']
            gen_stats['val_fitness'] = val_res['fitness']
            improved = val_res['fitness'] > best_val_fitness
            if improved:
                best_val_fitness = val_res['fitness']
                best_val_result = val_res
        else:
            improved = best_fitness > best_fitness_ever

        if improved:
            best_fitness_ever = max(best_fitness_ever, best_fitness)
            best_tree_ever = best_tree.clone()
            patience_counter = 0
        else:
            # A plateau in the DATA phase is MSE noise on narrow targets.
            # Count the plateau only in the GAME phase, where the signal is real.
            if use_ingame:
                patience_counter += 1

        if verbose and (gen % 10 == 0 or gen == generations - 1
                        or patience_counter >= patience):
            score_est = int(best_fitness // 100) if use_ingame else -1
            print(f"    Gen {gen:3d} [{phase_str}] | "
                  f"Fit: {best_fitness:10.1f} | "
                  f"D: {best_tree.depth()} S: {best_tree.size()} "
                  f"V: {best_tree.n_unique_vars()} | "
                  f"I: {intensity:.2f}" +
                  (f" | ~{score_est}p" if use_ingame else ""))

        if patience > 0 and patience_counter >= patience:
            stop_reason = f'plateau_{patience}'
            if verbose:
                print(f"    -> Early stop: no improvement for {patience} gens.")
            break

        # New generation
        population = search.breed(
            population, fitness_scores, basis,
            size=population_size, elitism=elitism, intensity=intensity,
            p_mutate=0.45, p_cross=0.75, p_cross_mutate=0.3, tournament_k=5)

    elapsed = time.perf_counter() - t0

    if best_tree_ever is None:
        best_tree_ever = population[0]

    if best_val_result is None:
        best_val_result = score_episodes(best_tree_ever, val_seeds, max_frames)

    return {
        'best_tree': best_tree_ever,
        'best_agent': EMLAgent(best_tree_ever),
        'history': history,
        'elapsed': elapsed,
        'stopped_early': bool(stop_reason),
        'stop_reason': stop_reason,
        # Selection metric. The reportable number comes from the TEST pool.
        'val_result': best_val_result,
        'val_score': best_val_result['avg_score'],
    }


# ── Full distillation pipeline ───────────────────────────────────────────────

def distill(
    oracle: GANet,
    mode: str = 'sigmoid',
    depth_penalty_name: str = 'medium',
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: dataset collection -> EML tree evolution.

    Returns:
        dict: 'best_agent', 'best_formula', 'dataset_size', + evolve results.
    """
    if verbose:
        print(f"  [1/2] Collecting dataset (mode={mode})...")
    states, targets = collect_dataset(oracle, mode=mode)
    if verbose:
        print(f"    Dataset: {len(states)} samples, "
              f"target range [{targets.min():.3f}, {targets.max():.3f}]")

    if verbose:
        print(f"  [2/2] Evolving EML trees (penalty={depth_penalty_name})...")
    result = evolve_eml(states, targets, mode=mode,
                        depth_penalty_name=depth_penalty_name,
                        verbose=verbose)

    result['dataset_size'] = len(states)
    result['best_formula'] = result['best_tree'].to_string()
    result['mode'] = mode
    result['depth_penalty'] = depth_penalty_name

    if verbose:
        print(f"    Best formula: {result['best_formula']}")
        print(f"    Depth: {result['best_tree'].depth()}, "
              f"Size: {result['best_tree'].size()}, "
              f"Vars: {result['best_tree'].n_unique_vars()}")

    return result
