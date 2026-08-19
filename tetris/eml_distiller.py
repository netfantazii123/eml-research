"""
eml_distiller.py — symbolic distillation of the CNN oracle into ONE EML formula.

Idea (v2, placement-based): the oracle and the formula choose a piece
PLACEMENT. The formula is a classic evaluation function (like Dellacherie's): it
is shown the features of the board after the placement (afterstate) and returns
a scalar. At inference:
    placement* = argmax_i f(features(afterstate_i))  over legal i.

Two phases:
  1. DATA phase — symbolic regression: f(features_i) ~ the oracle's logit_i
     (weighted MSE, vectorized in numpy). The formula learns to rank placements
     the same way the oracle does.
  2. GAME phase — a population of formulas is refined by evolution against the
     ACTUAL score in TetrisEnv: the formula plays by itself, and
     fitness = lines + survival.

EML operator:  eml(x, y) = exp(clamp(x, -10, 10)) - ln(|y| + eps).

The AST, the basis of primitives and the genetic search live in emlcore/; this
module only supplies what is specific to Tetris: the afterstate features, the
two fitness functions and the two-phase pipeline.
"""

import math
import os
import sys
import time
import random

import numpy as np

# emlcore lives at the repository root, one level above this task directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from config import DEPTH_PENALTIES, FEATURE_NAMES, N_FEATURES
from env import TetrisEnv

from emlcore import Node, eml_basis, search, static_metrics, pool_slice


# ── Basis ─────────────────────────────────────────────────────────────────────

def make_basis(max_depth: int | None = None,
               max_nodes: int | None = None) -> "object":
    """
    The basis this task searches in, built from config at call time.

    Built per call rather than once at import: batch_distill varies
    EML_MAX_DEPTH between runs, and a basis captured at import would silently
    ignore that. max_nodes is the node budget K — unset for now, and the hook
    the K sweep will use.
    """
    return eml_basis(
        N_FEATURES, FEATURE_NAMES,
        max_depth=config.EML_MAX_DEPTH if max_depth is None else max_depth,
        epsilon=config.EML_EPSILON,
        clamp=config.EML_CLAMP,
        max_nodes=max_nodes,
    )


# Basis used for deserialising saved formulas. Only the operator definitions and
# the variable names matter there, so the search limits are irrelevant.
TASK_BASIS = make_basis()


class EMLNode(Node):
    """
    Node bound to this task's basis.

    Exists so that storage.load_formulas(EMLNode, path) and the GUI keep calling
    from_dict() with one argument. Files written before the core was extracted
    load unchanged.
    """

    @classmethod
    def from_dict(cls, d: dict, basis=None) -> Node:
        return Node.from_dict(d, basis or TASK_BASIS)


def eml_op(x: float, y: float) -> float:
    """eml(x, y) for a single pair — kept for tests and quick checks."""
    return TASK_BASIS.default_op.scalar(x, y)



# ── DATA phase: regression of the formula to the oracle logits ────────────────

def _regression_fitness(tree: EMLNode, X: np.ndarray, y: np.ndarray,
                        w: np.ndarray, depth_penalty: float) -> float:
    """
    Fitness = var_bonus / (weighted_MSE + depth_penalty * depth).

    A constant tree (0 variables) is useless -> ~0.
    """
    n_vars = tree.n_unique_vars()
    if n_vars == 0:
        return 1e-6
    try:
        pred = tree.evaluate_batch(X)
    except (OverflowError, ValueError, FloatingPointError):
        return 1e-6
    if not np.all(np.isfinite(pred)):
        return 1e-6

    err = (pred - y) ** 2
    mse = float(np.average(err, weights=w))
    depth = tree.depth()
    var_bonus = 1.0 + config.EML_VAR_BONUS * n_vars
    return var_bonus / (mse + depth_penalty * depth + 1e-8)


def evolve_formula(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    generations: int,
    population_size: int,
    elitism: int,
    patience: int,
    depth_penalty: float,
    subsample: int | None = None,
    val_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    basis=None,
    verbose: bool = False,
    label: str = 'PLACE',
    should_stop=None,
    on_event=None,
) -> dict:
    """
    GA evolution of an EML formula (symbolic regression to the target y).

    The fitness subsample is redrawn every generation, so per-generation fitness
    values are NOT comparable to each other: a running max over them selects the
    tree that met the friendliest subsample. val_data (rows held out by group)
    breaks that tie honestly — the generation champion is re-scored there and the
    returned tree is the best by validation.

    on_event: an optional callback(dict) for progress (GUI). Invoked every
    generation with {'type':'data_gen', ...}.

    Returns: {'best_tree', 'best_fitness', 'best_val_fitness', 'history',
              'val_history'}.
    """
    if subsample is None:
        subsample = config.EML_DATA_SUBSAMPLE
    if basis is None:
        basis = make_basis()
    elitism = max(1, min(elitism, population_size))

    # Derived from the seeded global RNG: reproducible per run, yet actually
    # different across runs (a hardcoded default_rng(0) handed every "seed" the
    # very same subsample sequence).
    rng = np.random.default_rng(random.randrange(2 ** 63))
    population = search.seed_population(basis, population_size, max_depth=3)
    history: list[float] = []
    val_history: list[float] = []
    best_fitness_ever = -1.0
    best_val_fitness = -math.inf
    best_tree_ever: EMLNode | None = None
    patience_counter = 0

    for gen in range(generations):
        if should_stop is not None and should_stop():
            break

        intensity = max(0.1, 1.0 - gen / (generations * 0.7))

        # Subsample for speed (fresh every generation -> less overfitting).
        if len(X) > subsample:
            idx = rng.choice(len(X), subsample, replace=False)
            Xs, ys, ws = X[idx], y[idx], w[idx]
        else:
            Xs, ys, ws = X, y, w

        fitness_scores = [_regression_fitness(t, Xs, ys, ws, depth_penalty)
                          for t in population]

        ranked = sorted(zip(fitness_scores, population),
                        key=lambda p: p[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [t for _, t in ranked]

        best_fitness = fitness_scores[0]
        history.append(best_fitness)
        best_fitness_ever = max(best_fitness_ever, best_fitness)

        # Hall of fame on held-out groups (falls back to train fitness when no
        # validation split was supplied, e.g. tiny debug datasets).
        if val_data is not None:
            val_fitness = _regression_fitness(
                population[0], val_data[0], val_data[1], val_data[2],
                depth_penalty)
        else:
            val_fitness = best_fitness
        val_history.append(val_fitness)

        if val_fitness > best_val_fitness:
            best_val_fitness = val_fitness
            best_tree_ever = population[0].clone()
            patience_counter = 0
        else:
            patience_counter += 1

        bt = population[0]
        if verbose and (gen % 25 == 0 or gen == generations - 1):
            print(f"      [{label}] gen {gen:>4} | fit {best_fitness:10.2f} | "
                  f"val {val_fitness:10.2f} | "
                  f"D{bt.depth()} S{bt.size()} V{bt.n_unique_vars()} | "
                  f"i={intensity:.2f}")
        if on_event is not None:
            on_event({'type': 'data_gen', 'action': 0, 'name': label,
                      'gen': gen, 'generations': generations,
                      'fitness': best_fitness, 'val_fitness': val_fitness,
                      'depth': bt.depth(),
                      'size': bt.size(), 'nvars': bt.n_unique_vars()})

        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"      [{label}] early stop @ gen {gen} "
                      f"(no improve {patience})")
            break

        # New generation.
        population = search.breed(
            population, fitness_scores, basis,
            size=population_size, elitism=elitism, intensity=intensity,
            p_mutate=0.45, p_cross=0.75, p_cross_mutate=0.3, tournament_k=5)

    if best_tree_ever is None:
        best_tree_ever = population[0]
    return {'best_tree': best_tree_ever,
            'best_fitness': best_fitness_ever,
            'best_val_fitness': best_val_fitness,
            'history': history,
            'val_history': val_history}


# ── EML policy (one formula, argmax over placements) ──────────────────────────

class EMLPolicy:
    """
    A wrapper over an EML tree: placement* = argmax_i f(afterstate_i).

    Also accepts a list containing a single tree (the storage.save_formulas
    format).
    """

    def __init__(self, tree):
        if isinstance(tree, (list, tuple)):
            assert len(tree) >= 1, "empty formula list"
            tree = tree[0]
        self.tree = tree

    def choose(self, mask: np.ndarray, afeats: np.ndarray,
               scalars: np.ndarray | None = None) -> int:
        """
        The best legal placement based on the obs features.

        The formula input = afterstate features (19) + a one-hot of the next
        piece (7), taken from obs['scalars'][7:14]. If scalars are not passed,
        the next_* features are filled with zeros (formulas that do not use
        next_* are unaffected).
        """
        legal = np.flatnonzero(mask > 0)
        if len(legal) == 0:
            return -1
        X = np.zeros((len(legal), config.N_FEATURES), dtype=np.float64)
        X[:, :afeats.shape[1]] = afeats[legal]
        if scalars is not None:
            X[:, afeats.shape[1]:] = scalars[7:14]
        vals = self.tree.evaluate_batch(X)
        vals = np.where(np.isfinite(vals), vals, -np.inf)
        return int(legal[int(np.argmax(vals))])

    def get_action(self, env: TetrisEnv) -> int:
        """Choose the best legal placement for the current env state."""
        obs = env.get_placement_state()
        return self.choose(obs['mask'], obs['afeats'], obs['scalars'])


# ── GAME phase: in-game evolution of the formula ──────────────────────────────

def play_episodes(tree, seeds, max_placements: int) -> tuple[list[int], list[int]]:
    """
    Play exactly one game per seed. Returns (lines_per_game, placements_per_game).

    A FRESH TetrisEnv per seed — this is what makes the comparison fair.
    TetrisEnv.reset() does NOT re-seed self.rng, so reusing one env across games
    means game k starts wherever game k-1 left the bag stream, i.e. at a position
    that depends on how long the formula survived. Only the first game of such a
    run is comparable across formulas; the rest silently differ per individual.
    """
    policy = EMLPolicy(tree)
    lines: list[int] = []
    steps: list[int] = []
    for s in seeds:
        env = TetrisEnv(seed=int(s))
        obs = env.reset()
        f = 0
        while not env.done and f < max_placements:
            a = policy.choose(obs['mask'], obs['afeats'], obs['scalars'])
            if a < 0:
                break
            obs, _, _, _ = env.step_placement(a)
            f += 1
        lines.append(env.score)
        steps.append(f)
    return lines, steps


def score_episodes(tree, seeds, max_placements: int) -> dict:
    """play_episodes + the aggregates every caller needs (mean/std/fitness)."""
    lines, steps = play_episodes(tree, seeds, max_placements)
    la = np.asarray(lines, dtype=np.float64)
    sa = np.asarray(steps, dtype=np.float64)
    return {
        'lines': lines,
        'steps': steps,
        'seeds': list(seeds),
        'avg_lines': float(la.mean()),
        'std_lines': float(la.std(ddof=1)) if len(la) > 1 else 0.0,
        'avg_steps': float(sa.mean()),
        'fitness': float(la.mean() * 100.0 + sa.mean()),
    }


def _ingame_fitness(tree, seeds, max_placements) -> float:
    """fitness = avg_lines * 100 + avg_placements (survival is the tie-breaker)."""
    lines, steps = play_episodes(tree, seeds, max_placements)
    return sum(lines) / len(lines) * 100.0 + sum(steps) / len(steps)


def evolve_ingame(
    seed_tree: EMLNode,
    *,
    generations: int | None = None,
    population_size: int | None = None,
    elitism: int | None = None,
    patience: int | None = None,
    n_games: int | None = None,
    max_placements: int | None = None,
    train_pool: list[int] | None = None,
    val_pool: list[int] | None = None,
    val_games: int | None = None,
    basis=None,
    verbose: bool = True,
    should_stop=None,
    on_event=None,
) -> dict:
    """
    In-game evolution of the formula: fitness = the actual score in TetrisEnv.
    Seeded with the best formula of the DATA phase.

    Seed discipline (this is the part that decides whether the final number
    means anything):
      - every generation draws n_games seeds from train_pool at a rotating
        offset, so the population cannot memorise one piece sequence;
      - inside a generation all individuals share those seeds (common random
        numbers), so ranking compares formulas, not luck;
      - the generation champion is re-scored on the disjoint val_pool, and the
        returned tree is the best BY VALIDATION. Taking a max of train fitness
        across generations would select the luckiest rotation instead of the
        best formula — the rotation makes that failure mode certain, so the
        hall-of-fame is not optional here.

    on_event: an optional callback(dict), {'type':'joint_gen', ...} every
    generation.

    Returns: {'best_tree', 'best_fitness', 'best_lines', 'val_result',
              'history', 'val_history'}.
    """
    if generations is None:     generations = config.EML_JOINT_GENERATIONS
    if population_size is None:  population_size = config.EML_JOINT_POPULATION
    if elitism is None:         elitism = config.EML_JOINT_ELITISM
    if patience is None:        patience = config.EML_JOINT_PATIENCE
    if n_games is None:         n_games = config.EML_INGAME_GAMES
    if max_placements is None:  max_placements = config.EML_INGAME_MAX_PLACEMENTS
    if train_pool is None:      train_pool = config.EML_SEED_POOL_TRAIN
    if val_pool is None:        val_pool = config.EML_SEED_POOL_VAL
    if val_games is None:       val_games = config.EML_VAL_GAMES
    if basis is None:           basis = make_basis()
    elitism = max(1, min(elitism, population_size))

    val_seeds = pool_slice(val_pool, val_games)

    # Population: the original plus mutants of the seed formula.
    population = search.seed_population(basis, population_size,
                                        seed_tree=seed_tree, intensity=0.7)

    history: list[float] = []
    val_history: list[float] = []
    best_val_fitness = -math.inf
    best_val_result: dict | None = None
    best_tree_ever: EMLNode | None = None
    patience_counter = 0

    for gen in range(generations):
        if should_stop is not None and should_stop():
            break

        intensity = max(0.15, 1.0 - gen / (generations * 0.8))
        # Rotate the seeds every generation, share them within the generation.
        gen_seeds = pool_slice(train_pool, n_games, offset=gen * n_games)
        fitness_scores = [_ingame_fitness(t, gen_seeds, max_placements)
                          for t in population]

        ranked = sorted(zip(fitness_scores, population),
                        key=lambda p: p[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [t for _, t in ranked]

        best_fitness = fitness_scores[0]
        history.append(best_fitness)

        # Hall of fame: the champion is judged on seeds it never trained on.
        val_res = score_episodes(population[0], val_seeds, max_placements)
        val_history.append(val_res['fitness'])
        if val_res['fitness'] > best_val_fitness:
            best_val_fitness = val_res['fitness']
            best_val_result = val_res
            best_tree_ever = population[0].clone()
            patience_counter = 0
        else:
            patience_counter += 1

        best_lines = best_fitness / 100.0
        if verbose:
            print(f"    [GAME] gen {gen:>3} | train {best_fitness:9.1f} "
                  f"(~{best_lines:.1f} lines) | val {val_res['avg_lines']:6.2f}"
                  f"±{val_res['std_lines']:.2f} lines | i={intensity:.2f}")
        if on_event is not None:
            on_event({'type': 'joint_gen', 'gen': gen, 'generations': generations,
                      'fitness': best_fitness, 'lines': best_lines,
                      'val_lines': val_res['avg_lines']})

        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"    [GAME] early stop @ gen {gen}")
            break

        # Immigrants here are mutants of the DATA-phase formula rather than
        # random trees: a random tree scores zero in-game and would only waste
        # evaluations.
        population = search.breed(
            population, fitness_scores, basis,
            size=population_size, elitism=elitism, intensity=intensity,
            p_mutate=0.5, p_cross=0.85, p_cross_mutate=0.4, tournament_k=4,
            fresh=lambda: search.mutate_tree(seed_tree, basis,
                                             intensity=intensity))

    if best_tree_ever is None:
        best_tree_ever = population[0]
        best_val_result = score_episodes(best_tree_ever, val_seeds, max_placements)
    return {
        'best_tree': best_tree_ever,
        'best_fitness': best_val_result['fitness'],
        'best_lines': best_val_result['avg_lines'],
        'val_result': best_val_result,
        'history': history,
        'val_history': val_history,
    }


def split_groups(groups: np.ndarray, val_fraction: float,
                 rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Row masks for a GROUP-wise train/val split of the distillation dataset.

    The split has to be by group (= one board state with all its legal
    placements), not by row: the formula is only ever asked to rank placements
    within one step, so a step split across train and val would make validation
    almost free and useless as a check.
    """
    uniq = np.unique(groups)
    n_val = int(round(len(uniq) * val_fraction))
    if n_val < 1 or n_val >= len(uniq):
        return np.ones(len(groups), dtype=bool), np.zeros(len(groups), dtype=bool)
    val_groups = rng.choice(uniq, size=n_val, replace=False)
    val_mask = np.isin(groups, val_groups)
    return ~val_mask, val_mask


def _standardize_per_group(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """z-score of the target within each group (step): preserves the argmax."""
    out = np.empty_like(y)
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    start = 0
    for end in list(boundaries) + [len(groups)]:
        seg = y[start:end]
        out[start:end] = (seg - seg.mean()) / (seg.std() + 1e-8)
        start = end
    return out


# ── Full distillation pipeline ────────────────────────────────────────────────

def distill(
    features: np.ndarray,
    logits: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    groups: np.ndarray | None = None,
    depth_penalty_name: str = 'medium',
    data_generations: int | None = None,
    data_population: int | None = None,
    joint: bool = True,
    verbose: bool = True,
    should_stop=None,
    on_event=None,
) -> dict:
    """
    DATA phase (regression to the logits) -> GAME phase (in-game evolution).

    Args:
        features: (N, 19), logits: (N,) — from dataset_collector.
        weights:  (N,) per-sample weights (None -> by group if available, else 1).
        joint:    enable the GAME phase (can be turned off for debugging).
        on_event: an optional callback(dict) for progress (GUI).

    Returns:
        dict with 'trees' ([formula] — a list for storage compatibility),
        the DATA/GAME metrics and the history.
    """
    def _emit(ev):
        if on_event is not None:
            on_event(ev)

    if data_generations is None:
        data_generations = config.EML_GENERATIONS
    if data_population is None:
        data_population = config.EML_POPULATION
    if weights is None:
        if groups is not None:
            from dataset_collector import compute_sample_weights
            weights = compute_sample_weights(logits, groups)
        else:
            weights = np.ones(len(logits), dtype=np.float64)

    dp = DEPTH_PENALTIES[depth_penalty_name]
    # One basis for the whole run: both phases must search the same language,
    # and it is recorded in the result so a formula can be rebuilt later.
    basis = make_basis()
    X = np.ascontiguousarray(features, dtype=np.float64)
    y = np.asarray(logits, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)

    # Target normalization. The absolute scale of the oracle logits is arbitrary
    # (mean -46, std 48 on a real dataset) — only the RANKING of placements
    # within a step matters. A z-score within a group makes the target
    # scale-free (MSE ~ O(1)) and preserves the argmax; without groups a global
    # z-score is used.
    if groups is not None:
        y = _standardize_per_group(y, np.asarray(groups))
    else:
        y = (y - y.mean()) / (y.std() + 1e-8)

    # Group-wise train/val split of the dataset (see split_groups).
    if groups is not None:
        split_rng = np.random.default_rng(random.randrange(2 ** 63))
        tr_mask, va_mask = split_groups(
            np.asarray(groups), config.EML_DATA_VAL_FRACTION, split_rng)
        val_data = (X[va_mask], y[va_mask], w[va_mask]) if va_mask.any() else None
        X_tr, y_tr, w_tr = X[tr_mask], y[tr_mask], w[tr_mask]
    else:
        val_data = None
        X_tr, y_tr, w_tr = X, y, w

    # ── DATA phase: regression of a single formula ───────────────────────────
    _emit({'type': 'phase', 'phase': 'data', 'n_actions': 1})
    if verbose:
        n_val = 0 if val_data is None else len(val_data[0])
        print(f"  [DATA] evolving placement-value formula "
              f"({len(X_tr):,} train / {n_val:,} val samples)")
    _emit({'type': 'data_action_start', 'action': 0, 'name': 'PLACE'})
    res = evolve_formula(
        X_tr, y_tr, w_tr,
        basis=basis,
        generations=data_generations,
        population_size=data_population,
        elitism=config.EML_ELITISM,
        patience=config.EML_PATIENCE,
        depth_penalty=dp,
        val_data=val_data,
        verbose=verbose,
        label='PLACE',
        should_stop=should_stop,
        on_event=on_event,
    )
    data_tree = res['best_tree']
    if verbose:
        print(f"    -> fit={res['best_fitness']:.2f}  D{data_tree.depth()} "
              f"S{data_tree.size()} V{data_tree.n_unique_vars()}  "
              f"{data_tree.to_string()[:70]}")
    _emit({'type': 'data_action_done', 'action': 0, 'name': 'PLACE',
           'fitness': res['best_fitness'], 'depth': data_tree.depth(),
           'size': data_tree.size(), 'nvars': data_tree.n_unique_vars(),
           'formula': data_tree.to_string()})

    # Baseline in-game score of the DATA phase formula (before in-game tuning),
    # measured on the VAL pool — the same seeds the GAME phase champion will be
    # judged on, so the two phases are actually comparable.
    val_seeds = pool_slice(config.EML_SEED_POOL_VAL, config.EML_VAL_GAMES)
    base_val = score_episodes(data_tree, val_seeds,
                              config.EML_INGAME_MAX_PLACEMENTS)
    base_lines = base_val['avg_lines']
    if verbose:
        print(f"  DATA-phase val score: {base_lines:.2f}±{base_val['std_lines']:.2f} "
              f"lines/game ({len(val_seeds)} games)")
    _emit({'type': 'base_score', 'base_lines': base_lines})

    # ── GAME phase: in-game tuning ──────────────────────────────────────────
    if joint:
        if verbose:
            print("  [GAME] in-game evolution...")
        _emit({'type': 'phase', 'phase': 'joint'})
        jres = evolve_ingame(data_tree, basis=basis, verbose=verbose,
                             should_stop=should_stop, on_event=on_event)
        final_tree = jres['best_tree']
        val_result = jres['val_result']
        # Do not let the GAME phase make the result worse than DATA (both
        # numbers now come from the same val seeds).
        if val_result['avg_lines'] < base_lines:
            if verbose:
                print(f"  GAME ({val_result['avg_lines']:.2f}) < "
                      f"DATA ({base_lines:.2f}) -> keep DATA tree")
            final_tree = data_tree
            val_result = base_val
    else:
        jres = None
        final_tree = data_tree
        val_result = base_val

    return {
        'trees': [final_tree],
        'data_trees': [data_tree],
        'data_results': [res],
        'joint_result': jres,
        'base_lines': base_lines,
        'base_val': base_val,
        # Selection metric. NOT the number to report: the winner of a batch is
        # picked by it, so it is biased upward. The report comes from the TEST
        # pool (pipeline.evaluate_holdout), which nothing selects on.
        'val_lines': val_result['avg_lines'],
        'val_result': val_result,
        'static': static_metrics(final_tree),
        'basis': basis.name,
        'depth_penalty': depth_penalty_name,
        'dataset_size': len(features),
    }
