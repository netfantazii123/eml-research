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

from emlcore import (Node, search, static_metrics, pool_slice,
                     ParetoArchive, nsga2, checkpoint, parallel)
from emlcore.cache import FitnessCache
from emlcore import make_basis as _make_basis


# ── Basis ─────────────────────────────────────────────────────────────────────

def make_basis(name: str | None = None, max_depth: int | None = None,
               max_nodes: int | None = None) -> "object":
    """
    The basis this task searches in, built from config at call time.

    Built per call rather than once at import: batch_distill varies
    EML_MAX_DEPTH between runs, and a basis captured at import would silently
    ignore that. max_nodes is the node budget K, used by the fixed-K anchor runs
    that check the NSGA-II curve against dedicated ones.
    """
    return _make_basis(
        name or config.EML_BASIS,
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


def _search_opts(**over) -> dict:
    """The search options from config, overridable per call."""
    opts = {
        'unique': getattr(config, 'EML_UNIQUE_CHILDREN', False),
        'simplify': getattr(config, 'EML_FOLD_CONSTANTS', False),
        'tune_top': getattr(config, 'EML_TUNE_CONSTANTS_TOP', 0),
        'tune_steps': getattr(config, 'EML_TUNE_CONSTANTS_STEPS', 6),
    }
    opts.update({k: v for k, v in over.items() if v is not None})
    return opts


def _run_control(phase: str, basis, signature: dict, generations: int, *,
                 checkpoint_dir, resume, resume_point, extra_generations,
                 allow_changes, verbose) -> checkpoint.ResumableRun:
    return checkpoint.ResumableRun(
        phase=phase, basis=basis, signature=signature, generations=generations,
        checkpoint_dir=checkpoint_dir, resume=resume, point=resume_point,
        extra_generations=extra_generations, allow_changes=allow_changes,
        checkpoint_every=getattr(config, 'EML_CHECKPOINT_EVERY', 10),
        snapshot_every=getattr(config, 'EML_SNAPSHOT_EVERY', 50),
        keep_best=getattr(config, 'EML_KEEP_BEST_STATES', 5),
        hof_size=getattr(config, 'EML_HOF_SIZE', 20),
        verbose=verbose)


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
    multi_objective: bool | None = None,
    archive: ParetoArchive | None = None,
    verbose: bool = False,
    label: str = 'PLACE',
    should_stop=None,
    on_event=None,
    checkpoint_dir: str | None = None,
    resume=None,
    resume_point='latest',
    extra_generations: int = 0,
    allow_changes: tuple = (),
    init_trees: list | None = None,
    init_warm_fraction: float | None = None,
    search_opts: dict | None = None,
    cache: FitnessCache | None = None,
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

    Run state (see emlcore/checkpoint.py):
      checkpoint_dir — save latest / snapshots / best-performer states and the
          hall of fame there;
      resume — a checkpoint directory or state file to continue from, at
          resume_point ('latest', 'best' or a generation number);
      extra_generations — continue past the saved run's end (fine-tuning);
      allow_changes — signature keys allowed to differ, e.g. ('data',) to
          continue on a regrown dataset;
      init_trees — start a NEW run from these formulas (e.g. a hall of fame)
          instead of random trees.
      init_warm_fraction — with init_trees: only this share of the initial
          population are mutants of them (intensity 0.5, the trees included)
          and the rest stays random, so a poor warm start cannot capture the
          run (DAgger refits on a grown dataset use 0.5). None (default): the
          whole population starts around init_trees.

    Returns: {'best_tree', 'best_fitness', 'best_val_fitness', 'history',
              'val_history', 'archive', 'generation', 'cache', ...}.
    """
    if subsample is None:
        subsample = config.EML_DATA_SUBSAMPLE
    if basis is None:
        basis = make_basis()
    elitism = max(1, min(elitism, population_size))
    if multi_objective is None:
        multi_objective = config.EML_MULTI_OBJECTIVE
    opts = search_opts if search_opts is not None else _search_opts()
    if cache is None and getattr(config, 'EML_FITNESS_CACHE', True):
        cache = FitnessCache()

    # Size is an explicit objective under NSGA-II, so keeping a size penalty
    # inside the fitness too would count it twice: the front collapses onto the
    # smallest trees and R(K) stops rising well before the budget runs out.
    # Single-objective runs keep the penalty — it is the only size pressure they
    # have.
    if multi_objective:
        depth_penalty = 0.0

    signature = {
        'data': checkpoint.data_fingerprint(X, y, w),
        'val': checkpoint.data_fingerprint(*(val_data or (None,))),
        'population_size': int(population_size), 'elitism': int(elitism),
        'multi_objective': bool(multi_objective),
        'depth_penalty': float(depth_penalty), 'subsample': int(subsample),
        'search': {k: opts[k] for k in sorted(opts)},
    }
    run = _run_control('data', basis, signature, generations,
                       checkpoint_dir=checkpoint_dir, resume=resume,
                       resume_point=resume_point,
                       extra_generations=extra_generations,
                       allow_changes=allow_changes, verbose=verbose)

    if run.restored is not None:
        st = run.restored
        population = st['population']
        c = st['counters']
        history = list(c['history'])
        val_history = list(c['val_history'])
        best_fitness_ever = c['best_fitness_ever']
        best_val_fitness = c['best_val_fitness']
        best_tree_ever = c['best_tree_ever']
        patience_counter = c['patience_counter']
        rng = st['np_rng']
        if archive is None:
            archive = st['archive'] or ParetoArchive()
        else:
            archive.best.update((st['archive'] or ParetoArchive()).best)
    else:
        # Derived from the seeded global RNG: reproducible per run, yet actually
        # different across runs (a hardcoded default_rng(0) handed every "seed"
        # the very same subsample sequence).
        rng = np.random.default_rng(random.randrange(2 ** 63))
        if init_trees and init_warm_fraction is not None:
            population = search.seed_population(basis, population_size,
                                                max_depth=3)
            n_warm = int(population_size * init_warm_fraction)
            warm: list = []
            for i, t in enumerate(init_trees):
                share = (n_warm // len(init_trees)
                         + (i < n_warm % len(init_trees)))
                warm += search.seed_population(basis, share, seed_tree=t,
                                               intensity=0.5)
            population = warm[:n_warm] + population[:population_size - n_warm]
        else:
            population = search.seed_population(basis, population_size,
                                                max_depth=3,
                                                seed_trees=init_trees)
        history: list[float] = []
        val_history: list[float] = []
        best_fitness_ever = -1.0
        best_val_fitness = -math.inf
        best_tree_ever: EMLNode | None = None
        patience_counter = 0
        if archive is None:
            archive = ParetoArchive()

    def _fit(trees, Xs, ys, ws, ctx):
        def one(t):
            return _regression_fitness(t, Xs, ys, ws, depth_penalty)
        if cache is None:
            return [one(t) for t in trees]
        return cache.map(trees, ctx, one)

    def _counters():
        return {'history': history, 'val_history': val_history,
                'best_fitness_ever': best_fitness_ever,
                'best_val_fitness': best_val_fitness,
                'best_tree_ever': best_tree_ever,
                'patience_counter': patience_counter}

    gen = run.start_gen
    finished = False
    for gen in range(run.start_gen, run.generations):
        run.top(gen, population, counters=_counters(), rng=rng, archive=archive)
        if should_stop is not None and should_stop():
            break

        intensity = run.intensity(gen, floor=0.1, span=0.7)

        # Subsample for speed (fresh every generation -> less overfitting).
        if len(X) > subsample:
            idx = rng.choice(len(X), subsample, replace=False)
            Xs, ys, ws = X[idx], y[idx], w[idx]
            ctx = ('gen', gen)
        else:
            Xs, ys, ws = X, y, w
            ctx = 'full'

        fitness_scores = _fit(population, Xs, ys, ws, ctx)
        # The archive is what the R(K) curve is read from. Filled from training
        # fitness because it is already computed; the front is re-scored on
        # validation once the run is over, so no curve point is a subsample fluke.
        archive.update_many(population, fitness_scores)

        ranked = sorted(zip(fitness_scores, population),
                        key=lambda p: p[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [t for _, t in ranked]

        if opts.get('tune_top', 0) > 0:
            # Local constant search on the elites, on this generation's data.
            for i in range(min(opts['tune_top'], len(population))):
                t, f = search.tune_constants(
                    population[i],
                    lambda tr: _regression_fitness(tr, Xs, ys, ws, depth_penalty),
                    steps=opts.get('tune_steps', 6), current=fitness_scores[i])
                population[i], fitness_scores[i] = t, f
            order = sorted(range(len(population)),
                           key=lambda i: fitness_scores[i], reverse=True)
            population = [population[i] for i in order]
            fitness_scores = [fitness_scores[i] for i in order]

        best_fitness = fitness_scores[0]
        history.append(best_fitness)
        best_fitness_ever = max(best_fitness_ever, best_fitness)

        # Hall of fame on held-out groups (falls back to train fitness when no
        # validation split was supplied, e.g. tiny debug datasets).
        if val_data is not None:
            val_fitness = _fit([population[0]], *val_data, 'val')[0]
        else:
            val_fitness = best_fitness
        val_history.append(val_fitness)

        improved = val_fitness > best_val_fitness
        if improved:
            best_val_fitness = val_fitness
            best_tree_ever = population[0].clone()
            patience_counter = 0
        else:
            patience_counter += 1
        run.end_of_generation(improved, population[0], val_fitness,
                              train_fitness=best_fitness)

        bt = population[0]
        if verbose and (gen % 25 == 0 or gen == run.generations - 1):
            print(f"      [{label}] gen {gen:>4} | fit {best_fitness:10.2f} | "
                  f"val {val_fitness:10.2f} | "
                  f"D{bt.depth()} S{bt.size()} V{bt.n_unique_vars()} | "
                  f"i={intensity:.2f}")
        if on_event is not None:
            on_event({'type': 'data_gen', 'action': 0, 'name': label,
                      'gen': gen, 'generations': run.generations,
                      'fitness': best_fitness, 'val_fitness': val_fitness,
                      'depth': bt.depth(),
                      'size': bt.size(), 'nvars': bt.n_unique_vars()})

        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"      [{label}] early stop @ gen {gen} "
                      f"(no improve {patience})")
            break

        # New generation.
        if multi_objective:
            # NSGA-II over (quality, -size): survivors come from parents plus
            # offspring by front, so small trees keep selection pressure of
            # their own instead of being bred out by the best-scoring big ones.
            objectives = np.array(
                [[f, -t.size()] for f, t in zip(fitness_scores, population)],
                dtype=np.float64)
            ranks, crowd = nsga2.rank_and_crowding(objectives)
            offspring = search.breed_nsga2(
                population, ranks, crowd, basis,
                size=population_size, intensity=intensity,
                p_mutate=0.45, p_cross=0.75, p_cross_mutate=0.3,
                tournament_k=3, unique=opts.get('unique', False),
                simplify=opts.get('simplify', False))
            off_fitness = _fit(offspring, Xs, ys, ws, ctx)
            archive.update_many(offspring, off_fitness)

            combined = population + offspring
            comb_obj = np.array(
                [[f, -t.size()]
                 for f, t in zip(fitness_scores + off_fitness, combined)],
                dtype=np.float64)
            keep = nsga2.environmental_selection(comb_obj, population_size)
            population = [combined[i] for i in keep]
        else:
            population = search.breed(
                population, fitness_scores, basis,
                size=population_size, elitism=elitism, intensity=intensity,
                p_mutate=0.45, p_cross=0.75, p_cross_mutate=0.3,
                tournament_k=5, unique=opts.get('unique', False),
                simplify=opts.get('simplify', False))
    else:
        finished = True

    if finished:
        run.finish(population, counters=_counters(), rng=rng, archive=archive,
                   next_gen=run.generations)
    else:
        run.finish()

    if best_tree_ever is None:
        best_tree_ever = population[0]
    return {'best_tree': best_tree_ever,
            'best_fitness': best_fitness_ever,
            'best_val_fitness': best_val_fitness,
            'depth_penalty': depth_penalty,   # effective value, see above
            'archive': archive,
            'multi_objective': multi_objective,
            'history': history,
            'val_history': val_history,
            'generation': gen,
            'resumed_from': run.resumed_from,
            'cache': cache.stats() if cache is not None else None}


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
        env = TetrisEnv(seed=int(s), max_placements=0)   # the loop is the cap
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
        'n_capped': int((sa >= max_placements).sum()),
    }


def _ingame_fitness(tree, seeds, max_placements) -> float:
    """fitness = avg_lines * 100 + avg_placements (survival is the tie-breaker)."""
    lines, steps = play_episodes(tree, seeds, max_placements)
    return sum(lines) / len(lines) * 100.0 + sum(steps) / len(steps)


_WORKER_BASES: dict = {}


def _ingame_worker(payload: dict) -> float:
    """One GAME-phase fitness inside a worker process (see emlcore.parallel)."""
    spec = payload['basis']
    key = (spec['name'], spec['max_depth'], spec['max_nodes'])
    if key not in _WORKER_BASES:
        _WORKER_BASES[key] = make_basis(spec['name'], max_depth=spec['max_depth'],
                                        max_nodes=spec['max_nodes'])
    tree = Node.from_dict(payload['tree'], _WORKER_BASES[key])
    return _ingame_fitness(tree, payload['seeds'], payload['max_placements'])


def _ingame_scores(trees, seeds, max_placements, basis, *, workers: int = 1,
                   cache: FitnessCache | None = None) -> list[float]:
    """GAME fitness of a population: cached, and in parallel when workers > 1."""
    seeds = [int(s) for s in seeds]

    def batch(ts):
        if workers <= 1:
            return [_ingame_fitness(t, seeds, max_placements) for t in ts]
        spec = checkpoint.basis_spec(basis)
        return parallel.map_ordered(
            _ingame_worker,
            [{'tree': t.to_dict(), 'basis': spec, 'seeds': seeds,
              'max_placements': max_placements} for t in ts], workers)

    if cache is None:
        return batch(list(trees))
    return cache.map(trees, ('game', tuple(seeds), max_placements), None,
                     batch_compute=batch)


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
    checkpoint_dir: str | None = None,
    resume=None,
    resume_point='latest',
    extra_generations: int = 0,
    allow_changes: tuple = (),
    init_trees: list | None = None,
    search_opts: dict | None = None,
    workers: int | None = None,
    cache: FitnessCache | None = None,
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

    Cost: the val re-scoring of a champion that is still last generation's
    champion (the usual case once the run settles) and duplicate formulas inside
    a generation are served from the fitness cache instead of being replayed.
    workers > 1 plays a generation's games in parallel processes. Neither
    changes the result.

    Run state: the same checkpoint_dir / resume / resume_point /
    extra_generations / allow_changes / init_trees as evolve_formula.

    on_event: an optional callback(dict), {'type':'joint_gen', ...} every
    generation.

    Returns: {'best_tree', 'best_fitness', 'best_lines', 'val_result',
              'history', 'val_history', 'generation', 'cache'}.
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
    if workers is None:
        workers = getattr(config, 'EML_INGAME_WORKERS', 1)
    workers = parallel.resolve_workers(workers)
    if cache is None and getattr(config, 'EML_FITNESS_CACHE', True):
        cache = FitnessCache()
    opts = search_opts if search_opts is not None else _search_opts()
    elitism = max(1, min(elitism, population_size))

    val_seeds = pool_slice(val_pool, val_games)
    val_ctx = ('val', tuple(int(s) for s in val_seeds), max_placements)

    def _val(tree):
        if cache is None:
            return score_episodes(tree, val_seeds, max_placements)
        return cache.get(tree, val_ctx,
                         lambda t: score_episodes(t, val_seeds, max_placements))

    signature = {
        'seed_tree': seed_tree.key() if seed_tree is not None else None,
        'population_size': int(population_size), 'elitism': int(elitism),
        'n_games': int(n_games), 'max_placements': int(max_placements),
        'train_pool': [int(s) for s in train_pool],
        'val_seeds': [int(s) for s in val_seeds],
        'search': {k: opts[k] for k in ('unique', 'simplify')},
    }
    run = _run_control('game', basis, signature, generations,
                       checkpoint_dir=checkpoint_dir, resume=resume,
                       resume_point=resume_point,
                       extra_generations=extra_generations,
                       allow_changes=allow_changes, verbose=verbose)

    if run.restored is not None:
        c = run.restored['counters']
        population = run.restored['population']
        history = list(c['history'])
        val_history = list(c['val_history'])
        best_val_fitness = c['best_val_fitness']
        best_tree_ever = c['best_tree_ever']
        best_val_result = c['best_val_result']
        patience_counter = c['patience_counter']
    else:
        # Population: the original plus mutants of the seed formula (or of a
        # whole hall of fame, when fine-tuning from one).
        population = search.seed_population(
            basis, population_size, seed_tree=seed_tree, intensity=0.7,
            seed_trees=init_trees)
        history: list[float] = []
        val_history: list[float] = []
        best_val_fitness = -math.inf
        best_val_result: dict | None = None
        best_tree_ever: EMLNode | None = None
        patience_counter = 0

    def _counters():
        return {'history': history, 'val_history': val_history,
                'best_val_fitness': best_val_fitness,
                'best_val_result': best_val_result,
                'best_tree_ever': best_tree_ever,
                'patience_counter': patience_counter}

    gen = run.start_gen
    finished = False
    for gen in range(run.start_gen, run.generations):
        run.top(gen, population, counters=_counters())
        if should_stop is not None and should_stop():
            break

        intensity = run.intensity(gen, floor=0.15, span=0.8)
        # Rotate the seeds every generation, share them within the generation.
        gen_seeds = pool_slice(train_pool, n_games, offset=gen * n_games)
        fitness_scores = _ingame_scores(population, gen_seeds, max_placements,
                                        basis, workers=workers, cache=cache)

        ranked = sorted(zip(fitness_scores, population),
                        key=lambda p: p[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [t for _, t in ranked]

        best_fitness = fitness_scores[0]
        history.append(best_fitness)

        # Hall of fame: the champion is judged on seeds it never trained on.
        val_res = _val(population[0])
        val_history.append(val_res['fitness'])
        improved = val_res['fitness'] > best_val_fitness
        if improved:
            best_val_fitness = val_res['fitness']
            best_val_result = val_res
            best_tree_ever = population[0].clone()
            patience_counter = 0
        else:
            patience_counter += 1
        run.end_of_generation(improved, population[0], val_res['fitness'],
                              val_lines=val_res['avg_lines'])

        best_lines = best_fitness / 100.0
        if verbose:
            print(f"    [GAME] gen {gen:>3} | train {best_fitness:9.1f} "
                  f"(~{best_lines:.1f} lines) | val {val_res['avg_lines']:6.2f}"
                  f"±{val_res['std_lines']:.2f} lines | i={intensity:.2f}")
        if on_event is not None:
            on_event({'type': 'joint_gen', 'gen': gen,
                      'generations': run.generations,
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
            # No-oracle control (seed_tree None): random immigrants.
            fresh=(None if seed_tree is None else
                   lambda: search.mutate_tree(seed_tree, basis,
                                              intensity=intensity)),
            unique=opts.get('unique', False),
            simplify=opts.get('simplify', False))
    else:
        finished = True

    if finished:
        run.finish(population, counters=_counters(), next_gen=run.generations)
    else:
        run.finish()

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
        'generation': gen,
        'resumed_from': run.resumed_from,
        'cache': cache.stats() if cache is not None else None,
    }


def evaluate_pareto(archive: ParetoArchive, val_data, depth_penalty: float, *,
                    val_seeds, max_placements: int,
                    max_points: int | None = None,
                    refine: bool | None = None,
                    refine_generations: int | None = None,
                    basis=None,
                    verbose: bool = False) -> list[dict]:
    """
    Turn the size archive into the R(K) curve, scored on validation.

    Two numbers per budget, and they are not interchangeable: the regression
    fitness the search optimised, and the in-game lines the thesis is about.
    Where they disagree is itself a result — a formula can agree with the oracle
    on most placements and still fall apart in a game, or the reverse.

    refine matters more than it looks. Without it the curve is read off DATA
    phase trees, which score near zero lines no matter how good their regression
    fitness is — that is precisely why the GAME phase exists. So the unrefined
    curve is an R(K) in regression fitness only, and quoting its line counts as
    a game result would be wrong. With refine, each front member gets its own
    short GAME phase first, which costs one in-game evolution per point and
    gives the curve the thesis actually needs.

    In-game evaluation costs real games, so only max_points sizes are played,
    spread across the front rather than taken from its best-scoring end.
    """
    if max_points is None:
        max_points = config.EML_PARETO_MAX_POINTS
    front = archive.front()
    if not front:
        return []

    # Evenly spaced across the front: the small-K end is the interesting one and
    # would be dropped by any "take the top N" rule.
    if len(front) > max_points:
        idx = np.linspace(0, len(front) - 1, max_points).round().astype(int)
        picked = [front[i] for i in sorted(set(idx.tolist()))]
    else:
        picked = front

    if refine is None:
        refine = config.EML_PARETO_REFINE
    if refine_generations is None:
        refine_generations = config.EML_PARETO_REFINE_GENERATIONS

    rows = []
    for size, train_score, tree in picked:
        if val_data is not None:
            val_fit = _regression_fitness(tree, val_data[0], val_data[1],
                                          val_data[2], depth_penalty)
        else:
            val_fit = train_score

        played = tree
        if refine:
            # The budget must survive refinement, or the point stops being a
            # point about K: the basis carries max_nodes = this size.
            k_basis = make_basis(
                (basis.name if basis is not None else None), max_nodes=size)
            jres = evolve_ingame(
                tree, basis=k_basis, generations=refine_generations,
                verbose=False)
            if jres['val_result']['avg_lines'] >= 0:
                played = jres['best_tree']

        game = score_episodes(played, val_seeds, max_placements)
        stat = static_metrics(played)
        rows.append({
            'k': size,
            'refined': bool(refine),
            'formula_nodes': played.size(),
            'train_fitness': train_score,
            'val_fitness': val_fit,
            'val_lines': game['avg_lines'],
            'val_lines_std': game['std_lines'],
            'nodes': stat['nodes'], 'depth': stat['depth'],
            'n_vars': stat['n_vars_unique'],
            'n_exp': stat['n_exp'], 'n_log': stat['n_log'],
            'n_transcendental': stat['n_transcendental'],
            'est_code_bytes': stat['est_code_bytes'],
            'est_stack_bytes': stat['est_stack_bytes'],
            'formula': played.to_string(),
        })
        if verbose:
            print(f"      K={size:>4} | val_fit {val_fit:8.2f} | "
                  f"{game['avg_lines']:6.2f}±{game['std_lines']:.2f} lines | "
                  f"{stat['n_transcendental']:>3} transcendental")
    return rows


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

def _phase_dir(root: str | None, phase: str, must_exist: bool = False):
    """<root>/<phase>, or None (no root, or nothing saved there yet)."""
    if not root:
        return None
    d = os.path.join(root, phase)
    if must_exist and not checkpoint.list_points(d):
        return None
    return d


def _read_run_meta(root: str | None) -> dict:
    if not root:
        return {}
    p = os.path.join(root, 'run.json')
    try:
        return checkpoint.read_json(p)
    except (OSError, ValueError):
        return {}


def _write_run_meta(root: str | None, meta: dict) -> None:
    if root:
        checkpoint.atomic_write_json(os.path.join(root, 'run.json'),
                                     {**_read_run_meta(root), **meta})


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
    basis_name: str | None = None,
    max_nodes: int | None = None,
    multi_objective: bool | None = None,
    verbose: bool = True,
    should_stop=None,
    on_event=None,
    checkpoint_dir: str | None = None,
    resume: str | None = None,
    resume_point='latest',
    extra_generations: int = 0,
    allow_changes: tuple = (),
    workers: int | None = None,
) -> dict:
    """
    DATA phase (regression to the logits) -> GAME phase (in-game evolution).

    Run state: checkpoint_dir keeps each phase's states under <dir>/data and
    <dir>/game. resume=<an earlier checkpoint_dir> continues where that run
    stopped — the DATA phase from <resume>/data at resume_point, the GAME phase
    from <resume>/game if the earlier run got that far. extra_generations
    extends whichever phase is resumed. The dataset split is drawn from the
    global RNG, so resume with the same GA seed or the data signature will not
    match (allow_changes=('data', 'val') to continue on purpose anyway).

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
    # max_nodes set = a fixed-K anchor run, used to check the NSGA-II curve
    # against dedicated ones on at least one task.
    basis = make_basis(basis_name, max_nodes=max_nodes)
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
        # The split seed is drawn from the global RNG as before (so seeded runs
        # reproduce), and recorded next to the checkpoints: a resumed run must
        # split the dataset the same way, or its train/val data — and with it
        # the checkpoint signature — would differ from the run it continues.
        split_seed = random.randrange(2 ** 63)
        saved = _read_run_meta(resume)
        if saved.get('split_seed') is not None:
            split_seed = int(saved['split_seed'])
        _write_run_meta(checkpoint_dir, {'split_seed': split_seed})
        split_rng = np.random.default_rng(split_seed)
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
    archive = ParetoArchive()
    res = evolve_formula(
        X_tr, y_tr, w_tr,
        basis=basis,
        multi_objective=multi_objective,
        archive=archive,
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
        checkpoint_dir=_phase_dir(checkpoint_dir, 'data'),
        resume=_phase_dir(resume, 'data', must_exist=True),
        resume_point=resume_point,
        extra_generations=extra_generations,
        allow_changes=allow_changes,
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

    # ── R(K): quality against node budget, read off the archive ─────────────
    if verbose:
        print("  [PARETO] scoring the size front on validation...")
    pareto = evaluate_pareto(
        res['archive'], val_data, res['depth_penalty'],
        val_seeds=val_seeds,
        max_placements=config.EML_INGAME_MAX_PLACEMENTS,
        basis=basis, verbose=verbose)
    _emit({'type': 'pareto', 'points': [
        {k: v for k, v in row.items() if k != 'formula'} for row in pareto]})

    # ── GAME phase: in-game tuning ──────────────────────────────────────────
    if joint:
        if verbose:
            print("  [GAME] in-game evolution...")
        _emit({'type': 'phase', 'phase': 'joint'})
        game_resume = _phase_dir(resume, 'game', must_exist=True)
        jres = evolve_ingame(data_tree, basis=basis, verbose=verbose,
                             should_stop=should_stop, on_event=on_event,
                             checkpoint_dir=_phase_dir(checkpoint_dir, 'game'),
                             resume=game_resume,
                             # A chosen generation refers to the DATA phase;
                             # the GAME phase continues where it stopped.
                             resume_point='latest',
                             extra_generations=(extra_generations
                                                if game_resume else 0),
                             allow_changes=allow_changes, workers=workers)
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
        'max_nodes': basis.max_nodes,
        'multi_objective': res['multi_objective'],
        'pareto': pareto,
        'depth_penalty': depth_penalty_name,
        'dataset_size': len(features),
    }
