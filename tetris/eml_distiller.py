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

The AST engine is ported from flappy_bird_with_ai/eml_distiller.py; differences:
  - n_vars = 19 (afterstate features), names from config.FEATURE_NAMES;
  - a vectorized evaluate_batch() for fast evaluation over the dataset;
  - one formula instead of six — the argmax is now over placements rather than
    actions.
"""

import math
import time
import random

import numpy as np

import config
from config import EML_EPSILON, DEPTH_PENALTIES, FEATURE_NAMES, N_FEATURES
from env import TetrisEnv


# ── EML operator ──────────────────────────────────────────────────────────────

def eml_op(x: float, y: float) -> float:
    """eml(x, y) = exp(clamp(x)) - ln(|y| + eps). Scalar (numpy-free) version."""
    x_clamped = -10.0 if x < -10.0 else (10.0 if x > 10.0 else x)
    return math.exp(x_clamped) - math.log(abs(y) + EML_EPSILON)


# ── AST tree ──────────────────────────────────────────────────────────────────

class EMLNode:
    """
    A node of the EML formula AST.

    Kinds:
        'eml'   — eml(left, right)
        'var'   — input feature (index 0..N_FEATURES-1)
        'const' — numeric constant
    """

    __slots__ = ('kind', 'left', 'right', 'var_idx', 'value')

    def __init__(self, kind: str, left=None, right=None,
                 var_idx: int = 0, value: float = 1.0):
        self.kind = kind
        self.left = left
        self.right = right
        self.var_idx = var_idx
        self.value = value

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(self, inputs) -> float:
        """Scalar evaluation for a single feature vector (inference)."""
        if self.kind == 'const':
            return self.value
        if self.kind == 'var':
            return float(inputs[self.var_idx])
        # eml
        lv = self.left.evaluate(inputs)
        rv = self.right.evaluate(inputs)
        return eml_op(lv, rv)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation over the whole dataset X (N, 19) -> (N,).

        All operations are closed and numerically bounded (clamp + eps), so the
        result is always finite.
        """
        if self.kind == 'const':
            return np.full(X.shape[0], self.value, dtype=np.float64)
        if self.kind == 'var':
            return X[:, self.var_idx].astype(np.float64)
        lv = self.left.evaluate_batch(X)
        rv = self.right.evaluate_batch(X)
        return np.exp(np.clip(lv, -10.0, 10.0)) - np.log(np.abs(rv) + EML_EPSILON)

    # ── Metrics ───────────────────────────────────────────────────────────
    def depth(self) -> int:
        if self.kind in ('const', 'var'):
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        if self.kind in ('const', 'var'):
            return 1
        return 1 + self.left.size() + self.right.size()

    def _var_mask(self) -> int:
        if self.kind == 'var':
            return 1 << self.var_idx
        if self.kind == 'eml':
            return self.left._var_mask() | self.right._var_mask()
        return 0

    def n_unique_vars(self) -> int:
        return bin(self._var_mask()).count('1')

    # ── Serialization / printing ──────────────────────────────────────────
    def to_string(self) -> str:
        if self.kind == 'const':
            return f"{self.value:.3f}"
        if self.kind == 'var':
            return (FEATURE_NAMES[self.var_idx]
                    if self.var_idx < len(FEATURE_NAMES)
                    else f"x{self.var_idx}")
        return f"eml({self.left.to_string()}, {self.right.to_string()})"

    def clone(self) -> "EMLNode":
        if self.kind in ('const', 'var'):
            return EMLNode(self.kind, var_idx=self.var_idx, value=self.value)
        return EMLNode('eml', left=self.left.clone(), right=self.right.clone())

    def to_dict(self) -> dict:
        if self.kind == 'const':
            return {'kind': 'const', 'value': self.value}
        if self.kind == 'var':
            return {'kind': 'var', 'var_idx': self.var_idx}
        return {'kind': 'eml',
                'left': self.left.to_dict(),
                'right': self.right.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "EMLNode":
        kind = d['kind']
        if kind == 'const':
            return cls('const', value=float(d['value']))
        if kind == 'var':
            return cls('var', var_idx=int(d['var_idx']))
        if kind == 'eml':
            return cls('eml',
                       left=cls.from_dict(d['left']),
                       right=cls.from_dict(d['right']))
        raise ValueError(f"Unknown node kind: {kind}")


# ── Random tree generation ────────────────────────────────────────────────────

def _random_leaf() -> EMLNode:
    """Random leaf: a variable (70%) or a constant (30%)."""
    if random.random() < 0.7:
        return EMLNode('var', var_idx=random.randint(0, N_FEATURES - 1))
    return EMLNode('const', value=random.uniform(-2.0, 2.0))


def random_tree(max_depth: int = 3, force_eml: bool = False) -> EMLNode:
    """Random EML tree. force_eml -> the root is always eml (depth >= 1)."""
    if force_eml and max_depth >= 1:
        return EMLNode('eml',
                       left=random_tree(max_depth - 1, force_eml=False),
                       right=random_tree(max_depth - 1, force_eml=False))
    if max_depth <= 0 or random.random() < 0.3:
        return _random_leaf()
    return EMLNode('eml',
                   left=random_tree(max_depth - 1),
                   right=random_tree(max_depth - 1))


# ── Mutation / crossover ──────────────────────────────────────────────────────

def _collect_nodes(node: EMLNode) -> list[EMLNode]:
    nodes = [node]
    if node.kind == 'eml':
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    return nodes


def mutate_tree(root: EMLNode, intensity: float = 1.0) -> EMLNode:
    """Mutation with adaptive intensity (1.0 = wild, 0.1 = fine)."""
    tree = root.clone()
    n_mutations = 1 + int(intensity * 2 * random.random())

    for _ in range(n_mutations):
        nodes = _collect_nodes(tree)
        target = random.choice(nodes)
        r = random.random()

        if r < 0.2 and target.kind == 'const':
            noise_std = 0.2 + intensity * 1.0
            target.value += random.gauss(0, noise_std)
            target.value = max(-5.0, min(5.0, target.value))

        elif r < 0.35 and target.kind == 'var':
            target.var_idx = random.randint(0, N_FEATURES - 1)

        elif r < 0.6 and target.kind in ('const', 'var'):
            if tree.depth() < config.EML_MAX_DEPTH:
                target.kind = 'eml'
                target.left = _random_leaf()
                target.right = _random_leaf()

        elif r < 0.7 and target.kind == 'eml':
            if random.random() < 0.5:
                target.left = random_tree(max_depth=2)
            else:
                target.right = random_tree(max_depth=2)

        elif r < 0.8 and target.kind == 'eml' and intensity < 0.5:
            leaf = _random_leaf()
            target.kind = leaf.kind
            target.left = None
            target.right = None
            target.var_idx = leaf.var_idx
            target.value = leaf.value

        else:
            depth = 1 + int(intensity * 2)
            new_sub = random_tree(max_depth=depth, force_eml=True)
            target.kind = new_sub.kind
            target.left = new_sub.left
            target.right = new_sub.right
            target.var_idx = new_sub.var_idx
            target.value = new_sub.value

    return _prune_tree(tree)


def _prune_tree(node: EMLNode, max_depth: int | None = None) -> EMLNode:
    """Trim the tree to max_depth — deep subtrees become leaves."""
    if max_depth is None:
        max_depth = config.EML_MAX_DEPTH
    if node.kind in ('const', 'var'):
        return node
    if max_depth <= 0:
        return _random_leaf()
    node.left = _prune_tree(node.left, max_depth - 1)
    node.right = _prune_tree(node.right, max_depth - 1)
    return node


def crossover(parent1: EMLNode, parent2: EMLNode) -> EMLNode:
    """Replace a random subtree of parent1 with a subtree from parent2."""
    child = parent1.clone()
    target = random.choice(_collect_nodes(child))
    donor = random.choice(_collect_nodes(parent2)).clone()
    target.kind = donor.kind
    target.left = donor.left
    target.right = donor.right
    target.var_idx = donor.var_idx
    target.value = donor.value
    return _prune_tree(child)


def _tournament_select(fitness_scores: list[float], k: int = 5) -> int:
    n = len(fitness_scores)
    candidates = random.sample(range(n), min(k, n))
    return max(candidates, key=lambda i: fitness_scores[i])


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
    elitism = max(1, min(elitism, population_size))

    # Derived from the seeded global RNG: reproducible per run, yet actually
    # different across runs (a hardcoded default_rng(0) handed every "seed" the
    # very same subsample sequence).
    rng = np.random.default_rng(random.randrange(2 ** 63))
    population = [random_tree(max_depth=3, force_eml=True)
                 for _ in range(population_size)]
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
        new_pop = [population[i].clone() for i in range(elitism)]
        while len(new_pop) < population_size:
            r = random.random()
            if r < 0.45:
                i = _tournament_select(fitness_scores, k=5)
                child = mutate_tree(population[i], intensity=intensity)
            elif r < 0.75:
                i1 = _tournament_select(fitness_scores, k=5)
                i2 = _tournament_select(fitness_scores, k=5)
                child = crossover(population[i1], population[i2])
                if random.random() < 0.3:
                    child = mutate_tree(child, intensity=intensity * 0.5)
            else:
                child = random_tree(max_depth=2 + int(intensity * 2),
                                    force_eml=True)
            new_pop.append(child)
        population = new_pop

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

def pool_slice(pool: list[int], n_games: int, offset: int = 0) -> list[int]:
    """n_games seeds from the pool, starting at offset (wraps around)."""
    if not pool:
        raise ValueError("empty seed pool")
    return [pool[(offset + i) % len(pool)] for i in range(n_games)]


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
    elitism = max(1, min(elitism, population_size))

    val_seeds = pool_slice(val_pool, val_games)

    # Population: the original plus mutants of the seed formula.
    population = [seed_tree.clone()]
    while len(population) < population_size:
        population.append(mutate_tree(seed_tree, intensity=0.7))

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

        new_pop = [population[i].clone() for i in range(elitism)]
        while len(new_pop) < population_size:
            r = random.random()
            if r < 0.5:
                i = _tournament_select(fitness_scores, k=4)
                child = mutate_tree(population[i], intensity=intensity)
            elif r < 0.85:
                i1 = _tournament_select(fitness_scores, k=4)
                i2 = _tournament_select(fitness_scores, k=4)
                child = crossover(population[i1], population[i2])
                if random.random() < 0.4:
                    child = mutate_tree(child, intensity=intensity * 0.5)
            else:
                child = mutate_tree(seed_tree, intensity=intensity)
            new_pop.append(child)
        population = new_pop

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


def static_metrics(tree: EMLNode) -> dict:
    """
    Cheap resource proxies, readable without any hardware.

    Every 'eml' node compiles to exactly one expf + one logf + one fabsf in the
    generated C, so the transcendental call count — the number that decides
    whether this runs on a Cortex-M without an FPU — is just the operator count.
    """
    n_eml = n_var = n_const = 0
    for node in _collect_nodes(tree):
        if node.kind == 'eml':
            n_eml += 1
        elif node.kind == 'var':
            n_var += 1
        else:
            n_const += 1
    return {
        'nodes': n_eml + n_var + n_const,
        'depth': tree.depth(),
        'n_eml': n_eml,
        'n_exp': n_eml,
        'n_log': n_eml,
        'n_leaves': n_var + n_const,
        'n_const': n_const,
        'n_vars_unique': tree.n_unique_vars(),
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
        jres = evolve_ingame(data_tree, verbose=verbose,
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
        'depth_penalty': depth_penalty_name,
        'dataset_size': len(features),
    }
