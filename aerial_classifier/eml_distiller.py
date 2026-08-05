"""
eml_distiller.py — symbolic distillation of the classifier oracle into EML
formulas.

The idea: train NOT one formula but one per class. Each formula f_c produces a
scalar ("how much this looks like class c"), and the prediction is the argmax
over classes:

    class* = argmax_c f_c(inputs)

This is a direct analogue of Tetris, where the argmax was taken over piece
placements: there the formula ranked moves, here it ranks classes. The absolute
values of the formulas do not matter, only the order does, and that determines
both training phases.

EML operator:  eml(x, y) = exp(clamp(x, -10, 10)) - ln(|y| + eps).

Two phases:

  1. DATA phase — symbolic regression: f_c(inputs) ~ oracle logit_c.
     The logits are z-normalized beforehand WITHIN EACH EXAMPLE (across the 4
     classes). The absolute scale of the logits is arbitrary and differs from
     example to example; per-row normalization makes the target scale-free and,
     crucially, preserves the argmax — exactly the information we need.

  2. ACC phase — the analogue of the Tetris GAME phase ("the formula plays by
     itself"). MSE against the logits and accuracy are different things: a
     formula can reproduce the logits closely and still get the argmax wrong on
     borderline examples. The second phase therefore evolves THE WHOLE SET of 4
     formulas together, with fitness = actual balanced accuracy. It is seeded
     with the DATA phase result, so it starts from a meaningful approximation
     rather than from scratch.

On metric honesty: ACC-phase fitness is computed on a TRAIN SUBSAMPLE, never on
val/test. Otherwise val stops being an independent estimate — evolving against
it is the same kind of fitting as gradient training.

The AST engine (EMLNode, random_tree, mutate_tree, crossover) is ported from
tetris/eml_distiller.py; the differences are a set of formulas instead of one,
the ACC phase instead of the GAME phase, and a configurable input dimension
(32 features in branch A, CNN_EMBED_DIM embedding coordinates in branch B).
"""

import math
import random
import time

import numpy as np

import config
from config import EML_EPSILON

# ── Input space ───────────────────────────────────────────────────────────────
# Branches A and B feed the formula different vectors, so the dimension and the
# input names are set before evolution rather than taken from config rigidly.

_N_INPUTS = config.N_FEATURES
_INPUT_NAMES = list(config.FEATURE_NAMES)


def configure_inputs(n_inputs: int, names: list[str] | None = None):
    """Set the dimension and names of the formula inputs (call before evolving)."""
    global _N_INPUTS, _INPUT_NAMES
    _N_INPUTS = int(n_inputs)
    _INPUT_NAMES = list(names) if names else [f'x{i}' for i in range(n_inputs)]


def input_names() -> list[str]:
    return list(_INPUT_NAMES)


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
        'var'   — input coordinate (index 0.._N_INPUTS-1)
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
        """Scalar evaluation for a single input vector (inference)."""
        if self.kind == 'const':
            return self.value
        if self.kind == 'var':
            return float(inputs[self.var_idx])
        return eml_op(self.left.evaluate(inputs), self.right.evaluate(inputs))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized evaluation over the whole dataset X (N, n_inputs) -> (N,).

        The operations are closed and numerically bounded (clamp + eps), so the
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

    def used_vars(self) -> list[int]:
        """Indices of the inputs that actually occur in the tree."""
        mask = self._var_mask()
        return [i for i in range(_N_INPUTS) if mask >> i & 1]

    # ── Serialization / printing ──────────────────────────────────────────
    def to_string(self) -> str:
        if self.kind == 'const':
            return f"{self.value:.3f}"
        if self.kind == 'var':
            return (_INPUT_NAMES[self.var_idx]
                    if self.var_idx < len(_INPUT_NAMES)
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
            return cls('eml', left=cls.from_dict(d['left']),
                       right=cls.from_dict(d['right']))
        raise ValueError(f"unknown node kind: {kind}")


# ── Random tree generation ────────────────────────────────────────────────────

def _random_leaf() -> EMLNode:
    """Random leaf: a variable (70%) or a constant (30%)."""
    if random.random() < 0.7:
        return EMLNode('var', var_idx=random.randint(0, _N_INPUTS - 1))
    return EMLNode('const', value=random.uniform(-2.0, 2.0))


def random_tree(max_depth: int = 3, force_eml: bool = False) -> EMLNode:
    """Random EML tree. force_eml -> the root is always eml (depth >= 1)."""
    if force_eml and max_depth >= 1:
        return EMLNode('eml',
                       left=random_tree(max_depth - 1, force_eml=False),
                       right=random_tree(max_depth - 1, force_eml=False))
    if max_depth <= 0 or random.random() < 0.3:
        return _random_leaf()
    return EMLNode('eml', left=random_tree(max_depth - 1),
                   right=random_tree(max_depth - 1))


# ── Mutation / crossover ──────────────────────────────────────────────────────

def _collect_nodes(node: EMLNode) -> list[EMLNode]:
    nodes = [node]
    if node.kind == 'eml':
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    return nodes


def mutate_tree(root: EMLNode, intensity: float = 1.0,
                max_depth: int | None = None) -> EMLNode:
    """
    Mutation with adaptive intensity (1.0 = wild, 0.1 = fine).

    max_depth=None — the limit from config.EML_MAX_DEPTH (branches A and B).
    Branch C keeps one tree instead of N_CLASSES and passes its own, larger
    limit.
    """
    if max_depth is None:
        max_depth = config.EML_MAX_DEPTH
    tree = root.clone()
    n_mutations = 1 + int(intensity * 2 * random.random())

    for _ in range(n_mutations):
        target = random.choice(_collect_nodes(tree))
        r = random.random()

        if r < 0.2 and target.kind == 'const':
            target.value += random.gauss(0, 0.2 + intensity * 1.0)
            target.value = max(-5.0, min(5.0, target.value))

        elif r < 0.35 and target.kind == 'var':
            target.var_idx = random.randint(0, _N_INPUTS - 1)

        elif r < 0.6 and target.kind in ('const', 'var'):
            if tree.depth() < max_depth:
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
            target.kind, target.left, target.right = leaf.kind, None, None
            target.var_idx, target.value = leaf.var_idx, leaf.value

        else:
            new_sub = random_tree(max_depth=1 + int(intensity * 2), force_eml=True)
            target.kind = new_sub.kind
            target.left, target.right = new_sub.left, new_sub.right
            target.var_idx, target.value = new_sub.var_idx, new_sub.value

    return _prune_tree(tree, max_depth)


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


def crossover(parent1: EMLNode, parent2: EMLNode,
              max_depth: int | None = None) -> EMLNode:
    """Replace a random subtree of parent1 with a subtree from parent2."""
    child = parent1.clone()
    target = random.choice(_collect_nodes(child))
    donor = random.choice(_collect_nodes(parent2)).clone()
    target.kind = donor.kind
    target.left, target.right = donor.left, donor.right
    target.var_idx, target.value = donor.var_idx, donor.value
    return _prune_tree(child, max_depth)


def _tournament_select(fitness_scores, k: int = 5) -> int:
    candidates = random.sample(range(len(fitness_scores)),
                               min(k, len(fitness_scores)))
    return max(candidates, key=lambda i: fitness_scores[i])


# ── Formula-based classifier ──────────────────────────────────────────────────

class EMLClassifier:
    """
    A set of N_CLASSES formulas: prediction = argmax_c (f_c(inputs) - mu_c)/sd_c.

    This object is the end product of distillation. No torch inside — only
    numpy, and export.py translates the same thing into plain C.

    WHY CALIBRATION (mu_c, sd_c) IS NEEDED. It is not cosmetic but a condition of
    the thing working at all. In Tetris there was one formula and the argmax was
    taken over different INPUTS — the common output scale cancelled and did not
    matter. Here the argmax is taken between FOUR DIFFERENT formulas, each of
    which evolved independently and has its own arbitrary scale and offset.
    Without alignment the winner is not the most appropriate formula but the one
    that simply emits larger numbers: in the very first run class MOTO was never
    predicted once.

    The constants are taken as the mean and standard deviation of the formula
    output on TRAIN and then frozen. On a microcontroller this is 4 subtractions
    and 4 divisions — a negligible cost.
    """

    def __init__(self, trees: list[EMLNode], offsets=None, scales=None):
        self.trees = list(trees)
        n = len(self.trees)
        self.offsets = (np.zeros(n) if offsets is None
                        else np.asarray(offsets, dtype=np.float64))
        self.scales = (np.ones(n) if scales is None
                       else np.asarray(scales, dtype=np.float64))

    # ── Evaluation ────────────────────────────────────────────────────────
    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated formula outputs (N, n_classes)."""
        out = np.empty((len(X), len(self.trees)), dtype=np.float64)
        for c, tree in enumerate(self.trees):
            vals = tree.evaluate_batch(X)
            # A non-finite value = the formula broke on this example; such a
            # class simply must not win the argmax.
            out[:, c] = np.where(np.isfinite(vals), vals, -1e18)
        return out

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Calibrated scores (N, n_classes) — the argmax is taken over these."""
        return (self.raw_scores(X) - self.offsets) / self.scales

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(X), axis=1)

    def predict_one(self, inputs) -> int:
        """Inference for a single vector — the reference for the C export check."""
        vals = np.array([t.evaluate(inputs) for t in self.trees],
                        dtype=np.float64)
        return int(np.argmax((vals - self.offsets) / self.scales))

    # ── Calibration ───────────────────────────────────────────────────────
    def fit_calibration(self, X: np.ndarray) -> "EMLClassifier":
        """
        Fit mu_c and sd_c on sample X (train only!) and freeze them.

        A degenerate formula (a constant, sd ~ 0) receives sd = 1: division by
        zero is not allowed, and its contribution to the argmax is constant
        anyway.
        """
        raw = self.raw_scores(X)
        self.offsets, self.scales = _calibration_from_raw(raw)
        return self

    def calibration(self) -> dict:
        return {'offsets': self.offsets.tolist(), 'scales': self.scales.tolist()}

    # ── Complexity metrics (for the report table) ─────────────────────────
    def total_nodes(self) -> int:
        return sum(t.size() for t in self.trees)

    def max_depth(self) -> int:
        return max(t.depth() for t in self.trees)

    def used_inputs(self) -> list[int]:
        used = set()
        for t in self.trees:
            used.update(t.used_vars())
        return sorted(used)


# ── Metrics ───────────────────────────────────────────────────────────────────

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int | None = None) -> float:
    """
    Recall averaged over classes.

    At a 3.4:1 imbalance plain accuracy rewards the degenerate solution
    "everything is ROADS" (0.41 for free). The balanced one does not, which is
    why evolution is driven by it.
    """
    n_classes = n_classes or config.N_CLASSES
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.any():
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _calibration_from_raw(raw: np.ndarray):
    """
    Offsets and scales from the matrix of raw formula outputs (N, n_classes).

    Computed once on train and never changed — calibrating on val or test would
    mean looking into the held-out sample.
    """
    offsets = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)   # degenerate formula
    return offsets, scales


def _standardize_rows(logits: np.ndarray) -> np.ndarray:
    """
    z-score of the logits within each example (across classes).

    Preserves the argmax — i.e. all the information about the oracle's decision
    — and removes the arbitrary scale that would otherwise make MSE
    incomparable between examples.
    """
    mean = logits.mean(axis=1, keepdims=True)
    std = logits.std(axis=1, keepdims=True)
    return (logits - mean) / (std + 1e-8)


# ── DATA phase: regression of one formula to one class logit ──────────────────

def _regression_fitness(tree: EMLNode, X: np.ndarray, y: np.ndarray,
                        w: np.ndarray, depth_penalty: float) -> float:
    """
    Fitness = var_bonus / (weighted MSE + complexity penalty).

    The penalty accounts for both depth and NODE COUNT. Depth alone is not
    enough: it is bounded above by EML_MAX_DEPTH, so within the limit the tree
    can grow in width for free (see the comment on EML_SIZE_PENALTY_WEIGHT in
    config.py — it records the measurement this term came from).
    """
    n_vars = tree.n_unique_vars()
    if n_vars == 0:
        return 1e-6                       # a constant tree is useless
    try:
        pred = tree.evaluate_batch(X)
    except (OverflowError, ValueError, FloatingPointError):
        return 1e-6
    if not np.all(np.isfinite(pred)):
        return 1e-6

    mse = float(np.average((pred - y) ** 2, weights=w))
    var_bonus = 1.0 + config.EML_VAR_BONUS * n_vars
    complexity = tree.depth() + config.EML_SIZE_PENALTY_WEIGHT * tree.size()
    return var_bonus / (mse + depth_penalty * complexity + 1e-8)


def evolve_formula(X, y, w, *, generations: int, population_size: int,
                   elitism: int, patience: int, depth_penalty: float,
                   subsample: int | None = None, verbose: bool = False,
                   label: str = 'CLASS', rng_seed: int = 0,
                   progress_cb=None, should_stop=None) -> dict:
    """
    GA evolution of ONE formula for symbolic regression to the target y.

    Args:
        progress_cb: callable(dict) — invoked every generation. The GUI uses it
            to draw the fitness curve as it goes rather than half an hour later.
        should_stop: callable() -> bool — polled every generation. Interrupting
            returns the BEST tree found rather than discarding the work: stopping
            must stop, not cancel.

    Returns: {'best_tree', 'best_fitness', 'history', 'stopped'}.
    """
    if subsample is None:
        subsample = config.EML_DATA_SUBSAMPLE
    elitism = max(1, min(elitism, population_size))

    rng = np.random.default_rng(rng_seed)
    population = [random_tree(max_depth=3, force_eml=True)
                  for _ in range(population_size)]
    history: list[float] = []
    best_fitness_ever, best_tree_ever = -1.0, None
    stall = 0
    stopped = False

    for gen in range(generations):
        intensity = max(0.1, 1.0 - gen / (generations * 0.7))

        # A fresh subsample every generation: faster and less overfitting to
        # particular examples.
        if len(X) > subsample:
            idx = rng.choice(len(X), subsample, replace=False)
            Xs, ys, ws = X[idx], y[idx], w[idx]
        else:
            Xs, ys, ws = X, y, w

        scores = [_regression_fitness(t, Xs, ys, ws, depth_penalty)
                  for t in population]
        ranked = sorted(zip(scores, population), key=lambda p: p[0], reverse=True)
        scores = [s for s, _ in ranked]
        population = [t for _, t in ranked]

        best = scores[0]
        history.append(best)
        if best > best_fitness_ever:
            best_fitness_ever, best_tree_ever = best, population[0].clone()
            stall = 0
        else:
            stall += 1

        if verbose and (gen % 100 == 0 or gen == generations - 1):
            bt = population[0]
            print(f"      [{label}] gen {gen:>4} | fit {best:9.3f} | "
                  f"D{bt.depth()} S{bt.size()} V{bt.n_unique_vars()}")

        if progress_cb is not None:
            top = best_tree_ever or population[0]
            progress_cb({'phase': 'data', 'label': label, 'gen': gen,
                         'generations': generations, 'best': best,
                         'history': list(history), 'nodes': top.size(),
                         'depth': top.depth()})

        if should_stop is not None and should_stop():
            stopped = True
            if verbose:
                print(f"      [{label}] stopped by the user at gen {gen}")
            break

        if patience > 0 and stall >= patience:
            if verbose:
                print(f"      [{label}] early stop at gen {gen}")
            break

        new_pop = [population[i].clone() for i in range(elitism)]
        while len(new_pop) < population_size:
            r = random.random()
            if r < 0.45:
                child = mutate_tree(population[_tournament_select(scores)],
                                    intensity=intensity)
            elif r < 0.75:
                child = crossover(population[_tournament_select(scores)],
                                  population[_tournament_select(scores)])
                if random.random() < 0.3:
                    child = mutate_tree(child, intensity=intensity * 0.5)
            else:
                child = random_tree(max_depth=2 + int(intensity * 2),
                                    force_eml=True)
            new_pop.append(child)
        population = new_pop

    return {'best_tree': best_tree_ever or population[0],
            'best_fitness': best_fitness_ever,
            'history': history,
            'stopped': stopped}


# ── ACC phase: evolving the whole formula set by actual accuracy ──────────────

def _ensemble_fitness(trees: list[EMLNode], X: np.ndarray, y: np.ndarray,
                      depth_penalty: float) -> float:
    """
    Fitness of a set = balanced accuracy minus a size penalty.

    The calibration is fitted right here, on the same subsample: a formula set
    and its constants are one object, and evaluating trees with someone else's
    calibration is meaningless. It is computed from the already evaluated
    outputs, i.e. for free.

    The size penalty is mild and proportional to the node count: without it
    evolution inflates the trees for fractions of a percent of accuracy, whereas
    what is needed is a formula that fits into a microcontroller and can be read
    by a human.
    """
    clf = EMLClassifier(trees)
    raw = clf.raw_scores(X)
    offsets, scales = _calibration_from_raw(raw)
    pred = np.argmax((raw - offsets) / scales, axis=1)
    acc = balanced_accuracy(y, pred)
    return acc - depth_penalty * 0.01 * clf.total_nodes()


def evolve_ensemble(seed_trees: list[EMLNode], X: np.ndarray, y: np.ndarray, *,
                    generations: int | None = None,
                    population_size: int | None = None,
                    elitism: int | None = None,
                    patience: int | None = None,
                    subsample: int | None = None,
                    depth_penalty: float = 0.05,
                    verbose: bool = True, rng_seed: int = 0,
                    progress_cb=None, should_stop=None) -> dict:
    """
    Evolution of a SET of formulas with fitness = balanced accuracy.

    An individual = a list of N_CLASSES trees. Mutation changes one random tree
    of the set (changing all of them at once is too coarse a step and the set
    falls apart); crossover assembles a child from the formulas of two parents
    class by class.

    X, y — the train subsample ONLY (see the explanation in the module header).
    """
    generations = generations if generations is not None else config.EML_ACC_GENERATIONS
    population_size = population_size if population_size is not None else config.EML_ACC_POPULATION
    elitism = elitism if elitism is not None else config.EML_ACC_ELITISM
    patience = patience if patience is not None else config.EML_ACC_PATIENCE
    subsample = subsample if subsample is not None else config.EML_ACC_SUBSAMPLE
    elitism = max(1, min(elitism, population_size))

    rng = np.random.default_rng(rng_seed)
    population = [[t.clone() for t in seed_trees]]
    while len(population) < population_size:
        variant = [t.clone() for t in seed_trees]
        c = random.randrange(len(variant))
        variant[c] = mutate_tree(variant[c], intensity=0.7)
        population.append(variant)

    history: list[float] = []
    best_fitness_ever, best_trees_ever = -math.inf, None
    stall = 0
    stopped = False

    for gen in range(generations):
        intensity = max(0.15, 1.0 - gen / (generations * 0.8))

        # The same subsample for the whole generation — otherwise individuals
        # are compared on different data and selection becomes a lottery.
        if len(X) > subsample:
            idx = rng.choice(len(X), subsample, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y

        scores = [_ensemble_fitness(ind, Xs, ys, depth_penalty)
                  for ind in population]
        ranked = sorted(zip(scores, population), key=lambda p: p[0], reverse=True)
        scores = [s for s, _ in ranked]
        population = [ind for _, ind in ranked]

        best = scores[0]
        history.append(best)
        if best > best_fitness_ever:
            best_fitness_ever = best
            best_trees_ever = [t.clone() for t in population[0]]
            stall = 0
        else:
            stall += 1

        if verbose and (gen % 10 == 0 or gen == generations - 1):
            # Accuracy is computed through the same calibration as the fitness.
            # In the first version an EMLClassifier was created here WITHOUT
            # calibration, and the log showed 0.25 (a degenerate prediction)
            # while the fitness was growing — a direct illustration of why
            # calibration is needed.
            top = EMLClassifier(population[0])
            raw = top.raw_scores(Xs)
            off, sc = _calibration_from_raw(raw)
            acc = balanced_accuracy(ys, np.argmax((raw - off) / sc, axis=1))
            print(f"    [ACC] gen {gen:>3} | fit {best:.4f} | bal_acc {acc:.4f} | "
                  f"{top.total_nodes()} nodes")

        if progress_cb is not None:
            best_set = best_trees_ever or population[0]
            progress_cb({'phase': 'acc', 'gen': gen, 'generations': generations,
                         'best': best, 'history': list(history),
                         'nodes': sum(t.size() for t in best_set)})

        if should_stop is not None and should_stop():
            stopped = True
            if verbose:
                print(f"    [ACC] stopped by the user at gen {gen}")
            break

        if patience > 0 and stall >= patience:
            if verbose:
                print(f"    [ACC] early stop at gen {gen}")
            break

        new_pop = [[t.clone() for t in population[i]] for i in range(elitism)]
        while len(new_pop) < population_size:
            r = random.random()
            if r < 0.55:
                parent = population[_tournament_select(scores, k=4)]
                child = [t.clone() for t in parent]
                c = random.randrange(len(child))
                child[c] = mutate_tree(child[c], intensity=intensity)
            elif r < 0.85:
                p1 = population[_tournament_select(scores, k=4)]
                p2 = population[_tournament_select(scores, k=4)]
                child = [(p1 if random.random() < 0.5 else p2)[c].clone()
                         for c in range(len(p1))]
            else:
                child = [t.clone() for t in seed_trees]
                c = random.randrange(len(child))
                child[c] = mutate_tree(child[c], intensity=intensity)
            new_pop.append(child)
        population = new_pop

    return {'best_trees': best_trees_ever or population[0],
            'best_fitness': best_fitness_ever,
            'history': history,
            'stopped': stopped}


# ── Full distillation pipeline ────────────────────────────────────────────────

def distill(X_train: np.ndarray, logits_train: np.ndarray, y_train: np.ndarray,
            *, weights: np.ndarray | None = None,
            depth_penalty_name: str | None = None,
            data_generations: int | None = None,
            data_population: int | None = None,
            acc_phase: bool | None = None,
            verbose: bool = True, rng_seed: int = 0,
            progress_cb=None, should_stop=None) -> dict:
    """
    DATA phase (regression to the oracle logits) -> ACC phase (evolution by
    accuracy).

    Args:
        X_train:      (N, n_inputs) — features or embeddings.
        logits_train: (N, n_classes) — oracle logits on the same examples.
        y_train:      (N,) — ground truth labels (needed by the ACC phase only).
        weights:      (N,) example weights; None -> class weights from data.py.
        progress_cb:  callable(dict) — progress of both phases, for the GUI.
        should_stop:  callable() -> bool — soft stop. The result stays
            consistent: the formula set is complete, the calibration is
            computed, the metrics are honest. There was simply less evolution,
            and that is recorded in the 'stopped' flag — so that such a run is
            not passed off as a complete one.

    Returns: dict with 'trees' (the final set), metrics of both phases, and
    history.
    """
    depth_penalty_name = depth_penalty_name or config.DEPTH_PENALTY_DEFAULT
    data_generations = data_generations if data_generations is not None else config.EML_GENERATIONS
    data_population = data_population if data_population is not None else config.EML_POPULATION
    acc_phase = acc_phase if acc_phase is not None else config.EML_ACC_ENABLED

    random.seed(rng_seed)
    dp = config.DEPTH_PENALTIES[depth_penalty_name]
    X = np.ascontiguousarray(X_train, dtype=np.float64)
    n_classes = logits_train.shape[1]

    if weights is None:
        # Class weights: without them the rare BEACH drowns in the MSE and its
        # formula degenerates into a constant.
        import data as data_mod
        cw = data_mod.class_weights(y_train)
        weights = cw[y_train].astype(np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    targets = _standardize_rows(np.asarray(logits_train, dtype=np.float64))
    t0 = time.time()

    # ── DATA phase: one formula per class ────────────────────────────────────
    if verbose:
        print(f"  [DATA] symbolic regression to the oracle logits "
              f"({len(X):,} examples, {_N_INPUTS} inputs)")
    data_trees, data_results = [], []
    stopped = False
    for c in range(n_classes):
        name = config.CLASS_NAMES[c] if c < len(config.CLASS_NAMES) else f'C{c}'
        # The class index is mixed into the payload here rather than in
        # evolve_formula: that function knows nothing about classes, it evolves
        # a single formula.
        cb = None
        if progress_cb is not None:
            def cb(payload, _c=c, _name=name):
                progress_cb({**payload, 'class_idx': _c, 'class_name': _name,
                             'n_classes': n_classes})
        res = evolve_formula(
            X, targets[:, c], weights,
            generations=data_generations, population_size=data_population,
            elitism=config.EML_ELITISM, patience=config.EML_PATIENCE,
            depth_penalty=dp, verbose=verbose, label=name,
            rng_seed=rng_seed + c, progress_cb=cb, should_stop=should_stop,
        )
        tree = res['best_tree']
        data_trees.append(tree)
        data_results.append(res)
        stopped = stopped or res.get('stopped', False)
        if verbose:
            print(f"    {name:>7}: fit={res['best_fitness']:.3f}  "
                  f"D{tree.depth()} S{tree.size()} V{tree.n_unique_vars()}")

    def _fit_and_score(trees: list[EMLNode]) -> tuple[EMLClassifier, float]:
        """Freeze the calibration on full train and measure the accuracy."""
        clf = EMLClassifier(trees).fit_calibration(X)
        return clf, balanced_accuracy(y_train, clf.predict(X))

    data_clf, data_acc = _fit_and_score(data_trees)
    if verbose:
        print(f"  DATA phase: balanced accuracy on train = {data_acc:.4f}")

    # ── ACC phase ────────────────────────────────────────────────────────────
    acc_result = None
    final_trees, final_clf, final_acc = data_trees, data_clf, data_acc
    if acc_phase and not stopped:
        if verbose:
            print("  [ACC] evolving the formula set by actual accuracy...")
        acc_result = evolve_ensemble(data_trees, X, y_train,
                                     depth_penalty=dp, verbose=verbose,
                                     rng_seed=rng_seed, progress_cb=progress_cb,
                                     should_stop=should_stop)
        stopped = stopped or acc_result.get('stopped', False)
        candidate_trees = acc_result['best_trees']
        candidate_clf, candidate_acc = _fit_and_score(candidate_trees)
        # The ACC phase was evaluated on subsamples; on full train the result may
        # turn out worse than the original. In that case roll back — making
        # things worse is not allowed.
        if candidate_acc >= data_acc:
            final_trees, final_clf, final_acc = (candidate_trees, candidate_clf,
                                                 candidate_acc)
        elif verbose:
            print(f"  ACC ({candidate_acc:.4f}) < DATA ({data_acc:.4f}) "
                  f"-> keeping the DATA phase set")

    clf = final_clf
    return {
        'trees': final_trees,
        'classifier': clf,
        'calibration': clf.calibration(),
        'data_trees': data_trees,
        'data_results': data_results,
        'acc_result': acc_result,
        'train_bal_acc_data': data_acc,
        'train_bal_acc_final': final_acc,
        'total_nodes': clf.total_nodes(),
        'max_depth': clf.max_depth(),
        'used_inputs': clf.used_inputs(),
        'depth_penalty': depth_penalty_name,
        'n_inputs': _N_INPUTS,
        'dataset_size': len(X),
        'stopped': stopped,
        'seconds': time.time() - t0,
    }
