"""
eml_single.py — branch C: ONE EML formula for all classes.

Branches A and B keep one formula per class and take the argmax. The main
technical finding of the sub-project was that the method stumbles not over the
expressiveness of the formulas but over the COMPARABILITY OF THEIR SCALES: the
formulas evolve independently, each has its own scale, and without calibration
(mu_c, sd_c) the argmax is won not by the most appropriate formula but by the
one emitting larger numbers — in the first branch A run class MOTO was never
predicted once.

Branch C removes this defect by construction. There is one formula:

    f(features) -> v          class = labels[searchsorted(thresholds, v)]

There is nothing to compare, no calibration constants, and the device holds one
tree and a table of a few thresholds.

THE TABLE IS NOT FIXED IN ADVANCE — AND THAT IS THE KEY POINT. The obvious
variant "let class 0 be 0.0, class 1 be 0.33, class 2 be 0.6, class 3 be 1.0" is
bad twice over. First, it imposes on the classes an order and distances they do
not have: BEACH, MOTO, PEOPLE and ROADS are unordered, and regression to such a
code penalizes "distant" errors more than "near" ones without any grounds.
Second, a small MSE against the code and a correct class are different things
(exactly the argument that produced the ACC phase in branch A).

The table is therefore FITTED TO THE FORMULA at every fitness evaluation, by
exact dynamic programming maximizing balanced accuracy. The formula itself
decides in what order and at what distance to place the classes; where it
eventually put them is an output of the run, not an assumption of ours (see
IntervalDecoder.describe).

More intervals than classes are allowed, and INTERVAL LABELS MAY REPEAT. This is
the same effect a periodic activation function would target: a class occupies
several zones on the axis, and the linear ordering of classes stops being a
constraint. But without the periodicity itself: the EML tree output lies roughly
in [-10, +2.2e4] (exp(clamp(x, ±10)) plus a logarithm), a sine of such an
argument spans thousands of periods, the fitness landscape becomes discontinuous
— a small edit to the tree throws half the examples into a different class, and
a GA does not climb such a landscape but wanders over it. The threshold table
gives the same expressiveness, an exact fit, and K comparisons on the
microcontroller.

WHAT IS BEING LEARNED. The fitness target is the ORACLE'S ANSWERS, not the
ground truth labels: this is still distillation, as in stages 1 and 2, and the
formula reproduces the network's decision function specifically. Ground truth
labels are used only for the reported metrics.
"""

import json
import math
import os
import random
import time

import numpy as np

import config
from eml_distiller import (EMLNode, balanced_accuracy, crossover, mutate_tree,
                           random_tree)


# ── Decoder: threshold table over the formula output axis ─────────────────────

class IntervalDecoder:
    """
    Partition of the formula output axis into intervals with class labels.

    thresholds — an increasing array of length K-1, labels — K labels. Labels
    may repeat: a class is allowed to occupy several non-adjacent intervals.
    """

    __slots__ = ('thresholds', 'labels', 'n_classes', 'score')

    def __init__(self, thresholds, labels, n_classes: int,
                 score: float = float('nan')):
        self.thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
        self.labels = np.asarray(labels, dtype=np.int64).ravel()
        self.n_classes = int(n_classes)
        self.score = float(score)
        if len(self.labels) != len(self.thresholds) + 1:
            raise ValueError(f"{len(self.labels)} labels, "
                             f"{len(self.thresholds)} thresholds: there must be "
                             f"one more label")

    @property
    def n_intervals(self) -> int:
        return len(self.labels)

    def decode(self, values: np.ndarray) -> np.ndarray:
        """
        Vector of formula outputs -> vector of classes.

        Infinities pass through as ordinary very large numbers — they have a
        legitimate place on the axis. NaN has no place on the axis (the formula
        broke on that example) and is sent to the leftmost interval: the
        prediction must stay defined, otherwise the metric would be computed
        over a corrupted array.
        """
        values = np.asarray(values, dtype=np.float64)
        safe = np.where(np.isnan(values), -np.inf, values)
        return self.labels[np.searchsorted(self.thresholds, safe, side='right')]

    def decode_one(self, value: float) -> int:
        """Inference for a single number — the reference for the C export check."""
        if math.isnan(value):
            return int(self.labels[0])
        idx = 0
        while idx < len(self.thresholds) and value > self.thresholds[idx]:
            idx += 1
        return int(self.labels[idx])

    def class_intervals(self, class_idx: int) -> list[tuple[float, float]]:
        """Axis intervals given to a class, as a list of (left, right) pairs."""
        edges = np.concatenate(([-np.inf], self.thresholds, [np.inf]))
        return [(float(edges[i]), float(edges[i + 1]))
                for i in range(self.n_intervals) if self.labels[i] == class_idx]

    def describe(self, class_names: list[str] | None = None) -> str:
        """
        Where the formula placed the classes on its axis.

        This is a result rather than debug output: the formula chose the order
        and the distances between classes itself, and they are worth checking
        against the oracle confusion matrix — classes adjacent on the axis
        should be confused more often than others.
        """
        names = class_names or [f'C{i}' for i in range(self.n_classes)]
        edges = np.concatenate(([-np.inf], self.thresholds, [np.inf]))
        lines = []
        for i, label in enumerate(self.labels):
            lo, hi = edges[i], edges[i + 1]
            left = '-inf' if lo == -np.inf else f"{lo:+.4g}"
            right = '+inf' if hi == np.inf else f"{hi:+.4g}"
            name = names[label] if label < len(names) else f'C{label}'
            lines.append(f"    {left:>12} .. {right:<12}  ->  {name}")
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return {'thresholds': self.thresholds.tolist(),
                'labels': self.labels.tolist(),
                'n_classes': self.n_classes,
                'train_score': self.score}

    @classmethod
    def from_dict(cls, d: dict) -> "IntervalDecoder":
        return cls(d['thresholds'], d['labels'], d['n_classes'],
                   d.get('train_score', float('nan')))


def sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Example weights under which the sum of weights of correct answers equals
    balanced accuracy.

    Balanced accuracy is recall averaged over classes. Giving an example of
    class c the weight 1/(number of classes present * size of class c) makes the
    sum of weights of correctly predicted examples equal exactly that. The whole
    decoder optimization then reduces to "collect as much weight as possible",
    and the dynamic programming needs to know nothing else about classes.

    Classes absent from the sample drop out of the averaging — the same way as
    in eml_distiller.balanced_accuracy.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    present = max(int((counts > 0).sum()), 1)
    counts = np.maximum(counts, 1.0)
    return (1.0 / (present * counts))[y]


def fit_decoder(values: np.ndarray, y: np.ndarray, n_classes: int, *,
                weights: np.ndarray | None = None,
                max_intervals: int | None = None,
                bins: int | None = None,
                interval_penalty: float | None = None) -> IntervalDecoder:
    """
    Fit the threshold table to already computed formula outputs.

    The problem is solved EXACTLY rather than approximately: partitioning the
    axis into K labeled intervals is classic DP over the sorted values. An
    approximate threshold fit (greedy, by class means) would be dangerous here:
    the fitness of the whole evolution equals the quality of this fit, and its
    random failures would be read by the GA as "the tree got worse".

    The axis is cut by QUANTILES (bins of them) rather than by value: the EML
    tree output is distributed extremely unevenly (exp() at every node), and a
    uniform grid would give almost all thresholds to the empty tail. Bin
    boundaries are moved past the end of a run of equal values — a threshold in
    the middle of such a run cannot be realized anyway.

    The number of intervals is chosen together with the partition itself: each
    interval beyond the class count costs interval_penalty. Otherwise K would
    always hit the ceiling — an extra interval never worsens the training metric.
    """
    max_intervals = max_intervals or config.EMLC_MAX_INTERVALS
    bins = bins or config.EMLC_DECODER_BINS
    if interval_penalty is None:
        interval_penalty = config.EMLC_INTERVAL_PENALTY

    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if weights is None:
        weights = sample_weights(y, n_classes)

    # NaN has no place on the axis — it goes to the leftmost interval, as in
    # decode(); infinities sort themselves and stay at their edges.
    clean = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(clean, kind='stable')
    sv, sy, sw = clean[order], y[order], np.asarray(weights)[order]
    n = len(sv)

    # ── Quantile bins, boundaries only between distinct values ───────────────
    n_bins = max(1, min(bins, n))
    ranks = np.linspace(0, n, n_bins + 1).round().astype(np.int64)[1:-1]
    if len(ranks):
        ranks = np.searchsorted(sv, sv[np.clip(ranks, 0, n - 1)], side='right')
    bounds = np.unique(np.concatenate(([0], ranks, [n])))
    n_bins = len(bounds) - 1

    # Weight of each class in each bin and its cumulative sum over bins.
    bin_of = np.repeat(np.arange(n_bins), np.diff(bounds))
    per_bin = np.bincount(bin_of * n_classes + sy, weights=sw,
                          minlength=n_bins * n_classes).reshape(n_bins, n_classes)
    prefix = np.vstack([np.zeros((1, n_classes)), np.cumsum(per_bin, axis=0)])

    # gain[a, b] — how much weight an interval of bins [a, b) collects if given
    # to its best class. Computed for all pairs at once: (nb+1, nb+1, C) -> max.
    gain = (prefix[None, :, :] - prefix[:, None, :]).max(axis=2)
    idx = np.arange(n_bins + 1)
    valid = idx[None, :] > idx[:, None]

    # ── DP: best coverage of the first b bins by exactly k intervals ─────────
    n_steps = max(1, min(max_intervals, n_bins))
    prev = np.full(n_bins + 1, -np.inf)
    prev[0] = 0.0
    tables, pointers = [], []
    for _ in range(n_steps):
        cand = np.where(valid, prev[:, None] + gain, -np.inf)
        start = cand.argmax(axis=0)
        prev = cand[start, idx]
        tables.append(prev)
        pointers.append(start)

    totals = np.array([t[n_bins] for t in tables])
    extra = np.maximum(np.arange(1, n_steps + 1) - n_classes, 0)
    k_best = int(np.argmax(totals - interval_penalty * extra)) + 1

    # ── Reconstructing the intervals ─────────────────────────────────────────
    segments, b = [], n_bins
    for k in range(k_best, 0, -1):
        a = int(pointers[k - 1][b])
        label = int((prefix[b] - prefix[a]).argmax())
        segments.append((a, b, label))
        b = a
    segments.reverse()

    # Adjacent intervals with the same label are merged: an extra threshold
    # decides nothing but costs both a penalty and a comparison on the MCU.
    merged = [segments[0]]
    for a, b, label in segments[1:]:
        if label == merged[-1][2]:
            merged[-1] = (merged[-1][0], b, label)
        else:
            merged.append((a, b, label))

    thresholds = []
    for a, _, _ in merged[1:]:
        rank = bounds[a]
        thresholds.append((sv[rank - 1] + sv[rank]) / 2.0)
    labels = [label for _, _, label in merged]
    seg_weights = [float((prefix[b] - prefix[a])[label]) for a, b, label in merged]

    thresholds, labels, changed = _make_float32_safe(thresholds, labels,
                                                     seg_weights)
    decoder = IntervalDecoder(thresholds, labels, n_classes,
                              score=float(totals[k_best - 1]))
    if changed:
        # Merging intervals changes the answers, so the training metric is
        # recomputed: returning the pre-merge score would overstate the fitness.
        decoder.score = float(sw[decoder.decode(sv) == sy].sum())
    return decoder


def _make_float32_safe(thresholds: list[float], labels: list[int],
                       weights: list[float]):
    """
    Remove thresholds indistinguishable in float32, merging the affected
    intervals.

    The EML tree output reaches 2.2e4 (exp(clamp(x, ±10))), where the float32
    step is already ~0.002 — two neighboring thresholds in that region easily
    collapse into one number. On a microcontroller such an interval would simply
    disappear and its class would stop being predicted; the Python/C divergence
    would then look like a broken export even though the export is correct.

    It is cheaper not to allow indistinguishable thresholds at all: the table
    then behaves identically in float64 and float32 by construction. On a merge
    the label of the "heavier" interval survives — there is no sense in losing a
    larger part of the sample for a smaller one.
    """
    if not thresholds:
        return thresholds, labels, False

    out_t: list[float] = []
    out_l: list[int] = [labels[0]]
    out_w: list[float] = [weights[0]]
    changed = False
    for threshold, label, weight in zip(thresholds, labels[1:], weights[1:]):
        collides = bool(out_t) and np.float32(threshold) <= np.float32(out_t[-1])
        # A merge may have left two intervals of the same class side by side —
        # the threshold between them decides nothing but costs a comparison on
        # the device.
        if collides or label == out_l[-1]:
            if weight > out_w[-1]:
                out_l[-1] = label
            out_w[-1] += weight
            changed = True
            continue
        out_t.append(threshold)
        out_l.append(label)
        out_w.append(weight)
    return out_t, out_l, changed


# ── Branch C model ────────────────────────────────────────────────────────────

class EMLSingle:
    """
    The end product of branch C: one EML tree plus a threshold table.

    The interface deliberately matches EMLClassifier of branch A (predict,
    predict_one, total_nodes, max_depth, used_inputs), so that the pipeline, the
    reports and the test tabs work with both branches without branching.
    """

    def __init__(self, tree: EMLNode, decoder: IntervalDecoder):
        self.tree = tree
        self.decoder = decoder

    # ── Evaluation ────────────────────────────────────────────────────────
    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.tree.evaluate_batch(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decoder.decode(self.raw(X))

    def predict_one(self, inputs) -> int:
        return self.decoder.decode_one(self.tree.evaluate(inputs))

    def class_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Score of every class (N, n_classes) — for the test tabs.

        A single formula has no separate number per class, so the score is minus
        the distance from the output to the nearest interval of that class: 0
        for the predicted class, negative for the rest, and smaller the further
        along the axis. This is NOT a probability or a logit; the safest reading
        is "how far the output would have to move to yield this class".
        """
        values = self.raw(X)
        values = np.where(np.isfinite(values), values, 0.0)
        out = np.full((len(values), self.decoder.n_classes), -np.inf)
        for c in range(self.decoder.n_classes):
            for lo, hi in self.decoder.class_intervals(c):
                dist = np.maximum(np.maximum(lo - values, values - hi), 0.0)
                out[:, c] = np.maximum(out[:, c], -dist)
        return out

    # ── Complexity metrics ────────────────────────────────────────────────
    def total_nodes(self) -> int:
        return self.tree.size()

    def max_depth(self) -> int:
        return self.tree.depth()

    def used_inputs(self) -> list[int]:
        return self.tree.used_vars()

    # ── Serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {'tree': self.tree.to_dict(), 'decoder': self.decoder.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "EMLSingle":
        return cls(EMLNode.from_dict(d['tree']),
                   IntervalDecoder.from_dict(d['decoder']))


# ── Fitness ───────────────────────────────────────────────────────────────────

class FitnessParams:
    """Everything one formula evaluation needs, in one object (easy to ship to
    worker processes)."""

    __slots__ = ('n_classes', 'max_intervals', 'bins', 'interval_penalty',
                 'size_penalty', 'max_depth')

    def __init__(self, n_classes: int, *, max_intervals: int | None = None,
                 bins: int | None = None, interval_penalty: float | None = None,
                 size_penalty: float | None = None, max_depth: int | None = None):
        self.n_classes = int(n_classes)
        self.max_intervals = max_intervals or config.EMLC_MAX_INTERVALS
        self.bins = bins or config.EMLC_DECODER_BINS
        self.interval_penalty = (config.EMLC_INTERVAL_PENALTY
                                 if interval_penalty is None else interval_penalty)
        self.size_penalty = (
            config.EMLC_SIZE_PENALTIES[config.EMLC_SIZE_PENALTY_DEFAULT]
            if size_penalty is None else size_penalty)
        self.max_depth = max_depth or config.EMLC_MAX_DEPTH

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "FitnessParams":
        return cls(d['n_classes'], max_intervals=d['max_intervals'],
                   bins=d['bins'], interval_penalty=d['interval_penalty'],
                   size_penalty=d['size_penalty'], max_depth=d['max_depth'])


def evaluate_tree(tree: EMLNode, X: np.ndarray, y: np.ndarray,
                  weights: np.ndarray, params: FitnessParams) -> dict:
    """
    Evaluate one tree: fit a decoder to it and compute the fitness.

    fitness = balanced accuracy − node penalty − interval penalty.
    Both penalties are subtracted rather than dividing: the fitness scale then
    matches the metric scale, and "-0.05 per hundred nodes" reads directly off
    the constant.
    """
    if tree.n_unique_vars() == 0:
        # A constant tree yields exactly 1/n_classes (the decoder is forced to
        # give the whole axis to one class) — formally not zero, and without an
        # explicit penalty such trees fill the population as cheap options.
        return {'fitness': -1.0, 'score': 0.0, 'intervals': 1, 'nodes': tree.size()}
    try:
        values = tree.evaluate_batch(X)
    except (OverflowError, ValueError, FloatingPointError):
        return {'fitness': -1.0, 'score': 0.0, 'intervals': 1, 'nodes': tree.size()}

    decoder = fit_decoder(values, y, params.n_classes, weights=weights,
                          max_intervals=params.max_intervals, bins=params.bins,
                          interval_penalty=params.interval_penalty)
    nodes = tree.size()
    extra = max(0, decoder.n_intervals - params.n_classes)
    fitness = (decoder.score
               - params.size_penalty * nodes
               - params.interval_penalty * extra)
    return {'fitness': float(fitness), 'score': float(decoder.score),
            'intervals': decoder.n_intervals, 'nodes': nodes,
            'decoder': decoder}


def _tournament(scores: list[float], k: int = 5) -> int:
    candidates = random.sample(range(len(scores)), min(k, len(scores)))
    return max(candidates, key=lambda i: scores[i])


def _breed(population: list[EMLNode], scores: list[float], size: int,
           elitism: int, intensity: float, max_depth: int) -> list[EMLNode]:
    """New generation: elite unchanged, the rest by mutation/crossover/newcomer."""
    new_pop = [population[i].clone() for i in range(min(elitism, len(population)))]
    while len(new_pop) < size:
        r = random.random()
        if r < 0.45:
            child = mutate_tree(population[_tournament(scores)],
                                intensity=intensity, max_depth=max_depth)
        elif r < 0.75:
            child = crossover(population[_tournament(scores)],
                              population[_tournament(scores)],
                              max_depth=max_depth)
            if random.random() < 0.3:
                child = mutate_tree(child, intensity=intensity * 0.5,
                                    max_depth=max_depth)
        else:
            child = random_tree(max_depth=2 + int(intensity * 3), force_eml=True)
        new_pop.append(child)
    return new_pop


def run_island(trees: list[EMLNode], X: np.ndarray, y: np.ndarray,
               params: FitnessParams, *, generations: int, elitism: int,
               subsample: int, rng_seed: int, progress: float = 0.0) -> dict:
    """
    Run one population for `generations` generations.

    progress — the fraction of the whole run already completed (0..1); the
    mutation intensity depends on it: wild at first, fine toward the end.

    The subsample changes every generation (faster and less fitting to
    particular examples), but WITHIN a generation it is shared: otherwise
    individuals would be compared on different data and selection would turn
    into a lottery.
    """
    rng = np.random.default_rng(rng_seed)
    random.seed(rng_seed)
    population = [t.clone() for t in trees]
    best_fitness, best_tree, best_info = -math.inf, None, None
    history = []

    # Mutation intensity decays over the WHOLE run rather than within a round: a
    # round is tens of generations out of thousands and has no "wild to fine"
    # scale of its own. By the last third of the run mutations become pointwise.
    intensity = max(0.12, 1.0 - 1.2 * min(1.0, progress))

    for gen in range(generations):
        if len(X) > subsample:
            idx = rng.choice(len(X), subsample, replace=False)
            Xs, ys = X[idx], y[idx]
        else:
            Xs, ys = X, y
        ws = sample_weights(ys, params.n_classes)

        infos = [evaluate_tree(t, Xs, ys, ws, params) for t in population]
        scores = [i['fitness'] for i in infos]
        ranked = sorted(range(len(population)), key=lambda i: scores[i],
                        reverse=True)
        population = [population[i] for i in ranked]
        infos = [infos[i] for i in ranked]
        scores = [scores[i] for i in ranked]

        history.append(scores[0])
        if scores[0] > best_fitness:
            best_fitness = scores[0]
            best_tree = population[0].clone()
            best_info = {k: v for k, v in infos[0].items() if k != 'decoder'}

        if gen < generations - 1:
            population = _breed(population, scores, len(population), elitism,
                                intensity, params.max_depth)

    return {'population': population, 'best_tree': best_tree or population[0],
            'best_fitness': best_fitness, 'best_info': best_info or {},
            'history': history}


# ── Process-level parallelism ─────────────────────────────────────────────────
#
# A WHOLE ISLAND is parallelized for several generations, not individual fitness
# evaluations. Evaluating one tree takes fractions of a millisecond, and
# distributing that across processes would cost more in inter-process transfer
# than in computation. An island, by contrast, goes into a process once for tens
# of generations.

_WORKER: dict = {}


def _worker_init(X, y, n_inputs, input_names):
    """Data is placed into the process once at creation, not on every call."""
    import eml_distiller
    eml_distiller.configure_inputs(n_inputs, input_names)
    _WORKER['X'] = X
    _WORKER['y'] = y


def _worker_round(payload: dict) -> dict:
    """One round of one island inside a worker process."""
    params = FitnessParams.from_dict(payload['params'])
    trees = [EMLNode.from_dict(d) for d in payload['trees']]
    out = run_island(trees, _WORKER['X'], _WORKER['y'], params,
                     generations=payload['generations'],
                     elitism=payload['elitism'],
                     subsample=payload['subsample'],
                     rng_seed=payload['rng_seed'],
                     progress=payload['progress'])
    return {'island': payload['island'],
            'population': [t.to_dict() for t in out['population']],
            'best_tree': out['best_tree'].to_dict(),
            'best_fitness': out['best_fitness'],
            'best_info': out['best_info'],
            'history': out['history']}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, state: dict):
    """
    Write the run state atomically (via .tmp + replace).

    This is the only way a multi-hour run survives a reboot: writing directly
    into the target file would mean that a power loss mid-write leaves truncated
    JSON, destroying exactly what it was supposed to save.
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ── Full evolution ────────────────────────────────────────────────────────────

def evolve_single(X: np.ndarray, y: np.ndarray, *, n_classes: int | None = None,
                  generations: int | None = None,
                  population: int | None = None,
                  islands: int | None = None,
                  workers: int | None = None,
                  elitism: int | None = None,
                  patience: int | None = None,
                  subsample: int | None = None,
                  size_penalty: float | None = None,
                  max_intervals: int | None = None,
                  max_depth: int | None = None,
                  migrate_every: int | None = None,
                  migrants: int | None = None,
                  checkpoint_path: str | None = None,
                  checkpoint_every: int | None = None,
                  resume: bool = True,
                  input_names: list[str] | None = None,
                  rng_seed: int = 0, verbose: bool = True,
                  progress_cb=None, should_stop=None) -> dict:
    """
    Island evolution of a single formula.

    Islands are several independent populations that occasionally exchange their
    best individuals. The point is not speed (that comes from process-level
    parallelism) but DIVERSITY: the GA spread across seeds in branch A was
    0.525-0.623, i.e. one long evolution chain says next to nothing about the
    method. Several chains with rare migration provide both a reserve of
    diversity and the ability to compare whether they converge to one solution.

    THE GLOBAL BEST IS NOT CHOSEN BY ISLAND FITNESS. Within an island fitness is
    computed on its own random subsample, and numbers from different islands are
    not comparable — choosing by them would mean choosing by subsample noise.
    Candidates from all islands are therefore re-evaluated in the main process on
    the FULL training sample, identical for all of them.

    A run survives being stopped: the state of all islands is written to
    checkpoint_path, and resume=True continues from the last save.

    Returns: {'model' (EMLSingle), 'best_fitness', 'train_score', 'history',
              'generation', 'stopped', 'islands', 'seconds'}.
    """
    n_classes = n_classes if n_classes is not None else config.N_CLASSES
    generations = generations if generations is not None else config.EMLC_GENERATIONS
    population = population if population is not None else config.EMLC_POPULATION
    elitism = elitism if elitism is not None else config.EMLC_ELITISM
    patience = patience if patience is not None else config.EMLC_PATIENCE
    subsample = subsample if subsample is not None else config.EMLC_SUBSAMPLE
    migrate_every = (migrate_every if migrate_every is not None
                     else config.EMLC_MIGRATE_EVERY)
    migrants = migrants if migrants is not None else config.EMLC_MIGRANTS
    checkpoint_every = (checkpoint_every if checkpoint_every is not None
                        else config.EMLC_CHECKPOINT_EVERY)
    workers = workers if workers is not None else config.EMLC_WORKERS
    workers = max(1, workers or (os.cpu_count() or 1))
    islands = islands if islands is not None else config.EMLC_ISLANDS
    islands = max(1, islands or workers)

    params = FitnessParams(n_classes, max_intervals=max_intervals,
                           size_penalty=size_penalty, max_depth=max_depth)
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    full_weights = sample_weights(y, n_classes)
    t0 = time.time()

    # Only one's OWN run may be resumed. A checkpoint with a different number of
    # inputs, classes or individuals belongs to another experiment; picking it up
    # silently would produce a result matching none of the declared settings.
    signature = {'n_inputs': int(X.shape[1]), 'n_classes': n_classes,
                 'population': population, 'n_train': int(len(X)),
                 'max_depth': params.max_depth, 'seed': rng_seed}

    # ── Start or resume ──────────────────────────────────────────────────────
    state = load_checkpoint(checkpoint_path) if (resume and checkpoint_path) else None
    if state and state.get('signature') != signature:
        if verbose:
            print(f"  [C] checkpoint {checkpoint_path} belongs to another run "
                  f"({state.get('signature')}) — starting over")
        state = None
    if state:
        pops = [[EMLNode.from_dict(d) for d in island] for island in state['islands']]
        start_gen = int(state['generation'])
        history = list(state['history'])
        best_fitness = float(state['best_fitness'])
        best_tree = EMLNode.from_dict(state['best_tree'])
        stall = int(state.get('stall', 0))
        if verbose:
            print(f"  [C] resuming from generation {start_gen} "
                  f"({len(pops)} islands, best {best_fitness:.4f})")
    else:
        random.seed(rng_seed)
        pops = [[random_tree(max_depth=3, force_eml=True)
                 for _ in range(population)] for _ in range(islands)]
        start_gen, history, stall = 0, [], 0
        best_fitness, best_tree = -math.inf, pops[0][0]

    islands = len(pops)
    pool = None
    if workers > 1 and islands > 1:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        names = input_names or [f'x{i}' for i in range(X.shape[1])]
        pool = ctx.Pool(min(workers, islands), initializer=_worker_init,
                        initargs=(X, y, X.shape[1], names))

    if verbose:
        how = (f"in parallel on {min(workers, islands)} processes" if pool
               else "in a single process")
        print(f"  [C] {islands} islands x {population} individuals, {how}")

    stopped = False
    gen = start_gen
    try:
        while gen < generations:
            rounds = min(migrate_every, generations - gen)
            progress = gen / max(generations, 1)
            payloads = [{
                'island': i, 'trees': [t.to_dict() for t in pops[i]],
                'params': params.to_dict(), 'generations': rounds,
                'elitism': elitism, 'subsample': subsample,
                'rng_seed': rng_seed + 1000 * i + gen, 'progress': progress,
            } for i in range(islands)]

            if pool is not None:
                results = pool.map(_worker_round, payloads)
                for res in results:
                    pops[res['island']] = [EMLNode.from_dict(d)
                                           for d in res['population']]
                candidates = [EMLNode.from_dict(r['best_tree']) for r in results]
                round_history = [r['history'] for r in results]
            else:
                results = []
                for i in range(islands):
                    out = run_island(pops[i], X, y, params, generations=rounds,
                                     elitism=elitism, subsample=subsample,
                                     rng_seed=payloads[i]['rng_seed'],
                                     progress=progress)
                    pops[i] = out['population']
                    results.append(out)
                candidates = [r['best_tree'] for r in results]
                round_history = [r['history'] for r in results]

            gen += rounds

            # Re-evaluating the candidates on the FULL sample is the only common
            # yardstick. It costs a few tree evaluations per round.
            scored = [(evaluate_tree(t, X, y, full_weights, params), t)
                      for t in candidates]
            best_info, top_tree = max(scored, key=lambda p: p[0]['fitness'])
            history.append(float(best_info['fitness']))

            if best_info['fitness'] > best_fitness:
                best_fitness = float(best_info['fitness'])
                best_tree = top_tree.clone()
                stall = 0
            else:
                stall += rounds

            if verbose:
                spread = max(r[0]['fitness'] for r in scored) - min(
                    r[0]['fitness'] for r in scored)
                print(f"    [C] gen {gen:>5}/{generations} | best "
                      f"{best_fitness:.4f} | round {best_info['fitness']:.4f} | "
                      f"island spread {spread:.4f} | nodes "
                      f"{best_info['nodes']} | intervals {best_info['intervals']}")

            if progress_cb is not None:
                progress_cb({'phase': 'single', 'gen': gen,
                             'generations': generations,
                             'best': best_fitness, 'history': list(history),
                             'nodes': best_tree.size(),
                             'depth': best_tree.depth(),
                             'intervals': best_info['intervals'],
                             'islands': [max(h) if h else 0.0
                                         for h in round_history]})

            if checkpoint_path and (gen % max(checkpoint_every, 1) < rounds):
                save_checkpoint(checkpoint_path, {
                    'signature': signature,
                    'generation': gen, 'history': history,
                    'best_fitness': best_fitness,
                    'best_tree': best_tree.to_dict(),
                    'stall': stall,
                    'islands': [[t.to_dict() for t in p] for p in pops],
                })

            if should_stop is not None and should_stop():
                stopped = True
                if verbose:
                    print(f"    [C] stopped by the user at generation {gen}")
                break

            if patience > 0 and stall >= patience:
                if verbose:
                    print(f"    [C] early stop: {stall} generations "
                          f"without improvement")
                break

            if islands > 1 and migrants > 0:
                _migrate(pops, migrants)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # The final decoder is frozen on the FULL training sample, the same way as
    # the branch A calibration. Fitting it on val would mean looking ahead.
    # The threshold grid here is finer than during fitness: this is computed
    # once, and a coarse grid would cost a percent of accuracy for nothing.
    decoder = fit_decoder(best_tree.evaluate_batch(X), y, n_classes,
                          weights=full_weights,
                          max_intervals=params.max_intervals,
                          bins=config.EMLC_FINAL_BINS,
                          interval_penalty=params.interval_penalty)
    model = EMLSingle(best_tree, decoder)
    train_score = balanced_accuracy(y, model.predict(X), n_classes)

    return {'model': model, 'best_fitness': best_fitness,
            'train_score': float(train_score), 'history': history,
            'generation': gen, 'stopped': stopped, 'islands': islands,
            'workers': min(workers, islands) if islands > 1 else 1,
            'seconds': time.time() - t0}


def _migrate(pops: list[list[EMLNode]], migrants: int):
    """
    The best individuals of each island replace the worst on the next one
    (in a ring).

    Copies are sent rather than transferred: an island must not lose its elite,
    otherwise migration turns into shuffling and the diversity the islands exist
    for disappears within a few exchanges.
    """
    donors = [[t.clone() for t in pop[:migrants]] for pop in pops]
    for i, pop in enumerate(pops):
        incoming = donors[(i - 1) % len(pops)]
        for j, tree in enumerate(incoming):
            pop[-(j + 1)] = tree
