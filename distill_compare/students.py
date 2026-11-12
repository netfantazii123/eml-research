"""
students.py — the classical distillation students and the EML student, behind
one interface.

Every student is fit on the TEACHER's outputs (its logits / scores), never on
ground-truth labels: what is being measured is how well each model family can
reproduce a given policy, not which one is the better classifier.

Interface:

    s = make_student(name, seed=0)
    s.fit(X, T, kind='classify')                 # T: (N, K) teacher logits
    s.fit(X, T, kind='rank', groups=g)           # T: (N,) teacher scores
    s.scores(X)  -> (N, K) for classify, (N,) for rank; decision = argmax
    s.complexity() -> {'size', 'size_unit', 'flops', 'transcendental', ...}

Complexity is reported in one comparable unit, `flops`: arithmetic operations
plus comparisons per decision, with each transcendental call counted once and
also reported separately (on a Cortex-M without an FPU that is the number that
decides whether the model runs at all). For the formula students `size` is the
AST node count — the same number emlcore.static_metrics reports for EML.
"""

from __future__ import annotations

import os
import random
import sys
import warnings

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from emlcore import make_basis, search, static_metrics  # noqa: E402


# ── Target preparation ───────────────────────────────────────────────────────

def zscore_rows(T: np.ndarray) -> np.ndarray:
    """z-score each row of a logit matrix: removes the arbitrary logit scale and
    offset, preserves the argmax."""
    T = np.asarray(T, dtype=np.float64)
    return (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-8)


def zscore_groups(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """z-score within each group (one decision's candidates). Preserves argmax."""
    y = np.asarray(y, dtype=np.float64)
    _, inv = np.unique(groups, return_inverse=True)
    cnt = np.bincount(inv)
    mean = np.bincount(inv, y) / cnt
    var = np.bincount(inv, (y - mean[inv]) ** 2) / cnt
    return (y - mean[inv]) / (np.sqrt(var[inv]) + 1e-8)


def binary_margin(T: np.ndarray) -> np.ndarray:
    """Logit margin of class 1 over class 0, scaled to unit RMS. Not centred:
    the decision threshold is 0 and has to stay there."""
    m = np.asarray(T[:, 1] - T[:, 0], dtype=np.float64)
    return m / (np.sqrt(np.mean(m ** 2)) + 1e-8)


def group_gap(T: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-row weight = spread of the teacher's scores in that row's group."""
    _, inv = np.unique(groups, return_inverse=True)
    n = inv.max() + 1
    mx = np.full(n, -np.inf)
    mn = np.full(n, np.inf)
    np.maximum.at(mx, inv, T)
    np.minimum.at(mn, inv, T)
    return (mx - mn)[inv]


def softmax(z: np.ndarray, axis=-1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ── Base ─────────────────────────────────────────────────────────────────────

class Student:
    name = 'student'
    family = ''

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.kind = None
        self.n_classes = None

    def fit(self, X, T, *, kind='classify', groups=None, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        self.kind = kind
        if kind == 'classify':
            self.n_classes = T.shape[1]
        elif groups is None:
            raise ValueError("rank needs groups")
        w = (np.ones(len(X)) if sample_weight is None
             else np.asarray(sample_weight, dtype=np.float64))
        self._fit(X, T, groups, w)
        return self

    def actions(self, X) -> np.ndarray:
        """classify only: argmax class per row."""
        return np.argmax(self.scores(X), axis=1)

    # subclasses
    def _fit(self, X, T, groups, w): raise NotImplementedError
    def scores(self, X): raise NotImplementedError
    def complexity(self) -> dict: raise NotImplementedError
    def describe(self) -> str: return ''


# ── CART decision tree ───────────────────────────────────────────────────────

class TreeStudent(Student):
    """
    CART on the teacher's decisions — the standard interpretable-policy
    baseline (behavioural cloning into a tree). classify: classifier on the
    teacher's argmax; rank: regressor on the per-group z-scored teacher score.
    """
    family = 'tree'

    def __init__(self, max_depth=6, seed=0, min_samples_leaf=5):
        super().__init__(seed)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.name = f'tree_d{max_depth}'

    def _fit(self, X, T, groups, w):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        if self.kind == 'classify':
            y = np.argmax(T, axis=1)
            self.m = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed)
            self.m.fit(X, y, sample_weight=w)
        else:
            y = zscore_groups(T, groups)
            self.m = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                random_state=self.seed)
            self.m.fit(X, y, sample_weight=w)

    def scores(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.kind == 'rank':
            return self.m.predict(X)
        p = self.m.predict_proba(X)
        out = np.zeros((len(X), self.n_classes))
        out[:, self.m.classes_.astype(int)] = p
        return out

    def complexity(self):
        t = self.m.tree_
        return {'size': int(t.node_count), 'size_unit': 'nodes',
                'flops': int(self.m.get_depth()), 'transcendental': 0,
                'depth': int(self.m.get_depth()),
                'leaves': int(self.m.get_n_leaves())}


# ── Linear ───────────────────────────────────────────────────────────────────

class LinearStudent(Student):
    """
    Linear student. classify: multinomial logistic regression on the teacher's
    argmax; rank: ridge regression on the per-group z-scored teacher score (for
    Tetris this is exactly the classic hand-weighted afterstate evaluation,
    with weights learned from the oracle). Input standardisation is folded into
    the weights, so inference is one dot product per output.
    """
    family = 'linear'
    name = 'linear'

    def _fit(self, X, T, groups, w):
        from sklearn.linear_model import LogisticRegression, Ridge
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Z = (X - self.mu) / self.sd
        if self.kind == 'classify':
            y = np.argmax(T, axis=1)
            self.m = LogisticRegression(C=1.0, max_iter=3000)
            if len(np.unique(y)) < 2:
                self.const = int(y[0])
                self.m = None
                return
            self.m.fit(Z, y, sample_weight=w)
        else:
            self.m = Ridge(alpha=1.0)
            self.m.fit(Z, zscore_groups(T, groups), sample_weight=w)

    def scores(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        if self.kind == 'rank':
            return self.m.predict(Z)
        out = np.zeros((len(Z), self.n_classes))
        if self.m is None:
            out[:, self.const] = 1.0
            return out
        out[:, self.m.classes_.astype(int)] = self.m.predict_proba(Z)
        return out

    def complexity(self):
        if self.m is None:
            return {'size': 1, 'size_unit': 'params', 'flops': 0,
                    'transcendental': 0}
        coef = np.atleast_2d(self.m.coef_)
        n_out = coef.shape[0]
        nnz = int(np.count_nonzero(np.abs(coef) > 1e-12))
        return {'size': int(nnz + n_out), 'size_unit': 'params',
                'flops': int(2 * nnz), 'transcendental': 0}


# ── Small MLP student (knowledge distillation) ──────────────────────────────

class KDMLPStudent(Student):
    """
    A small MLP trained on the teacher's soft outputs.

    classify: Hinton et al. (2015) — loss = a*T^2*KL(softmax(t/T) || softmax(s/T))
              + (1-a)*CE(teacher argmax), T = temperature.
    rank:     Ba & Caruana (2014) logit matching — weighted MSE to the per-group
              z-scored teacher score.

    This is the "compress the network into a smaller network" end of the
    spectrum: the strongest fidelity per parameter, and no readability at all.
    """
    family = 'mlp'

    def __init__(self, hidden=16, temperature=4.0, alpha=0.9, epochs=200,
                 lr=3e-3, seed=0):
        super().__init__(seed)
        self.hidden = hidden
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.name = f'kd_mlp_h{hidden}'

    def _fit(self, X, T, groups, w):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        rng = np.random.default_rng(self.seed)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Z = ((X - self.mu) / self.sd).astype(np.float32)
        n_out = self.n_classes if self.kind == 'classify' else 1
        self.net = nn.Sequential(nn.Linear(X.shape[1], self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, n_out))
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        if self.kind == 'classify':
            tgt = torch.tensor(T, dtype=torch.float32)
        else:
            tgt = torch.tensor(zscore_groups(T, groups), dtype=torch.float32)
        Xt = torch.tensor(Z)
        Wt = torch.tensor(w / w.mean(), dtype=torch.float32)
        n = len(Z)
        perm = rng.permutation(n)
        n_val = max(1, n // 10) if n >= 50 else 0
        val, tr = perm[:n_val], perm[n_val:]
        bs = min(256, max(32, len(tr) // 8))
        T_ = self.temperature

        def loss_fn(idx):
            out = self.net(Xt[idx])
            if self.kind == 'classify':
                t = tgt[idx]
                soft = F.kl_div(F.log_softmax(out / T_, 1), F.softmax(t / T_, 1),
                                reduction='none').sum(1) * T_ * T_
                hard = F.cross_entropy(out, t.argmax(1), reduction='none')
                per = self.alpha * soft + (1 - self.alpha) * hard
            else:
                per = (out[:, 0] - tgt[idx]) ** 2
            return (per * Wt[idx]).mean()

        best, best_state, stall = np.inf, None, 0
        max_steps = 6000
        steps = 0
        for ep in range(self.epochs):
            rng.shuffle(tr)
            for i in range(0, len(tr), bs):
                idx = torch.tensor(tr[i:i + bs])
                opt.zero_grad()
                loss_fn(idx).backward()
                opt.step()
                steps += 1
            if n_val:
                with torch.no_grad():
                    v = float(loss_fn(torch.tensor(val)))
                if v < best - 1e-5:
                    best, stall = v, 0
                    best_state = {k: x.clone() for k, x in self.net.state_dict().items()}
                else:
                    stall += 1
                if stall >= 20:
                    break
            if steps >= max_steps:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def scores(self, X):
        import torch
        Z = ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd).astype(np.float32)
        with torch.no_grad():
            out = self.net(torch.from_numpy(Z)).numpy().astype(np.float64)
        return out[:, 0] if self.kind == 'rank' else out

    def complexity(self):
        n_params = sum(p.numel() for p in self.net.parameters())
        lin = [m for m in self.net if hasattr(m, 'weight')]
        macs = sum(m.weight.numel() for m in lin)
        return {'size': int(n_params), 'size_unit': 'params',
                'flops': int(2 * macs + self.hidden), 'transcendental': 0}


# ── Formula students (one regressor per output) ─────────────────────────────

class _FormulaStudent(Student):
    """
    Shared wiring for students that produce one scalar formula per output.

      classify K=2 : one formula on the scaled logit margin, class 1 iff f > 0
      classify K>2 : one formula per class on the row-z-scored logits, decision
                     argmax_c (f_c - mu_c) / sigma_c with mu, sigma measured on
                     train (the calibration aerial_classifier branch A uses)
      rank         : one formula on the per-group z-scored score, argmax
    """

    def _make_regressor(self, idx: int):
        raise NotImplementedError

    def _fit(self, X, T, groups, w):
        self.regs = []
        if self.kind == 'rank':
            self.regs.append(self._make_regressor(0).fit(X, zscore_groups(T, groups), w))
        elif self.n_classes == 2:
            self.regs.append(self._make_regressor(0).fit(X, binary_margin(T), w))
        else:
            Z = zscore_rows(T)
            for c in range(self.n_classes):
                self.regs.append(self._make_regressor(c).fit(X, Z[:, c], w))
            P = np.stack([r.predict(X) for r in self.regs], 1)
            P = np.where(np.isfinite(P), P, 0.0)
            self.cal_mu = P.mean(0)
            self.cal_sd = P.std(0) + 1e-8

    def scores(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.kind == 'rank':
            v = self.regs[0].predict(X)
            return np.where(np.isfinite(v), v, -np.inf)
        if self.n_classes == 2:
            f = self.regs[0].predict(X)
            f = np.where(np.isfinite(f), f, -1.0)
            return np.stack([np.zeros_like(f), f], 1)
        P = np.stack([r.predict(X) for r in self.regs], 1)
        P = (P - self.cal_mu) / self.cal_sd
        return np.where(np.isfinite(P), P, -np.inf)

    def complexity(self):
        parts = [r.complexity() for r in self.regs]
        out = {'size': sum(p['size'] for p in parts), 'size_unit': 'nodes',
               'flops': sum(p['flops'] for p in parts),
               'transcendental': sum(p['transcendental'] for p in parts)}
        if self.kind == 'classify' and self.n_classes > 2:
            out['flops'] += 2 * self.n_classes   # calibration
        return out

    def describe(self):
        return ' | '.join(r.describe() for r in self.regs)


class GARegressor:
    """
    emlcore's GA as a plain regressor: DATA-phase symbolic regression with a
    per-node parsimony penalty. Same search (search.breed), same basis code and
    same numerical guards as the task distillers; what it does NOT have is a
    task-specific GAME phase, which is why on the games the full task pipeline
    is run as well.

    Fitness = -NMSE - parsimony * nodes, on a fresh row subsample every
    generation (common to the whole population); the champion of each
    generation is re-scored on a fixed selection subset, and the best of those
    is returned — the same train/select split the task distillers use.
    """

    def __init__(self, basis='eml', population=300, generations=120, elitism=6,
                 max_depth=6, parsimony=2e-3, subsample=2000, patience=40,
                 seed=0):
        self.basis_name = basis
        self.population = population
        self.generations = generations
        self.elitism = elitism
        self.max_depth = max_depth
        self.parsimony = parsimony
        self.subsample = subsample
        self.patience = patience
        self.seed = seed

    def _fitness(self, tree, X, y, w, var_y):
        if tree.n_unique_vars() == 0:
            return -1e9
        with np.errstate(all='ignore'):
            p = tree.evaluate_batch(X)
            if not np.all(np.isfinite(p)):
                return -1e9
            mse = float(np.sum(w * (p - y) ** 2) / np.sum(w)) / var_y
        if not np.isfinite(mse):
            return -1e9
        return -mse - self.parsimony * tree.size()

    def fit(self, X, y, w=None):
        random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        w = np.ones(n) if w is None else w
        var_y = float(np.sum(w * (y - np.average(y, weights=w)) ** 2) / np.sum(w)) + 1e-12
        basis = make_basis(self.basis_name, d, max_depth=self.max_depth)
        self.basis = basis
        pop = [search.random_tree(basis, max_depth=3, force_op=True)
               for _ in range(self.population)]
        sel = rng.choice(n, min(n, 4000), replace=False)
        Xs, ys, ws = X[sel], y[sel], w[sel]
        best, best_fit, stall = None, -np.inf, 0
        self.history = []
        for gen in range(self.generations):
            intensity = max(0.1, 1.0 - gen / (self.generations * 0.7))
            idx = rng.choice(n, min(n, self.subsample), replace=False)
            Xg, yg, wg = X[idx], y[idx], w[idx]
            fits = [self._fitness(t, Xg, yg, wg, var_y) for t in pop]
            order = np.argsort(fits)[::-1]
            pop = [pop[i] for i in order]
            fits = [fits[i] for i in order]
            champ_fit = self._fitness(pop[0], Xs, ys, ws, var_y)
            self.history.append((gen, fits[0], champ_fit))
            if champ_fit > best_fit + 1e-6:
                best, best_fit, stall = pop[0].clone(), champ_fit, 0
            else:
                stall += 1
                if stall >= self.patience:
                    break
            pop = search.breed(pop, fits, basis, size=self.population,
                               elitism=self.elitism, intensity=intensity)
        self.tree = best if best is not None else pop[0]
        self.generations_run = gen + 1
        return self

    def predict(self, X):
        with np.errstate(all='ignore'):
            return self.tree.evaluate_batch(np.asarray(X, dtype=np.float64))

    def complexity(self):
        m = static_metrics(self.tree)
        # eml(x, y) = exp - log: two transcendental calls and one subtraction.
        flops = m['n_ops'] + m['n_eml'] * 2
        return {'size': m['nodes'], 'flops': int(flops),
                'transcendental': m['n_transcendental']}

    def describe(self):
        return self.tree.to_string()


class GAFormulaStudent(_FormulaStudent):
    """EML (or the control basis) through emlcore's GA, DATA phase only."""

    def __init__(self, basis='eml', seed=0, **ga_kw):
        super().__init__(seed)
        self.basis = basis
        self.ga_kw = ga_kw
        self.family = 'formula'
        self.name = 'eml_data' if basis == 'eml' else 'classic_gp'

    def _make_regressor(self, idx):
        return GARegressor(basis=self.basis, seed=self.seed * 1000 + idx,
                           **self.ga_kw)


class GPLearnRegressor:
    """gplearn's SymbolicRegressor — an independent GP-SR implementation."""

    FUNCS = ('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'neg')
    TRANSC = ('log', 'sqrt')

    def __init__(self, seed=0, population=500, generations=20, subsample=2000):
        self.seed = seed
        self.population = population
        self.generations = generations
        self.subsample = subsample

    def fit(self, X, y, w=None):
        from gplearn.genetic import SymbolicRegressor
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), min(len(X), self.subsample), replace=False)
        self.m = SymbolicRegressor(
            population_size=self.population, generations=self.generations,
            function_set=self.FUNCS, parsimony_coefficient=0.001,
            init_depth=(2, 4), max_samples=0.9, random_state=self.seed,
            n_jobs=1, verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.m.fit(X[idx], y[idx], sample_weight=None if w is None else w[idx])
        return self

    def predict(self, X):
        with np.errstate(all='ignore'):
            return self.m.predict(np.asarray(X, dtype=np.float64))

    def complexity(self):
        prog = self.m._program.program
        names = [getattr(n, 'name', None) for n in prog]
        n_ops = sum(1 for n in prog if hasattr(n, 'arity'))
        transc = sum(1 for nm in names if nm in self.TRANSC)
        return {'size': int(self.m._program.length_), 'flops': int(n_ops),
                'transcendental': int(transc)}

    def describe(self):
        return str(self.m._program)


class GPLearnStudent(_FormulaStudent):
    family = 'formula'
    name = 'gplearn_sr'

    def __init__(self, seed=0, **kw):
        super().__init__(seed)
        self.kw = kw

    def _make_regressor(self, idx):
        return GPLearnRegressor(seed=self.seed * 1000 + idx, **self.kw)


# ── Registry ─────────────────────────────────────────────────────────────────

def make_student(name: str, seed: int = 0, **kw) -> Student:
    """Build a student by name. Extra kwargs go to the constructor."""
    if name.startswith('tree_d'):
        return TreeStudent(max_depth=int(name[6:]), seed=seed, **kw)
    if name == 'linear':
        return LinearStudent(seed=seed)
    if name.startswith('kd_mlp_h'):
        return KDMLPStudent(hidden=int(name[8:]), seed=seed, **kw)
    if name == 'gplearn_sr':
        return GPLearnStudent(seed=seed, **kw)
    if name == 'eml_data':
        return GAFormulaStudent('eml', seed=seed, **kw)
    if name == 'classic_gp':
        return GAFormulaStudent('classic', seed=seed, **kw)
    raise ValueError(f"unknown student {name!r}")


DEFAULT_STUDENTS = ('tree_d3', 'tree_d6', 'linear', 'kd_mlp_h16',
                    'gplearn_sr', 'eml_data', 'classic_gp')
