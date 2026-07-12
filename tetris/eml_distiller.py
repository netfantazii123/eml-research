"""
eml_distiller.py — символьная дистилляция CNN-оракула в ОДНУ EML-формулу.

Идея (v2, placement-based): оракул и формула выбирают ПОСТАНОВКУ фигуры.
Формула — классическая оценочная функция (как у Dellacherie): ей показывают
признаки доски-после-постановки (afterstate) и она возвращает скаляр.
На инференсе:
    placement* = argmax_i f(features(afterstate_i))  по легальным i.

Две фазы:
  1. DATA-фаза  — символьная регрессия: f(features_i) ≈ logit_i оракула
     (взвешенный MSE, векторно на numpy). Формула учится ранжировать
     постановки так же, как оракул.
  2. GAME-фаза  — популяция формул доводится эволюцией по РЕАЛЬНОМУ score
     в TetrisEnv: формула играет сама, fitness = линии + выживание.

EML-оператор:  eml(x, y) = exp(clamp(x, -10, 10)) - ln(|y| + ε).

AST-движок портирован из flappy_bird_with_ai/eml_distiller.py; отличия:
  - n_vars = 19 (признаки afterstate), имена из config.FEATURE_NAMES;
  - векторная evaluate_batch() для быстрой оценки на датасете;
  - одна формула вместо шести — argmax теперь по постановкам, а не действиям.
"""

import math
import time
import random

import numpy as np

import config
from config import EML_EPSILON, DEPTH_PENALTIES, FEATURE_NAMES, N_FEATURES
from env import TetrisEnv


# ── EML-оператор ──────────────────────────────────────────────────────────────

def eml_op(x: float, y: float) -> float:
    """eml(x, y) = exp(clamp(x)) - ln(|y| + ε). Скалярная (numpy-free) версия."""
    x_clamped = -10.0 if x < -10.0 else (10.0 if x > 10.0 else x)
    return math.exp(x_clamped) - math.log(abs(y) + EML_EPSILON)


# ── AST-дерево ────────────────────────────────────────────────────────────────

class EMLNode:
    """
    Узел AST EML-формулы.

    Типы:
        'eml'   — eml(left, right)
        'var'   — входной признак (index 0..N_FEATURES-1)
        'const' — числовая константа
    """

    __slots__ = ('kind', 'left', 'right', 'var_idx', 'value')

    def __init__(self, kind: str, left=None, right=None,
                 var_idx: int = 0, value: float = 1.0):
        self.kind = kind
        self.left = left
        self.right = right
        self.var_idx = var_idx
        self.value = value

    # ── Вычисление ────────────────────────────────────────────────────────
    def evaluate(self, inputs) -> float:
        """Скалярное вычисление для одного вектора признаков (инференс)."""
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
        Векторное вычисление для всего датасета X (N, 19) → (N,).

        Все операции замкнуты и численно ограничены (clamp + ε), поэтому
        результат всегда конечен.
        """
        if self.kind == 'const':
            return np.full(X.shape[0], self.value, dtype=np.float64)
        if self.kind == 'var':
            return X[:, self.var_idx].astype(np.float64)
        lv = self.left.evaluate_batch(X)
        rv = self.right.evaluate_batch(X)
        return np.exp(np.clip(lv, -10.0, 10.0)) - np.log(np.abs(rv) + EML_EPSILON)

    # ── Метрики ───────────────────────────────────────────────────────────
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

    # ── Сериализация / печать ─────────────────────────────────────────────
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


# ── Генерация случайных деревьев ──────────────────────────────────────────────

def _random_leaf() -> EMLNode:
    """Случайный лист: переменная (70%) или константа (30%)."""
    if random.random() < 0.7:
        return EMLNode('var', var_idx=random.randint(0, N_FEATURES - 1))
    return EMLNode('const', value=random.uniform(-2.0, 2.0))


def random_tree(max_depth: int = 3, force_eml: bool = False) -> EMLNode:
    """Случайное EML-дерево. force_eml → корень всегда eml (depth >= 1)."""
    if force_eml and max_depth >= 1:
        return EMLNode('eml',
                       left=random_tree(max_depth - 1, force_eml=False),
                       right=random_tree(max_depth - 1, force_eml=False))
    if max_depth <= 0 or random.random() < 0.3:
        return _random_leaf()
    return EMLNode('eml',
                   left=random_tree(max_depth - 1),
                   right=random_tree(max_depth - 1))


# ── Мутации / кроссовер ───────────────────────────────────────────────────────

def _collect_nodes(node: EMLNode) -> list[EMLNode]:
    nodes = [node]
    if node.kind == 'eml':
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    return nodes


def mutate_tree(root: EMLNode, intensity: float = 1.0) -> EMLNode:
    """Мутация с адаптивной интенсивностью (1.0 = дикая, 0.1 = тонкая)."""
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
    """Обрезать дерево до max_depth — глубокие поддеревья → листы."""
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
    """Заменить случайное поддерево parent1 поддеревом из parent2."""
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


# ── DATA-фаза: регрессия формулы к логитам оракула ────────────────────────────

def _regression_fitness(tree: EMLNode, X: np.ndarray, y: np.ndarray,
                        w: np.ndarray, depth_penalty: float) -> float:
    """
    Fitness = var_bonus / (weighted_MSE + depth_penalty * depth).

    Дерево-константа (0 переменных) бесполезно → ~0.
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
    verbose: bool = False,
    label: str = 'PLACE',
    should_stop=None,
    on_event=None,
) -> dict:
    """
    GA-эволюция EML-формулы (символьная регрессия к таргету y).

    on_event: опциональный колбэк(dict) для прогресса (GUI). Вызывается
    каждое поколение с {'type':'data_gen', ...}.

    Returns: {'best_tree', 'best_fitness', 'history'}.
    """
    if subsample is None:
        subsample = config.EML_DATA_SUBSAMPLE
    elitism = max(1, min(elitism, population_size))

    rng = np.random.default_rng(0)
    population = [random_tree(max_depth=3, force_eml=True)
                 for _ in range(population_size)]
    history: list[float] = []
    best_fitness_ever = -1.0
    best_tree_ever: EMLNode | None = None
    patience_counter = 0

    for gen in range(generations):
        if should_stop is not None and should_stop():
            break

        intensity = max(0.1, 1.0 - gen / (generations * 0.7))

        # Подвыборка для скорости (свежая каждое поколение → меньше оверфита).
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

        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_tree_ever = population[0].clone()
            patience_counter = 0
        else:
            patience_counter += 1

        bt = population[0]
        if verbose and (gen % 25 == 0 or gen == generations - 1):
            print(f"      [{label}] gen {gen:>4} | fit {best_fitness:10.2f} | "
                  f"D{bt.depth()} S{bt.size()} V{bt.n_unique_vars()} | "
                  f"i={intensity:.2f}")
        if on_event is not None:
            on_event({'type': 'data_gen', 'action': 0, 'name': label,
                      'gen': gen, 'generations': generations,
                      'fitness': best_fitness, 'depth': bt.depth(),
                      'size': bt.size(), 'nvars': bt.n_unique_vars()})

        if patience > 0 and patience_counter >= patience:
            if verbose:
                print(f"      [{label}] early stop @ gen {gen} "
                      f"(no improve {patience})")
            break

        # Новое поколение.
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
            'history': history}


# ── EML-политика (одна формула, argmax по постановкам) ────────────────────────

class EMLPolicy:
    """
    Обёртка над EML-деревом: placement* = argmax_i f(afterstate_i).

    Принимает и список из одного дерева (формат storage.save_formulas).
    """

    def __init__(self, tree):
        if isinstance(tree, (list, tuple)):
            assert len(tree) >= 1, "пустой список формул"
            tree = tree[0]
        self.tree = tree

    def choose(self, mask: np.ndarray, afeats: np.ndarray,
               scalars: np.ndarray | None = None) -> int:
        """
        Лучшая легальная постановка по признакам obs.

        Вход формулы = afterstate-признаки (19) + one-hot следующей фигуры (7),
        который берётся из obs['scalars'][7:14]. Если scalars не переданы,
        next-признаки заполняются нулями (формулы без next_* не пострадают).
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
        """Выбрать лучшую легальную постановку для текущего состояния env."""
        obs = env.get_placement_state()
        return self.choose(obs['mask'], obs['afeats'], obs['scalars'])


# ── GAME-фаза: in-game эволюция формулы ───────────────────────────────────────

def play_episodes(tree, n_games: int, max_placements: int,
                  seed: int) -> tuple[float, float]:
    """Сыграть n_games формулой. Возвращает (avg_lines, avg_placements)."""
    env = TetrisEnv(seed=seed)
    policy = EMLPolicy(tree)
    total_lines = 0
    total_steps = 0
    for _ in range(n_games):
        obs = env.reset()
        f = 0
        while not env.done and f < max_placements:
            a = policy.choose(obs['mask'], obs['afeats'], obs['scalars'])
            if a < 0:
                break
            obs, _, _, _ = env.step_placement(a)
            f += 1
        total_lines += env.score
        total_steps += f
    return total_lines / n_games, total_steps / n_games


def _ingame_fitness(tree, n_games, max_placements, seed) -> float:
    """fitness = avg_lines * 100 + avg_placements (выживание — tie-breaker)."""
    avg_lines, avg_steps = play_episodes(tree, n_games, max_placements, seed)
    return avg_lines * 100.0 + avg_steps


def evolve_ingame(
    seed_tree: EMLNode,
    *,
    generations: int | None = None,
    population_size: int | None = None,
    elitism: int | None = None,
    patience: int | None = None,
    n_games: int | None = None,
    max_placements: int | None = None,
    eval_seed: int = 1000,
    verbose: bool = True,
    should_stop=None,
    on_event=None,
) -> dict:
    """
    In-game эволюция формулы: fitness = реальный score в TetrisEnv.
    Сидируется лучшей формулой DATA-фазы.

    on_event: опциональный колбэк(dict), {'type':'joint_gen', ...} каждое поколение.

    Returns: {'best_tree', 'best_fitness', 'best_lines', 'history'}.
    """
    if generations is None:     generations = config.EML_JOINT_GENERATIONS
    if population_size is None:  population_size = config.EML_JOINT_POPULATION
    if elitism is None:         elitism = config.EML_JOINT_ELITISM
    if patience is None:        patience = config.EML_JOINT_PATIENCE
    if n_games is None:         n_games = config.EML_INGAME_GAMES
    if max_placements is None:  max_placements = config.EML_INGAME_MAX_PLACEMENTS
    elitism = max(1, min(elitism, population_size))

    # Популяция: оригинал + мутанты сид-формулы.
    population = [seed_tree.clone()]
    while len(population) < population_size:
        population.append(mutate_tree(seed_tree, intensity=0.7))

    history: list[float] = []
    best_fitness_ever = -math.inf
    best_tree_ever: EMLNode | None = None
    patience_counter = 0

    for gen in range(generations):
        if should_stop is not None and should_stop():
            break

        intensity = max(0.15, 1.0 - gen / (generations * 0.8))
        # Все формулы оцениваются на ОДНИХ сидах → честное сравнение.
        fitness_scores = [_ingame_fitness(t, n_games, max_placements, eval_seed)
                          for t in population]

        ranked = sorted(zip(fitness_scores, population),
                        key=lambda p: p[0], reverse=True)
        fitness_scores = [f for f, _ in ranked]
        population = [t for _, t in ranked]

        best_fitness = fitness_scores[0]
        history.append(best_fitness)

        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_tree_ever = population[0].clone()
            patience_counter = 0
        else:
            patience_counter += 1

        best_lines = best_fitness / 100.0
        if verbose:
            print(f"    [GAME] gen {gen:>3} | fit {best_fitness:9.1f} | "
                  f"~{best_lines:.1f} lines | i={intensity:.2f}")
        if on_event is not None:
            on_event({'type': 'joint_gen', 'gen': gen, 'generations': generations,
                      'fitness': best_fitness, 'lines': best_lines})

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
    return {
        'best_tree': best_tree_ever,
        'best_fitness': best_fitness_ever,
        'best_lines': best_fitness_ever / 100.0,
        'history': history,
    }


def _standardize_per_group(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """z-score таргета внутри каждой группы (шага): сохраняет argmax."""
    out = np.empty_like(y)
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    start = 0
    for end in list(boundaries) + [len(groups)]:
        seg = y[start:end]
        out[start:end] = (seg - seg.mean()) / (seg.std() + 1e-8)
        start = end
    return out


# ── Полный пайплайн дистилляции ───────────────────────────────────────────────

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
    DATA-фаза (регрессия к логитам) → GAME-фаза (in-game эволюция).

    Args:
        features: (N, 19), logits: (N,) — из dataset_collector.
        weights:  (N,) per-sample веса (None → по группам, если есть, иначе 1).
        joint:    включить GAME-фазу (можно отключить для отладки).
        on_event: опциональный колбэк(dict) для прогресса (GUI).

    Returns:
        dict с 'trees' ([формула] — список для совместимости со storage),
        метриками DATA/GAME и историей.
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

    # Нормализация таргета. Абсолютный масштаб логитов оракула произволен
    # (mean -46, std 48 на реальном датасете) — важно только РАНЖИРОВАНИЕ
    # постановок внутри шага. z-score внутри группы делает таргет безмасштабным
    # (MSE ~ O(1)) и сохраняет argmax; без групп — глобальный z-score.
    if groups is not None:
        y = _standardize_per_group(y, np.asarray(groups))
    else:
        y = (y - y.mean()) / (y.std() + 1e-8)

    # ── DATA-фаза: регрессия одной формулы ───────────────────────────────────
    _emit({'type': 'phase', 'phase': 'data', 'n_actions': 1})
    if verbose:
        print(f"  [DATA] evolving placement-value formula "
              f"({len(X):,} samples)")
    _emit({'type': 'data_action_start', 'action': 0, 'name': 'PLACE'})
    res = evolve_formula(
        X, y, weights,
        generations=data_generations,
        population_size=data_population,
        elitism=config.EML_ELITISM,
        patience=config.EML_PATIENCE,
        depth_penalty=dp,
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

    # Базовый in-game score формулы DATA-фазы (до in-game доводки).
    base_lines, _ = play_episodes(
        data_tree, config.EML_INGAME_GAMES,
        config.EML_INGAME_MAX_PLACEMENTS, seed=1000)
    if verbose:
        print(f"  DATA-phase in-game score: {base_lines:.2f} lines/game")
    _emit({'type': 'base_score', 'base_lines': base_lines})

    # ── GAME-фаза: in-game доводка ──────────────────────────────────────────
    if joint:
        if verbose:
            print("  [GAME] in-game evolution...")
        _emit({'type': 'phase', 'phase': 'joint'})
        jres = evolve_ingame(data_tree, verbose=verbose,
                             should_stop=should_stop, on_event=on_event)
        final_tree = jres['best_tree']
        joint_lines = jres['best_lines']
        # Не даём GAME-фазе ухудшить результат относительно DATA.
        if joint_lines < base_lines:
            if verbose:
                print(f"  GAME ({joint_lines:.2f}) < DATA ({base_lines:.2f}) "
                      f"-> keep DATA tree")
            final_tree = data_tree
            joint_lines = base_lines
    else:
        jres = None
        final_tree = data_tree
        joint_lines = base_lines

    return {
        'trees': [final_tree],
        'data_trees': [data_tree],
        'data_results': [res],
        'joint_result': jres,
        'base_lines': base_lines,
        'final_lines': joint_lines,
        'depth_penalty': depth_penalty_name,
        'dataset_size': len(features),
    }
