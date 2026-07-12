"""
EML Distiller — символьная дистилляция нейросети в формулу.

Пайплайн:
    1. Собрать датасет (state → oracle_output) с GA-оракула.
    2. Эволюционировать EML-деревья (символьная регрессия).
    3. Получить компактную формулу jump = eml(...).

EML оператор: eml(x, y) = exp(x) - ln(|y| + ε)
"""

import math
import copy
import time
import random
import numpy as np

import config
from config import EML_EPSILON, DEPTH_PENALTIES
from env import FlappyEnv
from ga_net import GANet


# ── EML оператор ────────────────────────────────────────────────────────────

def eml_op(x: float, y: float) -> float:
    """eml(x, y) = exp(clamp(x)) - ln(|y| + ε). Безопасный от NaN."""
    x_clamped = max(-10.0, min(10.0, x))
    return math.exp(x_clamped) - math.log(abs(y) + EML_EPSILON)


# ── AST-дерево ──────────────────────────────────────────────────────────────

class EMLNode:
    """
    Узел AST-дерева EML-формулы.

    Типы:
        'eml'   — eml(left, right)
        'var'   — входная переменная (index 0..3)
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

    def evaluate(self, inputs: np.ndarray) -> float:
        """Вычислить значение формулы для данного state."""
        if self.kind == 'const':
            return self.value
        elif self.kind == 'var':
            return float(inputs[self.var_idx])
        elif self.kind == 'eml':
            lv = self.left.evaluate(inputs)
            rv = self.right.evaluate(inputs)
            return eml_op(lv, rv)
        return 0.0

    def depth(self) -> int:
        """Глубина дерева."""
        if self.kind in ('const', 'var'):
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def size(self) -> int:
        """Количество узлов в дереве."""
        if self.kind in ('const', 'var'):
            return 1
        return 1 + self.left.size() + self.right.size()

    def count_vars(self) -> int:
        """Сколько уникальных переменных используется."""
        if self.kind == 'var':
            return 1 << self.var_idx  # битовая маска
        elif self.kind == 'const':
            return 0
        elif self.kind == 'eml':
            return self.left.count_vars() | self.right.count_vars()
        return 0

    def n_unique_vars(self) -> int:
        """Количество уникальных переменных."""
        mask = self.count_vars()
        return bin(mask).count('1')

    def to_string(self) -> str:
        """Человекочитаемая формула."""
        VAR_NAMES = ['y', 'vel', 'dx', 'gap']
        if self.kind == 'const':
            return f"{self.value:.3f}"
        elif self.kind == 'var':
            return VAR_NAMES[self.var_idx] if self.var_idx < len(VAR_NAMES) \
                else f"x{self.var_idx}"
        elif self.kind == 'eml':
            return f"eml({self.left.to_string()}, {self.right.to_string()})"
        return "?"

    def clone(self) -> "EMLNode":
        """Глубокая копия дерева."""
        if self.kind in ('const', 'var'):
            return EMLNode(self.kind, var_idx=self.var_idx, value=self.value)
        return EMLNode('eml',
                       left=self.left.clone(),
                       right=self.right.clone())

    def to_dict(self) -> dict:
        """JSON-сериализация дерева."""
        if self.kind == 'const':
            return {'kind': 'const', 'value': self.value}
        if self.kind == 'var':
            return {'kind': 'var', 'var_idx': self.var_idx}
        return {
            'kind': 'eml',
            'left': self.left.to_dict(),
            'right': self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EMLNode":
        """Восстановление дерева из словаря."""
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


# ── Генерация случайных деревьев ─────────────────────────────────────────────

def _random_leaf() -> EMLNode:
    """Случайный лист: переменная (70%) или константа (30%)."""
    if random.random() < 0.7:
        return EMLNode('var', var_idx=random.randint(0, 3))
    else:
        return EMLNode('const', value=random.uniform(-2.0, 2.0))


def random_tree(max_depth: int = 3, force_eml: bool = False) -> EMLNode:
    """
    Случайное EML-дерево.

    Args:
        max_depth: максимальная глубина.
        force_eml: если True, корень всегда eml-узел (гарантирует depth >= 1).
    """
    if force_eml and max_depth >= 1:
        return EMLNode('eml',
                       left=random_tree(max_depth - 1, force_eml=False),
                       right=random_tree(max_depth - 1, force_eml=False))
    if max_depth <= 0 or random.random() < 0.3:
        return _random_leaf()
    return EMLNode('eml',
                   left=random_tree(max_depth - 1),
                   right=random_tree(max_depth - 1))


# ── Мутации ──────────────────────────────────────────────────────────────────

def _collect_nodes(node: EMLNode) -> list[EMLNode]:
    """Собрать все узлы дерева в плоский список."""
    nodes = [node]
    if node.kind == 'eml':
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    return nodes


def mutate_tree(root: EMLNode, intensity: float = 1.0) -> EMLNode:
    """
    Мутация дерева с адаптивной интенсивностью.

    Args:
        root: дерево для мутации.
        intensity: 0..1, чем выше — тем агрессивнее мутации.
            1.0 = начало эволюции (дикие мутации)
            0.1 = конец (тонкая настройка)
    """
    tree = root.clone()

    # Количество мутаций за раз: больше при высокой интенсивности
    n_mutations = 1 + int(intensity * 2 * random.random())

    for _ in range(n_mutations):
        nodes = _collect_nodes(tree)
        target = random.choice(nodes)

        r = random.random()

        if r < 0.2 and target.kind == 'const':
            # Подвинуть константу (сильнее при высокой intensity)
            noise_std = 0.2 + intensity * 1.0
            target.value += random.gauss(0, noise_std)
            target.value = max(-5.0, min(5.0, target.value))

        elif r < 0.35 and target.kind == 'var':
            # Сменить переменную
            target.var_idx = random.randint(0, 3)

        elif r < 0.6 and target.kind in ('const', 'var'):
            # Вырастить: лист → eml(лист, лист)
            if tree.depth() < config.EML_MAX_DEPTH:
                target.kind = 'eml'
                target.left = _random_leaf()
                target.right = _random_leaf()

        elif r < 0.7 and target.kind == 'eml':
            # Заменить одну ветку
            if random.random() < 0.5:
                target.left = random_tree(max_depth=2)
            else:
                target.right = random_tree(max_depth=2)

        elif r < 0.8 and target.kind == 'eml' and intensity < 0.5:
            # Упрощение: поддерево → лист (только при низкой intensity)
            leaf = _random_leaf()
            target.kind = leaf.kind
            target.left = None
            target.right = None
            target.var_idx = leaf.var_idx
            target.value = leaf.value

        else:
            # Полная замена поддерева
            depth = 1 + int(intensity * 2)
            new_sub = random_tree(max_depth=depth, force_eml=True)
            target.kind = new_sub.kind
            target.left = new_sub.left
            target.right = new_sub.right
            target.var_idx = new_sub.var_idx
            target.value = new_sub.value

    return tree


def _prune_tree(node: EMLNode, max_depth: int | None = None) -> EMLNode:
    """Обрезать дерево до max_depth — глубокие поддеревья заменяются листами."""
    if max_depth is None:
        max_depth = config.EML_MAX_DEPTH
    if node.kind in ('const', 'var'):
        return node
    if max_depth <= 0:
        # Заменить на лист
        return _random_leaf()
    node.left = _prune_tree(node.left, max_depth - 1)
    node.right = _prune_tree(node.right, max_depth - 1)
    return node


def crossover(parent1: EMLNode, parent2: EMLNode) -> EMLNode:
    """Кросовер: заменить случайное поддерево parent1 поддеревом из parent2."""
    child = parent1.clone()
    nodes1 = _collect_nodes(child)
    nodes2 = _collect_nodes(parent2)

    target = random.choice(nodes1)
    donor = random.choice(nodes2).clone()

    target.kind = donor.kind
    target.left = donor.left
    target.right = donor.right
    target.var_idx = donor.var_idx
    target.value = donor.value

    return child


# ── Сбор датасета ────────────────────────────────────────────────────────────

def collect_dataset(
    oracle: GANet,
    n_episodes: int | None = None,
    mode: str = 'sigmoid',
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Собрать датасет (states, targets) из игры GA-оракула.

    Args:
        oracle: обученная GANet.
        n_episodes: количество эпизодов (None → config.EML_DATASET_EPISODES).
        mode: 'sigmoid' (float 0..1) или 'binary' (0/1).
        max_frames: лимит кадров на эпизод (None → config.GA_MAX_FRAMES).

    Returns:
        (states, targets): np.ndarray shape (N, 4) и (N,).
    """
    if n_episodes is None:
        n_episodes = config.EML_DATASET_EPISODES
    if max_frames is None:
        max_frames = config.GA_MAX_FRAMES
    env = FlappyEnv()
    states_list = []
    targets_list = []

    for _ in range(n_episodes):
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
    Fitness дерева на датасете.

    Правила:
    - Дерево БЕЗ переменных (чистая константа) → почти ноль.
    - Для sigmoid: fitness = 1 / (MSE + depth_penalty * depth + eps)
    - Для binary:  fitness = balanced_accuracy - depth_penalty * depth
      (balanced = (TPR + TNR) / 2, чтобы класс-дисбаланс не давал
       тривиальное решение "всегда 0")
    - Бонус за использование >= 2 переменных.
    """
    n = len(states)
    if n == 0:
        return 0.0

    # Штраф за отсутствие переменных — дерево-константа бесполезно
    n_vars = tree.n_unique_vars()
    if n_vars == 0:
        return 1e-6

    errors = 0.0
    # Для balanced accuracy
    tp = 0  # true positive (target=1, pred=1)
    tn = 0  # true negative (target=0, pred=0)
    fp = 0  # false positive
    fn = 0  # false negative

    for i in range(n):
        try:
            pred = tree.evaluate(states[i])
        except (OverflowError, ValueError, ZeroDivisionError):
            return 1e-6

        if math.isnan(pred) or math.isinf(pred):
            return 1e-6

        if mode == 'sigmoid':
            errors += (pred - targets[i]) ** 2
        else:
            action = 1 if pred > 0.5 else 0
            t = int(targets[i])
            if action == 1 and t == 1:
                tp += 1
            elif action == 0 and t == 0:
                tn += 1
            elif action == 1 and t == 0:
                fp += 1
            else:
                fn += 1

    depth = tree.depth()

    # Бонус за разнообразие переменных (max +20% при 4 переменных)
    var_bonus = 1.0 + 0.05 * n_vars

    if mode == 'sigmoid':
        mse = errors / n
        base = 1.0 / (mse + depth_penalty * depth + 1e-8)
        return base * var_bonus
    else:
        # Balanced accuracy = (TPR + TNR) / 2
        tpr = tp / max(1, tp + fn)  # sensitivity (recall class 1)
        tnr = tn / max(1, tn + fp)  # specificity (recall class 0)
        balanced_acc = (tpr + tnr) / 2.0
        base = max(0.0, balanced_acc - depth_penalty * depth)
        return base * var_bonus


# ── EML-агент ────────────────────────────────────────────────────────────────

class EMLAgent:
    """Обгортка над EMLNode для гри у FlappyEnv."""

    def __init__(self, tree: EMLNode, name: str = "EML"):
        self.tree = tree
        self.name = name

    def get_action(self, state: np.ndarray) -> int:
        """Бінарна дія: 1 (flap) або 0 (no-flap)."""
        try:
            val = self.tree.evaluate(state)
            if math.isnan(val) or math.isinf(val):
                return 0
            return 1 if val > 0.5 else 0
        except (OverflowError, ValueError, ZeroDivisionError):
            return 0

    def to_string(self) -> str:
        return self.tree.to_string()


def _ingame_fitness(tree: EMLNode, n_games: int = 3,
                    max_frames: int | None = None) -> float:
    """
    In-game fitness: запустить формулу в FlappyEnv и посчитать score.
    fitness = avg_score * 100 + avg_frames (аналогично GA).
    """
    if max_frames is None:
        max_frames = config.GA_MAX_FRAMES
    env = FlappyEnv()
    agent = EMLAgent(tree)
    total_score = 0
    total_frames = 0

    for _ in range(n_games):
        state = env.reset()
        f = 0
        while not env.done and f < max_frames:
            action = agent.get_action(state)
            state, _, _ = env.step(action)
            f += 1
        total_score += env.score
        total_frames += f

    avg_score = total_score / n_games
    avg_frames = total_frames / n_games
    return avg_score * 100.0 + avg_frames


# ── Главный цикл эволюции ───────────────────────────────────────────────────

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
    GA-эволюция EML-деревьев.

    Гиперпараметры: если не переданы — берутся из текущего `config`.

    Двухфазный fitness:
    - Фаза 1 (первые 30% поколений): dataset fitness (быстрый скрининг).
    - Фаза 2 (остальные): in-game fitness (реальная оценка).

    Ключевые механизмы:
    - Адаптивная интенсивность мутаций.
    - Турнирная селекция.
    - Сейв лучшего каждого поколения.
    - patience > 0: ранний стоп при плато.

    Returns:
        dict: 'best_tree', 'best_agent', 'history', 'elapsed',
              'stopped_early', 'stop_reason'.
    """
    if generations is None:     generations = config.EML_GENERATIONS
    if population_size is None: population_size = config.EML_POPULATION
    if elitism is None:         elitism = config.EML_ELITISM
    if patience is None:        patience = config.EML_PATIENCE

    dp = DEPTH_PENALTIES[depth_penalty_name]
    # 10% генераций на DATA-фазу (быстрое разогревание), остальное — GAME.
    # При узких таргетах оракула DATA-фитнес = шум и переселяет в вырожденные деревья,
    # поэтому warm-up держим коротким и реально оцениваем игрой.
    phase_switch = max(1, int(generations * 0.1))

    # Инициализация
    population = [random_tree(max_depth=3, force_eml=True)
                  for _ in range(population_size)]
    history = []
    best_fitness_ever = -1.0
    best_tree_ever = None
    patience_counter = 0
    stop_reason = ''

    subsample_size = min(5000, len(states))

    t0 = time.perf_counter()

    for gen in range(generations):
        if should_stop is not None and should_stop():
            stop_reason = 'cancelled'
            if verbose:
                print(f"    -> Cancelled at gen {gen}.")
            break

        intensity = max(0.1, 1.0 - gen / (generations * 0.7))
        use_ingame = gen >= phase_switch

        # ── Переход DATA -> GAME: разные шкалы фитнеса, сбрасываем baseline ──
        # DATA фитнес = 1/MSE (десятки), GAME фитнес = score*100+frames (сотни-тысячи).
        # Без сброса либо GAME-фитнес "не сможет улучшить" DATA-рекорд, либо наоборот.
        if use_ingame and gen == phase_switch and phase_switch > 0:
            best_fitness_ever = -float('inf')
            patience_counter = 0
            if verbose:
                print(f"    -- Phase switch DATA -> GAME at gen {gen}, baseline reset.")

        # ── Fitness ──────────────────────────────────────────────────────
        if use_ingame:
            # In-game: каждое дерево играет 3 игры
            fitness_scores = []
            for tree in population:
                n_vars = tree.n_unique_vars()
                if n_vars == 0:
                    fitness_scores.append(0.0)
                else:
                    f = _ingame_fitness(tree, n_games=3)
                    # Штраф за глубину (но мягче чем в dataset mode)
                    f = max(0.0, f - dp * tree.depth() * 50.0)
                    fitness_scores.append(f)
        else:
            # Dataset: быстрый скрининг
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

        # Сортировать
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

        # Сейв лучшего
        if best_fitness > best_fitness_ever:
            best_fitness_ever = best_fitness
            best_tree_ever = best_tree.clone()
            patience_counter = 0
        else:
            # Plateau на DATA-фазе — это шум MSE на узких таргетах.
            # Считаем плато только на GAME-фазе, где сигнал реальный.
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

        # Новое поколение
        new_pop = []
        for i in range(elitism):
            new_pop.append(population[i].clone())

        while len(new_pop) < population_size:
            r = random.random()
            if r < 0.45:
                idx = _tournament_select(fitness_scores, k=5)
                child = mutate_tree(population[idx], intensity=intensity)
            elif r < 0.75:
                i1 = _tournament_select(fitness_scores, k=5)
                i2 = _tournament_select(fitness_scores, k=5)
                child = crossover(population[i1], population[i2])
                if random.random() < 0.3:
                    child = mutate_tree(child, intensity=intensity * 0.5)
            else:
                child = random_tree(max_depth=2 + int(intensity * 2),
                                    force_eml=True)
            # Обрезать если раздулось
            child = _prune_tree(child)
            new_pop.append(child)

        population = new_pop

    elapsed = time.perf_counter() - t0

    if best_tree_ever is None:
        best_tree_ever = population[0]

    return {
        'best_tree': best_tree_ever,
        'best_agent': EMLAgent(best_tree_ever),
        'history': history,
        'elapsed': elapsed,
        'stopped_early': bool(stop_reason),
        'stop_reason': stop_reason,
    }


def _tournament_select(fitness_scores: list[float], k: int = 5) -> int:
    """Турнирная селекция: выбрать лучшего из k случайных."""
    n = len(fitness_scores)
    candidates = random.sample(range(n), min(k, n))
    return max(candidates, key=lambda i: fitness_scores[i])


# ── Полный пайплайн дистилляции ──────────────────────────────────────────────

def distill(
    oracle: GANet,
    mode: str = 'sigmoid',
    depth_penalty_name: str = 'medium',
    verbose: bool = True,
) -> dict:
    """
    Полный пайплайн: сбор датасета → эволюция EML-дерева.

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
