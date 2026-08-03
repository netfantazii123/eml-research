"""
eml_distiller.py — символьная дистилляция оракула-классификатора в EML-формулы.

Идея: обучить НЕ одну формулу, а по одной на класс. Каждая формула f_c даёт
скаляр («насколько это похоже на класс c»), предсказание — argmax по классам:

    class* = argmax_c f_c(inputs)

Это прямой аналог Тетриса, где argmax брался по постановкам фигуры: там
формула ранжировала ходы, здесь — классы. Абсолютные значения формул не важны,
важен только порядок, и это определяет обе фазы обучения.

EML-оператор:  eml(x, y) = exp(clamp(x, -10, 10)) - ln(|y| + eps).

Две фазы:

  1. DATA-фаза — символьная регрессия: f_c(inputs) ~ logit_c оракула.
     Логиты предварительно z-нормируются ВНУТРИ КАЖДОГО ПРИМЕРА (по 4 классам).
     Абсолютный масштаб логитов произволен и от примера к примеру разный;
     нормировка по строке делает таргет безмасштабным и, главное, сохраняет
     argmax — то есть ровно ту информацию, которая нам нужна.

  2. ACC-фаза — аналог GAME-фазы Тетриса («формула играет сама»). MSE к логитам
     и accuracy — разные вещи: формула может точно повторять логиты и всё равно
     путать argmax на пограничных примерах. Поэтому вторая фаза эволюционирует
     ВЕСЬ НАБОР из 4 формул целиком, а fitness = реальная сбалансированная
     accuracy. Она сидируется результатом DATA-фазы, то есть стартует не с
     нуля, а с осмысленного приближения.

Про честность метрики: fitness ACC-фазы считается на ПОДВЫБОРКЕ TRAIN, никогда
на val/test. Иначе val перестаёт быть независимой оценкой — эволюция по нему
это такая же подгонка, как обучение градиентом.

AST-движок (EMLNode, random_tree, mutate_tree, crossover) портирован из
tetris/eml_distiller.py; отличия — набор формул вместо одной, ACC-фаза вместо
GAME-фазы и настраиваемая размерность входа (32 признака в ветке A,
CNN_EMBED_DIM координат эмбеддинга в ветке B).
"""

import math
import random
import time

import numpy as np

import config
from config import EML_EPSILON

# ── Пространство входов ───────────────────────────────────────────────────────
# Ветки A и B подают формуле разные векторы, поэтому размерность и имена входов
# задаются перед эволюцией, а не берутся из config жёстко.

_N_INPUTS = config.N_FEATURES
_INPUT_NAMES = list(config.FEATURE_NAMES)


def configure_inputs(n_inputs: int, names: list[str] | None = None):
    """Задать размерность и имена входов формул (вызывать до эволюции)."""
    global _N_INPUTS, _INPUT_NAMES
    _N_INPUTS = int(n_inputs)
    _INPUT_NAMES = list(names) if names else [f'x{i}' for i in range(n_inputs)]


def input_names() -> list[str]:
    return list(_INPUT_NAMES)


# ── EML-оператор ──────────────────────────────────────────────────────────────

def eml_op(x: float, y: float) -> float:
    """eml(x, y) = exp(clamp(x)) - ln(|y| + eps). Скалярная (numpy-free) версия."""
    x_clamped = -10.0 if x < -10.0 else (10.0 if x > 10.0 else x)
    return math.exp(x_clamped) - math.log(abs(y) + EML_EPSILON)


# ── AST-дерево ────────────────────────────────────────────────────────────────

class EMLNode:
    """
    Узел AST EML-формулы.

    Типы:
        'eml'   — eml(left, right)
        'var'   — входная координата (index 0.._N_INPUTS-1)
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
        """Скалярное вычисление для одного вектора входов (инференс)."""
        if self.kind == 'const':
            return self.value
        if self.kind == 'var':
            return float(inputs[self.var_idx])
        return eml_op(self.left.evaluate(inputs), self.right.evaluate(inputs))

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Векторное вычисление для всего датасета X (N, n_inputs) -> (N,).

        Операции замкнуты и численно ограничены (clamp + eps), поэтому
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

    def used_vars(self) -> list[int]:
        """Индексы входов, реально встречающихся в дереве."""
        mask = self._var_mask()
        return [i for i in range(_N_INPUTS) if mask >> i & 1]

    # ── Сериализация / печать ─────────────────────────────────────────────
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
        raise ValueError(f"неизвестный тип узла: {kind}")


# ── Генерация случайных деревьев ──────────────────────────────────────────────

def _random_leaf() -> EMLNode:
    """Случайный лист: переменная (70%) или константа (30%)."""
    if random.random() < 0.7:
        return EMLNode('var', var_idx=random.randint(0, _N_INPUTS - 1))
    return EMLNode('const', value=random.uniform(-2.0, 2.0))


def random_tree(max_depth: int = 3, force_eml: bool = False) -> EMLNode:
    """Случайное EML-дерево. force_eml -> корень всегда eml (depth >= 1)."""
    if force_eml and max_depth >= 1:
        return EMLNode('eml',
                       left=random_tree(max_depth - 1, force_eml=False),
                       right=random_tree(max_depth - 1, force_eml=False))
    if max_depth <= 0 or random.random() < 0.3:
        return _random_leaf()
    return EMLNode('eml', left=random_tree(max_depth - 1),
                   right=random_tree(max_depth - 1))


# ── Мутации / кроссовер ───────────────────────────────────────────────────────

def _collect_nodes(node: EMLNode) -> list[EMLNode]:
    nodes = [node]
    if node.kind == 'eml':
        nodes.extend(_collect_nodes(node.left))
        nodes.extend(_collect_nodes(node.right))
    return nodes


def mutate_tree(root: EMLNode, intensity: float = 1.0,
                max_depth: int | None = None) -> EMLNode:
    """
    Мутация с адаптивной интенсивностью (1.0 = дикая, 0.1 = тонкая).

    max_depth=None — предел из config.EML_MAX_DEPTH (ветки A и B). Ветка C
    держит одно дерево вместо N_CLASSES и передаёт свой, больший предел.
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
    """Обрезать дерево до max_depth — глубокие поддеревья становятся листами."""
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
    """Заменить случайное поддерево parent1 поддеревом из parent2."""
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


# ── Классификатор на формулах ─────────────────────────────────────────────────

class EMLClassifier:
    """
    Набор из N_CLASSES формул: предсказание = argmax_c (f_c(inputs) - mu_c)/sd_c.

    Ровно этот объект и есть конечный продукт дистилляции. Никакого torch
    внутри — только numpy, а в export.py то же самое переводится в чистый C.

    ЗАЧЕМ КАЛИБРОВКА (mu_c, sd_c). Это не косметика, а необходимое условие
    работоспособности. В Тетрисе была одна формула, и argmax брался по разным
    ВХОДАМ — общий масштаб выхода сокращался и роли не играл. Здесь argmax
    берётся между ЧЕТЫРЬМЯ РАЗНЫМИ формулами, каждая из которых эволюционировала
    независимо и имеет свой произвольный масштаб и сдвиг. Без выравнивания
    побеждает не самая уместная формула, а та, что просто выдаёт числа покрупнее:
    в первом же прогоне класс MOTO не был предсказан ни разу.

    Константы берутся как среднее и с.к.о. выхода формулы на TRAIN и дальше
    заморожены. На микроконтроллере это 4 вычитания и 4 деления — цена, которой
    можно пренебречь.
    """

    def __init__(self, trees: list[EMLNode], offsets=None, scales=None):
        self.trees = list(trees)
        n = len(self.trees)
        self.offsets = (np.zeros(n) if offsets is None
                        else np.asarray(offsets, dtype=np.float64))
        self.scales = (np.ones(n) if scales is None
                       else np.asarray(scales, dtype=np.float64))

    # ── Вычисление ────────────────────────────────────────────────────────
    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Некалиброванные выходы формул (N, n_classes)."""
        out = np.empty((len(X), len(self.trees)), dtype=np.float64)
        for c, tree in enumerate(self.trees):
            vals = tree.evaluate_batch(X)
            # Нефинитное значение = формула сломалась на этом примере; такой
            # класс просто не должен выиграть argmax.
            out[:, c] = np.where(np.isfinite(vals), vals, -1e18)
        return out

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Калиброванные оценки (N, n_classes) — по ним и берётся argmax."""
        return (self.raw_scores(X) - self.offsets) / self.scales

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.scores(X), axis=1)

    def predict_one(self, inputs) -> int:
        """Инференс для одного вектора — эталон для проверки C-экспорта."""
        vals = np.array([t.evaluate(inputs) for t in self.trees],
                        dtype=np.float64)
        return int(np.argmax((vals - self.offsets) / self.scales))

    # ── Калибровка ────────────────────────────────────────────────────────
    def fit_calibration(self, X: np.ndarray) -> "EMLClassifier":
        """
        Подобрать mu_c и sd_c по выборке X (только train!) и заморозить их.

        Вырожденная формула (константа, sd ~ 0) получает sd = 1: делить на ноль
        нельзя, а её вклад в argmax всё равно постоянный.
        """
        raw = self.raw_scores(X)
        self.offsets, self.scales = _calibration_from_raw(raw)
        return self

    def calibration(self) -> dict:
        return {'offsets': self.offsets.tolist(), 'scales': self.scales.tolist()}

    # ── Метрики сложности (для таблицы в отчёте) ──────────────────────────
    def total_nodes(self) -> int:
        return sum(t.size() for t in self.trees)

    def max_depth(self) -> int:
        return max(t.depth() for t in self.trees)

    def used_inputs(self) -> list[int]:
        used = set()
        for t in self.trees:
            used.update(t.used_vars())
        return sorted(used)


# ── Метрики ───────────────────────────────────────────────────────────────────

def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                      n_classes: int | None = None) -> float:
    """
    Среднее по классам recall.

    Обычная accuracy при дисбалансе 3.4:1 награждает вырожденное решение
    «всё это ROADS» (0.41 из воздуха). Сбалансированная — не награждает,
    поэтому эволюция ведётся именно по ней.
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
    Смещения и масштабы из матрицы сырых выходов формул (N, n_classes).

    Считается один раз по train и дальше не меняется — калибровать по val или
    test значило бы подглядывать в отложенную выборку.
    """
    offsets = raw.mean(axis=0)
    scales = raw.std(axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)   # вырожденная формула
    return offsets, scales


def _standardize_rows(logits: np.ndarray) -> np.ndarray:
    """
    z-score логитов внутри каждого примера (по классам).

    Сохраняет argmax — то есть всю информацию о решении оракула — и убирает
    произвольный масштаб, из-за которого MSE был бы несопоставим между
    примерами.
    """
    mean = logits.mean(axis=1, keepdims=True)
    std = logits.std(axis=1, keepdims=True)
    return (logits - mean) / (std + 1e-8)


# ── DATA-фаза: регрессия одной формулы к логиту одного класса ─────────────────

def _regression_fitness(tree: EMLNode, X: np.ndarray, y: np.ndarray,
                        w: np.ndarray, depth_penalty: float) -> float:
    """
    Fitness = var_bonus / (взвешенный MSE + штраф за сложность).

    Штраф считается и по глубине, и по ЧИСЛУ УЗЛОВ. Одной глубины мало: она
    ограничена сверху EML_MAX_DEPTH, поэтому внутри лимита дерево может расти
    вширь бесплатно (см. комментарий к EML_SIZE_PENALTY_WEIGHT в config.py —
    там замер, из-за которого этот член и появился).
    """
    n_vars = tree.n_unique_vars()
    if n_vars == 0:
        return 1e-6                       # дерево-константа бесполезно
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
    GA-эволюция ОДНОЙ формулы под символьную регрессию к таргету y.

    Args:
        progress_cb: callable(dict) — вызывается каждое поколение. Через него
            интерфейс рисует кривую fitness по ходу дела, а не через полчаса.
        should_stop: callable() -> bool — опрашивается каждое поколение.
            Прерывание возвращает ЛУЧШЕЕ найденное дерево, а не выбрасывает
            работу: остановка должна останавливать, а не отменять.

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

        # Свежая подвыборка каждое поколение: быстрее и меньше переобучения
        # под конкретные примеры.
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
                print(f"      [{label}] остановлено пользователем на gen {gen}")
            break

        if patience > 0 and stall >= patience:
            if verbose:
                print(f"      [{label}] ранняя остановка на gen {gen}")
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


# ── ACC-фаза: эволюция всего набора формул по реальной accuracy ────────────────

def _ensemble_fitness(trees: list[EMLNode], X: np.ndarray, y: np.ndarray,
                      depth_penalty: float) -> float:
    """
    Fitness набора = сбалансированная accuracy минус штраф за размер.

    Калибровка подбирается прямо здесь, на той же подвыборке: набор формул и
    его константы — единый объект, и оценивать деревья с чужой калибровкой
    бессмысленно. Считается она из уже вычисленных выходов, то есть даром.

    Штраф за размер мягкий и нормирован на число узлов: без него эволюция
    раздувает деревья ради долей процента точности, а нам нужна формула,
    которая влезает в микроконтроллер и читается человеком.
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
    Эволюция НАБОРА формул с fitness = сбалансированная accuracy.

    Особь = список из N_CLASSES деревьев. Мутация меняет одно случайное дерево
    набора (менять все сразу — слишком грубый шаг, набор разваливается),
    кроссовер собирает потомка из формул двух родителей поклассово.

    X, y — ТОЛЬКО train-подвыборка (см. пояснение в шапке модуля).
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

        # Одна и та же подвыборка на всё поколение — иначе особи сравниваются
        # на разных данных и отбор превращается в лотерею.
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
            # Точность считается через ту же калибровку, что и fitness.
            # В первой версии здесь создавался EMLClassifier БЕЗ калибровки, и
            # лог показывал 0.25 (вырожденное предсказание) в то время, как
            # fitness рос — наглядная иллюстрация того, зачем калибровка нужна.
            top = EMLClassifier(population[0])
            raw = top.raw_scores(Xs)
            off, sc = _calibration_from_raw(raw)
            acc = balanced_accuracy(ys, np.argmax((raw - off) / sc, axis=1))
            print(f"    [ACC] gen {gen:>3} | fit {best:.4f} | bal_acc {acc:.4f} | "
                  f"узлов {top.total_nodes()}")

        if progress_cb is not None:
            best_set = best_trees_ever or population[0]
            progress_cb({'phase': 'acc', 'gen': gen, 'generations': generations,
                         'best': best, 'history': list(history),
                         'nodes': sum(t.size() for t in best_set)})

        if should_stop is not None and should_stop():
            stopped = True
            if verbose:
                print(f"    [ACC] остановлено пользователем на gen {gen}")
            break

        if patience > 0 and stall >= patience:
            if verbose:
                print(f"    [ACC] ранняя остановка на gen {gen}")
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


# ── Полный пайплайн дистилляции ───────────────────────────────────────────────

def distill(X_train: np.ndarray, logits_train: np.ndarray, y_train: np.ndarray,
            *, weights: np.ndarray | None = None,
            depth_penalty_name: str | None = None,
            data_generations: int | None = None,
            data_population: int | None = None,
            acc_phase: bool | None = None,
            verbose: bool = True, rng_seed: int = 0,
            progress_cb=None, should_stop=None) -> dict:
    """
    DATA-фаза (регрессия к логитам оракула) -> ACC-фаза (эволюция по accuracy).

    Args:
        X_train:      (N, n_inputs) — признаки или эмбеддинги.
        logits_train: (N, n_classes) — логиты оракула на тех же примерах.
        y_train:      (N,) — истинные метки (нужны только ACC-фазе).
        weights:      (N,) веса примеров; None -> веса классов из data.py.
        progress_cb:  callable(dict) — прогресс обеих фаз для интерфейса.
        should_stop:  callable() -> bool — мягкая остановка. Результат остаётся
            согласованным: набор формул полный, калибровка посчитана, метрики
            честные. Просто эволюции было меньше, и это помечено флагом
            'stopped' — чтобы такой прогон не выдавали за полноценный.

    Returns: dict с 'trees' (итоговый набор), метриками обеих фаз и историей.
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
        # Веса классов: без них редкий BEACH тонет в MSE и его формула
        # вырождается в константу.
        import data as data_mod
        cw = data_mod.class_weights(y_train)
        weights = cw[y_train].astype(np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    targets = _standardize_rows(np.asarray(logits_train, dtype=np.float64))
    t0 = time.time()

    # ── DATA-фаза: по формуле на класс ───────────────────────────────────────
    if verbose:
        print(f"  [DATA] символьная регрессия к логитам оракула "
              f"({len(X):,} примеров, {_N_INPUTS} входов)")
    data_trees, data_results = [], []
    stopped = False
    for c in range(n_classes):
        name = config.CLASS_NAMES[c] if c < len(config.CLASS_NAMES) else f'C{c}'
        # Индекс класса подмешивается в payload здесь, а не в evolve_formula:
        # та ничего не знает про классы, она эволюционирует одну формулу.
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
        """Заморозить калибровку по полному train и померить точность."""
        clf = EMLClassifier(trees).fit_calibration(X)
        return clf, balanced_accuracy(y_train, clf.predict(X))

    data_clf, data_acc = _fit_and_score(data_trees)
    if verbose:
        print(f"  DATA-фаза: сбалансированная accuracy на train = {data_acc:.4f}")

    # ── ACC-фаза ─────────────────────────────────────────────────────────────
    acc_result = None
    final_trees, final_clf, final_acc = data_trees, data_clf, data_acc
    if acc_phase and not stopped:
        if verbose:
            print("  [ACC] эволюция набора формул по реальной accuracy...")
        acc_result = evolve_ensemble(data_trees, X, y_train,
                                     depth_penalty=dp, verbose=verbose,
                                     rng_seed=rng_seed, progress_cb=progress_cb,
                                     should_stop=should_stop)
        stopped = stopped or acc_result.get('stopped', False)
        candidate_trees = acc_result['best_trees']
        candidate_clf, candidate_acc = _fit_and_score(candidate_trees)
        # ACC-фаза оценивалась на подвыборках; на полном train результат может
        # оказаться хуже исходного. Тогда откатываемся — ухудшать нельзя.
        if candidate_acc >= data_acc:
            final_trees, final_clf, final_acc = (candidate_trees, candidate_clf,
                                                 candidate_acc)
        elif verbose:
            print(f"  ACC ({candidate_acc:.4f}) < DATA ({data_acc:.4f}) "
                  f"-> оставляем набор DATA-фазы")

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
