"""
eml_single.py — ветка C: ОДНА EML-формула на все классы.

Ветки A и B держат по формуле на класс и берут argmax. Главный технический
вывод подпроекта состоял в том, что метод спотыкается не о выразительность
формул, а о СОПОСТАВИМОСТЬ ИХ ШКАЛ: формулы эволюционируют независимо, масштаб
у каждой свой, и без калибровки (mu_c, sd_c) argmax выигрывает не самая
уместная формула, а та, что выдаёт числа покрупнее — в первом прогоне ветки A
класс MOTO не предсказывался ни разу.

Ветка C убирает этот дефект по построению. Формула одна:

    f(признаки) -> v          класс = labels[searchsorted(thresholds, v)]

Сравнивать нечего, калибровочных констант нет, на устройстве лежит одно дерево
и таблица из нескольких порогов.

ТАБЛИЦА НЕ ЗАДАНА ЗАРАНЕЕ — И ЭТО ГЛАВНОЕ. Напрашивающийся вариант «пусть
класс 0 это 0.0, класс 1 это 0.33, класс 2 это 0.6, класс 3 это 1.0» плох
дважды. Во-первых, он навязывает классам порядок и расстояния, которых у них
нет: BEACH, MOTO, PEOPLE и ROADS не упорядочены, и регрессия к такому коду
штрафует «дальние» ошибки сильнее «ближних» без всяких оснований. Во-вторых,
малый MSE к коду и правильный класс — разные вещи (ровно тот аргумент, из-за
которого в ветке A появилась ACC-фаза).

Поэтому таблица ПОДБИРАЕТСЯ ПОД ФОРМУЛУ на каждой оценке fitness — точным
динамическим программированием, максимизирующим сбалансированную accuracy.
Формула сама решает, в каком порядке и на каком расстоянии расположить классы;
куда она их в итоге расставила — результат прогона, а не наше допущение
(см. IntervalDecoder.describe).

Участков разрешено больше, чем классов, и МЕТКИ УЧАСТКОВ МОГУТ ПОВТОРЯТЬСЯ.
Это тот же эффект, ради которого напрашивалась периодическая функция
активации: класс занимает несколько зон на оси, и линейный порядок классов
перестаёт быть ограничением. Но без самой периодичности: выход EML-дерева
лежит примерно в [-10, +2.2e4] (exp(clamp(x, ±10)) плюс логарифм), синус на
таком аргументе это тысячи периодов, ландшафт fitness становится разрывным —
микроправка дерева перебрасывает половину примеров в другой класс, и GA по
такому ландшафту не идёт, а блуждает. Таблица порогов даёт ту же
выразительность, точный подбор и K сравнений на микроконтроллере.

ЧЕМУ УЧИМСЯ. Целью fitness служат ОТВЕТЫ ОРАКУЛА, а не истинные метки: это
по-прежнему дистилляция, как этапы 1 и 2, и формула воспроизводит именно
решающую функцию сети. Истинные метки используются только для отчётных метрик.
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


# ── Декодер: таблица порогов на оси выхода формулы ────────────────────────────

class IntervalDecoder:
    """
    Разбиение оси выхода формулы на участки с метками классов.

    thresholds — возрастающий массив длины K-1, labels — K меток. Метки могут
    повторяться: класс имеет право занимать несколько несмежных участков.
    """

    __slots__ = ('thresholds', 'labels', 'n_classes', 'score')

    def __init__(self, thresholds, labels, n_classes: int,
                 score: float = float('nan')):
        self.thresholds = np.asarray(thresholds, dtype=np.float64).ravel()
        self.labels = np.asarray(labels, dtype=np.int64).ravel()
        self.n_classes = int(n_classes)
        self.score = float(score)
        if len(self.labels) != len(self.thresholds) + 1:
            raise ValueError(f"меток {len(self.labels)}, порогов "
                             f"{len(self.thresholds)}: должно быть на 1 больше")

    @property
    def n_intervals(self) -> int:
        return len(self.labels)

    def decode(self, values: np.ndarray) -> np.ndarray:
        """
        Вектор выходов формулы -> вектор классов.

        Бесконечности проходят как обычные очень большие числа — у них есть
        законное место на оси. А вот NaN места на оси не имеет (формула
        сломалась на этом примере), и он отправляется в самый левый участок:
        предсказание обязано остаться определённым, иначе метрика посчитается
        по битому массиву.
        """
        values = np.asarray(values, dtype=np.float64)
        safe = np.where(np.isnan(values), -np.inf, values)
        return self.labels[np.searchsorted(self.thresholds, safe, side='right')]

    def decode_one(self, value: float) -> int:
        """Инференс для одного числа — эталон для проверки C-экспорта."""
        if math.isnan(value):
            return int(self.labels[0])
        idx = 0
        while idx < len(self.thresholds) and value > self.thresholds[idx]:
            idx += 1
        return int(self.labels[idx])

    def class_intervals(self, class_idx: int) -> list[tuple[float, float]]:
        """Участки оси, отданные классу, как список пар (левый, правый)."""
        edges = np.concatenate(([-np.inf], self.thresholds, [np.inf]))
        return [(float(edges[i]), float(edges[i + 1]))
                for i in range(self.n_intervals) if self.labels[i] == class_idx]

    def describe(self, class_names: list[str] | None = None) -> str:
        """
        Куда формула расставила классы на своей оси.

        Это не отладочная печать, а результат: порядок и расстояния между
        классами формула выбирала сама, и их стоит сверять с матрицей ошибок
        оракула — соседние на оси классы должны путаться чаще прочих.
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
    Веса примеров, при которых сумма весов угаданных = сбалансированная accuracy.

    Сбалансированная accuracy — это среднее по классам recall. Если дать
    примеру класса c вес 1/(число классов в выборке * размер класса c), то
    суммирование весов правильно предсказанных примеров даёт ровно её. Дальше
    вся оптимизация декодера сводится к «набрать как можно больше веса», и
    динамическому программированию не нужно знать про классы ничего сверх
    этого.

    Классы, которых в выборке нет, из усреднения выпадают — так же, как в
    eml_distiller.balanced_accuracy.
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
    Подобрать таблицу порогов под уже посчитанные выходы формулы.

    Задача решается ТОЧНО, а не приближённо: разбиение оси на K участков с
    метками — это классическое ДП по отсортированным значениям. Приближённый
    подбор порогов (жадный, по средним классов) в таком месте опасен: fitness
    всей эволюции равен качеству этого подбора, и его случайные провалы GA
    воспринимал бы как «дерево стало хуже».

    Ось режется не по значениям, а по КВАНТИЛЯМ (bins штук): выход EML-дерева
    распределён крайне неравномерно (exp() в каждом узле), и равномерная сетка
    отдала бы почти все пороги пустому хвосту. Границы бинов сдвигаются за
    конец серии одинаковых значений — порог посередине такой серии всё равно
    нельзя реализовать.

    Число участков выбирается вместе с самим разбиением: за каждый участок
    сверх числа классов взимается interval_penalty. Иначе K всегда упиралось бы
    в потолок — лишний участок никогда не ухудшает обучающую метрику.
    """
    max_intervals = max_intervals or config.EMLC_MAX_INTERVALS
    bins = bins or config.EMLC_DECODER_BINS
    if interval_penalty is None:
        interval_penalty = config.EMLC_INTERVAL_PENALTY

    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if weights is None:
        weights = sample_weights(y, n_classes)

    # NaN не имеет места на оси — отправляется в самый левый участок, как и в
    # decode(); бесконечности сортируются сами и остаются на своих краях.
    clean = np.where(np.isnan(values), -np.inf, values)
    order = np.argsort(clean, kind='stable')
    sv, sy, sw = clean[order], y[order], np.asarray(weights)[order]
    n = len(sv)

    # ── Бины по квантилям, границы — только между разными значениями ──────────
    n_bins = max(1, min(bins, n))
    ranks = np.linspace(0, n, n_bins + 1).round().astype(np.int64)[1:-1]
    if len(ranks):
        ranks = np.searchsorted(sv, sv[np.clip(ranks, 0, n - 1)], side='right')
    bounds = np.unique(np.concatenate(([0], ranks, [n])))
    n_bins = len(bounds) - 1

    # Вес каждого класса в каждом бине и его накопленная сумма по бинам.
    bin_of = np.repeat(np.arange(n_bins), np.diff(bounds))
    per_bin = np.bincount(bin_of * n_classes + sy, weights=sw,
                          minlength=n_bins * n_classes).reshape(n_bins, n_classes)
    prefix = np.vstack([np.zeros((1, n_classes)), np.cumsum(per_bin, axis=0)])

    # gain[a, b] — сколько веса наберёт участок из бинов [a, b), если отдать
    # его лучшему для себя классу. Разом для всех пар: (nb+1, nb+1, C) -> max.
    gain = (prefix[None, :, :] - prefix[:, None, :]).max(axis=2)
    idx = np.arange(n_bins + 1)
    valid = idx[None, :] > idx[:, None]

    # ── ДП: лучшее покрытие первых b бинов ровно k участками ─────────────────
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

    # ── Восстановление участков ──────────────────────────────────────────────
    segments, b = [], n_bins
    for k in range(k_best, 0, -1):
        a = int(pointers[k - 1][b])
        label = int((prefix[b] - prefix[a]).argmax())
        segments.append((a, b, label))
        b = a
    segments.reverse()

    # Соседние участки с одинаковой меткой сливаются: лишний порог ничего не
    # решает, но стоит и штрафа, и сравнения на микроконтроллере.
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
        # Слияние участков меняет ответы, поэтому обучающая метрика считается
        # заново: отдать наверх счёт до слияния значило бы завысить fitness.
        decoder.score = float(sw[decoder.decode(sv) == sy].sum())
    return decoder


def _make_float32_safe(thresholds: list[float], labels: list[int],
                       weights: list[float]):
    """
    Убрать пороги, неразличимые во float32, слив соответствующие участки.

    Выход EML-дерева доходит до 2.2e4 (exp(clamp(x, ±10))), а шаг float32 там
    уже ~0.002 — два соседних порога в этой области легко схлопываются в одно
    число. На микроконтроллере такой участок просто исчез бы, и класс перестал
    бы предсказываться; расхождение Python и C при этом выглядело бы как
    поломка экспорта, хотя экспорт исправен.

    Дешевле не допускать неразличимых порогов вовсе: тогда таблица одинаково
    работает во float64 и во float32 по построению. При слиянии выживает метка
    более «тяжёлого» участка — терять больший кусок выборки ради меньшего
    смысла нет.
    """
    if not thresholds:
        return thresholds, labels, False

    out_t: list[float] = []
    out_l: list[int] = [labels[0]]
    out_w: list[float] = [weights[0]]
    changed = False
    for threshold, label, weight in zip(thresholds, labels[1:], weights[1:]):
        collides = bool(out_t) and np.float32(threshold) <= np.float32(out_t[-1])
        # Слияние могло оставить рядом два участка одного класса — порог между
        # ними ничего не решает, но стоит сравнения на устройстве.
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


# ── Модель ветки C ────────────────────────────────────────────────────────────

class EMLSingle:
    """
    Итоговый продукт ветки C: одно EML-дерево плюс таблица порогов.

    Интерфейс намеренно совпадает с EMLClassifier ветки A (predict,
    predict_one, total_nodes, max_depth, used_inputs), чтобы pipeline, отчёты
    и вкладки теста работали с обеими ветками без развилок.
    """

    def __init__(self, tree: EMLNode, decoder: IntervalDecoder):
        self.tree = tree
        self.decoder = decoder

    # ── Вычисление ────────────────────────────────────────────────────────
    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.tree.evaluate_batch(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decoder.decode(self.raw(X))

    def predict_one(self, inputs) -> int:
        return self.decoder.decode_one(self.tree.evaluate(inputs))

    def class_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Оценка каждого класса (N, n_classes) — для вкладок теста.

        У одной формулы нет отдельного числа на класс, поэтому оценка = минус
        расстояние от выхода до ближайшего участка этого класса: 0 у
        предсказанного класса, отрицательное у остальных, и чем дальше по оси,
        тем меньше. Это НЕ вероятность и не логит; аккуратнее всего читать её
        как «насколько далеко пришлось бы сдвинуть выход, чтобы получить
        этот класс».
        """
        values = self.raw(X)
        values = np.where(np.isfinite(values), values, 0.0)
        out = np.full((len(values), self.decoder.n_classes), -np.inf)
        for c in range(self.decoder.n_classes):
            for lo, hi in self.decoder.class_intervals(c):
                dist = np.maximum(np.maximum(lo - values, values - hi), 0.0)
                out[:, c] = np.maximum(out[:, c], -dist)
        return out

    # ── Метрики сложности ─────────────────────────────────────────────────
    def total_nodes(self) -> int:
        return self.tree.size()

    def max_depth(self) -> int:
        return self.tree.depth()

    def used_inputs(self) -> list[int]:
        return self.tree.used_vars()

    # ── Сериализация ──────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {'tree': self.tree.to_dict(), 'decoder': self.decoder.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "EMLSingle":
        return cls(EMLNode.from_dict(d['tree']),
                   IntervalDecoder.from_dict(d['decoder']))

    def refit_decoder(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "EMLSingle":
        """Заморозить таблицу порогов по полной обучающей выборке."""
        kwargs.setdefault('bins', config.EMLC_FINAL_BINS)
        self.decoder = fit_decoder(self.raw(X), y, self.decoder.n_classes,
                                   **kwargs)
        return self


# ── Fitness ───────────────────────────────────────────────────────────────────

class FitnessParams:
    """Всё, что нужно оценке одной формулы, одним объектом (удобно слать в процессы)."""

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
    Оценить одно дерево: подобрать под него декодер и посчитать fitness.

    fitness = сбалансированная accuracy − штраф за узлы − штраф за участки.
    Оба штрафа вычитаются, а не делят: тогда шкала fitness совпадает со шкалой
    метрики и «-0.05 за сотню узлов» читается прямо из константы.
    """
    if tree.n_unique_vars() == 0:
        # Дерево-константа даёт ровно 1/n_classes (декодер вынужден отдать всю
        # ось одному классу) — формально это не ноль, и без явного штрафа такие
        # деревья заполняют популяцию как дешёвые.
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
    """Новое поколение: элита без изменений, остальные — мутация/кроссовер/новичок."""
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
    Прогнать одну популяцию `generations` поколений.

    progress — доля уже пройденного пути всего прогона (0..1); от неё зависит
    интенсивность мутаций: сначала дикие, к концу тонкие.

    Подвыборка меняется каждое поколение (быстрее и меньше подгонки под
    конкретные примеры), но ВНУТРИ поколения она общая: иначе особи
    сравнивались бы на разных данных и отбор превратился бы в лотерею.
    """
    rng = np.random.default_rng(rng_seed)
    random.seed(rng_seed)
    population = [t.clone() for t in trees]
    best_fitness, best_tree, best_info = -math.inf, None, None
    history = []

    # Интенсивность мутаций падает по ходу ВСЕГО прогона, а не внутри раунда:
    # раунд это десятки поколений из тысяч, и своей шкалы «от дикой к тонкой»
    # у него нет. К последней трети прогона мутации становятся точечными.
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


# ── Параллель по процессам ────────────────────────────────────────────────────
#
# Параллелится ОСТРОВ ЦЕЛИКОМ на несколько поколений, а не отдельные оценки
# fitness. Оценка одного дерева занимает доли миллисекунды, и раскладывать её
# по процессам значило бы платить за межпроцессную передачу больше, чем за сам
# счёт. Остров же уезжает в процесс один раз на десятки поколений.

_WORKER: dict = {}


def _worker_init(X, y, n_inputs, input_names):
    """Данные кладутся в процесс один раз при его создании, а не на каждый вызов."""
    import eml_distiller
    eml_distiller.configure_inputs(n_inputs, input_names)
    _WORKER['X'] = X
    _WORKER['y'] = y


def _worker_round(payload: dict) -> dict:
    """Один раунд одного острова внутри рабочего процесса."""
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


# ── Чекпоинт ──────────────────────────────────────────────────────────────────

def save_checkpoint(path: str, state: dict):
    """
    Записать состояние прогона атомарно (через .tmp + замена).

    Многочасовой прогон переживает перезагрузку только так: запись прямо в
    целевой файл означала бы, что выключение питания посреди неё оставляет
    обрезанный JSON, то есть губит ровно то, что он должен был спасти.
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


# ── Полная эволюция ───────────────────────────────────────────────────────────

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
    Островная эволюция одной формулы.

    Острова — несколько независимых популяций, изредка обменивающихся лучшими
    особями. Смысл не в скорости (её даёт параллель по процессам), а в
    РАЗНООБРАЗИИ: разброс GA по сидам в ветке A составил 0.525-0.623, то есть
    одна длинная цепочка эволюции говорит о методе примерно ничего. Несколько
    цепочек с редкой миграцией дают и запас разнообразия, и возможность
    сравнить, к одному ли решению они сходятся.

    ГЛОБАЛЬНЫЙ ЛУЧШИЙ ВЫБИРАЕТСЯ НЕ ПО FITNESS ОСТРОВОВ. Внутри острова
    fitness считается на своей случайной подвыборке, и числа с разных островов
    несопоставимы — выбирать по ним значило бы выбирать по шуму подвыборки.
    Поэтому кандидаты от всех островов переоцениваются в главном процессе на
    ПОЛНОЙ обучающей выборке, одинаковой для всех.

    Прогон переживает остановку: состояние всех островов пишется в
    checkpoint_path, и resume=True продолжает с последнего сохранения.

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

    # ── Старт или продолжение ────────────────────────────────────────────────
    state = load_checkpoint(checkpoint_path) if (resume and checkpoint_path) else None
    if state:
        pops = [[EMLNode.from_dict(d) for d in island] for island in state['islands']]
        start_gen = int(state['generation'])
        history = list(state['history'])
        best_fitness = float(state['best_fitness'])
        best_tree = EMLNode.from_dict(state['best_tree'])
        stall = int(state.get('stall', 0))
        if verbose:
            print(f"  [C] продолжаю с поколения {start_gen} "
                  f"({len(pops)} островов, лучший {best_fitness:.4f})")
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
        how = (f"параллельно на {min(workers, islands)} процессах" if pool
               else "в один процесс")
        print(f"  [C] {islands} островов x {population} особей, {how}")

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

            # Переоценка кандидатов на ПОЛНОЙ выборке — единственная общая
            # линейка. Стоит нескольких вычислений дерева на весь раунд.
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
                print(f"    [C] gen {gen:>5}/{generations} | лучший "
                      f"{best_fitness:.4f} | раунд {best_info['fitness']:.4f} | "
                      f"разброс островов {spread:.4f} | узлов "
                      f"{best_info['nodes']} | участков {best_info['intervals']}")

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
                    'generation': gen, 'history': history,
                    'best_fitness': best_fitness,
                    'best_tree': best_tree.to_dict(),
                    'stall': stall,
                    'islands': [[t.to_dict() for t in p] for p in pops],
                })

            if should_stop is not None and should_stop():
                stopped = True
                if verbose:
                    print(f"    [C] остановлено пользователем на поколении {gen}")
                break

            if patience > 0 and stall >= patience:
                if verbose:
                    print(f"    [C] ранняя остановка: {stall} поколений "
                          f"без улучшения")
                break

            if islands > 1 and migrants > 0:
                _migrate(pops, migrants)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Итоговый декодер замораживается по ПОЛНОЙ обучающей выборке — так же,
    # как калибровка ветки A. Подбирать его по val значило бы подглядывать.
    # Сетка порогов здесь мельче, чем в fitness: считается это один раз, а
    # грубая сетка стоила бы процента точности на пустом месте.
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
    Лучшие особи каждого острова заменяют худших на соседнем (по кольцу).

    Копии отправляются, а не переносятся: остров не должен терять свою элиту,
    иначе миграция превращается в перемешивание и разнообразие, ради которого
    острова и заведены, исчезает за несколько обменов.
    """
    donors = [[t.clone() for t in pop[:migrants]] for pop in pops]
    for i, pop in enumerate(pops):
        incoming = donors[(i - 1) % len(pops)]
        for j, tree in enumerate(incoming):
            pop[-(j + 1)] = tree
