"""
Тесты AST EML, классификатора на формулах и метрик.

Формулы — конечный артефакт проекта: их сохраняют в JSON, переводят в C и
увозят на микроконтроллер. Поэтому проверяется именно то, что при поломке
даёт не ошибку, а тихо неверный ответ: сериализация туда-обратно, совпадение
скалярного и векторного вычислений, калибровка и сбалансированная accuracy.
"""

import math
import random

import numpy as np
import pytest

import config
import eml_distiller as dist


@pytest.fixture(autouse=True)
def inputs_configured():
    """Все тесты работают в пространстве входов ветки A (32 признака)."""
    dist.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))
    yield
    dist.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))


def random_trees(n: int, seed: int = 0, max_depth: int = 4):
    random.seed(seed)
    return [dist.random_tree(max_depth=max_depth, force_eml=True)
            for _ in range(n)]


def random_X(n: int = 64, seed: int = 0) -> np.ndarray:
    """Входы в том же диапазоне, что настоящие признаки: [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, config.N_FEATURES))


# ── Оператор ──────────────────────────────────────────────────────────────────

def test_eml_op_matches_definition():
    assert dist.eml_op(0.5, 2.0) == pytest.approx(
        math.exp(0.5) - math.log(2.0 + config.EML_EPSILON))


def test_eml_op_clamps_before_exp():
    """
    Без clamp перед exp() формула улетает в inf на первом же выбросе —
    правило из update.md. Проверяем, что предел действительно стоит.
    """
    assert math.isfinite(dist.eml_op(1e9, 1.0))
    assert dist.eml_op(1e9, 1.0) == pytest.approx(dist.eml_op(10.0, 1.0))
    assert dist.eml_op(-1e9, 1.0) == pytest.approx(dist.eml_op(-10.0, 1.0))


def test_eml_op_survives_zero():
    """ln(0) защищён эпсилоном, иначе ноль в правой ветви ронял бы формулу."""
    assert math.isfinite(dist.eml_op(0.0, 0.0))


# ── Сериализация ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize('seed', range(5))
def test_tree_dict_roundtrip(seed):
    """
    to_dict/from_dict возвращают ТО ЖЕ дерево.

    Это формат хранения конечного артефакта: если он теряет узел или путает
    ветви, сохранённые формулы будут считать не то, что показывал отчёт.
    """
    X = random_X(seed=seed)
    for tree in random_trees(6, seed=seed):
        clone = dist.EMLNode.from_dict(tree.to_dict())
        assert clone.to_string() == tree.to_string()
        assert clone.size() == tree.size()
        assert clone.depth() == tree.depth()
        assert np.allclose(clone.evaluate_batch(X), tree.evaluate_batch(X))


def test_from_dict_rejects_unknown_node():
    with pytest.raises(ValueError):
        dist.EMLNode.from_dict({'kind': 'sin', 'left': {'kind': 'const',
                                                        'value': 1.0}})


def test_clone_is_deep():
    """Мутация клона не должна задевать оригинал — на этом стоит вся эволюция."""
    tree = dist.EMLNode('eml',
                        left=dist.EMLNode('var', var_idx=3),
                        right=dist.EMLNode('const', value=1.0))
    clone = tree.clone()
    clone.left.var_idx = 7
    clone.right.value = -5.0
    assert tree.left.var_idx == 3
    assert tree.right.value == 1.0


# ── Скалярное против векторного ───────────────────────────────────────────────

@pytest.mark.parametrize('seed', range(5))
def test_evaluate_matches_evaluate_batch(seed):
    """
    evaluate (по одному вектору) и evaluate_batch (по массиву) обязаны
    совпадать: первым проверяется C-экспорт, вторым считаются все метрики.
    Разъедутся — отчёт и артефакт будут говорить разное.
    """
    X = random_X(seed=seed)
    for tree in random_trees(6, seed=seed):
        batch = tree.evaluate_batch(X)
        one = np.array([tree.evaluate(row) for row in X])
        assert np.allclose(one, batch, rtol=1e-9, atol=1e-9)


def test_evaluate_batch_is_finite_on_extremes():
    """Даже на входах вне [-1, 1] результат конечен: клипы стоят на каждом узле."""
    X = np.full((4, config.N_FEATURES), 1e6)
    X[1] = -1e6
    for tree in random_trees(10, seed=3):
        assert np.all(np.isfinite(tree.evaluate_batch(X)))


# ── Метрики дерева ────────────────────────────────────────────────────────────

def test_size_depth_and_vars():
    tree = dist.EMLNode(
        'eml',
        left=dist.EMLNode('var', var_idx=2),
        right=dist.EMLNode('eml',
                           left=dist.EMLNode('var', var_idx=2),
                           right=dist.EMLNode('const', value=0.5)))
    assert tree.size() == 5
    assert tree.depth() == 2
    assert tree.n_unique_vars() == 1          # var_idx=2 дважды, но переменная одна
    assert tree.used_vars() == [2]


def test_prune_respects_max_depth():
    deep = dist.random_tree(max_depth=config.EML_MAX_DEPTH + 4, force_eml=True)
    pruned = dist._prune_tree(deep, max_depth=3)
    assert pruned.depth() <= 3


# ── Классификатор и калибровка ────────────────────────────────────────────────

def test_degenerate_calibration_gets_unit_scale():
    """
    Формула-константа имеет нулевой разброс; делить на ноль нельзя.

    Такая формула получает масштаб 1 и просто не влияет на argmax — это
    корректное поведение, а не заглушка: её вклад постоянен для всех примеров.
    """
    trees = [dist.EMLNode('const', value=float(c))
             for c in range(config.N_CLASSES)]
    clf = dist.EMLClassifier(trees).fit_calibration(random_X())
    assert np.allclose(clf.scales, 1.0)
    assert np.allclose(clf.offsets, [0.0, 1.0, 2.0, 3.0])
    assert np.all(np.isfinite(clf.scores(random_X())))


def test_calibration_equalises_scales():
    """
    Ради чего калибровка и заведена: формула с крупными числами не должна
    выигрывать argmax просто из-за масштаба.

    Здесь у первой формулы выход в сто раз крупнее остальных. Без калибровки
    argmax всегда отдаёт класс 0; после — распределение уже не вырождено.
    """
    X = random_X(n=256, seed=1)
    trees = [dist.EMLNode('var', var_idx=i) for i in range(config.N_CLASSES)]
    huge = dist.EMLNode('eml', left=dist.EMLNode('const', value=5.0),
                        right=dist.EMLNode('var', var_idx=0))
    trees[0] = huge

    raw_pred = np.argmax(dist.EMLClassifier(trees).raw_scores(X), axis=1)
    assert set(raw_pred.tolist()) == {0}, "подготовка теста: без калибровки ждём вырождение"

    clf = dist.EMLClassifier(trees).fit_calibration(X)
    assert len(set(clf.predict(X).tolist())) > 1


def test_predict_one_matches_predict():
    """Поштучный инференс (эталон для C) совпадает с батчевым."""
    X = random_X(n=32, seed=5)
    clf = dist.EMLClassifier(random_trees(config.N_CLASSES, seed=5))
    clf.fit_calibration(X)
    batch = clf.predict(X)
    one = np.array([clf.predict_one(row) for row in X])
    assert np.array_equal(batch, one)


def test_non_finite_formula_never_wins_argmax():
    """Сломавшаяся на примере формула не должна выигрывать класс."""
    class Broken(dist.EMLNode):
        def evaluate_batch(self, X):
            return np.full(len(X), np.nan)

    trees = [Broken('const', value=0.0)] + [
        dist.EMLNode('var', var_idx=i) for i in range(1, config.N_CLASSES)]
    clf = dist.EMLClassifier(trees)
    X = random_X(n=16, seed=2)
    assert 0 not in set(clf.predict(X).tolist())


def test_classifier_complexity_metrics():
    trees = [dist.EMLNode('var', var_idx=1),
             dist.EMLNode('eml', left=dist.EMLNode('var', var_idx=5),
                          right=dist.EMLNode('const', value=1.0)),
             dist.EMLNode('const', value=2.0),
             dist.EMLNode('var', var_idx=1)]
    clf = dist.EMLClassifier(trees)
    assert clf.total_nodes() == 1 + 3 + 1 + 1
    assert clf.max_depth() == 1
    assert clf.used_inputs() == [1, 5]


# ── Метрики качества ──────────────────────────────────────────────────────────

def test_balanced_accuracy_on_handmade_confusion():
    """
    Считается вручную. Классы 2:2:2:6, угадано 2, 1, 0 и 5 примеров:

        recall = 1.0, 0.5, 0.0, 5/6  ->  сбалансированная 0.5833
        обычная = (2 + 1 + 0 + 5) / 12 = 0.6667

    Две метрики на одних и тех же данных дают разные числа — ради этого
    различия сбалансированная и выбрана главной.
    """
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    y_pred = np.array([0, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 0])
    assert dist.balanced_accuracy(y_true, y_pred) == pytest.approx(
        (1.0 + 0.5 + 0.0 + 5 / 6) / 4)
    assert float((y_true == y_pred).mean()) == pytest.approx(8 / 12)


def test_balanced_accuracy_punishes_majority_guessing():
    """
    Ради этого свойства метрика и выбрана главной: «всё это ROADS» на
    несбалансированном датасете даёт высокую обычную accuracy и ровно 1/K
    сбалансированной.
    """
    y_true = np.concatenate([np.zeros(10), np.ones(10),
                             np.full(10, 2), np.full(70, 3)]).astype(int)
    y_pred = np.full(len(y_true), 3)
    assert float((y_true == y_pred).mean()) == pytest.approx(0.7)
    assert dist.balanced_accuracy(y_true, y_pred) == pytest.approx(0.25)


def test_balanced_accuracy_skips_absent_classes():
    """Класс, которого нет в выборке, не тянет метрику вниз нулевым recall."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert dist.balanced_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_standardize_rows_preserves_argmax():
    """
    z-нормировка логитов внутри примера обязана сохранять решение оракула —
    иначе DATA-фаза училась бы регрессии не к тому.
    """
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, config.N_CLASSES)) * rng.uniform(
        0.1, 50, (200, 1))
    normalized = dist._standardize_rows(logits)
    assert np.array_equal(logits.argmax(axis=1), normalized.argmax(axis=1))
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-6)


# ── Эволюция (короткий прогон) ────────────────────────────────────────────────

def test_evolve_formula_reports_progress_and_stops():
    """
    progress_cb вызывается каждое поколение, should_stop прерывает эволюцию,
    и результат при этом остаётся годным деревом, а не None.
    """
    X = random_X(n=200, seed=0)
    y = X[:, 0] * 2.0
    w = np.ones(len(X))

    seen = []
    result = dist.evolve_formula(
        X, y, w, generations=50, population_size=12, elitism=2, patience=0,
        depth_penalty=0.05, subsample=100, label='T', rng_seed=0,
        progress_cb=seen.append, should_stop=lambda: len(seen) >= 3)

    assert len(seen) == 3, "остановка должна срабатывать сразу после поколения"
    assert result['stopped'] is True
    assert isinstance(result['best_tree'], dist.EMLNode)
    assert len(result['history']) == 3
    assert seen[0]['phase'] == 'data' and seen[0]['label'] == 'T'


def test_evolve_formula_improves_fitness():
    """Эволюция должна улучшать fitness, иначе она бесполезна."""
    X = random_X(n=300, seed=1)
    y = X[:, 3] - X[:, 7]
    w = np.ones(len(X))
    result = dist.evolve_formula(
        X, y, w, generations=25, population_size=40, elitism=4, patience=0,
        depth_penalty=0.05, subsample=200, rng_seed=1)
    assert result['history'][-1] >= result['history'][0]
    assert result['stopped'] is False


def test_evolve_ensemble_stops_and_keeps_full_set():
    """Прерванная ACC-фаза возвращает полный набор формул, а не огрызок."""
    X = random_X(n=200, seed=2)
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    seed_trees = random_trees(config.N_CLASSES, seed=2)

    seen = []
    result = dist.evolve_ensemble(
        seed_trees, X, y, generations=30, population_size=8, elitism=2,
        patience=0, subsample=100, verbose=False, rng_seed=0,
        progress_cb=seen.append, should_stop=lambda: len(seen) >= 2)

    assert result['stopped'] is True
    assert len(result['best_trees']) == config.N_CLASSES
    assert seen[0]['phase'] == 'acc'
