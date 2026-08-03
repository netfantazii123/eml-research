"""
Тесты ветки C: одна формула на все классы плюс подобранная под неё таблица
порогов.

Проверяется то, что при поломке не падает, а тихо портит результат:

  * ДП декодера — сердце fitness. Приближённый или сбитый подбор порогов не
    даёт ошибки, он просто занижает fitness, и эволюция читает это как
    «дерево стало хуже». Поэтому оптимальность разбиения проверяется на
    случаях, где ответ известен заранее.
  * Совпадение обещанной метрики с реальной. fit_decoder возвращает score;
    если он разойдётся с настоящей сбалансированной accuracy того же
    декодера, весь отбор поедет, и заметить это будет не по чему.
  * Различимость порогов во float32. Выход дерева доходит до 2.2e4, шаг
    float32 там ~0.002; схлопнувшийся порог означает исчезнувший класс на
    микроконтроллере при исправном экспорте.
"""

import numpy as np
import pytest

import config
import eml_distiller as ed
import eml_single as es


@pytest.fixture(autouse=True)
def four_inputs():
    """Формулы ветки C индексируют входы — задаём их размерность явно."""
    ed.configure_inputs(4, ['a', 'b', 'c', 'd'])
    yield
    ed.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))


# ── Веса примеров ─────────────────────────────────────────────────────────────

def test_weights_turn_accuracy_into_balanced_accuracy():
    """
    Сумма весов угаданных примеров = сбалансированная accuracy.

    На этом равенстве стоит вся оптимизация декодера: ДП не знает про классы
    ничего, кроме «набрать побольше веса».
    """
    rng = np.random.default_rng(0)
    y = rng.choice(4, 500, p=[0.1, 0.2, 0.3, 0.4])
    pred = rng.choice(4, 500)
    weights = es.sample_weights(y, 4)

    assert (weights[pred == y].sum()
            == pytest.approx(ed.balanced_accuracy(y, pred, 4), abs=1e-12))


def test_weights_ignore_absent_classes():
    """Класс, которого нет в подвыборке, не тянет метрику вниз."""
    y = np.array([0, 0, 1, 1])
    weights = es.sample_weights(y, 4)
    assert weights.sum() == pytest.approx(1.0)


# ── Декодер ───────────────────────────────────────────────────────────────────

def test_decoder_finds_perfect_split():
    """Разделимые по оси классы разделяются полностью и ровно N-1 порогом."""
    y = np.repeat([0, 1, 2, 3], 50)
    values = y * 10.0 + np.linspace(0, 1, 200)
    decoder = es.fit_decoder(values, y, 4, bins=200)

    assert decoder.score == pytest.approx(1.0)
    assert decoder.n_intervals == 4
    assert np.array_equal(decoder.decode(values), y)


def test_grid_resolution_limits_the_decoder():
    """
    Порог встаёт только на границу бина — сетка задаёт достижимую точность.

    Это не дефект, а цена скорости: во время эволюции сетка грубая (промах
    одинаков для всех деревьев, сравнению между собой он не мешает), а
    итоговая таблица считается один раз по сетке в разы мельче. Тест
    закрепляет обе половины утверждения — иначе однажды кто-нибудь «оптимизирует»
    финальный подбор до эволюционной сетки и потеряет процент из ниоткуда.
    """
    y = np.repeat([0, 1, 2, 3], 50)
    values = y * 10.0 + np.linspace(0, 1, 200)

    coarse = es.fit_decoder(values, y, 4, bins=8)
    fine = es.fit_decoder(values, y, 4, bins=200)
    assert coarse.score < fine.score == pytest.approx(1.0)
    assert config.EMLC_FINAL_BINS > config.EMLC_DECODER_BINS


def test_decoder_gives_a_class_several_zones():
    """
    Класс имеет право занимать несколько участков оси.

    Это главная причина, по которой декодер не сводится к «N-1 порогу»:
    четыре класса не упорядочены, и требование расположить их на прямой в один
    ряд было бы ограничением, взятым с потолка. Здесь класс 0 сидит по краям, а
    класс 1 в середине — линейным порядком такое не решается вовсе.
    """
    y = np.array([0] * 60 + [1] * 60 + [0] * 60)
    values = np.concatenate([np.linspace(-3, -2, 60),
                             np.linspace(-1, 1, 60),
                             np.linspace(2, 3, 60)])
    decoder = es.fit_decoder(values, y, 2, bins=180)

    assert decoder.score == pytest.approx(1.0)
    assert decoder.labels.tolist() == [0, 1, 0]
    assert len(decoder.class_intervals(0)) == 2


def test_decoder_score_matches_its_own_predictions():
    """
    Обещанный score равен настоящей сбалансированной accuracy этого декодера.

    Расхождение здесь не упало бы: fitness просто перестал бы соответствовать
    метрике, и эволюция улучшала бы не то, что мы меряем.
    """
    for seed in range(5):
        rng = np.random.default_rng(seed)
        # Масштаб выхода формулы гуляет на порядки — декодер обязан работать
        # и на долях единицы, и на тысячах.
        values = rng.normal(size=800) * 10.0 ** rng.integers(-2, 4)
        y = rng.choice(4, 800)
        decoder = es.fit_decoder(values, y, 4)
        real = ed.balanced_accuracy(y, decoder.decode(values), 4)
        assert decoder.score == pytest.approx(real, abs=1e-12)


def test_decoder_beats_any_hand_made_split():
    """
    ДП находит оптимум, а не просто что-то приличное.

    Сравнение с перебором всех разбиений на маленькой задаче: если бы подбор
    был жадным, он бы здесь проиграл — распределение специально сделано так,
    что лучший первый порог не входит в лучшую пару порогов.
    """
    y = np.array([0, 1, 0, 1, 2, 2, 0, 0])
    values = np.arange(8, dtype=float)
    decoder = es.fit_decoder(values, y, 3, max_intervals=3, interval_penalty=0.0)

    best = -1.0
    for i in range(1, 8):                     # перебор двух порогов
        for j in range(i + 1, 8):
            for labels in np.ndindex(3, 3, 3):
                pred = np.array(labels)[np.searchsorted([i - 0.5, j - 0.5],
                                                        values, side='right')]
                best = max(best, ed.balanced_accuracy(y, pred, 3))
    assert decoder.score == pytest.approx(best, abs=1e-12)


def test_max_intervals_is_respected():
    """Потолок числа участков — жёсткий: на устройстве это длина таблицы."""
    rng = np.random.default_rng(2)
    y = rng.choice(4, 600)
    values = rng.normal(size=600)
    decoder = es.fit_decoder(values, y, 4, max_intervals=5,
                             interval_penalty=0.0)
    assert decoder.n_intervals <= 5


def test_interval_penalty_shortens_the_table():
    """
    Штраф за участок реально сокращает таблицу.

    Без него K всегда упирается в потолок: лишний участок никогда не ухудшает
    обучающую метрику, он только переобучается на шум.
    """
    rng = np.random.default_rng(3)
    y = rng.choice(3, 900)
    values = rng.normal(size=900)
    free = es.fit_decoder(values, y, 3, max_intervals=12, interval_penalty=0.0)
    taxed = es.fit_decoder(values, y, 3, max_intervals=12, interval_penalty=0.05)
    assert taxed.n_intervals < free.n_intervals


def test_constant_output_degrades_to_one_class():
    """Формула-константа получает ровно 1/n_classes, а не случайное число."""
    y = np.repeat([0, 1, 2, 3], 25)
    decoder = es.fit_decoder(np.full(100, 3.5), y, 4)
    assert decoder.n_intervals == 1
    assert decoder.score == pytest.approx(0.25)


def test_thresholds_survive_float32():
    """
    Пороги остаются строго возрастающими во float32.

    Выход EML-дерева доходит до 2.2e4 — там шаг float32 около 0.002, и два
    близких порога схлопываются в одно число. На микроконтроллере участок
    между ними исчезает вместе со своим классом, а сверка C с Python начинает
    краснеть при исправном экспорте.
    """
    rng = np.random.default_rng(4)
    base = 22000.0
    values = base + rng.normal(0, 1e-4, 900)
    y = rng.choice(4, 900)
    decoder = es.fit_decoder(values, y, 4, max_intervals=12,
                             interval_penalty=0.0)

    as_f32 = decoder.thresholds.astype(np.float32)
    assert np.all(np.diff(as_f32) > 0)
    assert decoder.score == pytest.approx(
        ed.balanced_accuracy(y, decoder.decode(values), 4), abs=1e-12)


def test_decode_one_matches_batch():
    """Скалярный инференс = векторному. По нему сверяется C-экспорт."""
    rng = np.random.default_rng(5)
    y = rng.choice(4, 300)
    values = rng.normal(size=300) * 100
    decoder = es.fit_decoder(values, y, 4)
    batch = decoder.decode(values)
    assert [decoder.decode_one(float(v)) for v in values] == batch.tolist()


def test_decoder_roundtrip():
    """JSON-сериализация без потерь: таблица порогов — часть артефакта."""
    decoder = es.IntervalDecoder([-1.5, 2.0, 7.25], [2, 0, 2, 1], 3, score=0.7)
    back = es.IntervalDecoder.from_dict(decoder.to_dict())
    assert np.array_equal(back.thresholds, decoder.thresholds)
    assert np.array_equal(back.labels, decoder.labels)
    assert back.n_classes == 3 and back.score == pytest.approx(0.7)


def test_decoder_rejects_inconsistent_table():
    """Меток должно быть ровно на одну больше, чем порогов."""
    with pytest.raises(ValueError):
        es.IntervalDecoder([0.0, 1.0], [0, 1], 2)


def test_non_finite_values_still_get_a_class():
    """
    NaN не роняет предсказание, а бесконечности идут по своим местам.

    Формула может сломаться на отдельном примере; предсказание при этом
    обязано остаться определённым, иначе метрика посчитается по битому
    массиву. У NaN места на оси нет — он уходит в левый участок; у +inf и -inf
    место есть, и загонять их туда же значило бы терять информацию.
    """
    decoder = es.IntervalDecoder([0.0], [0, 1], 2)
    got = decoder.decode(np.array([np.nan, np.inf, -np.inf, 5.0]))
    assert got.tolist() == [0, 1, 0, 1]
    assert decoder.decode_one(float('nan')) == 0
    assert decoder.decode_one(float('inf')) == 1


# ── Модель ────────────────────────────────────────────────────────────────────

def make_tree() -> ed.EMLNode:
    return ed.EMLNode('eml', left=ed.EMLNode('var', var_idx=0),
                      right=ed.EMLNode('var', var_idx=1))


def test_model_predict_matches_predict_one():
    """Векторный и скалярный путь дают один класс на каждом примере."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 4))
    y = rng.choice(3, 200)
    model = es.EMLSingle(make_tree(), es.IntervalDecoder([0.0, 1.0], [0, 1, 2], 3))
    batch = model.predict(X)
    assert [model.predict_one(row) for row in X] == batch.tolist()


def test_model_roundtrip():
    """Дерево вместе с таблицей порогов переживает сериализацию."""
    model = es.EMLSingle(make_tree(), es.IntervalDecoder([0.5], [1, 0], 2))
    back = es.EMLSingle.from_dict(model.to_dict())
    assert back.tree.to_string() == model.tree.to_string()
    assert np.array_equal(back.decoder.labels, model.decoder.labels)


def test_class_scores_agree_with_prediction():
    """Оценка классов во вкладке теста не должна спорить с самим ответом."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 4))
    model = es.EMLSingle(make_tree(),
                         es.IntervalDecoder([-1.0, 1.0], [0, 2, 1], 3))
    assert np.array_equal(model.class_scores(X).argmax(axis=1), model.predict(X))


# ── Fitness и эволюция ────────────────────────────────────────────────────────

def test_constant_tree_is_rejected_by_fitness():
    """
    Дерево без переменных получает отрицательный fitness.

    Формально оно даёт 1/n_classes (декодер вынужден отдать всю ось одному
    классу) и при этом стоит один узел — то есть без явного штрафа оказалось бы
    выгодным и заполнило бы популяцию.
    """
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 4))
    y = rng.choice(4, 200)
    params = es.FitnessParams(4)
    info = es.evaluate_tree(ed.EMLNode('const', value=1.0), X, y,
                            es.sample_weights(y, 4), params)
    assert info['fitness'] < 0


def test_size_penalty_prefers_small_trees():
    """При равном качестве побеждает дерево поменьше."""
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    small = make_tree()
    big = ed.EMLNode('eml', left=make_tree(), right=make_tree())
    params = es.FitnessParams(2, size_penalty=0.05)
    weights = es.sample_weights(y, 2)

    a = es.evaluate_tree(small, X, y, weights, params)
    b = es.evaluate_tree(big, X, y, weights, params)
    assert a['fitness'] - a['score'] > b['fitness'] - b['score']


def make_learnable(n: int = 400, seed: int = 0):
    """Задача, решаемая одной формулой: класс определяется первым входом."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 4))
    y = np.digitize(X[:, 0], [-0.3, 0.3]).astype(np.int64)
    return X, y


def test_island_improves_fitness():
    """Эволюция острова действительно улучшает fitness, а не топчется."""
    X, y = make_learnable()
    out = es.run_island([ed.random_tree(max_depth=3, force_eml=True)
                         for _ in range(30)],
                        X, y, es.FitnessParams(3), generations=12, elitism=3,
                        subsample=400, rng_seed=0)
    assert out['history'][-1] >= out['history'][0]
    assert out['best_fitness'] >= max(out['history'][:3])


def test_evolve_single_learns_and_freezes_decoder():
    """
    Полный прогон возвращает работоспособную модель.

    Таблица порогов в ней подобрана по ПОЛНОЙ обучающей выборке и заморожена —
    ровно как калибровка ветки A. Подбирать её потом по val значило бы
    подглядывать в отложенную выборку.
    """
    X, y = make_learnable(500, seed=1)
    out = es.evolve_single(X, y, n_classes=3, generations=30, population=40,
                           islands=2, workers=1, migrate_every=10,
                           subsample=500, patience=0, rng_seed=0, verbose=False)
    model = out['model']

    assert out['train_score'] > 0.5
    assert model.decoder.n_intervals >= 2
    assert out['train_score'] == pytest.approx(
        ed.balanced_accuracy(y, model.predict(X), 3), abs=1e-12)


def test_stop_request_returns_usable_model():
    """Прерванный прогон отдаёт согласованную модель и помечается флагом."""
    X, y = make_learnable(300, seed=2)
    out = es.evolve_single(X, y, n_classes=3, generations=200, population=20,
                           islands=1, workers=1, migrate_every=5, subsample=300,
                           patience=0, rng_seed=0, verbose=False,
                           should_stop=lambda: True)
    assert out['stopped'] is True
    assert out['generation'] < 200
    assert out['model'].predict(X).shape == y.shape


def test_checkpoint_resumes_where_it_stopped(tmp_path):
    """
    Продолжение подхватывает поколение, лучшего и все острова.

    Многочасовой прогон без этого теряется целиком при любой перезагрузке —
    а именно многочасовые прогоны тут и планируются.
    """
    X, y = make_learnable(300, seed=3)
    path = str(tmp_path / 'ckpt.json')
    common = dict(n_classes=3, population=20, islands=2, workers=1,
                  migrate_every=5, subsample=300, patience=0, verbose=False,
                  checkpoint_path=path, checkpoint_every=5)

    first = es.evolve_single(X, y, generations=10, resume=False, rng_seed=0,
                             **common)
    state = es.load_checkpoint(path)
    assert state['generation'] == 10
    assert len(state['islands']) == 2

    second = es.evolve_single(X, y, generations=20, resume=True, rng_seed=0,
                              **common)
    assert second['generation'] == 20
    # Продолжение не имеет права оказаться хуже: лучший переносится как есть.
    assert second['best_fitness'] >= first['best_fitness']


def test_migration_keeps_donor_elite():
    """
    Миграция отправляет КОПИИ лучших, а не переносит их.

    Иначе остров теряет свою элиту, обмен превращается в перемешивание, и
    разнообразие, ради которого острова и заведены, исчезает за пару обменов.
    """
    pops = [[ed.random_tree(max_depth=2, force_eml=True) for _ in range(6)]
            for _ in range(3)]
    elite = [pop[0].to_string() for pop in pops]
    es._migrate(pops, migrants=2)
    assert [pop[0].to_string() for pop in pops] == elite
    assert pops[1][-1].to_string() == elite[0]
