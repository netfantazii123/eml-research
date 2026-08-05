"""
Tests of branch C: one formula for all classes plus a threshold table fitted to
it.

What is checked is what, when broken, does not fail but quietly corrupts the
result:

  * The decoder DP is the heart of the fitness. An approximate or broken
    threshold fit produces no error, it merely understates the fitness, and
    evolution reads that as "the tree got worse". Optimality of the partition is
    therefore checked on cases where the answer is known in advance.
  * Agreement of the promised metric with the real one. fit_decoder returns a
    score; if it diverges from the actual balanced accuracy of the same decoder,
    the whole selection goes astray with nothing to notice it by.
  * Threshold distinguishability in float32. The tree output reaches 2.2e4,
    where the float32 step is ~0.002; a collapsed threshold means a class
    disappearing on the microcontroller with a correct export.
"""

import numpy as np
import pytest

import config
import eml_distiller as ed
import eml_single as es


@pytest.fixture(autouse=True)
def four_inputs():
    """Branch C formulas index their inputs — set the dimension explicitly."""
    ed.configure_inputs(4, ['a', 'b', 'c', 'd'])
    yield
    ed.configure_inputs(config.N_FEATURES, list(config.FEATURE_NAMES))


# ── Example weights ───────────────────────────────────────────────────────────

def test_weights_turn_accuracy_into_balanced_accuracy():
    """
    The sum of weights of correctly guessed examples = balanced accuracy.

    The whole decoder optimization rests on this equality: the DP knows nothing
    about classes beyond "collect as much weight as possible".
    """
    rng = np.random.default_rng(0)
    y = rng.choice(4, 500, p=[0.1, 0.2, 0.3, 0.4])
    pred = rng.choice(4, 500)
    weights = es.sample_weights(y, 4)

    assert (weights[pred == y].sum()
            == pytest.approx(ed.balanced_accuracy(y, pred, 4), abs=1e-12))


def test_weights_ignore_absent_classes():
    """A class absent from the subsample does not drag the metric down."""
    y = np.array([0, 0, 1, 1])
    weights = es.sample_weights(y, 4)
    assert weights.sum() == pytest.approx(1.0)


# ── Decoder ───────────────────────────────────────────────────────────────────

def test_decoder_finds_perfect_split():
    """Classes separable on the axis are separated fully by exactly N-1 thresholds."""
    y = np.repeat([0, 1, 2, 3], 50)
    values = y * 10.0 + np.linspace(0, 1, 200)
    decoder = es.fit_decoder(values, y, 4, bins=200)

    assert decoder.score == pytest.approx(1.0)
    assert decoder.n_intervals == 4
    assert np.array_equal(decoder.decode(values), y)


def test_grid_resolution_limits_the_decoder():
    """
    A threshold falls only on a bin boundary — the grid sets the achievable
    precision.

    This is not a defect but the price of speed: during evolution the grid is
    coarse (the miss is the same for all trees and does not hamper comparing
    them), while the final table is computed once on a far finer grid. The test
    pins both halves of the claim — otherwise someone will one day "optimize"
    the final fit down to the evolution grid and lose a percent for nothing.
    """
    y = np.repeat([0, 1, 2, 3], 50)
    values = y * 10.0 + np.linspace(0, 1, 200)

    coarse = es.fit_decoder(values, y, 4, bins=8)
    fine = es.fit_decoder(values, y, 4, bins=200)
    assert coarse.score < fine.score == pytest.approx(1.0)
    assert config.EMLC_FINAL_BINS > config.EMLC_DECODER_BINS


def test_decoder_gives_a_class_several_zones():
    """
    A class is allowed to occupy several axis intervals.

    This is the main reason the decoder does not reduce to "N-1 thresholds": the
    four classes are not ordered, and requiring them to line up on a line in one
    row would be an arbitrary constraint. Here class 0 sits at both edges and
    class 1 in the middle — a linear order cannot express that at all.
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
    The promised score equals the actual balanced accuracy of this decoder.

    A divergence here would not fail: the fitness would simply stop matching the
    metric, and evolution would improve something other than what we measure.
    """
    for seed in range(5):
        rng = np.random.default_rng(seed)
        # The formula output scale varies by orders of magnitude — the decoder
        # must work both on fractions of a unit and on thousands.
        values = rng.normal(size=800) * 10.0 ** rng.integers(-2, 4)
        y = rng.choice(4, 800)
        decoder = es.fit_decoder(values, y, 4)
        real = ed.balanced_accuracy(y, decoder.decode(values), 4)
        assert decoder.score == pytest.approx(real, abs=1e-12)


def test_decoder_beats_any_hand_made_split():
    """
    The DP finds the optimum rather than merely something decent.

    Compared against exhaustive search over all partitions on a small problem:
    a greedy fit would lose here — the distribution is constructed so that the
    best first threshold is not part of the best pair of thresholds.
    """
    y = np.array([0, 1, 0, 1, 2, 2, 0, 0])
    values = np.arange(8, dtype=float)
    decoder = es.fit_decoder(values, y, 3, max_intervals=3, interval_penalty=0.0)

    best = -1.0
    for i in range(1, 8):                     # enumerate two thresholds
        for j in range(i + 1, 8):
            for labels in np.ndindex(3, 3, 3):
                pred = np.array(labels)[np.searchsorted([i - 0.5, j - 0.5],
                                                        values, side='right')]
                best = max(best, ed.balanced_accuracy(y, pred, 3))
    assert decoder.score == pytest.approx(best, abs=1e-12)


def test_max_intervals_is_respected():
    """The interval count cap is hard: on the device it is the table length."""
    rng = np.random.default_rng(2)
    y = rng.choice(4, 600)
    values = rng.normal(size=600)
    decoder = es.fit_decoder(values, y, 4, max_intervals=5,
                             interval_penalty=0.0)
    assert decoder.n_intervals <= 5


def test_interval_penalty_shortens_the_table():
    """
    The per-interval penalty really does shorten the table.

    Without it K always hits the ceiling: an extra interval never worsens the
    training metric, it only overfits noise.
    """
    rng = np.random.default_rng(3)
    y = rng.choice(3, 900)
    values = rng.normal(size=900)
    free = es.fit_decoder(values, y, 3, max_intervals=12, interval_penalty=0.0)
    taxed = es.fit_decoder(values, y, 3, max_intervals=12, interval_penalty=0.05)
    assert taxed.n_intervals < free.n_intervals


def test_constant_output_degrades_to_one_class():
    """A constant formula gets exactly 1/n_classes, not a random number."""
    y = np.repeat([0, 1, 2, 3], 25)
    decoder = es.fit_decoder(np.full(100, 3.5), y, 4)
    assert decoder.n_intervals == 1
    assert decoder.score == pytest.approx(0.25)


def test_thresholds_survive_float32():
    """
    The thresholds stay strictly increasing in float32.

    The EML tree output reaches 2.2e4 — there the float32 step is about 0.002,
    and two nearby thresholds collapse into one number. On a microcontroller the
    interval between them disappears along with its class, and the C/Python
    comparison starts failing on a correct export.
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
    """Scalar inference = vectorized. The C export is checked against it."""
    rng = np.random.default_rng(5)
    y = rng.choice(4, 300)
    values = rng.normal(size=300) * 100
    decoder = es.fit_decoder(values, y, 4)
    batch = decoder.decode(values)
    assert [decoder.decode_one(float(v)) for v in values] == batch.tolist()


def test_decoder_roundtrip():
    """Lossless JSON serialization: the threshold table is part of the artifact."""
    decoder = es.IntervalDecoder([-1.5, 2.0, 7.25], [2, 0, 2, 1], 3, score=0.7)
    back = es.IntervalDecoder.from_dict(decoder.to_dict())
    assert np.array_equal(back.thresholds, decoder.thresholds)
    assert np.array_equal(back.labels, decoder.labels)
    assert back.n_classes == 3 and back.score == pytest.approx(0.7)


def test_decoder_rejects_inconsistent_table():
    """There must be exactly one more label than thresholds."""
    with pytest.raises(ValueError):
        es.IntervalDecoder([0.0, 1.0], [0, 1], 2)


def test_non_finite_values_still_get_a_class():
    """
    NaN does not break the prediction, and infinities go to their proper places.

    A formula may break on an individual example; the prediction must stay
    defined regardless, otherwise the metric would be computed over a corrupted
    array. NaN has no place on the axis — it goes to the left interval; +inf and
    -inf do have a place, and sending them there too would lose information.
    """
    decoder = es.IntervalDecoder([0.0], [0, 1], 2)
    got = decoder.decode(np.array([np.nan, np.inf, -np.inf, 5.0]))
    assert got.tolist() == [0, 1, 0, 1]
    assert decoder.decode_one(float('nan')) == 0
    assert decoder.decode_one(float('inf')) == 1


# ── Model ─────────────────────────────────────────────────────────────────────

def make_tree() -> ed.EMLNode:
    return ed.EMLNode('eml', left=ed.EMLNode('var', var_idx=0),
                      right=ed.EMLNode('var', var_idx=1))


def test_model_predict_matches_predict_one():
    """The vectorized and scalar paths give one class on every example."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(200, 4))
    y = rng.choice(3, 200)
    model = es.EMLSingle(make_tree(), es.IntervalDecoder([0.0, 1.0], [0, 1, 2], 3))
    batch = model.predict(X)
    assert [model.predict_one(row) for row in X] == batch.tolist()


def test_model_roundtrip():
    """The tree together with its threshold table survives serialization."""
    model = es.EMLSingle(make_tree(), es.IntervalDecoder([0.5], [1, 0], 2))
    back = es.EMLSingle.from_dict(model.to_dict())
    assert back.tree.to_string() == model.tree.to_string()
    assert np.array_equal(back.decoder.labels, model.decoder.labels)


def test_class_scores_agree_with_prediction():
    """The class scores in the test tab must not contradict the answer itself."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 4))
    model = es.EMLSingle(make_tree(),
                         es.IntervalDecoder([-1.0, 1.0], [0, 2, 1], 3))
    assert np.array_equal(model.class_scores(X).argmax(axis=1), model.predict(X))


# ── Fitness and evolution ─────────────────────────────────────────────────────

def test_constant_tree_is_rejected_by_fitness():
    """
    A tree without variables receives a negative fitness.

    Formally it yields 1/n_classes (the decoder is forced to give the whole axis
    to one class) while costing a single node — i.e. without an explicit penalty
    it would be advantageous and would fill the population.
    """
    rng = np.random.default_rng(8)
    X = rng.normal(size=(200, 4))
    y = rng.choice(4, 200)
    params = es.FitnessParams(4)
    info = es.evaluate_tree(ed.EMLNode('const', value=1.0), X, y,
                            es.sample_weights(y, 4), params)
    assert info['fitness'] < 0


def test_size_penalty_prefers_small_trees():
    """At equal quality the smaller tree wins."""
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
    """A task solvable by one formula: the class is set by the first input."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 4))
    y = np.digitize(X[:, 0], [-0.3, 0.3]).astype(np.int64)
    return X, y


def test_island_improves_fitness():
    """Island evolution really improves the fitness rather than treading water."""
    X, y = make_learnable()
    out = es.run_island([ed.random_tree(max_depth=3, force_eml=True)
                         for _ in range(30)],
                        X, y, es.FitnessParams(3), generations=12, elitism=3,
                        subsample=400, rng_seed=0)
    assert out['history'][-1] >= out['history'][0]
    assert out['best_fitness'] >= max(out['history'][:3])


def test_evolve_single_learns_and_freezes_decoder():
    """
    A full run returns a working model.

    Its threshold table is fitted on the FULL training sample and frozen —
    exactly like the branch A calibration. Fitting it afterwards on val would
    mean looking into the held-out sample.
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
    """An interrupted run yields a consistent model and is marked with a flag."""
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
    Resumption picks up the generation, the best individual and all islands.

    Without it a multi-hour run is lost entirely on any reboot — and multi-hour
    runs are exactly what is planned here.
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
    # The resumed run must not come out worse: the best is carried over as is.
    assert second['best_fitness'] >= first['best_fitness']


def test_migration_keeps_donor_elite():
    """
    Migration sends COPIES of the best rather than transferring them.

    Otherwise an island loses its elite, the exchange turns into shuffling, and
    the diversity the islands exist for disappears within a couple of exchanges.
    """
    pops = [[ed.random_tree(max_depth=2, force_eml=True) for _ in range(6)]
            for _ in range(3)]
    elite = [pop[0].to_string() for pop in pops]
    es._migrate(pops, migrants=2)
    assert [pop[0].to_string() for pop in pops] == elite
    assert pops[1][-1].to_string() == elite[0]
