"""The no-oracle control: the GAME phase must run from random trees.

Regression: with seed_tree=None the immigrants mutated seed_tree and the
second generation crashed (AttributeError on None.clone()).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eml_distiller import evolve_ingame, make_basis   # noqa: E402


def test_game_phase_without_seed_tree_runs_past_first_breed():
    import random
    random.seed(0)
    r = evolve_ingame(None, basis=make_basis('eml'), generations=4,
                      population_size=30, elitism=2, patience=10, n_games=1, val_games=1,
                      max_placements=20, verbose=False, workers=1)
    assert r is not None
    assert r.get('generations_run', 4) >= 2
