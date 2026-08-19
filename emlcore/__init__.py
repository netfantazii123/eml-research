"""
emlcore — the shared core of the EML distillation experiments.

The tasks (tetris, flappy_bird_with_ai, ...) own their environment, oracle,
dataset and fitness. Everything below that — the formula representation, the
basis of primitives, the genetic search, the static cost metrics and the C
exporter — lives here once, so a change to the search is a change to all tasks
rather than to whichever copy was edited last.

Typical use from a task:

    from emlcore import eml_basis, Node, search

    basis = eml_basis(config.N_FEATURES, config.FEATURE_NAMES,
                      max_depth=config.EML_MAX_DEPTH,
                      epsilon=config.EML_EPSILON)
    tree = search.random_tree(basis, max_depth=3, force_op=True)
"""

from .basis import Basis, Op, eml_basis, make_eml_op
from .node import Node
from .metrics import static_metrics, pool_slice
from . import search, cexport

__all__ = [
    'Basis', 'Op', 'eml_basis', 'make_eml_op',
    'Node', 'static_metrics', 'pool_slice',
    'search', 'cexport',
]
