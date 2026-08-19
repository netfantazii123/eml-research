"""
search.py — the genetic search over formula trees.

Nothing here knows what an EML operator is: every operator comes from the basis,
and arity is read off the Op. That is the whole reason the core was extracted —
the control-basis comparison has to run this exact search, not a second copy of
it that drifted.

Randomness comes from the `random` module, seeded by the caller. That is
deliberate: a run is reproduced by seeding once and replaying, and the seed is
recorded next to the result.
"""

from __future__ import annotations

import random

from .basis import Basis
from .node import Node


def random_leaf(basis: Basis) -> Node:
    """A leaf: a variable with probability var_leaf_prob, else a constant."""
    if random.random() < basis.var_leaf_prob:
        return Node.var(basis, random.randint(0, basis.n_vars - 1))
    return Node.const(basis, random.uniform(*basis.const_range))


def random_op(basis: Basis, max_arity: int | None = None):
    """A random operator, optionally restricted to at most max_arity children."""
    pool = basis.ops
    if max_arity is not None:
        pool = tuple(o for o in pool if o.arity <= max_arity)
    return random.choice(pool if pool else basis.ops)


def random_tree(basis: Basis, max_depth: int = 3,
                force_op: bool = False) -> Node:
    """Random tree. force_op -> the root is always an operator (depth >= 1)."""
    if force_op and max_depth >= 1:
        op = random_op(basis)
        return Node.apply(basis, op,
                          *(random_tree(basis, max_depth - 1)
                            for _ in range(op.arity)))
    if max_depth <= 0 or random.random() < 0.3:
        return random_leaf(basis)
    op = random_op(basis)
    return Node.apply(basis, op,
                      *(random_tree(basis, max_depth - 1)
                        for _ in range(op.arity)))


def collect_nodes(node: Node) -> list[Node]:
    """Every node of the tree, root first."""
    out = [node]
    for c in node.children:
        out.extend(collect_nodes(c))
    return out


def prune_depth(node: Node, max_depth: int) -> Node:
    """Trim to max_depth — subtrees below the limit collapse into leaves."""
    if node.kind != 'op':
        return node
    if max_depth <= 0:
        return random_leaf(node.basis)
    node.children = tuple(prune_depth(c, max_depth - 1) for c in node.children)
    return node


def within_budget(tree: Node, basis: Basis) -> bool:
    """Does the tree fit the basis limits (depth and, if set, node budget)?"""
    if tree.depth() > basis.max_depth:
        return False
    return basis.max_nodes is None or tree.size() <= basis.max_nodes


def mutate_tree(root: Node, basis: Basis, intensity: float = 1.0) -> Node:
    """
    Mutation with adaptive intensity (1.0 = wild, 0.1 = fine).

    The probability schedule is the one both task distillers used before the
    core was extracted, kept unchanged so results stay comparable across the
    refactor.
    """
    tree = root.clone()
    n_mutations = 1 + int(intensity * 2 * random.random())

    for _ in range(n_mutations):
        target = random.choice(collect_nodes(tree))
        r = random.random()

        if r < 0.2 and target.kind == 'const':
            noise_std = 0.2 + intensity * 1.0
            v = target.value + random.gauss(0, noise_std)
            lim = basis.const_abs_max
            target.value = max(-lim, min(lim, v))

        elif r < 0.35 and target.kind == 'var':
            target.var_idx = random.randint(0, basis.n_vars - 1)

        elif r < 0.6 and target.kind != 'op':
            # Grow: leaf -> op(leaf, ...)
            if tree.depth() < basis.max_depth:
                op = random_op(basis)
                target.become(Node.apply(
                    basis, op, *(random_leaf(basis) for _ in range(op.arity))))

        elif r < 0.7 and target.kind == 'op':
            i = random.randrange(len(target.children))
            target.replace_child(i, random_tree(basis, max_depth=2))

        elif r < 0.8 and target.kind == 'op' and intensity < 0.5:
            # Simplify: subtree -> leaf (only when fine-tuning).
            target.become(random_leaf(basis))

        else:
            depth = 1 + int(intensity * 2)
            target.become(random_tree(basis, max_depth=depth, force_op=True))

    return prune_depth(tree, basis.max_depth)


def crossover(parent1: Node, parent2: Node, basis: Basis) -> Node:
    """Replace a random subtree of parent1 with a subtree from parent2."""
    child = parent1.clone()
    target = random.choice(collect_nodes(child))
    donor = random.choice(collect_nodes(parent2)).clone()
    target.become(donor)
    return prune_depth(child, basis.max_depth)


def tournament_select(fitness_scores: list[float], k: int = 5) -> int:
    """Index of the best of k random candidates."""
    n = len(fitness_scores)
    candidates = random.sample(range(n), min(k, n))
    return max(candidates, key=lambda i: fitness_scores[i])


def breed(population: list[Node], fitness_scores: list[float], basis: Basis, *,
          size: int, elitism: int, intensity: float,
          p_mutate: float = 0.45, p_cross: float = 0.75,
          p_cross_mutate: float = 0.3,
          tournament_k: int = 5, fresh=None) -> list[Node]:
    """
    Build the next generation: elites, then mutation / crossover / fresh blood.

    `fresh` is a callable producing an immigrant tree; it defaults to a random
    tree, and the GAME phase passes a mutant of the seed formula instead.

    Children that violate the basis budget are rejected and re-drawn rather than
    trimmed: cutting a tree down to size destroys whatever structure made it
    worth keeping, and the search then spends generations rebuilding it.
    """
    if fresh is None:
        def fresh():
            return random_tree(basis, max_depth=2 + int(intensity * 2),
                               force_op=True)

    new_pop = [population[i].clone() for i in range(min(elitism, len(population)))]
    attempts = 0
    max_attempts = size * 20
    while len(new_pop) < size:
        attempts += 1
        r = random.random()
        if r < p_mutate:
            i = tournament_select(fitness_scores, k=tournament_k)
            child = mutate_tree(population[i], basis, intensity=intensity)
        elif r < p_cross:
            i1 = tournament_select(fitness_scores, k=tournament_k)
            i2 = tournament_select(fitness_scores, k=tournament_k)
            child = crossover(population[i1], population[i2], basis)
            if random.random() < p_cross_mutate:
                child = mutate_tree(child, basis, intensity=intensity * 0.5)
        else:
            child = fresh()

        if within_budget(child, basis) or attempts > max_attempts:
            # The fallback keeps a too-tight budget from hanging the run; a
            # budget that trips it is reported by the caller, not silenced.
            new_pop.append(child)
    return new_pop


def seed_population(basis: Basis, size: int, *, seed_tree: Node | None = None,
                    max_depth: int = 3, intensity: float = 0.7) -> list[Node]:
    """Initial population: random trees, or mutants around a seed formula."""
    if seed_tree is None:
        return [random_tree(basis, max_depth=max_depth, force_op=True)
                for _ in range(size)]
    pop = [seed_tree.clone()]
    while len(pop) < size:
        pop.append(mutate_tree(seed_tree, basis, intensity=intensity))
    return pop
