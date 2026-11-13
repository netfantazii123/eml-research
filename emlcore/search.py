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
    # NaN would make the pick depend on candidate order; count it as worst.
    return max(candidates, key=lambda i: (fitness_scores[i]
                                         if fitness_scores[i] == fitness_scores[i]
                                         else float('-inf')))


def fold_constants(node: Node) -> Node:
    """
    Replace every variable-free subtree by the constant it evaluates to.

    eml(0.7, -1.2) is a constant spelled with three nodes. The search breeds
    such subtrees all the time, and under a node budget — or with size as an
    NSGA-II objective — each one is pure waste: it costs nodes, exp/log calls on
    the microcontroller and readability, and buys nothing a single constant
    would not. Folding is exact up to float rounding, so the folded tree scores
    the same. Returns a new tree; the input is not touched.
    """
    if node.kind != 'op':
        return node.clone()
    kids = tuple(fold_constants(c) for c in node.children)
    if all(k.kind == 'const' for k in kids):
        try:
            v = node.op.scalar(*(k.value for k in kids))
        except (OverflowError, ValueError, ZeroDivisionError):
            v = float('nan')
        if v == v and abs(v) <= 1e6:          # finite and sane
            return Node.const(node.basis, v)
    return Node('op', node.basis, op=node.op, children=kids)


def tune_constants(tree: Node, score, *, steps: int = 8,
                   sigma: float = 0.3, current: float | None = None):
    """
    (1+1)-ES on the constants of one tree: perturb them all, keep if better.

    The GA moves constants by blind Gaussian mutation of one node at a time,
    and each attempt costs a whole child slot; a formula with the right shape
    and slightly wrong constants can sit unimproved for many generations.
    A few cheap local steps on the elites close that gap. score(tree) -> float
    is maximised. Returns (tree, score).
    """
    consts = [n for n in collect_nodes(tree) if n.kind == 'const']
    best = tree
    best_score = score(tree) if current is None else current
    if not consts:
        return best, best_score
    lim = tree.basis.const_abs_max
    for _ in range(steps):
        cand = best.clone()
        for n in collect_nodes(cand):
            if n.kind == 'const':
                v = n.value + random.gauss(0, sigma * (abs(n.value) + 0.1))
                n.value = max(-lim, min(lim, v))
        s = score(cand)
        if s > best_score:
            best, best_score = cand, s
        else:
            sigma *= 0.7
    return best, best_score


def _default_fresh(basis: Basis, intensity: float):
    def fresh():
        return random_tree(basis, max_depth=2 + int(intensity * 2),
                           force_op=True)
    return fresh


def _offspring(population: list[Node], select, basis: Basis, *, out: list,
               size: int, intensity: float, p_mutate: float, p_cross: float,
               p_cross_mutate: float, fresh, unique: bool,
               simplify: bool, seen: set | None = None) -> list[Node]:
    """
    Fill `out` up to `size` with mutants, crossovers and immigrants.

    The one place both breeders draw children from (they used to be two copies
    of this loop). select() returns a parent index.

    Children that violate the basis budget are rejected and re-drawn rather than
    trimmed: cutting a tree down to size destroys whatever structure made it
    worth keeping, and the search then spends generations rebuilding it.

    unique: also reject a child identical to one already in the generation
    (or in `seen`, extra keys to treat as taken). Without it, late generations
    fill up with copies of the elites — every copy costs an evaluation and adds
    nothing to the search.
    """
    if unique:
        seen = set(seen or ()) | {t.key() for t in out}
    else:
        seen = None
    attempts = 0
    max_attempts = size * 20
    while len(out) < size:
        attempts += 1
        r = random.random()
        if r < p_mutate:
            child = mutate_tree(population[select()], basis, intensity=intensity)
        elif r < p_cross:
            i1 = select()
            i2 = select()
            child = crossover(population[i1], population[i2], basis)
            if random.random() < p_cross_mutate:
                child = mutate_tree(child, basis, intensity=intensity * 0.5)
        else:
            child = fresh()
        if simplify:
            child = fold_constants(child)

        give_up = attempts > max_attempts
        if not within_budget(child, basis) and not give_up:
            continue
        if seen is not None:
            k = child.key()
            if k in seen and not give_up:
                continue
            seen.add(k)
        # The give-up fallback keeps a too-tight budget from hanging the run;
        # a budget that trips it is reported by the caller, not silenced.
        out.append(child)
    return out


def breed(population: list[Node], fitness_scores: list[float], basis: Basis, *,
          size: int, elitism: int, intensity: float,
          p_mutate: float = 0.45, p_cross: float = 0.75,
          p_cross_mutate: float = 0.3,
          tournament_k: int = 5, fresh=None,
          unique: bool = False, simplify: bool = False) -> list[Node]:
    """
    Build the next generation: elites, then mutation / crossover / fresh blood.

    `population` must be sorted best first (elites are its head).
    `fresh` is a callable producing an immigrant tree; it defaults to a random
    tree, and the GAME phase passes a mutant of the seed formula instead.
    With unique=False and simplify=False the random stream is consumed exactly
    as before these options existed, so old seeds reproduce old runs.
    """
    if fresh is None:
        fresh = _default_fresh(basis, intensity)

    # Elites are budget-checked too: a survivor that violates the budget would
    # otherwise carry the violation forward for the rest of the run.
    new_pop = [population[i].clone()
               for i in range(min(elitism, len(population)))
               if within_budget(population[i], basis)]
    return _offspring(
        population, lambda: tournament_select(fitness_scores, k=tournament_k),
        basis, out=new_pop, size=size, intensity=intensity,
        p_mutate=p_mutate, p_cross=p_cross, p_cross_mutate=p_cross_mutate,
        fresh=fresh, unique=unique, simplify=simplify)


def breed_nsga2(population: list[Node], ranks, crowd, basis: Basis, *,
                size: int, intensity: float,
                p_mutate: float = 0.45, p_cross: float = 0.75,
                p_cross_mutate: float = 0.3, tournament_k: int = 3,
                fresh=None, unique: bool = False,
                simplify: bool = False) -> list[Node]:
    """
    Offspring for an NSGA-II generation.

    No elites are copied here: survivors are decided afterwards by
    environmental selection over parents + offspring, which is what preserves
    the whole front rather than only its best-quality end. With unique, an
    offspring identical to a parent is rejected too — it would only enter the
    (mu + lambda) pool as a duplicate of a point already there.
    """
    from .nsga2 import crowded_tournament

    if fresh is None:
        fresh = _default_fresh(basis, intensity)
    return _offspring(
        population, lambda: crowded_tournament(ranks, crowd, k=tournament_k),
        basis, out=[], size=size, intensity=intensity,
        p_mutate=p_mutate, p_cross=p_cross, p_cross_mutate=p_cross_mutate,
        fresh=fresh, unique=unique, simplify=simplify,
        seen={t.key() for t in population} if unique else None)


def seed_population(basis: Basis, size: int, *, seed_tree: Node | None = None,
                    max_depth: int = 3, intensity: float = 0.7,
                    seed_trees: list[Node] | None = None) -> list[Node]:
    """
    Initial population: random trees, or mutants around a seed formula.

    Budget-checked like breeding is. Skipping the check here quietly breaks
    fixed-K runs: mutation grows trees, so a population seeded from a K-node
    formula starts above K and every later generation inherits the violation
    through the elites.

    seed_trees: several formulas to start from (e.g. the hall of fame of an
    earlier run). All of them enter unchanged, the rest are mutants of a random
    one — fine-tuning around a set of best performers rather than one.
    """
    if seed_trees:
        seeds = [t for t in seed_trees if within_budget(t, basis)][:size]
        if not seeds:
            seeds = list(seed_trees[:1])
    elif seed_tree is not None:
        seeds = [seed_tree]
    else:
        seeds = []

    def _draw():
        if not seeds:
            return random_tree(basis, max_depth=max_depth, force_op=True)
        src = seeds[0] if len(seeds) == 1 else random.choice(seeds)
        return mutate_tree(src, basis, intensity=intensity)

    pop = [t.clone() for t in seeds]
    attempts = 0
    max_attempts = size * 20
    while len(pop) < size:
        attempts += 1
        cand = _draw()
        if within_budget(cand, basis) or attempts > max_attempts:
            pop.append(cand)
    return pop
