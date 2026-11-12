"""
dagger.py — Dataset Aggregation (Ross, Gordon & Bagnell, 2011) for any student.

Why it is here. Plain distillation (behavioural cloning) fits the student on the
states the ORACLE visits. The student then plays by itself, makes a small
mistake, lands in a state the oracle never reached, and has no data there — the
errors compound with the horizon (O(T^2 eps) for BC against O(T eps) for
DAgger). That is the textbook reason a DATA-phase formula scores near zero in a
game while fitting the oracle well, and the reason the GAME phase had to exist.
DAgger attacks the same problem from the data side instead of from the fitness
side: roll the CURRENT student out, ask the oracle what it would have done in
the states the student actually reached, add those labels, refit.

Iteration i:
    1. pi_mix = beta_i * oracle + (1 - beta_i) * student_{i-1}
       (per-step coin: the oracle acts with probability beta_i)
    2. run pi_mix in the environment, label EVERY visited state with the oracle
    3. D <- D U D_i                         (aggregate=True, the DAgger rule)
       D <- D_i                             (aggregate=False, for the ablation)
    4. student_i = fit(D, warm_start=student_{i-1})
    5. score student_i on VAL; keep the best by VAL (never by TEST)

Nothing here knows about EML, Tetris or Flappy: the task supplies how to roll a
mixture out and label it (collect_fn), how to fit a student (fit_fn) and how to
score one (eval_fn). Data chunks are opaque — concatenating them is the task's
job, because a Tetris chunk is a set of placement GROUPS and a Flappy chunk is a
flat array of states.

Iteration 0 with beta_0 = 1 is exactly behavioural cloning, so the BC baseline
is the first row of every DAgger history and costs nothing extra.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ── beta schedules ───────────────────────────────────────────────────────────

def beta_schedule(kind: str = 'exp', *, p: float = 0.5,
                  beta0: float = 1.0) -> Callable[[int], float]:
    """
    Probability that the ORACLE acts at iteration i.

    'exp'    beta_i = beta0 * p**i      (Ross et al.: beta_0 = 1, then decay)
    'linear' beta_i = max(0, 1 - i*p)
    'zero'   beta_0 = 1, beta_i = 0 for i >= 1 (the student acts alone after BC;
             the variant Ross et al. report as usually best in practice)
    """
    if kind == 'exp':
        return lambda i: beta0 * (p ** i)
    if kind == 'linear':
        return lambda i: max(0.0, beta0 - i * p)
    if kind == 'zero':
        return lambda i: beta0 if i == 0 else 0.0
    raise ValueError(f"unknown beta schedule {kind!r}")


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class DaggerIter:
    """One iteration: what was collected, what was fitted, how it scored."""
    iteration: int
    beta: float
    n_new: int                 # rows (or groups) collected this iteration
    n_total: int               # rows in the training set the student saw
    fit_seconds: float
    collect_seconds: float
    metrics: dict = field(default_factory=dict)   # eval_fn output (VAL)
    fit_info: dict = field(default_factory=dict)  # whatever fit_fn reports


@dataclass
class DaggerResult:
    best_student: Any
    best_iteration: int
    best_score: float
    students: list
    history: list[DaggerIter]

    def table(self) -> list[dict]:
        """Flat rows for CSV / the report."""
        rows = []
        for h in self.history:
            rows.append({'iteration': h.iteration, 'beta': h.beta,
                         'n_new': h.n_new, 'n_total': h.n_total,
                         'collect_s': round(h.collect_seconds, 2),
                         'fit_s': round(h.fit_seconds, 2),
                         **{f'val_{k}': v for k, v in h.metrics.items()
                            if isinstance(v, (int, float, str, bool))},
                         **{k: v for k, v in h.fit_info.items()
                            if isinstance(v, (int, float, str, bool))}})
        return rows


# ── The loop ─────────────────────────────────────────────────────────────────

def run_dagger(
    collect_fn: Callable[[Any, float, int], Any],
    fit_fn: Callable[[list, Any], Any],
    eval_fn: Callable[[Any], dict] | None = None,
    *,
    n_iters: int = 5,
    beta: Callable[[int], float] | None = None,
    initial_data: Any | None = None,
    aggregate: bool = True,
    size_fn: Callable[[Any], int] = len,
    score_key: str = 'score',
    fit_info_fn: Callable[[Any], dict] | None = None,
    on_iter: Callable[[DaggerIter], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> DaggerResult:
    """
    Generic DAgger.

    collect_fn(student, beta, it) -> chunk
        Roll the mixture out (oracle acts with probability beta, otherwise the
        student; student is None at it=0), label every visited state with the
        oracle and return the chunk. Seeds of the rollouts must come from a
        stream disjoint from the VAL/TEST pools — the caller owns that.
    fit_fn(chunks, warm_start) -> student
        Fit on the list of chunks (one per iteration so far, or only the last
        one when aggregate=False). warm_start is the previous student or None.
    eval_fn(student) -> dict
        Scores on VAL. dict[score_key] is maximised for best-iteration
        selection. Without eval_fn the last student wins.
    initial_data
        A chunk to use as D_0 instead of collecting one (e.g. the BC dataset
        already on disk). Iteration 0 then only fits.
    """
    if beta is None:
        beta = beta_schedule('exp', p=0.5)

    chunks: list = []
    students: list = []
    history: list[DaggerIter] = []
    student = None
    best, best_it, best_score = None, -1, -math.inf

    for it in range(n_iters):
        if should_stop is not None and should_stop():
            break
        b = float(beta(it))

        t0 = time.perf_counter()
        if it == 0 and initial_data is not None:
            chunk = initial_data
        else:
            chunk = collect_fn(student, b, it)
        t_collect = time.perf_counter() - t0

        if aggregate:
            chunks.append(chunk)
            train = chunks
        else:
            train = [chunk]

        t0 = time.perf_counter()
        student = fit_fn(train, student)
        t_fit = time.perf_counter() - t0
        students.append(student)

        metrics = eval_fn(student) if eval_fn is not None else {}
        rec = DaggerIter(iteration=it, beta=b, n_new=size_fn(chunk),
                         n_total=sum(size_fn(c) for c in train),
                         fit_seconds=t_fit, collect_seconds=t_collect,
                         metrics=metrics,
                         fit_info=fit_info_fn(student) if fit_info_fn else {})
        history.append(rec)
        if on_iter is not None:
            on_iter(rec)

        score = metrics.get(score_key, it) if metrics else it
        if score > best_score:
            best, best_it, best_score = student, it, score

    return DaggerResult(best_student=best, best_iteration=best_it,
                        best_score=best_score, students=students,
                        history=history)
