"""
make_report.py — regenerate the tables of report/REPORT.md from the raw CSVs.

The prose of REPORT.md is written by hand; every table between a pair of
markers

    <!-- NAME:start -->
    ...
    <!-- NAME:end -->

is rebuilt from the experiment CSVs, so a new run never means retyping numbers.
The section of the classical-distillation comparison is copied from
distill_compare/REPORT.md (its own tables:start/tables:end block) when that file
exists.

    python report/make_report.py
"""

from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(ROOT, 'report', 'REPORT.md')

TETRIS_DIR = os.path.join(ROOT, 'tetris', 'results', 'dagger')
FLAPPY_DIR = os.path.join(ROOT, 'flappy_bird_with_ai', 'results')
COMPARE_MD = os.path.join(ROOT, 'distill_compare', 'REPORT.md')

METHOD_RU = {
    'BC': 'BC (только данные оракула)',
    'DAgger': 'DAgger',
    'BC+GAME': 'BC + GAME (текущий пайплайн)',
    'DAgger+GAME': 'DAgger + GAME',
    'EML-GAME': 'EML: DATA + GAME (текущий пайплайн)',
}


def read_csv(path):
    """One CSV, or the union of every same-named CSV below its directory
    (parallel runs write one sub-directory each)."""
    import glob
    base, name = os.path.split(path)
    paths = sorted(set(glob.glob(os.path.join(base, '**', name),
                                 recursive=True)))
    rows = []
    for p in paths:
        with open(p, encoding='utf-8') as fh:
            rows += list(csv.DictReader(fh))
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def mean_std(xs):
    xs = list(xs)
    bad = sum(1 for x in xs if math.isnan(x))
    if bad:
        # Dropping them would quietly break the "every seed counts" rule while
        # the table still printed len(rows) as n.
        raise ValueError(f'{bad} of {len(xs)} values are NaN (a missing or '
                         'unscored run); fix the data instead of averaging '
                         'around it')
    if not xs:
        return math.nan, math.nan
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt(m, s=None, nd=1):
    if math.isnan(m):
        return '—'
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} ± {s:.{nd}f}"


def summary_table(rows, metric, metric_name, oracle_key=None, order=(),
                  names=None):
    names = {**METHOD_RU, **(names or {})}
    by = defaultdict(list)
    for r in rows:
        by[r['method']].append(r)
    methods = [m for m in order if m in by] + [m for m in by if m not in order]
    out = [f"| Метод | {metric_name} (test) | % от оракула | лучший сид "
           f"| узлы | трансц. вызовы | сидов ГА | игр на сид |",
           "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        rs = by[m]
        vals = [_f(r[metric]) for r in rs]
        pct = [_f(r['pct_oracle']) for r in rs]
        nodes = [_f(r['nodes']) for r in rs]
        tr = [_f(r['n_transcendental']) for r in rs]
        vm, vs = mean_std(vals)
        pm, ps = mean_std(pct)
        nm, _ = mean_std(nodes)
        tm, _ = mean_std(tr)
        n_games = rs[0].get('n_games') or rs[0].get('test_games') or ''
        out.append(f"| {names.get(m, m)} | {fmt(vm, vs, 2)} | "
                   f"{fmt(pm, ps)} | {fmt(max(vals), nd=2)} | {fmt(nm, nd=0)} | "
                   f"{fmt(tm, nd=0)} | {len(rs)} | {n_games} |")
    return "\n".join(out)


def iteration_table(rows, metric, name):
    by = defaultdict(list)
    for r in rows:
        by[int(r['iteration'])].append(r)
    out = [f"| Итерация | β | новых {('шагов' if 'lines' in metric else 'кадров')} "
           f"(сред.) | всего в D (сред.) | {name} на VAL | узлы |",
           "|---|---|---|---|---|---|"]
    for it in sorted(by):
        rs = by[it]
        b = _f(rs[0]['beta'])
        nn, _ = mean_std([_f(r['n_new']) for r in rs])
        nt, _ = mean_std([_f(r['n_total']) for r in rs])
        vm, vs = mean_std([_f(r[metric]) for r in rs])
        nd, _ = mean_std([_f(r.get('nodes')) for r in rs])
        out.append(f"| {it} | {b:.3f} | {fmt(nn, nd=0)} | {fmt(nt, nd=0)} | "
                   f"{fmt(vm, vs, 2)} | {fmt(nd, nd=0)} |")
    return "\n".join(out)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test over non-tied pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def boot_ci(xs, n_boot: int = 10000, seed: int = 0) -> str:
    """Percentile bootstrap 95% CI of the mean."""
    import random
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return '—'
    rng = random.Random(seed)
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs)
                for _ in range(n_boot))
    return f"[{ms[int(0.025 * n_boot)]:+.1f}; {ms[int(0.975 * n_boot)]:+.1f}]"


def replace_block(text, name, body):
    pat = re.compile(rf"(<!-- {name}:start -->)(.*?)(<!-- {name}:end -->)",
                     re.S)
    if not pat.search(text):
        return text
    return pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--compare', default=COMPARE_MD,
                    help='distill_compare/REPORT.md, or a copy of it')
    compare_md = ap.parse_args().compare
    with open(REPORT, encoding='utf-8') as fh:
        text = fh.read()

    order = ('BC', 'DAgger', 'BC+GAME', 'DAgger+GAME', 'EML-GAME')

    t_sum = read_csv(os.path.join(TETRIS_DIR, 'dagger_summary.csv'))
    t_main = [r for r in t_sum if not r.get('tag')]
    for r in t_main:
        r['n_games'] = '30'
    if t_main:
        text = replace_block(text, 'tetris-summary', summary_table(
            t_main, 'test_lines', 'линий за игру', order=order))
    if t_main:
        methods = [m for m in order if any(r['method'] == m for r in t_main)]
        cell = {(r['ga_seed'], r['method']): _f(r['test_lines']) for r in t_main}
        seeds = sorted({r['ga_seed'] for r in t_main}, key=int)
        out = ["| Сид ГА | " + " | ".join(METHOD_RU.get(m, m) for m in methods)
               + " | Δ (DAgger+GAME − BC+GAME) |",
               "|---" * (len(methods) + 2) + "|"]
        deltas = []
        for sd in seeds:
            d = cell.get((sd, 'DAgger+GAME'), math.nan) - \
                cell.get((sd, 'BC+GAME'), math.nan)
            deltas.append(d)
            out.append(f"| {sd} | " + " | ".join(
                fmt(cell.get((sd, m), math.nan), nd=1) for m in methods)
                + f" | {d:+.1f} |")
        dm, ds = mean_std(deltas)
        wins = sum(1 for d in deltas if d > 0)
        losses = sum(1 for d in deltas if d < 0)
        out.append(f"\nΔ в среднем {dm:+.1f} ± {ds:.1f} линий "
                   f"(95% бутстрэп-интервал среднего {boot_ci(deltas)}); "
                   f"DAgger+GAME лучше в {wins} из {len(deltas)} сидов, хуже в "
                   f"{losses}, точный знаковый тест p = "
                   f"{sign_test(wins, losses):.2f}.")
        d2 = [cell.get((sd, 'DAgger'), math.nan) - cell.get((sd, 'BC'), math.nan)
              for sd in seeds]
        m2, s2 = mean_std(d2)
        w2 = sum(1 for d in d2 if d > 0)
        l2 = sum(1 for d in d2 if d < 0)
        out.append(f"\nБез GAME-фазы: DAgger − BC в среднем {m2:+.1f} ± {s2:.1f}"
                   f" линий (95% бутстрэп {boot_ci(d2)}); DAgger лучше в {w2}, "
                   f"хуже в {l2} из {len(d2)}, знаковый тест p = "
                   f"{sign_test(w2, l2):.3f}.")
        text = replace_block(text, 'tetris-perseed', "\n".join(out))
    t_it = [r for r in read_csv(os.path.join(TETRIS_DIR, 'dagger_iterations.csv'))
            if not r.get('tag')]
    if t_it:
        text = replace_block(text, 'tetris-iterations',
                             iteration_table(t_it, 'val_avg_lines', 'линий'))
    t_abl = [r for r in t_sum if r.get('tag')]
    if t_abl:
        out = ["| Вариант | Метод | линий (test) | % от оракула | сидов |",
               "|---|---|---|---|---|"]
        by = defaultdict(list)
        for r in t_abl:
            by[(r['tag'], r['method'])].append(r)
        for (tag, m), rs in sorted(by.items()):
            vm, vs = mean_std([_f(r['test_lines']) for r in rs])
            pm, ps = mean_std([_f(r['pct_oracle']) for r in rs])
            out.append(f"| {tag} | {METHOD_RU.get(m, m)} | {fmt(vm, vs, 2)} | "
                       f"{fmt(pm, ps)} | {len(rs)} |")
        text = replace_block(text, 'tetris-ablation', "\n".join(out))

    f_sum = read_csv(os.path.join(FLAPPY_DIR, 'dagger_summary.csv'))
    for r in f_sum:
        r['n_games'] = '30'
    if f_sum:
        text = replace_block(text, 'flappy-summary', summary_table(
            f_sum, 'test_pipes', 'труб за игру', order=order,
            # In Flappy the current pipeline starts from random trees; the
            # +GAME rows are the same GAME phase warm-started from BC / DAgger.
            names={'BC+GAME': 'BC + GAME (тёплый старт от BC)',
                   'DAgger+GAME': 'DAgger + GAME (тёплый старт от DAgger)'}))
    f_it = read_csv(os.path.join(FLAPPY_DIR, 'dagger_iterations.csv'))
    if f_it:
        text = replace_block(text, 'flappy-iterations',
                             iteration_table(f_it, 'val_avg_pipes', 'труб'))

    if os.path.exists(compare_md):
        with open(compare_md, encoding='utf-8') as fh:
            cmp_text = fh.read()
        m = re.search(r"<!-- tables:start -->(.*?)<!-- tables:end -->",
                      cmp_text, re.S)
        body = (m.group(1).strip() if m else cmp_text.strip())
        text = replace_block(text, 'classic-compare', body)

    with open(REPORT, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print(f"updated {REPORT}")


if __name__ == '__main__':
    main()
