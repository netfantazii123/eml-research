"""
report.py — results/*.csv -> the tables of REPORT.md.

    python -m distill_compare.report

The generated tables are written between the markers

    <!-- tables:start -->  ...  <!-- tables:end -->

of REPORT.md; everything outside them is hand-written analysis and is kept as
is. Seeds are aggregated as mean ± sample std; `n` is the number of seeds.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, 'results')
REPORT = os.path.join(_HERE, 'REPORT.md')
START, END = '<!-- tables:start -->', '<!-- tables:end -->'

NAMES = {
    'teacher': 'Оракул (учитель)',
    'tree_d3': 'CART, глубина 3',
    'tree_d6': 'CART, глубина 6',
    'viper_d6': 'VIPER, глубина 6',
    'linear': 'Линейная модель',
    'kd_mlp_h16': 'KD-MLP, 16 нейронов',
    'gplearn_sr': 'Символьная регрессия (gplearn)',
    'classic_gp': 'ГП, классический базис (emlcore)',
    'eml_data': 'EML, только DATA-фаза',
    'classic_task': 'Классический базис, пайплайн диплома',
    'eml_task': '**EML, пайплайн диплома**',
}
ORDER = list(NAMES)
METRIC_RU = {'accuracy': 'accuracy', 'return': 'награда', 'score': 'труб',
             'lines': 'линий'}


def _load(suite):
    path = os.path.join(RESULTS, f'{suite}.csv')
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _ms(vals, fmt='{:.2f}'):
    vals = list(vals)
    v = np.array([x for x in vals if np.isfinite(x)])
    # No run may vanish from a statistic silently: a non-finite value (a crashed
    # or unscored run) is printed next to the mean of the others.
    miss = len(vals) - len(v)
    tail = f" (+{miss} без значения)" if miss else ''
    if len(v) == 0:
        return '—'           # not applicable to this row (e.g. the teacher)
    if len(v) == 1:
        return fmt.format(v[0]) + tail
    return f"{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}" + tail


def _group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r['task'], r['method'])].append(r)
    return g


def _table(rows, *, control: bool):
    g = _group(rows)
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    head = ['задача', 'метод', 'узлы/параметры', 'FLOP', 'трансц.',
            'метрика на test', '% от оракула', 'fidelity']
    if control:
        head.append('fidelity on-policy')
    head.append('игр/seed' if control else 'n test/seed')
    head.append('сидов')
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for t in tasks:
        methods = sorted({m for (tt, m) in g if tt == t},
                         key=lambda m: ORDER.index(m) if m in ORDER else 99)
        for m in methods:
            rs = g[(t, m)]
            if m == 'teacher':
                rs = rs[:1]   # every run of a task re-scores the same teacher
            unit = rs[0]['size_unit']
            size = _ms([_f(r['size']) for r in rs], '{:.0f}')
            size = f"{size} {'узл.' if unit == 'nodes' else 'пар.'}" if unit else '—'
            metric_name = METRIC_RU.get(rs[0]['metric_name'], rs[0]['metric_name'])
            row = [t, NAMES.get(m, m), size,
                   _ms([_f(r['flops']) for r in rs], '{:.0f}'),
                   _ms([_f(r['transcendental']) for r in rs], '{:.0f}'),
                   f"{_ms([_f(r['metric']) for r in rs], '{:.3f}' if metric_name == 'accuracy' else '{:.1f}')} {'' if metric_name == 'accuracy' else metric_name}".strip(),
                   _ms([_f(r['pct_of_teacher']) for r in rs], '{:.1f}'),
                   _ms([_f(r['fidelity']) for r in rs], '{:.3f}')]
            if control:
                row.append(_ms([_f(r['fidelity_onpolicy']) for r in rs], '{:.3f}'))
            row.append(rs[0]['n_eval'])
            row.append(str(len(rs)))
            out.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(out)


def _summary(all_rows):
    """Mean % of teacher per method and suite (over tasks, then seeds pooled)."""
    by = defaultdict(lambda: defaultdict(list))
    for suite, rows in all_rows.items():
        for (t, m), rs in _group(rows).items():
            if m == 'teacher':
                continue
            by[m][suite].append(np.mean([_f(r['pct_of_teacher']) for r in rs]))
    suites = [s for s in all_rows if all_rows[s]]
    head = ['метод'] + [f'{s}: средний % от оракула' for s in suites]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for m in sorted(by, key=lambda m: ORDER.index(m) if m in ORDER else 99):
        cells = [NAMES.get(m, m)]
        for s in suites:
            v = by[m].get(s)
            cells.append(f"{np.mean(v):.1f} (задач: {len(v)})" if v else '—')
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def build() -> str:
    all_rows = {s: _load(s) for s in ('tabular', 'gym', 'games')}
    parts = ['### Сводка', '',
             'Средний процент от оракула по задачам набора (для сред с '
             'управлением нормирован на случайную политику: '
             '(m − random) / (оракул − random)).', '', _summary(all_rows), '']
    titles = {'tabular': ('### Табличные датасеты', False),
              'gym': ('### Классические среды управления (gymnasium)', True),
              'games': ('### Игры проекта (Flappy Bird, Tetris)', True)}
    for s, (title, control) in titles.items():
        if all_rows[s]:
            parts += [title, '', _table(all_rows[s], control=control), '']
    return '\n'.join(parts)


def main():
    tables = build()
    if os.path.exists(REPORT):
        text = open(REPORT, encoding='utf-8').read()
    else:
        text = f"# Сравнение EML с классической дистилляцией\n\n{START}\n{END}\n"
    if START not in text:
        text += f"\n{START}\n{END}\n"
    pre, rest = text.split(START, 1)
    post = rest.split(END, 1)[1] if END in rest else ''
    text = f"{pre}{START}\n\n{tables}\n{END}{post}"
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"wrote {REPORT}")


if __name__ == '__main__':
    main()
