"""
make_controls.py — the tables of the 2026-11 control study, from raw results.

Every seed that finished is in the table; nothing is filtered. Sources:
  seed_study/results/{tetris_garbage,flappy_ramp}_{eml,classic,nooracle}/seed_*.json
  medium_tasks/results/<task>/seed_*.jsonl
    python report/make_controls.py > report/CONTROLS_TABLES.md
"""
import glob, json, os, collections
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = (('eml', 'EML, полный пайплайн'), ('classic', 'классический базис, полный пайплайн'),
        ('nooracle', 'без оракула (GAME-фаза из случайных EML)'))


def ms(v):
    v = np.asarray(v, float)
    return f'{v.mean():.1f} ± {v.std(ddof=1):.1f}' if len(v) > 1 else f'{v.mean():.1f}'


def game(name, score_key, std_key, unit, oracle_txt):
    print(f'\n### {name}\n\n{oracle_txt}\n')
    print(f'| вариант | сидов | TEST, {unit} (среднее ± std по сидам) | худший сид | лучший сид | узлов | трансц. | fit, мин |')
    print('|---|---|---|---|---|---|---|---|')
    per_seed = {}
    for arm, label in ARMS:
        rows = [json.load(open(f)) for f in sorted(glob.glob(
            f'{ROOT}/seed_study/results/{name}_{arm}/seed_*.json'))]
        if not rows:
            continue
        s = [r[score_key] for r in rows]
        tr = [r.get('static', {}).get('n_transcendental', float('nan')) for r in rows]
        print(f'| {label} | {len(rows)} | {ms(s)} | {min(s):.1f} | {max(s):.1f} | '
              f'{np.mean([r["size"] for r in rows]):.0f} | {"—" if np.all(np.isnan(tr)) else f"{np.nanmean(tr):.0f}"} | '
              f'{np.mean([r["fit_sec"] for r in rows]) / 60:.0f} |')
        per_seed[arm] = {r['ga_seed']: r[score_key] for r in rows}
    seeds = sorted(set().union(*[set(v) for v in per_seed.values()]))
    print('\nПо сидам:\n')
    print('| сид | ' + ' | '.join(a for a in per_seed) + ' |')
    print('|---' * (len(per_seed) + 1) + '|')
    for s in seeds:
        print(f'| {s} | ' + ' | '.join(f'{per_seed[a][s]:.1f}' if s in per_seed[a] else '—'
                                       for a in per_seed) + ' |')


def medium():
    print('\n### Средние задачи: % от оракула на TEST (50 эпизодов), 10 сидов\n')
    order = ('eml_task', 'classic_task', 'nooracle_task', 'eml_data', 'classic_gp',
             'gplearn_sr', 'kd_mlp_h16', 'tree_d6', 'tree_d3', 'linear')
    tasks = ('dock', 'swingup', 'highway', 'forager')
    d = collections.defaultdict(list)
    meta = {}
    for t in tasks:
        for f in glob.glob(f'{ROOT}/medium_tasks/results/{t}/seed_*.jsonl'):
            for l in open(f):
                r = json.loads(l)
                d[(t, r['method'])].append(r)
                meta[t] = (r['level'], r['teacher_metric'], r['random_metric'], r['metric_name'])
    print('| метод | ' + ' | '.join(f'{t} (ур. {meta[t][0]})' for t in tasks) + ' |')
    print('|---' * (len(tasks) + 1) + '|')
    for m in order:
        cells = []
        for t in tasks:
            rs = d[(t, m)]
            p = [r['pct_of_teacher'] for r in rs]
            cells.append(f'{ms(p)} (мин {min(p):.0f}, n={len(p)})' if rs else '—')
        print(f'| {m} | ' + ' | '.join(cells) + ' |')
    print('\nОракул и случайная политика (TEST, сырая метрика): ' + '; '.join(
        f'{t}: оракул {meta[t][1]:.1f}, случайная {meta[t][2]:.1f} ({meta[t][3]})' for t in tasks))
    print('\nВремя полного пайплайна, с (среднее по сидам) и узлов в формуле:\n')
    print('| метод | ' + ' | '.join(tasks) + ' |')
    print('|---' * (len(tasks) + 1) + '|')
    for m in order[:5]:
        print(f'| {m} | ' + ' | '.join(
            f'{np.mean([r["fit_sec"] for r in d[(t, m)]]):.0f} с, {np.mean([r["size"] for r in d[(t, m)]]):.0f} узл.,'
            f' {np.mean([r["transcendental"] for r in d[(t, m)]]):.0f} трансц.' for t in tasks) + ' |')


if __name__ == '__main__':
    o = json.load(open(f'{ROOT}/tetris/models/garbage_ppo.test.json'))
    game('tetris_garbage', 'test_lines', 'test_std', 'линий',
         f'Профиль garbage. PPO-оракул на тех же 30 TEST-играх: {o["avg_lines"]:.1f} ± {o["std_lines"]:.1f} линий, '
         f'упёрлось в лимит {o["n_capped"]}.')
    game('flappy_ramp', 'test_score', 'test_std', 'труб',
         'Профиль ramp. Оракул (models/ramp_ga.pt) на TEST: 58.3 ± 4.2 трубы.')
    medium()
