"""
run.py — run the classical distillation baselines and write comparable rows.

    python -m distill_compare.run tabular [--datasets banknote,wine] [--seeds 3]
    python -m distill_compare.run gym     [--envs CartPole-v1,...]
    python -m distill_compare.run flappy  [--eml-generations 60]
    python -m distill_compare.run tetris  [--oracle path/to/best_ppo.pt]
    python -m distill_compare.report      # results/*.csv -> REPORT.md

Rows are appended to distill_compare/results/<suite>.csv as each (task, method,
seed) finishes, so an interrupted run keeps what it has done. Columns:

    suite, task, method, seed, size, size_unit, flops, transcendental,
    fidelity, fidelity_onpolicy, metric_name, metric, metric_std,
    teacher_metric, pct_of_teacher, n_eval, fit_sec, formula

`fidelity` is agreement with the teacher's decision on held-out teacher-
distribution data; `fidelity_onpolicy` (control tasks only) is agreement on the
states the student itself visits during the test episodes — the number that
exposes compounding error, which off-policy cloning cannot see.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from distill_compare.students import make_student, DEFAULT_STUDENTS  # noqa: E402
from distill_compare import envs as E  # noqa: E402

RESULTS_DIR = os.path.join(_HERE, 'results')
FIELDS = ['suite', 'task', 'method', 'seed', 'size', 'size_unit', 'flops',
          'transcendental', 'fidelity', 'fidelity_onpolicy', 'metric_name',
          'metric', 'metric_std', 'teacher_metric', 'random_metric',
          'pct_of_teacher', 'n_eval',
          'fit_sec', 'formula']


def _writer(suite: str, out: str | None = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = out or os.path.join(RESULTS_DIR, f'{suite}.csv')
    new = not os.path.exists(path)
    f = open(path, 'a', newline='')
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()

    def write(row):
        w.writerow({k: row.get(k, '') for k in FIELDS})
        f.flush()
        print(f"  -> {row['task']:<16} {row['method']:<14} s{row['seed']} | "
              f"{row['metric_name']} {row['metric']:.3f} "
              f"({row['pct_of_teacher']:.1f}% of teacher) | fid "
              f"{row['fidelity']:.3f} | size {row['size']} {row['size_unit']} | "
              f"{row['fit_sec']:.1f}s", flush=True)
    return write, path


def _ga_kw(args):
    return {'generations': args.ga_generations, 'population': args.ga_population}


def _student_kw(name, args):
    if name in ('eml_data', 'classic_gp'):
        return _ga_kw(args)
    if name == 'gplearn_sr':
        return {'generations': args.gp_generations}
    return {}


# ── Tabular ──────────────────────────────────────────────────────────────────

def run_tabular(args):
    from sklearn.neural_network import MLPClassifier
    from distill_compare import datasets as D
    write, path = _writer('tabular', args.out)
    names = args.datasets.split(',') if args.datasets else D.TABULAR
    methods = args.methods.split(',') if args.methods else DEFAULT_STUDENTS
    for name in names:
        X, y, _ = D.load(name)
        for seed in range(args.seeds):
            tr, va, te = D.split(X, y, seed)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            Z = np.clip((X - mu) / sd, -5, 5)
            # Early stopping only where its 10 % hold-out is big enough to mean
            # anything: on wine (106 train rows) it stops on 11 rows and leaves
            # a teacher that its own students then "beat".
            teacher = MLPClassifier((128, 64), max_iter=2000, alpha=1e-3,
                                    early_stopping=len(tr) >= 2000,
                                    random_state=seed).fit(Z[tr], y[tr])
            K = len(np.unique(y))

            def logits(rows):
                P = np.zeros((len(rows), K))
                P[:, teacher.classes_] = teacher.predict_proba(Z[rows])
                return np.log(P + 1e-9)

            T_tr, T_te = logits(tr), logits(te)
            t_acc = float(np.mean(T_te.argmax(1) == y[te]))
            print(f"[{name}] seed {seed}: n={len(y)} d={X.shape[1]} K={K} | "
                  f"teacher test acc {t_acc:.3f}", flush=True)
            for m in methods:
                t0 = time.time()
                s = make_student(m, seed=seed, **_student_kw(m, args)).fit(
                    Z[tr], T_tr, kind='classify')
                fit = time.time() - t0
                pred = s.actions(Z[te])
                acc = float(np.mean(pred == y[te]))
                c = s.complexity()
                write({'suite': 'tabular', 'task': name, 'method': s.name,
                       'seed': seed, **c,
                       'fidelity': float(np.mean(pred == T_te.argmax(1))),
                       'metric_name': 'accuracy', 'metric': acc, 'metric_std': 0.0,
                       'teacher_metric': t_acc,
                       'pct_of_teacher': 100 * acc / max(t_acc, 1e-9),
                       'n_eval': len(te), 'fit_sec': fit,
                       'formula': s.describe()[:2000]})
    return path


# ── Control (gym, flappy, tetris) ────────────────────────────────────────────

def _eval_control(adapter, student, test_seeds, held_out):
    metrics, onp = E.rollout(adapter, E.student_policy(adapter, student),
                             test_seeds, collect=True)
    return (float(np.mean(metrics)), float(np.std(metrics, ddof=1)),
            E.fidelity(adapter.kind, student, held_out),
            E.fidelity(adapter.kind, student, onp))


def run_control(adapter, *, suite, task, args, data_seeds, heldout_seeds,
                train_seeds, val_seeds, test_seeds, label_every=1, extra=None):
    """Teacher data -> every student -> VIPER -> extra (task-native EML)."""
    from distill_compare.viper import viper
    write, path = _writer(suite, args.out)
    methods = [] if args.methods == 'none' else args.methods.split(',') if args.methods else (
        list(DEFAULT_STUDENTS) + ['viper_d6'])
    t0 = time.time()
    t_metrics, data = E.rollout(adapter, E.teacher_policy(adapter), data_seeds,
                                collect=True, label_every=label_every)
    _, held = E.rollout(adapter, E.teacher_policy(adapter), heldout_seeds,
                        collect=True, label_every=max(1, label_every))
    t_test, _ = E.rollout(adapter, E.teacher_policy(adapter), test_seeds)
    t_mean = float(np.mean(t_test))
    # % of teacher is normalised against a uniform-random policy on the same
    # seeds: (m - random) / (teacher - random). Returns of Acrobot / LunarLander
    # are negative, where a plain ratio means nothing; on Flappy and Tetris the
    # random policy scores ~0 and this reduces to the plain ratio.
    rng_r = np.random.default_rng(0)
    if adapter.kind == 'rank':
        def rand_pol(obs):
            return int(rng_r.integers(len(adapter.inputs(obs)[1])))
    else:
        def rand_pol(obs):
            return int(rng_r.integers(adapter.n_actions))
    r_test, _ = E.rollout(adapter, rand_pol, test_seeds)
    r_mean = float(np.mean(r_test))
    span = t_mean - r_mean if abs(t_mean - r_mean) > 1e-9 else 1.0

    def pct(m):
        return 100.0 * (m - r_mean) / span
    print(f"[{task}] teacher: data {len(data['X'])} rows from {len(data_seeds)} "
          f"episodes ({time.time() - t0:.0f}s) | test {adapter.metric_name} "
          f"{t_mean:.2f} ± {np.std(t_test, ddof=1):.2f} | random {r_mean:.2f}",
          flush=True)
    write({'suite': suite, 'task': task, 'method': 'teacher', 'seed': 0,
           'size': '', 'size_unit': '', 'fidelity': 1.0, 'fidelity_onpolicy': 1.0,
           'metric_name': adapter.metric_name, 'metric': t_mean,
           'metric_std': float(np.std(t_test, ddof=1)), 'teacher_metric': t_mean,
           'random_metric': r_mean, 'pct_of_teacher': 100.0, 'n_eval': len(test_seeds), 'fit_sec': 0.0})

    def emit(student, seed, fit, method=None):
        m, sd, fid, fid_on = _eval_control(adapter, student, test_seeds, held)
        write({'suite': suite, 'task': task, 'method': method or student.name,
               'seed': seed, **student.complexity(), 'fidelity': fid,
               'fidelity_onpolicy': fid_on, 'metric_name': adapter.metric_name,
               'metric': m, 'metric_std': sd, 'teacher_metric': t_mean,
               'random_metric': r_mean, 'pct_of_teacher': pct(m),
               'n_eval': len(test_seeds), 'fit_sec': fit,
               'formula': student.describe()[:2000]})

    for seed in range(args.seed_offset, args.seed_offset + args.seeds):
        for m in methods:
            t0 = time.time()
            if m.startswith('viper_d'):
                s, _ = viper(adapter, max_depth=int(m[7:]), n_iters=args.viper_iters,
                             episodes_per_iter=args.viper_episodes,
                             train_seeds=train_seeds, val_seeds=val_seeds,
                             seed=seed, initial=data, label_every=label_every)
            else:
                s = make_student(m, seed=seed, **_student_kw(m, args)).fit(
                    data['X'], data['T'], kind=adapter.kind, groups=data['groups'])
            emit(s, seed, time.time() - t0)
        if extra is not None:
            t0 = time.time()
            s = extra(seed, data)
            if s is not None:
                emit(s, seed, time.time() - t0)
    return path


def run_gym(args):
    from distill_compare import ppo
    ids = args.envs.split(',') if args.envs else list(ppo.SETTINGS)
    for env_id in ids:
        teacher = ppo.load_or_train(env_id)
        a = E.GymAdapter(env_id, teacher)
        run_control(a, suite='gym', task=env_id, args=args,
                    data_seeds=range(10_000, 10_000 + args.data_episodes),
                    heldout_seeds=range(40_000, 40_010),
                    train_seeds=list(range(20_000, 20_200)),
                    val_seeds=list(range(30_000, 30_010)),
                    test_seeds=list(range(50_000, 50_000 + args.test_episodes)))


class _TreeAsStudent:
    """A task-native EML tree behind the Student interface (for evaluation)."""

    def __init__(self, tree, kind, threshold=None, name='eml_task'):
        self.tree, self.kind, self.threshold, self.name = tree, kind, threshold, name

    def scores(self, X):
        with np.errstate(all='ignore'):
            v = self.tree.evaluate_batch(np.asarray(X, dtype=np.float64))
        if self.kind == 'rank':
            return np.where(np.isfinite(v), v, -np.inf)
        v = np.where(np.isfinite(v), v, -np.inf)
        return np.stack([np.full_like(v, self.threshold), v], 1)

    def complexity(self):
        from emlcore import static_metrics
        m = static_metrics(self.tree)
        return {'size': m['nodes'], 'size_unit': 'nodes',
                'flops': m['n_ops'] + 2 * m['n_eml'],
                'transcendental': m['n_transcendental']}

    def describe(self):
        return self.tree.to_string()


def run_flappy(args):
    a = E.FlappyAdapter(max_frames=args.max_frames)
    E._import_task('flappy_bird_with_ai')
    import config as fcfg

    def eml_task(seed, data):
        """The thesis pipeline itself: DATA warm-up + GAME phase, VAL selection."""
        if args.skip_eml_task:
            return None
        import random
        import eml_distiller as ed
        from emlcore import classic_basis
        random.seed(seed)
        np.random.seed(seed)
        if args.task_basis == 'classic':
            # Flappy's evolve_eml builds its basis through make_basis(); the
            # control run swaps that one function and nothing else, so the
            # search, budget and seeds are the same as the EML run.
            ed.make_basis = lambda max_depth=None, max_nodes=None: classic_basis(
                fcfg.STATE_SIZE, ed.VAR_NAMES,
                max_depth=fcfg.EML_MAX_DEPTH if max_depth is None else max_depth,
                epsilon=fcfg.EML_EPSILON, clamp=fcfg.EML_CLAMP, max_nodes=max_nodes)
        # Sigmoid targets, as eml_distiller.collect_dataset produces them.
        p = 1.0 / (1.0 + np.exp(-(data['T'][:, 1] - data['T'][:, 0])))
        res = ed.evolve_eml(data['X'].astype(np.float32), p, mode='sigmoid',
                            generations=args.eml_generations,
                            population_size=args.eml_population,
                            patience=max(10, args.eml_generations // 4),
                            verbose=False)
        return _TreeAsStudent(res['best_tree'], 'classify', threshold=0.5,
                              name=f'{args.task_basis}_task')

    run_control(a, suite='games', task='flappy', args=args,
                data_seeds=[fcfg.EML_DATA_SEED * 100_000 + i
                            for i in range(args.data_episodes)],
                heldout_seeds=[fcfg.EML_DATA_SEED * 100_000 + 50_000 + i
                               for i in range(5)],
                train_seeds=list(fcfg.EML_SEED_POOL_TRAIN),
                val_seeds=list(fcfg.EML_SEED_POOL_VAL[:8]),
                test_seeds=list(fcfg.EML_SEED_POOL_TEST[:args.test_episodes]),
                label_every=2, extra=eml_task)


def run_tetris(args):
    a = E.TetrisAdapter(oracle_path=args.oracle)
    tcfg = a.config

    def eml_task(seed, data):
        if args.skip_eml_task:
            return None
        import random
        import eml_distiller as ed
        random.seed(tcfg.EML_GA_SEED_BASE + seed)
        np.random.seed(tcfg.EML_GA_SEED_BASE + seed)
        res = ed.distill(data['X'].astype(np.float32), data['T'].astype(np.float32),
                         groups=data['groups'], verbose=False,
                         basis_name=args.task_basis,
                         data_generations=args.eml_generations,
                         data_population=args.eml_population)
        return _TreeAsStudent(res['trees'][0], 'rank', name=f'{args.task_basis}_task')

    run_control(a, suite='games', task='tetris', args=args,
                data_seeds=range(args.data_episodes),
                heldout_seeds=range(900, 903),
                train_seeds=list(tcfg.EML_SEED_POOL_TRAIN),
                val_seeds=list(tcfg.EML_SEED_POOL_VAL[:8]),
                test_seeds=list(tcfg.EML_SEED_POOL_TEST[:args.test_episodes]),
                extra=eml_task)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('suite', choices=['tabular', 'gym', 'flappy', 'tetris'])
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--methods', default='')
    p.add_argument('--datasets', default='')
    p.add_argument('--envs', default='')
    p.add_argument('--out', default=None, help='CSV path (default results/<suite>.csv)')
    p.add_argument('--ga-generations', type=int, default=120)
    p.add_argument('--ga-population', type=int, default=300)
    p.add_argument('--gp-generations', type=int, default=20)
    p.add_argument('--data-episodes', type=int, default=30)
    p.add_argument('--test-episodes', type=int, default=30)
    p.add_argument('--viper-iters', type=int, default=8)
    p.add_argument('--viper-episodes', type=int, default=10)
    p.add_argument('--max-frames', type=int, default=5000, help='flappy only')
    p.add_argument('--oracle', default=None, help='tetris oracle .pt')
    p.add_argument('--eml-generations', type=int, default=100)
    p.add_argument('--eml-population', type=int, default=200)
    p.add_argument('--skip-eml-task', action='store_true')
    p.add_argument('--task-basis', default='eml', choices=['eml', 'classic'],
                   help='basis of the task-native pipeline run (classic = control)')
    p.add_argument('--seed-offset', type=int, default=0,
                   help='first seed (to add replicates to an existing CSV)')
    args = p.parse_args(argv)
    {'tabular': run_tabular, 'gym': run_gym, 'flappy': run_flappy,
     'tetris': run_tetris}[args.suite](args)


if __name__ == '__main__':
    main()
