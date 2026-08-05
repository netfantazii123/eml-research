"""
main.py — project entry point.

    python main.py info                  dataset and feature information
    python main.py train --branch A      train the oracle (A: MLP, B: CNN)
    python main.py distill --branch A    distill the oracle into EML formulas
    python main.py distill --batch       a batch of runs, best by val
    python main.py report --branch A     plots into results/
    python main.py export --branch A     export to C (results/c_export/)
    python main.py verify-c              compile the C and compare with Python
    python main.py bench                 speed comparison oracle vs formulas
    python main.py calibrate             feature percentiles (for _SCALE_*)
    python main.py gui                   graphical interface (all of the above)

Order for a clean run: train -> distill -> export -> verify-c.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

import config


# ── info ──────────────────────────────────────────────────────────────────────

def cmd_info(args):
    import data
    import features as features_mod

    images, labels, _ = data.load_dataset()
    train_idx, val_idx, test_idx = data.stratified_split(labels)
    print()
    print(data.describe(labels, 'total'))
    print(data.describe(labels[train_idx], 'train'))
    print(data.describe(labels[val_idx], 'val  '))
    print(data.describe(labels[test_idx], 'test '))
    print(f"\nimage: {config.IMG_SIZE}x{config.IMG_SIZE}x{config.IMG_CHANNELS}")
    print(f"features: {config.N_FEATURES}")

    feats = features_mod.load_or_compute(images)
    print("\nFeature F-statistics on train (how well a feature separates classes):")
    scores = features_mod.anova_f(feats[train_idx], labels[train_idx])
    for i in np.argsort(-scores):
        print(f"  {config.FEATURE_NAMES[i]:>18}  F = {scores[i]:8.1f}")


# ── train ─────────────────────────────────────────────────────────────────────

def cmd_train(args):
    import reports
    import train_oracle

    out = train_oracle.train(args.branch, epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size, name=args.name)
    path = os.path.join(config.RESULTS_DIR, f'training_{args.branch.upper()}.png')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    reports.plot_training(out['history'],
                          f"Oracle, branch {args.branch.upper()} "
                          f"({out['meta']['model']})", path)
    print(f"[report] {path}")


# ── distill ───────────────────────────────────────────────────────────────────

def cmd_distill(args):
    import pipeline

    if args.batch:
        pipeline.run_batch(args.branch, n_variants=args.variants,
                           generations=args.generations,
                           population=args.population)
    elif args.branch == 'C':
        # Branch C has its own controls: islands, processes and run resumption.
        # Mixing them with the generations/population of branches A and B in one
        # flag set is fine, but pretending that --no-acc-phase affects it is not:
        # it has no ACC phase.
        pipeline.run_single(
            'C', size_penalty=args.depth_penalty, generations=args.generations,
            population=args.population, islands=args.islands,
            workers=args.workers, max_intervals=args.max_intervals,
            seed=args.seed, resume=not args.restart, checkpoint=args.checkpoint,
            name=args.name)
    else:
        pipeline.run(args.branch, depth_penalty=args.depth_penalty,
                     generations=args.generations, population=args.population,
                     acc_phase=not args.no_acc_phase, seed=args.seed,
                     name=args.name)


# ── report ────────────────────────────────────────────────────────────────────

def cmd_report(args):
    import eml_distiller as dist
    import reports
    import storage

    branch = args.branch.upper()
    clf, payload = storage.load_classifier(branch=branch)
    dist.configure_inputs(len(payload['input_names']), payload['input_names'])
    meta = payload.get('meta', {})
    metrics = meta.get('metrics', {})

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    for split in ('val', 'test'):
        if split not in metrics:
            continue
        path = os.path.join(config.RESULTS_DIR, f'confusion_{branch}_{split}.png')
        reports.plot_confusion(
            metrics[split]['confusion'],
            f"EML, branch {branch}, {split} "
            f"(bal.acc {metrics[split]['eml_bal_acc']:.3f})", path)
        print(f"[report] {path}")

    if hasattr(clf, 'decoder'):                  # branch C: a single formula
        print(f"\nf = {clf.tree.to_string()}")
        print("\nclass layout on the formula axis:")
        print(clf.decoder.describe(config.CLASS_NAMES))
    else:
        print("\nformulas:")
        for c, tree in enumerate(clf.trees):
            print(f"  f_{config.CLASS_NAMES[c]} = {tree.to_string()}")


# ── export ────────────────────────────────────────────────────────────────────

def cmd_export(args):
    import eml_distiller as dist
    import export
    import storage

    branch = args.branch.upper()
    clf, payload = storage.load_classifier(branch=branch)
    dist.configure_inputs(len(payload['input_names']), payload['input_names'])
    export.export_all(clf, payload['input_names'], payload.get('meta', {}),
                      branch=branch, out_dir=args.out)


# ── verify-c ──────────────────────────────────────────────────────────────────

_HARNESS = r'''#include <stdio.h>
#include <stdlib.h>
#include "aerial_features.h"
#include "aerial_eml_{branch}.h"

/* Test harness: reads raw frames, prints the features and the class. */
int main(int argc, char **argv)
{{
    FILE *fp;
    int n, k, i;
    uint8_t *buf;
    float f[AERIAL_N_FEATURES];

    if (argc < 3) {{ fprintf(stderr, "usage: harness <file> <n>\n"); return 1; }}
    fp = fopen(argv[1], "rb");
    if (!fp) {{ fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }}
    n = atoi(argv[2]);
    buf = (uint8_t *)malloc(AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3);

    for (k = 0; k < n; ++k) {{
        if (fread(buf, 1, AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3, fp)
            != (size_t)(AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3)) break;
        aerial_extract_features(buf, f);
        for (i = 0; i < AERIAL_N_FEATURES; ++i) printf("%.6f ", f[i]);
        printf("%d", aerial_classify(f));
#ifdef AERIAL_N_INTERVALS
        /* Branch C: also print the formula output itself. It shows how far
         * float32 diverged from float64 — and therefore whether a class
         * mismatch on a frame lying right next to a threshold is legitimate. */
        printf(" %.9g", aerial_formula(f));
#endif
        printf("\n");
    }}
    free(buf);
    fclose(fp);
    return 0;
}}
'''


def cmd_verify_c(args):
    """
    Generate the C, compile it, and compare against Python on real frames.

    Both halves of the export are checked: the features (numerically) and the
    resulting class (by decision agreement). A feature divergence beyond the
    tolerance means features.py and the template in export.py are out of sync —
    the most likely breakage after edits.
    """
    import data
    import eml_distiller as dist
    import export
    import features as features_mod
    import storage

    n = args.n
    branch = (args.branch or 'A').upper()   # the feature extractor exists for A and C
    clf, payload = storage.load_classifier(branch=branch)
    dist.configure_inputs(len(payload['input_names']), payload['input_names'])

    out_dir = args.out or os.path.join(config.RESULTS_DIR, 'c_export')
    export.export_all(clf, payload['input_names'], payload.get('meta', {}),
                      branch=branch, out_dir=out_dir, verbose=False)

    images, labels, _ = data.load_dataset(verbose=False)
    _, _, test_idx = data.stratified_split(labels)
    rng = np.random.default_rng(0)
    sample = test_idx[rng.choice(len(test_idx), min(n, len(test_idx)),
                                 replace=False)]
    batch = np.ascontiguousarray(images[sample], dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, 'frames.bin')
        batch.tofile(raw_path)
        harness_path = os.path.join(out_dir, 'harness.c')
        with open(harness_path, 'w', encoding='utf-8') as f:
            f.write(_HARNESS.format(branch=branch))

        exe = os.path.join(tmp, 'harness.exe' if os.name == 'nt' else 'harness')
        compile_cmd = ['gcc', '-O2', '-I', out_dir, harness_path, '-o', exe, '-lm']
        proc = subprocess.run(compile_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("COMPILATION FAILED:")
            print(proc.stderr)
            return 1

        proc = subprocess.run([exe, raw_path, str(len(batch))],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("EXECUTION FAILED:")
            print(proc.stderr)
            return 1

    n_features = config.N_FEATURES
    rows = [line.split() for line in proc.stdout.strip().splitlines()]
    c_feats = np.array([[float(v) for v in r[:n_features]] for r in rows])
    c_pred = np.array([int(r[n_features]) for r in rows])
    c_value = (np.array([float(r[n_features + 1]) for r in rows])
               if len(rows[0]) > n_features + 1 else None)

    py_feats = features_mod.extract_batch(batch)
    diff = np.abs(c_feats - py_feats)
    max_diff = float(diff.max())

    # A class mismatch is not necessarily an error. C computes in float, Python
    # in float64; on a frame where the decision is marginal the answers may
    # legitimately differ, and requiring agree == 1.0 would produce random red
    # runs where the export is correct. Mismatches are therefore analyzed rather
    # than merely counted.
    if c_value is not None:
        # Branch C: one number decides against a threshold table. A mismatch is
        # legitimate exactly when the float32/float64 difference in that number
        # is no smaller than the distance to the nearest threshold — i.e. the
        # frame lies in the zone of numerical uncertainty rather than being
        # classified differently.
        py_value = clf.raw(py_feats.astype(np.float64))
        py_pred = clf.decoder.decode(py_value)
        thresholds = clf.decoder.thresholds
        gaps = (np.abs(py_value[:, None] - thresholds[None, :]).min(axis=1)
                if len(thresholds) else np.full(len(py_value), np.inf))
        noise = np.abs(py_value - c_value)
        borderline_mask = gaps <= noise
        gap_label = "distance to the nearest threshold"
    else:
        py_scores = clf.scores(py_feats.astype(np.float64))
        py_pred = py_scores.argmax(axis=1)
        top2 = np.sort(py_scores, axis=1)[:, -2:]
        gaps = top2[:, 1] - top2[:, 0]
        borderline_mask = gaps < 1e-3
        gap_label = "gap between the two best classes"

    agree = float((c_pred == py_pred).mean())
    mismatch = np.flatnonzero(c_pred != py_pred)
    borderline = mismatch[borderline_mask[mismatch]]
    real_mismatch = mismatch[~borderline_mask[mismatch]]

    print(f"\nC VS PYTHON COMPARISON, branch {branch} ({len(batch)} frames)")
    if c_value is not None:
        rel = np.abs(py_value - c_value) / np.maximum(np.abs(py_value), 1.0)
        print(f"  max output divergence  : {float(rel.max()):.2e} "
              f"(relative)")
    print(f"  max feature divergence : {max_diff:.2e}")
    print(f"  mean divergence        : {float(diff.mean()):.2e}")
    print(f"  class agreement        : {agree:.4f} "
          f"({int((c_pred == py_pred).sum())}/{len(batch)})")
    if len(mismatch):
        print(f"  borderline mismatches  : {len(borderline)} "
              f"({gap_label} within numerical noise)")
        print(f"  real mismatches        : {len(real_mismatch)}")
        for i in real_mismatch[:5]:
            print(f"    frame {i}: C={config.CLASS_NAMES[c_pred[i]]} "
                  f"Python={config.CLASS_NAMES[py_pred[i]]} "
                  f"gap {gaps[i]:.3e}")

    worst = np.argsort(-diff.max(axis=0))[:5]
    print("  worst features:")
    for i in worst:
        print(f"    {config.FEATURE_NAMES[i]:>18} {diff[:, i].max():.2e}")

    # Tolerance 1e-4: C computes in float, Python in float32/float64; exact
    # bitwise agreement will not happen and is not required.
    ok = max_diff < 1e-4 and len(real_mismatch) == 0
    print(f"\n  {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── models ────────────────────────────────────────────────────────────────────

def cmd_models(args):
    """
    Model library: list, activate, rename, delete.

    The best in each group (kind × branch) is marked with a star and chosen BY
    VAL — the same metric used for selection within a run. Test is not used for
    selection: it is inspected once, at the very end.
    """
    import storage

    branch = args.branch.upper() if args.branch else None

    if args.activate:
        return _models_apply(storage, args.activate, branch, 'activate')
    if args.delete:
        return _models_apply(storage, args.delete, branch, 'delete')
    if args.rename:
        old, new = args.rename
        return _models_apply(storage, old, branch, 'rename', new)

    entries = storage.list_models(kind=args.kind, branch=branch)
    if not entries:
        print("\nthe library is empty. A model gets there by itself on "
              "`train` and `distill`.")
        print(f"directory: {storage.library_dir()}")
        return 0

    print(f"\nMODEL LIBRARY ({storage.library_dir()})")
    print(f"current dataset: {config.dataset_summary()}")
    print("★ best by val among applicable   • currently active   "
          "≠ different class set\n")
    header = (f"    {'kind':<7} {'branch':<6} {'name':<34} {'val bal':>8} "
              f"{'test bal':>9} {'complexity':>11}")
    print(header)
    print('-' * len(header))
    for entry in entries:
        marks = (('★' if entry.is_best else ' ')
                 + ('•' if entry.is_active else ' ')
                 + (' ' if entry.fits_dataset else '≠'))
        val = f"{entry.val_score:.4f}" if entry.val_score is not None else '—'
        test = f"{entry.test_score:.4f}" if entry.test_score is not None else '—'
        unit = 'par.' if entry.kind == 'oracle' else 'nodes'
        size = f"{entry.size} {unit}" if entry.size is not None else '—'
        kind = 'oracle' if entry.kind == 'oracle' else 'formula'
        print(f"{marks} {kind:<7} {entry.branch:<6} {entry.name:<34} "
              f"{val:>8} {test:>9} {size:>11}")

    alien = [e for e in entries if not e.fits_dataset]
    if alien:
        print(f"\n≠ {len(alien)} models were trained on a different class set "
              f"and do not apply to the current dataset.")
        print("  Their metrics are not comparable with the rest and they do "
              "not take part in best-model selection.")

    print("\ncurrently active:")
    for kind in storage.KINDS:
        for br in ('A', 'B', 'C'):
            if kind == 'oracle' and br == 'C':
                continue                     # branch C learns from the branch A oracle
            name = storage.active_name(kind, br)
            label = 'oracle ' if kind == 'oracle' else 'formula'
            shown = name or '— (slot empty or model not from the library)'
            print(f"  {label} {br}: {shown}")
    print("\nmake active:  python main.py models --activate <name>")
    return 0


def _models_apply(storage, name: str, branch: str | None, action: str,
                  new_name: str | None = None) -> int:
    """Find a model by name and perform an action on it."""
    matches = [e for e in storage.list_models(branch=branch) if e.name == name]
    if not matches:
        print(f"model \"{name}\" not found. List: python main.py models")
        return 1
    if len(matches) > 1:
        # Names are unique within a group (kind × branch) but not across groups.
        print(f"several models found under the name \"{name}\" — narrow it down "
              f"with --branch and --kind:")
        for e in matches:
            print(f"  {e.kind} branch {e.branch}")
        return 1

    entry = matches[0]
    if action == 'activate':
        path = storage.activate(entry)
        print(f"\"{entry.name}\" became active -> {path}")
    elif action == 'delete':
        if entry.is_active:
            print(f"\"{entry.name}\" is currently active. The active slot "
                  f"({storage.active_path(entry.kind, entry.branch)}) "
                  f"will stay in place, the model will be removed from the "
                  f"library.")
        storage.delete_model(entry)
        print(f"deleted: {entry.path}")
    else:
        renamed = storage.rename_model(entry, new_name)
        print(f"\"{entry.name}\" -> \"{renamed.name}\"")
    return 0


# ── bench ─────────────────────────────────────────────────────────────────────

def cmd_bench(args):
    import reports
    reports.benchmark(n_images=args.n)


# ── gui ───────────────────────────────────────────────────────────────────────

def cmd_gui(args):
    from gui import main as gui_main
    gui_main(initial_tab=args.tab)


# ── calibrate ─────────────────────────────────────────────────────────────────

def cmd_calibrate(args):
    """
    Check of the _SCALE_* constants in features.py, so they are not arbitrary.

    The scale can be wrong in both directions, and both are visible in the
    distribution:
        scale TOO SMALL — many values are clipped at the boundary, and different
                          images receive the same feature value;
        scale TOO LARGE — the distribution occupies a narrow part of [-1, 1],
                          and the feature wastes resolution (and ahead of the
                          exp() in the EML operator small differences are hard
                          to tell apart anyway).

    Only empirically scaled features are checked. For fractions and correlations
    the range boundaries are set by mathematics rather than by fitting: if a
    third of the frames have exactly zero dark pixels, that is a property of the
    dataset and there is nothing to fix.

    Computed on the train split ONLY. The effect on the constants themselves is
    negligible here (percentiles over 3.5 thousand and 5 thousand frames nearly
    coincide), but normalization scales are part of the model, and fitting them
    on val/test would mean looking into the held-out sample.
    """
    import data
    import features as features_mod

    images, labels, _ = data.load_dataset(verbose=False)
    train_idx, _, _ = data.stratified_split(labels)
    feats = features_mod.extract_batch(images[train_idx])
    print(f"\n(percentiles are computed on train: {len(train_idx)} of "
          f"{len(images)} images)")

    print("\nFEATURE SCALES")
    print("clipped = fraction of values at the [-1, 1] boundary; coverage = "
          "how much of the range the distribution occupies\n")
    issues = []
    for i, name in enumerate(config.FEATURE_NAMES):
        col = feats[:, i]
        saturated = float(((col <= -0.999) | (col >= 0.999)).mean())
        span = float(col.max() - col.min()) / 2.0
        empirical = name in features_mod.EMPIRICALLY_SCALED

        note = ''
        if empirical and saturated > 0.05:
            note = '  <-- scale too small'
            issues.append((name, note))
        elif empirical and span < 0.5:
            note = f'  <-- scale too large ~x{1 / max(span, 1e-6):.1f}'
            issues.append((name, note))

        kind = ' ' if empirical else '.'
        print(f" {kind}{name:>18}  clipped {saturated * 100:5.1f}%  "
              f"coverage {span * 100:5.1f}%  "
              f"[{col.min():+.3f}, {col.max():+.3f}]{note}")

    print("\n('.' = scale fixed by construction, not checked)")
    if issues:
        print(f"\nneed attention: {len(issues)}")
        print("changing a scale changes the features — after that the oracle "
              "must be retrained and distillation repeated.")
    else:
        print("\nall empirical scales are fine.")


# ── CLI ───────────────────────────────────────────────────────────────────────

BRANCHES = ['A', 'B', 'C', 'a', 'b', 'c']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python main.py',
        description='Distillation of an image classifier into EML formulas.')
    # The dataset is a common flag for all commands: it determines the number of
    # classes and everything that depends on it, so setting it inside individual
    # subcommands would mean some of them silently work with a different class
    # set.
    parser.add_argument('--dataset', default=None, metavar='FOLDER',
                        help='folder with class subdirectories; the choice is '
                             'remembered')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('info', help='dataset and feature information')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('train', help='train the oracle')
    p.add_argument('--branch', default='A', choices=['A', 'B', 'a', 'b'],
                   help='A: MLP over features, B: CNN over pixels')
    p.add_argument('--epochs', type=int, default=None)
    # The same controls as on the "Training" tab: run parameters rather than
    # edits to config — otherwise the CLI and the GUI compute different things
    # from the same numbers on screen.
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--name', default=None,
                   help='name of the library copy (default: time + metric)')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('distill', help='distill the oracle into EML formulas')
    p.add_argument('--branch', default='A', choices=BRANCHES)
    p.add_argument('--batch', action='store_true',
                   help='several runs, best by val')
    p.add_argument('--variants', type=int, default=None)
    p.add_argument('--generations', type=int, default=None)
    p.add_argument('--population', type=int, default=None)
    p.add_argument('--depth-penalty', default=None,
                   choices=list(config.DEPTH_PENALTIES.keys()),
                   help='formula size penalty (for branch C — per node)')
    p.add_argument('--no-acc-phase', action='store_true',
                   help='DATA phase only (branches A and B, for debugging)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', default=None,
                   help='name of the library copy (default: time + metric)')
    group = p.add_argument_group('branch C (long runs)')
    group.add_argument('--islands', type=int, default=None,
                       help='independent populations (0 = one per process)')
    group.add_argument('--workers', type=int, default=None,
                       help='processes (0 = one per core, 1 = no parallelism)')
    group.add_argument('--max-intervals', type=int, default=None,
                       help='cap on the number of axis intervals in the table')
    group.add_argument('--checkpoint', default=None,
                       help='state file (default runs/checkpoints/...)')
    group.add_argument('--restart', action='store_true',
                       help='start over instead of resuming the saved run')
    p.set_defaults(func=cmd_distill)

    p = sub.add_parser('report', help='plots into results/')
    p.add_argument('--branch', default='A', choices=BRANCHES)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('export', help='export the formulas to C')
    p.add_argument('--branch', default='A', choices=BRANCHES)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser('verify-c', help='compile the C and compare with Python')
    p.add_argument('-n', type=int, default=200, help='how many frames to compare')
    p.add_argument('--branch', default='A', choices=['A', 'C', 'a', 'c'],
                   help='the C feature export exists for branches A and C')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_verify_c)

    p = sub.add_parser('models', help='model library: list and selection')
    p.add_argument('--kind', default=None, choices=['oracle', 'eml'],
                   help='show only oracles or only formulas')
    p.add_argument('--branch', default=None, choices=BRANCHES)
    p.add_argument('--activate', metavar='NAME',
                   help='make a model active')
    p.add_argument('--rename', nargs=2, metavar=('NAME', 'NEW'),
                   help='rename a model')
    p.add_argument('--delete', metavar='NAME',
                   help='delete a model from the library')
    p.set_defaults(func=cmd_models)

    p = sub.add_parser('bench', help='speed comparison oracle vs formulas')
    p.add_argument('-n', type=int, default=500)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser('calibrate', help='check the normalization scales')
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser('gui', help='graphical interface')
    p.add_argument('--tab', default=None,
                   choices=['data', 'features', 'train', 'distill', 'models',
                            'test'],
                   help='tab opened at startup')
    p.set_defaults(func=cmd_gui)

    return parser


def apply_dataset(args) -> int:
    """
    Select the dataset: an explicit --dataset or the one remembered last time.

    An explicit choice is remembered immediately — otherwise the next command
    would work with a different class set, and half the pipeline would compute
    one thing and half another.
    """
    if args.dataset:
        try:
            info = config.use_dataset(args.dataset)
        except (ValueError, OSError) as exc:
            print(f"dataset rejected: {exc}")
            return 1
        config.save_dataset_choice()
        print(f"[data] dataset: {info['root']} "
              f"({info['n_classes']} classes: {', '.join(info['classes'])})")
    else:
        config.load_dataset_choice(verbose=True)
    return 0


def main():
    args = build_parser().parse_args()
    # For `models` the branch is optional (None = all branches), so check not
    # only that the attribute exists but also its value.
    if getattr(args, 'branch', None):
        args.branch = args.branch.upper()
    if apply_dataset(args):
        sys.exit(1)
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
