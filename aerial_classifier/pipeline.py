"""
pipeline.py — the full distillation run and evaluation of the result.

Ties the three project phases together: takes a trained oracle, collects the
dataset (inputs -> oracle logits), evolves the EML formulas, and measures how
much quality is lost by replacing the neural network with a formula.

There is not one key metric here but three, and they must not be confused:

    accuracy    — fraction of correct answers against the GROUND TRUTH labels.
                  Answers the question "does the formula work".
    fidelity    — fraction of matches with the ORACLE's answer.
                  Answers the question "does the formula reproduce the oracle".
    bal_accuracy— accuracy averaged over classes.
                  The primary metric: the dataset is imbalanced 3.4:1, and plain
                  accuracy overrates a model that simply likes ROADS.

A formula can have high accuracy and low fidelity — then it solves the task in
its own way rather than copying the oracle. That is a result in itself rather
than an error, so both quantities are computed and printed.
"""

import json
import os
import shutil
import time

import numpy as np

import config
import data
import eml_distiller as dist
import features as features_mod
import oracle as oracle_mod
import storage


# ── Collecting the distillation dataset ───────────────────────────────────────

def oracle_branch_of(branch: str) -> str:
    """
    Whose oracle a branch distills.

    Branch C works in the same feature space as A and learns from the same MLP:
    otherwise the "one formula against four" comparison would also measure a
    difference between oracles.
    """
    return 'B' if branch.upper() == 'B' else 'A'


def collect_dataset(branch: str = 'A', verbose: bool = True) -> dict:
    """
    Collect everything the distiller needs: formula inputs, oracle logits, labels.

    Branches A and C: inputs = 32 engineered features, oracle = FeatureMLP.
    Branch B: inputs = the bottleneck embedding of the trained CNN.

    Returns a dict with keys inputs / logits / labels / splits / meta.
    """
    branch = branch.upper()
    images, labels, _ = data.load_dataset(verbose=verbose)
    train_idx, val_idx, test_idx = data.stratified_split(labels)
    device = oracle_mod.get_device()

    if oracle_branch_of(branch) == 'A':
        model = oracle_mod.FeatureMLP()
        meta = storage.load_oracle(model, branch='A', device=device)
        inputs = features_mod.load_or_compute(images, verbose=verbose)
        logits = oracle_mod.predict_logits(model, inputs, device)
        input_names = list(config.FEATURE_NAMES)
    else:
        model = oracle_mod.PixelCNN()
        meta = storage.load_oracle(model, branch='B', device=device)
        inputs = oracle_mod.predict_embeddings(model, images, device)
        logits = oracle_mod.predict_logits(model, images, device)
        input_names = [f'e{i}' for i in range(inputs.shape[1])]

    if verbose:
        print(f"[collect] branch {branch}: inputs {inputs.shape}, "
              f"logits {logits.shape}")
        print(f"[collect] oracle: val {meta.get('val_acc', float('nan')):.4f}, "
              f"test {meta.get('test_acc', float('nan')):.4f}")

    return {
        'inputs': inputs, 'logits': logits, 'labels': labels,
        'splits': (train_idx, val_idx, test_idx),
        'input_names': input_names, 'oracle_meta': meta, 'branch': branch,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(clf: dist.EMLClassifier, inputs: np.ndarray, logits: np.ndarray,
             labels: np.ndarray, idx: np.ndarray) -> dict:
    """Metrics of the formulas and the oracle on one split."""
    X, y = inputs[idx], labels[idx]
    eml_pred = clf.predict(X)
    oracle_pred = logits[idx].argmax(axis=1)
    return {
        'n': int(len(idx)),
        'eml_acc': float((eml_pred == y).mean()),
        'eml_bal_acc': dist.balanced_accuracy(y, eml_pred),
        'oracle_acc': float((oracle_pred == y).mean()),
        'oracle_bal_acc': dist.balanced_accuracy(y, oracle_pred),
        'fidelity': float((eml_pred == oracle_pred).mean()),
        'confusion': confusion_matrix(y, eml_pred).tolist(),
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int | None = None) -> np.ndarray:
    """Confusion matrix (n_classes, n_classes), rows = ground truth."""
    n_classes = n_classes or config.N_CLASSES
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


# ── Full run ──────────────────────────────────────────────────────────────────

def archive_run(trees: list, branch: str, meta: dict, input_names: list[str], *,
                run_dir: str | None = None, name: str | None = None,
                filename: str | None = None, decoder: dict | None = None):
    """
    Write the run into runs/ and place a named copy into the library.

    Decides nothing about being "best" — it only saves. The separation from
    promote_if_better is needed by the batch: there ALL variants go into the
    library, while only one claims the active slot.

    Returns: (ModelEntry, path of the copy in runs/).
    """
    branch = branch.upper()
    run_dir = run_dir or os.path.join(
        config.RUNS_DIR, time.strftime('%Y%m%d_%H%M%S') + f'_run_{branch}')
    os.makedirs(run_dir, exist_ok=True)

    # The name is written into the passed dict as well. Previously there was
    # only a copy here and the caller (and through it the GUI) never saw the
    # name: the run panel showed "in library: —" after every successful
    # distillation.
    meta['library_name'] = storage.reserve_name('eml', branch, name, meta)
    run_path = storage.save_formulas(
        trees, branch=branch, meta=dict(meta), input_names=input_names,
        decoder=decoder,
        path=os.path.join(run_dir, filename or f'formulas_{branch}.json'))
    entry = storage.add_to_library('eml', branch, meta['library_name'],
                                   dict(meta), source_path=run_path,
                                   overwrite=True)
    return entry, run_path


def promote_if_better(entry, *, verbose: bool = True) -> dict:
    """
    Make a model active only if it is not worse than the current one BY VAL.

    Previously any run with save=True (the default in both the GUI and the CLI)
    silently overwrote models/best_eml_X.json. One unlucky seed and the
    batch-selected formula set was lost without a trace.

    Rules:
      * comparison by val only — test is inspected once, at the very end;
      * the previous file is kept alongside as .prev.json;
      * a run interrupted by the user never claims the active slot.
    """
    branch = entry.branch
    new_val = entry.val_score if entry.val_score is not None else -1.0
    prev_val = storage.formulas_meta(branch).get(
        'metrics', {}).get('val', {}).get('eml_bal_acc')
    best_path = storage.formulas_path(branch)

    if entry.meta.get('stopped'):
        promoted, reason = False, "run was interrupted by the user"
    elif prev_val is None:
        promoted, reason = True, "there were no saved formulas"
    elif new_val >= float(prev_val):
        promoted = True
        reason = f"val {new_val:.4f} >= previous {float(prev_val):.4f}"
    else:
        promoted = False
        reason = f"val {new_val:.4f} < previous {float(prev_val):.4f}"

    if promoted:
        if os.path.exists(best_path):
            shutil.copyfile(best_path, best_path.replace('.json', '.prev.json'))
        storage.activate(entry)

    if verbose:
        if promoted:
            print(f"[distill] \"{entry.name}\" became active ({reason})"
                  f" -> {best_path}")
        else:
            print(f"[distill] active model NOT touched ({reason})")
    return {'promoted': promoted, 'reason': reason,
            'best_path': best_path if promoted else None,
            'entry': entry, 'library_name': entry.name}


def save_best(trees: list, branch: str, meta: dict, input_names: list[str], *,
              verbose: bool = True, run_dir: str | None = None,
              name: str | None = None, decoder: dict | None = None) -> dict:
    """
    Save the whole run: an archive in runs/, a copy in the library, and the
    active slot if the run is not worse than the previous one.

    Returns: {'run_path', 'best_path' | None, 'promoted', 'reason', 'entry'}.
    """
    entry, run_path = archive_run(trees, branch, meta, input_names,
                                  run_dir=run_dir, name=name, decoder=decoder)
    if verbose:
        print(f"\n[distill] run -> {run_path}")
        print(f"[distill] into the library as \"{entry.name}\" -> {entry.path}")
    result = promote_if_better(entry, verbose=verbose)
    result['run_path'] = run_path
    return result


def checkpoint_path(branch: str, seed: int) -> str:
    """Where the state of a long run lives so that it can be resumed."""
    return os.path.join(config.RUNS_DIR, 'checkpoints',
                        f'eml_{branch.upper()}_seed{seed}.json')


def run_single(branch: str = 'C', *, size_penalty: str | None = None,
               generations: int | None = None, population: int | None = None,
               islands: int | None = None, workers: int | None = None,
               max_intervals: int | None = None, seed: int = 0,
               resume: bool = True, checkpoint: str | None = None,
               save: bool = True, verbose: bool = True,
               progress_cb=None, should_stop=None,
               name: str | None = None) -> dict:
    """
    A branch C run: one formula plus a threshold table fitted to it.

    THE TRAINING TARGET IS THE ORACLE'S ANSWERS, not the ground truth labels.
    This is still distillation, as in stages 1 and 2: the formula reproduces the
    network's decision function, and its fidelity is a meaningful quantity.
    Ground truth labels only go into the report.

    The run is designed for hours: state is written to checkpoint, and
    resume=True continues an interrupted run from the last save.
    """
    import eml_single as single

    branch = branch.upper()
    t0 = time.time()
    penalty_name = size_penalty or config.EMLC_SIZE_PENALTY_DEFAULT
    bundle = collect_dataset(branch, verbose=verbose)
    inputs, logits, labels = bundle['inputs'], bundle['logits'], bundle['labels']
    train_idx, val_idx, test_idx = bundle['splits']
    oracle_labels = logits.argmax(axis=1)

    dist.configure_inputs(inputs.shape[1], bundle['input_names'])
    ckpt = checkpoint if checkpoint is not None else checkpoint_path(branch, seed)

    if verbose:
        print(f"\n[distill] branch {branch}: ONE formula for {config.N_CLASSES} "
              f"classes, {inputs.shape[1]} inputs, {len(train_idx)} examples")
        print(f"[distill] target — oracle answers (distillation), penalty "
              f"{penalty_name}, checkpoint {ckpt}")

    result = single.evolve_single(
        inputs[train_idx], oracle_labels[train_idx],
        n_classes=config.N_CLASSES,
        generations=generations, population=population, islands=islands,
        workers=workers, max_intervals=max_intervals,
        size_penalty=config.EMLC_SIZE_PENALTIES[penalty_name],
        checkpoint_path=ckpt, resume=resume,
        input_names=bundle['input_names'], rng_seed=seed, verbose=verbose,
        progress_cb=progress_cb, should_stop=should_stop)

    model = result['model']
    metrics = {name_: evaluate(model, inputs, logits, labels, idx)
               for name_, idx in (('train', train_idx), ('val', val_idx),
                                  ('test', test_idx))}
    used = model.used_inputs()
    meta = {
        'branch': branch,
        'oracle': bundle['oracle_meta'],
        'oracle_branch': oracle_branch_of(branch),
        'metrics': metrics,
        'total_nodes': model.total_nodes(),
        'max_depth': model.max_depth(),
        'n_inputs_available': inputs.shape[1],
        'n_inputs_used': len(used),
        'inputs_used': [bundle['input_names'][i] for i in used],
        'decoder': model.decoder.to_dict(),
        'n_intervals': model.decoder.n_intervals,
        'class_layout': model.decoder.describe(config.CLASS_NAMES),
        'depth_penalty': penalty_name,          # same field as in branches A and B
        'size_penalty': config.EMLC_SIZE_PENALTIES[penalty_name],
        'train_bal_acc_final': result['train_score'],
        'best_fitness': result['best_fitness'],
        'generations_run': result['generation'],
        'islands': result['islands'],
        'workers': result['workers'],
        'seed': seed,
        'stopped': result.get('stopped', False),
        'distill_seconds': result['seconds'],
        'total_seconds': time.time() - t0,
    }

    saved = None
    if save:
        saved = save_best([model.tree], branch, meta, bundle['input_names'],
                          verbose=verbose, name=name,
                          decoder=model.decoder.to_dict())

    # The checkpoint lives exactly as long as the run is unfinished. Keeping it
    # after normal completion would mean the next run with the same seed
    # silently continues an already finished one and immediately hits patience.
    if not result.get('stopped') and os.path.exists(ckpt):
        os.remove(ckpt)

    if verbose:
        print_report_single(meta, model)

    return {'trees': [model.tree], 'classifier': model, 'model': model,
            'meta': meta, 'result': result, 'bundle': bundle, 'saved': saved}


def print_report_single(meta: dict, model):
    """Branch C run summary: metrics, complexity, and class layout on the axis."""
    m = meta['metrics']
    print("\n" + "=" * 72)
    print(f"BRANCH {meta['branch']}: oracle -> ONE EML formula + "
          f"{meta['n_intervals']} axis intervals")
    print("=" * 72)
    print(f"{'split':<8} {'accuracy':>18} {'bal.accuracy':>20} {'fidelity':>10}")
    print(f"{'':8} {'EML':>8} {'oracle':>9} {'EML':>9} {'oracle':>10} {'':>10}")
    for split in ('train', 'val', 'test'):
        s = m[split]
        print(f"{split:<8} {s['eml_acc']:>8.4f} {s['oracle_acc']:>9.4f} "
              f"{s['eml_bal_acc']:>9.4f} {s['oracle_bal_acc']:>10.4f} "
              f"{s['fidelity']:>10.4f}")

    test = m['test']
    retained = (test['eml_bal_acc'] / test['oracle_bal_acc'] * 100
                if test['oracle_bal_acc'] > 0 else 0.0)
    print(f"\nthe formula holds {retained:.1f}% of the oracle's balanced "
          f"accuracy on test")
    print(f"complexity: {meta['total_nodes']} AST nodes, depth "
          f"{meta['max_depth']}, inputs used "
          f"{meta['n_inputs_used']}/{meta['n_inputs_available']}, "
          f"intervals {meta['n_intervals']}")
    print(f"used: {', '.join(meta['inputs_used'])}")

    print("\nEML confusion matrix on test (rows = ground truth):")
    header = ' ' * 9 + ''.join(f"{c[:6]:>8}" for c in config.CLASS_NAMES)
    print(header)
    for i, row in enumerate(test['confusion']):
        print(f"{config.CLASS_NAMES[i]:>8} " + ''.join(f"{v:>8}" for v in row))

    # The point of the branch: where the formula placed the classes itself.
    # Classes adjacent on the axis should be confused more often than others —
    # verifiable against the confusion matrix above.
    print("\nclass layout on the formula axis:")
    print(meta['class_layout'])
    print(f"\nf = {model.tree.to_string()}")
    print("=" * 72)


def run(branch: str = 'A', *, depth_penalty: str | None = None,
        generations: int | None = None, population: int | None = None,
        acc_phase: bool | None = None, seed: int = 0,
        save: bool = True, verbose: bool = True,
        progress_cb=None, should_stop=None, name: str | None = None,
        **single_kwargs) -> dict:
    """
    Collect the dataset -> distill -> evaluate -> save the formulas.

    progress_cb / should_stop are forwarded into the evolution: the first gives
    live curves, the second a soft stop that keeps the best result so far.

    Branch C is built differently (one formula instead of a set, a threshold
    table instead of calibration) and goes to run_single; it is intercepted here
    so that the CLI, the GUI and the batch keep a single entry point.

    Returns: dict with the formulas, metrics over all splits, and metadata.
    """
    branch = branch.upper()
    if branch == 'C':
        return run_single(branch, size_penalty=depth_penalty,
                          generations=generations, population=population,
                          seed=seed, save=save, verbose=verbose,
                          progress_cb=progress_cb, should_stop=should_stop,
                          name=name, **single_kwargs)
    t0 = time.time()
    bundle = collect_dataset(branch, verbose=verbose)
    inputs, logits, labels = bundle['inputs'], bundle['logits'], bundle['labels']
    train_idx, val_idx, test_idx = bundle['splits']

    # The formulas work in the input space of a particular branch — tell the
    # distiller about it before evolving, otherwise var_idx indexes the wrong
    # thing.
    dist.configure_inputs(inputs.shape[1], bundle['input_names'])

    if verbose:
        print(f"\n[distill] branch {branch}, {inputs.shape[1]} inputs, "
              f"{len(train_idx)} training examples")

    result = dist.distill(
        inputs[train_idx], logits[train_idx], labels[train_idx],
        depth_penalty_name=depth_penalty, data_generations=generations,
        data_population=population, acc_phase=acc_phase,
        verbose=verbose, rng_seed=seed,
        progress_cb=progress_cb, should_stop=should_stop,
    )

    # The calibration is already frozen by the distiller on train — take the
    # classifier as is without rebuilding it: recomputing on val/test would be
    # looking ahead.
    clf = result['classifier']
    metrics = {
        'train': evaluate(clf, inputs, logits, labels, train_idx),
        'val': evaluate(clf, inputs, logits, labels, val_idx),
        'test': evaluate(clf, inputs, logits, labels, test_idx),
    }

    used = result['used_inputs']
    meta = {
        'branch': branch,
        'oracle': bundle['oracle_meta'],
        'metrics': metrics,
        'total_nodes': result['total_nodes'],
        'max_depth': result['max_depth'],
        'n_inputs_available': inputs.shape[1],
        'n_inputs_used': len(used),
        'inputs_used': [bundle['input_names'][i] for i in used],
        'depth_penalty': result['depth_penalty'],
        'calibration': result['calibration'],
        'train_bal_acc_data_phase': result['train_bal_acc_data'],
        'train_bal_acc_final': result['train_bal_acc_final'],
        'seed': seed,
        'stopped': result.get('stopped', False),
        'generations': generations,
        'population': population,
        'distill_seconds': result['seconds'],
        'total_seconds': time.time() - t0,
    }

    saved = None
    if save:
        saved = save_best(result['trees'], branch, meta, bundle['input_names'],
                          verbose=verbose, name=name)

    if verbose:
        print_report(meta, result['trees'], bundle['input_names'])

    return {'trees': result['trees'], 'classifier': clf, 'meta': meta,
            'result': result, 'bundle': bundle, 'saved': saved}


def print_report(meta: dict, trees: list, input_names: list[str]):
    """Human-readable run summary."""
    m = meta['metrics']
    print("\n" + "=" * 72)
    print(f"BRANCH {meta['branch']}: oracle -> {len(trees)} EML formulas")
    print("=" * 72)
    print(f"{'split':<8} {'accuracy':>18} {'bal.accuracy':>20} {'fidelity':>10}")
    print(f"{'':8} {'EML':>8} {'oracle':>9} {'EML':>9} {'oracle':>10} {'':>10}")
    for split in ('train', 'val', 'test'):
        s = m[split]
        print(f"{split:<8} {s['eml_acc']:>8.4f} {s['oracle_acc']:>9.4f} "
              f"{s['eml_bal_acc']:>9.4f} {s['oracle_bal_acc']:>10.4f} "
              f"{s['fidelity']:>10.4f}")

    test = m['test']
    retained = (test['eml_bal_acc'] / test['oracle_bal_acc'] * 100
                if test['oracle_bal_acc'] > 0 else 0.0)
    print(f"\nthe formulas hold {retained:.1f}% of the oracle's balanced "
          f"accuracy on test")
    print(f"complexity: {meta['total_nodes']} AST nodes, depth "
          f"{meta['max_depth']}, inputs used "
          f"{meta['n_inputs_used']}/{meta['n_inputs_available']}")
    print(f"used: {', '.join(meta['inputs_used'])}")

    print("\nEML confusion matrix on test (rows = ground truth):")
    header = ' ' * 9 + ''.join(f"{c[:6]:>8}" for c in config.CLASS_NAMES)
    print(header)
    for i, row in enumerate(test['confusion']):
        print(f"{config.CLASS_NAMES[i]:>8} " + ''.join(f"{v:>8}" for v in row))

    print("\nformulas:")
    for c, tree in enumerate(trees):
        print(f"  f_{config.CLASS_NAMES[c]} = {tree.to_string()}")
    print("=" * 72)


# ── Batch distillation ────────────────────────────────────────────────────────

def run_batch(branch: str = 'A', n_variants: int | None = None,
              verbose: bool = True, progress_cb=None, should_stop=None,
              generations: int | None = None,
              population: int | None = None) -> dict:
    """
    Several runs with different seeds and depth penalties.

    A GA is stochastic: one run says little about what the method achieves. The
    best by val (NOT by test — test is touched once, at the very end) is saved
    as the main one, and a summary of all of them goes into runs/.
    """
    n_variants = n_variants if n_variants is not None else config.EML_BATCH_VARIANTS
    penalties = list(config.EMLC_SIZE_PENALTIES if branch.upper() == 'C'
                     else config.DEPTH_PENALTIES)
    runs, best, best_val = [], None, -1.0
    stamp = time.strftime('%Y%m%d-%H%M')
    summary_dir = os.path.join(config.RUNS_DIR,
                               time.strftime('%Y%m%d_%H%M%S') + f'_distill_{branch}')
    os.makedirs(summary_dir, exist_ok=True)

    for i in range(n_variants):
        penalty = penalties[i % len(penalties)]
        if verbose:
            print(f"\n{'#' * 72}\n# variant {i + 1}/{n_variants}: "
                  f"seed={i}, depth_penalty={penalty}\n{'#' * 72}")
        cb = None
        if progress_cb is not None:
            def cb(payload, _i=i):
                progress_cb({**payload, 'variant': _i, 'variants': n_variants})
        out = run(branch, depth_penalty=penalty, seed=i, save=False,
                  generations=generations, population=population,
                  verbose=verbose, progress_cb=cb, should_stop=should_stop)
        val_score = out['meta']['metrics']['val']['eml_bal_acc']

        # Every variant goes into the library under a descriptive name. Only
        # what was saved can be selected later, and batch variants differ from
        # each other more than many method changes do.
        entry, _ = archive_run(
            out['trees'], branch, out['meta'], out['bundle']['input_names'],
            run_dir=summary_dir, filename=f'variant_{i}_{penalty}.json',
            name=f"{stamp}_seed{i}-{penalty}_bal{val_score:.3f}")
        out['entry'] = entry

        runs.append({'seed': i, 'depth_penalty': penalty,
                     'val_bal_acc': val_score,
                     'test_bal_acc': out['meta']['metrics']['test']['eml_bal_acc'],
                     'nodes': out['meta']['total_nodes'],
                     'stopped': out['meta'].get('stopped', False),
                     'library_name': entry.name})
        if val_score > best_val:
            best_val, best = val_score, out
        if should_stop is not None and should_stop():
            if verbose:
                print(f"\n[batch] stopped after variant {i + 1}")
            break

    # The same protected path as for a single run: a batch must not overwrite
    # the best result either if it was unlucky with its seeds.
    saved = promote_if_better(best['entry'], verbose=False)
    with open(os.path.join(summary_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'branch': branch, 'runs': runs, 'best': best['meta'],
                   'saved': {k: v for k, v in saved.items() if k != 'entry'}},
                  f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'=' * 72}\nVARIANT SUMMARY (best chosen by val)")
        print(f"{'#':>3} {'penalty':>9} {'val bal':>9} {'test bal':>9} "
              f"{'nodes':>7}  library name")
        for r in runs:
            mark = ' <-' if r['val_bal_acc'] == best_val else '   '
            print(f"{r['seed']:>3} {r['depth_penalty']:>9} "
                  f"{r['val_bal_acc']:>9.4f} {r['test_bal_acc']:>9.4f} "
                  f"{r['nodes']:>7}{mark} {r['library_name']}")
        if saved['promoted']:
            print(f"\"{saved['library_name']}\" became active "
                  f"-> {saved['best_path']}")
        else:
            print(f"active NOT touched ({saved['reason']}); all variants "
                  f"are in the library")
        print(f"summary -> {summary_dir}/summary.json")

    return {'best': best, 'runs': runs, 'summary_dir': summary_dir,
            'saved': saved}
