"""
pipeline.py — полный прогон дистилляции и оценка результата.

Связывает вместе три фазы проекта: берёт обученного оракула, собирает датасет
(входы -> логиты оракула), эволюционирует EML-формулы и меряет, сколько
качества потеряно при замене нейросети на формулу.

Ключевая метрика тут не одна, а три, и путать их нельзя:

    accuracy    — доля правильных ответов относительно ИСТИННЫХ меток.
                  Отвечает на вопрос «работает ли формула».
    fidelity    — доля совпадений с ответом ОРАКУЛА.
                  Отвечает на вопрос «повторяет ли формула именно оракула».
    bal_accuracy— accuracy, усреднённая по классам.
                  Главная метрика: датасет несбалансирован 3.4:1, и обычная
                  accuracy переоценивает модель, которая просто любит ROADS.

Формула может иметь высокую accuracy и низкую fidelity — тогда она решает
задачу по-своему, а не копирует оракула. Для диплома это самостоятельный
результат, а не ошибка, поэтому обе величины считаются и печатаются.
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


# ── Сбор датасета для дистилляции ─────────────────────────────────────────────

def collect_dataset(branch: str = 'A', verbose: bool = True) -> dict:
    """
    Собрать всё, что нужно дистиллятору: входы формул, логиты оракула, метки.

    Ветка A: входы = 32 инженерных признака.
    Ветка B: входы = bottleneck-эмбеддинг обученной CNN.

    Возвращает словарь с ключами inputs / logits / labels / splits / meta.
    """
    branch = branch.upper()
    images, labels, _ = data.load_dataset(verbose=verbose)
    train_idx, val_idx, test_idx = data.stratified_split(labels)
    device = oracle_mod.get_device()

    if branch == 'A':
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
        print(f"[collect] ветка {branch}: входы {inputs.shape}, "
              f"логиты {logits.shape}")
        print(f"[collect] оракул: val {meta.get('val_acc', float('nan')):.4f}, "
              f"test {meta.get('test_acc', float('nan')):.4f}")

    return {
        'inputs': inputs, 'logits': logits, 'labels': labels,
        'splits': (train_idx, val_idx, test_idx),
        'input_names': input_names, 'oracle_meta': meta, 'branch': branch,
    }


# ── Оценка ────────────────────────────────────────────────────────────────────

def evaluate(clf: dist.EMLClassifier, inputs: np.ndarray, logits: np.ndarray,
             labels: np.ndarray, idx: np.ndarray) -> dict:
    """Метрики формул и оракула на одном сплите."""
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
    """Матрица ошибок (n_classes, n_classes), строки = истина."""
    n_classes = n_classes or config.N_CLASSES
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


# ── Полный прогон ─────────────────────────────────────────────────────────────

def archive_run(trees: list, branch: str, meta: dict, input_names: list[str], *,
                run_dir: str | None = None, name: str | None = None,
                filename: str | None = None):
    """
    Записать прогон в runs/ и положить именованную копию в библиотеку.

    Ничего не решает про «лучшесть» — только сохраняет. Разделение с
    promote_if_better нужно батчу: там в библиотеку идут ВСЕ варианты, а в
    активный слот претендует один.

    Returns: (ModelEntry, путь к копии в runs/).
    """
    branch = branch.upper()
    run_dir = run_dir or os.path.join(
        config.RUNS_DIR, time.strftime('%Y%m%d_%H%M%S') + f'_run_{branch}')
    os.makedirs(run_dir, exist_ok=True)

    meta = dict(meta)
    meta['library_name'] = storage.reserve_name('eml', branch, name, meta)
    run_path = storage.save_formulas(
        trees, branch=branch, meta=meta, input_names=input_names,
        path=os.path.join(run_dir, filename or f'formulas_{branch}.json'))
    entry = storage.add_to_library('eml', branch, meta['library_name'], meta,
                                   source_path=run_path, overwrite=True)
    return entry, run_path


def promote_if_better(entry, *, verbose: bool = True) -> dict:
    """
    Сделать модель активной, только если она не хуже уже лежащей ПО VAL.

    Раньше любой прогон с save=True (а это умолчание и в GUI, и в CLI) молча
    перезаписывал models/best_eml_X.json. Один неудачный сид — и отобранный
    батчем набор формул терялся без следа, причём заметить это было не по чему.

    Правила:
      * сравнение только по val — test смотрят один раз, в самом конце;
      * прежний файл на всякий случай остаётся рядом как .prev.json;
      * прерванный пользователем прогон не претендует на активный никогда.
    """
    branch = entry.branch
    new_val = entry.val_score if entry.val_score is not None else -1.0
    prev_val = storage.formulas_meta(branch).get(
        'metrics', {}).get('val', {}).get('eml_bal_acc')
    best_path = storage.formulas_path(branch)

    if entry.meta.get('stopped'):
        promoted, reason = False, "прогон прерван пользователем"
    elif prev_val is None:
        promoted, reason = True, "сохранённых формул не было"
    elif new_val >= float(prev_val):
        promoted = True
        reason = f"val {new_val:.4f} >= прежних {float(prev_val):.4f}"
    else:
        promoted = False
        reason = f"val {new_val:.4f} < прежних {float(prev_val):.4f}"

    if promoted:
        if os.path.exists(best_path):
            shutil.copyfile(best_path, best_path.replace('.json', '.prev.json'))
        storage.activate(entry)

    if verbose:
        if promoted:
            print(f"[distill] активной стала «{entry.name}» ({reason})"
                  f" -> {best_path}")
        else:
            print(f"[distill] активная модель НЕ тронута ({reason})")
    return {'promoted': promoted, 'reason': reason,
            'best_path': best_path if promoted else None,
            'entry': entry, 'library_name': entry.name}


def save_best(trees: list, branch: str, meta: dict, input_names: list[str], *,
              verbose: bool = True, run_dir: str | None = None,
              name: str | None = None) -> dict:
    """
    Сохранить прогон целиком: архив в runs/, копия в библиотеку, и активный
    слот, если прогон не хуже прежнего.

    Returns: {'run_path', 'best_path' | None, 'promoted', 'reason', 'entry'}.
    """
    entry, run_path = archive_run(trees, branch, meta, input_names,
                                  run_dir=run_dir, name=name)
    if verbose:
        print(f"\n[distill] прогон -> {run_path}")
        print(f"[distill] в библиотеку как «{entry.name}» -> {entry.path}")
    result = promote_if_better(entry, verbose=verbose)
    result['run_path'] = run_path
    return result


def run(branch: str = 'A', *, depth_penalty: str | None = None,
        generations: int | None = None, population: int | None = None,
        acc_phase: bool | None = None, seed: int = 0,
        save: bool = True, verbose: bool = True,
        progress_cb=None, should_stop=None, name: str | None = None) -> dict:
    """
    Собрать датасет -> дистиллировать -> оценить -> сохранить формулы.

    progress_cb / should_stop пробрасываются в эволюцию: первый даёт живые
    кривые, второй — мягкую остановку с сохранением лучшего на этот момент.

    Returns: dict с формулами, метриками по всем сплитам и метаданными.
    """
    branch = branch.upper()
    t0 = time.time()
    bundle = collect_dataset(branch, verbose=verbose)
    inputs, logits, labels = bundle['inputs'], bundle['logits'], bundle['labels']
    train_idx, val_idx, test_idx = bundle['splits']

    # Формулы работают в пространстве входов конкретной ветки — сообщаем его
    # дистиллятору до эволюции, иначе var_idx будет индексировать не то.
    dist.configure_inputs(inputs.shape[1], bundle['input_names'])

    if verbose:
        print(f"\n[distill] ветка {branch}, {inputs.shape[1]} входов, "
              f"{len(train_idx)} обучающих примеров")

    result = dist.distill(
        inputs[train_idx], logits[train_idx], labels[train_idx],
        depth_penalty_name=depth_penalty, data_generations=generations,
        data_population=population, acc_phase=acc_phase,
        verbose=verbose, rng_seed=seed,
        progress_cb=progress_cb, should_stop=should_stop,
    )

    # Калибровка уже заморожена дистиллятором по train — берём классификатор
    # как есть, не пересобирая: пересчёт по val/test был бы подглядыванием.
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
    """Человекочитаемая сводка прогона."""
    m = meta['metrics']
    print("\n" + "=" * 72)
    print(f"ВЕТКА {meta['branch']}: оракул -> {len(trees)} EML-формул")
    print("=" * 72)
    print(f"{'сплит':<8} {'accuracy':>18} {'bal.accuracy':>20} {'fidelity':>10}")
    print(f"{'':8} {'EML':>8} {'оракул':>9} {'EML':>9} {'оракул':>10} {'':>10}")
    for split in ('train', 'val', 'test'):
        s = m[split]
        print(f"{split:<8} {s['eml_acc']:>8.4f} {s['oracle_acc']:>9.4f} "
              f"{s['eml_bal_acc']:>9.4f} {s['oracle_bal_acc']:>10.4f} "
              f"{s['fidelity']:>10.4f}")

    test = m['test']
    retained = (test['eml_bal_acc'] / test['oracle_bal_acc'] * 100
                if test['oracle_bal_acc'] > 0 else 0.0)
    print(f"\nформулы держат {retained:.1f}% сбалансированной точности оракула "
          f"на test")
    print(f"сложность: {meta['total_nodes']} узлов AST, глубина "
          f"{meta['max_depth']}, задействовано входов "
          f"{meta['n_inputs_used']}/{meta['n_inputs_available']}")
    print(f"использованы: {', '.join(meta['inputs_used'])}")

    print("\nматрица ошибок EML на test (строки = истина):")
    header = ' ' * 9 + ''.join(f"{c[:6]:>8}" for c in config.CLASS_NAMES)
    print(header)
    for i, row in enumerate(test['confusion']):
        print(f"{config.CLASS_NAMES[i]:>8} " + ''.join(f"{v:>8}" for v in row))

    print("\nформулы:")
    for c, tree in enumerate(trees):
        print(f"  f_{config.CLASS_NAMES[c]} = {tree.to_string()}")
    print("=" * 72)


# ── Батч-дистилляция ──────────────────────────────────────────────────────────

def run_batch(branch: str = 'A', n_variants: int | None = None,
              verbose: bool = True, progress_cb=None, should_stop=None) -> dict:
    """
    Несколько прогонов с разными сидами и штрафами за глубину.

    GA стохастичен: один прогон говорит мало о том, чего метод достигает.
    Лучший по val (НЕ по test — test трогаем один раз, в самом конце)
    сохраняется как основной, сводка по всем — в runs/.
    """
    n_variants = n_variants if n_variants is not None else config.EML_BATCH_VARIANTS
    penalties = list(config.DEPTH_PENALTIES.keys())
    runs, best, best_val = [], None, -1.0
    stamp = time.strftime('%Y%m%d-%H%M')
    summary_dir = os.path.join(config.RUNS_DIR,
                               time.strftime('%Y%m%d_%H%M%S') + f'_distill_{branch}')
    os.makedirs(summary_dir, exist_ok=True)

    for i in range(n_variants):
        penalty = penalties[i % len(penalties)]
        if verbose:
            print(f"\n{'#' * 72}\n# вариант {i + 1}/{n_variants}: "
                  f"seed={i}, depth_penalty={penalty}\n{'#' * 72}")
        cb = None
        if progress_cb is not None:
            def cb(payload, _i=i):
                progress_cb({**payload, 'variant': _i, 'variants': n_variants})
        out = run(branch, depth_penalty=penalty, seed=i, save=False,
                  verbose=verbose, progress_cb=cb, should_stop=should_stop)
        val_score = out['meta']['metrics']['val']['eml_bal_acc']

        # Каждый вариант — в библиотеку под говорящим именем. Выбрать потом
        # можно только из сохранённого, а варианты батча отличаются друг от
        # друга сильнее, чем иные правки метода.
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
                print(f"\n[batch] остановлено после варианта {i + 1}")
            break

    # Тот же защищённый путь, что и у одиночного прогона: батч тоже не должен
    # затирать лучший результат, если ему не повезло с сидами.
    saved = promote_if_better(best['entry'], verbose=False)
    with open(os.path.join(summary_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'branch': branch, 'runs': runs, 'best': best['meta'],
                   'saved': {k: v for k, v in saved.items() if k != 'entry'}},
                  f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'=' * 72}\nСВОДКА ПО ВАРИАНТАМ (лучший выбран по val)")
        print(f"{'#':>3} {'penalty':>9} {'val bal':>9} {'test bal':>9} "
              f"{'узлов':>7}  имя в библиотеке")
        for r in runs:
            mark = ' <-' if r['val_bal_acc'] == best_val else '   '
            print(f"{r['seed']:>3} {r['depth_penalty']:>9} "
                  f"{r['val_bal_acc']:>9.4f} {r['test_bal_acc']:>9.4f} "
                  f"{r['nodes']:>7}{mark} {r['library_name']}")
        if saved['promoted']:
            print(f"активной стала «{saved['library_name']}» "
                  f"-> {saved['best_path']}")
        else:
            print(f"активная НЕ тронута ({saved['reason']}); все варианты "
                  f"лежат в библиотеке")
        print(f"сводка -> {summary_dir}/summary.json")

    return {'best': best, 'runs': runs, 'summary_dir': summary_dir,
            'saved': saved}
