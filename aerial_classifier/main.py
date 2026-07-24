"""
main.py — точка входа проекта.

    python main.py info                  сведения о датасете и признаках
    python main.py train --branch A      обучить оракула (A: MLP, B: CNN)
    python main.py distill --branch A    дистилляция оракула в EML-формулы
    python main.py distill --batch       пачка прогонов, лучший по val
    python main.py report --branch A     графики в results/
    python main.py export --branch A     экспорт в C (results/c_export/)
    python main.py verify-c              скомпилировать C и сверить с Python
    python main.py bench                 сравнение скорости оракул vs формулы
    python main.py calibrate             перцентили признаков (для _SCALE_*)
    python main.py gui                   графический интерфейс (всё то же в окне)

Порядок для чистого запуска: train -> distill -> export -> verify-c.
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
    print(data.describe(labels, 'всего'))
    print(data.describe(labels[train_idx], 'train'))
    print(data.describe(labels[val_idx], 'val  '))
    print(data.describe(labels[test_idx], 'test '))
    print(f"\nкартинка: {config.IMG_SIZE}x{config.IMG_SIZE}x{config.IMG_CHANNELS}")
    print(f"признаков: {config.N_FEATURES}")

    feats = features_mod.load_or_compute(images)
    print("\nF-статистика признаков на train (насколько признак разделяет классы):")
    scores = features_mod.anova_f(feats[train_idx], labels[train_idx])
    for i in np.argsort(-scores):
        print(f"  {config.FEATURE_NAMES[i]:>18}  F = {scores[i]:8.1f}")


# ── train ─────────────────────────────────────────────────────────────────────

def cmd_train(args):
    import reports
    import train_oracle

    out = train_oracle.train(args.branch, epochs=args.epochs)
    path = os.path.join(config.RESULTS_DIR, f'training_{args.branch.upper()}.png')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    reports.plot_training(out['history'],
                          f"Оракул, ветка {args.branch.upper()} "
                          f"({out['meta']['model']})", path)
    print(f"[report] {path}")


# ── distill ───────────────────────────────────────────────────────────────────

def cmd_distill(args):
    import pipeline

    if args.batch:
        pipeline.run_batch(args.branch, n_variants=args.variants)
    else:
        pipeline.run(args.branch, depth_penalty=args.depth_penalty,
                     generations=args.generations, population=args.population,
                     acc_phase=not args.no_acc_phase, seed=args.seed)


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
            f"EML, ветка {branch}, {split} "
            f"(bal.acc {metrics[split]['eml_bal_acc']:.3f})", path)
        print(f"[report] {path}")

    print("\nформулы:")
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
#include "aerial_eml_A.h"

/* Тестовая обвязка: читает сырые кадры, печатает признаки и класс. */
int main(int argc, char **argv)
{
    FILE *fp;
    int n, k, i;
    uint8_t *buf;
    float f[AERIAL_N_FEATURES];

    if (argc < 3) { fprintf(stderr, "usage: harness <file> <n>\n"); return 1; }
    fp = fopen(argv[1], "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    n = atoi(argv[2]);
    buf = (uint8_t *)malloc(AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3);

    for (k = 0; k < n; ++k) {
        if (fread(buf, 1, AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3, fp)
            != (size_t)(AERIAL_IMG_SIZE * AERIAL_IMG_SIZE * 3)) break;
        aerial_extract_features(buf, f);
        for (i = 0; i < AERIAL_N_FEATURES; ++i) printf("%.6f ", f[i]);
        printf("%d\n", aerial_classify(f));
    }
    free(buf);
    fclose(fp);
    return 0;
}
'''


def cmd_verify_c(args):
    """
    Сгенерировать C, скомпилировать и сверить с Python на реальных кадрах.

    Проверяются обе половины экспорта: признаки (числами) и итоговый класс
    (совпадением решения). Расхождение в признаках больше допуска — это
    рассинхрон features.py и шаблона в export.py, самая вероятная поломка
    при правках.
    """
    import data
    import eml_distiller as dist
    import export
    import features as features_mod
    import storage

    n = args.n
    branch = 'A'                     # C-экспорт признаков есть только у ветки A
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
            f.write(_HARNESS)

        exe = os.path.join(tmp, 'harness.exe' if os.name == 'nt' else 'harness')
        compile_cmd = ['gcc', '-O2', '-I', out_dir, harness_path, '-o', exe, '-lm']
        proc = subprocess.run(compile_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print("КОМПИЛЯЦИЯ НЕ УДАЛАСЬ:")
            print(proc.stderr)
            return 1

        proc = subprocess.run([exe, raw_path, str(len(batch))],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("ЗАПУСК НЕ УДАЛСЯ:")
            print(proc.stderr)
            return 1

    rows = [line.split() for line in proc.stdout.strip().splitlines()]
    c_feats = np.array([[float(v) for v in r[:-1]] for r in rows])
    c_pred = np.array([int(r[-1]) for r in rows])

    py_feats = features_mod.extract_batch(batch)
    py_scores = clf.scores(py_feats.astype(np.float64))
    py_pred = py_scores.argmax(axis=1)

    diff = np.abs(c_feats - py_feats)
    max_diff = float(diff.max())
    agree = float((c_pred == py_pred).mean())

    # Расхождение класса — не обязательно ошибка. C считает во float, Python во
    # float64; на семпле, где два лучших класса идут вплотную, argmax может
    # разойтись совершенно законно, и требовать agree == 1.0 значит ловить
    # случайные красные прогоны там, где экспорт исправен.
    #
    # Поэтому разошедшиеся семплы разбираются: если у Python зазор между двумя
    # лучшими оценками меньше BORDERLINE_GAP, семпл считается пограничным и
    # засчитывается; если зазор большой, значит C реально решает иначе — это
    # поломка. Допуск по признакам при этом остаётся жёстким (1e-4).
    BORDERLINE_GAP = 1e-3
    mismatch = np.flatnonzero(c_pred != py_pred)
    top2 = np.sort(py_scores, axis=1)[:, -2:]
    gaps = top2[:, 1] - top2[:, 0]
    borderline = mismatch[gaps[mismatch] < BORDERLINE_GAP]
    real_mismatch = mismatch[gaps[mismatch] >= BORDERLINE_GAP]

    print(f"\nСВЕРКА C ПРОТИВ PYTHON ({len(batch)} кадров)")
    print(f"  макс. расхождение признака : {max_diff:.2e}")
    print(f"  среднее расхождение        : {float(diff.mean()):.2e}")
    print(f"  совпадение класса          : {agree:.4f} "
          f"({int((c_pred == py_pred).sum())}/{len(batch)})")
    if len(mismatch):
        print(f"  из расхождений пограничных : {len(borderline)} "
              f"(зазор top-2 < {BORDERLINE_GAP:g})")
        print(f"  настоящих расхождений      : {len(real_mismatch)}")
        for i in real_mismatch[:5]:
            print(f"    кадр {i}: C={config.CLASS_NAMES[c_pred[i]]} "
                  f"Python={config.CLASS_NAMES[py_pred[i]]} "
                  f"зазор {gaps[i]:.3e}")

    worst = np.argsort(-diff.max(axis=0))[:5]
    print("  худшие признаки:")
    for i in worst:
        print(f"    {config.FEATURE_NAMES[i]:>18} {diff[:, i].max():.2e}")

    # Допуск 1e-4: C считает во float, Python — во float32/float64, полного
    # побитового совпадения не будет и не требуется.
    ok = max_diff < 1e-4 and len(real_mismatch) == 0
    print(f"\n  {'ПРОЙДЕНО' if ok else 'НЕ ПРОЙДЕНО'}")
    return 0 if ok else 1


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
    Проверка констант _SCALE_* в features.py — чтобы они не были взяты с потолка.

    Ошибиться масштабом можно в обе стороны, и обе видны по распределению:
        масштаб ЗАНИЖЕН — много значений упёрлось в границу, разные картинки
                          получают одинаковый признак;
        масштаб ЗАВЫШЕН — распределение занимает узкую часть [-1, 1], признак
                          зря отдаёт разрешение (а перед exp() в EML-операторе
                          мелкие различия и так плохо различимы).

    Проверяются только эмпирически отмасштабированные признаки. У долей и
    корреляций границы диапазона заданы математикой, а не подбором: если у
    трети кадров ровно ноль тёмных пикселей, то это свойство датасета, и
    «чинить» тут нечего.

    Считается ТОЛЬКО по train-сплиту. Влияние на сами константы тут
    пренебрежимо (перцентили 3.5 тысяч кадров и 5 тысяч почти совпадают), но
    масштабы нормировки — это часть модели, и подбирать их по val/test значило
    бы подглядывать в отложенную выборку. Дешевле сделать правильно, чем
    объяснять рецензенту, почему это не считается утечкой.
    """
    import data
    import features as features_mod

    images, labels, _ = data.load_dataset(verbose=False)
    train_idx, _, _ = data.stratified_split(labels)
    feats = features_mod.extract_batch(images[train_idx])
    print(f"\n(перцентили считаются по train: {len(train_idx)} из {len(images)} "
          f"картинок)")

    print("\nМАСШТАБЫ ПРИЗНАКОВ")
    print("упор = доля значений на границе [-1, 1]; охват = какую часть "
          "диапазона занимает распределение\n")
    issues = []
    for i, name in enumerate(config.FEATURE_NAMES):
        col = feats[:, i]
        saturated = float(((col <= -0.999) | (col >= 0.999)).mean())
        span = float(col.max() - col.min()) / 2.0
        empirical = name in features_mod.EMPIRICALLY_SCALED

        note = ''
        if empirical and saturated > 0.05:
            note = '  <-- масштаб занижен'
            issues.append((name, note))
        elif empirical and span < 0.5:
            note = f'  <-- масштаб завышен ~x{1 / max(span, 1e-6):.1f}'
            issues.append((name, note))

        kind = ' ' if empirical else '.'
        print(f" {kind}{name:>18}  упор {saturated * 100:5.1f}%  "
              f"охват {span * 100:5.1f}%  "
              f"[{col.min():+.3f}, {col.max():+.3f}]{note}")

    print("\n('.' = масштаб задан по построению, не проверяется)")
    if issues:
        print(f"\nтребуют внимания: {len(issues)}")
        print("правка масштаба меняет признаки — после неё нужно заново "
              "обучить оракула и повторить дистилляцию.")
    else:
        print("\nвсе эмпирические масштабы в норме.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python main.py',
        description='Дистилляция классификатора аэроснимков в EML-формулы.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('info', help='сведения о датасете и признаках')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('train', help='обучить оракула')
    p.add_argument('--branch', default='A', choices=['A', 'B', 'a', 'b'],
                   help='A: MLP на признаках, B: CNN на пикселях')
    p.add_argument('--epochs', type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('distill', help='дистилляция оракула в EML-формулы')
    p.add_argument('--branch', default='A', choices=['A', 'B', 'a', 'b'])
    p.add_argument('--batch', action='store_true',
                   help='несколько прогонов, лучший по val')
    p.add_argument('--variants', type=int, default=None)
    p.add_argument('--generations', type=int, default=None)
    p.add_argument('--population', type=int, default=None)
    p.add_argument('--depth-penalty', default=None,
                   choices=list(config.DEPTH_PENALTIES.keys()))
    p.add_argument('--no-acc-phase', action='store_true',
                   help='только DATA-фаза (для отладки)')
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=cmd_distill)

    p = sub.add_parser('report', help='графики в results/')
    p.add_argument('--branch', default='A', choices=['A', 'B', 'a', 'b'])
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('export', help='экспорт формул в C')
    p.add_argument('--branch', default='A', choices=['A', 'B', 'a', 'b'])
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser('verify-c', help='скомпилировать C и сверить с Python')
    p.add_argument('-n', type=int, default=200, help='сколько кадров сверять')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_verify_c)

    p = sub.add_parser('bench', help='сравнение скорости оракул vs формулы')
    p.add_argument('-n', type=int, default=500)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser('calibrate', help='проверка масштабов нормализации')
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser('gui', help='графический интерфейс')
    p.add_argument('--tab', default=None,
                   choices=['data', 'features', 'train', 'distill', 'test'],
                   help='вкладка, открытая при старте')
    p.set_defaults(func=cmd_gui)

    return parser


def main():
    args = build_parser().parse_args()
    if hasattr(args, 'branch'):
        args.branch = args.branch.upper()
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
