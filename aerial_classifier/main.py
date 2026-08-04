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

    out = train_oracle.train(args.branch, epochs=args.epochs, lr=args.lr,
                             batch_size=args.batch_size, name=args.name)
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
        pipeline.run_batch(args.branch, n_variants=args.variants,
                           generations=args.generations,
                           population=args.population)
    elif args.branch == 'C':
        # У ветки C свои ручки: острова, процессы и продолжение прогона.
        # Смешивать их с generations/population веток A и B в одном наборе
        # флагов можно, а вот делать вид, что --no-acc-phase на неё влияет, —
        # нельзя: ACC-фазы у неё нет.
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
            f"EML, ветка {branch}, {split} "
            f"(bal.acc {metrics[split]['eml_bal_acc']:.3f})", path)
        print(f"[report] {path}")

    if hasattr(clf, 'decoder'):                  # ветка C: одна формула
        print(f"\nf = {clf.tree.to_string()}")
        print("\nкуда формула расставила классы на своей оси:")
        print(clf.decoder.describe(config.CLASS_NAMES))
    else:
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
#include "aerial_eml_{branch}.h"

/* Тестовая обвязка: читает сырые кадры, печатает признаки и класс. */
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
        /* Ветка C: печатаем ещё и сам выход формулы. По нему видно, насколько
         * float32 разошёлся с float64, — а значит, законно ли расхождение
         * класса на кадре, легшем вплотную к порогу. */
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
    branch = (args.branch or 'A').upper()   # экстрактор признаков есть у A и C
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
            print("КОМПИЛЯЦИЯ НЕ УДАЛАСЬ:")
            print(proc.stderr)
            return 1

        proc = subprocess.run([exe, raw_path, str(len(batch))],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("ЗАПУСК НЕ УДАЛСЯ:")
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

    # Расхождение класса — не обязательно ошибка. C считает во float, Python во
    # float64; на кадре, где решение принимается «впритык», ответы могут
    # разойтись совершенно законно, и требовать agree == 1.0 значит ловить
    # случайные красные прогоны там, где экспорт исправен. Поэтому расхождения
    # разбираются, а не просто считаются.
    if c_value is not None:
        # Ветка C: решает одно число против таблицы порогов. Расхождение
        # законно ровно тогда, когда сама разница float32/float64 в этом числе
        # не меньше расстояния до ближайшего порога — то есть кадр лежит в
        # зоне численной неопределённости, а не классифицируется иначе.
        py_value = clf.raw(py_feats.astype(np.float64))
        py_pred = clf.decoder.decode(py_value)
        thresholds = clf.decoder.thresholds
        gaps = (np.abs(py_value[:, None] - thresholds[None, :]).min(axis=1)
                if len(thresholds) else np.full(len(py_value), np.inf))
        noise = np.abs(py_value - c_value)
        borderline_mask = gaps <= noise
        gap_label = "расстояние до ближайшего порога"
    else:
        py_scores = clf.scores(py_feats.astype(np.float64))
        py_pred = py_scores.argmax(axis=1)
        top2 = np.sort(py_scores, axis=1)[:, -2:]
        gaps = top2[:, 1] - top2[:, 0]
        borderline_mask = gaps < 1e-3
        gap_label = "зазор между двумя лучшими классами"

    agree = float((c_pred == py_pred).mean())
    mismatch = np.flatnonzero(c_pred != py_pred)
    borderline = mismatch[borderline_mask[mismatch]]
    real_mismatch = mismatch[~borderline_mask[mismatch]]

    print(f"\nСВЕРКА C ПРОТИВ PYTHON, ветка {branch} ({len(batch)} кадров)")
    if c_value is not None:
        rel = np.abs(py_value - c_value) / np.maximum(np.abs(py_value), 1.0)
        print(f"  макс. расхождение выхода   : {float(rel.max()):.2e} "
              f"(относительное)")
    print(f"  макс. расхождение признака : {max_diff:.2e}")
    print(f"  среднее расхождение        : {float(diff.mean()):.2e}")
    print(f"  совпадение класса          : {agree:.4f} "
          f"({int((c_pred == py_pred).sum())}/{len(batch)})")
    if len(mismatch):
        print(f"  из расхождений пограничных : {len(borderline)} "
              f"({gap_label} в пределах численного шума)")
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


# ── models ────────────────────────────────────────────────────────────────────

def cmd_models(args):
    """
    Библиотека моделей: список, активация, переименование, удаление.

    Лучшая в каждой группе (вид × ветка) помечается звёздочкой и выбирается
    ПО VAL — той же метрикой, по которой идёт отбор внутри прогона. Test для
    выбора не используется: его смотрят один раз, в самом конце.
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
        print("\nбиблиотека пуста. Модель попадает в неё сама при "
              "`train` и `distill`.")
        print(f"папка: {storage.library_dir()}")
        return 0

    print(f"\nБИБЛИОТЕКА МОДЕЛЕЙ ({storage.library_dir()})")
    print(f"датасет сейчас: {config.dataset_summary()}")
    print("★ лучшая по val среди подходящих   • активная сейчас   "
          "≠ другой набор классов\n")
    header = (f"    {'вид':<7} {'ветка':<5} {'имя':<34} {'val bal':>8} "
              f"{'test bal':>9} {'сложность':>10}")
    print(header)
    print('-' * len(header))
    for entry in entries:
        marks = (('★' if entry.is_best else ' ')
                 + ('•' if entry.is_active else ' ')
                 + (' ' if entry.fits_dataset else '≠'))
        val = f"{entry.val_score:.4f}" if entry.val_score is not None else '—'
        test = f"{entry.test_score:.4f}" if entry.test_score is not None else '—'
        unit = 'пар.' if entry.kind == 'oracle' else 'узл.'
        size = f"{entry.size} {unit}" if entry.size is not None else '—'
        kind = 'оракул' if entry.kind == 'oracle' else 'формулы'
        print(f"{marks} {kind:<7} {entry.branch:<5} {entry.name:<34} "
              f"{val:>8} {test:>9} {size:>10}")

    alien = [e for e in entries if not e.fits_dataset]
    if alien:
        print(f"\n≠ {len(alien)} моделей обучены на другом наборе классов и к "
              f"текущему датасету не подойдут.")
        print("  Их метрики несравнимы с остальными и в выборе лучшей не "
              "участвуют.")

    print("\nактивные сейчас:")
    for kind in storage.KINDS:
        for br in ('A', 'B', 'C'):
            if kind == 'oracle' and br == 'C':
                continue                     # ветка C учится у оракула ветки A
            name = storage.active_name(kind, br)
            label = 'оракул ' if kind == 'oracle' else 'формулы'
            shown = name or '— (слот пуст или модель не из библиотеки)'
            print(f"  {label} {br}: {shown}")
    print("\nсделать активной:  python main.py models --activate <имя>")
    return 0


def _models_apply(storage, name: str, branch: str | None, action: str,
                  new_name: str | None = None) -> int:
    """Найти модель по имени и выполнить над ней действие."""
    matches = [e for e in storage.list_models(branch=branch) if e.name == name]
    if not matches:
        print(f"модель «{name}» не найдена. Список: python main.py models")
        return 1
    if len(matches) > 1:
        # Имена уникальны внутри группы (вид × ветка), но не между группами.
        print(f"под именем «{name}» найдено несколько моделей — уточните "
              f"--branch и --kind:")
        for e in matches:
            print(f"  {e.kind} ветка {e.branch}")
        return 1

    entry = matches[0]
    if action == 'activate':
        path = storage.activate(entry)
        print(f"активной стала «{entry.name}» -> {path}")
    elif action == 'delete':
        if entry.is_active:
            print(f"«{entry.name}» сейчас активна. Активный слот "
                  f"({storage.active_path(entry.kind, entry.branch)}) "
                  f"останется на месте, из библиотеки модель будет удалена.")
        storage.delete_model(entry)
        print(f"удалено: {entry.path}")
    else:
        renamed = storage.rename_model(entry, new_name)
        print(f"«{entry.name}» -> «{renamed.name}»")
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

BRANCHES = ['A', 'B', 'C', 'a', 'b', 'c']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python main.py',
        description='Дистилляция классификатора изображений в EML-формулы.')
    # Датасет — общий флаг всех команд: он определяет и число классов, и всё,
    # что от него зависит, поэтому задавать его внутри отдельных подкоманд
    # означало бы, что часть из них молча работает с другим набором классов.
    parser.add_argument('--dataset', default=None, metavar='ПАПКА',
                        help='папка с подпапками-классами; выбор запоминается')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('info', help='сведения о датасете и признаках')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('train', help='обучить оракула')
    p.add_argument('--branch', default='A', choices=['A', 'B', 'a', 'b'],
                   help='A: MLP на признаках, B: CNN на пикселях')
    p.add_argument('--epochs', type=int, default=None)
    # Те же ручки, что и во вкладке «Обучение»: параметры прогона, а не правка
    # config — иначе CLI и GUI считают разное при одинаковых числах на экране.
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--name', default=None,
                   help='имя копии в библиотеке (по умолчанию время + метрика)')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('distill', help='дистилляция оракула в EML-формулы')
    p.add_argument('--branch', default='A', choices=BRANCHES)
    p.add_argument('--batch', action='store_true',
                   help='несколько прогонов, лучший по val')
    p.add_argument('--variants', type=int, default=None)
    p.add_argument('--generations', type=int, default=None)
    p.add_argument('--population', type=int, default=None)
    p.add_argument('--depth-penalty', default=None,
                   choices=list(config.DEPTH_PENALTIES.keys()),
                   help='штраф за размер формулы (у ветки C — за число узлов)')
    p.add_argument('--no-acc-phase', action='store_true',
                   help='только DATA-фаза (ветки A и B, для отладки)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--name', default=None,
                   help='имя копии в библиотеке (по умолчанию время + метрика)')
    group = p.add_argument_group('ветка C (длинные прогоны)')
    group.add_argument('--islands', type=int, default=None,
                       help='независимых популяций (0 = по числу процессов)')
    group.add_argument('--workers', type=int, default=None,
                       help='процессов (0 = по числу ядер, 1 = без параллели)')
    group.add_argument('--max-intervals', type=int, default=None,
                       help='потолок числа участков оси в таблице порогов')
    group.add_argument('--checkpoint', default=None,
                       help='файл состояния (по умолчанию runs/checkpoints/...)')
    group.add_argument('--restart', action='store_true',
                       help='начать заново, не продолжая сохранённый прогон')
    p.set_defaults(func=cmd_distill)

    p = sub.add_parser('report', help='графики в results/')
    p.add_argument('--branch', default='A', choices=BRANCHES)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser('export', help='экспорт формул в C')
    p.add_argument('--branch', default='A', choices=BRANCHES)
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser('verify-c', help='скомпилировать C и сверить с Python')
    p.add_argument('-n', type=int, default=200, help='сколько кадров сверять')
    p.add_argument('--branch', default='A', choices=['A', 'C', 'a', 'c'],
                   help='C-экспорт признаков есть у веток A и C')
    p.add_argument('--out', default=None)
    p.set_defaults(func=cmd_verify_c)

    p = sub.add_parser('models', help='библиотека моделей: список и выбор')
    p.add_argument('--kind', default=None, choices=['oracle', 'eml'],
                   help='показать только оракулов или только формулы')
    p.add_argument('--branch', default=None, choices=BRANCHES)
    p.add_argument('--activate', metavar='ИМЯ',
                   help='сделать модель активной')
    p.add_argument('--rename', nargs=2, metavar=('ИМЯ', 'НОВОЕ'),
                   help='переименовать модель')
    p.add_argument('--delete', metavar='ИМЯ',
                   help='удалить модель из библиотеки')
    p.set_defaults(func=cmd_models)

    p = sub.add_parser('bench', help='сравнение скорости оракул vs формулы')
    p.add_argument('-n', type=int, default=500)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser('calibrate', help='проверка масштабов нормализации')
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser('gui', help='графический интерфейс')
    p.add_argument('--tab', default=None,
                   choices=['data', 'features', 'train', 'distill', 'models',
                            'test'],
                   help='вкладка, открытая при старте')
    p.set_defaults(func=cmd_gui)

    return parser


def apply_dataset(args) -> int:
    """
    Выбрать датасет: явный --dataset или запомненный в прошлый раз.

    Явный выбор сразу запоминается — иначе следующая команда работала бы с
    другим набором классов, и половина пайплайна считала бы одно, а половина
    другое.
    """
    if args.dataset:
        try:
            info = config.use_dataset(args.dataset)
        except (ValueError, OSError) as exc:
            print(f"датасет не подошёл: {exc}")
            return 1
        config.save_dataset_choice()
        print(f"[data] датасет: {info['root']} "
              f"({info['n_classes']} классов: {', '.join(info['classes'])})")
    else:
        config.load_dataset_choice(verbose=True)
    return 0


def main():
    args = build_parser().parse_args()
    # У models ветка необязательна (None = все ветки), поэтому проверяем не
    # только наличие атрибута, но и значение.
    if getattr(args, 'branch', None):
        args.branch = args.branch.upper()
    if apply_dataset(args):
        sys.exit(1)
    sys.exit(args.func(args) or 0)


if __name__ == '__main__':
    main()
