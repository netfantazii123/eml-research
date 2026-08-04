"""
reports.py — графики и замеры скорости для отчёта.

Всё пишется в results/. Графики — matplotlib без seaborn и без заданных
цветов: единственный стиль, который одинаково выглядит и в вёрстке диплома,
и в терминале preview.
"""

import os
import time

import numpy as np

import config


# ── Графики ───────────────────────────────────────────────────────────────────

def plot_confusion(cm, title: str, path: str, normalize: bool = True):
    """Матрица ошибок. Нормализованная по строкам — иначе крупный класс давит."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    shown = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)) if normalize else cm

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(shown, interpolation='nearest', cmap='Blues', vmin=0,
                   vmax=shown.max())
    ax.set_title(title)
    ax.set_xlabel('предсказано')
    ax.set_ylabel('истина')
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha='right')
    ax.set_yticklabels(config.CLASS_NAMES)

    threshold = shown.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            label = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, label, ha='center', va='center',
                    color='white' if shown[i, j] > threshold else 'black')

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training(history: dict, title: str, path: str):
    """
    Кривые обучения оракула: лосс и обе val accuracy.

    Рисуются обе — обычная и сбалансированная. Отбор чекпоинта идёт по
    сбалансированной, а расстояние между кривыми показывает, насколько модель
    держится за многочисленный класс.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history['train_loss'])
    ax1.set_title('train loss')
    ax1.set_xlabel('эпоха')
    ax1.grid(alpha=0.3)

    bal = history.get('val_bal_acc') or []
    ax2.plot(history['val_acc'], label='val accuracy', alpha=0.7)
    if bal:
        ax2.plot(bal, label='val bal. accuracy')
        best = int(np.argmax(bal))
        ax2.set_title(f"val bal.accuracy (лучшая {bal[best]:.4f} "
                      f"на эпохе {best})")
    else:                       # чекпоинт, обученный до появления второй кривой
        best = int(np.argmax(history['val_acc']))
        ax2.set_title(f"val accuracy (лучшая {history['val_acc'][best]:.4f} "
                      f"на эпохе {best})")
    ax2.axvline(best, linestyle='--', alpha=0.6)
    ax2.set_xlabel('эпоха')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Замер скорости ────────────────────────────────────────────────────────────

def benchmark(n_images: int = 500, repeats: int = 3, verbose: bool = True) -> dict:
    """
    Сравнить время инференса: оракул против дистиллированных формул.

    Меряется то, что реально выполняется на устройстве:
        ветка A — экстрактор признаков + MLP    против  экстрактор + формулы;
        ветка B — CNN                            против  CNN + формулы.

    Оговорка, без которой цифры вводят в заблуждение: это Python/numpy на
    десктопе, а не C на STM32. Величина показывает соотношение стоимости
    моделей, а не абсолютную скорость на целевом железе; для неё есть
    C-экспорт.
    """
    import torch

    import data
    import eml_distiller as dist
    import features as features_mod
    import oracle as oracle_mod
    import storage

    images, labels, _ = data.load_dataset(verbose=False)
    images = images[:n_images]
    device = torch.device('cpu')          # честное сравнение с MCU-сценарием
    out = {'n_images': n_images, 'repeats': repeats}

    def timeit(fn) -> float:
        """Лучшее из repeats — минимум устойчивее среднего к фоновой нагрузке."""
        best = float('inf')
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best / n_images * 1e6      # мкс на картинку

    out['features_us'] = timeit(lambda: features_mod.extract_batch(images))
    feats = features_mod.extract_batch(images)

    # Модель, обученная на другом наборе классов, теперь отвергается загрузкой
    # (ValueError), и это штатная ситуация: после смены датасета часть моделей
    # просто неприменима. Замер должен пропустить такую строку, а не упасть.
    missing = (FileNotFoundError, ValueError)

    try:
        mlp = oracle_mod.FeatureMLP()
        storage.load_oracle(mlp, branch='A', device=device)
        mlp.eval()
        out['mlp_us'] = timeit(
            lambda: oracle_mod.predict_logits(mlp, feats, device))
    except missing:
        out['mlp_us'] = None

    try:
        cnn = oracle_mod.PixelCNN()
        storage.load_oracle(cnn, branch='B', device=device)
        cnn.eval()
        out['cnn_us'] = timeit(
            lambda: oracle_mod.predict_logits(cnn, images, device))
    except missing:
        out['cnn_us'] = None

    X = feats.astype(np.float64)
    for branch, key in (('A', 'eml'), ('C', 'eml_c')):
        try:
            clf, payload = storage.load_classifier(branch=branch)
            dist.configure_inputs(len(payload['input_names']),
                                  payload['input_names'])
            out[f'{key}_us'] = timeit(lambda c=clf: c.predict(X))
            out[f'{key}_nodes'] = clf.total_nodes()
        except missing:
            out[f'{key}_us'] = None

    if verbose:
        print(f"\nЗАМЕР СКОРОСТИ (CPU, numpy/torch, {n_images} картинок, "
              f"лучшее из {repeats})")
        print(f"  экстрактор признаков : {out['features_us']:8.1f} мкс/кадр")
        if out['mlp_us'] is not None:
            print(f"  MLP-оракул (ветка A) : {out['mlp_us']:8.1f} мкс/кадр")
        if out['cnn_us'] is not None:
            print(f"  CNN-оракул (ветка B) : {out['cnn_us']:8.1f} мкс/кадр")
        if out['eml_us'] is not None:
            print(f"  EML, ветка A         : {out['eml_us']:8.1f} мкс/кадр "
                  f"(формул {config.N_CLASSES}, узлов AST {out['eml_nodes']})")
        if out.get('eml_c_us') is not None:
            print(f"  EML, ветка C         : {out['eml_c_us']:8.1f} мкс/кадр "
                  f"(формула одна, узлов AST {out['eml_c_nodes']})")
        if out['mlp_us'] and out['eml_us']:
            print(f"\n  ветка A: формулы вместо MLP -> "
                  f"x{out['mlp_us'] / out['eml_us']:.1f} по времени модели")
            total_nn = out['features_us'] + out['mlp_us']
            total_eml = out['features_us'] + out['eml_us']
            print(f"  полный кадр: {total_nn:.1f} -> {total_eml:.1f} мкс "
                  f"(x{total_nn / total_eml:.2f}); "
                  f"доминирует экстрактор признаков")
        if out['cnn_us'] and out['eml_us']:
            print(f"  ветка B: CNN {out['cnn_us']:.1f} мкс против "
                  f"{out['features_us'] + out['eml_us']:.1f} мкс у ветки A "
                  f"целиком -> x{out['cnn_us'] / (out['features_us'] + out['eml_us']):.1f}")

    return out
