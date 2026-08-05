"""
reports.py — plots and speed measurements for the report.

Everything is written to results/. Plots use matplotlib without seaborn and
without explicit colors.
"""

import os
import time

import numpy as np

import config


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion(cm, title: str, path: str, normalize: bool = True):
    """Confusion matrix. Row-normalized, otherwise the large class dominates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    shown = (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)) if normalize else cm

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(shown, interpolation='nearest', cmap='Blues', vmin=0,
                   vmax=shown.max())
    ax.set_title(title)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
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
    Oracle training curves: loss and both val accuracies.

    Both are drawn — plain and balanced. Checkpoint selection uses the balanced
    one, and the distance between the curves shows how much the model leans on
    the frequent class.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history['train_loss'])
    ax1.set_title('train loss')
    ax1.set_xlabel('epoch')
    ax1.grid(alpha=0.3)

    bal = history.get('val_bal_acc') or []
    ax2.plot(history['val_acc'], label='val accuracy', alpha=0.7)
    if bal:
        ax2.plot(bal, label='val bal. accuracy')
        best = int(np.argmax(bal))
        ax2.set_title(f"val bal.accuracy (best {bal[best]:.4f} "
                      f"at epoch {best})")
    else:                       # checkpoint trained before the second curve existed
        best = int(np.argmax(history['val_acc']))
        ax2.set_title(f"val accuracy (best {history['val_acc'][best]:.4f} "
                      f"at epoch {best})")
    ax2.axvline(best, linestyle='--', alpha=0.6)
    ax2.set_xlabel('epoch')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Speed measurement ─────────────────────────────────────────────────────────

def benchmark(n_images: int = 500, repeats: int = 3, verbose: bool = True) -> dict:
    """
    Compare inference time: the oracle against the distilled formulas.

    What is measured is what actually runs on the device:
        branch A — feature extractor + MLP  against  extractor + formulas;
        branch B — CNN                      against  CNN + formulas.

    Note: this is Python/numpy on a desktop, not C on an STM32. The numbers show
    the ratio of model costs rather than absolute speed on the target hardware;
    the C export exists for that.
    """
    import torch

    import data
    import eml_distiller as dist
    import features as features_mod
    import oracle as oracle_mod
    import storage

    images, labels, _ = data.load_dataset(verbose=False)
    images = images[:n_images]
    device = torch.device('cpu')          # fair comparison with the MCU scenario
    out = {'n_images': n_images, 'repeats': repeats}

    def timeit(fn) -> float:
        """Best of `repeats` — the minimum is more robust to background load."""
        best = float('inf')
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best / n_images * 1e6      # µs per image

    out['features_us'] = timeit(lambda: features_mod.extract_batch(images))
    feats = features_mod.extract_batch(images)

    # A model trained on a different class set is now rejected at load time
    # (ValueError), and that is a normal situation: after a dataset change some
    # models are simply inapplicable. The measurement must skip such a row
    # rather than fail.
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
        print(f"\nSPEED MEASUREMENT (CPU, numpy/torch, {n_images} images, "
              f"best of {repeats})")
        print(f"  feature extractor    : {out['features_us']:8.1f} us/frame")
        if out['mlp_us'] is not None:
            print(f"  MLP oracle (branch A): {out['mlp_us']:8.1f} us/frame")
        if out['cnn_us'] is not None:
            print(f"  CNN oracle (branch B): {out['cnn_us']:8.1f} us/frame")
        if out['eml_us'] is not None:
            print(f"  EML, branch A        : {out['eml_us']:8.1f} us/frame "
                  f"({config.N_CLASSES} formulas, {out['eml_nodes']} AST nodes)")
        if out.get('eml_c_us') is not None:
            print(f"  EML, branch C        : {out['eml_c_us']:8.1f} us/frame "
                  f"(one formula, {out['eml_c_nodes']} AST nodes)")
        if out['mlp_us'] and out['eml_us']:
            print(f"\n  branch A: formulas instead of MLP -> "
                  f"x{out['mlp_us'] / out['eml_us']:.1f} on model time")
            total_nn = out['features_us'] + out['mlp_us']
            total_eml = out['features_us'] + out['eml_us']
            print(f"  full frame: {total_nn:.1f} -> {total_eml:.1f} us "
                  f"(x{total_nn / total_eml:.2f}); "
                  f"the feature extractor dominates")
        if out['cnn_us'] and out['eml_us']:
            print(f"  branch B: CNN {out['cnn_us']:.1f} us against "
                  f"{out['features_us'] + out['eml_us']:.1f} us for the whole "
                  f"branch A -> x{out['cnn_us'] / (out['features_us'] + out['eml_us']):.1f}")

    return out
