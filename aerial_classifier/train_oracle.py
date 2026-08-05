"""
train_oracle.py — oracle training (branch A: MLP over features, branch B: CNN).

A common loop for both branches: class-weighted CrossEntropy, Adam with weight
decay, LR decay on a plateau, early stopping on BALANCED val accuracy. The best
checkpoint by validation is saved — not the last one: the final epoch is almost
always more overfitted than the best.

Why selection uses the balanced metric. The dataset is skewed 3.4 : 1, the loss
is already class-weighted, and the project's stated primary metric is balanced
as well. Selecting the checkpoint by plain accuracy would pull the model exactly
where the weighted loss moves it away from: an epoch that predicts the frequent
ROADS better at the cost of the rare BEACH wins on plain accuracy and loses on
balanced. Both curves are written to history — the gap between them measures the
skew.

Augmentation is applied only to branch B (branch A input is already a set of
statistics reduced to scalars, and rotating them is meaningless) and only to
train.

Note on the original TF scripts the dataset came from: there data_augmentation
was attached to inputs, but the backbone received the same inputs bypassing it,
so the augmentation had no effect. Here it is applied to the batch tensor
explicitly.
"""

import time

import numpy as np
import torch
import torch.nn as nn

import config
import data
import eml_distiller as dist
import features as features_mod
import oracle as oracle_mod
import storage


# ── Augmentation (branch B only) ──────────────────────────────────────────────

def augment(batch: torch.Tensor) -> torch.Tensor:
    """
    Geometry + brightness for a batch of images (B, 3, 64, 64).

    Rotations and reflections are chosen per batch rather than per sample: an
    aerial frame has no top or bottom, so all 8 orientations are equivalent, and
    batches are reshuffled every epoch anyway — over a hundred epochs each image
    sees every variant. This way augmentation costs one call instead of
    per-element indexing.
    """
    if config.AUG_ROT90:
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            batch = torch.rot90(batch, k, dims=(2, 3))
    if config.AUG_FLIP and torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=(3,))
    if config.AUG_BRIGHTNESS > 0:
        # Brightness is per-sample: a cheap element-wise multiplication.
        scale = 1.0 + (torch.rand(batch.shape[0], 1, 1, 1, device=batch.device)
                       * 2.0 - 1.0) * config.AUG_BRIGHTNESS
        batch = torch.clamp(batch * scale, 0.0, 1.0)
    return batch


# ── Training ──────────────────────────────────────────────────────────────────

def train(branch: str = 'A', epochs: int | None = None, verbose: bool = True,
          seed: int = 0, lr: float | None = None, batch_size: int | None = None,
          progress_cb=None, should_stop=None, name: str | None = None) -> dict:
    """
    Train the oracle of the selected branch.

    Args:
        branch: 'A' — FeatureMLP over 32 features, 'B' — PixelCNN over pixels.
        lr, batch_size: override config for THIS run. Passed as parameters
            rather than written into config: the GUI used to write the entered
            values into config.TRAIN_LR, and they leaked into every subsequent
            run of the session, including neighboring tabs.
        progress_cb: callable(dict) — invoked after every epoch with the current
            curves. The GUI uses it to draw training live.
        should_stop: callable() -> bool — polled between epochs. True = stop and
            return the BEST checkpoint so far rather than garbage: the "Stop"
            button must stop training, not discard it.
        name: name of the copy in the model library (models/library/). Without
            it the name is assembled from the time and the metric. The active
            slot models/oracle_X.pt is updated in any case.

    Returns:
        dict with the model, metrics, and training history.
    """
    branch = branch.upper()
    if branch not in ('A', 'B'):
        raise ValueError(f"branch must be 'A' or 'B', got {branch!r}")

    epochs = epochs if epochs is not None else config.TRAIN_EPOCHS
    lr = lr if lr is not None else config.TRAIN_LR
    batch_size = batch_size if batch_size is not None else config.TRAIN_BATCH_SIZE
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = oracle_mod.get_device()

    images, labels, _ = data.load_dataset(verbose=verbose)
    train_idx, val_idx, test_idx = data.stratified_split(labels)

    if branch == 'A':
        X = features_mod.load_or_compute(images, verbose=verbose)
        model = oracle_mod.FeatureMLP()
    else:
        X = images
        model = oracle_mod.PixelCNN()
    model = model.to(device)

    if verbose:
        print(f"\n[train] branch {branch}: {type(model).__name__}, "
              f"{oracle_mod.count_params(model):,} parameters, device={device}")
        print(f"[train] {data.describe(labels[train_idx], 'train')}")
        print(f"[train] {data.describe(labels[val_idx], 'val')}")

    y_train = torch.from_numpy(labels[train_idx]).to(device)
    weights = (torch.from_numpy(data.class_weights(labels[train_idx])).to(device)
               if config.USE_CLASS_WEIGHTS else None)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=config.TRAIN_WEIGHT_DECAY)

    n_train = len(train_idx)
    history = {'train_loss': [], 'val_acc': [], 'val_bal_acc': [], 'lr': []}
    best_val_bal_acc, best_state, best_epoch = -1.0, None, -1
    best_val_acc = 0.0
    epochs_since_best = 0
    epochs_since_lr_drop = 0
    stopped_early = False
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0

        for start in range(0, n_train, batch_size):
            sel = perm[start:start + batch_size]
            batch_idx = train_idx[sel.numpy()]
            xb = oracle_mod.to_input_tensor(X[batch_idx], device)
            if branch == 'B':
                xb = augment(xb)
            yb = y_train[sel]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(sel)

        train_loss = total_loss / n_train
        val_acc, val_bal_acc = evaluate(model, X[val_idx], labels[val_idx], device)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_bal_acc'].append(val_bal_acc)
        history['lr'].append(lr)

        # Selection, plateau detection and early stopping all use BALANCED
        # accuracy (see the module header).
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc, best_val_acc = val_bal_acc, val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
            epochs_since_lr_drop = 0
        else:
            epochs_since_best += 1
            epochs_since_lr_drop += 1

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:>3} | loss {train_loss:.4f} | "
                  f"val bal {val_bal_acc:.4f} | val acc {val_acc:.4f} | "
                  f"best {best_val_bal_acc:.4f} | lr {lr:.2e}")

        if progress_cb is not None:
            progress_cb({'epoch': epoch, 'epochs': epochs,
                         'train_loss': train_loss, 'val_acc': val_acc,
                         'val_bal_acc': val_bal_acc, 'lr': lr,
                         'best_val_bal_acc': best_val_bal_acc,
                         'best_epoch': best_epoch,
                         'history': {k: list(v) for k, v in history.items()}})

        # Plateau -> LR decay. A separate counter, so that one decay does not
        # fire every epoch in a row after the first plateau is reached.
        if epochs_since_lr_drop >= config.TRAIN_LR_PATIENCE:
            lr *= config.TRAIN_LR_DECAY
            for group in optimizer.param_groups:
                group['lr'] = lr
            epochs_since_lr_drop = 0
            if verbose:
                print(f"    plateau -> LR {lr:.2e}")

        if should_stop is not None and should_stop():
            stopped_early = True
            if verbose:
                print(f"  stopped by the user at epoch {epoch}")
            break

        if epochs_since_best >= config.TRAIN_PATIENCE:
            if verbose:
                print(f"  early stop at epoch {epoch} "
                      f"(no improvement for {config.TRAIN_PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_bal_acc = evaluate(model, X[test_idx], labels[test_idx], device)
    elapsed = time.time() - t0

    meta = {
        'branch': branch,
        'model': type(model).__name__,
        'params': oracle_mod.count_params(model),
        'best_epoch': best_epoch,
        # Both metrics go into meta and both are labeled: they differ in the
        # report tables, and "0.88" without saying which one misleads the reader.
        'val_acc': best_val_acc,
        'val_bal_acc': best_val_bal_acc,
        'test_acc': test_acc,
        'test_bal_acc': test_bal_acc,
        'selected_by': 'val_bal_acc',
        'epochs_run': len(history['val_acc']),
        'stopped_early': stopped_early,
        'lr': lr,
        'batch_size': batch_size,
        'seconds': elapsed,
        'class_names': config.CLASS_NAMES,
        'feature_names': config.FEATURE_NAMES if branch == 'A' else None,
        # The training curves go into the checkpoint so the GUI can draw them
        # for an already trained model without retraining it.
        'history': history,
    }
    # The library copy name is written into meta BEFORE saving: it is what later
    # identifies which model sits in the active slot.
    meta['library_name'] = storage.reserve_name('oracle', branch, name, meta)
    path = storage.save_oracle(model, branch=branch, meta=meta)
    entry = storage.add_to_library('oracle', branch, meta['library_name'], meta,
                                   source_path=path, overwrite=True)

    if verbose:
        print(f"\n[train] best epoch {best_epoch}: "
              f"val bal {best_val_bal_acc:.4f} (acc {best_val_acc:.4f}), "
              f"test bal {test_bal_acc:.4f} (acc {test_acc:.4f})  ({elapsed:.0f} s)")
        print(f"[train] active model -> {path}")
        print(f"[train] into the library as \"{entry.name}\" -> {entry.path}")
        _warn_if_not_best(entry, verbose=True)

    return {'model': model, 'meta': meta, 'history': history, 'entry': entry,
            'regressed': _warn_if_not_best(entry, verbose=False),
            'splits': (train_idx, val_idx, test_idx)}


def _warn_if_not_best(entry, verbose: bool = True) -> dict | None:
    """
    Warn if the model just trained is worse than the best one in the library.

    Training, unlike distillation, always occupies the active slot: after
    `train` the slot must hold the model that was just trained, otherwise the
    command would misreport its action. But silently replacing a good oracle
    with an unlucky twenty-epoch run is not acceptable either, so the regression
    is stated out loud, and the previous model is not lost: it stays in the
    library and one command brings it back.
    """
    best = next((e for e in storage.list_models('oracle', entry.branch)
                 if e.is_best), None)
    if (best is None or best.name == entry.name
            or best.val_score is None or entry.val_score is None
            or entry.val_score >= best.val_score):
        return None
    if verbose:
        print(f"\n[train] WARNING: the active model is WORSE than the best "
              f"saved one")
        print(f"        now active \"{entry.name}\" — val {entry.val_score:.4f}")
        print(f"        best       \"{best.name}\" — val {best.val_score:.4f}")
        print(f"        revert: python main.py models --activate {best.name}")
    return {'best_name': best.name, 'best_val': best.val_score,
            'current_val': entry.val_score}


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device=None,
                   batch_size: int = 512) -> np.ndarray:
    """Predicted classes of the model for array X — no gradients, in eval mode."""
    device = device or oracle_mod.get_device()
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(X), batch_size):
        xb = oracle_mod.to_input_tensor(X[start:start + batch_size], device)
        out.append(model(xb).argmax(dim=1).cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None,
             batch_size: int = 512) -> tuple[float, float]:
    """(plain accuracy, balanced accuracy) in a single pass over the data."""
    pred = predict_labels(model, X, device, batch_size)
    return float((pred == y).mean()), dist.balanced_accuracy(y, pred)


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None,
             batch_size: int = 512) -> float:
    """Fraction of correct model answers on (X, y)."""
    return evaluate(model, X, y, device, batch_size)[0]


if __name__ == '__main__':
    config.load_dataset_choice(verbose=True)
    train('A')
