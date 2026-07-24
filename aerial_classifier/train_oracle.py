"""
train_oracle.py — обучение оракула (ветка A: MLP на признаках, ветка B: CNN).

Общий цикл на обе ветки: взвешенный по классам CrossEntropy, Adam с weight
decay, затухание LR на плато, ранняя остановка по СБАЛАНСИРОВАННОЙ val
accuracy. Лучший по валидации чекпоинт сохраняется — не последний: последняя
эпоха почти всегда переобучена сильнее лучшей.

Почему отбор именно по сбалансированной. Датасет перекошен 3.4 : 1, лосс уже
взвешен по классам, и заявленная главная метрика проекта — тоже
сбалансированная. Если при этом отбирать чекпоинт по обычной accuracy, отбор
тянет модель ровно туда, откуда её уводит взвешенный лосс: эпоха, лучше
угадывающая многочисленный ROADS ценой редкого BEACH, выигрывает по обычной
accuracy и проигрывает по сбалансированной. Обе кривые пишутся в history —
разрыв между ними и есть мера перекоса.

Аугментация применяется только к ветке B (у ветки A вход — уже свёрнутые в
скаляры статистики, и вертеть их бессмысленно) и только к train.

Замечание про исходные TF-скрипты, из которых пришёл датасет: там
data_augmentation подключалась к inputs, но backbone получал те же самые
inputs в обход неё, так что аугментация фактически не работала. Здесь она
применяется к тензору батча явно.
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


# ── Аугментация (только ветка B) ──────────────────────────────────────────────

def augment(batch: torch.Tensor) -> torch.Tensor:
    """
    Геометрия + яркость для пачки картинок (B, 3, 64, 64).

    Повороты и отражения выбираются на весь батч, а не на каждый семпл: у
    аэроснимка нет верха и низа, поэтому все 8 ориентаций равноправны, а
    батчи всё равно перемешиваются каждую эпоху — за сотню эпох каждая
    картинка увидит все варианты. Так аугментация стоит один вызов вместо
    поэлементной индексации.
    """
    if config.AUG_ROT90:
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            batch = torch.rot90(batch, k, dims=(2, 3))
    if config.AUG_FLIP and torch.rand(1).item() < 0.5:
        batch = torch.flip(batch, dims=(3,))
    if config.AUG_BRIGHTNESS > 0:
        # Яркость — поштучно: это дешёвое поэлементное умножение.
        scale = 1.0 + (torch.rand(batch.shape[0], 1, 1, 1, device=batch.device)
                       * 2.0 - 1.0) * config.AUG_BRIGHTNESS
        batch = torch.clamp(batch * scale, 0.0, 1.0)
    return batch


# ── Обучение ──────────────────────────────────────────────────────────────────

def train(branch: str = 'A', epochs: int | None = None, verbose: bool = True,
          seed: int = 0, lr: float | None = None, batch_size: int | None = None,
          progress_cb=None, should_stop=None, name: str | None = None) -> dict:
    """
    Обучить оракула выбранной ветки.

    Args:
        branch: 'A' — FeatureMLP на 32 признаках, 'B' — PixelCNN на пикселях.
        lr, batch_size: переопределяют config для ЭТОГО прогона. Именно
            параметрами, а не записью в config: интерфейс раньше писал введённые
            значения в config.TRAIN_LR, и они протекали во все последующие
            запуски сессии, включая соседние вкладки.
        progress_cb: callable(dict) — вызывается после каждой эпохи с текущими
            кривыми. Через него интерфейс рисует обучение вживую.
        should_stop: callable() -> bool — опрашивается между эпохами. True =
            прекратить и вернуть ЛУЧШИЙ на этот момент чекпоинт, а не мусор:
            кнопка «Стоп» должна останавливать обучение, а не выбрасывать его.
        name: имя копии в библиотеке моделей (models/library/). Без него имя
            собирается из времени и метрики. Активный слот models/oracle_X.pt
            обновляется в любом случае.

    Returns:
        dict с моделью, метриками и историей обучения.
    """
    branch = branch.upper()
    if branch not in ('A', 'B'):
        raise ValueError(f"ветка должна быть 'A' или 'B', получено {branch!r}")

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
        print(f"\n[train] ветка {branch}: {type(model).__name__}, "
              f"{oracle_mod.count_params(model):,} параметров, device={device}")
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

        # Отбор, плато и ранняя остановка — все три по СБАЛАНСИРОВАННОЙ
        # accuracy (см. шапку модуля).
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

        # Плато -> затухание LR. Отдельный счётчик, чтобы одно затухание не
        # срабатывало каждую эпоху подряд после первого выхода на плато.
        if epochs_since_lr_drop >= config.TRAIN_LR_PATIENCE:
            lr *= config.TRAIN_LR_DECAY
            for group in optimizer.param_groups:
                group['lr'] = lr
            epochs_since_lr_drop = 0
            if verbose:
                print(f"    плато -> LR {lr:.2e}")

        if should_stop is not None and should_stop():
            stopped_early = True
            if verbose:
                print(f"  остановлено пользователем на эпохе {epoch}")
            break

        if epochs_since_best >= config.TRAIN_PATIENCE:
            if verbose:
                print(f"  ранняя остановка на эпохе {epoch} "
                      f"(нет улучшения {config.TRAIN_PATIENCE} эпох)")
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
        # Обе метрики в meta и обе подписаны: в таблицах отчёта они разные,
        # и «0.88» без указания, какая именно, читателя обманывает.
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
        # Кривые обучения кладутся в чекпоинт, чтобы GUI мог нарисовать их для
        # уже обученной модели, не переобучая её заново.
        'history': history,
    }
    # Имя библиотечной копии пишется в meta ДО сохранения: по нему потом
    # опознаётся, какая именно модель лежит в активном слоте.
    meta['library_name'] = storage.reserve_name('oracle', branch, name, meta)
    path = storage.save_oracle(model, branch=branch, meta=meta)
    entry = storage.add_to_library('oracle', branch, meta['library_name'], meta,
                                   source_path=path, overwrite=True)

    if verbose:
        print(f"\n[train] лучшая эпоха {best_epoch}: "
              f"val bal {best_val_bal_acc:.4f} (acc {best_val_acc:.4f}), "
              f"test bal {test_bal_acc:.4f} (acc {test_acc:.4f})  ({elapsed:.0f} c)")
        print(f"[train] активная модель -> {path}")
        print(f"[train] в библиотеку как «{entry.name}» -> {entry.path}")
        _warn_if_not_best(entry, verbose=True)

    return {'model': model, 'meta': meta, 'history': history, 'entry': entry,
            'regressed': _warn_if_not_best(entry, verbose=False),
            'splits': (train_idx, val_idx, test_idx)}


def _warn_if_not_best(entry, verbose: bool = True) -> dict | None:
    """
    Предупредить, если только что обученная модель хуже лучшей в библиотеке.

    Обучение, в отличие от дистилляции, всегда занимает активный слот: после
    `train` там должна лежать именно та модель, которую только что обучили —
    иначе команда врала бы о своём действии. Но тихо заменить хорошего оракула
    неудачным прогоном на двадцать эпох тоже нельзя, поэтому регресс
    проговаривается вслух, а прежняя модель никуда не девается: она осталась в
    библиотеке, и вернуть её — одна команда.
    """
    best = next((e for e in storage.list_models('oracle', entry.branch)
                 if e.is_best), None)
    if (best is None or best.name == entry.name
            or best.val_score is None or entry.val_score is None
            or entry.val_score >= best.val_score):
        return None
    if verbose:
        print(f"\n[train] ВНИМАНИЕ: активной стала модель ХУЖЕ лучшей "
              f"сохранённой")
        print(f"        сейчас активна «{entry.name}» — val {entry.val_score:.4f}")
        print(f"        лучшая  «{best.name}» — val {best.val_score:.4f}")
        print(f"        вернуть: python main.py models --activate {best.name}")
    return {'best_name': best.name, 'best_val': best.val_score,
            'current_val': entry.val_score}


@torch.no_grad()
def predict_labels(model: nn.Module, X: np.ndarray, device=None,
                   batch_size: int = 512) -> np.ndarray:
    """Предсказанные классы модели для массива X — без градиентов и в eval."""
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
    """(обычная accuracy, сбалансированная accuracy) одним проходом по данным."""
    pred = predict_labels(model, X, device, batch_size)
    return float((pred == y).mean()), dist.balanced_accuracy(y, pred)


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None,
             batch_size: int = 512) -> float:
    """Доля правильных ответов модели на (X, y)."""
    return evaluate(model, X, y, device, batch_size)[0]


if __name__ == '__main__':
    train('A')
