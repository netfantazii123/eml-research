"""
dataset_collector.py — сбор датасета для EML-дистилляции (placement-based).

Прогоняет обученный оракул в TetrisEnv. На каждом шаге для КАЖДОЙ легальной
постановки i сохраняется тройка:
    (19 признаков afterstate_i, логит оракула z_i, group_id шага)

EML-формула затем учится воспроизводить оценку оракула как функцию признаков
доски-после-постановки: f(features(afterstate_i)) ≈ z_i. На инференсе
placement* = argmax_i f — ранжирование внутри группы, как у оракула.

group_id нужен для весов: важнее всего точность на постановках, которые
оракул считает хорошими (softmax внутри группы), а не на заведомо плохих.

Интерфейс:
    collect_dataset(oracle, ...) -> (features, logits, groups)
    compute_sample_weights(logits, groups) -> weights
    save_dataset / load_dataset   -> .npz в results/
"""

import os
import time

import numpy as np
import torch

import config
from env import TetrisEnv
from cnn_oracle import get_device, obs_to_tensors


def collect_dataset(
    oracle,
    n_episodes: int | None = None,
    *,
    epsilon_explore: float | None = None,
    max_placements: int | None = None,
    device=None,
    seed: int = 0,
    verbose: bool = True,
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Прогнать оракул и собрать (features, logits, groups).

    Args:
        oracle: обученная TetrisCNN.
        n_episodes: эпизодов (None → config.EML_DATASET_EPISODES).
        epsilon_explore: доля случайных постановок для покрытия пространства
            состояний (None → config.EML_DATA_EPSILON_EXPLORE).
        max_placements: лимит постановок на эпизод
            (None → config.EML_DATA_MAX_PLACEMENTS).
        device: torch-устройство (None → авто).
        seed: сид среды/исследования.

    Returns:
        features: (N, 26) float32 — afterstate-признаки постановки + one-hot
                  следующей фигуры (вход EML-формулы, см. config.FEATURE_NAMES).
        logits:   (N,)   float32 — логит оракула для этой постановки.
        groups:   (N,)   int64   — id шага (группы argmax-конкурентов).
    """
    if n_episodes is None:
        n_episodes = config.EML_DATASET_EPISODES
    if epsilon_explore is None:
        epsilon_explore = config.EML_DATA_EPSILON_EXPLORE
    if max_placements is None:
        max_placements = config.EML_DATA_MAX_PLACEMENTS
    if device is None:
        device = get_device()

    oracle.eval()
    env = TetrisEnv(seed=seed)
    rng = np.random.default_rng(seed + 12345)

    feats_buf: list[np.ndarray] = []
    logits_buf: list[float] = []
    groups_buf: list[int] = []
    group_id = 0

    t0 = time.perf_counter()
    report_every = max(1, n_episodes // 20)

    for ep in range(n_episodes):
        obs = env.reset()
        steps = 0
        while not env.done and steps < max_placements:
            mask = obs['mask']
            legal = np.flatnonzero(mask > 0)
            if len(legal) == 0:
                break

            grid, scalars, afeats, m = obs_to_tensors(obs, device)
            with torch.no_grad():
                logits = oracle.get_logits(grid, scalars, afeats,
                                           m)[0].cpu().numpy()

            # Строка датасета = afterstate-признаки постановки + one-hot
            # СЛЕДУЮЩЕЙ фигуры (scalars[7:14]) → 26-мерный вход формулы.
            next_onehot = obs['scalars'][7:14]
            for a in legal:
                feats_buf.append(
                    np.concatenate([obs['afeats'][a], next_onehot]))
                logits_buf.append(float(logits[a]))
                groups_buf.append(group_id)
            group_id += 1

            # Действие для траектории: argmax оракула + ε-исследование.
            if rng.random() < epsilon_explore:
                action = int(rng.choice(legal))
            else:
                action = int(legal[np.argmax(logits[legal])])

            obs, _, _, _ = env.step_placement(action)
            steps += 1

        if (ep + 1) % report_every == 0 or ep == n_episodes - 1:
            n = len(feats_buf)
            if verbose:
                sps = n / max(1e-9, time.perf_counter() - t0)
                print(f"    ep {ep + 1:>4}/{n_episodes} | "
                      f"samples: {n:>8,} | {sps:,.0f} rows/s")
            if progress_cb is not None:
                progress_cb(ep + 1, n_episodes, n)

    features = np.asarray(feats_buf, dtype=np.float32)
    logits = np.asarray(logits_buf, dtype=np.float32)
    groups = np.asarray(groups_buf, dtype=np.int64)
    return features, logits, groups


def compute_sample_weights(
    logits: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """
    Per-sample веса: базовый 1 + бонус за вероятность постановки у оракула.

    Внутри каждой группы (одного шага) считается softmax логитов; постановки,
    которые оракул реально выбирает, получают больший вес — формула должна
    точнее всего ранжировать верхушку, а не хвост заведомо плохих ходов.
    Веса нормированы к среднему 1 (масштаб MSE не меняется).

    Returns:
        weights: (N,) float64.
    """
    weights = np.ones(len(logits), dtype=np.float64)
    top_w = config.EML_DATA_TOP_WEIGHT
    # Группы идут подряд — обходим отрезками.
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    start = 0
    for end in list(boundaries) + [len(groups)]:
        z = logits[start:end].astype(np.float64)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        # p*K: при равномерном распределении бонус одинаков для всех.
        weights[start:end] = 1.0 + top_w * p * p.size
        start = end
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights


# ── Сохранение / загрузка ────────────────────────────────────────────────────

def save_dataset(features: np.ndarray, logits: np.ndarray,
                 groups: np.ndarray, path: str | None = None) -> str:
    """Сохранить датасет в .npz (по умолчанию results/eml_dataset.npz)."""
    if path is None:
        path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, features=features, logits=logits, groups=groups)
    return path


def load_dataset(path: str | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загрузить датасет из .npz. Возвращает (features, logits, groups)."""
    if path is None:
        path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    data = np.load(path)
    return data['features'], data['logits'], data['groups']


def summarize(features: np.ndarray, logits: np.ndarray,
              groups: np.ndarray) -> str:
    """Короткая сводка по датасету (для логов)."""
    n_groups = len(np.unique(groups))
    per_group = len(features) / max(1, n_groups)
    return (f"samples={len(features):,}  groups={n_groups:,} "
            f"(~{per_group:.1f} placements/step)  "
            f"logit_range=[{logits.min():.2f}, {logits.max():.2f}]")
