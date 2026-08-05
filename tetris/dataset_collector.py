"""
dataset_collector.py — dataset collection for EML distillation
(placement-based).

Runs the trained oracle in TetrisEnv. At every step, for EVERY legal placement
i, a triple is stored:
    (19 afterstate_i features, oracle logit z_i, step group_id)

The EML formula then learns to reproduce the oracle's score as a function of the
board-after-placement features: f(features(afterstate_i)) ~ z_i. At inference
placement* = argmax_i f — ranking within a group, as the oracle does.

group_id is needed for the weights: what matters most is accuracy on the
placements the oracle considers good (softmax within a group), not on obviously
bad ones.

Interface:
    collect_dataset(oracle, ...) -> (features, logits, groups)
    compute_sample_weights(logits, groups) -> weights
    save_dataset / load_dataset   -> .npz in results/
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
    Run the oracle and collect (features, logits, groups).

    Args:
        oracle: a trained TetrisCNN.
        n_episodes: episodes (None -> config.EML_DATASET_EPISODES).
        epsilon_explore: fraction of random placements for state space coverage
            (None -> config.EML_DATA_EPSILON_EXPLORE).
        max_placements: placement limit per episode
            (None -> config.EML_DATA_MAX_PLACEMENTS).
        device: torch device (None -> automatic).
        seed: environment/exploration seed.

    Returns:
        features: (N, 26) float32 — the placement's afterstate features plus a
                  one-hot of the next piece (the EML formula input, see
                  config.FEATURE_NAMES).
        logits:   (N,)   float32 — the oracle logit for this placement.
        groups:   (N,)   int64   — step id (groups of argmax competitors).
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

            # A dataset row = the placement's afterstate features + a one-hot of
            # the NEXT piece (scalars[7:14]) -> the 26-dim formula input.
            next_onehot = obs['scalars'][7:14]
            for a in legal:
                feats_buf.append(
                    np.concatenate([obs['afeats'][a], next_onehot]))
                logits_buf.append(float(logits[a]))
                groups_buf.append(group_id)
            group_id += 1

            # Trajectory action: the oracle's argmax + epsilon exploration.
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
    Per-sample weights: a base of 1 plus a bonus for the oracle's placement
    probability.

    Within every group (a single step) a softmax of the logits is computed; the
    placements the oracle actually chooses receive a larger weight — the formula
    must rank the top most accurately, not the tail of obviously bad moves. The
    weights are normalized to a mean of 1 (the MSE scale does not change).

    Returns:
        weights: (N,) float64.
    """
    weights = np.ones(len(logits), dtype=np.float64)
    top_w = config.EML_DATA_TOP_WEIGHT
    # The groups are contiguous — walk them segment by segment.
    boundaries = np.flatnonzero(np.diff(groups)) + 1
    start = 0
    for end in list(boundaries) + [len(groups)]:
        z = logits[start:end].astype(np.float64)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        # p*K: under a uniform distribution the bonus is the same for everyone.
        weights[start:end] = 1.0 + top_w * p * p.size
        start = end
    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights


# ── Saving / loading ─────────────────────────────────────────────────────────

def save_dataset(features: np.ndarray, logits: np.ndarray,
                 groups: np.ndarray, path: str | None = None) -> str:
    """Save the dataset to .npz (default results/eml_dataset.npz)."""
    if path is None:
        path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, features=features, logits=logits, groups=groups)
    return path


def load_dataset(path: str | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset from .npz. Returns (features, logits, groups)."""
    if path is None:
        path = os.path.join(config.RESULTS_DIR, 'eml_dataset.npz')
    data = np.load(path)
    return data['features'], data['logits'], data['groups']


def summarize(features: np.ndarray, logits: np.ndarray,
              groups: np.ndarray) -> str:
    """Short dataset summary (for the logs)."""
    n_groups = len(np.unique(groups))
    per_group = len(features) / max(1, n_groups)
    return (f"samples={len(features):,}  groups={n_groups:,} "
            f"(~{per_group:.1f} placements/step)  "
            f"logit_range=[{logits.min():.2f}, {logits.max():.2f}]")
