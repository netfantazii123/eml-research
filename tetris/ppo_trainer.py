"""
ppo_trainer.py — обучение CNN-оракула методом PPO (placement-based).

Содержит:
    VecTetris   — синхронная векторизованная обёртка над N средами
                  с авто-ресетом и сбором статистики эпизодов.
    train_ppo() — основной цикл: rollout → GAE → clipped surrogate update.

Один шаг среды = одна ПОСТАНОВКА фигуры (см. env.step_placement), поэтому
награда плотная и кредит присваивается мгновенно. Нелегальные постановки
закрываются маской в Categorical-распределении.

Реализация ванильного PPO (Schulman et al. 2017) с GAE (lambda).
Device-aware. Гиперпараметры — из config (резолвятся при вызове).
"""

import time
import collections

import numpy as np
import torch
import torch.nn as nn

import config
import storage
from env import TetrisEnv
from cnn_oracle import (TetrisCNN, get_device, GRID_CHANNELS, SCALAR_SIZE,
                        AFEAT_SIZE)


# ── Векторизованная среда ────────────────────────────────────────────────────

class VecTetris:
    """
    Синхронная векторизованная обёртка над N экземплярами TetrisEnv.

    step() авто-ресетит завершившиеся среды и записывает (return, lines)
    эпизода в очередь статистики, доступную через pop_episode_stats().
    """

    def __init__(self, n_envs: int, seed: int = 0):
        self.n = n_envs
        self.envs = [TetrisEnv(seed=seed + i) for i in range(n_envs)]
        self.ep_returns = np.zeros(n_envs, dtype=np.float64)
        self.ep_lines = np.zeros(n_envs, dtype=np.int64)
        self.ep_lens = np.zeros(n_envs, dtype=np.int64)
        self._stats = collections.deque(maxlen=1000)

    def reset(self):
        obs = [e.reset() for e in self.envs]
        self.ep_returns[:] = 0
        self.ep_lines[:] = 0
        self.ep_lens[:] = 0
        return self._stack(obs)

    def step(self, actions: np.ndarray):
        """
        Args:
            actions: (N,) int — placement-действия.
        Returns:
            grids (N,1,24,10), scalars (N,14), afeats (N,40,19), masks (N,40),
            rewards (N,), dones (N,)
        """
        grids, scalars, afeats, masks, rewards, dones = [], [], [], [], [], []
        for i, env in enumerate(self.envs):
            obs, r, done, info = env.step_placement(int(actions[i]))
            self.ep_returns[i] += r
            self.ep_lines[i] = info['score']
            self.ep_lens[i] += 1
            if done:
                self._stats.append((self.ep_returns[i], self.ep_lines[i],
                                    self.ep_lens[i]))
                self.ep_returns[i] = 0
                self.ep_lines[i] = 0
                self.ep_lens[i] = 0
                obs = env.reset()            # авто-ресет
            grids.append(obs['grid'])
            scalars.append(obs['scalars'])
            afeats.append(obs['afeats'])
            masks.append(obs['mask'])
            rewards.append(r)
            dones.append(done)
        return (np.stack(grids), np.stack(scalars), np.stack(afeats),
                np.stack(masks),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    @staticmethod
    def _stack(obs_list):
        return (np.stack([o['grid'] for o in obs_list]),
                np.stack([o['scalars'] for o in obs_list]),
                np.stack([o['afeats'] for o in obs_list]),
                np.stack([o['mask'] for o in obs_list]))

    def pop_episode_stats(self):
        """Вернуть и очистить накопленную статистику эпизодов."""
        stats = list(self._stats)
        self._stats.clear()
        return stats


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """
    Generalized Advantage Estimation.

    Args:
        rewards, values, dones: (T, N) np.ndarray.
        last_value: (N,) бутстрап-значение состояния после последнего шага.
    Returns:
        advantages (T, N), returns (T, N).
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    last_gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# ── Основной цикл обучения ───────────────────────────────────────────────────

def train_ppo(
    total_steps: int | None = None,
    n_envs: int | None = None,
    rollout: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    gamma: float | None = None,
    gae_lambda: float | None = None,
    clip: float | None = None,
    vf_coef: float | None = None,
    ent_coef: float | None = None,
    target_score: float | None = None,
    seed: int = 0,
    verbose: bool = True,
    log_every: int = 1,
    model: TetrisCNN | None = None,
    should_stop=None,
    on_update=None,
    on_log=None,
    overrides=None,
    autopilot: bool = False,
) -> dict:
    """
    Обучить TetrisCNN методом PPO.

    Гиперпараметры: None → берётся из config на момент вызова.

    Returns:
        dict: 'model', 'history', 'elapsed', 'total_steps', 'best_avg_lines',
              'stop_reason'.
    """
    if total_steps is None: total_steps = config.PPO_TOTAL_STEPS
    if n_envs is None:      n_envs = config.PPO_N_ENVS
    if rollout is None:     rollout = config.PPO_ROLLOUT
    if epochs is None:      epochs = config.PPO_EPOCHS
    if batch_size is None:  batch_size = config.PPO_BATCH_SIZE
    if lr is None:          lr = config.PPO_LR
    if gamma is None:       gamma = config.PPO_GAMMA
    if gae_lambda is None:  gae_lambda = config.PPO_GAE_LAMBDA
    if clip is None:        clip = config.PPO_CLIP
    if vf_coef is None:     vf_coef = config.PPO_VF_COEF
    if ent_coef is None:    ent_coef = config.PPO_ENT_COEF
    if target_score is None: target_score = config.PPO_TARGET_SCORE

    device = get_device()
    if model is None:
        model = TetrisCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    vec = VecTetris(n_envs, seed=seed)
    grids_np, scalars_np, afeats_np, masks_np = vec.reset()

    batch_per_rollout = rollout * n_envs
    n_updates = total_steps // batch_per_rollout

    history = []
    recent_lines = collections.deque(maxlen=100)
    recent_lens = collections.deque(maxlen=100)
    best_avg_lines = -1.0
    stop_reason = ''
    global_step = 0
    t_start = time.perf_counter()

    def _log(msg: str):
        if on_log is not None:
            on_log(msg)
        elif verbose:
            print(f"  [autopilot] {msg}")

    # Автопилот: состояние плато-детектора и авто-сейва.
    overrides = overrides if overrides is not None else {}
    ap_since_improve = 0           # апдейтов без улучшения best
    ap_saved_best = 0.0            # best avg на момент последнего авто-сейва

    def _set_lr(new_lr: float):
        nonlocal lr
        lr = new_lr
        for _pg in optimizer.param_groups:
            _pg['lr'] = new_lr
        overrides['lr'] = new_lr   # чтобы верх цикла не откатил значение

    def _set_ent_coef(new_ec: float):
        nonlocal ent_coef
        ent_coef = new_ec
        overrides['ent_coef'] = new_ec

    for update in range(n_updates):
        if should_stop is not None and should_stop():
            stop_reason = 'cancelled'
            break

        # Apply live hyperparameter overrides from GUI
        if overrides:
            if 'lr' in overrides:
                lr = float(overrides['lr'])
                for _pg in optimizer.param_groups:
                    _pg['lr'] = lr
            if 'gamma' in overrides:      gamma       = float(overrides['gamma'])
            if 'clip' in overrides:       clip        = float(overrides['clip'])
            if 'ent_coef' in overrides:   ent_coef    = float(overrides['ent_coef'])
            if 'vf_coef' in overrides:    vf_coef     = float(overrides['vf_coef'])
            if 'epochs' in overrides:     epochs      = int(overrides['epochs'])
            if 'batch_size' in overrides: batch_size  = int(overrides['batch_size'])
            if 'target' in overrides:     target_score = float(overrides['target'])

        # ── Сбор rollout ──────────────────────────────────────────────────
        b_grids = np.zeros((rollout, n_envs, GRID_CHANNELS,
                            config.BOARD_H_TOTAL, config.BOARD_W),
                           dtype=np.float32)
        b_scalars = np.zeros((rollout, n_envs, SCALAR_SIZE), dtype=np.float32)
        b_afeats = np.zeros((rollout, n_envs, config.N_PLACEMENTS,
                             AFEAT_SIZE), dtype=np.float32)
        b_masks = np.zeros((rollout, n_envs, config.N_PLACEMENTS),
                           dtype=np.float32)
        b_actions = np.zeros((rollout, n_envs), dtype=np.int64)
        b_logprobs = np.zeros((rollout, n_envs), dtype=np.float32)
        b_values = np.zeros((rollout, n_envs), dtype=np.float32)
        b_rewards = np.zeros((rollout, n_envs), dtype=np.float32)
        b_dones = np.zeros((rollout, n_envs), dtype=np.float32)

        model.eval()
        for t in range(rollout):
            b_grids[t] = grids_np
            b_scalars[t] = scalars_np
            b_afeats[t] = afeats_np
            b_masks[t] = masks_np
            g = torch.from_numpy(grids_np).to(device)
            s = torch.from_numpy(scalars_np).to(device)
            af = torch.from_numpy(afeats_np).to(device)
            m = torch.from_numpy(masks_np).to(device)
            action, log_prob, value = model.act(g, s, af, m)
            a_np = action.cpu().numpy()

            (grids_np, scalars_np, afeats_np, masks_np,
             rewards, dones) = vec.step(a_np)

            b_actions[t] = a_np
            b_logprobs[t] = log_prob.cpu().numpy()
            b_values[t] = value.cpu().numpy()
            b_rewards[t] = rewards
            b_dones[t] = dones
            global_step += n_envs

        # Бутстрап значения для последнего состояния
        with torch.no_grad():
            g = torch.from_numpy(grids_np).to(device)
            s = torch.from_numpy(scalars_np).to(device)
            af = torch.from_numpy(afeats_np).to(device)
            m = torch.from_numpy(masks_np).to(device)
            _, last_value = model(g, s, af, m)
            last_value = last_value.cpu().numpy()

        advantages, returns = compute_gae(
            b_rewards, b_values, b_dones, last_value, gamma, gae_lambda)

        # ── Flatten ───────────────────────────────────────────────────────
        f_grids = b_grids.reshape(-1, GRID_CHANNELS, config.BOARD_H_TOTAL,
                                  config.BOARD_W)
        f_scalars = b_scalars.reshape(-1, SCALAR_SIZE)
        f_afeats = b_afeats.reshape(-1, config.N_PLACEMENTS, AFEAT_SIZE)
        f_masks = b_masks.reshape(-1, config.N_PLACEMENTS)
        f_actions = b_actions.reshape(-1)
        f_logprobs = b_logprobs.reshape(-1)
        f_advantages = advantages.reshape(-1)
        f_returns = returns.reshape(-1)

        # Нормализация advantage
        f_advantages = (f_advantages - f_advantages.mean()) / \
                       (f_advantages.std() + 1e-8)

        # ── PPO-обновление (K эпох, мини-батчи) ──────────────────────────
        model.train()
        n_samples = f_grids.shape[0]
        idx = np.arange(n_samples)
        last_pg, last_vf, last_ent = 0.0, 0.0, 0.0
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0, n_samples, batch_size):
                mb = idx[start:start + batch_size]
                mb_grids = torch.from_numpy(f_grids[mb]).to(device)
                mb_scalars = torch.from_numpy(f_scalars[mb]).to(device)
                mb_afeats = torch.from_numpy(f_afeats[mb]).to(device)
                mb_masks = torch.from_numpy(f_masks[mb]).to(device)
                mb_actions = torch.from_numpy(f_actions[mb]).to(device)
                mb_old_logprobs = torch.from_numpy(f_logprobs[mb]).to(device)
                mb_advantages = torch.from_numpy(f_advantages[mb]).to(device)
                mb_returns = torch.from_numpy(f_returns[mb]).to(device)

                new_logprobs, entropy, values = model.evaluate_actions(
                    mb_grids, mb_scalars, mb_afeats, mb_masks, mb_actions)

                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * mb_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                vf_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                last_pg, last_vf, last_ent = (pg_loss.item(), vf_loss.item(),
                                              ent_loss.item())

        # ── Статистика ────────────────────────────────────────────────────
        for _, lines, ep_len in vec.pop_episode_stats():
            recent_lines.append(lines)
            recent_lens.append(ep_len)
        avg_lines = float(np.mean(recent_lines)) if recent_lines else 0.0
        max_lines = int(np.max(recent_lines)) if recent_lines else 0
        avg_len = float(np.mean(recent_lens)) if recent_lens else 0.0

        improved = avg_lines > best_avg_lines
        if improved:
            best_avg_lines = avg_lines

        # ── Автопилот (обучение без присмотра) ───────────────────────────
        if autopilot:
            if improved:
                ap_since_improve = 0
                # Авто-сейв при заметном росте best — прогресс не теряется.
                if best_avg_lines >= ap_saved_best + config.AUTOPILOT_SAVE_DELTA:
                    try:
                        path = storage.save_oracle(
                            model, path=config.AUTOPILOT_SAVE_PATH,
                            meta={'update': update, 'global_step': global_step,
                                  'avg_lines': best_avg_lines})
                        ap_saved_best = best_avg_lines
                        _log(f"saved best avg={best_avg_lines:.2f} -> {path}")
                    except Exception as exc:  # noqa: BLE001
                        _log(f"auto-save failed: {exc!r}")
            else:
                ap_since_improve += 1
                if ap_since_improve >= config.AUTOPILOT_PATIENCE:
                    ap_since_improve = 0
                    # Плато: затухание LR.
                    if lr > config.AUTOPILOT_LR_FLOOR:
                        new_lr = max(config.AUTOPILOT_LR_FLOOR,
                                     lr * config.AUTOPILOT_LR_DECAY)
                        _set_lr(new_lr)
                        _log(f"plateau {config.AUTOPILOT_PATIENCE} upd -> LR "
                             f"{new_lr:.2e}")
                    # Коллапс энтропии: подбросить ent_coef ради разведки.
                    if last_ent < config.AUTOPILOT_ENT_FLOOR and \
                            ent_coef < config.AUTOPILOT_ENT_COEF_MAX:
                        new_ec = min(config.AUTOPILOT_ENT_COEF_MAX,
                                     ent_coef * config.AUTOPILOT_ENT_BOOST)
                        _set_ent_coef(new_ec)
                        _log(f"entropy {last_ent:.2f} low -> ent_coef "
                             f"{new_ec:.3f}")

        elapsed = time.perf_counter() - t_start
        sps = global_step / elapsed
        rec = {
            'update': update,
            'global_step': global_step,
            'avg_lines': avg_lines,
            'max_lines': max_lines,
            'avg_len': avg_len,
            'pg_loss': last_pg,
            'vf_loss': last_vf,
            'entropy': last_ent,
            'sps': sps,
        }
        history.append(rec)
        if on_update is not None:
            on_update(rec)

        if verbose and (update % log_every == 0 or update == n_updates - 1):
            print(f"  Upd {update:4d} | step {global_step:>9,} | "
                  f"lines avg {avg_lines:6.2f} max {max_lines:4d} | "
                  f"len {avg_len:5.0f} | "
                  f"pg {last_pg:+.3f} vf {last_vf:.3f} ent {last_ent:.3f} | "
                  f"{sps:,.0f} sps")

        if target_score and target_score > 0 and avg_lines >= target_score:
            stop_reason = f'target_lines>={target_score}'
            if verbose:
                print(f"  -> Target avg lines {target_score} reached at "
                      f"update {update}.")
            break

    elapsed = time.perf_counter() - t_start
    return {
        'model': model,
        'history': history,
        'elapsed': elapsed,
        'total_steps': global_step,
        'best_avg_lines': best_avg_lines,
        'stop_reason': stop_reason,
    }
