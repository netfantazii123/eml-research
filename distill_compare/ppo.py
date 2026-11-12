"""
ppo.py — a compact PPO for discrete-action gymnasium tasks: the teachers for the
classical control benchmarks.

Kept deliberately small (one file, CPU, a few minutes per task). The teachers
only need to be good, not state of the art: the comparison is between students
of the same teacher, and the teacher's own score is the 100 % line every student
is reported against.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

TEACHER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'teachers')

# Per-task budgets; LunarLander follows the rl-baselines3-zoo PPO settings
# (rollout 1024 x 16 envs, gamma 0.999, GAE 0.98, entropy 0.01).
SETTINGS = {
    'CartPole-v1':    dict(steps=150_000, n_envs=8, rollout=256, lr=3e-4,
                           ent=0.0, gamma=0.99, gae=0.95),
    'Acrobot-v1':     dict(steps=300_000, n_envs=8, rollout=256, lr=3e-4,
                           ent=0.0, gamma=0.99, gae=0.95),
    'LunarLander-v3': dict(steps=2_000_000, n_envs=16, rollout=1024, lr=3e-4,
                           ent=0.01, gamma=0.999, gae=0.98),
}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, n_actions))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(),
                                    nn.Linear(hidden, 1))

    def logits(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.actor(torch.as_tensor(np.atleast_2d(obs),
                                              dtype=torch.float32)).numpy()


def train(env_id: str, seed: int = 0, verbose: bool = True) -> ActorCritic:
    import gymnasium as gym
    cfg = SETTINGS[env_id]
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id)
                                     for _ in range(cfg['n_envs'])])
    obs, _ = envs.reset(seed=seed)
    obs_dim = envs.single_observation_space.shape[0]
    n_act = envs.single_action_space.n
    ac = ActorCritic(obs_dim, n_act)
    opt = torch.optim.Adam(ac.parameters(), lr=cfg['lr'], eps=1e-5)
    n_envs, T = cfg['n_envs'], cfg['rollout']
    n_updates = cfg['steps'] // (n_envs * T)
    ep_ret = np.zeros(n_envs)
    recent: list[float] = []
    t0 = time.time()
    for upd in range(n_updates):
        frac = 1.0 - upd / n_updates
        for g in opt.param_groups:
            g['lr'] = cfg['lr'] * frac
        b_obs = np.zeros((T, n_envs, obs_dim), np.float32)
        b_act = np.zeros((T, n_envs), np.int64)
        b_logp = np.zeros((T, n_envs), np.float32)
        b_rew = np.zeros((T, n_envs), np.float32)
        b_done = np.zeros((T, n_envs), np.float32)
        b_val = np.zeros((T + 1, n_envs), np.float32)
        b_term = np.zeros((T, n_envs), np.float32)
        for t in range(T):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=ac.actor(ot))
                a = dist.sample()
                b_val[t] = ac.critic(ot)[:, 0].numpy()
            b_obs[t] = obs
            b_act[t] = a.numpy()
            b_logp[t] = dist.log_prob(a).numpy()
            obs, r, term, trunc, info = envs.step(a.numpy())
            b_rew[t] = r
            b_term[t] = term
            b_done[t] = np.logical_or(term, trunc)
            ep_ret += r
            for i in np.flatnonzero(b_done[t]):
                recent.append(ep_ret[i])
                ep_ret[i] = 0.0
        with torch.no_grad():
            b_val[T] = ac.critic(torch.as_tensor(obs, dtype=torch.float32))[:, 0].numpy()
        # GAE. Truncation is treated as termination here (a small bias that the
        # time limits of these tasks make harmless).
        adv = np.zeros((T, n_envs), np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t]
            delta = b_rew[t] + cfg['gamma'] * b_val[t + 1] * nonterm - b_val[t]
            last = delta + cfg['gamma'] * cfg['gae'] * nonterm * last
            adv[t] = last
        ret = adv + b_val[:T]
        f_obs = torch.as_tensor(b_obs.reshape(-1, obs_dim))
        f_act = torch.as_tensor(b_act.reshape(-1))
        f_logp = torch.as_tensor(b_logp.reshape(-1))
        f_adv = torch.as_tensor(adv.reshape(-1))
        f_ret = torch.as_tensor(ret.reshape(-1))
        n = len(f_obs)
        for _ in range(4):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                dist = torch.distributions.Categorical(logits=ac.actor(f_obs[idx]))
                logp = dist.log_prob(f_act[idx])
                ratio = torch.exp(logp - f_logp[idx])
                a_ = f_adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, torch.clamp(ratio, 0.8, 1.2) * a_).mean()
                vf = ((ac.critic(f_obs[idx])[:, 0] - f_ret[idx]) ** 2).mean()
                loss = pg + 0.5 * vf - cfg['ent'] * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
        if verbose and (upd % 20 == 0 or upd == n_updates - 1) and recent:
            print(f"  [{env_id}] upd {upd:4d}/{n_updates} | "
                  f"ret(last 50) {np.mean(recent[-50:]):8.1f} | "
                  f"{time.time() - t0:5.0f}s", flush=True)
    envs.close()
    ac.eval()
    return ac


def teacher_path(env_id: str) -> str:
    return os.path.join(TEACHER_DIR, f'{env_id}.pt')


def load_or_train(env_id: str, seed: int = 0, verbose: bool = True) -> ActorCritic:
    import gymnasium as gym
    path = teacher_path(env_id)
    env = gym.make(env_id)
    ac = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    if os.path.exists(path):
        ac.load_state_dict(torch.load(path, map_location='cpu'))
        ac.eval()
        return ac
    torch.set_num_threads(1)
    ac = train(env_id, seed=seed, verbose=verbose)
    os.makedirs(TEACHER_DIR, exist_ok=True)
    torch.save(ac.state_dict(), path)
    return ac
