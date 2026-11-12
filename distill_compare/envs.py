"""
envs.py — one rollout interface over gymnasium tasks, Flappy Bird and Tetris.

An adapter turns an environment observation into the student's input and the
teacher's output, so that behavioural cloning, VIPER and evaluation are written
once:

    a = Adapter(...)
    obs = a.reset(seed)
    X, legal = a.inputs(obs)    # classify: X (1, d), legal None
                                # rank:     X (n_legal, d), legal action ids
    t = a.teacher(obs)          # classify: (K,) logits; rank: (n_legal,)
    obs, done = a.step(action)  # action = class id (classify) / index into legal (rank)
    a.metric()                  # the task's score for the finished episode

Every evaluation uses a fresh environment per seed, so every policy meets the
same episodes (the seed discipline the task code already follows).
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_task(task_dir: str):
    """Make a task directory importable as top-level modules (as it expects)."""
    path = os.path.join(_ROOT, task_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    if _ROOT not in sys.path:
        sys.path.insert(1, _ROOT)


# ── gymnasium ────────────────────────────────────────────────────────────────

class GymAdapter:
    kind = 'classify'

    def __init__(self, env_id: str, teacher):
        import gymnasium as gym
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.n_actions = self.env.action_space.n
        self.teacher_net = teacher
        self.max_steps = self.env.spec.max_episode_steps or 1000
        self.metric_name = 'return'

    def reset(self, seed):
        obs, _ = self.env.reset(seed=int(seed))
        self._ret = 0.0
        return obs

    def inputs(self, obs):
        return np.asarray(obs, dtype=np.float64)[None, :], None

    def teacher(self, obs):
        return self.teacher_net.logits(obs)[0].astype(np.float64)

    def step(self, action):
        obs, r, term, trunc, _ = self.env.step(int(action))
        self._ret += float(r)
        return obs, bool(term or trunc)

    def metric(self):
        return self._ret


# ── Flappy Bird ──────────────────────────────────────────────────────────────

class FlappyAdapter:
    """
    The GA+MLP oracle outputs a flap probability p; its logits are taken as
    [0, logit(p)] so the binary students see the same margin the oracle
    thresholds at 0.5. Metric = pipes passed, as in flappy's benchmark.
    """
    kind = 'classify'
    n_actions = 2
    metric_name = 'score'

    def __init__(self, oracle_path: str | None = None, max_frames: int = 5000):
        _import_task('flappy_bird_with_ai')
        import torch  # noqa: F401
        from env import FlappyEnv
        from storage import load_ga_model
        self._Env = FlappyEnv
        path = oracle_path or os.path.join(_ROOT, 'flappy_bird_with_ai',
                                           'models', 'best_ga.pt')
        self.oracle = load_ga_model(path)
        self.max_steps = max_frames

    def reset(self, seed):
        self.env = self._Env(seed=int(seed), max_placements=0)  # max_steps caps
        return self.env.reset()

    def inputs(self, obs):
        return np.asarray(obs, dtype=np.float64)[None, :], None

    def teacher(self, obs):
        p = float(np.clip(self.oracle.get_sigmoid(np.asarray(obs, np.float32)),
                          1e-6, 1 - 1e-6))
        return np.array([0.0, np.log(p / (1 - p))])

    def step(self, action):
        obs, _, done = self.env.step(int(action))
        return obs, bool(done)

    def metric(self):
        return float(self.env.score)


# ── Tetris ───────────────────────────────────────────────────────────────────

class TetrisAdapter:
    """
    Placement-level ranking, exactly the EML formula's view: one candidate per
    legal placement, input = 19 afterstate features + next-piece one-hot (26),
    teacher score = the CNN oracle's logit of that placement. Metric = lines.
    """
    kind = 'rank'
    metric_name = 'lines'

    def __init__(self, oracle_path: str | None = None,
                 max_placements: int | None = None):
        _import_task('tetris')
        import torch
        import config
        from env import TetrisEnv
        from cnn_oracle import TetrisCNN, obs_to_tensors
        from storage import load_oracle
        self.torch = torch
        self.config = config
        self._Env = TetrisEnv
        self._to_t = obs_to_tensors
        self.oracle = TetrisCNN()
        load_oracle(self.oracle, oracle_path or os.path.join(
            _ROOT, 'tetris', 'models', 'best_ppo.pt'))
        self.oracle.eval()
        self.device = torch.device('cpu')
        self.max_steps = max_placements or config.EML_INGAME_MAX_PLACEMENTS
        self.feature_names = list(config.FEATURE_NAMES)

    def reset(self, seed):
        self.env = self._Env(seed=int(seed))
        return self.env.reset()

    def inputs(self, obs):
        legal = np.flatnonzero(obs['mask'] > 0)
        X = np.empty((len(legal), self.config.N_FEATURES), dtype=np.float64)
        na = obs['afeats'].shape[1]
        X[:, :na] = obs['afeats'][legal]
        X[:, na:] = obs['scalars'][7:14]
        return X, legal

    def teacher(self, obs):
        legal = np.flatnonzero(obs['mask'] > 0)
        g, s, a, m = self._to_t(obs, self.device)
        with self.torch.no_grad():
            z = self.oracle.get_logits(g, s, a, m)[0].numpy()
        return z[legal].astype(np.float64)

    def step(self, action):
        obs, _, _, _ = self.env.step_placement(int(action))
        return obs, bool(self.env.done)

    def metric(self):
        return float(self.env.score)


# ── Rollouts ─────────────────────────────────────────────────────────────────

def teacher_policy(adapter):
    """Deterministic argmax of the teacher."""
    def act(obs):
        return int(np.argmax(adapter.teacher(obs)))
    return act


def student_policy(adapter, student):
    def act(obs):
        X, _ = adapter.inputs(obs)
        s = student.scores(X)
        if adapter.kind == 'rank':
            return int(np.argmax(np.where(np.isfinite(s), s, -np.inf)))
        return int(np.argmax(s[0]))
    return act


def rollout(adapter, policy, seeds, *, collect: bool = False,
            beta: float = 0.0, rng=None, label_every: int = 1):
    """
    Play one episode per seed with `policy`.

    collect: also record every visited state's student input and the teacher's
    label (the DAgger / VIPER data). beta: probability of executing the
    teacher's action instead of the policy's (DAgger's mixture).
    label_every: record only every n-th visited state (thins long episodes).

    Returns (metrics list, data dict or None). data has X, T (classify: (N, K);
    rank: (N,)), groups (decision id per row).
    """
    rng = rng or np.random.default_rng(0)
    metrics = []
    Xs, Ts, Gs = [], [], []
    gid = 0
    for s in seeds:
        obs = adapter.reset(s)
        done = False
        steps = 0
        while not done and steps < adapter.max_steps:
            need_teacher = collect or beta > 0
            t = adapter.teacher(obs) if need_teacher else None
            if adapter.kind == 'rank':
                X, legal = adapter.inputs(obs)
                if len(legal) == 0:
                    break
            if collect and steps % label_every == 0:
                X, _ = adapter.inputs(obs)
                Xs.append(X)
                if adapter.kind == 'rank':
                    Ts.append(t)
                    Gs.append(np.full(len(t), gid))
                else:
                    Ts.append(t[None, :])
                    Gs.append(np.array([gid]))
                gid += 1
            if beta > 0 and rng.random() < beta:
                a = int(np.argmax(t))
            else:
                a = policy(obs)
            if adapter.kind == 'rank':
                a = int(legal[a])
            obs, done = adapter.step(a)
            steps += 1
        metrics.append(adapter.metric())
    data = None
    if collect:
        data = {'X': np.concatenate(Xs), 'T': np.concatenate(Ts),
                'groups': np.concatenate(Gs)}
    return metrics, data


def concat_data(chunks: list[dict]) -> dict:
    """Aggregate rollout data, keeping group ids distinct across chunks."""
    out = {'X': [], 'T': [], 'groups': []}
    off = 0
    for c in chunks:
        out['X'].append(c['X'])
        out['T'].append(c['T'])
        out['groups'].append(c['groups'] + off)
        off += int(c['groups'].max()) + 1 if len(c['groups']) else 0
    return {k: np.concatenate(v) for k, v in out.items()}


def fidelity(adapter_kind: str, student, data: dict) -> float:
    """Fraction of decisions where the student picks the teacher's action."""
    s = student.scores(data['X'])
    if adapter_kind == 'classify':
        return float(np.mean(np.argmax(s, 1) == np.argmax(data['T'], 1)))
    g = data['groups']
    s = np.where(np.isfinite(s), s, -np.inf)
    _, inv = np.unique(g, return_inverse=True)
    hits = []
    start = 0
    bounds = np.flatnonzero(np.diff(inv)) + 1
    for end in list(bounds) + [len(inv)]:
        hits.append(np.argmax(s[start:end]) == np.argmax(data['T'][start:end]))
        start = end
    return float(np.mean(hits))
