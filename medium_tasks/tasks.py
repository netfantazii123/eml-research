"""
tasks.py — four control tasks of rising difficulty, vectorised over episodes.

Every task follows the Tetris design, which is the one where the EML pipeline
showed something: a policy is ONE scoring function over the afterstate
features of each candidate action, and the action taken is the argmax. That
keeps a single formula per task for every method, and it lets the GAME phase
play many episodes at once.

API (B parallel episodes, K candidate actions, F afterstate features):

    st = task.reset(seeds)                 # seeds: (B,) ints
    feats, legal = task.candidates(st)     # (B, K, F) float64, (B, K) bool
    r, done = task.step(st, actions)       # actions (B,), mutates st in place

All randomness of an episode is drawn from np.random.default_rng(seed) at
reset, so an episode depends only on its seed and on the actions taken, never
on which other episodes share the batch.

Difficulty ladder (decided before any result, by what the oracle has to
represent):
  1. dock      smooth, 2 features: stop a cart at the target under wind
               (an LQR-like quadratic value).
  2. swingup   smooth but non-convex: pump energy into an underpowered
               pendulum, then balance it (needs trig and a regime switch).
  3. highway   discrete and piecewise: change lanes through traffic whose
               density grows until every episode ends (min / threshold logic).
  4. forager   two competing goals over a long horizon: collect food while a
               predator that keeps getting faster closes in.
"""

from __future__ import annotations

import numpy as np


def _rngs(seeds):
    return [np.random.default_rng(int(s)) for s in seeds]


# ── 1. Dock ──────────────────────────────────────────────────────────────────

class Dock:
    name = 'dock'
    level = 1
    feature_names = ('x', 'v')
    K = 3
    max_steps = 200
    metric_name = 'docked_steps'
    DT, THRUST, WIND = 0.1, 0.5, 0.04

    def reset(self, seeds):
        B = len(seeds)
        x = np.empty(B); v = np.empty(B)
        wind = np.empty((B, self.max_steps))
        for i, r in enumerate(_rngs(seeds)):
            x[i] = r.uniform(-1, 1)
            v[i] = r.uniform(-0.5, 0.5)
            wind[i] = r.normal(0, self.WIND, self.max_steps)
        return {'x': x, 'v': v, 'wind': wind, 't': 0,
                'alive': np.ones(B, dtype=bool)}

    def _next(self, x, v, a, w=0.0):
        v2 = v + self.DT * self.THRUST * a + w
        return x + self.DT * v2, v2

    def candidates(self, st):
        a = np.array([-1.0, 0.0, 1.0])
        x2, v2 = self._next(st['x'][:, None], st['v'][:, None], a[None, :])
        feats = np.stack([x2, v2], -1)
        return feats, np.ones(feats.shape[:2], dtype=bool)

    def step(self, st, actions):
        a = np.asarray(actions, dtype=np.float64) - 1.0
        w = st['wind'][:, st['t']]
        st['x'], st['v'] = self._next(st['x'], st['v'], a, w)
        st['t'] += 1
        # Bounded score: steps spent docked (|x| < 0.05, |v| < 0.1). Leaving
        # the track (|x| > 3) ends the episode. A quadratic cost was unbounded
        # below, which made "% of oracle" meaningless for diverging students.
        r = ((np.abs(st['x']) < 0.05) & (np.abs(st['v']) < 0.1)).astype(float)
        r = np.where(st['alive'], r, 0.0)
        st['alive'] &= np.abs(st['x']) <= 3.0
        done = ~st['alive'] | (st['t'] >= self.max_steps)
        return r, done


# ── 2. Swing-up ──────────────────────────────────────────────────────────────

class SwingUp:
    """theta = 0 is upright. The torque limit (2) is far below gravity (10), so
    the pendulum cannot be lifted directly: energy has to be pumped in."""
    name = 'swingup'
    level = 2
    feature_names = ('cos', 'sin', 'omega')
    K = 3
    max_steps = 400
    metric_name = 'return'
    DT, G, UMAX = 0.05, 10.0, 2.0

    def reset(self, seeds):
        B = len(seeds)
        th = np.empty(B); om = np.empty(B)
        for i, r in enumerate(_rngs(seeds)):
            th[i] = np.pi + r.uniform(-0.3, 0.3)
            om[i] = r.uniform(-0.3, 0.3)
        return {'th': th, 'om': om, 't': 0}

    def _next(self, th, om, a):
        om2 = np.clip(om + self.DT * (self.G * np.sin(th) + self.UMAX * a), -8, 8)
        th2 = th + self.DT * om2
        return th2, om2

    def candidates(self, st):
        a = np.array([-1.0, 0.0, 1.0])
        th2, om2 = self._next(st['th'][:, None], st['om'][:, None], a[None, :])
        feats = np.stack([np.cos(th2), np.sin(th2), om2 / 8.0], -1)
        return feats, np.ones(feats.shape[:2], dtype=bool)

    def step(self, st, actions):
        a = np.asarray(actions, dtype=np.float64) - 1.0
        st['th'], st['om'] = self._next(st['th'], st['om'], a)
        st['t'] += 1
        r = np.cos(st['th'])
        done = np.full(len(r), st['t'] >= self.max_steps)
        return r, done


# ── 3. Highway ───────────────────────────────────────────────────────────────

class Highway:
    """5 lanes; cars approach at their own speeds. Each step the agent keeps
    its lane or moves one lane; a car within 1.0 ahead in the agent's lane (or
    in the target lane during a change) ends the episode. Traffic gets denser
    every step, so every episode ends; return = steps survived."""
    name = 'highway'
    level = 3
    feature_names = ('gap', 'gap_l', 'gap_r', 'closing', 'edge', 'move')
    K = 3
    max_steps = 4000
    metric_name = 'steps'
    LANES, SLOTS, VIEW = 5, 6, 12.0

    def reset(self, seeds):
        B = len(seeds)
        # Pre-drawn spawn stream: for every step, per lane, a uniform and a speed.
        u = np.empty((B, self.max_steps, self.LANES))
        sp = np.empty((B, self.max_steps, self.LANES))
        for i, r in enumerate(_rngs(seeds)):
            u[i] = r.random((self.max_steps, self.LANES))
            sp[i] = r.uniform(0.15, 0.45, (self.max_steps, self.LANES))
        # Cars: (B, LANES, SLOTS) distance ahead (inf = empty) and speed.
        d = np.full((B, self.LANES, self.SLOTS), np.inf)
        s = np.zeros((B, self.LANES, self.SLOTS))
        return {'lane': np.full(B, self.LANES // 2), 'd': d, 's': s,
                'u': u, 'sp': sp, 't': 0, 'alive': np.ones(B, dtype=bool)}

    def _gap(self, d, lane):
        """Nearest car ahead in `lane` (B,) or (B,K); lanes off-road -> 0."""
        B = d.shape[0]
        nearest = d.min(axis=2)                                # (B, LANES)
        lane = np.asarray(lane)
        ok = (lane >= 0) & (lane < self.LANES)
        lc = np.clip(lane, 0, self.LANES - 1)
        g = np.take_along_axis(nearest, lc.reshape(B, -1), 1).reshape(lane.shape)
        g = np.where(ok, np.maximum(g, 0.0), 0.0)
        return np.minimum(g, self.VIEW)

    def _closing(self, st, lane):
        B = st['d'].shape[0]
        lc = np.clip(lane, 0, self.LANES - 1).reshape(B, -1)
        idx = st['d'].argmin(axis=2)                           # (B, LANES)
        spd = np.take_along_axis(st['s'], idx[:, :, None], 2)[:, :, 0]
        return np.take_along_axis(spd, lc, 1).reshape(np.shape(lane))

    def candidates(self, st):
        mv = np.array([-1, 0, 1])
        tgt = st['lane'][:, None] + mv[None, :]                # (B, K)
        legal = (tgt >= 0) & (tgt < self.LANES)
        g = self._gap(st['d'], tgt)
        gl = self._gap(st['d'], tgt - 1)
        gr = self._gap(st['d'], tgt + 1)
        cl = self._closing(st, tgt)
        edge = ((tgt == 0) | (tgt == self.LANES - 1)).astype(float)
        move = np.broadcast_to(np.abs(mv)[None, :], tgt.shape).astype(float)
        feats = np.stack([g / self.VIEW, gl / self.VIEW, gr / self.VIEW,
                          cl, edge, move], -1)
        return feats, legal

    def _lane_min(self, d, lane):
        nearest = d.min(axis=2)
        return np.take_along_axis(nearest, lane[:, None], 1)[:, 0]

    def step(self, st, actions):
        t = st['t']
        mv = np.asarray(actions) - 1
        new = np.clip(st['lane'] + mv, 0, self.LANES - 1)
        # A lane change is blocked by a car level with the agent in the target lane.
        blocked = np.abs(self._lane_min(st['d'], new)) < 0.6
        st['lane'] = np.where((mv != 0) & blocked, st['lane'], new)
        st['d'] = st['d'] - st['s']
        gone = st['d'] < -0.5                                  # passed the agent
        st['d'][gone] = np.inf
        st['s'][gone] = 0.0
        # Density ramp: spawn probability per lane and step.
        p = min(0.9, 0.03 + t * 0.00025)
        fin = np.isfinite(st['d'])
        crowded = (fin & (st['d'] > self.VIEW - 2.5)).any(2)  # keep spacing
        full = fin.sum(2) >= self.SLOTS
        spawn = (st['u'][:, t, :] < p) & ~crowded & ~full
        empty = ~np.isfinite(st['d'])
        slot = empty.argmax(2)
        bi, li = np.nonzero(spawn)
        st['d'][bi, li, slot[bi, li]] = self.VIEW
        st['s'][bi, li, slot[bi, li]] = st['sp'][bi, t, li]
        near = st['d'].min(axis=2)
        hit = np.abs(np.take_along_axis(near, st['lane'][:, None], 1)[:, 0]) < 0.5
        st['t'] += 1
        r = np.where(st['alive'], 1.0, 0.0)
        st['alive'] &= ~hit
        done = ~st['alive'] | (st['t'] >= self.max_steps)
        return r, done


# ── 4. Forager ───────────────────────────────────────────────────────────────

class Forager:
    """Unit square, 3 food items, one predator that chases at a speed rising
    every step. +1 per food; caught -> episode ends."""
    name = 'forager'
    level = 4
    feature_names = ('food1', 'food2', 'pred', 'pred_dx', 'pred_dy', 'wall',
                     'food_dx', 'food_dy')
    K = 5
    max_steps = 4000
    metric_name = 'food'
    SPEED, P0, PGAIN, EAT, CATCH, NFOOD = 0.03, 0.012, 0.000006, 0.04, 0.04, 3
    MOVES = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.float64)

    def reset(self, seeds):
        B = len(seeds)
        pos = np.empty((B, 2)); pred = np.empty((B, 2))
        food = np.empty((B, self.NFOOD, 2))
        stream = np.empty((B, 600, 2))         # respawn positions
        for i, r in enumerate(_rngs(seeds)):
            pos[i] = r.uniform(0.3, 0.7, 2)
            pred[i] = r.uniform(0, 1, 2)
            while np.linalg.norm(pred[i] - pos[i]) < 0.4:
                pred[i] = r.uniform(0, 1, 2)
            food[i] = r.uniform(0.05, 0.95, (self.NFOOD, 2))
            stream[i] = r.uniform(0.05, 0.95, (600, 2))
        return {'pos': pos, 'pred': pred, 'food': food, 'stream': stream,
                'k': np.zeros(B, dtype=int), 't': 0,
                'alive': np.ones(B, dtype=bool)}

    def _pspeed(self, t):
        return self.P0 + self.PGAIN * t

    def candidates(self, st):
        p2 = np.clip(st['pos'][:, None, :] + self.SPEED * self.MOVES[None], 0, 1)
        df = np.linalg.norm(st['food'][:, None, :, :] - p2[:, :, None, :], axis=-1)
        dsort = np.sort(df, axis=-1)                           # (B, K, NFOOD)
        near = np.take_along_axis(
            st['food'][:, None, :, :].repeat(self.K, 1),
            df.argmin(-1)[:, :, None, None].repeat(2, -1), 2)[:, :, 0, :]
        # The predator moves after the agent: where it will be next.
        rel = st['pred'][:, None, :] - p2
        dist = np.linalg.norm(rel, axis=-1)
        step = self._pspeed(st['t'])
        pred2 = st['pred'][:, None, :] - step * rel / (dist[..., None] + 1e-9)
        rel2 = pred2 - p2
        wall = np.minimum(p2, 1 - p2).min(-1)
        fd = near - p2
        feats = np.stack([dsort[..., 0], dsort[..., 1], np.linalg.norm(rel2, axis=-1),
                          rel2[..., 0], rel2[..., 1], wall, fd[..., 0], fd[..., 1]], -1)
        return feats, np.ones(feats.shape[:2], dtype=bool)

    def step(self, st, actions):
        B = len(st['pos'])
        st['pos'] = np.clip(st['pos'] + self.SPEED * self.MOVES[np.asarray(actions)], 0, 1)
        rel = st['pos'] - st['pred']
        dist = np.linalg.norm(rel, axis=-1)
        st['pred'] = st['pred'] + self._pspeed(st['t']) * rel / (dist[:, None] + 1e-9)
        df = np.linalg.norm(st['food'] - st['pos'][:, None, :], axis=-1)
        eat = (df < self.EAT) & st['alive'][:, None]
        r = eat.sum(1).astype(float)
        bi, fi = np.nonzero(eat)
        for b, f in zip(bi, fi):
            st['food'][b, f] = st['stream'][b, st['k'][b] % 600]
            st['k'][b] += 1
        caught = np.linalg.norm(st['pos'] - st['pred'], axis=-1) < self.CATCH
        st['alive'] &= ~caught
        st['t'] += 1
        done = ~st['alive'] | (st['t'] >= self.max_steps)
        return np.where(st['alive'] | caught, r, 0.0), done


TASKS = {c.name: c for c in (Dock, SwingUp, Highway, Forager)}


def play(task, score_fn, seeds, *, collect=False, teacher_fn=None, batch=64):
    """
    Run one episode per seed with the policy argmax(score_fn(feats)).

    score_fn(feats (N, F)) -> (N,) scores of candidate afterstates.
    Returns per-episode metrics (np.ndarray) and, with collect=True, the
    teacher-labelled data {'X', 'T', 'groups'} of every visited decision
    (teacher_fn scores the same candidates; the rank-kind format of
    distill_compare).
    """
    seeds = list(seeds)
    out = np.zeros(len(seeds))
    Xs, Ts, Gs = [], [], []
    gid = 0
    for b0 in range(0, len(seeds), batch):
        sl = seeds[b0:b0 + batch]
        st = task.reset(sl)
        ret = np.zeros(len(sl))
        live = np.ones(len(sl), dtype=bool)
        for _ in range(task.max_steps):
            feats, legal = task.candidates(st)
            B, K, F = feats.shape
            flat = feats.reshape(-1, F)
            with np.errstate(all='ignore'):
                s = np.asarray(score_fn(flat), dtype=np.float64).reshape(B, K)
            s = np.where(np.isfinite(s) & legal, s, -np.inf)
            # All -inf (a NaN formula): the first legal action.
            a = np.where(np.isfinite(s).any(1), s.argmax(1), legal.argmax(1))
            if collect:
                tch = np.asarray(teacher_fn(flat)).reshape(B, K)
                for i in np.flatnonzero(live):
                    lg = np.flatnonzero(legal[i])
                    Xs.append(feats[i, lg]); Ts.append(tch[i, lg])
                    Gs.append(np.full(len(lg), gid)); gid += 1
            r, done = task.step(st, a)
            ret += np.where(live, r, 0.0)
            live &= ~done
            if not live.any():
                break
        out[b0:b0 + len(sl)] = ret
    if collect:
        return out, {'X': np.concatenate(Xs), 'T': np.concatenate(Ts),
                     'groups': np.concatenate(Gs)}
    return out
