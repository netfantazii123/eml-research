"""
checkpoint.py — saving a GA run and resuming it from a chosen point.

Why this exists: a DATA phase is hundreds of generations and a GAME phase plays
thousands of games, and until now a run lived only in memory. A crash, a closed
GUI or a reboot threw the whole run away, and there was no way to say "take the
population from generation 300, where the curve was still rising, and continue
it for another 200 with the new dataset".

What is saved is the state at the TOP of a generation — before the subsample is
drawn or anything is evaluated: the population, every counter the loop keeps,
the archive, and the state of both random generators. Resuming from that point
replays the run exactly: a run stopped at generation 100 and resumed produces
the same trees, bit for bit, as one that was never stopped (tests check this).
That is what makes a resumed result as citable as an uninterrupted one.

Layout of a checkpoint directory (one per phase):

    latest.json                rolling state, rewritten every `every` generations
                               and when the run ends or is stopped
    snapshots/gen_000050.json  kept every `snapshot_every` generations — the
                               points a run can be resumed from later
    best/gen_000123.json       the state of the generation where a new best (by
                               validation) appeared; the last `keep_best` are kept
    hall_of_fame.json          the top `hof_size` DISTINCT formulas seen, with
                               their validation score and generation

A "best" state is not just the best tree: it is the whole population that
produced it, so continuing from it explores around the best performer with the
diversity that found it, rather than restarting from one tree.

Everything is JSON and written atomically (tmp + replace): a power loss mid-write
must not destroy the save it was meant to make.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import time

import numpy as np

FORMAT_VERSION = 1


# ── Low-level IO ──────────────────────────────────────────────────────────────

def atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ── Random generator state ────────────────────────────────────────────────────

def python_rng_state() -> list:
    version, internal, gauss_next = random.getstate()
    return [version, list(internal), gauss_next]


def set_python_rng_state(state) -> None:
    version, internal, gauss_next = state
    random.setstate((version, tuple(internal), gauss_next))


def numpy_rng_state(rng: np.random.Generator) -> dict:
    # PCG64 state is a dict of (big) ints — JSON keeps arbitrary-size integers.
    return rng.bit_generator.state


def numpy_rng_from_state(state: dict) -> np.random.Generator:
    bg = getattr(np.random, state['bit_generator'])()
    bg.state = state
    return np.random.Generator(bg)


# ── Signatures ────────────────────────────────────────────────────────────────

def data_fingerprint(*arrays) -> str:
    """
    A short hash of the training data.

    A resumed run that silently picked up a different dataset would produce a
    result matching neither run. The fingerprint lets the loader say so — and
    lets the caller decide that continuing on new data (e.g. a DAgger-grown
    dataset) is exactly what is wanted.
    """
    h = hashlib.sha1()
    for a in arrays:
        if a is None:
            h.update(b'none')
            continue
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def basis_spec(basis) -> dict:
    return {'name': basis.name, 'n_vars': basis.n_vars,
            'max_depth': basis.max_depth, 'max_nodes': basis.max_nodes}


class SignatureMismatch(RuntimeError):
    pass


def check_signature(saved: dict, current: dict, *, allow: tuple = ()) -> list[str]:
    """
    Keys whose values differ. Raises unless every differing key is in `allow`.

    The basis is never allowed to differ: trees from one basis cannot even be
    parsed in another.
    """
    diff = sorted(k for k in set(saved) | set(current)
                  if saved.get(k) != current.get(k))
    bad = [k for k in diff if k not in allow]
    if bad:
        detail = ", ".join(f"{k}: saved={saved.get(k)!r} now={current.get(k)!r}"
                           for k in bad)
        raise SignatureMismatch(
            f"checkpoint belongs to a different run ({detail}). Pass "
            f"allow_changes={tuple(bad)!r} if continuing on changed settings "
            f"is intended.")
    return diff


# ── State capture / serialisation ─────────────────────────────────────────────

def capture(gen: int, population, *, counters: dict, rng=None,
            archive=None, extra: dict | None = None) -> dict:
    """
    In-memory state at the top of generation `gen`.

    Cheap on purpose (references and small copies, no serialisation), so it can
    be taken every generation: trees are never mutated in place by the search —
    mutation and crossover clone — so holding references is safe.
    """
    return {
        'generation': int(gen),
        'population': list(population),
        # Lists are copied: the loop keeps appending to its history after the
        # capture, and the saved state must be the one at the top.
        'counters': {k: (list(v) if isinstance(v, list) else v)
                     for k, v in counters.items()},
        'py_rng': random.getstate(),
        'np_rng': numpy_rng_state(rng) if rng is not None else None,
        'archive': dict(archive.best) if archive is not None else None,
        'extra': dict(extra or {}),
    }


def serialise(cap: dict, *, phase: str, signature: dict,
              generations: int, schedule: int | None = None) -> dict:
    version, internal, gauss_next = cap['py_rng']
    archive = None
    if cap['archive'] is not None:
        archive = [{'size': int(size), 'score': float(score), 'tree': t.to_dict()}
                   for size, (score, t) in sorted(cap['archive'].items())]

    def _tree(t):
        return t.to_dict() if t is not None else None

    counters = {}
    for k, v in cap['counters'].items():
        if hasattr(v, 'to_dict'):
            counters[k] = {'__tree__': v.to_dict()}
        elif isinstance(v, (list, tuple)):
            counters[k] = list(v)
        else:
            counters[k] = v
    return {
        'format': FORMAT_VERSION,
        'phase': phase,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'signature': signature,
        'generation': cap['generation'],
        'generations': int(generations),
        'schedule': int(schedule if schedule is not None else generations),
        'population': [_tree(t) for t in cap['population']],
        'counters': counters,
        'rng': {'python': [version, list(internal), gauss_next],
                'numpy': cap['np_rng']},
        'archive': archive,
        'extra': cap['extra'],
    }


def restore(state: dict, basis, *, restore_rng: bool = True) -> dict:
    """
    Rebuild the run state: trees bound to `basis`, counters, archive, RNG.

    Returns {'generation', 'generations', 'population', 'counters', 'archive',
    'np_rng', 'extra', 'signature'}. Sets the global `random` state when
    restore_rng (the default — that is what makes the replay exact).
    """
    from .node import Node
    from .nsga2 import ParetoArchive

    if state.get('format') != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {state.get('format')}")

    pop = [Node.from_dict(d, basis) for d in state['population']]
    counters = {}
    for k, v in state['counters'].items():
        if isinstance(v, dict) and '__tree__' in v:
            counters[k] = Node.from_dict(v['__tree__'], basis)
        else:
            counters[k] = v

    archive = None
    if state.get('archive') is not None:
        archive = ParetoArchive()
        for row in state['archive']:
            archive.best[int(row['size'])] = (
                float(row['score']), Node.from_dict(row['tree'], basis))

    np_rng = None
    if restore_rng:
        set_python_rng_state(state['rng']['python'])
        if state['rng'].get('numpy') is not None:
            np_rng = numpy_rng_from_state(state['rng']['numpy'])

    return {'generation': int(state['generation']),
            'generations': int(state['generations']),
            'schedule': int(state.get('schedule', state['generations'])),
            'population': pop, 'counters': counters, 'archive': archive,
            'np_rng': np_rng, 'extra': state.get('extra', {}),
            'signature': state.get('signature', {})}


# ── The checkpointer a GA loop talks to ───────────────────────────────────────

class RunCheckpointer:
    """
    Decides what to write and when; the GA loop only reports what happened.

    Per generation the loop calls on_generation(cap, improved=...) with a cheap
    capture() of the TOP of that generation. When the generation's champion
    turns out to be a new best, the same capture is written under best/ —
    resuming from it re-evaluates that generation and finds the best again.
    """

    def __init__(self, directory: str, *, phase: str, signature: dict,
                 generations: int, schedule: int | None = None,
                 every: int = 10, snapshot_every: int = 50,
                 keep_best: int = 5, hof_size: int = 20,
                 resume_gen: int | None = None):
        self.directory = directory
        self.phase = phase
        self.signature = signature
        self.generations = generations
        self.schedule = schedule if schedule is not None else generations
        self.every = max(1, int(every))
        self.snapshot_every = int(snapshot_every)
        self.keep_best = int(keep_best)
        self.hof_size = int(hof_size)
        self._hof: dict[str, dict] = {}
        self._dirty = False
        os.makedirs(directory, exist_ok=True)
        self._set_aside_stale(resume_gen)
        hof_path = os.path.join(directory, 'hall_of_fame.json')
        if resume_gen is not None and os.path.exists(hof_path):
            try:
                for row in read_json(hof_path).get('entries', []):
                    self._hof[row['key']] = row
            except (OSError, ValueError, KeyError):
                pass

    def _set_aside_stale(self, resume_gen: int | None) -> None:
        """
        Move files that do not belong to this run's timeline out of the way.

        A fresh run in a directory that already holds checkpoints would otherwise
        inherit the old hall of fame, and best/ pruning (by file name) would
        delete the NEW run's states while keeping the old run's later
        generations — resolve_point('best') then returns another run's state.
        A resume from generation g keeps everything up to g and sets aside the
        best/ and snapshots/ files after g (an abandoned continuation).
        Nothing is deleted: the files go to superseded/<timestamp>/.
        """
        d = self.directory
        stale: list[str] = []
        for sub in ('best', 'snapshots'):
            for p in glob.glob(os.path.join(d, sub, 'gen_*.json')):
                try:
                    g = int(os.path.basename(p)[4:-5])
                except ValueError:
                    continue
                if resume_gen is None or g > resume_gen:
                    stale.append(p)
        latest = os.path.join(d, 'latest.json')
        if os.path.exists(latest):
            try:
                lg = int(read_json(latest)['generation'])
            except (OSError, ValueError, KeyError):
                lg = None
            if resume_gen is None or lg is None or lg > resume_gen:
                stale.append(latest)
        hof = os.path.join(d, 'hall_of_fame.json')
        if resume_gen is None and os.path.exists(hof):
            stale.append(hof)
        if not stale:
            return
        dest = os.path.join(d, 'superseded',
                            time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}')
        for p in stale:
            rel = os.path.relpath(p, d)
            os.makedirs(os.path.dirname(os.path.join(dest, rel)), exist_ok=True)
            os.replace(p, os.path.join(dest, rel))

    # paths
    @property
    def latest_path(self) -> str:
        return os.path.join(self.directory, 'latest.json')

    def _write(self, path: str, cap: dict) -> str:
        atomic_write_json(path, serialise(cap, phase=self.phase,
                                          signature=self.signature,
                                          generations=self.generations,
                                          schedule=self.schedule))
        return path

    def on_generation(self, cap: dict, *, improved: bool) -> None:
        gen = cap['generation']
        if gen % self.every == 0:
            self._write(self.latest_path, cap)
        if self.snapshot_every > 0 and gen % self.snapshot_every == 0:
            self._write(os.path.join(self.directory, 'snapshots',
                                     f'gen_{gen:06d}.json'), cap)
        if improved and self.keep_best > 0:
            self._write(os.path.join(self.directory, 'best',
                                     f'gen_{gen:06d}.json'), cap)
            old = sorted(glob.glob(os.path.join(self.directory, 'best',
                                                'gen_*.json')))
            for p in old[:-self.keep_best]:
                try:
                    os.remove(p)
                except OSError:
                    pass
        if improved or gen % self.every == 0:
            self.flush()

    def finish(self, cap: dict) -> None:
        """Write latest.json at the end of a run (finished, stopped or crashed)."""
        self._write(self.latest_path, cap)
        self._flush_hof()

    def record(self, tree, score: float, gen: int, **info) -> None:
        """Offer one validated formula to the hall of fame (top-N distinct)."""
        key = tree.key()
        cur = self._hof.get(key)
        if cur is not None and cur['score'] >= score:
            return
        self._hof[key] = {'key': key, 'score': float(score), 'generation': int(gen),
                          'nodes': tree.size(), 'depth': tree.depth(),
                          'formula': tree.to_string(), 'tree': tree.to_dict(),
                          **info}
        if len(self._hof) > self.hof_size:
            worst = min(self._hof.values(), key=lambda r: r['score'])
            del self._hof[worst['key']]
        self._dirty = True

    def _flush_hof(self) -> None:
        rows = sorted(self._hof.values(), key=lambda r: -r['score'])
        atomic_write_json(os.path.join(self.directory, 'hall_of_fame.json'),
                          {'phase': self.phase, 'signature': self.signature,
                           'entries': rows})

    def flush(self) -> None:
        if self._dirty:
            self._flush_hof()
            self._dirty = False


# ── Finding and loading a resume point ────────────────────────────────────────

def list_points(directory: str) -> list[dict]:
    """Every state a run can be resumed from: [{'generation','kind','path'}]."""
    out = []
    for kind in ('snapshots', 'best'):
        for p in glob.glob(os.path.join(directory, kind, 'gen_*.json')):
            gen = int(os.path.basename(p)[4:10])
            out.append({'generation': gen, 'kind': kind.rstrip('s'), 'path': p})
    latest = os.path.join(directory, 'latest.json')
    if os.path.exists(latest):
        try:
            out.append({'generation': int(read_json(latest)['generation']),
                        'kind': 'latest', 'path': latest})
        except (OSError, ValueError, KeyError):
            pass
    return sorted(out, key=lambda r: (r['generation'], r['kind']))


def resolve_point(source: str, point='latest') -> str:
    """
    Path of the state to resume from.

    source: a checkpoint directory or a state file. point (directory only):
      'latest'  — where the run stopped;
      'best'    — the most recent new-best state;
      int / '120' — the newest saved state at or before that generation.
    """
    if os.path.isfile(source):
        return source
    if not os.path.isdir(source):
        raise FileNotFoundError(f"no checkpoint at {source}")
    points = list_points(source)
    if not points:
        raise FileNotFoundError(f"{source} holds no saved states")
    if point in (None, 'latest'):
        latest = [p for p in points if p['kind'] == 'latest']
        return (latest or points)[-1]['path']
    if point == 'best':
        best = [p for p in points if p['kind'] == 'best']
        if not best:
            raise FileNotFoundError(f"{source} holds no best-performer states")
        return best[-1]['path']
    gen = int(point)
    before = [p for p in points if p['generation'] <= gen]
    if not before:
        raise FileNotFoundError(
            f"no state at or before generation {gen} in {source} "
            f"(earliest is {points[0]['generation']})")
    return before[-1]['path']


def load_state(source: str, point='latest') -> dict:
    return read_json(resolve_point(source, point))


def load_hall_of_fame(directory: str, basis=None) -> list[dict]:
    """Hall-of-fame rows, best first; with basis, each row gains a 'node'."""
    path = directory if directory.endswith('.json') else os.path.join(
        directory, 'hall_of_fame.json')
    rows = read_json(path).get('entries', [])
    if basis is not None:
        from .node import Node
        for r in rows:
            r['node'] = Node.from_dict(r['tree'], basis)
    return rows


# ── The glue a GA loop uses ───────────────────────────────────────────────────

class ResumableRun:
    """
    Everything a GA loop needs to be stoppable and resumable, in one object.

        run = ResumableRun(phase='data', basis=basis, signature=sig,
                           generations=G, checkpoint_dir=d, resume=src)
        if run.restored: population, counters... = run.restored[...]
        for gen in range(run.start_gen, run.generations):
            run.top(gen, population, counters=..., rng=rng, archive=archive)
            ... evaluate, pick champion ...
            run.end_of_generation(improved, champion, val_score)
            ... breed ...
        run.finish()

    resume: a checkpoint directory, a state file, or an already loaded state.
    point: which state of a directory — 'latest', 'best' or a generation.
    extra_generations: continue past the saved run's end. The mutation
        intensity schedule keeps the ORIGINAL length (see schedule), so the
        extra generations run at the floor intensity: continuing a converged
        run is fine-tuning, not a restart with wild mutations.
    allow_changes: signature keys that may differ from the saved run (e.g.
        ('data',) to continue the search on a regrown dataset).
    """

    def __init__(self, *, phase: str, basis, signature: dict, generations: int,
                 checkpoint_dir: str | None = None, resume=None,
                 point='latest', extra_generations: int = 0,
                 allow_changes: tuple = (), checkpoint_every: int = 10,
                 snapshot_every: int = 50, keep_best: int = 5,
                 hof_size: int = 20, verbose: bool = False):
        self.phase = phase
        self.signature = dict(signature, phase=phase, basis=basis_spec(basis))
        self.restored = None
        self.start_gen = 0
        self.generations = int(generations)
        self.schedule = int(generations)
        self.changed: list[str] = []
        self.resumed_from: str | None = None

        if resume is not None:
            if isinstance(resume, dict):
                state = resume
            else:
                path = resolve_point(resume, point)
                state = read_json(path)
                self.resumed_from = path
            if state.get('phase') != phase:
                raise SignatureMismatch(
                    f"checkpoint is from phase {state.get('phase')!r}, "
                    f"not {phase!r}")
            self.changed = check_signature(state.get('signature', {}),
                                           self.signature,
                                           allow=tuple(allow_changes))
            self.restored = restore(state, basis)
            self.start_gen = self.restored['generation']
            self.schedule = self.restored['schedule']
            self.generations = (self.restored['generations']
                                + int(extra_generations))
            if extra_generations > 0 and 'patience_counter' in self.restored['counters']:
                # A run that ended on patience would stop again at once.
                self.restored['counters']['patience_counter'] = 0
            if verbose:
                src = self.resumed_from or 'given state'
                print(f"  [{phase}] resuming at generation {self.start_gen} "
                      f"of {self.generations} from {src}"
                      + (f" (changed: {', '.join(self.changed)})"
                         if self.changed else ""))

        self.ckpt = None
        if checkpoint_dir:
            self.ckpt = RunCheckpointer(
                checkpoint_dir, phase=phase, signature=self.signature,
                generations=self.generations, schedule=self.schedule,
                every=checkpoint_every,
                snapshot_every=snapshot_every, keep_best=keep_best,
                hof_size=hof_size,
                resume_gen=self.start_gen if self.restored is not None else None)
        self._cap = None
        self._gen = self.start_gen

    def intensity(self, gen: int, *, floor: float, span: float) -> float:
        """The usual wild-to-fine schedule, over the original run length."""
        return max(floor, 1.0 - gen / (self.schedule * span))

    def top(self, gen: int, population, *, counters: dict, rng=None,
            archive=None, extra: dict | None = None) -> None:
        self._gen = gen
        if self.ckpt is not None:
            self._cap = capture(gen, population, counters=counters, rng=rng,
                                archive=archive, extra=extra)

    def end_of_generation(self, improved: bool, champion=None,
                          val_score: float | None = None, **info) -> None:
        if self.ckpt is None or self._cap is None:
            return
        if champion is not None and val_score is not None:
            self.ckpt.record(champion, val_score, self._gen, **info)
        self.ckpt.on_generation(self._cap, improved=improved)

    def finish(self, population=None, *, counters: dict | None = None,
               rng=None, archive=None, next_gen: int | None = None) -> None:
        """
        Save where the run ended.

        Pass the population/counters as they stand at the top of next_gen when
        the loop ran to the end; with nothing passed, the last top-of-generation
        capture is written (a stop or early stop: resuming re-runs that
        generation, which replays exactly).
        """
        if self.ckpt is None:
            return
        if population is not None:
            cap = capture(next_gen if next_gen is not None else self._gen + 1,
                          population, counters=counters or {}, rng=rng,
                          archive=archive)
        else:
            cap = self._cap
        if cap is not None:
            self.ckpt.finish(cap)
