"""
storage.py — saving/loading oracles (PyTorch) and EML formulas (JSON).

ACTIVE artifacts (used by export, verify-c, bench and the test tabs):
    models/oracle_A.pt        FeatureMLP
    models/oracle_B.pt        PixelCNN
    models/best_eml_A.json    4 branch A formulas
    models/best_eml_B.json    4 branch B formulas

LIBRARY — models/library/: named copies of every run.

    models/library/oracle_A__20260724-1230_bal0.908.pt
    models/library/eml_A__medium-seed4.json

Why the library exists. Previously every training run overwrote the single slot,
and two models could only be compared from memory. Runs cost from 14 seconds to
an hour, and the GA spread across seeds exceeds the difference between changes —
so a choice among several saved sets is a condition of meaningful comparison
rather than a convenience.

The library keeps no index file: the list is assembled by scanning the
directory, and all characteristics are read from the meta of the artifacts
themselves. An index would have to be repaired every time a file was deleted
outside the program.

The best model is determined BY VAL — the same metric used for selection within
a run. Test is never touched for selection: it is inspected once, at the very
end.

Formulas are stored as JSON rather than pickle deliberately: this is the final
artifact of the project, and it must be readable by eye, diffable in git, and
parseable by any other language when porting to a microcontroller.
"""

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field

import torch

import config


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)


def oracle_path(branch: str) -> str:
    return os.path.join(config.MODELS_DIR, f'oracle_{branch.upper()}.pt')


def formulas_path(branch: str) -> str:
    return os.path.join(config.MODELS_DIR, f'best_eml_{branch.upper()}.json')


# ── Oracle ───────────────────────────────────────────────────────────────────

def save_oracle(model, branch: str = 'A', path: str | None = None,
                meta: dict | None = None) -> str:
    """Save the oracle state_dict + training metadata."""
    path = path or oracle_path(branch)
    _ensure_dir(path)
    meta = dict(meta or {})
    meta.setdefault('class_names', list(config.CLASS_NAMES))
    torch.save({'state_dict': model.state_dict(), 'meta': meta}, path)
    return path


def load_oracle(model, branch: str = 'A', path: str | None = None,
                device=None, check_classes: bool = True) -> dict:
    """
    Load weights into an existing model instance. Returns meta.

    A class set mismatch is caught BEFORE load_state_dict: with a different
    number of classes torch would fail anyway, but with a message about tensor
    dimensions from which the actual cause ("a different dataset was selected")
    cannot be read at all. And with the same number of classes but different
    names, nothing would fail.
    """
    path = path or oracle_path(branch)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no oracle checkpoint {path}. First run: python main.py train "
            f"--branch {branch.upper()}")
    ckpt = torch.load(path, map_location=device or 'cpu', weights_only=False)
    meta = ckpt.get('meta', {})
    if check_classes:
        check_class_names(meta.get('class_names'),
                          f"branch {branch.upper()} oracle")
    model.load_state_dict(ckpt['state_dict'])
    return meta


def oracle_meta(branch: str = 'A') -> dict:
    """Read only the oracle metadata, without constructing a model."""
    path = oracle_path(branch)
    if not os.path.exists(path):
        return {}
    return torch.load(path, map_location='cpu',
                      weights_only=False).get('meta', {})


# ── EML formulas ─────────────────────────────────────────────────────────────

def save_formulas(trees: list, branch: str = 'A', path: str | None = None,
                  meta: dict | None = None,
                  input_names: list[str] | None = None,
                  calibration: dict | None = None,
                  decoder: dict | None = None) -> str:
    """
    Save EML trees as JSON.

    Args:
        trees: branches A and B — a list of EMLNode of length N_CLASSES in
            config.CLASS_NAMES order; branch C — a list with a single tree.
        input_names: names of the formula inputs — features (branches A and C)
            or embedding coordinate names (branch B).
        calibration: {'offsets': [...], 'scales': [...]} — the constants that
            align the formula outputs (branches A and B). Without them the
            argmax between formulas is meaningless, so they are part of the
            artifact rather than debug information.
        decoder: the branch C threshold table. Exactly the same role: without it
            the formula output is just a number.

    class_names are always written into the artifact. The class set now depends
    on the selected folder, and formulas trained on four classes mean something
    different with five — applying them silently would be worse than refusing.
    """
    path = path or formulas_path(branch)
    _ensure_dir(path)
    if input_names is None:
        input_names = (config.FEATURE_NAMES if branch.upper() != 'B'
                       else [f'e{i}' for i in range(config.CNN_EMBED_DIM)])
    if calibration is None and meta:
        calibration = meta.get('calibration')
    if decoder is None and meta:
        decoder = meta.get('decoder')
    payload = {
        'branch': branch.upper(),
        'class_names': list(config.CLASS_NAMES),
        'input_names': input_names,
        'calibration': calibration,
        'decoder': decoder,
        'trees': [t.to_dict() for t in trees],
        'formulas': [t.to_string() for t in trees],   # human-readable copy
        'meta': meta or {},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def check_class_names(payload_classes, what: str):
    """
    Verify the artifact's class set against the current dataset.

    A silent mismatch is the most expensive error after a folder change: the
    model runs, the metrics are computed, and everything means something else.
    Hence a refusal with a ready recipe rather than a warning.
    """
    if not payload_classes:
        return                                   # old artifact without a list
    if list(payload_classes) == list(config.CLASS_NAMES):
        return
    raise ValueError(
        f"{what} was trained on classes {list(payload_classes)}, while the "
        f"currently selected dataset has classes {list(config.CLASS_NAMES)}. "
        f"Either restore the previous dataset "
        f"(python main.py --dataset <folder> ...) or retrain.")


def formulas_meta(branch: str = 'A', path: str | None = None) -> dict:
    """
    Metadata of the saved formulas without parsing the trees ({} if no file).

    Needed to compare a new run against the stored one and avoid overwriting a
    better result with a worse one (see pipeline.save_best).
    """
    path = path or formulas_path(branch)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f).get('meta', {})
    except (json.JSONDecodeError, OSError):
        # A corrupted file counts as absent: let a new run replace it.
        return {}


def load_formulas(node_cls, branch: str = 'A', path: str | None = None):
    """
    Load EML trees. node_cls is the EMLNode class (with .from_dict).

    Returns: (trees, payload) — the trees and the whole parsed JSON.
    """
    path = path or formulas_path(branch)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no formula file {path}. First run: python main.py distill "
            f"--branch {branch.upper()}")
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    trees = [node_cls.from_dict(d) for d in payload['trees']]
    return trees, payload


def load_classifier(branch: str = 'A', path: str | None = None,
                    check_classes: bool = True):
    """
    Load a ready model together with its constants.

    Branches A and B — an EMLClassifier with calibration, branch C — an
    EMLSingle with a threshold table. A separate function, because forgetting to
    apply the constants after loading is the most likely mistake when using the
    artifact: the formulas would run and the answer would silently be garbage.
    """
    from eml_distiller import EMLClassifier, EMLNode

    trees, payload = load_formulas(EMLNode, branch=branch, path=path)
    if check_classes:
        check_class_names(payload.get('class_names'),
                          f"branch {branch.upper()} formulas")

    if payload.get('decoder'):
        from eml_single import EMLSingle, IntervalDecoder
        return EMLSingle(trees[0],
                         IntervalDecoder.from_dict(payload['decoder'])), payload

    calib = payload.get('calibration') or {}
    return EMLClassifier(trees, offsets=calib.get('offsets'),
                         scales=calib.get('scales')), payload


# ── Named model library ──────────────────────────────────────────────────────

KINDS = ('oracle', 'eml')
_EXT = {'oracle': '.pt', 'eml': '.json'}


def library_dir() -> str:
    return os.path.join(config.MODELS_DIR, 'library')


def slugify(name: str) -> str:
    """
    Model name -> safe file name.

    The user types the name by hand ("branch A, medium, seed 4"), and it ends up
    in a path. Passing it to the file system as is will not do: spaces, colons
    and slashes either break the path or create nested directories.
    """
    slug = re.sub(r'[^\w.-]+', '-', name.strip(), flags=re.UNICODE)
    slug = re.sub(r'-{2,}', '-', slug).strip('-.')
    return slug[:80] or 'model'


def auto_name(kind: str, branch: str, meta: dict) -> str:
    """
    Default name: timestamp plus the primary metric.

    The timestamp keeps runs distinct, the metric makes the list readable
    without expanding every item.
    """
    score = main_score(kind, meta)
    stamp = time.strftime('%Y%m%d-%H%M')
    return f"{stamp}_bal{score:.3f}" if score is not None else stamp


def main_score(kind: str, meta: dict) -> float | None:
    """
    The primary characteristic for comparing models — balanced accuracy on
    VALIDATION. The same metric used for selection within a run; test is never
    used for selection.
    """
    if kind == 'oracle':
        value = meta.get('val_bal_acc')
    else:
        value = meta.get('metrics', {}).get('val', {}).get('eml_bal_acc')
    return float(value) if value is not None else None


def _test_score(kind: str, meta: dict) -> float | None:
    if kind == 'oracle':
        value = meta.get('test_bal_acc')
    else:
        value = meta.get('metrics', {}).get('test', {}).get('eml_bal_acc')
    return float(value) if value is not None else None


@dataclass
class ModelEntry:
    """One model in the library together with its characteristics."""

    kind: str                    # 'oracle' | 'eml'
    branch: str
    name: str
    path: str
    meta: dict = field(repr=False, default_factory=dict)
    saved_at: float = 0.0
    is_active: bool = False
    is_best: bool = False

    @property
    def val_score(self) -> float | None:
        return main_score(self.kind, self.meta)

    @property
    def test_score(self) -> float | None:
        return _test_score(self.kind, self.meta)

    @property
    def size(self) -> int | None:
        """Complexity: parameters for an oracle, AST nodes for formulas."""
        return (self.meta.get('params') if self.kind == 'oracle'
                else self.meta.get('total_nodes'))

    @property
    def classes(self) -> list[str] | None:
        """Class set the model was trained on (None for old artifacts)."""
        value = self.meta.get('class_names')
        return list(value) if value else None

    @property
    def fits_dataset(self) -> bool:
        """
        Whether the model is applicable to the current dataset.

        With folder selection, models for different tasks sit side by side in
        the library. They must be distinguishable in the list: otherwise the
        difference surfaces only on an attempt to load, and it looks like a
        failure rather than "the model is simply from another dataset".
        """
        return self.classes is None or self.classes == list(config.CLASS_NAMES)

    def label(self) -> str:
        """Line for the dropdown: name, metric, complexity, marks."""
        marks = ('★' if self.is_best else ' ') + ('•' if self.is_active else ' ')
        val = f"{self.val_score:.4f}" if self.val_score is not None else "  ?   "
        unit = 'par.' if self.kind == 'oracle' else 'nodes'
        size = f"{self.size} {unit}" if self.size is not None else ''
        alien = '' if self.fits_dataset else '  [other dataset]'
        return f"{marks} {self.name:<28} val {val}  {size}{alien}"

    def details(self) -> str:
        """Multi-line summary for the panel."""
        lines = [f"name: {self.name}",
                 f"branch {self.branch}, "
                 f"{'oracle' if self.kind == 'oracle' else 'formulas'}"]
        if self.classes:
            lines.append(f"classes: {', '.join(self.classes)}")
        if not self.fits_dataset:
            lines.append("NOT FROM THE CURRENT DATASET")
        if self.is_active:
            lines.append("ACTIVE")
        if self.is_best:
            lines.append("best by val")
        if self.val_score is not None:
            lines.append(f"val bal:  {self.val_score:.4f}")
        if self.test_score is not None:
            lines.append(f"test bal: {self.test_score:.4f}")
        if self.kind == 'oracle':
            if self.meta.get('params'):
                lines.append(f"parameters: {self.meta['params']:,}")
            if self.meta.get('epochs_run'):
                lines.append(f"epochs: {self.meta['epochs_run']}")
            if self.meta.get('seconds'):
                lines.append(f"training: {self.meta['seconds']:.0f} s")
        else:
            if self.meta.get('total_nodes'):
                lines.append(f"AST nodes: {self.meta['total_nodes']}, "
                             f"depth {self.meta.get('max_depth', '?')}")
            if self.meta.get('n_inputs_used') is not None:
                lines.append(f"inputs: {self.meta['n_inputs_used']}"
                             f"/{self.meta.get('n_inputs_available', '?')}")
            fidelity = self.meta.get('metrics', {}).get('test', {}).get('fidelity')
            if fidelity is not None:
                lines.append(f"fidelity: {fidelity:.4f}")
            if self.meta.get('depth_penalty'):
                lines.append(f"penalty: {self.meta['depth_penalty']}, "
                             f"seed {self.meta.get('seed', '?')}")
        lines.append(time.strftime('%Y-%m-%d %H:%M',
                                   time.localtime(self.saved_at)))
        return '\n'.join(lines)


def library_path(kind: str, branch: str, name: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be 'oracle' or 'eml', got {kind!r}")
    return os.path.join(library_dir(),
                        f"{kind}_{branch.upper()}__{slugify(name)}{_EXT[kind]}")


def _parse_library_filename(filename: str):
    """'oracle_A__name.pt' -> ('oracle', 'A', 'name'); None if not ours."""
    stem, ext = os.path.splitext(filename)
    for kind in KINDS:
        if ext == _EXT[kind] and stem.startswith(f"{kind}_") and '__' in stem:
            head, name = stem.split('__', 1)
            branch = head[len(kind) + 1:]
            if branch and name:
                return kind, branch.upper(), name
    return None


def active_path(kind: str, branch: str) -> str:
    return oracle_path(branch) if kind == 'oracle' else formulas_path(branch)


def active_name(kind: str, branch: str) -> str | None:
    """
    Name of the library model currently sitting in the active slot.

    Determined from the meta of the active file itself rather than by comparing
    contents: the name is written there on save, so the active model is
    identifiable even after the file was copied by hand.
    """
    meta = (oracle_meta(branch) if kind == 'oracle' else formulas_meta(branch))
    name = meta.get('library_name')
    return str(name) if name else None


def _read_meta(kind: str, path: str) -> dict:
    try:
        if kind == 'oracle':
            return torch.load(path, map_location='cpu',
                              weights_only=False).get('meta', {})
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        meta = payload.get('meta', {})
        # For formulas the class list sits at the top level of the artifact
        # rather than in meta. Pull it in here so the model list can show which
        # class set the model was trained on: with dataset selection, models for
        # different tasks sit side by side in the library, and they must be
        # distinguishable in the list rather than by a refusal at load time.
        if 'class_names' not in meta and payload.get('class_names'):
            meta = {**meta, 'class_names': payload['class_names']}
        return meta
    except Exception:                                  # noqa: BLE001
        # A corrupted or foreign file must not break the whole list: show it
        # without characteristics and let the user decide to delete it.
        return {}


def list_models(kind: str | None = None, branch: str | None = None
                ) -> list[ModelEntry]:
    """
    Everything in the library, sorted by the primary metric (descending).

    Models without a metric go to the end: there is nothing to compare them to.
    """
    directory = library_dir()
    if not os.path.isdir(directory):
        return []

    entries: list[ModelEntry] = []
    for filename in sorted(os.listdir(directory)):
        parsed = _parse_library_filename(filename)
        if parsed is None:
            continue
        entry_kind, entry_branch, name = parsed
        if kind is not None and entry_kind != kind:
            continue
        if branch is not None and entry_branch != branch.upper():
            continue
        path = os.path.join(directory, filename)
        entries.append(ModelEntry(
            kind=entry_kind, branch=entry_branch, name=name, path=path,
            meta=_read_meta(entry_kind, path),
            saved_at=os.path.getmtime(path)))

    entries.sort(key=lambda e: (e.val_score is None,
                                -(e.val_score or 0.0), e.name))

    # The active model name is read once per (kind, branch) pair rather than per
    # entry: that is a checkpoint read from disk, and the active file is the
    # same for every entry in the group.
    active: dict[tuple[str, str], str | None] = {}
    for entry in entries:
        key = (entry.kind, entry.branch)
        if key not in active:
            active[key] = active_name(*key)
        entry.is_active = (entry.name == active[key])

    # The marks are computed within each (kind, branch) pair: the best branch A
    # oracle and the best branch B formula set are different entities with
    # nothing to compete over.
    #
    # Models from other datasets are excluded from the competition. Their
    # metrics are simply not comparable: on two classes balanced accuracy starts
    # at 0.5, and the star would go to them automatically whatever sat nearby.
    for entry_kind in KINDS:
        for entry_branch in {e.branch for e in entries if e.kind == entry_kind}:
            group = [e for e in entries
                     if e.kind == entry_kind and e.branch == entry_branch
                     and e.val_score is not None and e.fits_dataset]
            if group:
                max(group, key=lambda e: e.val_score).is_best = True
    return entries


def find_model(kind: str, branch: str, name: str) -> ModelEntry | None:
    for entry in list_models(kind, branch):
        if entry.name == name:
            return entry
    return None


def _unique_name(kind: str, branch: str, name: str) -> str:
    """A free name: a taken one is extended with -2, -3 and so on."""
    if not os.path.exists(library_path(kind, branch, name)):
        return name
    for suffix in range(2, 1000):
        candidate = f"{name}-{suffix}"
        if not os.path.exists(library_path(kind, branch, candidate)):
            return candidate
    raise RuntimeError(f"could not find a free name for {name!r}")


def reserve_name(kind: str, branch: str, name: str | None, meta: dict) -> str:
    """
    Reserve a free library name BEFORE saving.

    The order matters: the name is written inside the artifact (it is what
    identifies the active model), so it cannot be adjusted after writing — the
    active file and the library copy would then disagree on the name, and the
    active model would stop being recognized.
    """
    return _unique_name(kind, branch.upper(),
                        slugify(name) if name else auto_name(kind, branch, meta))


def add_to_library(kind: str, branch: str, name: str | None, meta: dict,
                   source_path: str | None = None,
                   overwrite: bool = False) -> ModelEntry:
    """
    Place a copy of the artifact into the library under a name.

    The already saved file is copied rather than re-serialized: this way the
    library copy is guaranteed to be byte-identical to what was measured, and a
    divergence between "the model that was evaluated" and "the model that was
    saved" is impossible in principle.

    A taken name is not overwritten silently (unless overwrite=True): runs cost
    from minutes to an hour, and losing another run's result to a name collision
    is worse than seeing "name-2" in the list.
    """
    branch = branch.upper()
    name = slugify(name) if name else auto_name(kind, branch, meta)
    if not overwrite:
        name = _unique_name(kind, branch, name)
    source_path = source_path or active_path(kind, branch)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"nothing to copy into the library: {source_path}")

    target = library_path(kind, branch, name)
    _ensure_dir(target)
    shutil.copyfile(source_path, target)
    return ModelEntry(kind=kind, branch=branch, name=name, path=target,
                      meta=meta, saved_at=os.path.getmtime(target))


def rename_model(entry: ModelEntry, new_name: str) -> ModelEntry:
    """
    Rename a model in the library.

    The name lives both in the file path and in meta (which is what identifies
    the active model), so it is changed in both places — otherwise a renamed
    model would stop being recognized as active.
    """
    name = _unique_name(entry.kind, entry.branch, slugify(new_name))
    target = library_path(entry.kind, entry.branch, name)
    os.rename(entry.path, target)

    meta = dict(entry.meta)
    meta['library_name'] = name
    _write_library_name(entry.kind, target, name)
    if entry.is_active:
        _write_library_name(entry.kind, active_path(entry.kind, entry.branch),
                            name)
    return ModelEntry(kind=entry.kind, branch=entry.branch, name=name,
                      path=target, meta=meta,
                      saved_at=os.path.getmtime(target), is_best=entry.is_best,
                      is_active=entry.is_active)


def _write_library_name(kind: str, path: str, name: str):
    """Rewrite library_name inside an already saved artifact."""
    if not os.path.exists(path):
        return
    if kind == 'oracle':
        payload = torch.load(path, map_location='cpu', weights_only=False)
        payload.setdefault('meta', {})['library_name'] = name
        torch.save(payload, path)
        return
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    payload.setdefault('meta', {})['library_name'] = name
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def activate(entry: ModelEntry) -> str:
    """
    Make a library model active: a copy goes into the main slot.

    A copy rather than a symbolic link: the artifact must remain an ordinary
    file that can be carried to another machine, and must not depend on link
    creation privileges on Windows.
    """
    target = active_path(entry.kind, entry.branch)
    _ensure_dir(target)
    shutil.copyfile(entry.path, target)
    return target


def delete_model(entry: ModelEntry) -> str:
    """Delete a model from the library. The active slot is not touched."""
    os.remove(entry.path)
    return entry.path
