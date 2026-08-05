"""
data.py — loading an image dataset from class folders and stratified splitting.

DATASET LAYOUT — one subdirectory per class:

    <any folder>/
        BEACH/*.bmp      MOTO/*.png      PEOPLE/*.jpg      ROADS/...

Class names, their count and their order come from the file system rather than
from a constant: the user selects the folder. The order is alphabetical, because
the class index enters both the oracle weights and the branch C threshold table
and cannot depend on the order in which the OS returned the directory entries.

BMP is read by an OWN numpy parser (imaging.read_bmp), without Pillow/OpenCV.
The reference dataset consists of 24-bit uncompressed BMP files with a
BITMAPINFOHEADER (verified against the headers of all 5049 files); for that
format the parser takes 30 lines, and the project targets a microcontroller
where heavy dependencies are undesirable at any stage. Other formats and
arbitrary sizes are supported in imaging.py.

The first call to load_dataset() reads the files and stores the decoded array in
a cache (~62 MB uint8 on the reference dataset). Subsequent loads are instant.
"""

import os

import numpy as np

import config
import imaging
from imaging import IMAGE_EXTS, read_bmp, read_image     # noqa: F401 (re-export)


# ── Directory scanning ────────────────────────────────────────────────────────

def _image_files(directory: str) -> list[str]:
    """Sorted image file names in a folder (non-recursive)."""
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return sorted(n for n in names
                  if n.lower().endswith(IMAGE_EXTS)
                  and os.path.isfile(os.path.join(directory, n)))


def scan_classes(root: str) -> list[str]:
    """
    Class names = subdirectories of root containing at least one image.

    Empty subdirectories are skipped silently: a class without a single example
    would break both stratification and balanced accuracy (division by zero in
    recall), and most often it is just a stray folder such as
    `.ipynb_checkpoints`.
    """
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if name.startswith('.'):
            continue
        if _image_files(os.path.join(root, name)):
            out.append(name)
    return out


def describe_root(root: str) -> dict:
    """
    What the folder contains — before it is selected.

    Returns the class names, per-class file counts, the extensions encountered,
    and the sizes of the first images of each class. Used by the GUI: showing
    this BEFORE training starts is cheaper than explaining afterwards why the
    model trained on three classes instead of four.
    """
    classes = scan_classes(root)
    counts, exts, sizes, total_bytes = {}, set(), set(), 0
    for class_name in classes:
        class_dir = os.path.join(root, class_name)
        files = _image_files(class_dir)
        counts[class_name] = len(files)
        for name in files:
            exts.add(os.path.splitext(name)[1].lower())
            total_bytes += os.path.getsize(os.path.join(class_dir, name))
        for name in files[:1]:
            try:
                img = imaging.read_image(os.path.join(class_dir, name))
                sizes.add(f"{img.shape[1]}x{img.shape[0]}")
            except (OSError, ValueError, RuntimeError):
                sizes.add('?')
    return {'root': root, 'classes': classes, 'counts': counts,
            'n_files': sum(counts.values()), 'total_bytes': total_bytes,
            'exts': sorted(exts), 'sizes': sorted(sizes)}


# ── Dataset loading ───────────────────────────────────────────────────────────

def dataset_fingerprint() -> str:
    """
    Cheap fingerprint of the dataset contents: classes, file count, total size.

    Fully hashing 83 MB on every run is pointlessly expensive, while these three
    quantities are enough to detect "images were added to or removed from the
    folder" or "a different class set was selected". Without a fingerprint the
    cache is NEVER invalidated: after switching datasets the user would silently
    keep training on the old one.

    The class names are a mandatory part of the fingerprint: when switching to a
    folder with the same file count (for example a renamed copy) the size and
    count would match while the labels would mean something entirely different.

    An empty string = the fingerprint could not be computed (no class folders).
    This is not an error: that is what a machine holding only the cache looks
    like, and then the cache must be accepted as is rather than rebuilt from
    files that do not exist.
    """
    n_files, total_bytes = 0, 0
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            return ''
        for name in _image_files(class_dir):
            n_files += 1
            total_bytes += os.path.getsize(os.path.join(class_dir, name))
    if n_files == 0:
        return ''
    return f"{'|'.join(config.CLASS_NAMES)}:{n_files}:{total_bytes}"


def load_dataset(use_cache: bool = True, verbose: bool = True):
    """
    Load the whole dataset.

    Returns:
        images: (N, IMG_SIZE, IMG_SIZE, 3) uint8
        labels: (N,) int64 — index into config.CLASS_NAMES
        paths:  (N,) list of relative paths (for error reports)

    An image of a different size is center-cropped and scaled rather than
    stretched: stretching would change the texture scale that coherence,
    block_std and other features rely on.
    """
    cache = config.CACHE_PATH
    if use_cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        current = dataset_fingerprint()
        cached = str(d['fingerprint']) if 'fingerprint' in d.files else None
        # An empty `current` = the source images are not on this machine, so
        # there is nothing to recompute from; an old cache without a fingerprint
        # is accepted, with a warning.
        if not current or cached == current:
            if verbose:
                print(f"[data] cache: {cache}")
                if cached is None:
                    print("[data] cache has no fingerprint (created by an old "
                          "version) — it will be updated on the next rebuild")
            return d['images'], d['labels'], list(d['paths'])
        if verbose:
            print(f"[data] dataset changed ({cached} -> {current}) — "
                  f"re-reading the images")

    images, labels, paths, resized = [], [], [], 0
    for class_idx, class_name in enumerate(config.CLASS_NAMES):
        class_dir = os.path.join(config.DATASET_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"no class folder {class_dir}. Expected the layout "
                f"{config.DATASET_DIR}/<CLASS NAME>/*{'|*'.join(IMAGE_EXTS)}")
        files = _image_files(class_dir)
        if verbose:
            print(f"[data] {class_name:>12}: {len(files)} files")
        for name in files:
            img = imaging.read_image(os.path.join(class_dir, name))
            if img.shape != (config.IMG_SIZE, config.IMG_SIZE, 3):
                img = imaging.fit_to_size(img, config.IMG_SIZE)
                resized += 1
            images.append(img)
            labels.append(class_idx)
            paths.append(f"{class_name}/{name}")

    if not images:
        raise FileNotFoundError(
            f"no images of classes {', '.join(config.CLASS_NAMES)} found "
            f"in {config.DATASET_DIR}")

    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int64)

    os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
    np.savez_compressed(cache, images=images, labels=labels,
                        paths=np.array(paths, dtype=object),
                        fingerprint=dataset_fingerprint())
    if verbose:
        if resized:
            print(f"[data] converted to {config.IMG_SIZE}x{config.IMG_SIZE}: "
                  f"{resized} of {len(images)} (center crop + scale)")
        print(f"[data] loaded {len(images)} images -> cache {cache}")
    return images, labels, paths


# ── Splits ────────────────────────────────────────────────────────────────────

def stratified_split(labels: np.ndarray, seed: int | None = None):
    """
    Stratified train/val/test split: class proportions are preserved in every
    part. Returns three index arrays.

    Stratification is mandatory here: BEACH is three times rarer than ROADS, and
    under a random split the rare class could almost vanish from validation and
    skew the metric.
    """
    if seed is None:
        seed = config.SPLIT_SEED
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for class_idx in range(config.N_CLASSES):
        idx = np.flatnonzero(labels == class_idx)
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * config.SPLIT_TRAIN))
        n_val = int(round(n * config.SPLIT_VAL))
        train_idx.append(idx[:n_train])
        val_idx.append(idx[n_train:n_train + n_val])
        test_idx.append(idx[n_train + n_val:])

    out = tuple(np.sort(np.concatenate(parts))
                for parts in (train_idx, val_idx, test_idx))
    return out


def class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Class weights 1/freq, normalized to a mean of 1.

    Needed because of the 3.4:1 imbalance — without them ROADS and PEOPLE
    dominate the loss.
    """
    counts = np.bincount(labels, minlength=config.N_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (config.N_CLASSES * counts)
    return (w / w.mean()).astype(np.float32)


def describe(labels: np.ndarray, name: str = 'dataset') -> str:
    """A line like 'train: 3534 (BEACH 428, MOTO 572, ...)' for logs."""
    counts = np.bincount(labels, minlength=config.N_CLASSES)
    parts = ', '.join(f"{c} {counts[i]}" for i, c in enumerate(config.CLASS_NAMES))
    return f"{name}: {len(labels)} ({parts})"


if __name__ == '__main__':
    # As in the other standalone entry points, apply the selected dataset:
    # otherwise `python data.py` reports the contents of a folder OTHER than the
    # one the rest of the project works with.
    config.load_dataset_choice(verbose=True)
    images, labels, paths = load_dataset()
    tr, va, te = stratified_split(labels)
    print(describe(labels, 'total'))
    print(describe(labels[tr], 'train'))
    print(describe(labels[va], 'val  '))
    print(describe(labels[te], 'test '))
    print('class weights:', dict(zip(config.CLASS_NAMES,
                                     class_weights(labels[tr]).round(3))))
