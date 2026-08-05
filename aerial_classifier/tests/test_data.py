"""
Tests of the BMP parser and the splits.

The BMP parser is written by hand (data.read_bmp), and it is the most fragile
part of loading: an error in the row padding or in the row order does not fail
but silently corrupts the images — training then proceeds on garbage of
seemingly correct shape. The BMPs here are therefore assembled inside the test,
byte by byte, in every sub-format the parser promises to understand.
"""

import json
import os
import struct

import numpy as np
import pytest

import config
import data


# ── Assembling a BMP in the test ──────────────────────────────────────────────

def make_bmp(rgb: np.ndarray, top_down: bool = False, bpp: int = 24,
             compression: int = 0) -> bytes:
    """
    Assemble a 24-bit BMP from an (H, W, 3) uint8 RGB array.

    top_down=False — rows bottom-up (as in the dataset), True — top-down
    (negative height in the header).
    """
    height, width = rgb.shape[:2]
    row_bytes = width * 3
    padded = (row_bytes + 3) & ~3
    pad = b'\x00' * (padded - row_bytes)

    rows = [rgb[y][:, ::-1].tobytes() + pad          # RGB -> BGR
            for y in range(height)]
    if not top_down:
        rows = rows[::-1]
    pixels = b''.join(rows)

    offset = 14 + 40
    file_header = b'BM' + struct.pack('<IHHI', offset + len(pixels), 0, 0, offset)
    info_header = struct.pack('<IiiHHIIiiII', 40, width,
                              -height if top_down else height,
                              1, bpp, compression, len(pixels), 2835, 2835, 0, 0)
    return file_header + info_header + pixels


def write_bmp(tmp_path, name: str, rgb: np.ndarray, **kwargs) -> str:
    path = tmp_path / name
    path.write_bytes(make_bmp(rgb, **kwargs))
    return str(path)


# ── Parser ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('top_down', [False, True])
def test_read_bmp_roundtrip(tmp_path, top_down):
    """The pixels are read back exactly, in both row orders."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (7, 12, 3), dtype=np.uint8)
    got = data.read_bmp(write_bmp(tmp_path, 'a.bmp', rgb, top_down=top_down))
    assert got.shape == rgb.shape
    assert np.array_equal(got, rgb)


@pytest.mark.parametrize('width', [1, 2, 3, 4, 5, 6, 7, 8])
def test_read_bmp_row_padding(tmp_path, width):
    """
    A BMP row is padded to a multiple of 4 bytes.

    At a width not divisible by 4 there are filler bytes between rows. If they
    are not discarded, the image shears sideways with a growing offset — the
    most insidious parser breakage, because with it the first row still looks
    correct.
    """
    rng = np.random.default_rng(width)
    rgb = rng.integers(0, 256, (5, width, 3), dtype=np.uint8)
    got = data.read_bmp(write_bmp(tmp_path, f'w{width}.bmp', rgb))
    assert np.array_equal(got, rgb)


def test_read_bmp_channel_order(tmp_path):
    """BMP stores BGR; the parser must return RGB rather than swap the channels."""
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 200        # red
    rgb[..., 1] = 100        # green
    rgb[..., 2] = 50         # blue
    got = data.read_bmp(write_bmp(tmp_path, 'rgb.bmp', rgb))
    assert got[0, 0].tolist() == [200, 100, 50]


def test_read_bmp_rejects_foreign_format(tmp_path):
    """A foreign sub-format fails rather than silently corrupting the data."""
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        data.read_bmp(write_bmp(tmp_path, 'p8.bmp', rgb, bpp=8))
    with pytest.raises(ValueError):
        data.read_bmp(write_bmp(tmp_path, 'rle.bmp', rgb, compression=1))

    not_bmp = tmp_path / 'x.bmp'
    not_bmp.write_bytes(b'PNG\x00' + b'\x00' * 60)
    with pytest.raises(ValueError):
        data.read_bmp(str(not_bmp))


# ── Class folder ──────────────────────────────────────────────────────────────

@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """
    A temporary dataset folder; the global config settings are restored.

    Selecting a dataset changes config.DATASET_DIR, CLASS_NAMES, N_CLASSES and
    the cache paths right in the module. Without restoring them, one such test
    would spoil all the following ones — not by failing, but by quietly changing
    their number of classes.
    """
    import config as config_mod

    for attr in ('DATASET_DIR', 'CLASS_NAMES', 'N_CLASSES', 'CACHE_PATH',
                 'FEATURES_CACHE_PATH', 'DATASET_CHOICE_PATH'):
        monkeypatch.setattr(config_mod, attr, getattr(config_mod, attr))
    monkeypatch.setattr(config_mod, 'DATASET_CHOICE_PATH',
                        str(tmp_path / 'choice.json'))
    return tmp_path


def put_images(root, class_name: str, n: int, ext: str = '.bmp',
               size: int = 64, start: int = 0):
    """Drop n images of the given format and size into <root>/<class_name>."""
    from test_imaging import make_png

    class_dir = root / class_name
    class_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(abs(hash(class_name)) % 1000)
    for i in range(start, start + n):
        rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        blob = make_bmp(rgb) if ext == '.bmp' else make_png(rgb)
        (class_dir / f"{i:03d}{ext}").write_bytes(blob)
    return class_dir


def test_scan_classes_reads_subfolders(sandbox):
    """Classes = subdirectories with images, in alphabetical order."""
    put_images(sandbox, 'ROADS', 2)
    put_images(sandbox, 'BEACH', 3, ext='.png')
    assert data.scan_classes(str(sandbox)) == ['BEACH', 'ROADS']


def test_scan_classes_skips_empty_and_hidden(sandbox):
    """
    An empty subdirectory does not count as a class.

    A class without examples would break both stratification and recall
    (division by zero), and most often it is a stray service folder such as
    .ipynb_checkpoints.
    """
    put_images(sandbox, 'ROADS', 2)
    (sandbox / 'EMPTY').mkdir()
    (sandbox / '.hidden').mkdir()
    put_images(sandbox, '.hidden', 2)
    (sandbox / 'readme.txt').write_text('not a class', encoding='utf-8')
    assert data.scan_classes(str(sandbox)) == ['ROADS']


def test_use_dataset_switches_everything(sandbox):
    """Choosing a folder changes the class names, their count and the cache path."""
    import config as config_mod

    put_images(sandbox, 'ONE', 2)
    put_images(sandbox, 'TWO', 2)
    put_images(sandbox, 'THREE', 2)
    info = config_mod.use_dataset(str(sandbox))

    assert info['classes'] == ['ONE', 'THREE', 'TWO']
    assert config_mod.N_CLASSES == 3
    assert config_mod.CLASS_NAMES == ['ONE', 'THREE', 'TWO']
    # The cache must move with the dataset: otherwise the second dataset would
    # read the images of the first and silently train on the wrong data.
    assert str(sandbox) in config_mod.CACHE_PATH
    assert str(sandbox) in config_mod.FEATURES_CACHE_PATH


def test_load_dataset_labels_follow_folder_order(sandbox):
    """The label = the class index in alphabetical order of folder names."""
    import config as config_mod

    put_images(sandbox, 'AAA', 3)
    put_images(sandbox, 'BBB', 2)
    config_mod.use_dataset(str(sandbox))
    images, labels, paths = data.load_dataset(verbose=False)

    assert images.shape == (5, config_mod.IMG_SIZE, config_mod.IMG_SIZE, 3)
    assert labels.tolist() == [0, 0, 0, 1, 1]
    assert paths[0].startswith('AAA/') and paths[-1].startswith('BBB/')


def test_load_dataset_normalizes_mixed_sizes_and_formats(sandbox):
    """Different formats and sizes in one folder are converted to the model input."""
    import config as config_mod

    put_images(sandbox, 'MIX', 2, ext='.bmp', size=40)
    put_images(sandbox, 'MIX', 2, ext='.png', size=100, start=10)
    put_images(sandbox, 'OTHER', 2, size=64)
    config_mod.use_dataset(str(sandbox))
    images, labels, _ = data.load_dataset(verbose=False)

    assert images.shape == (6, config_mod.IMG_SIZE, config_mod.IMG_SIZE, 3)
    assert images.dtype == np.uint8
    assert labels.tolist() == [0, 0, 0, 0, 1, 1]


def test_cache_is_reused_and_invalidated(sandbox):
    """
    The cache is reused but stops being valid when the folder is edited.

    The cache was never invalidated at all before the fingerprint appeared:
    after adding images the user silently kept training on the old set.
    """
    import config as config_mod

    put_images(sandbox, 'AAA', 2)
    put_images(sandbox, 'BBB', 2)
    config_mod.use_dataset(str(sandbox))
    first, _, _ = data.load_dataset(verbose=False)
    assert os.path.exists(config_mod.CACHE_PATH)

    again, _, _ = data.load_dataset(verbose=False)
    assert np.array_equal(first, again)

    put_images(sandbox, 'BBB', 1, start=50)
    third, labels, _ = data.load_dataset(verbose=False)
    assert len(third) == 5


def test_fingerprint_includes_class_names(sandbox):
    """
    The fingerprint changes on a change of the CLASS SET, not only the file count.

    Otherwise switching to a folder with the same number and size of files (a
    renamed copy, a different labeling) would go through the old cache, and the
    labels would mean something entirely different.
    """
    import config as config_mod

    put_images(sandbox, 'AAA', 2)
    put_images(sandbox, 'BBB', 2)
    config_mod.use_dataset(str(sandbox))
    both = data.dataset_fingerprint()

    config_mod.use_dataset(str(sandbox), ['AAA'])
    assert data.dataset_fingerprint() != both


def test_dataset_choice_survives_restart(sandbox):
    """The selected dataset is remembered and applied on the next start."""
    import config as config_mod

    put_images(sandbox, 'ONE', 2)
    put_images(sandbox, 'TWO', 2)
    config_mod.use_dataset(str(sandbox))
    config_mod.save_dataset_choice()

    config_mod.use_dataset('dataset', ['X', 'Y', 'Z'])     # "another run"
    assert config_mod.load_dataset_choice() is True
    assert config_mod.CLASS_NAMES == ['ONE', 'TWO']


def test_missing_dataset_choice_does_not_break_startup(sandbox):
    """
    A missing folder must not break startup.

    The dataset may live on a removable drive; not finding it is a reason to
    stay on the default and say so, not to refuse to start.
    """
    import config as config_mod

    config_mod.use_dataset(str(sandbox), ['X'])
    config_mod.save_dataset_choice()
    config_mod.DATASET_DIR = 'dataset'
    config_mod.CLASS_NAMES = ['A', 'B']

    saved = json.loads((sandbox / 'choice.json').read_text(encoding='utf-8'))
    saved['root'] = str(sandbox / 'no-such-folder')
    (sandbox / 'choice.json').write_text(json.dumps(saved), encoding='utf-8')

    assert config_mod.load_dataset_choice() is False
    assert config_mod.CLASS_NAMES == ['A', 'B']


def test_describe_root_reports_contents(sandbox):
    """The folder summary is assembled BEFORE selection — the GUI has what to show."""
    put_images(sandbox, 'ONE', 3, ext='.bmp', size=64)
    put_images(sandbox, 'TWO', 2, ext='.png', size=32)
    info = data.describe_root(str(sandbox))

    assert info['classes'] == ['ONE', 'TWO']
    assert info['counts'] == {'ONE': 3, 'TWO': 2}
    assert info['n_files'] == 5
    assert info['exts'] == ['.bmp', '.png']
    assert set(info['sizes']) == {'64x64', '32x32'}


# ── Splits ────────────────────────────────────────────────────────────────────

def make_labels() -> np.ndarray:
    """Imbalanced labels, as in the real dataset (about 3.4:1)."""
    counts = [61, 92, 130, 206]
    return np.concatenate([np.full(n, c, dtype=np.int64)
                           for c, n in enumerate(counts)])


def test_stratified_split_is_a_partition():
    """The three parts do not overlap and together cover the dataset entirely."""
    labels = make_labels()
    train, val, test = data.stratified_split(labels, seed=1337)

    joined = np.concatenate([train, val, test])
    assert len(joined) == len(labels)
    assert len(np.unique(joined)) == len(labels)
    assert np.array_equal(np.sort(joined), np.arange(len(labels)))
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(a.tolist()) & set(b.tolist())


def test_stratified_split_is_deterministic():
    """One seed — one and the same split; otherwise metrics across runs differ."""
    labels = make_labels()
    first = data.stratified_split(labels, seed=7)
    second = data.stratified_split(labels, seed=7)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)

    other = data.stratified_split(labels, seed=8)
    assert not all(np.array_equal(a, b) for a, b in zip(first, other))


def test_stratified_split_keeps_class_proportions():
    """
    Class proportions are preserved in every part.

    That is what stratification is for: under a random split the rare class can
    almost vanish from validation and skew the metric.
    """
    labels = make_labels()
    train, val, test = data.stratified_split(labels, seed=1337)
    overall = np.bincount(labels, minlength=config.N_CLASSES) / len(labels)
    for part in (train, val, test):
        share = (np.bincount(labels[part], minlength=config.N_CLASSES)
                 / len(part))
        assert np.allclose(share, overall, atol=0.02)


def test_stratified_split_sizes_match_config():
    labels = make_labels()
    train, val, _ = data.stratified_split(labels, seed=1337)
    assert len(train) / len(labels) == pytest.approx(config.SPLIT_TRAIN, abs=0.01)
    assert len(val) / len(labels) == pytest.approx(config.SPLIT_VAL, abs=0.01)


def test_class_weights_compensate_imbalance():
    """The weight is inversely proportional to frequency, and the mean weight is 1."""
    labels = make_labels()
    w = data.class_weights(labels)
    assert len(w) == config.N_CLASSES
    assert w.mean() == pytest.approx(1.0, abs=1e-6)
    counts = np.bincount(labels, minlength=config.N_CLASSES)
    assert w[np.argmin(counts)] > w[np.argmax(counts)]
    # weight * frequency is the same for every class — that is the point of 1/freq.
    products = w * counts
    assert np.allclose(products, products[0], rtol=1e-6)
