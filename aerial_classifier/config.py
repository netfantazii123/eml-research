"""
aerial_classifier — configuration.
All project hyperparameters in one place (modeled on tetris/config.py).

Task: classification of 64x64 aerial images into 4 classes. Unlike stages 1-2
(Flappy, Tetris) there is no environment and no policy: the oracle solves a
supervised classification task, and the EML formulas approximate its decision
function.

Three pipeline phases — three blocks of constants:
  1. Data (BMP loading, splits, features)
  2. Oracle (MLP over features / CNN over pixels)
  3. EML distillation

THREE BRANCHES (see README):
  A (main)      — the oracle consumes 32 engineered features, EML consumes THE
                  SAME features, N_CLASSES formulas, decision = argmax.
                  Nothing of the neural network remains on the microcontroller.
  B (ablation)  — a CNN oracle consumes pixels, EML consumes its bottleneck
                  embedding. Upper bound on quality; convolutions stay on the MCU.
  C (single)    — the same features as in A, but ONE formula for all classes:
                  a scalar plus a threshold table fitted to it. Removes the scale
                  comparability problem that forces branch A to calibrate
                  (see the "Branch C" block below).
"""

import os

# ── Data ─────────────────────────────────────────────────────────────────────
#
# NOTE: DATASET_DIR, CLASS_NAMES, N_CLASSES and the cache paths are not
# constants. The user selects the class folder (in the GUI or via --dataset),
# and use_dataset() rewrites all of them at once. The values below are the
# default for when no choice has been made: the aerial dataset next to the
# project.
#
# Hence the rule for the rest of the code: access them through `config.X` rather
# than `from config import X` — otherwise the module captures the class names at
# import time and keeps using the old ones after a dataset switch.

DATASET_DIR = 'dataset'
CLASS_NAMES = ['BEACH', 'MOTO', 'PEOPLE', 'ROADS']
N_CLASSES = len(CLASS_NAMES)

IMG_SIZE = 64               # model input; images of other sizes are cropped
IMG_CHANNELS = 3

# Cache of the decoded dataset: files are read once, then loaded from .npz.
# It lives next to the dataset itself (easier to find and delete); if that
# folder is not writable, it moves to the project CACHE_DIR.
CACHE_DIR = 'cache'
CACHE_PATH = 'dataset/cache.npz'
FEATURES_CACHE_PATH = 'dataset/features.npz'

# The selected dataset is remembered here so that the CLI and the GUI work with
# the same class set rather than each with its own.
DATASET_CHOICE_PATH = 'models/dataset.json'

# Stratified split (fractions; test = remainder).
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_SEED = 1337           # the same seed as in the original TF scripts

# The dataset is imbalanced (ROADS 2057 : BEACH 612 = 3.4:1) — the oracle loss
# uses 1/freq class weights, otherwise the network learns the "everything is
# ROADS" prior.
USE_CLASS_WEIGHTS = True


# ── Features (branch A) ──────────────────────────────────────────────────────
#
# 32 features, all computed in 1-2 passes over the image with integer arithmetic
# plus a few divisions => realistic for an STM32.
# The ORDER MATTERS: var_idx in the EML tree indexes exactly this list.

FEATURE_NAMES = [
    # -- Color: channel and opponent axis moments (14) --
    'mean_r', 'mean_g', 'mean_b',
    'std_r', 'std_g', 'std_b',
    'mean_gray', 'std_gray',
    'mean_rg', 'mean_yb',          # opponency (R-G) and (R+G-2B)
    'std_rg', 'std_yb',
    'mean_sat', 'std_sat',         # saturation (max-min)/max

    # -- Texture and edges (8) --
    'grad_x', 'grad_y',            # mean |dI/dx|, |dI/dy|
    'grad_mag', 'grad_std',
    'edge_density',                # fraction of pixels with a strong gradient
    'anisotropy',                  # (gx-gy)/(gx+gy): texture elongation
    'coherence',                   # structure tensor: linearity (roads!)
    'entropy',                     # entropy of the luminance histogram

    # -- Local roughness over 8x8 blocks (2) --
    'block_std',                   # mean spread WITHIN blocks
    'block_spread',                # spread of block MEANS

    # -- Spatial layout (8) --
    'center_gray_diff',            # central 32x32 minus periphery
    'center_sat_diff',
    'center_edge_diff',            # object in the center => more edges there
    'radial_slope',                # correlation of luminance with radius
    'block16_spread',              # spread of 16x16 block means (coarse scale)
    'quad_spread',                 # luminance spread across the 4 quadrants
    'dark_frac', 'bright_frac',    # fractions of dark and bright pixels
]
N_FEATURES = len(FEATURE_NAMES)          # 32

EDGE_THRESHOLD = 24.0       # |grad| threshold for edge_density (0..255 scale)
DARK_THRESHOLD = 60.0       # "dark pixel" threshold (0..255)
BRIGHT_THRESHOLD = 190.0    # "bright pixel" threshold (0..255)
ENTROPY_BINS = 16           # histogram bins for entropy
BLOCK_SIZE = 8              # block size for block_std / block_spread


# ── Oracle ───────────────────────────────────────────────────────────────────

# Branch A: MLP over 32 features.
MLP_HIDDEN = [128, 64]
MLP_DROPOUT = 0.2

# Branch B: a small CNN over pixels with a narrow bottleneck embedding.
# The embedding is deliberately narrow: the EML formula must receive a vector of
# the same order of dimensionality as the branch A features, otherwise the
# comparison is unfair.
CNN_CHANNELS = [32, 64, 128]
CNN_EMBED_DIM = 32          # = N_FEATURES, so the branches are comparable
CNN_DROPOUT = 0.3

TRAIN_EPOCHS = 120
TRAIN_BATCH_SIZE = 64
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_PATIENCE = 25         # epochs without val accuracy growth -> stop
TRAIN_LR_DECAY = 0.5        # LR multiplier on a plateau
TRAIN_LR_PATIENCE = 8

# Augmentation (branch B only, train only). In the original TF scripts the
# augmentation was wired around the backbone and had no effect.
AUG_FLIP = True             # an aerial frame has no canonical orientation
AUG_ROT90 = True
AUG_BRIGHTNESS = 0.15       # +-15% brightness


# ── EML distillation ─────────────────────────────────────────────────────────
#
# Classification => NOT one formula but N_CLASSES formulas: f_c(features) ->
# scalar, prediction = argmax_c f_c. This is a direct analogue of tetris (argmax
# over placements), with the argmax taken over classes instead.

EML_EPSILON = 1e-10         # guard against ln(0)
EML_POPULATION = 600
EML_GENERATIONS = 600
EML_ELITISM = 20
EML_MAX_DEPTH = 7
EML_PATIENCE = 120
EML_DATA_SUBSAMPLE = 4000   # train subsample for fitness evaluation (speed)
EML_VAR_BONUS = 0.03        # fitness bonus per unique variable used

# The ACC phase is the analogue of the Tetris GAME phase. The DATA phase trains
# the formulas by regression to the oracle logits (MSE), but what matters is not
# MSE but ACCURACY: only the argmax counts. The second phase therefore evolves
# the whole SET of 4 formulas together, with fitness = actual balanced accuracy
# on a TRAIN SUBSAMPLE (EML_ACC_SUBSAMPLE), never on val/test: evolving against
# val is the same kind of fitting as gradient training, after which val is no
# longer an independent estimate.
# The same logic as "the formula plays by itself".
EML_ACC_ENABLED = True
EML_ACC_POPULATION = 60
EML_ACC_GENERATIONS = 60
EML_ACC_ELITISM = 6
EML_ACC_PATIENCE = 15
EML_ACC_SUBSAMPLE = 3000    # train subsample for accuracy in the ACC phase

DEPTH_PENALTIES = {
    'weak':   0.01,
    'medium': 0.05,
    'strong': 0.15,
}
DEPTH_PENALTY_DEFAULT = 'medium'

# Weight of the node count in the DATA phase penalty, in units of depth.
#
# A depth-only penalty (as in Tetris) does NOT restrain growth: depth is bounded
# by EML_MAX_DEPTH anyway, and a depth-7 tree may have 8 nodes or 127 — they pay
# the same. Measurement: at 600 generations the formulas grew to 106 nodes and
# scored 0.5399 on test, whereas a 40-generation run gave 14 nodes and 0.5504,
# i.e. the extra nodes hurt rather than helped. Here a node costs
# EML_SIZE_PENALTY_WEIGHT units of depth, so a hundred nodes add as much to the
# penalty as 5 depth levels.
EML_SIZE_PENALTY_WEIGHT = 0.05

# Batch distillation: several runs with different seeds/parameters, the best ->
# models/best_eml.json, all -> runs/<date>_distill/.
EML_BATCH_VARIANTS = 5
EML_BATCH_SPREAD = 0.3


# ── Branch C: ONE formula for all classes ────────────────────────────────────
#
# Branches A and B keep one formula per class and take the argmax. The main
# defect of that scheme is described in the README: the formulas evolve
# independently, their scales differ, and without calibration (mu_c, sd_c) the
# winner is whichever formula emits larger numbers rather than the most
# appropriate one.
#
# Branch C removes the defect by construction: there is one formula, its output
# is a single number, and there is nothing to compare it against. The class is
# recovered from that number by a THRESHOLD TABLE:
#
#     f(x) -> v ;  class = labels[searchsorted(thresholds, v)]
#
# There are not N_CLASSES-1 thresholds but up to EMLC_MAX_INTERVALS-1, and
# interval labels MAY REPEAT. This is essential: otherwise the classes would
# have to line up in a linear order on the axis, and no order exists between
# BEACH, MOTO, PEOPLE and ROADS. Allowing a class to occupy several intervals
# gives the effect a periodic activation would target, without one: a sine of
# the EML tree output (range roughly [-10, +2.2e4]) puts thousands of periods
# across the data, the fitness landscape becomes discontinuous, and the GA
# wanders over it instead of climbing.
#
# The table itself is NOT FIXED IN ADVANCE. It is fitted to the formula output
# at every fitness evaluation, by exact dynamic programming maximizing balanced
# accuracy. The formula decides in what order and at what distance to place the
# classes; where it put them is an output of the run, not an assumption.

EMLC_MAX_INTERVALS = 12     # K: how many axis intervals the decoder may use

# Threshold grid resolution. A threshold can only fall on a bin boundary, so the
# grid is the precision with which the decoder can separate classes at all: with
# 64 bins and 3000 examples a bin holds ~47 values, and a class boundary inside
# a bin costs up to one and a half percent of recall. During evolution this is
# sufficient (fitness only has to compare trees against each other, and the miss
# is the same for all of them), while the final table is fitted on a far finer
# grid: it is computed once, and there is no reason to lose percent there.
EMLC_DECODER_BINS = 64
EMLC_FINAL_BINS = 512

EMLC_INTERVAL_PENALTY = 0.004   # penalty per interval beyond the class count

# Depth 9 against 7 in branch A is a condition of a fair comparison, not a
# concession: there N_CLASSES trees work (120-310 nodes per set in the branch A
# measurements), here a single one. At depth 7 we would be measuring "four times
# fewer nodes" rather than "one formula against four".
EMLC_MAX_DEPTH = 9
EMLC_POPULATION = 400
EMLC_GENERATIONS = 3000
EMLC_ELITISM = 8
EMLC_PATIENCE = 400         # generations without improvement -> stop (0 = never)
EMLC_SUBSAMPLE = 3000       # train subsample for fitness evaluation

# The per-node penalty is subtracted directly from balanced accuracy, so the
# scale reads off immediately: medium gives -0.05 per hundred nodes. The same
# quantity as in the branch A ACC phase (there it came out as
# depth_penalty * 0.01 * nodes).
EMLC_SIZE_PENALTIES = {
    'weak':   0.0002,
    'medium': 0.0005,
    'strong': 0.0015,
}
EMLC_SIZE_PENALTY_DEFAULT = 'medium'

# Long run. Island model: several independent populations occasionally
# exchanging their best individuals. The point is diversity rather than speed:
# the GA spread across seeds in branch A (0.525-0.623) exceeded the effect of
# any single change, i.e. one long evolution chain says little about the method.
EMLC_ISLANDS = 0            # 0 = one per worker process
EMLC_WORKERS = 0            # 0 = os.cpu_count(); 1 = no parallelism
EMLC_MIGRATE_EVERY = 25     # generations between exchanges
EMLC_MIGRANTS = 2           # how many best individuals go to the next island
EMLC_CHECKPOINT_EVERY = 20  # generations between state writes to disk


# ── Directories ──────────────────────────────────────────────────────────────

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
RUNS_DIR = 'runs'


# ── Dataset selection ────────────────────────────────────────────────────────
#
# Classes = subdirectories of the selected folder. Nothing is hard-coded: the
# names, their count and their order come from the file system (the order is
# alphabetical, so that the class index does not depend on the order in which
# the OS returned the directory entries).

def _cache_paths(root: str) -> tuple[str, str]:
    """
    Where to place the decoded image and feature caches for dataset `root`.

    Next to the dataset if that is writable: the cache weighs tens of megabytes,
    belongs to that particular folder, and is deleted together with it. If the
    folder is not writable (read-only disk, network share), the cache moves to
    the project CACHE_DIR under a name derived from the path.
    """
    if os.access(root, os.W_OK):
        return (os.path.join(root, 'cache.npz'),
                os.path.join(root, 'features.npz'))

    import hashlib
    digest = hashlib.md5(os.path.abspath(root).encode('utf-8')).hexdigest()[:8]
    slug = f"{os.path.basename(os.path.abspath(root)) or 'dataset'}_{digest}"
    return (os.path.join(CACHE_DIR, slug, 'cache.npz'),
            os.path.join(CACHE_DIR, slug, 'features.npz'))


def use_dataset(root: str, class_names: list[str] | None = None) -> dict:
    """
    Switch the project to another class folder.

    class_names=None — determine them from the subdirectories (the usual case).
    An explicit list is needed when some subdirectories must be excluded.

    Changes DATASET_DIR, CLASS_NAMES, N_CLASSES and the cache paths. Returns a
    dict describing the result, which can be shown to the user before they start
    training on an unexpectedly different class set.
    """
    global DATASET_DIR, CLASS_NAMES, N_CLASSES, CACHE_PATH, FEATURES_CACHE_PATH

    root = os.path.normpath(root)
    if class_names is None:
        import data
        class_names = data.scan_classes(root)
    if not class_names:
        raise ValueError(
            f"no subdirectories with images in {root}. Expected the layout "
            f"{root}/<CLASS NAME>/*.bmp|png|jpg")

    DATASET_DIR = root
    CLASS_NAMES = list(class_names)
    N_CLASSES = len(CLASS_NAMES)
    CACHE_PATH, FEATURES_CACHE_PATH = _cache_paths(root)
    return {'root': DATASET_DIR, 'classes': CLASS_NAMES,
            'n_classes': N_CLASSES, 'cache': CACHE_PATH}


def save_dataset_choice() -> str:
    """Remember the current choice so the CLI and the GUI see one dataset."""
    import json
    os.makedirs(os.path.dirname(DATASET_CHOICE_PATH) or '.', exist_ok=True)
    with open(DATASET_CHOICE_PATH, 'w', encoding='utf-8') as f:
        json.dump({'root': DATASET_DIR, 'classes': CLASS_NAMES}, f,
                  indent=2, ensure_ascii=False)
    return DATASET_CHOICE_PATH


def load_dataset_choice(verbose: bool = False) -> bool:
    """
    Apply the remembered choice. True if there was one and it applied.

    Called EXPLICITLY from main.py and gui.py rather than at config import time:
    the tests must not depend on which folder the user selected last time. A
    missing folder is not a startup error: we fall back to the default and say
    so, otherwise the program would stop starting because of an unplugged drive.
    """
    import json
    if not os.path.exists(DATASET_CHOICE_PATH):
        return False
    try:
        with open(DATASET_CHOICE_PATH, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        root = saved['root']
        if not os.path.isdir(root):
            if verbose:
                print(f"[config] remembered dataset {root} not found — "
                      f"staying on {DATASET_DIR}")
            return False
        use_dataset(root, saved.get('classes') or None)
        if verbose:
            print(f"[config] dataset: {DATASET_DIR} "
                  f"({N_CLASSES} classes: {', '.join(CLASS_NAMES)})")
        return True
    except (OSError, ValueError, KeyError) as exc:
        if verbose:
            print(f"[config] could not read {DATASET_CHOICE_PATH}: {exc}")
        return False


def dataset_summary() -> str:
    """One-line description of the current dataset — for logs and window titles."""
    return f"{DATASET_DIR} — {N_CLASSES} cls.: {', '.join(CLASS_NAMES)}"
