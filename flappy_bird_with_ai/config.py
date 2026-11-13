"""
Flappy Bird AI — configuration.
All project constants in one place.
"""

# ── Environment constants ────────────────────────────────────────────────────

SCREEN_W = 400          # logical field width (px)
SCREEN_H = 600          # logical field height (px)
PIPE_GAP = 150          # distance between the upper and lower pipe (px)
PIPE_SPEED = 3          # pipe movement speed to the left (px/frame)
PIPE_FREQ = 90          # a new pipe spawns every N frames
GRAVITY = 0.5           # falling acceleration (px/frame^2)
FLAP_POWER = -8         # jump impulse (negative = upward)
BIRD_X = 60             # fixed X position of the bird
BIRD_RADIUS = 12        # bird hitbox radius (circle)
PIPE_WIDTH = 50         # pipe width (px)
STATE_SIZE = 4          # dimensionality of the state vector

# ── Environment difficulty profiles ──────────────────────────────────────────
# 'classic' is the original game (fixed 150 px gap). Its oracle never dies
# (1110 pipes in 100 000 frames), so every score is set by the frame cap, not by
# skill. 'ramp' makes the game end on its own: every pipe's gap is gap_shrink px
# narrower than the previous one, down to gap_min, so the score measures how
# narrow a gap a policy can still thread, and no frame cap is needed for it.
# Select with the environment variable EML_FLAPPY_ENV (read at import).
#   gap        — the first pipe's gap (px)
#   gap_shrink — px removed per spawned pipe
#   gap_min    — floor of the gap (px); the bird is 2*BIRD_RADIUS = 24 px
FLAPPY_PROFILES = {
    'classic': {'gap': PIPE_GAP, 'gap_shrink': 0.0, 'gap_min': PIPE_GAP},
    'ramp':    {'gap': PIPE_GAP, 'gap_shrink': 1.0, 'gap_min': 40},
}
import os as _os
FLAPPY_ENV = _os.environ.get('EML_FLAPPY_ENV', 'classic')
if FLAPPY_ENV not in FLAPPY_PROFILES:
    raise ValueError(f'EML_FLAPPY_ENV={FLAPPY_ENV!r}, expected one of '
                     f'{sorted(FLAPPY_PROFILES)}')

# ── Genetic algorithm constants (neuroevolution) ─────────────────────────────

GA_POPULATION = 1000    # population size (number of birds)
GA_MUTATION_RATE = 0.1  # mutation probability per weight
GA_MUTATION_STD = 0.3   # standard deviation of the mutation noise
GA_ELITISM = 5          # number of best individuals copied unchanged
GA_GENERATIONS = 50     # maximum number of generations
GA_MAX_FRAMES = 3000    # frame limit for a single game
GA_PATIENCE = 15        # generations without best_fitness improvement -> stop
GA_TARGET_SCORE = 0     # 0 = off; otherwise stop early at this score

# ── EML distillation constants ───────────────────────────────────────────────

EML_BASIS = 'eml'       # 'eml' or 'classic' (the control basis)
EML_POPULATION = 500    # EML tree population size
EML_GENERATIONS = 500   # maximum number of generations
EML_ELITISM = 10         # elite individuals carried over unchanged
EML_MAX_DEPTH = 5       # maximum depth at initialization
EML_PATIENCE = 100       # generations without improvement -> stop
EML_DATASET_EPISODES = 500  # episodes for dataset collection
EML_EPSILON = 1e-10     # guard against ln(0)
# exp() clamp. Both guards belong to THIS operator rather than to the EML
# operator of the literature, and the clamp dominates eps by orders of
# magnitude — it is the one worth a sensitivity sweep.
EML_CLAMP = 10.0

# ── Environment seed pools (disjoint) ────────────────────────────────────────
# Same discipline as tetris/config.py: fitness, selection and reporting never
# share pipe layouts. TRAIN rotates across generations with common random
# numbers inside a generation, VAL judges the champion, TEST is the reported
# number and is touched once. Dataset collection uses EML_DATA_SEED, disjoint
# from all three.
EML_SEED_POOL_TRAIN = list(range(1000, 1064))
EML_SEED_POOL_VAL = list(range(3000, 3032))
EML_SEED_POOL_TEST = list(range(5000, 5064))
EML_DATA_SEED = 7

EML_INGAME_GAMES = 3    # games per fitness evaluation (TRAIN pool)
EML_VAL_GAMES = 8       # games per hall-of-fame re-scoring (VAL pool)
EML_TEST_GAMES = 30     # games in the final holdout report (TEST pool)

DEPTH_PENALTIES = {
    'weak':   0.01,
    'medium': 0.05,
    'strong': 0.15,
}

# ── Directories ──────────────────────────────────────────────────────────────

MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# ── Frame caps per profile ───────────────────────────────────────────────────
# classic keeps the historical caps (GA/VAL 3000 frames = 32 pipes, TEST 5000
# = 55 pipes) so old numbers reproduce; that VAL and TEST disagree there is a
# known defect of the classic setup (9 of 10 seed-study seeds tie at VAL 32).
# ramp has no ceiling: one cap, far above any game (the ramp oracle dies after
# ~60 pipes, ~5500 frames), for training, VAL and TEST alike, and VAL uses the
# whole VAL pool so champion selection is not decided by 8 layouts.
EML_TEST_MAX_FRAMES = 5000
if FLAPPY_ENV == 'ramp':
    GA_MAX_FRAMES = 1_000_000
    EML_TEST_MAX_FRAMES = 1_000_000
    EML_VAL_GAMES = len(EML_SEED_POOL_VAL)
