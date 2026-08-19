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

EML_POPULATION = 500    # EML tree population size
EML_GENERATIONS = 500   # maximum number of generations
EML_ELITISM = 10         # elite individuals carried over unchanged
EML_MAX_DEPTH = 5       # maximum depth at initialization
EML_PATIENCE = 100       # generations without improvement -> stop
EML_DATASET_EPISODES = 500  # episodes for dataset collection
EML_EPSILON = 1e-10     # guard against ln(0)

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
