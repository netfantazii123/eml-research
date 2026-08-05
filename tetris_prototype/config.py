"""
Tetris AI — configuration.
All project hyperparameters in one place (modeled on
flappy_bird_with_ai/config.py).

Three pipeline phases — three blocks of constants:
  1. Environment (TetrisEnv)
  2. Oracle (CNN + PPO)
  3. EML distillation
"""

# ── Environment constants (TetrisEnv) ────────────────────────────────────────

BOARD_W = 10            # board width (columns)
BOARD_H = 20            # visible board height (rows)
BOARD_BUFFER = 4        # top buffer for piece spawning (hidden rows)
BOARD_H_TOTAL = BOARD_H + BOARD_BUFFER   # full board array height = 24

GRAVITY_TICKS = 20      # frames between one-cell piece drops
MAX_EPISODE_FRAMES = 10000   # episode length limit (anti-stall)

N_ACTIONS = 6           # NO-OP, left, right, rotate CW, soft drop, hard drop
N_PIECES = 7            # I, O, T, S, Z, J, L
N_ROTATIONS = 4         # rotations per piece

# ── Reward shaping ───────────────────────────────────────────────────────────

REWARD_SURVIVE = 0.01       # per surviving step
REWARD_GAME_OVER = -1.0     # for a game over
# Line clear bonus = REWARD_LINE_BASE * (cleared_lines ** 2)
#   1 line -> 1, 2 -> 4, 3 -> 9, 4 (tetris) -> 16. This encourages tetrises.
REWARD_LINE_BASE = 1.0
# Mild penalty for stack growth: REWARD_HEIGHT_PENALTY * delta(aggregate_height).
#   0.0 = off. Enable with care — the policy is easy to "hack" with it.
REWARD_HEIGHT_PENALTY = 0.0

# ── Features for EML (features.py) ───────────────────────────────────────────

# Feature names (order = var_idx in the EML tree). 28 features.
FEATURE_NAMES = [
    'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',  # column heights
    'holes', 'bumpiness', 'agg_h', 'max_h', 'wells',
    'row_trans', 'col_trans',
    'piece_I', 'piece_O', 'piece_T', 'piece_S', 'piece_Z', 'piece_J', 'piece_L',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
]
N_FEATURES = len(FEATURE_NAMES)   # 28

# ── Oracle: CNN + PPO ────────────────────────────────────────────────────────

PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.01
PPO_N_ENVS = 16             # parallel environments
PPO_ROLLOUT = 2048          # steps per env per rollout collection
PPO_EPOCHS = 10             # SGD epochs over a collected rollout
PPO_BATCH_SIZE = 256
PPO_TOTAL_STEPS = 5_000_000  # starting budget; increase on stagnation
PPO_TARGET_SCORE = 20       # avg lines per episode -> oracle readiness criterion

# ── EML distillation ─────────────────────────────────────────────────────────

EML_POPULATION = 800        # EML tree population size
EML_GENERATIONS = 1000      # maximum generations
EML_ELITISM = 20            # elite individuals carried over unchanged
EML_MAX_DEPTH = 7           # maximum tree depth
EML_PATIENCE = 150          # generations without improvement -> stop
EML_DATASET_EPISODES = 1000  # oracle episodes for dataset collection
EML_EPSILON = 1e-10         # guard against ln(0)
EML_PHASE_SWITCH = 0.15     # fraction of generations for the DATA phase (warm-up)

DEPTH_PENALTIES = {
    'weak':   0.01,
    'medium': 0.05,
    'strong': 0.15,
}

# ── Directories ──────────────────────────────────────────────────────────────

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
