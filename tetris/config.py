"""
Tetris AI — configuration.
All project hyperparameters in one place (modeled on
flappy_bird_with_ai/config.py).

Three pipeline phases — three blocks of constants:
  1. Environment (TetrisEnv)
  2. Oracle (CNN + PPO)
  3. EML distillation

ARCHITECTURE v2 — placement-based action space.
The oracle chooses a piece PLACEMENT (rotation × column = 40 actions with a
legality mask) rather than pressing buttons frame by frame. One environment step
= one placed piece. Frame-by-frame PPO (v1) did not train: over 4.2M steps it
averaged 1.45 lines — the line signal drowned in the hundreds of frames between
the action and the result.
"""

# ── Environment constants (TetrisEnv) ────────────────────────────────────────

BOARD_W = 10            # board width (columns)
BOARD_H = 20            # visible board height (rows)
BOARD_BUFFER = 4        # top buffer for piece spawning (hidden rows)
BOARD_H_TOTAL = BOARD_H + BOARD_BUFFER   # full board array height = 24

GRAVITY_TICKS = 20      # frames between drops (legacy frame mode only)
MAX_EPISODE_FRAMES = 10000   # episode length limit in frame mode (anti-stall)

N_ACTIONS = 6           # legacy frame actions: NO-OP, left, right, rot, soft, hard
N_PIECES = 7            # I, O, T, S, Z, J, L
N_ROTATIONS = 4         # rotations per piece

# ── Placement action space (the main one for the oracle) ─────────────────────
# Action = rot * BOARD_W + xi, where xi is the column of the leftmost occupied
# cell of the piece. Illegal combinations (past the wall, blocked spawn,
# duplicate O/I/S/Z rotations) are closed off by a mask.
N_PLACEMENTS = N_ROTATIONS * BOARD_W     # 40
MAX_EPISODE_PLACEMENTS = 1000            # placements per episode limit (anti-stall)

# ── Reward shaping (per placement) ───────────────────────────────────────────
#
# Every placement step ends with a piece lock, so the reward is dense:
#   + REWARD_PIECE                       — bonus for "placed a piece and survived"
#   + REWARD_LINE_BASE * n_cleared**2    — for cleared lines (tetris → 16)
#   - REWARD_W_HOLES * max(0, Δholes)    — penalty for new holes (the main enemy)
#   - REWARD_W_BUMP  * max(0, Δbumpiness)— penalty for surface unevenness
#   - REWARD_W_AGG   * max(0, Δagg_h)    — mild penalty for stack growth
# where Δ = metric(board after lock+clear) − metric(board before lock).

# IMPORTANT (lesson from the first v2 run): at REWARD_PIECE=0.1 and w_holes=0.5
# random play gave ~-1/step, while a quick suicide cost only -14 per episode. PPO
# fell into the local optimum "die faster" and the entropy collapsed. The
# placement bonus must therefore DOMINATE the penalties: survival is always
# profitable, and the shaping is only a mild quality gradient on top.
REWARD_SURVIVE = 0.0        # per-frame bonus (legacy frame mode only)
REWARD_GAME_OVER = -5.0     # game over penalty (suicide is strictly unprofitable)
REWARD_PIECE = 1.0          # bonus for each successfully placed piece

# Line clear bonus = REWARD_LINE_BASE * (cleared_lines ** 2)
#   1 line → 2, 2 → 8, 3 → 18, 4 (tetris) → 32. This encourages tetrises.
REWARD_LINE_BASE = 2.0

# Dense shaping weights (by the delta of board metrics at the lock). Only
# DEGRADATION is penalized (max(0, Δ)); clearing lines decreases the metrics ->
# no penalty, plus a separate line bonus. Holes weigh the most (Dellacherie).
REWARD_W_HOLES = 0.3        # penalty for an increase in the number of holes
REWARD_W_BUMP = 0.03        # penalty for an increase in bumpiness
REWARD_W_AGG = 0.0          # penalty for an increase in total height (0 = off)

# Cap on the total shaping penalty per placement — against spikes where one lock
# on a dirty board buries dozens of cells (Δholes ~ 40). Below REWARD_PIECE:
# even the worst placement is no worse than -0.5 in total.
REWARD_SHAPING_CLIP = 1.5

# ── Afterstate features for EML (features.py) ────────────────────────────────
#
# The EML formula evaluates the BOARD AFTER PLACEMENT (afterstate) — like a
# classic Dellacherie evaluation function. One formula rather than 6:
#   placement* = argmax_i f(features(afterstate_i))
#
# The formula input = 19 afterstate features + a one-hot of the NEXT piece (7):
# the oracle plans with the next piece in mind (it sees it in scalars), and the
# formula needs the same data to reproduce that behavior. The oracle receives
# only the 19 afterstate features per candidate (the next piece is in its
# context).

AFTERSTATE_FEATURE_NAMES = [
    'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',  # column heights
    'holes', 'bumpiness', 'agg_h', 'max_h', 'wells',
    'row_trans', 'col_trans',
    'cleared',      # lines cleared by this placement (0..4)
    'landing_h',    # height at which the piece landed
]
N_AFTERSTATE_FEATURES = len(AFTERSTATE_FEATURE_NAMES)   # 19 — the oracle input

FEATURE_NAMES = AFTERSTATE_FEATURE_NAMES + [
    'next_I', 'next_O', 'next_T', 'next_S', 'next_Z', 'next_J', 'next_L',
]
N_FEATURES = len(FEATURE_NAMES)   # 26 — the EML formula input

# ── Oracle: CNN + PPO (over placements) ──────────────────────────────────────

PPO_LR = 2.5e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.02         # 40 actions: higher than usual, against early
                            # determinization (entropy collapse kills exploration)
PPO_N_ENVS = 16             # parallel environments
PPO_ROLLOUT = 128           # PLACEMENTS per env per rollout collection
PPO_EPOCHS = 4              # SGD epochs over a collected rollout
PPO_BATCH_SIZE = 256
PPO_TOTAL_STEPS = 3_000_000  # budget in placements (~1 placement = 20+ frames)
PPO_TARGET_SCORE = 100      # avg lines per episode -> oracle readiness criterion

# ── Autopilot (unattended training) ──────────────────────────────────────────
# Enabled by a checkbox in the GUI. On every update it reacts to metric dynamics:
#   • auto-save the model when best avg lines improves (progress is not lost);
#   • on a plateau (no improvement for PATIENCE updates) — LR decay;
#   • on entropy collapse — raise ent_coef for the sake of exploration.
# IMPORTANT (lesson from v3): LR decay with patience 40 strangled training — LR
# fell to 1e-5 by a third of the budget, while the agent was still LEARNING
# rather than converged. The plateau "not learned yet" and the plateau
# "converged" are indistinguishable to a simple heuristic, so patience is large
# and the floor prevents killing the gradient entirely.
AUTOPILOT_SAVE_PATH = 'models/autosave_ppo.pt'
AUTOPILOT_SAVE_DELTA = 0.5      # how much best avg must grow for an auto-save
AUTOPILOT_PATIENCE = 200        # updates without improvement -> corrective action
AUTOPILOT_LR_DECAY = 0.7        # LR multiplier on a plateau
AUTOPILOT_LR_FLOOR = 5e-5       # LR is not lowered below this
AUTOPILOT_ENT_FLOOR = 0.5       # entropy below the threshold -> the policy collapses
AUTOPILOT_ENT_BOOST = 1.5       # ent_coef multiplier on entropy collapse
AUTOPILOT_ENT_COEF_MAX = 0.05   # ent_coef ceiling

# ── EML distillation (one formula, afterstate value) ─────────────────────────

EML_POPULATION = 600        # EML tree population size
EML_GENERATIONS = 600       # maximum DATA phase generations
EML_ELITISM = 20            # elite individuals carried over unchanged
EML_MAX_DEPTH = 7           # maximum tree depth
EML_PATIENCE = 120          # generations without improvement -> stop
EML_DATASET_EPISODES = 40   # oracle episodes for dataset collection
EML_EPSILON = 1e-10         # guard against ln(0)
# exp() clamp. Both guards are parameters of THIS operator, not of the EML
# operator in the literature: exp(x) - ln(y) is complete over the complex
# numbers, exp(clamp(x)) - ln(|y| + eps) is a different, real-computable
# function. Named here so the gap can be swept and reported rather than assumed
# away; the clamp dominates eps by orders of magnitude, so sweep this one.
EML_CLAMP = 10.0

# Which set of primitives the search may use: 'eml' (the single operator this
# work is about) or 'classic' (+ - * / exp log tanh). The control basis is not
# decoration: an EML number means nothing until the same pipeline, at the same
# budget and the same seeds, has been run on an ordinary one.
EML_BASIS = 'eml'

# Multi-objective search over (quality, size). It reads the whole quality-vs-K
# curve off one run instead of needing one run per budget — the difference
# between ~160 and ~1440 GA runs for the full sweep.
EML_MULTI_OBJECTIVE = True
EML_PARETO_MAX_POINTS = 12   # front members evaluated in-game (games cost time)
# Give each curve point its own short GAME phase before scoring it. Off by
# default because it costs one in-game evolution per point; ON is what the curve
# needs to be about game score at all — DATA-phase formulas score near zero
# lines whatever their regression fitness, which is why the GAME phase exists.
EML_PARETO_REFINE = False
EML_PARETO_REFINE_GENERATIONS = 12
EML_PHASE_SWITCH = 0.15     # fraction of generations for the DATA phase (warm-up)

# ── Dataset collection ─────────────────────────────────────────────────────
EML_DATA_EPSILON_EXPLORE = 0.05   # fraction of random oracle placements (coverage)
EML_DATA_MAX_PLACEMENTS = 400     # placement limit per episode during collection
EML_DATA_SUBSAMPLE = 8000         # dataset subsample for fitness (speed)
EML_VAR_BONUS = 0.03              # fitness bonus per unique variable used
EML_DATA_TOP_WEIGHT = 3.0         # extra weight for examples the oracle is sure of

# ── Batch distillation ───────────────────────────────────────────────────────
# A batch is a grid: EML_BATCH_CONFIGS configurations x EML_BATCH_SEEDS GA seeds.
# The dataset is collected ONCE for the whole batch. The best run (by VAL) ->
# best_eml.json, every run + per-config spread -> runs/<date>_distill_batch/.
#
# The two axes are kept apart deliberately. Jittering hyper-parameters and the
# seed together — as the old EML_BATCH_VARIANTS did — produces runs that differ
# in everything at once, so the seed-to-seed spread cannot be separated from the
# effect of the parameters, and a batch can only rank runs it cannot distinguish.
# Default shape: one configuration, five seeds (a replicate study).
EML_BATCH_CONFIGS = 1       # GA configurations per batch
EML_BATCH_SEEDS = 5         # GA seeds per configuration (replicates)
EML_BATCH_SPREAD = 0.3      # parameter spread: ±30% on numeric values
EML_GA_SEED_BASE = 20       # GA seeds are BASE, BASE+1, ... (not env seeds)

# ── Joint (in-game) distillation phase ───────────────────────────────────────
# The DATA phase regresses the formula to the oracle logits, the GAME phase
# refines it by evolution against the actual score in TetrisEnv (placement mode).
EML_INGAME_GAMES = 3              # games per formula evaluation
EML_INGAME_MAX_PLACEMENTS = 300   # placement limit per evaluation game (speed)
EML_JOINT_POPULATION = 60         # GAME phase population size
EML_JOINT_GENERATIONS = 40        # GAME phase generations
EML_JOINT_ELITISM = 6             # elite formulas
EML_JOINT_PATIENCE = 12           # generations without improvement -> stop

# ── Environment seed pools (disjoint) ────────────────────────────────────────
# Fitness, selection and reporting must not share environment seeds. Selecting a
# formula by the same number that is later reported turns the report into a
# maximum over noise, and the reported score is then biased upward regardless of
# how many episodes it averages. Three disjoint pools, mirroring train/val/test:
#
#   TRAIN — GA fitness only. Rotated across generations so no single piece
#           sequence can be memorised; inside one generation every individual
#           gets the SAME seeds (common random numbers), otherwise selection
#           compares formulas across different games and becomes a lottery.
#   VAL   — hall-of-fame re-scoring and picking the best batch variant. Never
#           enters a fitness function.
#   TEST  — the final reported score. Touched once per run, on the winner only.
#
# The distillation dataset uses its own stream (dataset_collector seed=0), which
# is disjoint from all three.
EML_SEED_POOL_TRAIN = list(range(1000, 1064))   # 64 seeds, rotated per generation
EML_SEED_POOL_VAL = list(range(3000, 3032))     # 32 seeds
EML_SEED_POOL_TEST = list(range(5000, 5064))    # 64 seeds, reporting only

EML_VAL_GAMES = 8        # games per hall-of-fame re-scoring (VAL pool)
EML_TEST_GAMES = 30      # games in the final holdout report (TEST pool)
EML_DATA_VAL_FRACTION = 0.15   # fraction of dataset GROUPS held out for DATA val

DEPTH_PENALTIES = {
    'weak':   0.01,
    'medium': 0.05,
    'strong': 0.15,
}

# ── Directories ──────────────────────────────────────────────────────────────

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
