"""
Tetris AI — конфигурация.
Все гиперпараметры проекта в одном месте (по образцу flappy_bird_with_ai/config.py).

Три фазы пайплайна — три блока констант:
  1. Среда (TetrisEnv)
  2. Оракул (CNN + PPO)
  3. EML-дистилляция
"""

# ── Константы среды (TetrisEnv) ──────────────────────────────────────────────

BOARD_W = 10            # ширина доски (колонок)
BOARD_H = 20            # видимая высота доски (строк)
BOARD_BUFFER = 4        # буфер сверху для спавна фигур (скрытые строки)
BOARD_H_TOTAL = BOARD_H + BOARD_BUFFER   # полная высота массива доски = 24

GRAVITY_TICKS = 20      # кадров между падениями фигуры на 1 клетку
MAX_EPISODE_FRAMES = 10000   # лимит длины эпизода (anti-stall)

N_ACTIONS = 6           # NO-OP, left, right, rotate CW, soft drop, hard drop
N_PIECES = 7            # I, O, T, S, Z, J, L
N_ROTATIONS = 4         # поворотов на фигуру

# ── Reward shaping ───────────────────────────────────────────────────────────

REWARD_SURVIVE = 0.01       # за каждый шаг выживания
REWARD_GAME_OVER = -1.0     # за game over
# Бонус за очистку линий = REWARD_LINE_BASE * (cleared_lines ** 2)
#   1 линия → 1, 2 → 4, 3 → 9, 4 (тетрис) → 16. Стимулирует тетрисы.
REWARD_LINE_BASE = 1.0
# Мягкий штраф за рост стека: REWARD_HEIGHT_PENALTY * delta(aggregate_height).
#   0.0 = выключено. Включать осторожно — легко "взломать" политику.
REWARD_HEIGHT_PENALTY = 0.0

# ── Признаки для EML (features.py) ───────────────────────────────────────────

# Имена признаков (порядок = var_idx в EML-дереве). 28 признаков.
FEATURE_NAMES = [
    'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',  # высоты колонок
    'holes', 'bumpiness', 'agg_h', 'max_h', 'wells',
    'row_trans', 'col_trans',
    'piece_I', 'piece_O', 'piece_T', 'piece_S', 'piece_Z', 'piece_J', 'piece_L',
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
]
N_FEATURES = len(FEATURE_NAMES)   # 28

# ── Оракул: CNN + PPO ────────────────────────────────────────────────────────

PPO_LR = 3e-4
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
PPO_VF_COEF = 0.5
PPO_ENT_COEF = 0.01
PPO_N_ENVS = 16             # параллельных сред
PPO_ROLLOUT = 2048          # шагов на env за один сбор rollout
PPO_EPOCHS = 10             # эпох SGD на собранном rollout
PPO_BATCH_SIZE = 256
PPO_TOTAL_STEPS = 5_000_000  # стартовый бюджет; увеличить при стагнации
PPO_TARGET_SCORE = 20       # avg линий за эпизод → критерий готовности оракула

# ── EML-дистилляция ──────────────────────────────────────────────────────────

EML_POPULATION = 800        # размер популяции EML-деревьев
EML_GENERATIONS = 1000      # максимум поколений
EML_ELITISM = 20            # элитные особи без изменений
EML_MAX_DEPTH = 7           # максимальная глубина дерева
EML_PATIENCE = 150          # поколений без улучшения → стоп
EML_DATASET_EPISODES = 1000  # эпизодов оракула для сбора датасета
EML_EPSILON = 1e-10         # защита от ln(0)
EML_PHASE_SWITCH = 0.15     # доля генераций на DATA-фазу (warm-up)

DEPTH_PENALTIES = {
    'weak':   0.01,
    'medium': 0.05,
    'strong': 0.15,
}

# ── Директории ───────────────────────────────────────────────────────────────

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'
