"""
cnn_oracle.py — CNN actor-critic оракул для Tetris (обучается PPO).

Placement-based (v2.1): политика оценивает КАЖДУЮ легальную постановку по
признакам её afterstate + контексту доски от CNN.

Вход:
    grid:    (B, 1, 24, 10)  — зафиксированные ячейки доски.
    scalars: (B, 14)         — one-hot текущей и следующей фигуры.
    afeats:  (B, 40, 19)     — признаки afterstate каждой постановки.
    mask:    (B, 40)         — 1 для легальных постановок (rot × column).

Архитектура:
    context h = FC(CNN(grid) ⊕ scalars)              — «что происходит на доске»
    logit_i   = MLP(afeats_i ⊕ h)  для каждой i      — «насколько хороша постановка»
    value     = FC(h)

Ключевая идея: сети не нужно мысленно симулировать геометрию падения — ей
сразу показывают измеримые последствия каждого хода (дыры, высоты, линии).
Это то же признаковое пространство, что у EML-формулы; CNN-контекст добавляет
то, что признаки не ловят (структура стека, синергия со следующей фигурой).

Выход:
    logits: (B, 40) — логиты политики; нелегальные действия → -inf.
    value:  (B,)    — оценка состояния V(s) (critic).

Device-aware: автоматически использует CUDA, если доступна.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BOARD_W, BOARD_H_TOTAL, N_PLACEMENTS, N_AFTERSTATE_FEATURES


GRID_CHANNELS = 1
SCALAR_SIZE = 14
AFEAT_SIZE = N_AFTERSTATE_FEATURES   # 19 afterstate-признаков на постановку
                                     # (next-фигура — в scalars контекста)
CONTEXT_SIZE = 96              # размер контекст-вектора доски
_MASK_FILL = -1e9              # логит нелегального действия


def get_device() -> torch.device:
    """CUDA, если доступна, иначе CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def describe_device() -> str:
    """
    Человекочитаемое описание устройства + подсказка, если в машине есть
    NVIDIA GPU, а установлен CPU-only torch (частая ситуация на серверах).
    """
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    import shutil
    if shutil.which('nvidia-smi'):
        return ("cpu  [!] найден nvidia-smi, но torch без CUDA. Установите:\n"
                "    pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124")
    return "cpu"


class TetrisCNN(nn.Module):
    """Actor-Critic: CNN-контекст доски + per-placement MLP-оценщик."""

    def __init__(self):
        super().__init__()
        # ── Conv-ствол (контекст доски) ──────────────────────────────────
        self.conv1 = nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)

        conv_flat = 8 * BOARD_H_TOTAL * BOARD_W           # 8*24*10 = 1920
        self.fc_ctx = nn.Linear(conv_flat + SCALAR_SIZE, CONTEXT_SIZE)

        # ── Per-placement оценщик: logit_i = MLP(afeats_i ⊕ context) ─────
        self.score1 = nn.Linear(AFEAT_SIZE + CONTEXT_SIZE, 64)
        self.score2 = nn.Linear(64, 1)

        # ── Critic ───────────────────────────────────────────────────────
        self.critic1 = nn.Linear(CONTEXT_SIZE, 64)
        self.critic2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Ортогональная инициализация (стандарт для PPO)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Голова оценщика — малый gain для стартовой near-uniform политики.
        nn.init.orthogonal_(self.score2.weight, gain=0.01)
        nn.init.orthogonal_(self.critic2.weight, gain=1.0)

    def _context(self, grid: torch.Tensor,
                 scalars: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, scalars], dim=1)
        return F.relu(self.fc_ctx(x))                     # (B, CONTEXT_SIZE)

    def forward(self, grid: torch.Tensor, scalars: torch.Tensor,
                afeats: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, N_PLACEMENTS) — с маской, если она передана.
            value:  (B,).
        """
        h = self._context(grid, scalars)                  # (B, C)
        h_exp = h.unsqueeze(1).expand(-1, N_PLACEMENTS, -1)   # (B, 40, C)
        z = torch.cat([afeats, h_exp], dim=-1)            # (B, 40, 19+C)
        z = F.relu(self.score1(z))
        logits = self.score2(z).squeeze(-1)               # (B, 40)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, _MASK_FILL)
        v = F.relu(self.critic1(h))
        value = self.critic2(v).squeeze(-1)
        return logits, value

    # ── Helpers для rollout / инференса ──────────────────────────────────

    @torch.no_grad()
    def act(self, grid: torch.Tensor, scalars: torch.Tensor,
            afeats: torch.Tensor, mask: torch.Tensor,
            deterministic: bool = False):
        """
        Выбрать постановку(и) среди легальных.

        Returns:
            action: (B,) long
            log_prob: (B,)
            value: (B,)
        """
        logits, value = self.forward(grid, scalars, afeats, mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, grid: torch.Tensor, scalars: torch.Tensor,
                         afeats: torch.Tensor, mask: torch.Tensor,
                         actions: torch.Tensor):
        """
        Для PPO-обновления: log_prob, entropy, value по заданным действиям.
        """
        logits, value = self.forward(grid, scalars, afeats, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    @torch.no_grad()
    def get_logits(self, grid: torch.Tensor, scalars: torch.Tensor,
                   afeats: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
        """Логиты политики — цель для EML-дистилляции."""
        logits, _ = self.forward(grid, scalars, afeats, mask)
        return logits


# ── Утилиты обёртки obs → тензоры ────────────────────────────────────────────

def obs_to_tensors(obs: dict, device: torch.device):
    """Один placement-obs (dict из env) → батч-тензоры размера 1."""
    grid = torch.from_numpy(obs['grid']).unsqueeze(0).to(device)
    scalars = torch.from_numpy(obs['scalars']).unsqueeze(0).to(device)
    afeats = torch.from_numpy(obs['afeats']).unsqueeze(0).to(device)
    mask = torch.from_numpy(obs['mask']).unsqueeze(0).to(device)
    return grid, scalars, afeats, mask


def batch_obs_to_tensors(obs_list: list[dict], device: torch.device):
    """Список obs → батч-тензоры (B, ...)."""
    grids = np.stack([o['grid'] for o in obs_list])
    scalars = np.stack([o['scalars'] for o in obs_list])
    afeats = np.stack([o['afeats'] for o in obs_list])
    masks = np.stack([o['mask'] for o in obs_list])
    return (torch.from_numpy(grids).to(device),
            torch.from_numpy(scalars).to(device),
            torch.from_numpy(afeats).to(device),
            torch.from_numpy(masks).to(device))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
