"""
cnn_oracle.py — CNN actor-critic оракул для Tetris (обучается PPO).

Вход:
    grid:    (B, 3, 24, 10) — locked / current piece / ghost.
    scalars: (B, 19)        — one-hot фигуры/следующей/поворота + drop progress.

Выход:
    logits: (B, 6) — сырые логиты политики (до softmax).
    value:  (B,)   — оценка состояния V(s) (critic).

Архитектура (shared trunk, две головы) — см. PROJECT_BLUEPRINT.md § 3.3.
Device-aware: автоматически использует CUDA, если доступна.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BOARD_W, BOARD_H_TOTAL, N_ACTIONS


SCALAR_SIZE = 19


def get_device() -> torch.device:
    """CUDA, если доступна, иначе CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TetrisCNN(nn.Module):
    """
    Actor-Critic CNN.

    Conv-ствол обрабатывает 3-канальную доску, результат конкатенируется
    со скалярными признаками и идёт в общий FC, затем в две головы.
    """

    def __init__(self):
        super().__init__()
        # ── Conv-ствол ───────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        conv_flat = 64 * BOARD_H_TOTAL * BOARD_W          # 64*24*10 = 15360
        fc_in = conv_flat + SCALAR_SIZE                   # + 19 = 15379

        # ── Общий FC ─────────────────────────────────────────────────────
        self.fc1 = nn.Linear(fc_in, 256)
        self.fc2 = nn.Linear(256, 128)

        # ── Головы ───────────────────────────────────────────────────────
        self.actor = nn.Linear(128, N_ACTIONS)            # политика π
        self.critic = nn.Linear(128, 1)                   # ценность V

        self._init_weights()

    def _init_weights(self):
        """Ортогональная инициализация (стандарт для PPO)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Голова политики — малый gain для стартовой near-uniform политики.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, grid: torch.Tensor,
                scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, N_ACTIONS), value: (B,).
        """
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, scalars], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    # ── Helpers для rollout / инференса ──────────────────────────────────

    @torch.no_grad()
    def act(self, grid: torch.Tensor, scalars: torch.Tensor,
            deterministic: bool = False):
        """
        Выбрать действие(я).

        Returns:
            action: (B,) long
            log_prob: (B,)
            value: (B,)
        """
        logits, value = self.forward(grid, scalars)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, grid: torch.Tensor, scalars: torch.Tensor,
                         actions: torch.Tensor):
        """
        Для PPO-обновления: log_prob, entropy, value по заданным действиям.
        """
        logits, value = self.forward(grid, scalars)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    @torch.no_grad()
    def get_logits(self, grid: torch.Tensor,
                   scalars: torch.Tensor) -> torch.Tensor:
        """Сырые логиты — цель для EML-дистилляции."""
        logits, _ = self.forward(grid, scalars)
        return logits


# ── Утилиты обёртки obs → тензоры ────────────────────────────────────────────

def obs_to_tensors(obs: dict, device: torch.device):
    """Один obs (dict из env) → батч-тензоры размера 1."""
    grid = torch.from_numpy(obs['grid']).unsqueeze(0).to(device)
    scalars = torch.from_numpy(obs['scalars']).unsqueeze(0).to(device)
    return grid, scalars


def batch_obs_to_tensors(obs_list: list[dict], device: torch.device):
    """Список obs → батч-тензоры (B, ...)."""
    grids = np.stack([o['grid'] for o in obs_list])
    scalars = np.stack([o['scalars'] for o in obs_list])
    return (torch.from_numpy(grids).to(device),
            torch.from_numpy(scalars).to(device))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
