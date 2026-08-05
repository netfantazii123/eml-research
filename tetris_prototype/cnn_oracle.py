"""
cnn_oracle.py — CNN actor-critic oracle for Tetris (trained with PPO).

Input:
    grid:    (B, 3, 24, 10) — locked / current piece / ghost.
    scalars: (B, 19)        — one-hot of piece/next/rotation + drop progress.

Output:
    logits: (B, 6) — raw policy logits (before softmax).
    value:  (B,)   — state value V(s) (critic).

Architecture: a shared trunk with two heads.
Device-aware: uses CUDA automatically when available.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BOARD_W, BOARD_H_TOTAL, N_ACTIONS


SCALAR_SIZE = 19


def get_device() -> torch.device:
    """CUDA if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TetrisCNN(nn.Module):
    """
    Actor-Critic CNN.

    The conv trunk processes the 3-channel board, the result is concatenated
    with the scalar features and goes into a shared FC, then into two heads.
    """

    def __init__(self):
        super().__init__()
        # ── Conv trunk ───────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        conv_flat = 64 * BOARD_H_TOTAL * BOARD_W          # 64*24*10 = 15360
        fc_in = conv_flat + SCALAR_SIZE                   # + 19 = 15379

        # ── Shared FC ────────────────────────────────────────────────────
        self.fc1 = nn.Linear(fc_in, 256)
        self.fc2 = nn.Linear(256, 128)

        # ── Heads ────────────────────────────────────────────────────────
        self.actor = nn.Linear(128, N_ACTIONS)            # policy pi
        self.critic = nn.Linear(128, 1)                   # value V

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (the PPO standard)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Policy head — a small gain for a near-uniform initial policy.
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

    # ── Helpers for rollout / inference ──────────────────────────────────

    @torch.no_grad()
    def act(self, grid: torch.Tensor, scalars: torch.Tensor,
            deterministic: bool = False):
        """
        Choose action(s).

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
        For the PPO update: log_prob, entropy and value for the given actions.
        """
        logits, value = self.forward(grid, scalars)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    @torch.no_grad()
    def get_logits(self, grid: torch.Tensor,
                   scalars: torch.Tensor) -> torch.Tensor:
        """Raw logits — the target for EML distillation."""
        logits, _ = self.forward(grid, scalars)
        return logits


# ── obs -> tensor wrapper utilities ──────────────────────────────────────────

def obs_to_tensors(obs: dict, device: torch.device):
    """One obs (dict from env) -> batch tensors of size 1."""
    grid = torch.from_numpy(obs['grid']).unsqueeze(0).to(device)
    scalars = torch.from_numpy(obs['scalars']).unsqueeze(0).to(device)
    return grid, scalars


def batch_obs_to_tensors(obs_list: list[dict], device: torch.device):
    """List of obs -> batch tensors (B, ...)."""
    grids = np.stack([o['grid'] for o in obs_list])
    scalars = np.stack([o['scalars'] for o in obs_list])
    return (torch.from_numpy(grids).to(device),
            torch.from_numpy(scalars).to(device))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
