"""
cnn_oracle.py — CNN actor-critic oracle for Tetris (trained with PPO).

Placement-based (v2.1): the policy scores EVERY legal placement from its
afterstate features plus a board context produced by the CNN.

Input:
    grid:    (B, 1, 24, 10)  — locked board cells.
    scalars: (B, 14)         — one-hot of the current and the next piece.
    afeats:  (B, 40, 19)     — afterstate features of every placement.
    mask:    (B, 40)         — 1 for legal placements (rot × column).

Architecture:
    context h = FC(CNN(grid) + scalars)          — "what is going on on the board"
    logit_i   = MLP(afeats_i + h) for every i    — "how good this placement is"
    value     = FC(h)

Key idea: the network does not have to mentally simulate the drop geometry — it
is shown the measurable consequences of every move directly (holes, heights,
lines). This is the same feature space as the EML formula's; the CNN context
adds what the features do not capture (stack structure, synergy with the next
piece).

Output:
    logits: (B, 40) — policy logits; illegal actions -> -inf.
    value:  (B,)    — state value V(s) (critic).

Device-aware: uses CUDA automatically when available.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BOARD_W, BOARD_H_TOTAL, N_PLACEMENTS, N_AFTERSTATE_FEATURES


GRID_CHANNELS = 1
SCALAR_SIZE = 14
AFEAT_SIZE = N_AFTERSTATE_FEATURES   # 19 afterstate features per placement
                                     # (the next piece lives in the context scalars)
CONTEXT_SIZE = 96              # size of the board context vector
_MASK_FILL = -1e9              # logit of an illegal action


def get_device() -> torch.device:
    """CUDA if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def describe_device() -> str:
    """
    Human-readable device description plus a hint when the machine has an NVIDIA
    GPU but a CPU-only torch is installed (a common situation on servers).
    """
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    import shutil
    if shutil.which('nvidia-smi'):
        return ("cpu  [!] nvidia-smi found, but torch has no CUDA. Install:\n"
                "    pip install torch --index-url "
                "https://download.pytorch.org/whl/cu124")
    return "cpu"


class TetrisCNN(nn.Module):
    """Actor-Critic: CNN board context + a per-placement MLP scorer."""

    def __init__(self):
        super().__init__()
        # ── Conv trunk (board context) ───────────────────────────────────
        self.conv1 = nn.Conv2d(GRID_CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, padding=1)

        conv_flat = 8 * BOARD_H_TOTAL * BOARD_W           # 8*24*10 = 1920
        self.fc_ctx = nn.Linear(conv_flat + SCALAR_SIZE, CONTEXT_SIZE)

        # ── Per-placement scorer: logit_i = MLP(afeats_i + context) ──────
        self.score1 = nn.Linear(AFEAT_SIZE + CONTEXT_SIZE, 64)
        self.score2 = nn.Linear(64, 1)

        # ── Critic ───────────────────────────────────────────────────────
        self.critic1 = nn.Linear(CONTEXT_SIZE, 64)
        self.critic2 = nn.Linear(64, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (the PPO standard)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Scorer head — a small gain for a near-uniform initial policy.
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
            logits: (B, N_PLACEMENTS) — masked if a mask was passed.
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

    # ── Helpers for rollout / inference ──────────────────────────────────

    @torch.no_grad()
    def act(self, grid: torch.Tensor, scalars: torch.Tensor,
            afeats: torch.Tensor, mask: torch.Tensor,
            deterministic: bool = False):
        """
        Choose placement(s) among the legal ones.

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
        For the PPO update: log_prob, entropy and value for the given actions.
        """
        logits, value = self.forward(grid, scalars, afeats, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    @torch.no_grad()
    def get_logits(self, grid: torch.Tensor, scalars: torch.Tensor,
                   afeats: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
        """Policy logits — the target for EML distillation."""
        logits, _ = self.forward(grid, scalars, afeats, mask)
        return logits


# ── obs -> tensor wrapper utilities ──────────────────────────────────────────

def obs_to_tensors(obs: dict, device: torch.device):
    """One placement obs (dict from env) -> batch tensors of size 1."""
    grid = torch.from_numpy(obs['grid']).unsqueeze(0).to(device)
    scalars = torch.from_numpy(obs['scalars']).unsqueeze(0).to(device)
    afeats = torch.from_numpy(obs['afeats']).unsqueeze(0).to(device)
    mask = torch.from_numpy(obs['mask']).unsqueeze(0).to(device)
    return grid, scalars, afeats, mask


def batch_obs_to_tensors(obs_list: list[dict], device: torch.device):
    """List of obs -> batch tensors (B, ...)."""
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
