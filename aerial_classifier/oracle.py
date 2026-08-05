"""
oracle.py — the oracle networks whose decisions are later distilled into EML
formulas.

Two models for the two experiment branches:

    FeatureMLP (branch A) — input: 32 engineered features from features.py.
        This is the main branch. The EML formulas receive EXACTLY THE SAME
        vector, so after distillation the neural network is discarded entirely:
        the microcontroller keeps the feature extractor and 4 formulas.

    PixelCNN (branch B)   — input: raw 64x64x3 pixels.
        Ablation branch: an upper bound on quality. The convolutions compress
        the image into a narrow embedding (CNN_EMBED_DIM = 32, as many as there
        are features), and EML approximates the head over that embedding. The
        convolutions remain on the MCU here — the branch exists to measure how
        much accuracy dropping them costs.

Important for both branches: the PixelCNN embedding passes through tanh and lies
in [-1, 1], like the branch A features. This is not cosmetic: the EML operator
contains exp(), and unbounded activations must not be fed into it (see
update.md, the rule about clamping before exp). It also makes the branches
comparable in input scale, so the comparison does not come down to differing
normalization rather than quality.
"""

import numpy as np
import torch
import torch.nn as nn

import config


def get_device() -> torch.device:
    """CUDA if available, otherwise CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Branch A: MLP over engineered features ────────────────────────────────────

class FeatureMLP(nn.Module):
    """Classifier over 32 features. Input (B, 32) -> logits (B, 4)."""

    def __init__(self, n_features: int | None = None,
                 hidden: list[int] | None = None,
                 n_classes: int | None = None,
                 dropout: float | None = None):
        super().__init__()
        n_features = n_features if n_features is not None else config.N_FEATURES
        hidden = hidden if hidden is not None else config.MLP_HIDDEN
        n_classes = n_classes if n_classes is not None else config.N_CLASSES
        dropout = dropout if dropout is not None else config.MLP_DROPOUT

        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Branch B: CNN over pixels with a narrow bottleneck ────────────────────────

class PixelCNN(nn.Module):
    """
    Convolutional classifier. Input (B, 3, 64, 64) float in [0, 1] -> (B, 4).

    Three conv-bn-relu-conv-bn-relu-pool blocks: 64 -> 32 -> 16 -> 8, then
    global average pooling, a narrow embedding, and a linear head.
    """

    def __init__(self, channels: list[int] | None = None,
                 embed_dim: int | None = None,
                 n_classes: int | None = None,
                 dropout: float | None = None):
        super().__init__()
        channels = channels if channels is not None else config.CNN_CHANNELS
        embed_dim = embed_dim if embed_dim is not None else config.CNN_EMBED_DIM
        n_classes = n_classes if n_classes is not None else config.N_CLASSES
        dropout = dropout if dropout is not None else config.CNN_DROPOUT

        blocks: list[nn.Module] = []
        prev = config.IMG_CHANNELS
        for ch in channels:
            blocks += [
                nn.Conv2d(prev, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.embed_fc = nn.Linear(prev, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bottleneck embedding (B, embed_dim) in [-1, 1] — the input of the branch
        B EML formulas.

        tanh is mandatory here: without bounded activations a formula containing
        exp() blows up to inf on the first outlier.
        """
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return torch.tanh(self.embed_fc(self.dropout(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ── Inference helpers ─────────────────────────────────────────────────────────

@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, device=None,
                   batch_size: int = 512) -> np.ndarray:
    """
    Model logits for a numpy array. Returns (N, N_CLASSES) float32.

    X is either (N, 32) features (FeatureMLP) or (N, 64, 64, 3) uint8 images
    (PixelCNN); the required conversion is chosen by dimensionality.
    """
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        tensor = to_input_tensor(batch, device)
        out.append(model(tensor).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def to_input_tensor(batch: np.ndarray, device) -> torch.Tensor:
    """numpy batch -> a tensor in the form the model expects."""
    if batch.ndim == 4:                                    # images (N,H,W,3)
        t = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        return t.permute(0, 3, 1, 2).float() / 255.0
    return torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)


@torch.no_grad()
def predict_embeddings(model: PixelCNN, images: np.ndarray, device=None,
                       batch_size: int = 512) -> np.ndarray:
    """PixelCNN embeddings for images — the input of branch B EML distillation."""
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for start in range(0, len(images), batch_size):
        batch = to_input_tensor(images[start:start + batch_size], device)
        out.append(model.embed(batch).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters — for the comparison table in the report."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
