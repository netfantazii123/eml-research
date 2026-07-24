"""
oracle.py — нейросети-оракулы, чьё решение потом дистиллируется в EML-формулы.

Две модели под две ветки эксперимента:

    FeatureMLP (ветка A) — вход: 32 инженерных признака из features.py.
        Это основная ветка. EML-формулы получают РОВНО ТОТ ЖЕ вектор, поэтому
        после дистилляции нейросеть выбрасывается целиком: на микроконтроллере
        остаются экстрактор признаков и 4 формулы.

    PixelCNN (ветка B)   — вход: сырые пиксели 64x64x3.
        Ablation-ветка: верхняя граница качества. Свёртки сжимают картинку в
        узкий эмбеддинг (CNN_EMBED_DIM = 32, столько же, сколько признаков),
        и EML аппроксимирует голову поверх эмбеддинга. Честно оговорено, что
        на MCU здесь остаются свёртки — ветка нужна, чтобы измерить, сколько
        точности стоит отказ от них.

Важно для обеих веток: эмбеддинг PixelCNN проходит через tanh и лежит в
[-1, 1] — как и признаки ветки A. Это не косметика: EML-оператор содержит
exp(), и подавать в него неограниченные активации нельзя (см. update.md,
правило про clamp перед exp). Заодно ветки становятся сопоставимы по масштабу
входа, иначе сравнение упирается в разную нормировку, а не в качество.
"""

import numpy as np
import torch
import torch.nn as nn

import config


def get_device() -> torch.device:
    """CUDA, если доступна, иначе CPU."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Ветка A: MLP на инженерных признаках ──────────────────────────────────────

class FeatureMLP(nn.Module):
    """Классификатор поверх 32 признаков. Вход (B, 32) -> логиты (B, 4)."""

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


# ── Ветка B: CNN на пикселях с узким bottleneck ───────────────────────────────

class PixelCNN(nn.Module):
    """
    Свёрточный классификатор. Вход (B, 3, 64, 64) float в [0, 1] -> (B, 4).

    Три блока conv-bn-relu-conv-bn-relu-pool: 64 -> 32 -> 16 -> 8, затем
    global average pooling, узкий эмбеддинг и линейная голова.
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
        Bottleneck-эмбеддинг (B, embed_dim) в [-1, 1] — вход EML-формул ветки B.

        tanh здесь обязателен: без ограничения активации формула с exp()
        улетает в inf на первом же выбросе.
        """
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return torch.tanh(self.embed_fc(self.dropout(h)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


# ── Инференс-хелперы ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, device=None,
                   batch_size: int = 512) -> np.ndarray:
    """
    Логиты модели для массива numpy. Возвращает (N, N_CLASSES) float32.

    X — либо (N, 32) признаки (FeatureMLP), либо (N, 64, 64, 3) uint8 картинки
    (PixelCNN); нужное преобразование выбирается по размерности.
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
    """numpy-пачка -> тензор нужного модели вида."""
    if batch.ndim == 4:                                    # картинки (N,H,W,3)
        t = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
        return t.permute(0, 3, 1, 2).float() / 255.0
    return torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)


@torch.no_grad()
def predict_embeddings(model: PixelCNN, images: np.ndarray, device=None,
                       batch_size: int = 512) -> np.ndarray:
    """Эмбеддинги PixelCNN для картинок — вход EML-дистилляции ветки B."""
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for start in range(0, len(images), batch_size):
        batch = to_input_tensor(images[start:start + batch_size], device)
        out.append(model.embed(batch).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def count_params(model: nn.Module) -> int:
    """Число обучаемых параметров — для таблицы сравнения в отчёте."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
