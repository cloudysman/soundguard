"""SoundGuard — classifier heads (serving side).

Verbatim copies of the head classes from the training notebook so the saved
state_dicts load exactly. Imported lazily by predictor.py (needs torch).
"""
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Single-layer Linear classifier head."""
    def __init__(self, in_features: int = 512, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.classifier(x)


class MLPHead(nn.Module):
    """Two-layer MLP classifier head."""
    def __init__(self, in_features: int = 512, num_classes: int = 10,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


def load_head(name, ckpt_path, in_features=512, num_classes=10, device="cpu"):
    name = name.lower()
    head = (LinearHead(in_features, num_classes) if name == "linear"
            else MLPHead(in_features, num_classes))
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    head.load_state_dict(state)
    return head.to(device).eval()