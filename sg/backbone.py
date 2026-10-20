"""The frozen feature extractor: PANNs CNN14, pretrained on AudioSet.

The classifier is a single linear layer on top of these features and nothing here is ever trained.
Two things must not be done to CNN14, and both are easy to do by accident:

  - do not apply a per-recording min-max normalisation on top of it. CNN14 carries its own
    spectrogram front end and a dataset-level batch normalisation over the frequency axis. The
    per-recording rescaling of the ImageNet-transfer recipe would cancel exactly the absolute level
    that this front end preserves.
  - do not feed it 16 kHz audio. It was trained at 32 kHz with an upper edge of 14 kHz, so audio
    low-passed at 8 kHz throws away a third of the band it expects.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SR = 32000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ConvBlock(nn.Module):
    def __init__(self, i: int, o: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(i, o, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(o), nn.BatchNorm2d(o)

    def forward(self, x, pool=(2, 2)):
        x = torch.relu_(self.bn1(self.conv1(x)))
        x = torch.relu_(self.bn2(self.conv2(x)))
        return nn.functional.avg_pool2d(x, kernel_size=pool)


class Cnn14(nn.Module):
    """Reproduces the published architecture exactly, so the released weights load with no gaps."""

    EMBED_DIM = 2048

    def __init__(self) -> None:
        super().__init__()
        from torchlibrosa.augmentation import SpecAugmentation
        from torchlibrosa.stft import LogmelFilterBank, Spectrogram

        self.spectrogram_extractor = Spectrogram(
            n_fft=1024, hop_length=320, win_length=1024, window="hann",
            center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=SR, n_fft=1024, n_mels=64, fmin=50, fmax=14000,
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.spec_augmenter = SpecAugmentation(64, 2, 8, 2)
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)      # unused; kept so the load is exact

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.logmel_extractor(self.spectrogram_extractor(x))
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        for blk, pool in ((self.conv_block1, (2, 2)), (self.conv_block2, (2, 2)),
                          (self.conv_block3, (2, 2)), (self.conv_block4, (2, 2)),
                          (self.conv_block5, (2, 2)), (self.conv_block6, (1, 1))):
            x = blk(x, pool)
        x = torch.mean(x, dim=3)                                # pool over frequency
        x = (nn.functional.max_pool1d(x, 3, 1, 1) +
             nn.functional.avg_pool1d(x, 3, 1, 1))
        x = torch.mean(x, dim=2)                                # pool over time
        return torch.relu_(self.fc1(x))                         # the 2048-d embedding


def load(ckpt: str = "ckpt/Cnn14_mAP=0.431.pth", verbose: bool = True) -> Cnn14:
    m = Cnn14()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if verbose:
        print(f"  CNN14: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
    if missing or unexpected:
        raise RuntimeError("the checkpoint does not match the architecture")
    return m.to(DEVICE).eval()


@torch.no_grad()
def embed(model: Cnn14, waves: np.ndarray, batch: int = 128) -> torch.Tensor:
    out = []
    for i in range(0, len(waves), batch):
        w = torch.from_numpy(np.ascontiguousarray(waves[i:i + batch])).float().to(DEVICE)
        out.append(model(w).cpu())
    return torch.cat(out)
