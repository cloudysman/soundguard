"""The report's own system: a frozen ResNet-18, pretrained on ImageNet, over log-mel spectrograms.

This reproduces exactly the preprocessing of features.py, so the Q1 to Q4 findings can be measured
on the same backbone the report describes, rather than on a stronger one. Interface matches
sg/backbone.py: load() returns a handle, embed(handle, waves) returns [N, 512].

The recipe, verbatim from features.py:
    resample to 16 kHz -> log-mel (64 bands, n_fft 1024, hop 512) -> amplitude to decibels
    -> replicate 1 channel to 3 -> per-recording min-max to [0, 1] -> ImageNet normalisation
    -> frozen ResNet-18 with the final layer removed -> 512 numbers
The pipeline decodes at 32 kHz for the other backbones, so the waveform is resampled to 16 kHz here.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR_IN, SR = 32000, 16000
EMBED_DIM = 512

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


def load(verbose: bool = True):
    mel = T.MelSpectrogram(sample_rate=SR, n_fft=1024, hop_length=512, n_mels=64).to(DEVICE)
    db = T.AmplitudeToDB().to(DEVICE)
    rs = T.Resample(SR_IN, SR).to(DEVICE)
    bb = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    bb.fc = nn.Identity()
    bb = bb.to(DEVICE).eval()
    if verbose:
        print(f"  ResNet-18: frozen ImageNet backbone, embedding dim {EMBED_DIM}", flush=True)
    return mel, db, rs, bb


@torch.no_grad()
def embed(bundle, waves: np.ndarray, batch: int = 128) -> torch.Tensor:
    mel, db, rs, bb = bundle
    out = []
    for i in range(0, len(waves), batch):
        w = torch.from_numpy(np.ascontiguousarray(waves[i:i + batch])).float().to(DEVICE)
        w = rs(w)                                        # 32 kHz -> 16 kHz
        f = db(mel(w)).unsqueeze(1).expand(-1, 3, -1, -1).clone()
        lo = f.amin(dim=(1, 2, 3), keepdim=True)
        hi = f.amax(dim=(1, 2, 3), keepdim=True)
        f = (f - lo) / (hi - lo + 1e-6)                  # per-recording min-max
        f = (f - _MEAN) / _STD                           # ImageNet normalisation
        out.append(bb(f).float().cpu())
    return torch.cat(out)
