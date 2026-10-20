"""A second frozen backbone: the audio spectrogram transformer (AST), pretrained on AudioSet.

The purpose is one specific rebuttal. A reviewer will say the reliability findings are an artefact
of a weak CNN14 with a linear head. AST is a stronger backbone (AudioSet mAP 0.459 vs CNN14's 0.431)
and a different architecture family (attention, not convolution). If the same findings hold on AST,
they are not an artefact of the CNN.

Interface matches sg/backbone.py exactly: load() returns a handle, embed(handle, waves) returns an
[N, dim] tensor, so pipeline.py treats the two backbones identically.

Two things must not be done, and both are silent if wrong:
  - AST wants 16 kHz. The pipeline decodes at 32 kHz for CNN14, so the waveform is resampled to
    16 kHz here before the feature extractor. Feeding 32 kHz produces wrong mel scaling and garbage.
  - Do NOT add any per-recording normalisation. The AST feature extractor already applies the
    AudioSet mean and standard deviation; a second normalisation destroys the absolute level.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR_IN, SR_AST = 32000, 16000
EMBED_DIM = 768


def load(verbose: bool = True):
    """Return (feature_extractor, model). The bare ASTModel gives the 768-d pooled embedding; the
    527-way AudioSet head is dropped, which is what the load report's 'unexpected classifier' means."""
    from transformers import ASTFeatureExtractor, ASTModel

    fe = ASTFeatureExtractor.from_pretrained(MODEL_ID)
    model = ASTModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    if verbose:
        print(f"  AST: loaded {MODEL_ID}, embedding dim {EMBED_DIM}", flush=True)
    return fe, model


_resample = torchaudio.transforms.Resample(SR_IN, SR_AST)


@torch.no_grad()
def embed(bundle, waves: np.ndarray, batch: int = 64) -> torch.Tensor:
    """waves: [N, T] mono numpy at 32 kHz (the pipeline's decode rate). Returns [N, 768]."""
    fe, model = bundle
    out = []
    for i in range(0, len(waves), batch):
        w = torch.from_numpy(np.ascontiguousarray(waves[i:i + batch])).float()   # [B, T] @ 32k
        w16 = _resample(w)                                                       # [B, T'] @ 16k
        feats = fe([x.numpy() for x in w16], sampling_rate=SR_AST, return_tensors="pt")
        iv = feats["input_values"].to(DEVICE)
        out.append(model(iv).pooler_output.float().cpu())                        # [B, 768]
    return torch.cat(out)
