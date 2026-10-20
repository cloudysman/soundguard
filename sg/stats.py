"""Front-end statistics of a waveform.

These are the quantities that the preprocessing of an ImageNet-transfer pipeline throws away.
The per-recording min-max step rescales every recording by its own minimum and maximum, so the
absolute level and the absolute noise floor never reach the network. A classifier trained that
way is invariant to gain by construction, which is desirable for the PREDICTION and disastrous
for the CONFIDENCE: the model cannot notice that its input has been degraded.

The statistics are computed from the raw waveform, before any preprocessing, and are therefore
available even to a system whose features have discarded them. They are cheap: one pass over the
samples plus one short-time Fourier transform.
"""
from __future__ import annotations

import numpy as np

FRAME_MS = 25
EPS = 1e-12

NAMES = [
    "level_rms_db",      # root-mean-square level, decibels relative to full scale
    "level_peak_db",     # peak level, decibels relative to full scale
    "noise_floor_db",    # energy of the quietest decile of frames, in decibels
    "dynamic_range_db",  # gap between the loudest and the quietest decile of frames
    "crest_db",          # peak minus root-mean-square: how impulsive the recording is
    "flatness",          # spectral flatness: 1 for white noise, near 0 for a tone
    "clip_frac",         # fraction of samples at or above full scale
    "zero_frac",         # fraction of samples that are exactly zero (padding, dropout, masking)
    "hf_ratio",          # fraction of the spectral energy above a quarter of the Nyquist rate
    "centroid_norm",     # spectral centroid, divided by the Nyquist rate
]


def _frame_energy(x: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * FRAME_MS / 1000))
    k = len(x) // n
    if k < 2:
        return np.array([float(np.mean(x ** 2)) + EPS])
    return np.mean(x[: k * n].reshape(k, n) ** 2, axis=1) + EPS


def compute(wav: np.ndarray, sr: int) -> np.ndarray:
    """Return the statistics of one waveform, in the order of NAMES."""
    x = np.asarray(wav, dtype=np.float64)
    e = _frame_energy(x, sr)

    rms = float(np.sqrt(np.mean(x ** 2) + EPS))
    peak = float(np.max(np.abs(x)) + EPS)
    lo = float(np.percentile(e, 10))
    hi = float(np.percentile(e, 90))

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2 + EPS
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    total = float(spec.sum())

    return np.array([
        20.0 * np.log10(rms),                                        # level_rms_db
        20.0 * np.log10(peak),                                       # level_peak_db
        10.0 * np.log10(lo),                                         # noise_floor_db
        10.0 * np.log10(hi / lo),                                    # dynamic_range_db
        20.0 * np.log10(peak / rms),                                 # crest_db
        float(np.exp(np.mean(np.log(spec))) / np.mean(spec)),        # flatness
        float(np.mean(np.abs(x) >= 0.99)),                           # clip_frac
        float(np.mean(x == 0.0)),                                    # zero_frac
        float(spec[freqs > sr / 8.0].sum() / total),                 # hf_ratio
        float((spec * freqs).sum() / total / (sr / 2.0)),            # centroid_norm
    ], dtype=np.float32)


def compute_many(waves: np.ndarray, sr: int) -> np.ndarray:
    """Statistics for a batch of waveforms: returns [N, len(NAMES)]."""
    return np.stack([compute(w, sr) for w in waves])


class Standardiser:
    """Fit the mean and the standard deviation on the training folds only, never on the test fold."""

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, s: np.ndarray) -> "Standardiser":
        self.mu = s.mean(axis=0)
        self.sd = s.std(axis=0) + 1e-6
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        return ((s - self.mu) / self.sd).astype(np.float32)
