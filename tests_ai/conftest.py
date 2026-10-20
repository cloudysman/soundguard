"""Harness for the SoundGuard AI / model test suites (graduated: light -> maximum).

Tests the science, not the HTTP layer: preprocessing train/serve parity, metric-formula correctness,
model quality/behaviour on held-out audio, and reproducibility. CPU-forced for determinism and so
several suites run in parallel without GPU VRAM contention.

Fixtures:
    predictor     -> get_predictor() (predictor.py): .available, .classes, predict_wav, apply_slider_corruption
    model_ok      -> bool
    extractor     -> FeatureExtractor (features.py) on CPU: wav_to_features(wav, sr) -> torch [1,512]
    config        -> checkpoints/feature_config.json (dict)
    CLASSES        -> the 10 class names in index order
    rerun_logits  -> (logits[N,10], labels[N]) = concat of LAST-epoch mlp logits over the 10 folds
                     (honest, matches results/dashboard.json). Use for metric / calibration tests, no model needed.
    rerun_best    -> same but BEST-epoch logits (optimistic).
    us8k          -> factory us8k(n_per_class=2, seed=0) -> list[(wav float32 mono, sr, label_idx, class_name)]
                     from the local UrbanSound8K, or None if the dataset is absent (skip then).

Reference implementations to cross-check against (read them):
    sg/backbone_resnet18.py : load()/embed() — the TRAINING-side feature pipeline (parity reference)
    sg/reliability.py, sg/stats.py : reference metric/CI implementations
    metrics.py : the serving metrics under test (softmax/accuracy/nll/brier/ece/reliability_curve/risk_coverage_curve/per_class_accuracy)
See tests_ai/AI_CONTRACT.md.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU: determinism + no GPU VRAM contention. NB "-1" not ""

import glob
import json

import numpy as np
import pytest

# Constants live in helpers_ai.py — see the note there on why they are not in this file.
from helpers_ai import CLASSES  # noqa: E402,F401  (re-exported for convenience)

# Resolve UrbanSound8K across machines (laptop vs PTIT server) — first candidate that has the metadata csv.
_US8K_CANDIDATES = [
    os.environ.get("SG_US8K", ""),
    r"C:\Users\trong\Downloads\us8k\UrbanSound8K",
    os.path.expanduser("~/HieuDT/soundguard/data/UrbanSound8K"),
    "/home/hieudt/HieuDT/soundguard/data/UrbanSound8K",
]
US8K_ROOT = next((p for p in _US8K_CANDIDATES
                  if p and os.path.exists(os.path.join(p, "metadata", "UrbanSound8K.csv"))), None)


@pytest.fixture(scope="session")
def predictor():
    from predictor import get_predictor
    return get_predictor()


@pytest.fixture(scope="session")
def model_ok(predictor):
    return predictor.available


@pytest.fixture(scope="session")
def extractor():
    from features import FeatureExtractor
    return FeatureExtractor("checkpoints/feature_config.json", device="cpu")


@pytest.fixture(scope="session")
def config():
    with open("checkpoints/feature_config.json") as f:
        return json.load(f)


def _load_rerun(key):
    import torch
    logits, labels = [], []
    for f in sorted(glob.glob("rerun_results/mlp_fold*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        logits.append(np.asarray(d[key], dtype=np.float64))
        labels.append(np.asarray(d["labels"]).astype(int))
    if not logits:
        return None
    return np.concatenate(logits), np.concatenate(labels)


@pytest.fixture(scope="session")
def rerun_logits():
    r = _load_rerun("last_logits")
    if r is None:
        pytest.skip("rerun_results/mlp_fold*.pt not present")
    return r


@pytest.fixture(scope="session")
def rerun_best():
    r = _load_rerun("best_logits")
    if r is None:
        pytest.skip("rerun_results/mlp_fold*.pt not present")
    return r


@pytest.fixture(scope="session")
def us8k():
    if US8K_ROOT is None:
        return None
    meta = os.path.join(US8K_ROOT, "metadata", "UrbanSound8K.csv")
    if not os.path.exists(meta):
        return None
    import pandas as pd
    import soundfile as sf
    df = pd.read_csv(meta)
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def load(n_per_class=2, seed=0):
        out = []
        for cls in CLASSES:
            sub = df[df["class"] == cls].sample(n=min(n_per_class, (df["class"] == cls).sum()),
                                                random_state=seed)
            for _, r in sub.iterrows():
                path = os.path.join(US8K_ROOT, "audio", f"fold{r['fold']}", r["slice_file_name"])
                if not os.path.exists(path):
                    continue
                wav, sr = sf.read(path)
                wav = np.asarray(wav, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                out.append((wav, sr, cls_to_idx[cls], cls))
        return out
    return load
