"""The two datasets, behind one interface.

Both are distributed with a fixed partition into folds, and both ask that the partition be used
rather than a random one, because clips cut from the same source recording are placed in the same
fold. A random partition would put related clips on both sides of the split and would inflate the
measured accuracy. Neither dataset is re-partitioned anywhere in this work.

Everything is decoded once, to 32 kHz mono, and cached as a memory-mappable array. 32 kHz is what
the PANNs backbone was trained at; feeding it 16 kHz audio would throw away a third of the band it
expects.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

SR = 32000


@dataclass(frozen=True)
class Dataset:
    name: str
    n_classes: int
    n_folds: int
    clip_s: int
    root: str

    @property
    def n_samples(self) -> int:
        return SR * self.clip_s


US8K = Dataset("us8k", 10, 10, 4, "data/UrbanSound8K")
ESC50 = Dataset("esc50", 50, 5, 5, "data/ESC-50")
# TAU has one official location-independent split, so it uses a single fold: fold 1 is the training
# set, fold 2 the evaluation set. clip_s is 4 to match the other two; the 10 s clips are cropped.
# The zips extract into a versioned subdirectory, which is the actual root.
TAU = Dataset("tau", 10, 2, 4, "data/tau/TAU-urban-acoustic-scenes-2019-development")

REGISTRY = {d.name: d for d in (US8K, ESC50, TAU)}


def _index(ds: Dataset) -> pd.DataFrame:
    """One row per recording: path, label, fold. Nothing dataset-specific escapes this function."""
    if ds.name == "us8k":
        df = pd.read_csv(os.path.join(ds.root, "metadata", "UrbanSound8K.csv"))
        return pd.DataFrame({
            "path": [os.path.join(ds.root, "audio", f"fold{f}", n)
                     for f, n in zip(df["fold"], df["slice_file_name"])],
            "label": df["classID"].astype(int).values,
            "fold": df["fold"].astype(int).values,
        })

    if ds.name == "esc50":
        df = pd.read_csv(os.path.join(ds.root, "meta", "esc50.csv"))
        return pd.DataFrame({
            "path": [os.path.join(ds.root, "audio", n) for n in df["filename"]],
            "label": df["target"].astype(int).values,
            "fold": df["fold"].astype(int).values,
        })

    if ds.name == "tau":
        # the official split: fold1_train.csv is the training set, fold1_evaluate.csv the test set.
        # A single split, not cross-validation, so map train -> fold 1 and evaluate -> fold 2. The
        # random-fold trap is avoided by using the location-independent official split verbatim.
        setup = os.path.join(ds.root, "evaluation_setup")
        tr = pd.read_csv(os.path.join(setup, "fold1_train.csv"), sep="\t")
        ev = pd.read_csv(os.path.join(setup, "fold1_evaluate.csv"), sep="\t")
        classes = sorted(pd.concat([tr, ev])["scene_label"].unique())
        cmap = {c: i for i, c in enumerate(classes)}
        rows = []
        for split, fold in ((tr, 1), (ev, 2)):
            for _, r in split.iterrows():
                rows.append((os.path.join(ds.root, r["filename"]), cmap[r["scene_label"]], fold))
        out = pd.DataFrame(rows, columns=["path", "label", "fold"])
        return out

    raise KeyError(ds.name)


def cache(ds: Dataset, cache_dir: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Decode once. Returns the path of the waveform array, the labels and the folds."""
    import librosa
    import soundfile as sf

    os.makedirs(cache_dir, exist_ok=True)
    wp = os.path.join(cache_dir, f"{ds.name}_waves.npy")
    mp = os.path.join(cache_dir, f"{ds.name}_meta.npz")

    if not (os.path.exists(wp) and os.path.exists(mp)):
        idx = _index(ds)
        arr = np.zeros((len(idx), ds.n_samples), dtype=np.float32)
        for i, p in enumerate(idx["path"]):
            w, sr = sf.read(p, dtype="float32", always_2d=True)
            w = w.mean(axis=1)
            if sr != SR:
                w = librosa.resample(w, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
            arr[i, : min(ds.n_samples, len(w))] = w[: ds.n_samples]
        np.save(wp, arr)
        np.savez(mp, labels=idx["label"].values, folds=idx["fold"].values)

    m = np.load(mp)
    return wp, m["labels"].astype(int), m["folds"].astype(int)
