"""Rebuild rerun_results/quality.csv: the three recording-axis statistics per recording.

For every UrbanSound8K recording the table holds the statistics of the clean waveform and of the
same waveform under each of the fifty corruption cells, so the audit of the recording axis can be
reproduced without re-reading the audio.

Columns
    file, classID, fold          identity of the recording
    corruption, severity         the cell (one row per recording per cell)
    snr_clean,  snr_corr         dynamic_range_db: the statistic the policy calls a signal-to-noise
                                 ratio, which is 10*log10(Q90/Q10) of the frame energy
    level_clean, level_corr      level_rms_db
    clip_clean,  clip_corr       clip_frac as a percentage

    python make_quality_table.py
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import corruption as C
from sg import data
from sg import stats as S

CACHE, OUT = "cache", "rerun_results"
SEED = 0
I_SNR = S.NAMES.index("dynamic_range_db")
I_LEVEL = S.NAMES.index("level_rms_db")
I_CLIP = S.NAMES.index("clip_frac")

_W = None


def _init(path):
    global _W
    _W = np.load(path, mmap_mode="r")


def _chunk(args):
    name, sev, lo, hi = args
    out = np.empty((hi - lo, len(S.NAMES)), dtype=np.float32)
    for j, i in enumerate(range(lo, hi)):
        w = np.asarray(_W[i])
        if name is not None:
            w = C.corrupt(w, data.SR, name, sev, seed=SEED)
        out[j] = S.compute(w, data.SR)
    return out


def all_stats(wpath, n, keys, workers):
    bounds = [(i, min(i + 512, n)) for i in range(0, n, 512)]
    st = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(wpath,)) as pool:
        for key in keys:
            name, sev = (None, 0) if key == "clean" else (key.rsplit("_", 1)[0],
                                                          int(key.rsplit("_", 1)[1]))
            st[key] = np.concatenate(list(pool.map(
                _chunk, [(name, sev, lo, hi) for lo, hi in bounds])))
            print(f"  stats {key:24s}", flush=True)
    return st


def main():
    ds = data.REGISTRY["us8k"]
    wpath, labels, folds = data.cache(ds, CACHE)
    n = len(labels)
    names = [os.path.basename(p) for p in data._index(ds)["path"]]
    cells = [f"{c}_{s}" for c in sorted(C.CORRUPTIONS) for s in range(1, 6)]
    workers = int(os.environ.get("SG_WORKERS", "24"))
    print(f"{n} recordings, {len(cells)} cells, {workers} workers", flush=True)

    st = all_stats(wpath, n, ["clean"] + cells, workers)
    clean = st["clean"]

    rows = []
    for key in cells:
        corr_name, sev = key.rsplit("_", 1)
        c = st[key]
        rows.append(pd.DataFrame({
            "file": names,
            "classID": labels,
            "fold": folds,
            "corruption": corr_name,
            "severity": int(sev),
            "snr_clean": clean[:, I_SNR],
            "snr_corr": c[:, I_SNR],
            "level_clean": clean[:, I_LEVEL],
            "level_corr": c[:, I_LEVEL],
            "clip_clean": clean[:, I_CLIP] * 100.0,
            "clip_corr": c[:, I_CLIP] * 100.0,
        }))

    os.makedirs(OUT, exist_ok=True)
    table = pd.concat(rows, ignore_index=True)
    dst = os.path.join(OUT, "quality.csv")
    table.to_csv(dst, index=False)
    print(f"wrote {dst}: {len(table)} rows", flush=True)

    # also keep the clean statistics on their own, one row per recording
    pd.DataFrame({
        "file": names, "classID": labels, "fold": folds,
        "snr": clean[:, I_SNR], "level": clean[:, I_LEVEL], "clip": clean[:, I_CLIP] * 100.0,
    }).to_csv(os.path.join(OUT, "quality_clean.csv"), index=False)
    print("wrote rerun_results/quality_clean.csv", flush=True)


if __name__ == "__main__":
    main()
