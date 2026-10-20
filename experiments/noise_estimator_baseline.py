"""The audited statistic against two estimators that do measure noise.

Reviewer 2 asks for a comparison against a theoretically grounded alternative; reviewer 1
asks how the statistic responds to noise imposed at a known ratio. One table answers both.
For every recording the audited statistic and two published noise estimators are computed on
clean input and under Gaussian and background noise at five prescribed ratios:

  WADA               Kim and Stern: for a signal whose magnitude is Gamma-distributed in
                     additive Gaussian noise, the ratio of the arithmetic to the geometric
                     mean of the magnitude is monotone in the signal-to-noise ratio. The
                     lookup table is built here by simulation from that assumption.
  minimum statistics Martin: the noise power spectral density is read from the quietest
                     frames of each frequency bin and the signal power is what remains.

Writes rerun_results/noise_estimator.csv, one row per recording per condition.

    SG_WORKERS=8 python noise_estimator_baseline.py
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
N_FFT, HOP, Q_NOISE = 1024, 512, 0.05
GAMMA_SHAPE, EPS = 0.4, 1e-12
GRID = np.arange(-30.0, 80.1, 1.0)
I_SNR = S.NAMES.index("dynamic_range_db")

IMPOSED = {"gaussian_noise": {1: 20.0, 2: 15.0, 3: 10.0, 4: 5.0, 5: 0.0},
           "background_noise": {1: 18.0, 2: 13.0, 3: 8.0, 4: 3.0, 5: -2.0}}
CATEGORIES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
              "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
CONTINUOUS = {"air_conditioner", "drilling", "engine_idling", "jackhammer", "siren",
              "street_music"}

_W = None
_TABLE = None
_WIN = np.hanning(N_FFT).astype(np.float32)


def _ag_ratio(a):
    a = np.asarray(a, dtype=np.float64) + EPS
    return float(a.mean() / np.exp(np.log(a).mean()))


def build_table(n=200000, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.gamma(GAMMA_SHAPE, 1.0, n) * rng.choice((-1.0, 1.0), n)
    noise = rng.standard_normal(n)
    ps, pn = (sig ** 2).mean(), (noise ** 2).mean()
    tab = np.array([_ag_ratio(np.abs(sig + np.sqrt(ps / (pn * 10 ** (s / 10))) * noise))
                    for s in GRID])
    o = np.argsort(tab)
    return tab[o], GRID[o]


def _init(path):
    global _W, _TABLE
    _W = np.load(path, mmap_mode="r")
    _TABLE = build_table()


def wada_snr_db(x):
    ratios, snrs = _TABLE
    return float(np.interp(_ag_ratio(np.abs(x)), ratios, snrs))


def ms_snr_db(w):
    n = len(w)
    if n < N_FFT:
        w = np.pad(w, (0, N_FFT - n))
        n = len(w)
    frames = np.stack([w[i:i + N_FFT] * _WIN for i in range(0, n - N_FFT + 1, HOP)])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    if power.shape[0] < 4:
        return float("nan")
    noise = np.quantile(power, Q_NOISE, axis=0)
    p_noise = float(noise.sum())
    p_signal = float(power.mean(axis=0).sum()) - p_noise
    if p_noise <= 0.0 or p_signal <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(p_signal / p_noise))


def _chunk(args):
    name, sev, lo, hi = args
    out = np.empty((hi - lo, 3), dtype=np.float32)
    for j, i in enumerate(range(lo, hi)):
        w = np.asarray(_W[i], dtype=np.float32)
        if name is not None:
            w = C.corrupt(w, data.SR, name, sev, seed=SEED)
        out[j, 0] = S.compute(w, data.SR)[I_SNR]
        out[j, 1] = wada_snr_db(w)
        out[j, 2] = ms_snr_db(w)
    return out


QUANTITIES = ("axis_statistic_db", "wada_db", "ms_estimator_db")


def main():
    ds = data.REGISTRY["us8k"]
    wpath, labels, folds = data.cache(ds, CACHE)
    n = len(labels)
    names = [os.path.basename(p) for p in data._index(ds)["path"]]
    workers = int(os.environ.get("SG_WORKERS", "8"))
    cells = [("clean", 0)] + [(c, s) for c in IMPOSED for s in range(1, 6)]
    print(f"{n} recordings, {len(cells)} conditions, {workers} workers", flush=True)

    bounds = [(i, min(i + 512, n)) for i in range(0, n, 512)]
    rows = []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(wpath,)) as pool:
        for cname, sev in cells:
            name = None if cname == "clean" else cname
            got = np.concatenate(list(pool.map(
                _chunk, [(name, sev, lo, hi) for lo, hi in bounds])))
            rows.append(pd.DataFrame({
                "file": names, "classID": labels, "fold": folds,
                "condition": cname, "severity": sev,
                "imposed_snr_db": np.nan if cname == "clean" else IMPOSED[cname][sev],
                "axis_statistic_db": got[:, 0],
                "wada_db": got[:, 1],
                "ms_estimator_db": got[:, 2],
            }))
            print(f"  done {cname}_{sev}", flush=True)

    out = pd.concat(rows, ignore_index=True)
    out["category"] = out.classID.map(lambda i: CATEGORIES[i])
    out["continuous"] = out.category.isin(CONTINUOUS)
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, "noise_estimator.csv")
    out.to_csv(dst, index=False)
    print(f"wrote {dst}  rows {len(out)}", flush=True)

    for cname, table in IMPOSED.items():
        sub = out[out.condition == cname]
        imposed_span = table[1] - table[5]
        print(f"\n=== {cname}: median by prescribed ratio "
              f"({', '.join(f'{table[s]:g}' for s in sorted(table))} dB)")
        for group, mask in (("all", np.ones(len(sub), bool)),
                            ("continuous", sub.continuous.values),
                            ("intermittent", ~sub.continuous.values)):
            g = sub[mask]
            for col in QUANTITIES:
                med = [g[g.severity == s][col].median() for s in sorted(table)]
                span = med[0] - med[-1]
                imp = g.severity.map(table).astype(float).values
                ok = np.isfinite(g[col].values)
                r = np.corrcoef(imp[ok], g[col].values[ok])[0, 1]
                print(f"  {group:12s} {col:18s} " + " ".join(f"{v:7.2f}" for v in med) +
                      f"   span {span:6.2f}/{imposed_span:4.1f} = {span / imposed_span:5.2f}"
                      f"   pearson {r:+.3f}")


if __name__ == "__main__":
    main()
