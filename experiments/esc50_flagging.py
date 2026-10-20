"""How often the recording axis fires on clean ESC-50, with the same eight thresholds."""
import numpy as np
import pandas as pd

from sg import data
from sg import stats as S

SNR_U, LEVEL_U, CLIP_U = 4.0, -40.0, 12.0
I_SNR = S.NAMES.index("dynamic_range_db")
I_LEVEL = S.NAMES.index("level_rms_db")
I_CLIP = S.NAMES.index("clip_frac")

ds = data.REGISTRY["esc50"]
wpath, labels, folds = data.cache(ds, "cache")
W = np.load(wpath, mmap_mode="r")
labels = np.asarray(labels)
print(f"esc50 recordings {len(labels)}  categories {len(set(labels.tolist()))}", flush=True)

st = np.empty((len(labels), len(S.NAMES)), dtype=np.float64)
for i in range(len(labels)):
    st[i] = S.compute(np.asarray(W[i], dtype=np.float32), data.SR)
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}", flush=True)

snr, level, clip = st[:, I_SNR], st[:, I_LEVEL], st[:, I_CLIP] * 100.0
gate_snr, gate_level, gate_clip = snr < SNR_U, level < LEVEL_U, clip > CLIP_U
axis = gate_snr | gate_level | gate_clip

print(f"\naxis flags            {100 * axis.mean():.2f}%")
print(f"  signal-to-noise gate {100 * gate_snr.mean():.2f}%")
print(f"  level gate           {100 * gate_level.mean():.2f}%")
print(f"  clipping gate        {100 * gate_clip.mean():.2f}%")

per = pd.DataFrame({"label": labels, "axis": axis, "snr": gate_snr})
rate = per.groupby("label")["axis"].mean() * 100
rate_snr = per.groupby("label")["snr"].mean() * 100
print(f"\ncategories never flagged by the axis      {int((rate == 0).sum())} of {len(rate)}")
print(f"categories never flagged by the snr gate  {int((rate_snr == 0).sum())} of {len(rate_snr)}")
print(f"highest category rate {rate.max():.1f}%  median {rate.median():.1f}%")
