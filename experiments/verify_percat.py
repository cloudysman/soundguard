"""Re-derive Table 2 of the paper: the per-category flagging rate of the recording axis
on clean UrbanSound8K, beside the per-category accuracy, and the axis AUROC."""
import numpy as np
import pandas as pd

SNR_U, LEVEL_U, CLIP_U = 4.0, -40.0, 12.0
CATEGORIES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
              "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

q = pd.read_csv("rerun_results/quality.csv")
one = q.drop_duplicates("file").reset_index(drop=True)
print(f"recordings {len(one)}  columns {list(one.columns)}")

flagged = ~((one.snr_clean >= SNR_U) & (one.level_clean >= LEVEL_U) & (one.clip_clean <= CLIP_U))
print(f"\noverall flagging rate {100*flagged.mean():.2f}%")
for g, thr, side in (("snr", SNR_U, "<"), ("level", LEVEL_U, "<"), ("clip", CLIP_U, ">")):
    v = one[f"{g}_clean"]
    f = v < thr if side == "<" else v > thr
    print(f"  gate {g:6s} alone {100*f.mean():.2f}%")

z = np.load("rerun_results/clean_conf.npz")
correct, label, conf = z["correct"].astype(bool), z["label"], z["conf"].astype(float)
print(f"\naccuracy {100*correct.mean():.2f}%  mean confidence {100*conf.mean():.2f}%")

rows = []
for i, name in enumerate(CATEGORIES):
    m = one.classID.values == i
    rows.append({"category": name.replace("_", " "),
                 "flagging_pct": 100 * flagged.values[m].mean(),
                 "accuracy_pct": 100 * correct[label == i].mean(),
                 "n": int(m.sum())})
t = pd.DataFrame(rows).sort_values("flagging_pct", ascending=False)
print("\n" + t.round(1).to_string(index=False))


def auroc(score, ok):
    o = np.argsort(score, kind="mergesort")
    s, c = score[o], ok[o]
    r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    p, n = int(c.sum()), int((~c).sum())
    return float("nan") if p == 0 or n == 0 else float((r[c].sum() - p * (p + 1) / 2) / (p * n))


acc_binary = (~flagged.values).astype(float)
print(f"\naxis AUROC, binary accept score  {auroc(acc_binary, correct):.4f}")
print(f"axis AUROC, snr statistic alone  {auroc(one.snr_clean.values, correct):.4f}")
print(f"confidence AUROC                 {auroc(conf, correct):.4f}")
wc = [auroc(acc_binary[label == i], correct[label == i]) for i in range(10)]
wc = [v for v in wc if np.isfinite(v)]
print(f"within-category axis AUROC: median {np.median(wc):.3f} range {min(wc):.3f}..{max(wc):.3f}")

fl, ac = t.flagging_pct.values, t.accuracy_pct.values
pear = np.corrcoef(fl, ac)[0, 1]
rf, ra = np.argsort(np.argsort(fl)), np.argsort(np.argsort(ac))
print(f"\nflagging vs accuracy: Pearson {pear:.3f}  Spearman {np.corrcoef(rf, ra)[0,1]:.3f}")
