"""The recording axis against the confidence threshold at the coverage the axis itself sets.

e1 matched the confidence to the coverage of the combined rule. This matches it to the
coverage of the recording axis alone, which is the comparison the paper quotes.
"""
import json

import numpy as np
import pandas as pd

OUT = "rerun_results"
SNR_R, SNR_U = 2.0, 4.0
LEVEL_R, LEVEL_U = -50.0, -40.0
CLIP_R, CLIP_U = 30.0, 12.0


def selective_risk(correct, accept):
    k = int(accept.sum())
    return float(1.0 - correct[accept].mean()) if k else float("nan")


def risk_at_coverage(conf, correct, coverage):
    tau = float(np.quantile(conf, 1.0 - coverage))
    return selective_risk(correct, conf >= tau)


def axis_accept(snr, level, clip, strict):
    if strict:
        return (snr >= SNR_R) & (level >= LEVEL_R) & (clip <= CLIP_R)
    return (snr >= SNR_U) & (level >= LEVEL_U) & (clip <= CLIP_U)


q = pd.read_csv(f"{OUT}/quality.csv")
order = {f: i for i, f in enumerate(pd.unique(q.file))}
q["file_order"] = q.file.map(order)
one = q.drop_duplicates("file").sort_values("file_order")

z = np.load(f"{OUT}/clean_conf.npz")
clean_conf, clean_ok = z["conf"].astype(np.float64), z["correct"].astype(bool)
grid = dict(np.load(f"{OUT}/grid_conf.npz"))

res = {}
for strict in (False, True):
    gate = "reject" if strict else "uncertain"
    a = axis_accept(one.snr_clean.values, one.level_clean.values, one.clip_clean.values, strict)
    cov = float(a.mean())
    row = {"clean": {"coverage_axis": cov,
                     "risk_axis": selective_risk(clean_ok, a),
                     "risk_confidence_at_that_coverage":
                         risk_at_coverage(clean_conf, clean_ok, cov)}}
    rows = []
    for (name, sev), g in q.groupby(["corruption", "severity"], sort=True):
        key = f"{name}_{sev}"
        if f"{key}_conf" not in grid:
            continue
        g = g.sort_values("file_order")
        conf = grid[f"{key}_conf"].astype(np.float64)
        ok = grid[f"{key}_correct"].astype(bool)
        aa = axis_accept(g.snr_corr.values, g.level_corr.values, g.clip_corr.values, strict)
        c = float(aa.mean())
        rows.append((c, selective_risk(ok, aa), risk_at_coverage(conf, ok, c)))
    r = np.array(rows)
    row["corrupted_mean"] = {"coverage_axis": float(r[:, 0].mean()),
                             "risk_axis": float(r[:, 1].mean()),
                             "risk_confidence_at_that_coverage": float(r[:, 2].mean())}
    row["conditions_axis_worse"] = int((r[:, 1] > r[:, 2]).sum())
    row["conditions_total"] = int(len(r))
    res[gate] = row

print(json.dumps(res, indent=2))
with open(f"{OUT}/e1b_axis_vs_confidence.json", "w") as fh:
    json.dump(res, fh, indent=2)
print(f"\nwrote {OUT}/e1b_axis_vs_confidence.json")
