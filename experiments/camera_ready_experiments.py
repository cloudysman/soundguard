"""The experiments the reviewers asked for.

E1  the comparison both reviewers named: the deployed policy, which accepts only when the
    confidence and all three waveform statistics pass, against the confidence alone at the
    same coverage, on clean input and under each corruption condition.
E2  the response of the recording-axis statistic to noise imposed at a known ratio, which
    tests whether a quantity that is not an SNR might still track one.
E3  the three gates separately: how often each fires and how well each detects an error,
    since the audit removes all three but only analyses the first.

Reads  rerun_results/quality.csv        statistics per recording per condition
       rerun_results/clean_conf.npz     per-recording confidence on clean input
       rerun_results/grid_conf.npz      per-recording confidence under each condition
Writes rerun_results/camera_ready.json  every number quoted in the revision

    python camera_ready_experiments.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

OUT = "rerun_results"

# the shipped configuration, as listed in Table 1 of the paper
SNR_REJECT, SNR_UNCERTAIN = 2.0, 4.0
LEVEL_REJECT, LEVEL_UNCERTAIN = -50.0, -40.0
CLIP_REJECT, CLIP_UNCERTAIN = 30.0, 12.0
CONF_REJECT, CONF_UNCERTAIN = 0.45, 0.70

IMPOSED = {"gaussian_noise": {1: 20, 2: 15, 3: 10, 4: 5, 5: 0},
           "background_noise": {1: 18, 2: 13, 3: 8, 4: 3, 5: -2}}
CATEGORIES = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
              "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
CONTINUOUS = {"air_conditioner", "drilling", "engine_idling", "jackhammer", "siren",
              "street_music"}


def selective_risk(correct, accept):
    """Error rate among the accepted recordings; nan when nothing is accepted."""
    k = int(accept.sum())
    return float(1.0 - correct[accept].mean()) if k else float("nan")


def risk_at_coverage(conf, correct, coverage):
    """Risk of the confidence alone when it accepts exactly the given fraction."""
    if not np.isfinite(coverage) or coverage <= 0:
        return float("nan")
    tau = float(np.quantile(conf, 1.0 - coverage))
    return selective_risk(correct, conf >= tau)


def axis_accept(snr, level, clip, strict):
    """The recording axis passes a recording. strict uses the reject thresholds."""
    if strict:
        return (snr >= SNR_REJECT) & (level >= LEVEL_REJECT) & (clip <= CLIP_REJECT)
    return (snr >= SNR_UNCERTAIN) & (level >= LEVEL_UNCERTAIN) & (clip <= CLIP_UNCERTAIN)


def e1_combined_versus_confidence(q, clean, grid):
    """The deployed conjunction against the confidence alone at a matched coverage."""
    rows = []
    one = q.drop_duplicates("file").sort_values("file_order")
    for strict in (True, False):
        conf_t = CONF_REJECT if strict else CONF_UNCERTAIN
        # clean
        acc_axis = axis_accept(one.snr_clean.values, one.level_clean.values,
                               one.clip_clean.values, strict)
        acc_conf = clean["conf"] >= conf_t
        both = acc_axis & acc_conf
        cov = float(both.mean())
        rows.append({
            "condition": "clean", "severity": 0, "gate": "reject" if strict else "uncertain",
            "coverage_combined": cov,
            "risk_combined": selective_risk(clean["correct"], both),
            "risk_confidence_matched": risk_at_coverage(clean["conf"], clean["correct"], cov),
            "risk_axis_matched_proxy": selective_risk(clean["correct"], acc_axis),
            "coverage_axis_only": float(acc_axis.mean()),
        })
        # each corruption condition
        for (name, sev), g in q.groupby(["corruption", "severity"], sort=True):
            key = f"{name}_{sev}"
            if f"{key}_conf" not in grid:
                continue
            g = g.sort_values("file_order")
            conf = grid[f"{key}_conf"]
            correct = grid[f"{key}_correct"]
            acc_axis = axis_accept(g.snr_corr.values, g.level_corr.values, g.clip_corr.values,
                                   strict)
            both = acc_axis & (conf >= conf_t)
            cov = float(both.mean())
            rows.append({
                "condition": name, "severity": int(sev),
                "gate": "reject" if strict else "uncertain",
                "coverage_combined": cov,
                "risk_combined": selective_risk(correct, both),
                "risk_confidence_matched": risk_at_coverage(conf, correct, cov),
                "risk_axis_matched_proxy": selective_risk(correct, acc_axis),
                "coverage_axis_only": float(acc_axis.mean()),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(os.path.join(OUT, "e1_combined_vs_confidence.csv"), index=False)

    summary = {}
    for gate in ("reject", "uncertain"):
        f = frame[frame.gate == gate]
        clean_row = f[f.condition == "clean"].iloc[0]
        corr = f[f.condition != "clean"]
        delta = corr.risk_combined - corr.risk_confidence_matched
        summary[gate] = {
            "clean": {k: float(clean_row[k]) for k in
                      ("coverage_combined", "risk_combined", "risk_confidence_matched")},
            "corrupted_mean": {
                "coverage_combined": float(corr.coverage_combined.mean()),
                "risk_combined": float(corr.risk_combined.mean()),
                "risk_confidence_matched": float(corr.risk_confidence_matched.mean()),
            },
            "conditions_where_combined_is_worse": int((delta > 0).sum()),
            "conditions_total": int(len(corr)),
            "mean_risk_increase_from_the_axis": float(delta.mean()),
        }
    return summary


def e2_imposed_ratio(q):
    """How the statistic moves when noise is mixed in at a known ratio."""
    out = {}
    for name, table in IMPOSED.items():
        sub = q[q.corruption == name]
        if sub.empty:
            continue
        rec = {}
        for group, mask in (("continuous", sub.continuous), ("intermittent", ~sub.continuous)):
            med = {table[s]: float(sub[mask & (sub.severity == s)].snr_corr.median())
                   for s in sorted(table)}
            span_imposed = table[1] - table[5]
            span_seen = med[table[1]] - med[table[5]]
            rec[group] = {
                "median_statistic_by_imposed_ratio": med,
                "imposed_span_db": float(span_imposed),
                "observed_span_db": float(span_seen),
                "fraction_of_a_true_estimate": float(span_seen / span_imposed),
            }
        # correlation between the imposed ratio and the statistic, over all recordings
        imposed = sub.severity.map(table).astype(float).values
        rec["pearson_imposed_vs_statistic"] = float(np.corrcoef(imposed, sub.snr_corr.values)[0, 1])
        out[name] = rec
    return out


def e3_three_gates(q, clean, grid):
    """Each gate on its own: how often it fires, and whether it detects errors."""
    one = q.drop_duplicates("file").sort_values("file_order")

    def auroc(score, correct):
        """Probability that an incorrect prediction scores below a correct one."""
        order = np.argsort(score, kind="mergesort")
        s, c = score[order], correct[order]
        ranks = np.empty(len(s), dtype=np.float64)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            ranks[i:j + 1] = 0.5 * (i + j) + 1.0
            i = j + 1
        npos, nneg = int(c.sum()), int((~c).sum())
        if npos == 0 or nneg == 0:
            return float("nan")
        return float((ranks[c].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))

    gates = {
        "snr": (one.snr_clean.values, SNR_UNCERTAIN, "below"),
        "level": (one.level_clean.values, LEVEL_UNCERTAIN, "below"),
        "clipping": (one.clip_clean.values, CLIP_UNCERTAIN, "above"),
    }
    out = {}
    for gname, (vals, thr, side) in gates.items():
        fires = vals < thr if side == "below" else vals > thr
        score = vals if side == "below" else -vals
        out[gname] = {
            "flagging_rate_clean": float(fires.mean()),
            "error_detection_auroc": auroc(score, clean["correct"]),
        }
    axis = ~axis_accept(one.snr_clean.values, one.level_clean.values, one.clip_clean.values,
                        False)
    out["axis_combined"] = {"flagging_rate_clean": float(axis.mean())}
    out["confidence"] = {"error_detection_auroc": auroc(clean["conf"], clean["correct"])}
    return out


def main():
    q = pd.read_csv(os.path.join(OUT, "quality.csv"))
    order = {f: i for i, f in enumerate(pd.unique(q.file))}
    q["file_order"] = q.file.map(order)
    q["category"] = q.classID.map(lambda i: CATEGORIES[i])
    q["continuous"] = q.category.isin(CONTINUOUS)

    clean_npz = np.load(os.path.join(OUT, "clean_conf.npz"))
    clean = {"conf": clean_npz["conf"].astype(np.float64),
             "correct": clean_npz["correct"].astype(bool)}
    grid = dict(np.load(os.path.join(OUT, "grid_conf.npz")))
    grid = {k: (v.astype(np.float64) if v.dtype != bool else v) for k, v in grid.items()}

    result = {
        "e1_combined_versus_confidence": e1_combined_versus_confidence(q, clean, grid),
        "e2_imposed_ratio": e2_imposed_ratio(q),
        "e3_three_gates": e3_three_gates(q, clean, grid),
    }
    dst = os.path.join(OUT, "camera_ready.json")
    with open(dst, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
