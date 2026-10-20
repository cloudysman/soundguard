"""The master table across every dataset and backbone.

Reads every results_paper/*_{cnn14,ast}_headline.csv and the validity table, and prints one table
covering the four questions on each (dataset, backbone) pair, so the paper's central claim can be
read at a glance: which findings reproduce across datasets AND across backbones. Conclusions are
computed from the numbers; none is asserted regardless of them.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

RES = "results_paper/q1up"
ACC_FLOOR = 0.55


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return "%.3f" % v.mean()
    h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return "%.3f±%.3f" % (v.mean(), h)


def analyse(df):
    out = {}
    lad = df[df.q == "ladder"]
    out["acc"] = lad[lad.frac == 1.0].clean_acc.mean()
    out["clean_ece"] = lad[lad.frac == 1.0].clean_ece.mean()
    out["q1_corr"] = float(np.mean([np.corrcoef(s.clean_acc, s.ece_factor)[0, 1]
                                    for _, s in lad.groupby("seed")]))

    rep = df[df.q == "repair"]
    piv = rep.pivot_table(index=["held_out", "fold", "seed"], columns="method", values="auroc")
    base = piv["none"].values
    best, bd, bp = "none", 0.0, 1.0
    for m in ("global_T", "temp_nll", "temp_rank", "detector"):
        if m in piv:
            d, p = (piv[m].values - base).mean() * 100, sps.wilcoxon(piv[m].values, base).pvalue
            if p < 0.05 and d > bd:
                best, bd, bp = m, d, p
    out["q2_repair"] = "none helps" if best == "none" else f"{best} +{bd:.2f} (p={bp:.0e})"

    cf = df[df.q == "conformal"]
    out["q2_conformal_drift"] = cf.coverage_drift.mean() if len(cf) else float("nan")

    tr = df[(df.q == "train") & (df.condition != "clean")]
    pt = tr.pivot_table(index=["held_out", "fold", "seed"], columns="arm", values="auroc")
    dau = (pt["aug_matched"] - pt["clean_only"]).values
    out["q3_gain"], out["q3_p"] = dau.mean() * 100, sps.wilcoxon(dau, np.zeros_like(dau)).pvalue

    mech = df[(df.q == "mechanism") & (df.clean_acc > ACC_FLOOR)]
    g = mech.groupby(["frac", "seed"]).agg(ece=("clean_ece", "mean"),
                                           gain=("auroc_gain", "mean")).reset_index()
    if len(g) > 3:
        out["q4_rho"], out["q4_p"] = sps.spearmanr(g.ece, g.gain)
    else:
        out["q4_rho"], out["q4_p"] = float("nan"), float("nan")
    return out


def main():
    files = sorted(glob.glob(os.path.join(RES, "*_headline.csv")))
    if not files:
        raise SystemExit(f"no headline files in {RES}")

    print("=" * 100)
    print("MASTER TABLE — four questions on each (dataset, backbone) pair")
    print("=" * 100)
    print("  %-16s %6s %8s %8s %-22s %-20s %-16s" %
          ("dataset/backbone", "acc", "cleanECE", "Q1 corr", "Q3 aug (unseen)",
           "Q2 post-hoc repair", "Q4 mechanism"))
    rows = []
    for f in files:
        tag = os.path.basename(f).replace("_headline.csv", "")
        r = analyse(pd.read_csv(f))
        rows.append((tag, r))
        print("  %-16s %6.3f %8.3f %+8.3f  %+.2f (p=%.0e)%3s %-20s rho=%+.2f (p=%.0e)" %
              (tag, r["acc"], r["clean_ece"], r["q1_corr"],
               r["q3_gain"], r["q3_p"], "", r["q2_repair"][:20], r["q4_rho"], r["q4_p"]))

    print("\n  reading guide:")
    print("   Q1 corr > 0  : more accurate -> more fragile calibration (the core claim)")
    print("   Q3 aug  > 0 with small p : corruption-augmented training transfers to unseen corruption")
    print("   Q2 repair 'none helps'  : post-hoc rescaling does not beat doing nothing (us8k regime)")
    print("   Q4 rho  > 0 with small p : clean ECE explains the sign of the repair gain (the mechanism)")

    # validity
    vfiles = sorted(glob.glob(os.path.join(RES, "*_validity.csv")))
    if vfiles:
        v = pd.read_csv(vfiles[0])
        piv = v.groupby(["corruption", "pipeline"]).mean_delta.mean().unstack()
        for p in piv.columns:
            piv[p] = piv[p] / piv[p].max()
        print("\n" + "=" * 100)
        print("C5 VALIDITY — normalised change to the input tensor; < 0.05 is a no-op for that pipeline")
        print("=" * 100)
        print("  %-17s %s" % ("corruption", "  ".join("%-13s" % p for p in piv.columns)))
        for n in piv.index:
            marks = "  ".join(("%.3f NO-OP" % piv.loc[n, p]) if piv.loc[n, p] < 0.05
                              else ("%.3f      " % piv.loc[n, p]) for p in piv.columns)
            print("  %-17s %s" % (n, marks))


if __name__ == "__main__":
    main()
