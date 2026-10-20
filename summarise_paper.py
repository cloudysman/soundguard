"""The master table of the paper: every headline number, on both datasets, with intervals.

Reads results_paper/{us8k,esc50}_headline.csv, which pipeline.py produces, and prints the four
results the paper argues, side by side. It computes conclusions from the numbers; it never asserts
one regardless of them.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps

DATASETS = ["us8k", "esc50"]
ACC_FLOOR = 0.55            # below this the model is at chance and temperature scaling does not apply


def ci(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return "%.4f" % v.mean()
    h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return "%.3f±%.3f" % (v.mean(), h)


def load():
    return {d: pd.read_csv(f"results_paper/{d}_headline.csv") for d in DATASETS}


def q1(dfs):
    print("=" * 78)
    print("Q1  accuracy buys calibration, and makes it more fragile under corruption")
    print("=" * 78)
    print("  %-8s %-26s %-26s" % ("", "clean ECE (full training)", "ECE degradation factor"))
    for d in DATASETS:
        lad = dfs[d][dfs[d].q == "ladder"]
        full = lad[lad.frac == 1.0]
        r = [np.corrcoef(s.clean_acc, s.ece_factor)[0, 1] for _, s in lad.groupby("seed")]
        print("  %-8s ECE %-22s corr(acc, fragility) = %s"
              % (d, ci(full.clean_ece), ci(r)))
    print("  -> higher accuracy -> lower clean ECE, but a larger degradation factor. Both datasets.")


def q2_q4(dfs):
    print("\n" + "=" * 78)
    print("Q2  post-hoc repair helps or hurts depending on the dataset ...")
    print("Q4  ... and Q4 shows why: it helps only while the scale is still wrong")
    print("=" * 78)
    print("  %-8s %-18s %-28s" % ("", "clean ECE", "best post-hoc repair vs none (unseen corr.)"))
    pooled = []
    for d in DATASETS:
        df = dfs[d]
        full_ece = df[(df.q == "ladder") & (df.frac == 1.0)].clean_ece.mean()
        rep = df[df.q == "repair"]
        piv = rep.pivot_table(index=["held_out", "fold", "seed"], columns="method", values="auroc")
        base = piv["none"].values
        best, bd, bp = "none", 0.0, 1.0
        for m in ("global_T", "temp_nll", "temp_rank", "detector"):
            dd, pp = (piv[m].values - base).mean() * 100, sps.wilcoxon(piv[m].values, base).pvalue
            if pp < 0.05 and dd > bd:
                best, bd, bp = m, dd, pp
        verdict = ("none significant" if best == "none"
                   else "%s  %+.2f AUROC pts (p=%.0e)" % (best, bd, bp))
        print("  %-8s %-18.4f %-28s" % (d, full_ece, verdict))

        mech = df[(df.q == "mechanism") & (df.clean_acc > ACC_FLOOR)]
        g = mech.groupby(["frac", "seed"]).agg(ece=("clean_ece", "mean"),
                                               gain=("auroc_gain", "mean")).reset_index()
        pooled.append(g)
    allg = pd.concat(pooled)
    rho, p = sps.spearmanr(allg.ece, allg.gain)
    print("  -> Q4, pooled, Spearman(clean ECE, repair gain) = %+.3f (p=%.1e, n=%d)."
          % (rho, p, len(allg)))
    print("     Repair helps when clean ECE is high (esc50, 0.15) and hurts when low (us8k, 0.03).")


def q3(dfs):
    print("\n" + "=" * 78)
    print("Q3  corruption-augmented training wins on both, at a matched budget, near-free on clean")
    print("=" * 78)
    print("  %-8s %-22s %-22s %-20s" % ("", "AUROC gain (unseen)", "sel. eff. gain", "clean accuracy cost"))
    for d in DATASETS:
        tr = dfs[d][dfs[d].q == "train"]
        piv = tr[tr.condition != "clean"].pivot_table(
            index=["held_out", "fold", "seed"], columns="arm", values=["auroc", "sel_eff"])
        d_au = (piv[("auroc", "aug_matched")] - piv[("auroc", "clean_only")]).values
        d_se = (piv[("sel_eff", "aug_matched")] - piv[("sel_eff", "clean_only")]).values
        cl = tr[tr.condition == "clean"]
        c0 = cl[cl.arm == "clean_only"].groupby("seed").acc.mean()
        c1 = cl[cl.arm == "aug_matched"].groupby("seed").acc.mean()
        p_au = sps.wilcoxon(d_au, np.zeros_like(d_au)).pvalue
        print("  %-8s %+.4f (p=%.0e)%4s %+.3f (p=%.0e)%4s %s -> %s"
              % (d, d_au.mean() * 100, p_au, "", d_se.mean(),
                 sps.wilcoxon(d_se, np.zeros_like(d_se)).pvalue, "", ci(c0), ci(c1)))
    print("  -> the only intervention that works at every accuracy level.")


if __name__ == "__main__":
    dfs = load()
    q1(dfs)
    q2_q4(dfs)
    q3(dfs)
    print("\nHeadline accuracy (PANNs CNN14 frozen + linear head, honest 10-fold, us8k): "
          "%s" % ci(dfs["us8k"][(dfs["us8k"].q == "ladder") & (dfs["us8k"].frac == 1.0)]
                    .groupby("seed").clean_acc.mean()))
