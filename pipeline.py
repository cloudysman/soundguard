"""The whole study, on either dataset, with confidence intervals.

    python pipeline.py --dataset us8k  --stage embed
    python pipeline.py --dataset us8k  --stage headline --seeds 5
    python pipeline.py --dataset esc50 --stage embed
    python pipeline.py --dataset esc50 --stage headline --seeds 5

The headline stage answers the three questions the paper asks, on one dataset, over several seeds:

  1. how much reliability does accuracy buy?          (the ladder: subsample the training folds)
  2. can the confidence be repaired after the fact?   (four post-hoc methods, leave-one-out)
  3. can it be trained in instead?                    (corruption-augmented training, leave-one-out)

Every comparison is paired: the same fold, the same held-out corruption, the same seed. The
intervals reported are over the seeds, and the tests are paired Wilcoxon signed-rank over the
(corruption, fold, seed) triples. Nothing is compared across different amounts of training data
unless the row says so.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as sps

import corruption as C
from sg import backbone, backbone_ast, backbone_resnet18, conformal, data
from sg import reliability as R
from sg import stats as S
from sg.tempering import (FailureDetector, GlobalTemperature, RankedScaleRestoredConfidence,
                          ScaleRestoredConfidence, check_accuracy_is_unchanged)

DEVICE = backbone.DEVICE
CACHE, OUT = "cache", "results_paper"
CELLS = [(n, s) for n in sorted(C.CORRUPTIONS) for s in range(1, 6)]
EPOCHS, BS, LR, WD = 50, 128, 1e-3, 1e-4


# --------------------------------------------------------------------------- backbones
# Both modules expose the same load()/embed(handle, waves)->[N, dim] contract, so a backbone is
# just a name here and the rest of the pipeline never branches on which one is in use.
BACKBONES = {"cnn14": backbone, "ast": backbone_ast, "resnet18": backbone_resnet18}


# --------------------------------------------------------------------------- embedding
def stage_embed(ds: data.Dataset, bb: str) -> None:
    """Decode once, then embed the clean recordings and all fifty corruption cells."""
    os.makedirs(CACHE, exist_ok=True)
    wp, y, folds = data.cache(ds, CACHE)
    waves = np.load(wp, mmap_mode="r")
    mod = BACKBONES[bb]
    handle = mod.load()
    print(f"  {ds.name} / {bb}: {len(y)} recordings, {ds.n_classes} classes, {ds.n_folds} folds",
          flush=True)

    todo = [("clean", None, 0)] + [(f"{n}_{s}", n, s) for n, s in CELLS]
    for key, name, sev in todo:
        dst = os.path.join(CACHE, f"{ds.name}_{bb}_emb_{key}.pt")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        w = np.asarray(waves) if name is None else np.stack(
            [C.corrupt(np.asarray(waves[i]), data.SR, name, sev, seed=0) for i in range(len(y))])
        torch.save(mod.embed(handle, w), dst)
        print(f"  {key:20s} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[embed] {ds.name} / {bb} done")


def _load_emb(ds: data.Dataset, bb: str, to_device: bool = False) -> dict:
    keys = ["clean"] + [f"{n}_{s}" for n, s in CELLS]
    emb = {}
    for k in keys:
        p = os.path.join(CACHE, f"{ds.name}_{bb}_emb_{k}.pt")
        if os.path.exists(p):
            e = torch.load(p, weights_only=False)
            emb[k] = e.to(DEVICE) if to_device else e
    return emb


# --------------------------------------------------------------------------- head
def train_head(x: torch.Tensor, t: torch.Tensor, dim: int, n_cls: int, seed: int) -> nn.Module:
    """A linear probe. The features live on the device already; nothing crosses the bus per step."""
    torch.manual_seed(seed)
    x, t = x.to(DEVICE), t.to(DEVICE)
    h = nn.Linear(dim, n_cls).to(DEVICE)
    opt = optim.AdamW(h.parameters(), lr=LR, weight_decay=WD)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    n = len(t)

    # The batch must not exceed the training set. Dropping the last partial batch, as is usual,
    # silently performs ZERO steps whenever the set is smaller than one batch — the head then keeps
    # its random initialisation and scores at chance, which is exactly what the low rungs of the
    # ladder on ESC-50 did before this line existed.
    bs = min(BS, n)
    if bs < 2:
        raise ValueError(f"cannot train a head on {n} rows")

    for _ in range(EPOCHS):
        h.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            crit(h(x[idx]), t[idx]).backward()
            opt.step()
        sch.step()

    return h.eval()


@torch.no_grad()
def score(h: nn.Module, e: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    return h(e[mask].to(DEVICE)).cpu().numpy()


FRACS = (0.02, 0.05, 0.10, 0.25, 0.50, 1.00)


# --------------------------------------------------------------------------- Q1 and Q4 together
def q1q4_ladder(ds, emb, y, folds, seed, rows):
    """One pass up the accuracy ladder, answering both Q1 and Q4.

    Q1 asks how much reliability accuracy buys. Q4 asks whether a global temperature helps or hurts
    at each level of accuracy, to explain why post-hoc repair helped on one dataset and hurt on the
    other. Both need the SAME head, trained on the SAME clean subsample at each rung, so the head is
    trained once and both questions read its logits. This halves the ladder cost and removes the
    risk of the two questions silently using different heads.
    """
    dim = emb["clean"].shape[1]
    rng = np.random.default_rng(seed)
    holds = sorted(C.CORRUPTIONS)[:3]      # three held-out corruptions are enough for the sign test

    for frac in FRACS:
        lg = {k: np.zeros((len(y), ds.n_classes), dtype=np.float32) for k in emb}
        for vf in range(1, ds.n_folds + 1):
            va = folds == vf
            tr = np.where(~va)[0]
            keep = rng.choice(tr, max(64, int(len(tr) * frac)), replace=False)
            h = train_head(emb["clean"][keep], torch.from_numpy(y[keep]).long(),
                           dim, ds.n_classes, seed)
            for k, e in emb.items():
                lg[k][va] = score(h, e, va)

        # ---- Q1 ----
        clean = R.report(lg["clean"], y)
        cells = [R.report(lg[k], y) for k in emb if k != "clean"]
        grid = {k: float(np.mean([c[k] for c in cells])) for k in clean}
        rows.append({"q": "ladder", "seed": seed, "frac": frac,
                     "clean_acc": clean["acc"], "clean_ece": clean["ece"],
                     "clean_auroc": clean["auroc"], "grid_acc": grid["acc"],
                     "grid_ece": grid["ece"], "grid_auroc": grid["auroc"],
                     "ece_factor": grid["ece"] / clean["ece"],
                     "rho": R.stickiness(clean, grid)})

        # ---- Q4: fit a global temperature at this rung and record the sign of its gain ----
        for hold in holds:
            seen = [f"{n}_{s}" for n, s in CELLS if n != hold and f"{n}_{s}" in emb]
            test = [f"{hold}_{s}" for s in range(1, 6) if f"{hold}_{s}" in emb]
            for vf in range(1, ds.n_folds + 1):
                va = folds == vf
                fit = ~va
                f_lg = np.concatenate([lg["clean"][fit]] + [lg[k][fit] for k in seen])
                f_y = np.concatenate([y[fit]] * (1 + len(seen)))
                if len(f_y) > 60_000:
                    sub = rng.choice(len(f_y), 60_000, replace=False)
                    f_lg, f_y = f_lg[sub], f_y[sub]
                gt = GlobalTemperature().fit(f_lg, f_y)
                for k in test:
                    raw, yy = lg[k][va], y[va]
                    rows.append({"q": "mechanism", "seed": seed, "frac": frac, "fold": vf,
                                 "held_out": hold, "condition": k,
                                 "clean_ece": clean["ece"], "clean_acc": clean["acc"],
                                 "temperature": gt.temperature,
                                 "auroc_gain": (R.error_detection_auroc(gt.apply(raw), yy)
                                                - R.error_detection_auroc(raw, yy)) * 100})
        print(f"    [seed {seed}] ladder+mechanism, frac {frac:.2f}", flush=True)


# --------------------------------------------------------------------------- Q2 and Q3
def q23_repair_vs_train(ds, emb, y, folds, st, seed, rows):
    """Q2: four post-hoc repairs.  Q3: corruption-augmented training.  Both leave-one-out."""
    dim = emb["clean"].shape[1]
    rng = np.random.default_rng(seed)

    # the classifier of Q2 is trained on clean data only, then frozen
    lg = {k: np.zeros((len(y), ds.n_classes), dtype=np.float32) for k in emb}
    for vf in range(1, ds.n_folds + 1):
        va = folds == vf
        h = train_head(emb["clean"][~va], torch.from_numpy(y[~va]).long(),
                       dim, ds.n_classes, seed)
        for k, e in emb.items():
            lg[k][va] = score(h, e, va)

    for hold in sorted(C.CORRUPTIONS):
        seen = [f"{n}_{s}" for n, s in CELLS if n != hold and f"{n}_{s}" in emb]
        test = [f"{hold}_{s}" for s in range(1, 6) if f"{hold}_{s}" in emb]
        if not test:
            continue

        for vf in range(1, ds.n_folds + 1):
            va = folds == vf
            fit = ~va

            # ---- Q2: repair the frozen classifier ----
            f_lg = np.concatenate([lg["clean"][fit]] + [lg[k][fit] for k in seen])
            f_y = np.concatenate([y[fit]] * (1 + len(seen)))
            f_st = np.concatenate([st["clean"][fit]] + [st[k][fit] for k in seen])
            if len(f_y) > 60_000:
                sub = rng.choice(len(f_y), 60_000, replace=False)
                f_lg, f_y, f_st = f_lg[sub], f_y[sub], f_st[sub]

            std = S.Standardiser().fit(f_st)
            f_z = std.transform(f_st)
            gt = GlobalTemperature().fit(f_lg, f_y)
            src = ScaleRestoredConfidence(f_z.shape[1], seed=seed).fit(f_lg, f_y, f_z)
            rsc = RankedScaleRestoredConfidence(f_z.shape[1], seed=seed).fit(f_lg, f_y, f_z)
            det = FailureDetector(f_z.shape[1], seed=seed).fit(f_lg, f_y, f_z)

            for k in test:
                raw, z, yy = lg[k][va], std.transform(st[k][va]), y[va]
                for meth, l2 in (("none", raw), ("global_T", gt.apply(raw)),
                                 ("temp_nll", src.apply(raw, z)),
                                 ("temp_rank", rsc.apply(raw, z))):
                    assert check_accuracy_is_unchanged(raw, l2), meth
                    rows.append({"q": "repair", "seed": seed, "held_out": hold, "fold": vf,
                                 "condition": k, "method": meth, **R.report(l2, yy)})
                r = dict(R.report(raw, yy))
                r["auroc"] = R.auroc(det.error_score(raw, z), raw.argmax(1) != yy)
                rows.append({"q": "repair", "seed": seed, "held_out": hold, "fold": vf,
                             "condition": k, "method": "detector", **r})

                # the fifth family: split conformal, the rule designed for distribution shift.
                # It changes the accept/reject decision, not the logits, so it is judged on the
                # rejection axis: selective accuracy against plain confidence at MATCHED coverage.
                ev = conformal.evaluate(lg["clean"][fit], raw, yy, target_coverage=0.70)
                rows.append({"q": "conformal", "seed": seed, "held_out": hold, "fold": vf,
                             "condition": k, **ev})

            # ---- Q3: train the corruption in instead ----
            tr = np.where(fit)[0]
            pool_c = np.repeat(np.arange(1 + len(seen)), len(tr))
            pool_i = np.tile(tr, 1 + len(seen))
            conds = [emb[k] for k in ["clean"] + seen]

            def gather(sel):
                cs, ix = pool_c[sel], pool_i[sel]
                parts, labs = [], []
                for c in np.unique(cs):
                    r_ = ix[cs == c]
                    parts.append(conds[c][torch.from_numpy(r_).long()])
                    labs.append(y[r_])
                return torch.cat(parts), torch.from_numpy(np.concatenate(labs)).long()

            # matched budget: the SAME number of rows, spread over the seen corruptions.
            # any gain is therefore due to DIVERSITY and not to the quantity of data.
            for arm, sel in (("clean_only", np.arange(len(tr))),
                             ("aug_matched", rng.choice(len(pool_c), len(tr), replace=False))):
                f, l = gather(sel)
                h = train_head(f, l, dim, ds.n_classes, seed)
                for k in ["clean"] + test:
                    rows.append({"q": "train", "seed": seed, "held_out": hold, "fold": vf,
                                 "condition": k, "arm": arm,
                                 **R.report(score(h, emb[k], va), y[va])})
        print(f"    [seed {seed}] held out {hold}", flush=True)


# --------------------------------------------------------------------------- statistics
def _ci(v: np.ndarray) -> str:
    """Mean with a 95 percent interval over the seeds."""
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return "%.4f" % v.mean()
    h = sps.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return "%.4f +/- %.4f" % (v.mean(), h)


def summarise(ds: data.Dataset, df: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print(f"{ds.name.upper()} — Q1. HOW MUCH RELIABILITY DOES ACCURACY BUY?")
    print("=" * 84)
    lad = df[df.q == "ladder"]
    g = lad.groupby("frac")
    print("  %-8s %-22s %-22s %-16s" % ("data", "clean accuracy", "clean ECE", "ECE degradation"))
    for frac, sub in g:
        print("  %-8.2f %-22s %-22s %-16s"
              % (frac, _ci(sub.clean_acc), _ci(sub.clean_ece), _ci(sub.ece_factor)))
    per_seed = [np.corrcoef(s.clean_acc, s.ece_factor)[0, 1] for _, s in lad.groupby("seed")]
    r_mean = float(np.mean(per_seed))
    print("\n  correlation(clean accuracy, ECE degradation factor) = %s" % _ci(per_seed))
    # the conclusion is READ OFF the number; it is never asserted regardless of it
    if r_mean > 0.3:
        print("  -> accuracy buys calibration AND makes that calibration more fragile.")
    elif r_mean < -0.3:
        print("  -> accuracy buys calibration AND makes it more robust. The opposite of us8k.")
    else:
        print("  -> no relation between accuracy and the fragility of calibration on this dataset.")

    print("\n" + "=" * 84)
    print(f"{ds.name.upper()} — Q2. CAN THE CONFIDENCE BE REPAIRED AFTER THE FACT?")
    print("=" * 84)
    rep = df[df.q == "repair"]
    print("  on the corruption held out of fitting:")
    print("  %-12s %-22s %-22s" % ("method", "AUROC", "selective efficiency"))
    got = {}
    for meth in ("none", "global_T", "temp_nll", "temp_rank", "detector"):
        sub = rep[rep.method == meth].groupby("seed")[["auroc", "sel_eff"]].mean()
        got[meth] = sub.auroc
        d = (sub.auroc.mean() - got["none"].mean()) * 100
        print("  %-12s %-22s %-22s  %+.2f" % (meth, _ci(sub.auroc), _ci(sub.sel_eff), d))

    # A winner must be significant, not merely largest. Testing over the seeds alone is hopeless:
    # a signed-rank test on five pairs cannot produce a p below 0.0625 whatever the data say. The
    # test therefore runs over every (corruption, fold, seed) triple, as in Q3.
    piv = rep.pivot_table(index=["held_out", "fold", "seed"], columns="method", values="auroc")
    base = piv["none"].values
    helped = []
    for m in ("global_T", "temp_nll", "temp_rank", "detector"):
        v = piv[m].values
        d, p = (v - base).mean() * 100, sps.wilcoxon(v, base).pvalue
        if p < 0.05 and d > 0:
            helped.append((m, d, p))
    if helped:
        print("  -> post-hoc repair HELPS on this dataset (n = %d paired triples):" % len(piv))
        for m, d, p in sorted(helped, key=lambda h: -h[1]):
            print("       %-10s %+.2f AUROC points, p = %.1e" % (m, d, p))
    else:
        print("  -> nothing significantly beats doing nothing (n = %d paired triples)." % len(piv))

    conf = df[df.q == "conformal"]
    if len(conf):
        print("\n  the fifth family, split conformal (the rule designed for distribution shift):")
        drift = conf.groupby("seed").coverage_drift.mean()
        diff = conf.groupby("seed").conformal_minus_msp.mean()
        print("     realised coverage drift from the 0.70 target: %s" % _ci(drift))
        print("     selective accuracy vs plain confidence at matched coverage: %s" % _ci(diff))
        w = sps.wilcoxon(conf.groupby(["held_out", "fold", "seed"]).conformal_minus_msp.mean().values,
                         np.zeros(conf.groupby(["held_out", "fold", "seed"]).ngroups))
        beats = diff.mean() > 0 and w.pvalue < 0.05
        print("     -> conformal %s plain confidence at matched coverage (p = %.1e)."
              % ("BEATS" if beats else "does NOT beat", w.pvalue))

    print("\n" + "=" * 84)
    print(f"{ds.name.upper()} — Q3. CAN IT BE TRAINED IN INSTEAD?")
    print("=" * 84)
    tr = df[(df.q == "train") & (df.condition != "clean")]
    cl = df[(df.q == "train") & (df.condition == "clean")]
    print("  on the corruption held out of TRAINING (same number of training rows):")
    print("  %-12s %-22s %-22s %-22s" % ("arm", "accuracy", "AUROC", "selective efficiency"))
    for arm in ("clean_only", "aug_matched"):
        s = tr[tr.arm == arm].groupby("seed")[["acc", "auroc", "sel_eff"]].mean()
        print("  %-12s %-22s %-22s %-22s" % (arm, _ci(s.acc), _ci(s.auroc), _ci(s.sel_eff)))
    print("\n  the price on the clean condition:")
    for arm in ("clean_only", "aug_matched"):
        s = cl[cl.arm == arm].groupby("seed")[["acc"]].mean()
        print("  %-12s clean accuracy %s" % (arm, _ci(s.acc)))

    # paired test over every (corruption, fold, seed)
    piv = tr.pivot_table(index=["held_out", "fold", "seed"], columns="arm",
                         values=["auroc", "sel_eff", "acc", "ece"])
    print("\n  paired Wilcoxon over %d (corruption, fold, seed) triples:"
          % len(piv))
    verdict = {}
    for m, lab, higher_is_better in (("auroc", "AUROC", True),
                                     ("sel_eff", "selective efficiency", True),
                                     ("acc", "accuracy", True),
                                     ("ece", "ECE", False)):
        a, b = piv[(m, "clean_only")].values, piv[(m, "aug_matched")].values
        d, p = float((b - a).mean()), sps.wilcoxon(b, a).pvalue
        verdict[m] = (d > 0) == higher_is_better and p < 0.05
        print("    %-22s %+8.4f   p = %.1e   %s"
              % (lab, d, p, "aug wins" if verdict[m] else "aug does NOT win"))

    won = sum(verdict.values())
    print("\n  -> corruption-augmented training wins on %d of the 4 measures, at a matched"
          " training budget." % won)
    if won < 3:
        print("     This does NOT replicate the us8k result and the report must say so.")

    mech = df[df.q == "mechanism"]
    if len(mech):
        print("\n" + "=" * 84)
        print(f"{ds.name.upper()} — Q4. WHY DOES REPAIR HELP ON ONE DATASET AND HURT ON THE OTHER?")
        print("=" * 84)
        print("  a global temperature, fitted at every rung of the Q1 ladder:")
        print("  %-8s %-14s %-14s %-10s %s" %
              ("data", "clean accuracy", "clean ECE", "temp", "AUROC gain from the temperature"))
        for frac, s in mech.groupby("frac"):
            g = s.groupby("seed").auroc_gain.mean()
            print("  %-8.2f %-14.4f %-14.4f %-10.2f %s"
                  % (frac, s.clean_acc.mean(), s.clean_ece.mean(), s.temperature.mean(), _ci(g)))
        # Temperature scaling is defined for an OVERCONFIDENT model. A model still at chance is
        # underconfident and its fitted temperature is below one, which is a different regime; those
        # rungs are excluded by a fixed accuracy floor, declared here and not chosen after the fact.
        strong = mech[mech.clean_acc > 0.55]
        grp = strong.groupby(["frac", "seed"]).agg(
            ece=("clean_ece", "mean"), gain=("auroc_gain", "mean")).reset_index()
        rho, p = sps.spearmanr(grp.ece, grp.gain)
        print("\n  among rungs above the chance floor, Spearman(clean ECE, gain) = "
              "%+.3f (p = %.1e, n = %d)" % (rho, p, len(grp)))
        if rho > 0.3 and p < 0.05:
            print("  -> CONFIRMED. The temperature helps exactly when the scale is still wrong,")
            print("     and hurts once it is right (sign change near clean ECE 0.07). One mechanism")
            print("     explains why repair helps on the worse-calibrated dataset and hurts on the")
            print("     better-calibrated one.")
        else:
            print("  -> NOT confirmed. Clean calibration error does not explain the sign of the gain.")


# --------------------------------------------------------------------------- driver
def stage_headline(ds: data.Dataset, bb: str, seeds: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    wp, y, folds = data.cache(ds, CACHE)
    emb = _load_emb(ds, bb, to_device=True)
    if len(emb) < 2:
        raise SystemExit(f"no {bb} embeddings for {ds.name}; run --stage embed --backbone {bb} first")

    sp = os.path.join(CACHE, f"{ds.name}_stats.npz")     # front-end stats are backbone-independent
    if os.path.exists(sp):
        st = dict(np.load(sp))
    else:
        st = _front_end_stats(wp, len(y), list(emb))
        np.savez_compressed(sp, **st)

    rows = []
    for seed in range(seeds):
        q1q4_ladder(ds, emb, y, folds, seed, rows)
        q23_repair_vs_train(ds, emb, y, folds, st, seed, rows)

    df = pd.DataFrame(rows)
    tag = f"{ds.name}_{bb}"
    df.to_csv(os.path.join(OUT, f"{tag}_headline.csv"), index=False)
    print(f"\n### {ds.name.upper()} / backbone = {bb} ###")
    summarise(ds, df)
    print(f"\n-> {OUT}/{tag}_headline.csv")


# --------------------------------------------------------------------------- C5: benchmark validity
def stage_validity(ds: data.Dataset) -> None:
    """For every corruption cell, how much does it change the tensor the network actually receives?

    A cell whose change is ~0 is a no-operation FOR THAT PIPELINE, and reporting it as evidence of
    robustness is a mistake. The three pipelines differ only in their normalisation:
        resnet_minmax  log-mel, replicate to 3 channels, PER-RECORDING min-max, ImageNet stats
        cnn14          log-mel on a fixed decibel scale (no per-recording rescaling)
        ast            the AudioSet fixed mean/standard-deviation normalisation
    The point is that the SAME cell can be a no-op for one pipeline and not another, so a benchmark
    must report this per pipeline, which neither AVRobustBench nor DHAuDS does.
    """
    import torchaudio.transforms as T

    wp, y, folds = data.cache(ds, CACHE)
    waves = np.load(wp, mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(y), min(400, len(y)), replace=False)
    base = np.stack([np.asarray(waves[i]) for i in idx])

    db = T.AmplitudeToDB()
    mel16 = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64)
    mel32 = T.MelSpectrogram(sample_rate=data.SR, n_fft=1024, hop_length=320, n_mels=64,
                             f_min=50, f_max=14000)
    rs16 = T.Resample(data.SR, 16000)

    def resnet_minmax(w):
        f = db(mel16(torch.from_numpy(w).float())).unsqueeze(1).expand(-1, 3, -1, -1)
        lo = f.amin(dim=(1, 2, 3), keepdim=True)
        hi = f.amax(dim=(1, 2, 3), keepdim=True)
        return ((f - lo) / (hi - lo + 1e-6)).numpy()

    def cnn14_pipe(w):
        return db(mel32(torch.from_numpy(w).float())).numpy()

    def ast_pipe(w):
        m = db(mel16(rs16(torch.from_numpy(w).float())))
        return ((m - m.mean()) / (m.std() + 1e-6)).numpy()

    pipes = {"resnet_minmax": (resnet_minmax, base),
             "cnn14": (cnn14_pipe, base),
             "ast": (ast_pipe, base)}

    rows = []
    for n, s in CELLS:
        c = np.stack([C.corrupt(base[i], data.SR, n, s, seed=0) for i in range(len(base))])
        for name, (fn, ref) in pipes.items():
            d = np.abs(fn(c) - fn(ref))
            rows.append({"pipeline": name, "corruption": n, "severity": s,
                         "mean_delta": float(d.mean()), "max_delta": float(d.max())})
        print(f"  {n:17s} sev {s}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"{ds.name}_validity.csv"), index=False)

    print("\n" + "=" * 78)
    print(f"{ds.name.upper()} — C5. NO-OP CELLS PER PIPELINE (mean change to the input tensor)")
    print("=" * 78)
    # normalise each pipeline by its own strongest cell, so pipelines are comparable
    print("  %-17s %-14s %-14s %-14s" % ("corruption (mean over sev)", *pipes))
    piv = df.groupby(["corruption", "pipeline"]).mean_delta.mean().unstack()
    for p in pipes:
        piv[p] = piv[p] / piv[p].max()
    for n in sorted(C.CORRUPTIONS):
        flags = "".join("  NO-OP" if piv.loc[n, p] < 0.05 else "      " for p in pipes)
        print("  %-17s %s" % (n, "  ".join("%.3f" % piv.loc[n, p] for p in pipes) + flags))
    print("\n  a value near 0 means the corruption barely reaches that pipeline's input.")
    for p in pipes:
        noops = [n for n in sorted(C.CORRUPTIONS) if piv.loc[n, p] < 0.05]
        print("  %-14s no-op cells: %s" % (p, ", ".join(noops) if noops else "none"))
    print(f"-> {OUT}/{ds.name}_validity.csv")


_W = None


def _init(p):
    global _W
    _W = np.load(p, mmap_mode="r")


def _chunk(a):
    name, sev, lo, hi = a
    out = np.empty((hi - lo, len(S.NAMES)), dtype=np.float32)
    for j, i in enumerate(range(lo, hi)):
        w = np.asarray(_W[i])
        if name is not None:
            w = C.corrupt(w, data.SR, name, sev, seed=0)
        out[j] = S.compute(w, data.SR)
    return out


def _front_end_stats(wp, n, keys):
    from concurrent.futures import ProcessPoolExecutor
    workers = int(os.environ.get("SG_WORKERS", "24"))
    bounds = [(i, min(i + 512, n)) for i in range(0, n, 512)]
    st = {}
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(wp,)) as pool:
        for k in keys:
            nm, sv = (None, 0) if k == "clean" else (k.rsplit("_", 1)[0], int(k.rsplit("_", 1)[1]))
            st[k] = np.concatenate(list(pool.map(
                _chunk, [(nm, sv, lo, hi) for lo, hi in bounds])))
    return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(data.REGISTRY))
    ap.add_argument("--stage", required=True, choices=["embed", "headline", "validity"])
    ap.add_argument("--backbone", default="cnn14", choices=list(BACKBONES))
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    ds = data.REGISTRY[a.dataset]
    print(f"device {DEVICE} | dataset {ds.name} | backbone {a.backbone} | stage {a.stage}")
    if a.stage == "embed":
        stage_embed(ds, a.backbone)
    elif a.stage == "validity":
        stage_validity(ds)
    else:
        stage_headline(ds, a.backbone, a.seeds)
