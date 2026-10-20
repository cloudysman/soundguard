# Reproducing the paper

*A Measurement Error in an Urban Sound Rejection Policy* — SOICT 2026, Springer CCIS.

The paper audits the rejection policy this repository serves. It makes three claims: the policy's
"signal-to-noise ratio" is equal by an algebraic identity to the inter-decile range of frame energy in
decibels, so it measures dispersion and not noise; under corruption the confidence ordering survives
while the confidence scale drifts; and a rejection rule should therefore anchor its operating point to
a coverage target rather than to an absolute confidence value.

## The classifier is not the one the web app serves

The app serves `checkpoints/mlp_prod.pth`, an MLP head trained on all of UrbanSound8K, which reaches
70.0 % out of fold. The paper uses a **linear** probe on the same frozen ResNet-18 features, evaluated
over the ten official folds and averaged over five seeds, which reaches **70.6 %**. The two numbers
belong to two different heads and must not be mixed. Everything below refers to the linear probe.

## Running the chain

```bash
# UrbanSound8K under data/UrbanSound8K, ESC-50 under data/ESC-50
python experiments/make_quality_table.py        # the three statistics, per recording per condition
python experiments/clean_confidence.py          # per-recording confidence on clean input
python experiments/camera_ready_experiments.py  # E1, E2, E3
python experiments/e1b_axis_vs_confidence.py    # the axis at its own coverage
python experiments/noise_estimator_baseline.py  # WADA and minimum statistics
python experiments/esc50_flagging.py            # the same thresholds on ESC-50
python experiments/verify_percat.py             # the per-category table
```

`SG_WORKERS` sets the pool size; the default assumes a machine you are not using for anything else.

## Which script produces which number

| number in the paper | where it comes from |
|---|---|
| recording axis flags 30.0 % of clean UrbanSound8K | `verify_percat.py` |
| the three gates: 23.8 % at AUROC 0.579, 8.2 % at 0.514, 0.0 % | `camera_ready_experiments.py`, key `e3_three_gates` |
| axis AUROC 0.553, within-category median 0.515 (0.457–0.712) | `verify_percat.py` |
| per-category flagging and accuracy (Table 3) | `verify_percat.py` |
| flagging against accuracy, Pearson −0.59, Spearman −0.70 | `verify_percat.py` |
| ESC-50: axis 13.1 %, nine of fifty categories never flagged | `esc50_flagging.py` |
| statistic recovers 0.19 and 0.11 of the imposed span | `noise_estimator_baseline.py` |
| WADA recovers 0.29 and 0.44; minimum statistics 0.12 and 0.08 | `noise_estimator_baseline.py` |
| split by category: 0.41/0.30 and 0.15/0.07 against 0.64/0.72 and 0.31/0.48 | `noise_estimator_baseline.py` |
| combined rule 44.9 % at risk 0.120 against 0.083 (clean) | `camera_ready_experiments.py`, key `e1_combined_versus_confidence` |
| combined rule worse in 49 of 50 conditions, by 0.057 | `camera_ready_experiments.py` |
| axis alone: 70.0 % at 0.262 against 0.174 (clean) | `e1b_axis_vs_confidence.py` |
| axis alone under corruption: 67.2 % at 0.444 against 0.372 | `e1b_axis_vs_confidence.py` |

The reliability figures of Tables 4 and 5 (ECE, AUROC, Brier, NLL over six backbone–dataset cases) come
from the training pipeline in `../pipeline.py`, branch `q=ladder` at full data over five seeds.

## The eight thresholds

`../rejection.py` holds the deployed configuration the paper audits, verbatim: the statistic rejects
below 2.0 dB and marks uncertain below 4.0, the level rejects below −50 dBFS and marks uncertain below
−40, clipping rejects above 30 % and marks uncertain above 12, and the confidence rejects below 0.45
and marks uncertain below 0.70. The experiment scripts restate these constants rather than importing
them, so a change to the served policy cannot silently rewrite a published number.

## The corruption model

`../corruption.py` holds the ten corruption types at five severities each, plus the mixed condition
that draws one type and one severity uniformly at random. Every recording is corrupted with a fixed
seed, so the fifty conditions are identical across backbones and seeds. Table 2 of the paper lists
every parameter.

## What is not here

The audio is not redistributed beyond the seven demo clips in `../samples/`. UrbanSound8K and ESC-50
are downloaded separately. Cached embeddings and waveform caches are rebuilt by the scripts on first
run and are large.
