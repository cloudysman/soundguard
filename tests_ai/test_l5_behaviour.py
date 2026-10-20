"""L5 — model quality / behaviour / reproducibility (the science, CPU-forced).

Validates that the served model is a REAL trained ~70% classifier and that the project's
core claims hold. Two data sources are used deliberately:

  * `us8k` (predict_wav path) — proves the full serving pipeline works and recognises every
    class. NOTE: `mlp_prod.pth` was trained on ALL of UrbanSound8K, so this sample is NOT
    held-out w.r.t. the prod head and scores ~100% (train contamination — see the xfail
    `test_us8k_sample_is_not_heldout` and the findings entry). It is therefore unsuitable for
    the "sane band, not a suspicious 1.0" bar and for error-detection (no errors to detect).
  * `rerun_logits` (honest cross-validated LAST-epoch logits, matches results/dashboard.json,
    acc 0.700) — the genuinely held-out signal used for the "sane band" and the ERROR-DETECTION
    thesis (confidence discriminates correct from wrong).

Behavioural error-detection is additionally confirmed end-to-end through predict_wav on a
mixed clean+heavy-noise us8k set (which has real errors).

All numeric comparisons are float64 with explicit tolerances and fixed seeds.
Reference: predictor.Predictor / features.FeatureExtractor / heads.MLPHead / corruption.py / metrics.py.
"""
import os

import numpy as np
import pytest

import metrics as M
from helpers_ai import CLASSES

# demo-clip filename stem -> the class it is supposed to be recognised as
DEMO_MAP = {
    "ac": "air_conditioner", "dog": "dog_bark", "gun": "gun_shot", "horn": "car_horn",
    "jack": "jackhammer", "kids": "children_playing", "siren": "siren",
}
SAMPLES_DIR = "samples"
SEED = 0


# ----------------------------------------------------------------------------- helpers
def _read_wav(path):
    import soundfile as sf
    wav, sr = sf.read(path)
    wav = np.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def _auroc(scores, correct):
    """AUROC of `scores` discriminating correct(=1) from wrong(=0) via Mann-Whitney U."""
    scores = np.asarray(scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    pos, neg = scores[correct], scores[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for a in pos:                                   # O(n_pos) python x vectorised inner
        wins += np.sum(a > neg) + 0.5 * np.sum(a == neg)
    return float(wins / (len(pos) * len(neg)))


def _need(model_ok, us8k):
    if not model_ok:
        pytest.skip("model unavailable")
    if us8k is None:
        pytest.skip("UrbanSound8K not present")


# ---------------------------------------------- shared held-out prediction passes (CPU)
@pytest.fixture(scope="module")
def preds(predictor, model_ok, us8k):
    """Clean predictions over a us8k sample (n_per_class=3): {label,pred,conf,correct}."""
    _need(model_ok, us8k)
    clips = us8k(n_per_class=3, seed=SEED)
    if len(clips) < 8:
        pytest.skip("too few us8k clips")
    rows = []
    for wav, sr, label, _name in clips:
        r = predictor.predict_wav(wav, sr)
        pred = int(r["order"][0][0])
        rows.append({"label": int(label), "pred": pred,
                     "conf": float(r["conf"]), "correct": pred == int(label)})
    return rows


@pytest.fixture(scope="module")
def clips2(model_ok, us8k):
    _need(model_ok, us8k)
    c = us8k(n_per_class=2, seed=SEED)
    if len(c) < 8:
        pytest.skip("too few us8k clips")
    return c


@pytest.fixture(scope="module")
def clean2(predictor, clips2):
    rows = []
    for wav, sr, label, _name in clips2:
        r = predictor.predict_wav(wav, sr)
        pred = int(r["order"][0][0])
        rows.append({"label": int(label), "pred": pred,
                     "conf": float(r["conf"]), "correct": pred == int(label)})
    return rows


@pytest.fixture(scope="module")
def preds_mixed(predictor, clips2, clean2):
    """clean (all correct) + heavy-noise corrupted (introduces errors) -> both classes present."""
    rows = list(clean2)
    for wav, sr, label, _name in clips2:
        w = predictor.apply_slider_corruption(wav, sr, noise=0.9, seed=SEED)
        r = predictor.predict_wav(w, sr)
        pred = int(r["order"][0][0])
        rows.append({"label": int(label), "pred": pred,
                     "conf": float(r["conf"]), "correct": pred == int(label)})
    return rows


# ============================================================ quality: inference works
def test_us8k_inference_works(preds):
    acc = float(np.mean([r["correct"] for r in preds]))
    assert acc > 0.5, f"us8k top-1 acc {acc:.3f} too low — serving pipeline broken"


def test_per_class_recognised_on_us8k(preds):
    by_cls = {}
    for r in preds:
        by_cls.setdefault(r["label"], []).append(r["correct"])
    for c, v in by_cls.items():
        acc_c = float(np.mean(v))
        assert acc_c > 0.10, (f"class {CLASSES[c]} us8k acc {acc_c:.3f} (={int(sum(v))}/{len(v)}) "
                              "at/below chance 0.1")


@pytest.mark.xfail(reason="mlp_prod.pth was trained on ALL of UrbanSound8K, so the us8k fixture "
                          "is not held-out: predict_wav scores 30/30=1.000 (n=3, seeds 0/1/2). "
                          "The honest held-out accuracy is 0.700 (rerun CV logits).",
                   strict=False)
def test_us8k_sample_is_not_heldout(preds):
    acc = float(np.mean([r["correct"] for r in preds]))
    assert acc < 1.0, f"us8k acc is a suspicious perfect {acc:.3f} (train contamination)"


# ============================================================ honest held-out (rerun CV)
def test_heldout_top1_sane_band(rerun_logits):
    logits, labels = rerun_logits
    acc = M.accuracy(logits, labels)
    # genuinely held-out CV logits: a ~70% model — well above chance, NOT a suspicious 1.0
    assert 0.55 < acc < 0.85, f"held-out acc {acc:.3f} outside the sane ~0.70 band"
    assert acc < 0.99, f"held-out acc {acc:.3f} is a suspicious near-perfect score"


def test_head_far_better_than_chance(rerun_logits):
    logits, labels = rerun_logits
    acc = M.accuracy(logits, labels)
    assert acc > 4 * 0.10, f"held-out acc {acc:.3f} not far above the 0.1 chance level"


def test_heldout_per_class_beats_chance(rerun_logits):
    logits, labels = rerun_logits
    for row in M.per_class_accuracy(logits, labels, num_classes=len(CLASSES)):
        if row["count"] == 0:
            continue
        assert row["accuracy"] > 0.10, (f"held-out class {CLASSES[row['class']]} "
                                        f"acc {row['accuracy']:.3f} at/below chance")


# ============================================================ error-detection (thesis)
def test_error_detection_mean_confidence_heldout(rerun_logits):
    logits, labels = rerun_logits
    probs = M.softmax(logits)
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(labels).astype(int)
    m_ok, m_bad = conf[correct].mean(), conf[~correct].mean()
    assert m_ok > m_bad, (f"held-out mean conf on correct {m_ok:.3f} not > on wrong {m_bad:.3f} "
                          "— confidence carries no error signal")


def test_error_detection_auroc_heldout(rerun_logits):
    logits, labels = rerun_logits
    probs = M.softmax(logits)
    conf = probs.max(1)
    correct = probs.argmax(1) == np.asarray(labels).astype(int)
    auroc = _auroc(conf, correct)
    assert auroc is not None and auroc > 0.5, (
        f"confidence error-detection AUROC {auroc} is no better than random")


def test_error_detection_through_predict_wav(preds_mixed):
    conf = np.array([r["conf"] for r in preds_mixed], dtype=np.float64)
    correct = np.array([r["correct"] for r in preds_mixed], dtype=bool)
    if not (correct.any() and (~correct).any()):
        pytest.skip("mixed set had no errors to detect")
    m_ok, m_bad = conf[correct].mean(), conf[~correct].mean()
    auroc = _auroc(conf, correct)
    assert m_ok > m_bad, f"audio-path mean conf correct {m_ok:.3f} !> wrong {m_bad:.3f}"
    assert auroc > 0.5, f"audio-path error-detection AUROC {auroc:.3f} !> 0.5"


# ============================================================ demo clips
@pytest.mark.parametrize("stem,expected", sorted(DEMO_MAP.items()))
def test_demo_clip_classifies_to_own_class(predictor, model_ok, stem, expected):
    if not model_ok:
        pytest.skip("model unavailable")
    path = os.path.join(SAMPLES_DIR, f"{stem}.wav")
    if not os.path.exists(path):
        pytest.skip(f"{path} not present on this machine")
    pytest.importorskip("soundfile")
    wav, sr = _read_wav(path)
    r = predictor.predict_wav(wav, sr)
    pred = CLASSES[int(r["order"][0][0])]
    assert pred == expected, (f"demo clip {stem}.wav predicted {pred} "
                              f"(conf {r['conf']:.3f}), expected {expected}")


# ============================================================ trained-weights evidence
def test_head_batchnorm_running_stats_are_trained(predictor, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    import torch
    bns = [m for m in predictor.head.modules()
           if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
    assert bns, "MLP head has no BatchNorm layer to inspect"
    rv = bns[0].running_var.detach().cpu().numpy().astype(np.float64)
    rm = bns[0].running_mean.detach().cpu().numpy().astype(np.float64)
    # untrained defaults are running_var==all-ones, running_mean==all-zeros
    assert not np.allclose(rv, 1.0, atol=1e-3), "BatchNorm running_var == all-ones (untrained)"
    assert not np.allclose(rm, 0.0, atol=1e-4), "BatchNorm running_mean == all-zeros (untrained)"
    assert np.isfinite(rv).all() and (rv > 0).all()


# ============================================================ reproducibility
def test_predict_wav_deterministic_repeated(predictor, model_ok, us8k):
    _need(model_ok, us8k)
    wav, sr, _l, _n = us8k(n_per_class=1, seed=SEED)[0]
    p1 = np.asarray(predictor.predict_wav(wav, sr)["probs"], dtype=np.float64)
    p2 = np.asarray(predictor.predict_wav(wav, sr)["probs"], dtype=np.float64)
    assert np.allclose(p1, p2, atol=1e-6, rtol=0), f"max abs diff {np.abs(p1 - p2).max():.2e}"


def test_predict_wav_deterministic_fresh_predictor(predictor, model_ok, us8k):
    _need(model_ok, us8k)
    from predictor import Predictor
    wav, sr, _l, _n = us8k(n_per_class=1, seed=SEED)[0]
    p1 = np.asarray(predictor.predict_wav(wav, sr)["probs"], dtype=np.float64)
    fresh = Predictor()
    if not fresh.available:
        pytest.skip(f"fresh Predictor unavailable: {fresh.note}")
    p2 = np.asarray(fresh.predict_wav(wav, sr)["probs"], dtype=np.float64)
    assert np.allclose(p1, p2, atol=1e-5, rtol=0), f"max abs diff {np.abs(p1 - p2).max():.2e}"


def test_fresh_extractor_identical_features(extractor, model_ok, us8k):
    _need(model_ok, us8k)
    from features import FeatureExtractor
    fe2 = FeatureExtractor("checkpoints/feature_config.json", device="cpu")
    wav, sr, _l, _n = us8k(n_per_class=1, seed=SEED)[0]
    f1 = extractor.wav_to_features(wav, sr).detach().cpu().numpy().astype(np.float64)
    f2 = fe2.wav_to_features(wav, sr).detach().cpu().numpy().astype(np.float64)
    assert f1.shape == (1, 512) == f2.shape
    assert np.allclose(f1, f2, atol=1e-5, rtol=0), f"max abs diff {np.abs(f1 - f2).max():.2e}"


# ============================================================ behavioural invariance
@pytest.fixture(scope="module")
def confident_clip(predictor, model_ok, us8k):
    """A us8k clip that is confidently & correctly classified (available on the server)."""
    _need(model_ok, us8k)
    for wav, sr, label, _name in us8k(n_per_class=3, seed=SEED):
        r = predictor.predict_wav(wav, sr)
        if int(r["order"][0][0]) == int(label) and r["conf"] > 0.7:
            return np.asarray(wav, dtype=np.float32), sr, int(label), float(r["conf"])
    pytest.skip("no confidently-correct us8k clip found")


def test_tiny_gain_does_not_flip_prediction(predictor, confident_clip):
    wav, sr, cls, conf = confident_clip
    quieter = (wav.astype(np.float64) * 0.98).astype(np.float32)
    pred = int(predictor.predict_wav(quieter, sr)["order"][0][0])
    assert pred == cls, (f"x0.98 gain flipped a confident (conf {conf:.3f}) prediction "
                         f"{CLASSES[cls]} -> {CLASSES[pred]}")


def test_small_pad_shift_does_not_flip_prediction(predictor, confident_clip):
    wav, sr, cls, conf = confident_clip
    shifted = np.concatenate([np.zeros(64, dtype=np.float32), wav.astype(np.float32)])
    pred = int(predictor.predict_wav(shifted, sr)["order"][0][0])
    assert pred == cls, (f"a 64-sample pad shift flipped a confident (conf {conf:.3f}) "
                         f"prediction {CLASSES[cls]} -> {CLASSES[pred]}")


# ============================================================ corruption trend
def _corrupt_acc(predictor, clips, **slider):
    hit = 0
    for wav, sr, label, _name in clips:
        w = predictor.apply_slider_corruption(wav, sr, **slider)
        pred = int(predictor.predict_wav(w, sr)["order"][0][0])
        hit += int(pred == int(label))
    return hit / len(clips)


def test_noise_slider_trends_accuracy_down(predictor, clips2, clean2):
    clean = float(np.mean([r["correct"] for r in clean2]))
    heavy = _corrupt_acc(predictor, clips2, noise=0.9, seed=SEED)
    # heavy additive noise (SNR ~ -0.6 dB) must not raise accuracy; clean >= heavy within tol
    assert clean + 0.05 >= heavy, f"noise raised accuracy: clean {clean:.3f} < heavy {heavy:.3f}"


def test_clip_slider_trends_accuracy_down(predictor, clips2, clean2):
    clean = float(np.mean([r["correct"] for r in clean2]))
    heavy = _corrupt_acc(predictor, clips2, clip=0.9, seed=SEED)
    assert clean + 0.05 >= heavy, f"clip raised accuracy: clean {clean:.3f} < heavy {heavy:.3f}"
