"""L1 — foundations for the SoundGuard model / feature extractor (the science, not HTTP).

Covers:
  * predictor availability + class-name identity (predictor.classes == CLASSES == config['classes'])
  * FeatureExtractor: config reflected in extractor.sr / num_samples / mel transform
  * wav_to_features on tone / noise -> torch [1,512], finite
  * 1-D and 2-D [C,T] input both accepted (downmix); any length -> exactly num_samples internally
  * predict_wav dict schema: probs len10, sum==1, all in [0,1]; order desc permutation;
    conf == order[0][1]; entropy in [0,1]; status/cause in their allowed sets
  * determinism: same wav -> identical probs and identical head logits

All model-dependent tests skip when model_ok is False. Numerical checks use float64 with
explicit tolerances and fixed seeds; audio samples are tiny.
"""
import math

import numpy as np
import pytest

SR = 16000
STATUS_SET = {"reliable", "uncertain", "reject"}
CAUSE_SET = {"noise", "quiet", "clip", "uncert", "clean"}

# Canonical class order (matches conftest.CLASSES / feature_config.json).
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


# ---------------- small deterministic waveforms ----------------

def _tone(freq=440.0, secs=2.0, sr=SR, amp=0.5):
    t = np.arange(int(secs * sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(secs=2.0, sr=SR, seed=0, amp=0.3):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(secs * sr))).astype(np.float32)


# ======================= predictor availability / classes =======================

def test_predictor_available(model_ok, predictor):
    if not model_ok:
        pytest.skip(f"model unavailable: {predictor.note}")
    assert predictor.available is True


def test_classes_identity(model_ok, predictor, config):
    if not model_ok:
        pytest.skip(f"model unavailable: {predictor.note}")
    assert list(predictor.classes) == list(CLASSES)
    assert list(config["classes"]) == list(CLASSES)
    assert len(CLASSES) == 10
    assert len(set(CLASSES)) == 10


# ======================= FeatureExtractor config reflection =======================

def test_config_values(config):
    assert config["sample_rate"] == 16000
    assert config["n_fft"] == 1024
    assert config["hop_length"] == 512
    assert config["n_mels"] == 64
    assert config["max_length_s"] == 4
    assert config["in_features"] == 512
    assert config["num_classes"] == 10


def test_extractor_reflects_config(extractor, config):
    assert extractor.sr == config["sample_rate"] == 16000
    assert extractor.num_samples == config["sample_rate"] * config["max_length_s"] == 64000
    mel = extractor.mel
    assert mel.sample_rate == config["sample_rate"]
    assert mel.n_fft == config["n_fft"]
    assert mel.hop_length == config["hop_length"]
    assert mel.n_mels == config["n_mels"]


# ======================= wav_to_features basics =======================

def test_features_shape_and_finite_tone(extractor):
    import torch
    feats = extractor.wav_to_features(_tone(), SR)
    assert torch.is_tensor(feats)
    assert tuple(feats.shape) == (1, 512)
    assert torch.isfinite(feats).all()


def test_features_shape_and_finite_noise(extractor):
    import torch
    feats = extractor.wav_to_features(_noise(seed=1), SR)
    assert tuple(feats.shape) == (1, 512)
    assert torch.isfinite(feats).all()


def test_features_accept_1d_and_2d_downmix(extractor):
    import torch
    x = _tone()
    f1 = extractor.wav_to_features(x, SR)
    # stereo with two identical channels must downmix to the same mono signal
    stereo = np.stack([x, x], axis=0)  # [2, T]
    f2 = extractor.wav_to_features(stereo, SR)
    assert tuple(f2.shape) == (1, 512)
    assert torch.allclose(f1, f2, atol=1e-5, rtol=0), \
        f"stereo(identical)!=mono, max|diff|={ (f1-f2).abs().max().item() }"


def test_pad_or_crop_normalises_length(extractor):
    import torch
    ns = extractor.num_samples
    for n in (1000, ns, 3 * ns + 7):
        w = torch.zeros(1, n)
        out = extractor._pad_or_crop(w)
        assert out.shape[1] == ns, f"length {n} -> {out.shape[1]}, expected {ns}"


def test_any_length_produces_valid_features(extractor):
    import torch
    # very short and very long both succeed and give [1,512] finite
    for secs in (0.1, 6.0):
        feats = extractor.wav_to_features(_tone(secs=secs), SR)
        assert tuple(feats.shape) == (1, 512)
        assert torch.isfinite(feats).all()


# ======================= predict_wav schema =======================

@pytest.fixture(scope="module")
def one_prediction(request):
    predictor = request.getfixturevalue("predictor")
    if not predictor.available:
        pytest.skip(f"model unavailable: {predictor.note}")
    return predictor.predict_wav(_tone(), SR)


def test_predict_probs_valid_simplex(one_prediction):
    probs = np.asarray(one_prediction["probs"], dtype=np.float64)
    assert probs.shape == (10,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert abs(probs.sum() - 1.0) < 1e-6, f"probs sum={probs.sum()!r}"


def test_predict_order_is_desc_permutation(one_prediction):
    order = one_prediction["order"]
    assert len(order) == 10
    idxs = [i for i, _ in order]
    assert sorted(idxs) == list(range(10)), "order indices not a permutation of 0..9"
    ps = [p for _, p in order]
    assert all(ps[k] >= ps[k + 1] for k in range(len(ps) - 1)), "order not descending"
    # order probabilities match the probs vector at those indices
    probs = one_prediction["probs"]
    for i, p in order:
        assert p == probs[i]


def test_predict_conf_matches_top(one_prediction):
    order = one_prediction["order"]
    conf = one_prediction["conf"]
    assert conf == order[0][1]
    assert conf == max(one_prediction["probs"])


def test_predict_entropy_in_unit_range(one_prediction):
    ent = one_prediction["entropy"]
    assert 0.0 <= ent <= 1.0 + 1e-9, f"entropy={ent!r} out of [0,1]"
    # cross-check the normalised-entropy formula
    probs = one_prediction["probs"]
    ref = -sum(p * math.log(p) for p in probs if p > 0) / math.log(len(probs))
    assert abs(ent - ref) < 1e-9


def test_predict_status_and_cause_sets(one_prediction):
    assert one_prediction["status"] in STATUS_SET
    assert one_prediction["cause"] in CAUSE_SET


def test_predict_dict_has_all_keys(one_prediction):
    for k in ("probs", "order", "conf", "entropy", "snrDb", "levelDb",
              "clipPct", "status", "cause"):
        assert k in one_prediction


# ======================= determinism =======================

def test_predict_deterministic_probs(model_ok, predictor):
    if not model_ok:
        pytest.skip(f"model unavailable: {predictor.note}")
    x = _tone(freq=523.25)
    p1 = predictor.predict_wav(x, SR)["probs"]
    p2 = predictor.predict_wav(x, SR)["probs"]
    assert p1 == p2, "predict_wav probs not identical across calls"


def test_head_logits_deterministic(model_ok, predictor):
    import torch
    if not model_ok:
        pytest.skip(f"model unavailable: {predictor.note}")
    x = _tone(freq=330.0)
    f1 = predictor.extractor.wav_to_features(x, SR)
    f2 = predictor.extractor.wav_to_features(x, SR)
    assert torch.equal(f1, f2), "features not identical across calls"
    with torch.no_grad():
        l1 = predictor.head(f1)
        l2 = predictor.head(f2)
    assert torch.equal(l1, l2), "head logits not identical across calls"
