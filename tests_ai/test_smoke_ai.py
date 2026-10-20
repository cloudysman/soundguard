"""AI harness smoke test — model on CPU, extractor, ground-truth logits, metric reproduction."""
import numpy as np

import metrics as M
from helpers_ai import CLASSES


def test_model_and_classes(predictor, model_ok):
    if not model_ok:
        import pytest; pytest.skip("model unavailable")
    assert list(predictor.classes) == CLASSES


def test_extractor_shape(extractor):
    wav = np.sin(np.linspace(0, 200, 16000)).astype("float32")
    feats = extractor.wav_to_features(wav, 16000)
    assert tuple(feats.shape) == (1, 512)
    assert np.isfinite(feats.detach().numpy()).all()


def test_rerun_logits_shape(rerun_logits):
    logits, labels = rerun_logits
    assert logits.shape[1] == 10
    assert len(labels) == len(logits) > 1000


def test_metrics_reproduce_dashboard_accuracy(rerun_logits):
    logits, labels = rerun_logits
    acc = M.accuracy(logits, labels)
    # results/dashboard.json reports 0.700 clean accuracy from these same last-epoch logits
    assert abs(acc - 0.700) < 0.01, acc


def test_us8k_available(us8k):
    if us8k is None:
        import pytest; pytest.skip("UrbanSound8K not present")
    clips = us8k(n_per_class=1, seed=0)
    assert len(clips) >= 8
    wav, sr, idx, name = clips[0]
    assert wav.ndim == 1 and 0 <= idx < 10 and name in CLASSES
