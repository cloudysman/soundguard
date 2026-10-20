"""Tests for the metrics + eval modules.

    pip install pytest
    pytest -v test_metrics.py
"""
import os
import json
import tempfile

import numpy as np

import metrics as M
import eval as E


# ---------- metrics.py property checks ----------

def _onehot_logits(labels, scale=10.0, num_classes=10):
    z = np.zeros((len(labels), num_classes))
    z[np.arange(len(labels)), labels] = scale
    return z


def test_softmax_sums_to_one():
    p = M.softmax(np.random.randn(50, 10))
    assert np.allclose(p.sum(1), 1.0)


def test_confident_correct_is_well_calibrated():
    labels = np.arange(100) % 10
    logits = _onehot_logits(labels, scale=10.0)         # confident AND correct
    assert M.accuracy(logits, labels) == 1.0
    assert M.ece(logits, labels) < 0.05                 # conf~1, acc=1 -> tiny gap
    assert M.nll(logits, labels) < 0.05
    assert M.brier(logits, labels) < 0.05


def test_confident_but_wrong_is_badly_calibrated():
    labels = np.zeros(100, dtype=int)
    logits = _onehot_logits(np.ones(100, dtype=int), scale=10.0)   # confident on the WRONG class
    assert M.accuracy(logits, labels) == 0.0
    assert M.ece(logits, labels) > 0.8                  # conf~1, acc=0 -> huge gap


def test_metric_ranges():
    labels = np.random.randint(0, 10, 200)
    logits = np.random.randn(200, 10)
    assert 0.0 <= M.accuracy(logits, labels) <= 1.0
    assert 0.0 <= M.ece(logits, labels) <= 1.0
    assert M.nll(logits, labels) >= 0.0
    assert M.brier(logits, labels) >= 0.0


def test_reliability_curve_shape():
    labels = np.random.randint(0, 10, 200)
    logits = np.random.randn(200, 10)
    rel = M.reliability_curve(logits, labels, n_bins=15)
    assert len(rel["centers"]) == 15 and len(rel["accuracy"]) == 15


def test_risk_decreases_when_only_confident_kept():
    # 100 confident+correct, 100 low-confidence+mostly-wrong
    lab_conf = np.arange(100) % 10
    log_conf = _onehot_logits(lab_conf, scale=10.0)
    lab_unc = np.random.randint(0, 10, 100)
    log_unc = np.zeros((100, 10))                        # uniform -> conf ~0.1
    logits = np.vstack([log_conf, log_unc])
    labels = np.concatenate([lab_conf, lab_unc])
    rc = M.risk_coverage_curve(logits, labels)
    assert rc["risk"][0] <= rc["risk"][-1]               # low coverage (confident) -> lower risk


def test_per_class_accuracy_covers_all_classes():
    labels = np.random.randint(0, 10, 300)
    logits = np.random.randn(300, 10)
    pca = M.per_class_accuracy(logits, labels, num_classes=10)
    assert len(pca) == 10
    assert sum(p["count"] for p in pca) == 300


def test_no_temperature_scaling_leakage():
    # scaling logits by a constant must NOT change argmax-accuracy (sanity: we read raw)
    labels = np.arange(100) % 10
    logits = np.random.randn(100, 10)
    assert M.accuracy(logits, labels) == M.accuracy(logits * 3.0, labels)


# ---------- eval.py dashboard checks (loader monkeypatched -> no torch needed) ----------

def _signal_logits(n=300, signal=4.0):
    labels = np.random.randint(0, 10, n)
    logits = np.random.randn(n, 10)
    logits[np.arange(n), labels] += signal       # give it real signal
    return logits, labels


def _patch_loader(monkeypatch, clean, corr=None):
    def loader(head, tag, results_dir):
        if tag == "val":
            return clean
        if tag == "val_corrupted":
            return corr
        return None
    monkeypatch.setattr(E, "load_dumps", loader)


def test_dashboard_schema_clean_only(monkeypatch):
    _patch_loader(monkeypatch, clean=_signal_logits(), corr=None)
    with tempfile.TemporaryDirectory() as d:
        dash, path = E.build_dashboard("mlp", d)
        assert {"reliability", "riskCoverage", "classAcc", "metrics"} <= set(dash.keys())
        assert len(dash["classAcc"]) == 10
        assert {m["k"] for m in dash["metrics"]} == {"ECE", "NLL", "Brier", "Accuracy"}
        assert dash["_corrupted_pending"] is True
        assert dash["reliability"]["corrupted"] is None
        assert all(c["corrupt"] is None for c in dash["classAcc"])
        with open(path) as f:
            json.load(f)                          # valid JSON on disk


def test_dashboard_fills_corrupted_when_present(monkeypatch):
    _patch_loader(monkeypatch, clean=_signal_logits(signal=4.0),
                  corr=_signal_logits(signal=1.0))   # weaker -> worse
    with tempfile.TemporaryDirectory() as d:
        dash, _ = E.build_dashboard("mlp", d)
        assert dash["_corrupted_pending"] is False
        assert dash["reliability"]["corrupted"] is not None
        assert all(c["corrupt"] is not None for c in dash["classAcc"])