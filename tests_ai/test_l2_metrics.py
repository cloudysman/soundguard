"""L2 — metric-formula correctness for metrics.py.

Prove metrics.py is mathematically correct on inputs whose answers are known by hand, cross-check
against the reference implementations in sg/reliability.py & sg/stats.py, and reproduce the published
results/dashboard.json numbers from the rerun logits. Tests the science, not HTTP.

All numeric work in float64 with explicit tolerances. No model / audio needed here.
"""
import math

import numpy as np
import pytest

import metrics as M


# ----------------------------------------------------------------------------- helpers
def logits_from_probs(probs):
    """softmax(log p) == p for a normalised p, so log-prob logits give an exactly known softmax."""
    p = np.asarray(probs, dtype=np.float64)
    return np.log(np.clip(p, 1e-300, 1.0))


def two_class_logits_from_conf(confs):
    """2-class logits [L, 0] with L = logit(c); for c>0.5 the max-prob (confidence) is exactly c and
    the prediction is always class 0."""
    confs = np.asarray(confs, dtype=np.float64)
    L = np.log(confs / (1.0 - confs))
    return np.stack([L, np.zeros_like(L)], axis=1)


def ref_softmax_rows(logits):
    """Independent, row-by-row softmax reference (max-shifted, pure-python math)."""
    out = []
    for row in np.asarray(logits, dtype=np.float64):
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        out.append([v / s for v in e])
    return np.asarray(out, dtype=np.float64)


# ----------------------------------------------------------------------------- softmax
def test_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(50, 10)) * 3.0
    p = M.softmax(logits)
    assert p.shape == (50, 10)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert (p >= 0).all() and (p <= 1).all()


def test_softmax_additive_constant_invariance():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(20, 7)) * 2.0
    shift = rng.normal(size=(20, 1)) * 5.0          # per-row additive constant
    np.testing.assert_allclose(M.softmax(logits), M.softmax(logits + shift), atol=1e-12)


def test_softmax_matches_independent_reference():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(30, 5)) * 4.0
    np.testing.assert_allclose(M.softmax(logits), ref_softmax_rows(logits), atol=1e-12)


def test_softmax_matches_naive_reference_moderate():
    """Correctness independent of the max-subtraction trick: for moderate logits the naive
    exp/sum agrees."""
    logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]], dtype=np.float64)
    naive = np.array([[math.exp(v) for v in row] for row in logits])
    naive = naive / naive.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(M.softmax(logits), naive, atol=1e-12)


def test_softmax_uniform_on_equal_logits():
    logits = np.zeros((4, 8), dtype=np.float64)
    np.testing.assert_allclose(M.softmax(logits), np.full((4, 8), 1.0 / 8), atol=1e-12)


def test_softmax_stable_on_extreme_logits():
    logits = np.array([[1e4, -1e4, 0.0], [-1e4, -1e4, -1e4], [1e4, 1e4, 1e4]], dtype=np.float64)
    p = M.softmax(logits)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    # first row: class 0 dominates
    np.testing.assert_allclose(p[0], [1.0, 0.0, 0.0], atol=1e-12)
    # rows 1 & 2: all-equal logits -> uniform
    np.testing.assert_allclose(p[1], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)
    np.testing.assert_allclose(p[2], [1 / 3, 1 / 3, 1 / 3], atol=1e-12)


# ------------------------------------------------------------------- accuracy/nll/brier by hand
def test_accuracy_nll_brier_hand_values():
    # 2 samples, 3 classes; logits = log(prob) so softmax reproduces the probs exactly.
    p = np.array([[0.7, 0.2, 0.1],   # label 0 -> argmax 0 -> correct
                  [0.2, 0.5, 0.3]])  # label 2 -> argmax 1 -> wrong
    logits = logits_from_probs(p)
    labels = np.array([0, 2])

    assert M.accuracy(logits, labels) == pytest.approx(0.5, abs=1e-12)

    nll_hand = -(math.log(0.7) + math.log(0.3)) / 2          # 0.7803238...
    assert M.nll(logits, labels) == pytest.approx(nll_hand, abs=1e-12)

    # brier: s0 label0 -> (0.7-1)^2+0.2^2+0.1^2 = 0.14 ; s1 label2 -> 0.2^2+0.5^2+(0.3-1)^2 = 0.78
    brier_hand = (0.14 + 0.78) / 2                            # 0.46
    assert M.brier(logits, labels) == pytest.approx(brier_hand, abs=1e-12)


def test_nll_zero_when_perfectly_confident_correct():
    logits = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0]], dtype=np.float64)
    labels = np.array([0, 1])
    assert M.nll(logits, labels) == pytest.approx(0.0, abs=1e-9)
    assert M.brier(logits, labels) == pytest.approx(0.0, abs=1e-9)
    assert M.accuracy(logits, labels) == pytest.approx(1.0, abs=1e-12)


def test_brier_uniform_prediction():
    # uniform 4-class prediction, true class present: sum_k (p_k - oh_k)^2 = (1-1/4)^2 + 3*(1/4)^2
    logits = np.zeros((1, 4), dtype=np.float64)
    labels = np.array([2])
    expected = (1 - 0.25) ** 2 + 3 * (0.25 ** 2)             # 0.5625 + 0.1875 = 0.75
    assert M.brier(logits, labels) == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------------------------------------- ECE
def test_ece_zero_perfectly_calibrated():
    # All samples at confidence 0.8 (2-class), 80% correct -> single bin, acc==conf -> ECE 0.
    n = 100
    conf = np.full(n, 0.8)
    logits = two_class_logits_from_conf(conf)
    labels = np.zeros(n, dtype=int)
    labels[: int(0.2 * n)] = 1                                # 20% wrong (prediction is class 0)
    assert M.ece(logits, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_equals_abs_conf_minus_acc():
    # K samples all at confidence p, accuracy a -> ECE == |p - a| (single occupied bin).
    p, a, K = 0.8, 0.6, 100
    logits = two_class_logits_from_conf(np.full(K, p))
    labels = np.zeros(K, dtype=int)
    labels[int(a * K):] = 1                                   # a fraction correct (predict class 0)
    assert M.accuracy(logits, labels) == pytest.approx(a, abs=1e-12)
    assert M.ece(logits, labels) == pytest.approx(abs(p - a), abs=1e-9)


def test_ece_bounds():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(200, 10)) * 3.0
    labels = rng.integers(0, 10, size=200)
    e = M.ece(logits, labels)
    assert 0.0 <= e <= 1.0 and np.isfinite(e)


# ------------------------------------------------------------------ reliability_curve by hand
def test_reliability_curve_hand():
    # confidences 0.55 (two samples, one correct) and 0.95 (one correct). 15 equal-width bins.
    # 0.55 -> bin 8 (edges 8/15=0.5333 .. 9/15=0.6]; 0.95 -> bin 14 (14/15=0.9333 .. 1.0].
    confs = np.array([0.55, 0.55, 0.95])
    logits = two_class_logits_from_conf(confs)
    labels = np.array([0, 1, 0])                              # predictions all class 0
    rel = M.reliability_curve(logits, labels, n_bins=15)

    assert rel["counts"][8] == 2
    assert rel["accuracy"][8] == pytest.approx(0.5, abs=1e-12)
    assert rel["confidence"][8] == pytest.approx(0.55, abs=1e-9)

    assert rel["counts"][14] == 1
    assert rel["accuracy"][14] == pytest.approx(1.0, abs=1e-12)
    assert rel["confidence"][14] == pytest.approx(0.95, abs=1e-9)

    # empty bins: count 0, accuracy 0.0, confidence == bin center
    for i in range(15):
        if rel["counts"][i] == 0:
            assert rel["accuracy"][i] == 0.0
            assert rel["confidence"][i] == pytest.approx(rel["centers"][i], abs=1e-12)
    assert sum(rel["counts"]) == 3


# ------------------------------------------------------------------ per_class_accuracy by hand
def test_per_class_accuracy_hand():
    # pred index via a spiked logit; labels chosen to give a known per-class tally.
    preds = [0, 1, 1, 2]
    labels = np.array([0, 0, 1, 2])   # class0: {pred0 correct, pred1 wrong}=1/2 ; class1: 1/1 ; class2: 1/1
    logits = np.zeros((4, 10), dtype=np.float64)
    for i, pcls in enumerate(preds):
        logits[i, pcls] = 10.0
    out = M.per_class_accuracy(logits, labels, num_classes=10)

    assert out[0]["count"] == 2 and out[0]["accuracy"] == pytest.approx(0.5, abs=1e-12)
    assert out[1]["count"] == 1 and out[1]["accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert out[2]["count"] == 1 and out[2]["accuracy"] == pytest.approx(1.0, abs=1e-12)
    for c in range(3, 10):
        assert out[c]["count"] == 0 and out[c]["accuracy"] == 0.0
    # overall accuracy consistency: correct = 3/4
    assert M.accuracy(logits, labels) == pytest.approx(0.75, abs=1e-12)


# ------------------------------------------------------------------- risk_coverage closed form
def test_risk_coverage_monotone_closed_form():
    # confidence strictly decreasing; top 7 correct, bottom 3 wrong.
    L = np.array([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.5])   # descending -> descending confidence
    logits = np.stack([L, np.zeros_like(L)], axis=1)          # all predict class 0
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])         # first 7 correct, last 3 wrong
    n = 10
    rc = M.risk_coverage_curve(logits, labels)
    cov = np.asarray(rc["coverage"])
    risk = np.asarray(rc["risk"])

    correct_sorted = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=float)
    for c, r in zip(cov, risk):
        k = max(1, int(round(c * n)))
        assert r == pytest.approx(1.0 - correct_sorted[:k].mean(), abs=1e-12)

    # risk is non-decreasing here (well-ordered case)
    assert np.all(np.diff(risk) >= -1e-12)
    # risk at coverage == 1.0 equals 1 - accuracy
    j = int(np.argmin(np.abs(cov - 1.0)))
    assert cov[j] == pytest.approx(1.0)
    assert risk[j] == pytest.approx(1.0 - M.accuracy(logits, labels), abs=1e-12)
    assert risk[j] == pytest.approx(0.3, abs=1e-12)


def test_risk_at_full_coverage_equals_one_minus_accuracy_random():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(137, 10)) * 2.5
    labels = rng.integers(0, 10, size=137)
    rc = M.risk_coverage_curve(logits, labels, coverages=[1.0])
    assert rc["risk"][0] == pytest.approx(1.0 - M.accuracy(logits, labels), abs=1e-12)


# --------------------------------------------------- cross-check metrics.py vs sg reference impls
def test_crosscheck_against_sg_reference_random():
    from sg import reliability as R
    rng = np.random.default_rng(11)
    logits = rng.normal(size=(300, 10)) * 3.0
    labels = rng.integers(0, 10, size=300)

    np.testing.assert_allclose(M.softmax(logits), R.softmax(logits), atol=1e-12)
    assert M.accuracy(logits, labels) == pytest.approx(R.accuracy(logits, labels), abs=1e-12)
    assert M.nll(logits, labels) == pytest.approx(R.nll(logits, labels), abs=1e-12)
    assert M.brier(logits, labels) == pytest.approx(R.brier(logits, labels), abs=1e-12)
    assert M.ece(logits, labels, n_bins=15) == pytest.approx(R.ece(logits, labels, bins=15), abs=1e-12)

    # risk-coverage: metrics default coverages == sg COVERAGES; curves must match.
    m_rc = M.risk_coverage_curve(logits, labels)
    np.testing.assert_allclose(np.asarray(m_rc["coverage"]), R.COVERAGES, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m_rc["risk"]),
                               R.risk_coverage(logits, labels, R.COVERAGES), atol=1e-12)


# ------------------------------------------------------------- reproduce results/dashboard.json
def test_reproduce_dashboard_numbers(rerun_logits):
    logits, labels = rerun_logits
    acc = M.accuracy(logits, labels)
    ece = M.ece(logits, labels, n_bins=15)
    nll = M.nll(logits, labels)
    brier = M.brier(logits, labels)

    # published (results/dashboard.json, clean = rerun last-epoch)
    assert acc == pytest.approx(0.700, abs=2e-3), f"accuracy={acc}"
    assert ece == pytest.approx(0.1859, abs=2e-3), f"ece={ece}"
    assert nll == pytest.approx(1.466, abs=3e-3), f"nll={nll}"
    assert brier == pytest.approx(0.476, abs=3e-3), f"brier={brier}"


def test_dashboard_metrics_finite_and_in_range(rerun_logits):
    logits, labels = rerun_logits
    acc = M.accuracy(logits, labels)
    ece = M.ece(logits, labels)
    nll = M.nll(logits, labels)
    brier = M.brier(logits, labels)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= ece <= 1.0
    assert nll >= 0.0 and np.isfinite(nll)
    assert 0.0 <= brier <= 2.0 and np.isfinite(brier)

    # cross-check the dashboard numbers with the sg reference on the SAME logits
    from sg import reliability as R
    assert acc == pytest.approx(R.accuracy(logits, labels), abs=1e-12)
    assert ece == pytest.approx(R.ece(logits, labels), abs=1e-12)
    assert nll == pytest.approx(R.nll(logits, labels), abs=1e-12)
    assert brier == pytest.approx(R.brier(logits, labels), abs=1e-12)
