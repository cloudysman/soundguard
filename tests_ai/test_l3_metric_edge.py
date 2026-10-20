"""L3 (cao) — metric robustness / edge cases for metrics.py.

Goal: the calibration & selective-prediction metrics stay VALID and FINITE on adversarial /
degenerate inputs. Everything is analytical (no model, no audio) so it is deterministic and cheap.

References cross-checked before asserting: sg/reliability.py (same softmax/ece/nll/brier/risk_coverage
formulas) and eval.py (how reliability_curve / risk_coverage_curve / per_class_accuracy feed the
dashboard). A metric "disagreement" that is only a definition choice is documented, not failed.
"""
import numpy as np
import pytest

import metrics as M

K = 10  # num classes used throughout


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _finite(x):
    return np.all(np.isfinite(np.asarray(x, dtype=np.float64)))


def _rand_logits(n, k=K, scale=3.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k)).astype(np.float64) * scale


# --------------------------------------------------------------------------------------
# 1. Extreme logits -> softmax finite, rows sum to 1, no overflow / NaN
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("fill", [1e4, -1e4, 1e-8, -1e-8, 1e8, -1e8])
def test_softmax_extreme_uniform_rows_sum_to_one(fill):
    logits = np.full((5, K), fill, dtype=np.float64)
    p = M.softmax(logits)
    assert _finite(p)
    assert (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    # all-equal logits -> exactly uniform regardless of magnitude
    np.testing.assert_allclose(p, 1.0 / K, atol=1e-12)


def test_softmax_extreme_mixed_no_overflow():
    # one class saturated high, one saturated low, rest tiny — the classic overflow trap
    logits = np.zeros((4, K), dtype=np.float64)
    logits[:, 0] = 1e4
    logits[:, 1] = -1e4
    logits[:, 2] = 1e-8
    p = M.softmax(logits)
    assert _finite(p)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    # class 0 dominates completely, class 1 underflows to ~0 (never negative, never NaN)
    assert (p[:, 0] > 1 - 1e-9).all()
    assert (p[:, 1] >= 0).all() and (p[:, 1] < 1e-9).all()


def test_softmax_two_way_tie_saturated():
    # two classes at +1e4 -> exactly 0.5 / 0.5 after the max-subtraction, others underflow
    logits = np.full((3, K), -1e4, dtype=np.float64)
    logits[:, 0] = 1e4
    logits[:, 5] = 1e4
    p = M.softmax(logits)
    assert _finite(p)
    np.testing.assert_allclose(p[:, 0], 0.5, atol=1e-9)
    np.testing.assert_allclose(p[:, 5], 0.5, atol=1e-9)


@pytest.mark.parametrize("fill", [1e4, -1e4, 1e8])
def test_downstream_metrics_finite_on_extreme(fill):
    logits = _rand_logits(32, seed=1)
    logits[::3, 0] = fill          # spike a subset to an extreme value
    labels = (np.arange(32) % K).astype(int)
    for val, name in [
        (M.accuracy(logits, labels), "acc"),
        (M.nll(logits, labels), "nll"),
        (M.brier(logits, labels), "brier"),
        (M.ece(logits, labels), "ece"),
    ]:
        assert np.isfinite(val), name
    assert 0.0 <= M.accuracy(logits, labels) <= 1.0
    assert M.nll(logits, labels) >= 0.0
    assert 0.0 <= M.brier(logits, labels) <= 2.0
    assert 0.0 <= M.ece(logits, labels) <= 1.0


def test_nll_finite_when_true_class_underflows():
    # true class logit driven to -inf-ish; softmax there underflows to 0, nll clips at 1e-12
    logits = np.full((1, K), 1e4, dtype=np.float64)
    logits[0, 3] = -1e4
    labels = np.array([3])
    v = M.nll(logits, labels)
    assert np.isfinite(v)
    # -log(1e-12) ~= 27.63 is the clipped ceiling; we must be at/under it and positive
    assert 0.0 < v <= -np.log(1e-12) + 1e-9


# --------------------------------------------------------------------------------------
# 2. Degenerate shapes: single sample, single class present
# --------------------------------------------------------------------------------------
def test_single_sample_all_metrics_finite():
    logits = _rand_logits(1, seed=2)
    labels = np.array([int(logits.argmax(1)[0])])  # correct
    assert M.accuracy(logits, labels) == 1.0
    assert np.isfinite(M.nll(logits, labels))
    assert np.isfinite(M.brier(logits, labels))
    e = M.ece(logits, labels)
    assert 0.0 <= e <= 1.0
    rc = M.risk_coverage_curve(logits, labels)
    assert all(0.0 <= r <= 1.0 for r in rc["risk"])
    # single correct sample -> risk 0 at every coverage
    assert all(r == 0.0 for r in rc["risk"])
    rel = M.reliability_curve(logits, labels)
    assert sum(rel["counts"]) == 1


def test_single_class_present_all_labels_same():
    logits = _rand_logits(40, seed=3)
    labels = np.zeros(40, dtype=int)  # only class 0 appears
    assert np.isfinite(M.nll(logits, labels))
    assert 0.0 <= M.ece(logits, labels) <= 1.0
    pc = M.per_class_accuracy(logits, labels, num_classes=K)
    assert sum(p["count"] for p in pc) == 40
    assert pc[0]["count"] == 40
    # classes 1..9 never appear -> count 0
    assert all(p["count"] == 0 for p in pc[1:])


# --------------------------------------------------------------------------------------
# 3. All-identical confidences
# --------------------------------------------------------------------------------------
def test_all_identical_confidences_uniform_logits():
    # zero logits -> uniform -> every confidence == 1/K, argmax == 0 for all rows
    logits = np.zeros((50, K), dtype=np.float64)
    labels = (np.arange(50) % K).astype(int)
    p = M.softmax(logits)
    conf = p.max(1)
    np.testing.assert_allclose(conf, 1.0 / K, atol=1e-12)
    e = M.ece(logits, labels)
    acc = M.accuracy(logits, labels)
    # single occupied bin: ECE == |acc - conf| exactly
    np.testing.assert_allclose(e, abs(acc - 1.0 / K), atol=1e-12)
    rc = M.risk_coverage_curve(logits, labels)
    # risk at full coverage == 1 - accuracy
    np.testing.assert_allclose(rc["risk"][-1], 1.0 - acc, atol=1e-12)


# --------------------------------------------------------------------------------------
# 4. Exact bin-edge confidences — the >=/> first-bin boundary in _bin_mask
# --------------------------------------------------------------------------------------
def test_bin_mask_first_bin_includes_left_edge():
    # first=True uses >= on the low edge; every other bin uses > (half-open on the left)
    conf = np.array([0.0, 0.25, 0.5])
    assert list(M._bin_mask(conf, 0.0, 0.5, first=True)) == [True, True, True]
    assert list(M._bin_mask(conf, 0.0, 0.5, first=False)) == [False, True, True]
    # right edge is inclusive in both cases
    assert bool(M._bin_mask(np.array([0.5]), 0.0, 0.5, first=False)[0]) is True


def test_bin_edges_assign_each_sample_to_exactly_one_bin():
    # uniform 10-class logits -> conf == 0.1 which is exactly edges[1] for n_bins=10.
    # It must land in bin 0 (right-inclusive) and NOT be double counted / dropped.
    logits = np.zeros((7, K), dtype=np.float64)
    labels = (np.arange(7) % K).astype(int)
    rel = M.reliability_curve(logits, labels, n_bins=10)
    assert sum(rel["counts"]) == 7
    assert rel["counts"][0] == 7            # 0.1 falls in the first bin (0.0, 0.1]
    assert all(c == 0 for c in rel["counts"][1:])


def test_confidence_exactly_one_lands_in_last_bin():
    # near-one-hot logits give conf ~= 1.0; the top bin's right edge (1.0) is inclusive
    logits = np.full((5, K), -50.0, dtype=np.float64)
    logits[:, 2] = 50.0
    labels = np.full(5, 2, dtype=int)
    for nb in (1, 15, 100):
        rel = M.reliability_curve(logits, labels, n_bins=nb)
        assert sum(rel["counts"]) == 5
        assert rel["counts"][-1] == 5


# --------------------------------------------------------------------------------------
# 5. Ties in confidence for risk-coverage -> deterministic
# --------------------------------------------------------------------------------------
def test_risk_coverage_deterministic_under_ties():
    logits = np.zeros((30, K), dtype=np.float64)  # all confidences identical (ties everywhere)
    labels = (np.arange(30) % K).astype(int)
    a = M.risk_coverage_curve(logits, labels)
    b = M.risk_coverage_curve(logits, labels)
    assert a == b                                  # identical output across calls
    assert all(0.0 <= r <= 1.0 for r in a["risk"])
    assert all(np.isfinite(r) for r in a["risk"])


def test_risk_coverage_full_equals_one_minus_accuracy():
    logits = _rand_logits(120, seed=4)
    labels = (np.arange(120) % K).astype(int)
    rc = M.risk_coverage_curve(logits, labels)
    np.testing.assert_allclose(rc["risk"][-1], 1.0 - M.accuracy(logits, labels), atol=1e-12)


# --------------------------------------------------------------------------------------
# 6. n_bins in {1, 15, 100}: ECE finite & in [0,1]; reliability counts sum to N
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_bins", [1, 15, 100])
def test_ece_range_and_reliability_counts(n_bins):
    logits = _rand_logits(200, seed=5)
    labels = (np.arange(200) % K).astype(int)
    e = M.ece(logits, labels, n_bins=n_bins)
    assert np.isfinite(e) and 0.0 <= e <= 1.0
    rel = M.reliability_curve(logits, labels, n_bins=n_bins)
    assert sum(rel["counts"]) == 200
    assert len(rel["centers"]) == n_bins
    # with a single bin, ECE == |overall acc - overall mean conf|
    if n_bins == 1:
        p = M.softmax(logits)
        expected = abs((p.argmax(1) == labels).mean() - p.max(1).mean())
        np.testing.assert_allclose(e, expected, atol=1e-12)


def test_ece_in_unit_interval_on_extremes():
    for seed in range(6):
        logits = _rand_logits(64, scale=20.0, seed=seed)  # very peaky -> high confidence
        labels = np.random.default_rng(seed).integers(0, K, 64)
        e = M.ece(logits, labels)
        assert 0.0 <= e <= 1.0, (seed, e)


# --------------------------------------------------------------------------------------
# 7. labels accepted as python int / numpy / torch
# --------------------------------------------------------------------------------------
def test_label_types_python_numpy_torch_agree():
    logits = _rand_logits(20, seed=7)
    lab_np = (np.arange(20) % K).astype(int)
    lab_py = [int(x) for x in lab_np]
    variants = {"numpy": lab_np, "python": lab_py}
    try:
        import torch
        variants["torch"] = torch.tensor(lab_np)
    except ImportError:
        pass
    refs = {"acc": M.accuracy(logits, lab_np), "nll": M.nll(logits, lab_np),
            "brier": M.brier(logits, lab_np), "ece": M.ece(logits, lab_np)}
    for name, lab in variants.items():
        assert M.accuracy(logits, lab) == pytest.approx(refs["acc"]), name
        assert M.nll(logits, lab) == pytest.approx(refs["nll"]), name
        assert M.brier(logits, lab) == pytest.approx(refs["brier"]), name
        assert M.ece(logits, lab) == pytest.approx(refs["ece"]), name


def test_logits_accepted_as_torch_tensor():
    np_logits = _rand_logits(15, seed=8)
    labels = (np.arange(15) % K).astype(int)
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")
    t = torch.tensor(np_logits)
    np.testing.assert_allclose(M.softmax(t), M.softmax(np_logits), atol=1e-12)
    assert M.accuracy(t, labels) == pytest.approx(M.accuracy(np_logits, labels))


# --------------------------------------------------------------------------------------
# 8. per-class with a 0-count class -> 0.0  (documented convention)
# --------------------------------------------------------------------------------------
def test_per_class_zero_count_returns_zero():
    # labels only in classes 0..4; classes 5..9 have zero support
    logits = _rand_logits(50, seed=9)
    labels = (np.arange(50) % 5).astype(int)
    pc = M.per_class_accuracy(logits, labels, num_classes=K)
    assert len(pc) == K
    for c in range(5, K):
        assert pc[c]["count"] == 0
        assert pc[c]["accuracy"] == 0.0   # convention: undefined accuracy reported as 0.0
    assert sum(p["count"] for p in pc) == 50


def test_per_class_accuracy_matches_manual():
    # hand-built: class 0 has 2 samples, both predicted 0 (acc 1.0);
    #             class 1 has 2 samples, both predicted 0 (acc 0.0)
    logits = np.array([
        [9.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [9.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float64)
    labels = np.array([0, 0, 1, 1])
    pc = M.per_class_accuracy(logits, labels, num_classes=K)
    assert pc[0]["accuracy"] == 1.0 and pc[0]["count"] == 2
    assert pc[1]["accuracy"] == 0.0 and pc[1]["count"] == 2


# --------------------------------------------------------------------------------------
# 9. reliability_curve counts sum to N (random, several bin counts)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n_bins", [1, 2, 15, 37, 100])
def test_reliability_counts_sum_to_n_random(n_bins):
    logits = _rand_logits(150, seed=11)
    labels = (np.arange(150) % K).astype(int)
    rel = M.reliability_curve(logits, labels, n_bins=n_bins)
    assert sum(rel["counts"]) == 150
    # occupied bins have acc/conf in [0,1]; empty bins fall back to the center
    for cnt, acc, conf in zip(rel["counts"], rel["accuracy"], rel["confidence"]):
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= conf <= 1.0


# --------------------------------------------------------------------------------------
# 10. risk in [0,1] and non-decreasing for a well-ordered case
# --------------------------------------------------------------------------------------
def test_risk_coverage_nondecreasing_well_ordered():
    # Construct confidence that perfectly orders correctness: the most confident samples are
    # correct, the least confident are wrong. Risk over the accepted (top-conf) set must then
    # be monotonically non-decreasing as coverage grows.
    n = 100
    n_correct = 70
    logits = np.zeros((n, K), dtype=np.float64)
    labels = np.zeros(n, dtype=int)
    # correct samples: high margin (high confidence), predicted == label == 0
    for i in range(n_correct):
        margin = 8.0 - 4.0 * (i / n_correct)   # descending confidence, still class 0
        logits[i, 0] = margin
        labels[i] = 0
    # wrong samples: LOWER confidence than any correct one, predicted 0 but label 1
    for i in range(n_correct, n):
        margin = 3.5 - 3.0 * ((i - n_correct) / (n - n_correct))
        logits[i, 0] = margin
        labels[i] = 1
    fine = np.round(np.linspace(0.05, 1.0, 20), 4)
    rc = M.risk_coverage_curve(logits, labels, coverages=fine)
    risk = np.array(rc["risk"])
    assert ((risk >= -1e-12) & (risk <= 1.0 + 1e-12)).all()
    diffs = np.diff(risk)
    assert (diffs >= -1e-9).all(), risk           # non-decreasing
    # top-confident set is all-correct -> risk starts at 0
    assert risk[0] == pytest.approx(0.0, abs=1e-12)
    # full coverage risk == 1 - accuracy == 0.30
    np.testing.assert_allclose(risk[-1], 1.0 - n_correct / n, atol=1e-12)


# --------------------------------------------------------------------------------------
# 11. Cross-check metrics.py softmax against the reference sg/reliability.py
# --------------------------------------------------------------------------------------
def test_softmax_matches_reference_implementation():
    try:
        from sg import reliability as R
    except Exception:
        pytest.skip("sg.reliability reference not importable")
    logits = _rand_logits(64, scale=6.0, seed=12)
    np.testing.assert_allclose(M.softmax(logits), R.softmax(logits), atol=1e-12)
    labels = (np.arange(64) % K).astype(int)
    assert M.accuracy(logits, labels) == pytest.approx(R.accuracy(logits, labels))
    assert M.nll(logits, labels) == pytest.approx(R.nll(logits, labels))
    assert M.brier(logits, labels) == pytest.approx(R.brier(logits, labels))
    assert M.ece(logits, labels, n_bins=15) == pytest.approx(R.ece(logits, labels, bins=15))


# --------------------------------------------------------------------------------------
# 12. best-epoch logits are optimistic vs last-epoch logits
# --------------------------------------------------------------------------------------
def test_best_epoch_more_accurate_than_last(rerun_best, rerun_logits):
    best_logits, best_labels = rerun_best
    last_logits, last_labels = rerun_logits
    acc_best = M.accuracy(best_logits, best_labels)
    acc_last = M.accuracy(last_logits, last_labels)
    assert acc_best >= acc_last - 1e-9, (acc_best, acc_last)


def test_best_epoch_better_calibrated_than_last(rerun_best, rerun_logits):
    best_logits, best_labels = rerun_best
    last_logits, last_labels = rerun_logits
    ece_best = M.ece(best_logits, best_labels, n_bins=15)
    ece_last = M.ece(last_logits, last_labels, n_bins=15)
    nll_best = M.nll(best_logits, best_labels)
    nll_last = M.nll(last_logits, last_labels)
    # best epoch (early-stopped) should be at least as well-calibrated as the over-trained last epoch
    assert ece_best <= ece_last + 1e-9, (ece_best, ece_last)
    assert nll_best <= nll_last + 1e-9, (nll_best, nll_last)
