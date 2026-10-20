"""Conformal abstention, as a fifth family in the post-hoc comparison of Q2.

Q2 asks whether the confidence of a frozen classifier can be repaired after training, on a
corruption it did not see. The four families already tested all rescale the logits. A reviewer who
sees Gibbs and Candes cited will ask why the one procedure designed for distribution shift — a
conformal one — was not among them. This module supplies it, so the negative result of Q2 covers
the obvious escape hatch rather than leaving it open.

Two variants:

    split conformal (Vovk et al.; Angelopoulos and Bates tutorial)
        Fit a threshold on a held-out calibration split of CLEAN data so that a target fraction of
        it is retained. Apply the same threshold at test time. This is the honest transfer test: the
        threshold sees only clean data, exactly as a deployed rule would, and we measure what it does
        under corruption.

    adaptive conformal, split form (Gibbs and Candes, 2021)
        The threshold is a quantile of the non-conformity score. Under corruption the score
        distribution shifts, so the fixed clean quantile no longer retains its target fraction. The
        online update of Gibbs and Candes adjusts the quantile from the realised error, which needs a
        feedback signal the deployed system does not have; the split form below is therefore the
        version that is fairly comparable to the other post-hoc methods, and its drift under shift is
        the point, not a defect.

Neither variant touches the predicted category, so accuracy is unchanged, exactly as for the
temperature methods. What conformal changes is the ACCEPT/REJECT decision, so it is compared on the
rejection axis (retained accuracy at a matched coverage), not on a rescaled-confidence axis.
"""
from __future__ import annotations

import numpy as np

from . import reliability as R


def _nonconformity(logits: np.ndarray) -> np.ndarray:
    """One minus the softmax probability of the predicted class. Larger means less confident."""
    p = R.softmax(logits)
    return 1.0 - p.max(axis=1)


class SplitConformal:
    """A threshold on the non-conformity score, fitted on clean calibration data for a target coverage."""

    def __init__(self, target_coverage: float = 0.70) -> None:
        self.target = target_coverage
        self.threshold: float | None = None

    def fit(self, calib_logits: np.ndarray) -> "SplitConformal":
        s = _nonconformity(calib_logits)
        # accept the recordings whose score is below the target-coverage quantile
        self.threshold = float(np.quantile(s, self.target))
        return self

    def accept(self, logits: np.ndarray) -> np.ndarray:
        return _nonconformity(logits) <= self.threshold

    def realised_coverage(self, logits: np.ndarray) -> float:
        return float(self.accept(logits).mean())

    def selective_accuracy(self, logits: np.ndarray, labels: np.ndarray) -> float:
        a = self.accept(logits)
        if a.sum() == 0:
            return float("nan")
        return float((logits[a].argmax(1) == labels[a]).mean())


def evaluate(clean_calib_logits: np.ndarray,
             test_logits: np.ndarray, test_labels: np.ndarray,
             target_coverage: float = 0.70) -> dict:
    """Fit on clean, apply to a (possibly corrupted) test set, report what the rule realises.

    The comparison against the confidence baseline is at MATCHED coverage: read off the coverage
    the conformal rule actually achieves, then take the confidence threshold that accepts the same
    fraction, and compare the selective accuracy of the two. If conformal does not beat the plain
    confidence at matched coverage, it is one more post-hoc method that does not transfer.
    """
    cp = SplitConformal(target_coverage).fit(clean_calib_logits)
    cov = cp.realised_coverage(test_logits)
    sel_conf = cp.selective_accuracy(test_logits, test_labels)

    conf = R.softmax(test_logits).max(1)
    correct = test_logits.argmax(1) == test_labels
    order = np.argsort(-conf)
    k = max(1, int(round(cov * len(test_labels))))
    sel_msp = float(correct[order][:k].mean())

    return {
        "target_coverage": target_coverage,
        "realised_coverage": cov,          # drifts from target under corruption; that is the finding
        "coverage_drift": cov - target_coverage,
        "sel_acc_conformal": sel_conf,
        "sel_acc_msp_matched": sel_msp,    # plain confidence at the SAME coverage
        "conformal_minus_msp": (sel_conf - sel_msp) if sel_conf == sel_conf else float("nan"),
    }
