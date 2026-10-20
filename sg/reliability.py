"""Reliability measures. All of them operate on raw logits and labels.

Correctness is measured by the accuracy. Everything else here measures whether the confidence of
the classifier can be trusted, and the three questions are kept apart on purpose:

    the SCALE     — does a reported confidence of 0.8 mean the model is right 80 percent of the
                    time?  Measured by the expected calibration error.
    the ORDERING  — does the confidence rank the mistakes below the correct predictions?
                    Measured by the area under the receiver operating characteristic curve of the
                    confidence used as a detector of the model's own errors. This measure does not
                    depend on how many mistakes there are, which is why it can separate a genuine
                    loss of ordering from the mechanical effect of a higher error rate.
    the OPERATING POINT — if the system may decline to answer, how much better is it on the
                    recordings it does answer?  Measured by the risk-coverage curve and its area.
"""
from __future__ import annotations

import numpy as np

COVERAGES = np.round(np.arange(0.1, 1.001, 0.1), 2)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((softmax(logits).argmax(1) == labels).mean())


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    p = softmax(logits)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def brier(logits: np.ndarray, labels: np.ndarray) -> float:
    p = softmax(logits)
    oh = np.zeros_like(p)
    oh[np.arange(len(labels)), labels] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def ece(logits: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error over `bins` equal-width bins of confidence."""
    p = softmax(logits)
    conf, correct = p.max(1), (p.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out, n = 0.0, len(labels)
    for i in range(bins):
        m = (conf >= edges[i]) & (conf <= edges[i + 1]) if i == 0 else \
            (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            out += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a random positive outranks a random negative. Ties take the average rank."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(positive, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def error_detection_auroc(logits: np.ndarray, labels: np.ndarray) -> float:
    """The confidence, used as a detector of the model's own errors. Base-rate invariant."""
    p = softmax(logits)
    return auroc(-p.max(1), p.argmax(1) != labels)


def risk_coverage(logits: np.ndarray, labels: np.ndarray,
                  coverages: np.ndarray = COVERAGES) -> np.ndarray:
    """Selective risk: the error rate among the most confident `coverage` fraction."""
    p = softmax(logits)
    correct = (p.argmax(1) == labels)[np.argsort(-p.max(1))].astype(float)
    n = len(labels)
    return np.array([1.0 - correct[: max(1, int(round(c * n)))].mean() for c in coverages])


def aurc(logits: np.ndarray, labels: np.ndarray) -> float:
    r = risk_coverage(logits, labels)
    return float(np.trapz(r, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))


def selective_efficiency(logits: np.ndarray, labels: np.ndarray) -> float:
    """The area under the risk-coverage curve, normalised so that a random ordering scores 0 and a
    perfect ordering scores 1. This is a normalised form of the excess area of Geifman et al.

    It does NOT remove the dependence on the base error rate: both reference curves move when the
    error rate moves, so a fall in this quantity is not on its own evidence that the ordering has
    deteriorated. Use `error_detection_auroc` for that.
    """
    err = 1.0 - accuracy(logits, labels)
    area = lambda r: float(np.trapz(r, COVERAGES) / (COVERAGES[-1] - COVERAGES[0]))
    got = aurc(logits, labels)
    rnd = area(np.full_like(COVERAGES, err))
    orc = area(np.array([max(0.0, (c - (1 - err)) / c) for c in COVERAGES]))
    return float((rnd - got) / (rnd - orc) * 100.0)


def report(logits: np.ndarray, labels: np.ndarray) -> dict:
    p = softmax(logits)
    return {
        "acc": accuracy(logits, labels),
        "mean_conf": float(p.max(1).mean()),
        "ece": ece(logits, labels),
        "nll": nll(logits, labels),
        "brier": brier(logits, labels),
        "aurc": aurc(logits, labels),
        "sel_eff": selective_efficiency(logits, labels),
        "auroc": error_detection_auroc(logits, labels),
    }


def stickiness(clean: dict, corrupted: dict) -> float:
    """How much of the damage the confidence notices.

    rho = (fall in mean confidence) / (fall in accuracy).
    rho = 1 means the confidence tracks the damage exactly. rho near 0 means it is blind to it.
    """
    d_acc = clean["acc"] - corrupted["acc"]
    if abs(d_acc) < 1e-9:
        return float("nan")
    return float((clean["mean_conf"] - corrupted["mean_conf"]) / d_acc)
