"""SoundGuard — calibration & selective-prediction metrics.

IMPORTANT: every metric uses the RAW softmax of the logits. Do NOT apply
temperature scaling or any post-hoc recalibration here — that would confound the
head-to-head (Linear vs MLP) calibration comparison, which is the whole point.

All functions accept either torch tensors or numpy arrays:
    logits: [N, num_classes]   labels: [N]
"""
import numpy as np


def _np(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def softmax(logits):
    logits = _np(logits).astype(np.float64)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def accuracy(logits, labels):
    probs, labels = softmax(logits), _np(labels).astype(int)
    return float((probs.argmax(1) == labels).mean())


def nll(logits, labels):
    """Mean negative log-likelihood of the true class."""
    probs, labels = softmax(logits), _np(labels).astype(int)
    p_true = probs[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())


def brier(logits, labels):
    """Multi-class Brier score: mean squared error between prob vector and one-hot."""
    probs, labels = softmax(logits), _np(labels).astype(int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _bin_mask(conf, lo, hi, first):
    return (conf >= lo) & (conf <= hi) if first else (conf > lo) & (conf <= hi)


def ece(logits, labels, n_bins=15):
    """Expected Calibration Error over `n_bins` equal-width confidence bins."""
    probs, labels = softmax(logits), _np(labels).astype(int)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e, n = 0.0, len(labels)
    for i in range(n_bins):
        m = _bin_mask(conf, edges[i], edges[i + 1], i == 0)
        if m.sum() > 0:
            e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability_curve(logits, labels, n_bins=15):
    """Per-bin accuracy & confidence for the reliability diagram (ideal: acc == conf)."""
    probs, labels = softmax(logits), _np(labels).astype(int)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        m = _bin_mask(conf, edges[i], edges[i + 1], i == 0)
        c = (edges[i] + edges[i + 1]) / 2
        centers.append(float(c))
        counts.append(int(m.sum()))
        accs.append(float(correct[m].mean()) if m.sum() else 0.0)
        confs.append(float(conf[m].mean()) if m.sum() else float(c))
    return {"centers": centers, "accuracy": accs, "confidence": confs, "counts": counts}


def risk_coverage_curve(logits, labels, coverages=None):
    """Selective prediction: accept the top-`coverage` fraction by confidence,
    report the error rate (risk) among accepted samples."""
    probs, labels = softmax(logits), _np(labels).astype(int)
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    correct_sorted = correct[np.argsort(-conf)]          # most confident first
    n = len(labels)
    if coverages is None:
        coverages = np.round(np.linspace(0.1, 1.0, 10), 2)
    cov_out, risk_out = [], []
    for c in coverages:
        k = max(1, int(round(c * n)))
        cov_out.append(float(c))
        risk_out.append(float(1.0 - correct_sorted[:k].mean()))
    return {"coverage": cov_out, "risk": risk_out}


def per_class_accuracy(logits, labels, num_classes=10):
    probs, labels = softmax(logits), _np(labels).astype(int)
    pred = probs.argmax(1)
    out = []
    for c in range(num_classes):
        m = labels == c
        out.append({"class": c,
                    "accuracy": float((pred[m] == c).mean()) if m.sum() else 0.0,
                    "count": int(m.sum())})
    return out