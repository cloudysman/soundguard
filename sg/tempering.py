"""Two ways to rescale the logits of a frozen classifier, and why the second one can do more.

Dividing the logits of ONE recording by a positive scalar cannot change which logit is largest.
Both methods below therefore leave the predicted category, and so the accuracy, EXACTLY unchanged.
This is an identity, not an empirical finding, and it is what removes accuracy from the axis of
competition: neither method can buy a single point of it.

    global temperature (Guo et al., 2017)
        one scalar for the whole dataset. Because it is the SAME scalar everywhere, it is a
        monotone map of every logit vector by the same amount, and it can therefore do very little
        to the ORDER in which recordings are ranked by confidence. It corrects the scale.

    scale-restored confidence (this work)
        a scalar PER RECORDING, predicted from the front-end statistics of that recording. Because
        the scalar differs between recordings, the map is NOT the same everywhere, and two
        recordings CAN exchange their order of confidence. It can therefore correct the ordering
        as well as the scale — and it does so using information that the network never saw,
        because the preprocessing discarded it.

The classifier is frozen throughout. Only the temperature is fitted.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels)


class GlobalTemperature:
    """One temperature for the whole dataset. The baseline of Guo et al."""

    def __init__(self) -> None:
        self.log_t = torch.zeros(1, requires_grad=True, device=DEVICE)

    def fit(self, logits: np.ndarray, labels: np.ndarray, iters: int = 100) -> "GlobalTemperature":
        lg = torch.from_numpy(logits).float().to(DEVICE)
        lb = torch.from_numpy(labels).long().to(DEVICE)
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=iters)

        def step():
            opt.zero_grad()
            loss = _nll(lg / self.log_t.exp(), lb)
            loss.backward()
            return loss

        opt.step(step)
        return self

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp().item())

    def apply(self, logits: np.ndarray, stats: np.ndarray | None = None) -> np.ndarray:
        return logits / self.temperature

    def to(self, device):
        self.log_t = self.log_t.detach().to(device).requires_grad_(True)
        return self


class ScaleRestoredConfidence:
    """A temperature predicted, per recording, from the front-end statistics of that recording.

    T(s) = softplus(g(s)) + t_min, with g a small multi-layer perceptron. The softplus keeps the
    temperature positive, so the predicted category is unchanged for every recording; t_min keeps
    it away from zero.
    """

    def __init__(self, n_stats: int, hidden: int = 32, t_min: float = 0.25, seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.t_min = t_min
        self.net = nn.Sequential(
            nn.Linear(n_stats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(DEVICE)
        # start near a temperature of one, so the untrained model is close to a no-operation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.55)

    def _temp(self, s: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.net(s)) + self.t_min

    def fit(self, logits: np.ndarray, labels: np.ndarray, stats: np.ndarray,
            epochs: int = 60, lr: float = 1e-2, weight_decay: float = 1e-4,
            batch: int = 4096) -> "ScaleRestoredConfidence":
        lg = torch.from_numpy(logits).float().to(DEVICE)
        lb = torch.from_numpy(labels).long().to(DEVICE)
        st = torch.from_numpy(stats).float().to(DEVICE)

        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = len(lb)

        for _ in range(epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = _nll(lg[idx] / self._temp(st[idx]), lb[idx])
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def temperatures(self, stats: np.ndarray) -> np.ndarray:
        t = self._temp(torch.from_numpy(stats).float().to(DEVICE))
        return t.squeeze(-1).cpu().numpy()

    @torch.no_grad()
    def apply(self, logits: np.ndarray, stats: np.ndarray) -> np.ndarray:
        t = self._temp(torch.from_numpy(stats).float().to(DEVICE))
        lg = torch.from_numpy(logits).float().to(DEVICE)
        return (lg / t).cpu().numpy()


class RankedScaleRestoredConfidence(ScaleRestoredConfidence):
    """The same per-recording temperature, fitted to a RANKING objective instead of the likelihood.

    Fitting the temperature by the negative log-likelihood makes it correct the SCALE, because that
    is what the likelihood is dominated by. It does not make it correct the ORDERING, and the
    measurement shows that it does not: on corruptions never seen while fitting, the likelihood-
    fitted temperature improves the calibration error and DEGRADES the error-detection area.

    The ordering is what a rejection rule actually uses, so fit it directly. For every pair of one
    correct and one incorrect prediction, the confidence of the incorrect one should be lower. The
    loss below is the smooth relaxation of exactly that, and it is the same quantity the
    error-detection area measures.

    The temperature is still a positive scalar per recording, so the predicted category, and hence
    the accuracy, is still unchanged for every recording.
    """

    def fit(self, logits: np.ndarray, labels: np.ndarray, stats: np.ndarray,
            epochs: int = 60, lr: float = 1e-2, weight_decay: float = 1e-4,
            batch: int = 4096, margin: float = 0.05,
            lam: float = 0.2) -> "RankedScaleRestoredConfidence":
        lg = torch.from_numpy(logits).float().to(DEVICE)
        lb = torch.from_numpy(labels).long().to(DEVICE)
        st = torch.from_numpy(stats).float().to(DEVICE)
        correct = (lg.argmax(1) == lb)                    # fixed: the temperature cannot change it

        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = len(lb)

        for _ in range(epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                ok = correct[idx]
                if ok.sum() == 0 or (~ok).sum() == 0:
                    continue
                opt.zero_grad()
                conf = torch.softmax(lg[idx] / self._temp(st[idx]), dim=1).max(1).values
                # every (correct, incorrect) pair in the batch, at once
                diff = conf[~ok].unsqueeze(0) - conf[ok].unsqueeze(1) + margin
                rank_loss = nn.functional.softplus(diff).mean()
                # a light likelihood term keeps the scale sane while the ranking is fixed
                loss = rank_loss + lam * _nll(lg[idx] / self._temp(st[idx]), lb[idx])
                loss.backward()
                opt.step()
            sched.step()
        return self


class FailureDetector:
    """Detect the model's own errors, instead of trying to repair its confidence.

    The two temperature methods above force every piece of evidence through a single positive
    scalar multiplying the logits. That funnel can rescale the confidence; it has nowhere near
    enough freedom to say "this recording is broken". The measurement bears this out: on a
    corruption never seen while fitting, both of them make error detection WORSE than doing
    nothing.

    So stop trying to repair the confidence and detect the failure directly. This model takes the
    confidence of the classifier AND the front-end statistics that the preprocessing discarded, and
    predicts whether the prediction is wrong. It never touches the logits, so the predicted category
    and the accuracy are untouched by construction, exactly as before — but it is free to combine
    the evidence in any way, rather than through a scalar.
    """

    def __init__(self, n_stats: int, hidden: int = 64, seed: int = 42) -> None:
        torch.manual_seed(seed)
        # inputs: the confidence, the entropy, the logit margin, plus the front-end statistics
        self.net = nn.Sequential(
            nn.Linear(3 + n_stats, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(DEVICE)

    @staticmethod
    def _score_features(logits: np.ndarray) -> np.ndarray:
        """The three quantities any failure detector should already have from the classifier."""
        z = torch.from_numpy(logits).float()
        p = torch.softmax(z, dim=1)
        conf = p.max(1).values
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1) / np.log(p.shape[1])
        top2 = z.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        return torch.stack([conf, ent, margin], dim=1).numpy()

    def _x(self, logits: np.ndarray, stats: np.ndarray) -> torch.Tensor:
        f = np.concatenate([self._score_features(logits), stats], axis=1)
        return torch.from_numpy(f).float().to(DEVICE)

    def fit(self, logits: np.ndarray, labels: np.ndarray, stats: np.ndarray,
            epochs: int = 60, lr: float = 1e-3, weight_decay: float = 1e-4,
            batch: int = 4096) -> "FailureDetector":
        x = self._x(logits, stats)
        wrong = torch.from_numpy(
            (logits.argmax(1) != labels).astype(np.float32)).to(DEVICE).unsqueeze(1)

        # the errors are the minority, so weight them up
        pos = float(wrong.mean().item())
        w = torch.tensor([(1 - pos) / max(pos, 1e-6)], device=DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=w)

        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        n = len(x)
        for _ in range(epochs):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                crit(self.net(x[idx]), wrong[idx]).backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def error_score(self, logits: np.ndarray, stats: np.ndarray) -> np.ndarray:
        """Higher means more likely to be wrong. Used directly as the rejection score."""
        return self.net(self._x(logits, stats)).squeeze(-1).cpu().numpy()


def check_accuracy_is_unchanged(before: np.ndarray, after: np.ndarray) -> bool:
    """The predicted category must be identical for every recording. This is an identity; if it
    ever fails, the temperature has gone negative and there is a bug."""
    return bool((before.argmax(1) == after.argmax(1)).all())
