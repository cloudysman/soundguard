"""SoundGuard — real-inference predictor (Phase 5).

Ties the pieces together for serving:
    wav -> (optional slider corruption) -> features (frozen ResNet) -> head -> softmax
        -> quality heuristics (corruption.py) -> 3-state rejection (rejection.py)

Loaded as a lazy singleton via get_predictor(). If torch / torchaudio / the checkpoint
are missing, `available` stays False and `note` records why; the app then returns an explicit
503 rather than a fabricated answer. corruption.py and rejection.py are numpy/pure so they
import unconditionally.
"""
import math
import os

import corruption as C
import rejection as R

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


class Predictor:
    def __init__(self, head_name="mlp", ckpt_dir="checkpoints", config_path=None):
        self.available = False
        self.note = None
        self.head_name = head_name
        try:
            import torch
            from features import FeatureExtractor
            from heads import load_head

            config_path = config_path or os.path.join(ckpt_dir, "feature_config.json")
            ckpt = os.path.join(ckpt_dir, f"{head_name}_prod.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(ckpt)

            self._torch = torch
            self.extractor = FeatureExtractor(config_path)
            self.cfg = self.extractor.cfg
            self.classes = self.cfg.get("classes", CLASSES)
            self.head = load_head(head_name, ckpt,
                                  self.cfg.get("in_features", 512),
                                  self.cfg.get("num_classes", 10),
                                  self.extractor.device)
            self.available = True
        except Exception as e:                       # torch missing, no checkpoint, etc.
            self.note = f"{type(e).__name__}: {e}"

    # ---- continuous slider corruption (smooth UI response; mirrors the mock's intent) ----
    def apply_slider_corruption(self, wav, sr, noise=0.0, volume=0.0, clip=0.0, seed=0):
        import numpy as np
        w = np.asarray(wav, dtype=np.float32).copy()
        if volume > 0:
            w = w * (1.0 - 0.94 * float(volume))                 # quieter
        if noise > 0:
            snr_db = 30 - 34 * float(noise)
            sig_p = np.mean(w ** 2) + 1e-12
            n = np.random.default_rng(seed).normal(0, 1, len(w))
            n = n / (np.sqrt(np.mean(n ** 2)) + 1e-12)
            w = w + n * np.sqrt(sig_p / (10 ** (snr_db / 10)))
        if clip > 0:
            w = np.clip(w * (1 + 11 * float(clip)), -1, 1)       # drive + hard clip
        return np.clip(w, -1, 1).astype(np.float32)

    @staticmethod
    def _mel_for_display(log_mel):
        """[1, n_mels, time] in decibels -> list of rows of ints in 0..255, lowest mel band first.

        Quantised because this is for drawing, not for computation: 8 bits is past what a canvas
        resolves, and floats would triple the size of the response for no visible gain. The scale is
        per-recording min to max, which is the same normalisation step 7 of the pipeline applies, so
        what the caller draws is what the backbone is given."""
        mel = log_mel[0]
        lo, hi = float(mel.min()), float(mel.max())
        scaled = (mel - lo) / (hi - lo + 1e-6) * 255.0
        return [[int(v) for v in row] for row in scaled.round().cpu().tolist()]

    def predict_wav(self, wav, sr):
        """Full result dict. `mel` is the spectrogram the backbone receives, for display."""
        torch = self._torch
        log_mel = self.extractor.wav_to_log_mel(wav, sr)
        feats = self.extractor.log_mel_to_features(log_mel)
        with torch.no_grad():
            probs = torch.softmax(self.head(feats), dim=1)[0].cpu().tolist()

        order = sorted(enumerate(probs), key=lambda kv: kv[1], reverse=True)
        order = [[i, p] for i, p in order]
        conf = order[0][1]
        entropy = -sum(p * math.log(p) for p in probs if p > 0) / math.log(len(probs))

        q = C.quality_report(wav, sr)                            # snrDb / clipPct / levelDb
        dec = R.decide(q["snrDb"], q["levelDb"], q["clipPct"], conf, entropy)

        return {
            "probs": probs, "order": order, "conf": conf, "entropy": entropy,
            "snrDb": q["snrDb"], "levelDb": q["levelDb"], "clipPct": q["clipPct"],
            "status": dec["status"], "cause": dec["cause"],
            "mel": self._mel_for_display(log_mel),
        }


_PRED = None


def get_predictor():
    global _PRED
    if _PRED is None:
        _PRED = Predictor()
    return _PRED