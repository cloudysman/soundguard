"""SoundGuard — 3-state rejection decision (reliable / uncertain / reject).

Combines two axes:
  - input quality : SNR (dB), level (dBFS), clipping (%)   [from corruption.py heuristics]
  - model uncertainty : top-class confidence               [from the head's softmax]

The input-quality floors below are calibrated to the SCALE the corruption.py heuristics
actually produce on real UrbanSound8K audio, measured across the demo clips:
    clean clips   : SNR-contrast 3.3–25.6 dB,  RMS level -21 to -47 dBFS,  clip 0%
    heavy noise   : SNR-contrast 1.0–3.5 dB
    heavy attenuation : level down to ~-53 dBFS
estimate_snr_db is a temporal energy-contrast proxy, not an absolute SNR; on real audio its
clean range sits far below the 30 dB the earlier mock assumed, so the floors are set to the gap
between the clean and degraded clusters. This gate feeds only the live demo's status badge — it is
never used to compute any dashboard/report metric — and the estimator's clean/degraded overlap is
the crude-heuristic limitation discussed in the report; recalibrating the floors fixes the scale
mismatch, not the estimator's precision.
"""
from dataclasses import dataclass


@dataclass
class Thresholds:
    snr_poor: float = 2.0       # below this -> severe (heavy noise reads 1.0-1.9; clean stays above)
    snr_weak: float = 4.0       # below this -> mild   (only the steadiest clean clips dip here)
    level_poor: float = -50.0   # dBFS below this -> severe (attenuated input; quietest clean is -47)
    level_weak: float = -40.0   # a genuinely quiet-but-valid recording -> uncertain, not rejected
    clip_poor: float = 30.0     # clip% above this -> severe (percentages are scale-independent)
    clip_weak: float = 12.0
    conf_reject: float = 0.45   # confidence below this -> reject (model too unsure)
    conf_uncertain: float = 0.70


def decide(snr_db, level_db, clip_pct, conf, entropy=None, t: Thresholds = Thresholds()):
    """Return {'status', 'cause'}. Input defects take priority over the uncertainty axis."""
    poor = (snr_db < t.snr_poor) or (level_db < t.level_poor) or (clip_pct > t.clip_poor)
    weak = (snr_db < t.snr_weak) or (level_db < t.level_weak) or (clip_pct > t.clip_weak)

    if poor or conf < t.conf_reject:
        status = "reject"
    elif weak or conf < t.conf_uncertain:
        status = "uncertain"
    else:
        status = "reliable"

    bad = []
    if snr_db < t.snr_weak:
        bad.append(("noise", (t.snr_weak - snr_db) / max(t.snr_weak, 1e-6)))
    if level_db < t.level_weak:
        bad.append(("quiet", (t.level_weak - level_db) / max(abs(t.level_weak), 1e-6)))
    if clip_pct > t.clip_weak:
        bad.append(("clip", (clip_pct - t.clip_weak) / max(t.clip_weak, 1e-6)))
    bad.sort(key=lambda x: x[1], reverse=True)

    if bad:
        cause = bad[0][0]
    elif conf < t.conf_uncertain:
        cause = "uncert"
    else:
        cause = "clean"

    return {"status": status, "cause": cause}