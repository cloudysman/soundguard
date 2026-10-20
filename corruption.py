"""SoundGuard — audio corruptions + input-quality heuristics (Hien's module).

Two roles:
  1. corrupt(wav, sr, name, severity)  — apply a "common corruption" to a waveform,
     severity 1..5. Used to build the CORRUPTED evaluation set (the robustness story).
  2. quality_report / quality_verdict  — cheap input-quality heuristics (SNR, clipping,
     level). These feed the INPUT-QUALITY axis of the 3-state rejection logic, judging the
     recording itself, independent of model uncertainty.

numpy-only on purpose (no librosa / torchaudio) to avoid version conflicts.
`wav` is a 1-D float32 array in roughly [-1, 1]; `sr` is the sample rate in Hz.
"""
import numpy as np


# ---------- small helpers ----------
def _rng(seed):
    return np.random.default_rng(seed)


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _scale_noise_to_snr(wav, noise, snr_db):
    """Scale `noise` so that wav vs noise sits at the target SNR, return wav+noise."""
    sig_p = np.mean(wav ** 2) + 1e-12
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12)   # unit power
    noise_p = sig_p / (10 ** (snr_db / 10))
    return wav + noise * np.sqrt(noise_p)


# ---------- corruptions (severity 1..5, increasing) ----------

def add_gaussian_noise(wav, sr, severity=3, seed=0):
    snr_db = [20, 15, 10, 5, 0][severity - 1]
    noise = _rng(seed).normal(0, 1, size=len(wav))
    return _f32(np.clip(_scale_noise_to_snr(wav, noise, snr_db), -1, 1))


def add_background_noise(wav, sr, severity=3, seed=0):
    """Pink-ish ambient noise at decreasing SNR."""
    snr_db = [18, 13, 8, 3, -2][severity - 1]
    n = len(wav)
    white = _rng(seed).normal(0, 1, n)
    f = np.fft.rfftfreq(n, d=1.0 / sr)
    f[0] = f[1] if len(f) > 1 else 1.0                      # avoid /0 at DC
    spec = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spec, n=n)
    return _f32(np.clip(_scale_noise_to_snr(wav, pink, snr_db), -1, 1))


def gain_quiet(wav, sr, severity=3):
    factor = [0.7, 0.5, 0.3, 0.15, 0.06][severity - 1]
    return _f32(wav * factor)


def clip_distortion(wav, sr, severity=3):
    """Over-drive then hard-clip (saturation)."""
    drive = [1.5, 2.5, 4.0, 7.0, 12.0][severity - 1]
    return _f32(np.clip(wav * drive, -1, 1))


def time_mask(wav, sr, severity=3, seed=0):
    """Zero out one contiguous chunk (dropout / silence gap)."""
    frac = [0.05, 0.1, 0.2, 0.35, 0.5][severity - 1]
    n = len(wav)
    m = int(n * frac)
    out = wav.copy()
    if m > 0:
        start = int(_rng(seed).integers(0, max(1, n - m)))
        out[start:start + m] = 0.0
    return _f32(out)


def sample_dropout(wav, sr, severity=3, seed=0):
    """Random per-sample dropout (packet loss / glitching)."""
    p = [0.02, 0.05, 0.1, 0.2, 0.35][severity - 1]
    mask = _rng(seed).random(len(wav)) > p
    return _f32(wav * mask)


def bitcrush(wav, sr, severity=3):
    """Quantize to fewer bits (codec / low-quality recording artifact)."""
    bits = [10, 8, 6, 4, 3][severity - 1]
    levels = float(2 ** bits)
    return _f32(np.round(wav * levels) / levels)


def lowpass_muffle(wav, sr, severity=3):
    """Brick-wall low-pass via FFT (muffled / underwater)."""
    cutoff = [6000, 4000, 2500, 1500, 800][severity - 1]
    n = len(wav)
    spec = np.fft.rfft(wav)
    f = np.fft.rfftfreq(n, d=1.0 / sr)
    spec[f > cutoff] = 0.0
    return _f32(np.fft.irfft(spec, n=n))


def echo_reverb(wav, sr, severity=3):
    """Single delayed, decayed copy (room reverb / echo)."""
    delay_ms = [40, 80, 120, 180, 250][severity - 1]
    decay = [0.3, 0.4, 0.5, 0.6, 0.7][severity - 1]
    d = int(sr * delay_ms / 1000)
    out = wav.astype(np.float64).copy()
    if 0 < d < len(wav):
        out[d:] += decay * wav[:-d]
    return _f32(np.clip(out, -1, 1))


def pitch_shift(wav, sr, severity=3):
    """Resample-based speed/pitch change, re-fit to original length."""
    rate = [1.05, 0.95, 1.12, 0.88, 1.25][severity - 1]
    n = len(wav)
    idx = np.arange(0, n, rate)
    res = np.interp(idx, np.arange(n), wav)
    if len(res) < n:
        res = np.pad(res, (0, n - len(res)))
    else:
        res = res[:n]
    return _f32(res)


CORRUPTIONS = {
    "gaussian_noise": add_gaussian_noise,
    "background_noise": add_background_noise,
    "quiet": gain_quiet,
    "clipping": clip_distortion,
    "time_mask": time_mask,
    "dropout": sample_dropout,
    "bitcrush": bitcrush,
    "lowpass": lowpass_muffle,
    "reverb": echo_reverb,
    "pitch": pitch_shift,
}


def corrupt(wav, sr, name, severity=3, seed=0):
    """Dispatch to a named corruption. severity in 1..5."""
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption '{name}'. options: {list(CORRUPTIONS)}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be 1..5")
    wav = _f32(wav)
    fn = CORRUPTIONS[name]
    try:
        return fn(wav, sr, severity, seed=seed)
    except TypeError:                                       # deterministic fns take no seed
        return fn(wav, sr, severity)


def random_corruption(wav, sr, seed=0, severities=(1, 2, 3, 4, 5)):
    """Pick one random corruption + severity (for a mixed corrupted test set)."""
    r = _rng(seed)
    name = list(CORRUPTIONS)[int(r.integers(0, len(CORRUPTIONS)))]
    sev = int(r.choice(severities))
    return corrupt(wav, sr, name, sev, seed=seed), name, sev


# ---------- input-quality heuristics (rejection's input-quality axis) ----------

def level_dbfs(wav):
    """RMS level in dBFS (0 = full scale)."""
    return float(20 * np.log10(_rms(_f32(wav)) + 1e-12))


def clipping_pct(wav, thresh=0.99):
    """Percent of samples at/near full scale (hard-clipping indicator)."""
    wav = _f32(wav)
    return float(np.mean(np.abs(wav) >= thresh) * 100.0)


def estimate_snr_db(wav, sr, frame_ms=25):
    """Cheap SNR proxy: contrast between high-energy (signal) and low-energy (noise
    floor) frames. Works for clips that contain both active sound and quieter gaps."""
    wav = _f32(wav)
    fl = max(1, int(sr * frame_ms / 1000))
    nf = len(wav) // fl
    if nf < 2:
        return float(99.0)                                  # too short to estimate -> assume clean
    e = np.mean(wav[:nf * fl].reshape(nf, fl) ** 2, axis=1) + 1e-12
    noise = np.percentile(e, 10)
    signal = np.percentile(e, 90)
    return float(10 * np.log10(signal / noise))


def quality_report(wav, sr):
    """The three numbers the UI shows: snrDb, clipPct, levelDb."""
    wav = _f32(wav)
    return {
        "snrDb": round(estimate_snr_db(wav, sr), 1),
        "clipPct": round(clipping_pct(wav), 1),
        "levelDb": round(level_dbfs(wav), 1),
    }


def quality_verdict(wav, sr, snr_floor=8.0, clip_max=1.0, level_floor=-45.0):
    """Input-quality gate for rejection. Returns the report plus an ok flag and the
    list of reasons the input looks degraded (independent of the model)."""
    q = quality_report(wav, sr)
    causes = []
    if q["snrDb"] < snr_floor:
        causes.append("low SNR")
    if q["clipPct"] > clip_max:
        causes.append("clipping")
    if q["levelDb"] < level_floor:
        causes.append("too quiet")
    return {**q, "ok": len(causes) == 0, "causes": causes}