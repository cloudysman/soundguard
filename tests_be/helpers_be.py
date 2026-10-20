"""Audio helpers for the backend suite.

These live outside conftest.py on purpose. Every suite has its own conftest.py, so a test module that
does `from conftest import ...` resolves to whichever conftest landed on sys.path first — which makes
the suites impossible to run in one pytest invocation. A uniquely named module cannot collide.
"""
import io

import numpy as np
import soundfile as sf


def enc(signal, sr=16000, fmt="WAV", subtype=None):
    """Encode a float array to container bytes via libsndfile. fmt: WAV/FLAC/OGG/AIFF/..."""
    buf = io.BytesIO()
    sf.write(buf, np.asarray(signal, dtype="float32"), sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def tone(freq=440.0, secs=1.0, sr=16000, amp=0.3):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def noise(secs=1.0, sr=16000, amp=0.3, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(int(sr * secs))).astype("float32")


def silence(secs=1.0, sr=16000):
    return np.zeros(int(sr * secs), dtype="float32")


def chirp(f0=200.0, f1=4000.0, secs=1.0, sr=16000, amp=0.3):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return (amp * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / (2 * secs)) * t)).astype("float32")


def upload(client, data, name="clip.wav", ctype="audio/wav", **form):
    return client.post("/api/predict/upload", files={"file": (name, data, ctype)}, data=form or None)
