"""Constants and audio helpers for the system suite.

These live outside conftest.py on purpose. Every suite has its own conftest.py, so a test module that
does `from conftest import ...` resolves to whichever conftest landed on sys.path first — which makes
the suites impossible to run in one pytest invocation. A uniquely named module cannot collide.
"""
import io
import os

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_IDS = ("dog", "siren", "jack", "horn", "kids", "gun", "ac")
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


def enc(signal, sr=16000, fmt="WAV", subtype=None):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(signal, dtype="float32"), sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def tone(freq=440.0, secs=1.0, sr=16000, amp=0.3):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def upload(client, data, name="clip.wav", ctype="audio/wav", **form):
    return client.post("/api/predict/upload", files={"file": (name, data, ctype)}, data=form or None)
