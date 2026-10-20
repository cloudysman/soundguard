"""Shared harness for the SoundGuard BACKEND test suites (graduated: light -> maximum).

These use FastAPI's in-process TestClient (no browser, no uvicorn) so they are fast and can run in
parallel. The model is forced onto CPU so several test processes never contend for GPU VRAM.

Fixtures / helpers:
    client               -> TestClient(app.app); the real model + endpoints
    model_ok             -> bool, whether the predictor loaded (skip model-dependent asserts if False)
    enc(signal, sr, ...) -> bytes of an encoded audio file (WAV/FLAC/OGG/... via libsndfile)
    tone / noise / silence / chirp  -> quick signal generators (float32 in [-1, 1])
    upload(client, data, name, ctype) -> POST /api/predict/upload, returns the Response

See tests_be/BACKEND_CONTRACT.md for the endpoint contract and known formats.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU-only: no GPU VRAM contention across parallel runs
# NB: "-1" (not "") — an empty string leaves CUDA in a broken half-state where torch.load tries to
# deserialize onto "CUDA device 0" while device_count()==0. "-1" hides the GPU cleanly.

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app as app_module


@pytest.fixture(scope="session")
def client():
    # raise_server_exceptions=False so an uncaught 500 comes back as a Response (a finding to assert on),
    # instead of propagating into the test as an exception.
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture(scope="session")
def model_ok():
    from predictor import get_predictor
    return get_predictor().available


# Audio helpers live in helpers_be.py — see the note there on why they are not in this file.
@pytest.fixture
def make_audio():
    """Factory the suites can call: make_audio(tone(...), sr=16000, fmt='WAV') -> bytes."""
    from helpers_be import enc
    return enc
