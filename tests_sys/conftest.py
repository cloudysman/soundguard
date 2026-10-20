"""Harness for the SYSTEM-level suites: the whole stack together, not one layer.

What this layer adds on top of tests_fe (browser), tests_be (endpoints) and tests_ai (model):
deployment integrity, end-to-end flows, cross-layer consistency of the numbers, performance, and
graceful degradation. In-process TestClient, model forced to CPU.

Fixtures:
    client        -> TestClient(app.app, raise_server_exceptions=False); a 500 comes back as a Response
    predictor     -> the shared Predictor (predictor.get_predictor())
    model_ok      -> bool
    ROOT          -> project root path
    dashboard     -> the parsed results/dashboard.json
    rerun_logits  -> (logits[N,10], labels[N]) last-epoch out-of-fold logits; the honest ground truth
    DEMO_IDS      -> the seven demo clip ids
    enc/tone      -> audio helpers (bytes / signal)
See tests_sys/SYSTEM_CONTRACT.md.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # CPU. NB "-1", not "" (an empty value breaks torch.load)

import glob
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
# Constants and audio helpers live in helpers_sys.py — see the note there on why not in this file.
from helpers_sys import CLASSES, DEMO_IDS, ROOT  # noqa: F401  (re-exported for convenience)


@pytest.fixture(scope="session")
def client():
    return TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture(scope="session")
def predictor():
    from predictor import get_predictor
    return get_predictor()


@pytest.fixture(scope="session")
def model_ok(predictor):
    return predictor.available


@pytest.fixture(scope="session")
def dashboard():
    p = os.path.join(ROOT, "results", "dashboard.json")
    if not os.path.exists(p):
        pytest.skip("results/dashboard.json absent")
    with open(p) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def rerun_logits():
    import torch
    logits, labels = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "rerun_results", "mlp_fold*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        logits.append(np.asarray(d["last_logits"], dtype=np.float64))
        labels.append(np.asarray(d["labels"]).astype(int))
    if not logits:
        pytest.skip("rerun_results/mlp_fold*.pt absent")
    return np.concatenate(logits), np.concatenate(labels)


