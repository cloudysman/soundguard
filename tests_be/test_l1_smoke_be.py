"""L1 smoke / routing tests for the SoundGuard FastAPI backend.

Shallow liveness checks: health shape, static index, OpenAPI/docs, 404 routing,
dashboard sections, method-not-allowed on the static mount, and one prediction per
demo sample plus a single real WAV upload. See tests_be/BACKEND_CONTRACT.md.
"""
import json

import numpy as np
import pytest

from helpers_be import enc, tone, upload

# The seven bundled demo clips (app.DEMO_IDS / SampleId Literal).
SAMPLES = ["dog", "siren", "jack", "horn", "kids", "gun", "ac"]
STATUS_SET = {"reliable", "uncertain", "reject"}
CAUSE_SET = {"clean", "uncert", "noise", "quiet", "clip"}


def _strict(resp):
    """Parse a response body as strict JSON, rejecting NaN/Infinity tokens."""
    return json.loads(
        resp.text,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite token: {c}")),
    )


# ---- health ----------------------------------------------------------------
def test_health_shape(client):
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    body = _strict(r)
    assert set(body) == {"model_available", "note", "missing_demo_clips"}
    assert isinstance(body["model_available"], bool)
    assert isinstance(body["missing_demo_clips"], list)


# ---- static frontend -------------------------------------------------------
def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200, r.text
    ctype = r.headers.get("content-type", "")
    assert "text/html" in ctype, ctype
    assert r.content, "index body should be non-empty"


# ---- OpenAPI / docs --------------------------------------------------------
def test_openapi_json(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200, r.text
    body = _strict(r)
    assert body.get("info", {}).get("title") == "SoundGuard"
    # the three API routes are documented
    assert "/api/predict" in body["paths"]
    assert "/api/predict/upload" in body["paths"]
    assert "/api/dashboard" in body["paths"]


def test_docs_served(client):
    r = client.get("/docs")
    assert r.status_code == 200, r.text
    assert "text/html" in r.headers.get("content-type", "")


# ---- routing ---------------------------------------------------------------
def test_unknown_route_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404, r.text


# ---- dashboard -------------------------------------------------------------
def test_dashboard_sections(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text
    body = _strict(r)
    for section in ("reliability", "riskCoverage", "classAcc", "metrics"):
        assert section in body, f"missing dashboard section {section!r}"
    assert body["_corrupted_pending"] is False


def test_dashboard_post_405(client):
    """The dashboard is GET-only; the static mount rejects POST with 405."""
    r = client.post("/api/dashboard")
    assert r.status_code == 405, r.text


# ---- predictions over the seven demo clips ---------------------------------
@pytest.mark.parametrize("sample", SAMPLES)
def test_predict_sample(client, model_ok, sample):
    if not model_ok:
        pytest.skip("model unavailable")
    r = client.post("/api/predict", json={"sample": sample})
    assert r.status_code != 500, r.text
    assert r.status_code == 200, r.text
    body = _strict(r)
    assert len(body["probs"]) == 10
    assert all(np.isfinite(body["probs"]))
    assert abs(sum(body["probs"]) - 1.0) < 1e-4
    assert body["status"] in STATUS_SET
    assert body["cause"] in CAUSE_SET


# ---- one real WAV upload ---------------------------------------------------
def test_upload_tone_classifies(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    r = upload(client, enc(tone(secs=1.0)), "tone.wav", "audio/wav")
    assert r.status_code != 500, r.text
    assert r.status_code == 200, r.text
    body = _strict(r)
    assert len(body["probs"]) == 10
    assert all(np.isfinite(body["probs"]))
    assert body["status"] in STATUS_SET
