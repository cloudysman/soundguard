"""Backend harness smoke test — proves TestClient + CPU model + audio encoders work."""
import numpy as np

from helpers_be import enc, tone, upload


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "model_available" in r.json()


def test_real_wav_classifies(client, model_ok):
    if not model_ok:
        import pytest; pytest.skip("model unavailable")
    r = upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav")
    assert r.status_code == 200, r.text
    assert len(r.json()["probs"]) == 10


def test_response_is_valid_json_and_finite(client, model_ok):
    """The response body must be strict JSON with finite numbers (NaN/Inf would break a browser)."""
    if not model_ok:
        import pytest; pytest.skip("model unavailable")
    import json
    r = upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav")
    # strict parse: json.loads rejects NaN/Infinity only with parse_constant raising
    body = json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite {c}")))
    assert all(np.isfinite(body["probs"]))
    assert np.isfinite(body["conf"])
