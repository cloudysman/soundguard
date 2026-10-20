"""System harness smoke test — the whole stack answers, and the served numbers match the file."""
import json

from helpers_sys import enc, tone, upload


def test_stack_answers(client, model_ok):
    if not model_ok:
        import pytest; pytest.skip("model unavailable")
    r = client.post("/api/predict", json={"sample": "dog"})
    assert r.status_code == 200, r.text
    assert r.json()["order"][0][0] == 3          # dog_bark


def test_root_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["model_available"] in (True, False)


def test_served_dashboard_equals_file(client, dashboard):
    served = client.get("/api/dashboard").json()
    assert served == dashboard


def test_upload_roundtrip(client, model_ok):
    if not model_ok:
        import pytest; pytest.skip("model unavailable")
    r = upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav")
    assert r.status_code == 200
    json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
