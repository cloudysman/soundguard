"""L5 — resilience / graceful degradation of the whole stack.

The thesis of this project is that a fabricated answer is worse than no answer (appendix F). This
layer removes, one at a time, each artifact the deployment depends on (results/dashboard.json, a
demo clip, the whole samples/ dir, the model itself) and asserts the system degrades the way the
contract promises: an explicit 503/4xx carrying {detail}, never a 500 with a stack trace and never
an invented prediction. It also checks the app is not wedged by a failure, that /api/health keeps
answering while the model is down, and that no error body leaks a traceback or a host path.

Nothing here deletes a real file: every test monkeypatches app.py's own path constants / factory
(BASE_DIR, SAMPLES_DIR, get_predictor) and an autouse fixture restores them afterwards.
"""
import json
import os
import re
import shutil

import pytest

import app as app_module
from helpers_sys import CLASSES, DEMO_IDS, ROOT, enc, tone, upload

PREDICT_ROUTES = ("/api/predict", "/api/predict/upload")


# --------------------------------------------------------------------------- helpers
def sj(r):
    """Parse a response body as STRICT json: NaN/Infinity/-Infinity are not JSON and must not appear."""
    return json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def detail_of(r):
    body = sj(r)
    assert isinstance(body, dict), f"error body must be an object, got {type(body).__name__}"
    assert "detail" in body, f"error body must carry {{detail}}: {body!r}"
    return body["detail"]


# A response body must never expose the innards of the host. Project-relative mentions
# ("samples/", "results/dashboard.json", "requirements.txt") are documentation, and allowed.
_LEAK_PATTERNS = (
    r"Traceback \(most recent call last\)",
    r'File "[^"]+", line \d+',
    r"site-packages",
    r"(?i)\b(?:[A-Z]:\\|\\\\)",          # windows absolute / UNC path
    r"(?<![\w.])/(?:home|usr|root|etc|var|tmp|opt|Users)/",   # posix absolute path outside the project
    r"(?i)0x[0-9a-f]{6,}",               # object repr
    r"<[a-zA-Z_.]+ object at ",
)


def assert_no_leak(r):
    text = r.text
    for pat in _LEAK_PATTERNS:
        m = re.search(pat, text)
        assert m is None, f"error body leaks host internals ({pat!r} -> {m.group(0)!r}): {text[:400]}"


def assert_clean_error(r, status=None):
    """Not a 500; a machine-readable {detail}; strict JSON; no leak."""
    assert r.status_code != 500, f"a 500 is a defect, body={r.text[:400]}"
    assert 400 <= r.status_code < 600, r.status_code
    if status is not None:
        assert r.status_code == status, f"expected {status}, got {r.status_code}: {r.text[:300]}"
    d = detail_of(r)
    assert isinstance(d, (str, list)) and d, f"detail must be non-empty: {d!r}"
    assert_no_leak(r)
    return d


class _DownPredictor:
    """What predictor.Predictor looks like when torch / the checkpoint are missing."""
    available = False
    note = "ModuleNotFoundError: No module named 'torch'"

    def apply_slider_corruption(self, *a, **k):
        raise AssertionError("apply_slider_corruption called while the model is unavailable")

    def predict_wav(self, *a, **k):
        raise AssertionError("predict_wav called while the model is unavailable")


def kill_model(monkeypatch):
    monkeypatch.setattr(app_module, "get_predictor", lambda: _DownPredictor())


def point_base_dir(monkeypatch, tmp_path, dashboard_bytes=None):
    """Repoint app.BASE_DIR (used only to resolve results/dashboard.json at request time)."""
    if dashboard_bytes is not None:
        (tmp_path / "results").mkdir(parents=True, exist_ok=True)
        (tmp_path / "results" / "dashboard.json").write_bytes(dashboard_bytes)
    monkeypatch.setattr(app_module, "BASE_DIR", str(tmp_path))


def point_samples_dir(monkeypatch, path):
    monkeypatch.setattr(app_module, "SAMPLES_DIR", str(path))


@pytest.fixture(autouse=True)
def _restore_app_state():
    """Belt-and-braces: whatever a test patches, app.py is byte-identical afterwards."""
    saved = {k: getattr(app_module, k) for k in ("BASE_DIR", "SAMPLES_DIR", "get_predictor")}
    yield
    for k, v in saved.items():
        setattr(app_module, k, v)


def good_predict(client):
    return client.post("/api/predict", json={"sample": "dog"})


# =========================================================================== dashboard.json missing
def test_dashboard_missing_is_503_with_detail(client, monkeypatch, tmp_path):
    point_base_dir(monkeypatch, tmp_path)                    # no results/ at all
    r = client.get("/api/dashboard")
    d = assert_clean_error(r, 503)
    assert "eval.py" in d, f"the 503 must tell the operator how to fix it: {d!r}"


def test_dashboard_missing_body_is_only_detail(client, monkeypatch, tmp_path):
    """The frontend's empty-state contract: a failed /api/dashboard must not hand back a body the
    charts could mistake for data (soundguard-app.js does Object.assign(SG, await res.json()))."""
    point_base_dir(monkeypatch, tmp_path)
    r = client.get("/api/dashboard")
    body = sj(r)
    assert set(body) == {"detail"}, f"error body must carry nothing but detail: {body!r}"
    for k in ("reliability", "riskCoverage", "classAcc", "metrics", "accuracy", "ece"):
        assert k not in body, f"fabricated dashboard key {k!r} in an error body"


def test_dashboard_missing_never_fabricates_numbers(client, monkeypatch, tmp_path):
    point_base_dir(monkeypatch, tmp_path)
    text = client.get("/api/dashboard").text
    for honest in ("0.7", "0.186", "1.466", "0.476", "0.527", "0.323"):
        assert honest not in text, f"the error body echoes a headline number ({honest})"


def test_dashboard_missing_leaves_frontend_empty_state_reachable(client, monkeypatch, tmp_path):
    """The status must be non-ok so app.js's `if (!res.ok) throw` fires showChartsUnavailable()."""
    point_base_dir(monkeypatch, tmp_path)
    assert not (200 <= client.get("/api/dashboard").status_code < 300)
    js = client.get("/soundguard-app.js")
    assert js.status_code == 200, "the static frontend must keep serving while results are missing"
    assert "showChartsUnavailable" in js.text and "chart-empty" in js.text
    assert "res.ok" in js.text


def test_dashboard_recovers_when_the_file_returns(client, monkeypatch, tmp_path):
    point_base_dir(monkeypatch, tmp_path)
    assert client.get("/api/dashboard").status_code == 503
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join(ROOT, "results", "dashboard.json"),
                str(tmp_path / "results" / "dashboard.json"))
    r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text
    assert sj(r) == json.load(open(os.path.join(ROOT, "results", "dashboard.json")))


# =========================================================================== dashboard.json corrupt
# app.dashboard() wraps the read in try/except and rejects the non-standard NaN/Infinity tokens via
# parse_constant, so every unreadable state below degrades to a 503 {detail} rather than a 500.
def test_dashboard_corrupt_json_is_a_clean_failure(client, monkeypatch, tmp_path):
    point_base_dir(monkeypatch, tmp_path, dashboard_bytes=b'{"metrics": {"accuracy": 0.7,')
    r = client.get("/api/dashboard")
    assert_clean_error(r)


def test_dashboard_corrupt_json_fails_as_json_not_as_a_crash(client, monkeypatch, tmp_path):
    """Seen from the client's side: the failure must arrive as JSON the frontend can parse."""
    point_base_dir(monkeypatch, tmp_path, dashboard_bytes=b"<html>not json at all</html>")
    r = client.get("/api/dashboard")
    assert_no_leak(r)
    assert r.status_code != 500
    assert r.headers["content-type"].startswith("application/json")


def test_dashboard_with_nan_token_is_a_clean_failure(client, monkeypatch, tmp_path):
    point_base_dir(monkeypatch, tmp_path, dashboard_bytes=b'{"metrics": {"accuracy": NaN}}')
    r = client.get("/api/dashboard")
    assert r.status_code != 500, r.text[:300]
    if r.status_code == 200:
        sj(r)                        # if it is served, it must at least be strict, finite JSON
    else:
        assert_clean_error(r)


def test_dashboard_empty_file_is_a_clean_failure(client, monkeypatch, tmp_path):
    """A zero-byte dashboard.json is the classic half-written-by-eval.py state."""
    point_base_dir(monkeypatch, tmp_path, dashboard_bytes=b"")
    r = client.get("/api/dashboard")
    assert_clean_error(r)


def test_corrupt_dashboard_does_not_wedge_the_app(client, monkeypatch, tmp_path):
    """Even where the corrupt file 500s, the failure must not poison the next request."""
    point_base_dir(monkeypatch, tmp_path, dashboard_bytes=b"{oops")
    for _ in range(3):
        client.get("/api/dashboard")
    monkeypatch.undo()
    r = client.get("/api/dashboard")
    assert r.status_code == 200, f"the app stayed broken after a corrupt-file request: {r.text[:200]}"
    assert client.get("/api/health").status_code == 200


# =========================================================================== demo clips missing
def test_missing_demo_clip_is_503_naming_the_clip(client, monkeypatch, tmp_path):
    """Only 'dog' is removed; the detail must name it so an operator knows what to restore."""
    stage = tmp_path / "samples"
    stage.mkdir()
    for cid in DEMO_IDS:
        if cid != "dog":
            shutil.copy(os.path.join(ROOT, "samples", f"{cid}.wav"), str(stage / f"{cid}.wav"))
    point_samples_dir(monkeypatch, stage)
    r = client.post("/api/predict", json={"sample": "dog"})
    d = assert_clean_error(r, 503)
    assert "dog" in d, f"the 503 must name the missing clip: {d!r}"
    assert "samples" in d.lower()


def test_missing_demo_clip_returns_no_prediction(client, monkeypatch, tmp_path):
    stage = tmp_path / "samples"
    stage.mkdir()
    point_samples_dir(monkeypatch, stage)
    body = sj(client.post("/api/predict", json={"sample": "siren"}))
    assert set(body) == {"detail"}
    for k in ("probs", "order", "conf", "status", "cause"):
        assert k not in body, f"fabricated prediction key {k!r} in a degraded response"


def test_samples_dir_missing_entirely_503s_every_clip(client, monkeypatch, tmp_path):
    point_samples_dir(monkeypatch, tmp_path / "gone")           # never created
    for cid in DEMO_IDS:
        r = client.post("/api/predict", json={"sample": cid})
        d = assert_clean_error(r, 503)
        assert cid in d, f"{cid}: detail must name the clip, got {d!r}"


def test_samples_dir_missing_does_not_break_upload(client, monkeypatch, tmp_path, model_ok):
    """The upload path does not touch samples/: losing the demo clips must not take the app down."""
    if not model_ok:
        pytest.skip("model unavailable")
    point_samples_dir(monkeypatch, tmp_path / "gone")
    r = upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav")
    assert r.status_code == 200, r.text
    assert len(sj(r)["probs"]) == len(CLASSES)


def test_undecodable_demo_clip_is_503_not_500(client, monkeypatch, tmp_path):
    """A present-but-garbage samples/dog.wav (truncated deploy, LFS pointer) must 503, not crash."""
    stage = tmp_path / "samples"
    stage.mkdir()
    (stage / "dog.wav").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\n")
    point_samples_dir(monkeypatch, stage)
    r = client.post("/api/predict", json={"sample": "dog"})
    d = assert_clean_error(r, 503)
    assert "dog" in d


def test_empty_demo_clip_file_is_a_clean_failure(client, monkeypatch, tmp_path):
    stage = tmp_path / "samples"
    stage.mkdir()
    (stage / "horn.wav").write_bytes(b"")
    point_samples_dir(monkeypatch, stage)
    r = client.post("/api/predict", json={"sample": "horn"})
    assert_clean_error(r, 503)


def test_samples_dir_missing_still_reports_via_health(client, monkeypatch, tmp_path):
    point_samples_dir(monkeypatch, tmp_path / "gone")
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    body = sj(r)
    assert sorted(body["missing_demo_clips"]) == sorted(DEMO_IDS), \
        f"health must report every missing clip: {body['missing_demo_clips']}"


def test_health_reports_clips_healthy_again_after_restore(client, monkeypatch, tmp_path):
    point_samples_dir(monkeypatch, tmp_path / "gone")
    assert sj(client.get("/api/health"))["missing_demo_clips"]
    monkeypatch.undo()
    assert sj(client.get("/api/health"))["missing_demo_clips"] == [], \
        "state did not restore: the real samples/ should be complete"


# =========================================================================== model unavailable
def test_model_unavailable_predict_is_503_model_unavailable(client, monkeypatch):
    kill_model(monkeypatch)
    r = client.post("/api/predict", json={"sample": "dog"})
    d = assert_clean_error(r, 503)
    assert "Model unavailable" in d, d
    assert "requirements.txt" in d, f"the 503 must say how to fix it: {d!r}"


def test_model_unavailable_upload_is_503_model_unavailable(client, monkeypatch):
    kill_model(monkeypatch)
    r = upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav")
    d = assert_clean_error(r, 503)
    assert "Model unavailable" in d, d


def test_model_unavailable_every_predict_route_and_every_clip_refuses(client, monkeypatch):
    kill_model(monkeypatch)
    for cid in DEMO_IDS:
        r = client.post("/api/predict", json={"sample": cid,
                                              "corr": {"noise": 0.5, "volume": 0.2, "clip": 0.9}})
        assert r.status_code == 503, f"{cid}: {r.status_code} {r.text[:200]}"
    r = upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav", noise="0.5")
    assert r.status_code == 503, r.text[:200]


def test_model_unavailable_never_returns_a_mock_answer(client, monkeypatch):
    """The incident this project is about: a plausible-looking answer produced without the model."""
    kill_model(monkeypatch)
    for route, kw in (("/api/predict", dict(json={"sample": "dog"})), (None, None)):
        r = client.post(route, **kw) if route else upload(client, enc(tone()), "a.wav", "audio/wav")
        assert r.status_code == 503
        body = sj(r)
        assert set(body) == {"detail"}, body
        for k in ("probs", "order", "conf", "entropy", "snrDb", "levelDb", "clipPct", "status", "cause"):
            assert k not in body, f"fabricated key {k!r} while the model is down"
        for name in CLASSES:
            assert name not in r.text, f"a class label ({name}) leaked into a model-down response"


def test_model_unavailable_health_still_answers_and_says_so(client, monkeypatch):
    kill_model(monkeypatch)
    r = client.get("/api/health")
    assert r.status_code == 200, f"/api/health must stay up while the model is down: {r.text[:200]}"
    body = sj(r)
    assert body["model_available"] is False
    assert body["note"], "health must report WHY the model is unavailable"
    assert "torch" in body["note"]
    assert isinstance(body["missing_demo_clips"], list)


def test_model_unavailable_frontend_still_served(client, monkeypatch):
    kill_model(monkeypatch)
    for path in ("/", "/soundguard-app.js", "/soundguard.css"):
        assert client.get(path).status_code == 200, f"{path} must survive a model outage"


def test_model_unavailable_dashboard_still_served(client, monkeypatch):
    """The evaluation results are a static artifact: they do not depend on the model being loadable."""
    kill_model(monkeypatch)
    r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text[:200]
    sj(r)


def test_model_unavailable_validation_still_precedes_the_model(client, monkeypatch):
    """A bad request must still be a 4xx (the client's fault), not masked as a 503 (ours)."""
    kill_model(monkeypatch)
    assert client.post("/api/predict", json={"sample": "nope"}).status_code == 422
    assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": 5}}).status_code == 422
    assert upload(client, b"hello", "notes.txt", "text/plain").status_code == 415
    assert upload(client, b"", "a.wav", "audio/wav").status_code == 400


def test_model_unavailable_all_error_bodies_are_clean(client, monkeypatch):
    kill_model(monkeypatch)
    for r in (client.post("/api/predict", json={"sample": "dog"}),
              upload(client, enc(tone()), "a.wav", "audio/wav"),
              client.post("/api/predict", json={"sample": "nope"}),
              upload(client, b"hi", "x.txt", "text/plain")):
        assert_clean_error(r)


def test_model_recovers_after_being_unavailable(client, monkeypatch, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    kill_model(monkeypatch)
    assert client.post("/api/predict", json={"sample": "dog"}).status_code == 503
    monkeypatch.undo()
    r = good_predict(client)
    assert r.status_code == 200, r.text[:300]
    assert sj(r)["order"][0][0] == CLASSES.index("dog_bark")


# =========================================================================== no wedged state
def test_app_serves_the_next_good_request_after_every_failure_mode(client, monkeypatch, tmp_path, model_ok):
    """Cycle through each degradation, restoring between, and demand a correct answer each time."""
    if not model_ok:
        pytest.skip("model unavailable")

    def broke_samples():
        point_samples_dir(monkeypatch, tmp_path / "gone")

    def broke_model():
        kill_model(monkeypatch)

    def broke_dashboard():
        point_base_dir(monkeypatch, tmp_path / "nodash")

    for break_it in (broke_samples, broke_model, broke_dashboard, broke_model, broke_samples):
        break_it()
        client.post("/api/predict", json={"sample": "dog"})
        client.get("/api/dashboard")
        monkeypatch.undo()
        r = good_predict(client)
        assert r.status_code == 200, f"wedged after {break_it.__name__}: {r.status_code} {r.text[:200]}"
        assert sj(r)["order"][0][0] == CLASSES.index("dog_bark")
        assert client.get("/api/dashboard").status_code == 200
        assert client.get("/api/health").status_code == 200


def test_repeated_failures_do_not_leak_state_into_the_answer(client, monkeypatch, tmp_path, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    baseline = sj(good_predict(client))
    point_samples_dir(monkeypatch, tmp_path / "gone")
    for _ in range(5):
        assert client.post("/api/predict", json={"sample": "dog"}).status_code == 503
    monkeypatch.undo()
    after = sj(good_predict(client))
    assert after["probs"] == pytest.approx(baseline["probs"], abs=1e-6), \
        "a run of failures changed the model's answer"
    assert after["status"] == baseline["status"]


def test_interleaved_broken_and_good_requests(client, monkeypatch, tmp_path, model_ok):
    """A degraded route must not degrade its neighbours: the samples/ outage is scoped to /api/predict."""
    if not model_ok:
        pytest.skip("model unavailable")
    point_samples_dir(monkeypatch, tmp_path / "gone")
    for _ in range(3):
        assert client.post("/api/predict", json={"sample": "dog"}).status_code == 503
        assert client.get("/api/dashboard").status_code == 200
        assert upload(client, enc(tone(secs=1.0)), "a.wav", "audio/wav").status_code == 200
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 200


# =========================================================================== no information leak
def test_no_degraded_body_leaks_a_host_path_or_traceback(client, monkeypatch, tmp_path):
    """Sweep every failure mode; no body may name a directory outside the project or show a frame."""
    kill_model(monkeypatch)
    point_samples_dir(monkeypatch, tmp_path / "gone")
    point_base_dir(monkeypatch, tmp_path / "nodash")
    responses = [
        client.post("/api/predict", json={"sample": "dog"}),
        client.get("/api/dashboard"),
        upload(client, enc(tone()), "a.wav", "audio/wav"),
        upload(client, b"x" * 32, "a.bin", "application/octet-stream"),
        upload(client, b"", "a.wav", "audio/wav"),
        client.post("/api/predict", json={"sample": "../../etc/passwd"}),
        client.get("/api/nonexistent"),
    ]
    for r in responses:
        assert r.status_code != 500, f"{r.request.url}: 500 {r.text[:200]}"
        assert_no_leak(r)


def test_tmp_staging_path_never_appears_in_a_detail(client, monkeypatch, tmp_path):
    """The patched directory is a host path; the 503 text must describe the deployment, not the host."""
    point_samples_dir(monkeypatch, tmp_path / "gone")
    r = client.post("/api/predict", json={"sample": "dog"})
    assert str(tmp_path) not in r.text, f"the handler echoed its filesystem path: {r.text[:300]}"
    assert "gone" not in r.text


def test_degraded_bodies_are_strict_json(client, monkeypatch, tmp_path):
    kill_model(monkeypatch)
    point_samples_dir(monkeypatch, tmp_path / "gone")
    for r in (client.post("/api/predict", json={"sample": "gun"}),
              upload(client, enc(tone()), "a.wav", "audio/wav"),
              client.get("/api/health")):
        assert r.headers["content-type"].startswith("application/json"), r.headers["content-type"]
        sj(r)
