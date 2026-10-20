"""L5 — robustness at scale: concurrency / thread-safety, determinism under load,
sequential stability + latency, and security/validation edges.

The model + FeatureExtractor are a shared singleton reused across every request, so a
data race would surface as a 500 or a corrupted (non-finite) body under concurrency.
Every prediction response must be 200 with a STRICTLY-valid JSON body whose numeric
fields are all FINITE, or a clean 4xx {detail} — never a 500, never a NaN/Infinity token.

Run:
    python -m pytest tests_be/test_l5_stress.py -q
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from helpers_be import enc, tone, noise, chirp, silence, upload


# ---- helpers ---------------------------------------------------------------

def _strict_json(r):
    """json.loads that REJECTS NaN/Infinity tokens (they break a browser's JSON.parse)."""
    return json.loads(
        r.text,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite token {c!r}")),
    )


def _assert_valid_prediction(r):
    """A prediction response is correct iff: not a 500, status 200, strict JSON, all numbers finite."""
    assert r.status_code != 500, f"500 (uncaught server error): {r.text[:300]}"
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:300]}"
    body = _strict_json(r)                      # raises on NaN/Infinity
    assert len(body["probs"]) == 10
    assert all(np.isfinite(p) for p in body["probs"]), body["probs"]
    s = sum(body["probs"])
    assert abs(s - 1.0) < 1e-3, f"probs sum {s} != 1"
    for k in ("conf", "entropy", "snrDb", "levelDb", "clipPct"):
        assert np.isfinite(body[k]), f"{k}={body[k]} not finite"
    assert body["status"] in {"reliable", "uncertain", "reject"}
    assert body["cause"] in {"clean", "uncert", "noise", "quiet", "clip"}
    return body


def _need_model(model_ok):
    if not model_ok:
        pytest.skip("model unavailable")


# distinct in-memory clips so concurrent requests don't all share one buffer/path
def _clip_bank(n):
    bank = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            sig = tone(freq=220.0 + 40 * i, secs=1.0, amp=0.3)
        elif kind == 1:
            sig = noise(secs=1.0, amp=0.25, seed=i)
        elif kind == 2:
            sig = chirp(f0=150.0, f1=3500.0, secs=1.5, amp=0.3)
        else:
            sig = tone(freq=440.0, secs=2.0, amp=0.2)
        bank.append(enc(sig))
    return bank


# ---- concurrency / thread-safety -------------------------------------------

def test_concurrent_uploads_no_race(client, model_ok):
    """~24 uploads fired concurrently through the shared model: every one must be a valid,
    finite 200 and none may 500. A race on the shared FeatureExtractor/head surfaces here."""
    _need_model(model_ok)
    N = 24
    bank = _clip_bank(N)

    def one(i):
        r = upload(client, bank[i], name=f"c{i}.wav", ctype="audio/wav")
        return i, r.status_code, r.text

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, code, text in ex.map(one, range(N)):
            results.append((i, code, text))

    # rebuild a Response-like check from captured text/status
    bad = [(i, c) for i, c, _ in results if c != 200]
    assert not bad, f"non-200 under concurrency: {bad}"
    for i, code, text in results:
        assert code != 500
        body = json.loads(
            text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite {c}")))
        assert len(body["probs"]) == 10
        assert all(np.isfinite(p) for p in body["probs"])
        assert np.isfinite(body["conf"])


def test_concurrent_demo_predict_true_parallel(client, model_ok):
    """The /api/predict (sync) handler runs in the threadpool -> TRUE parallelism across the
    shared model. Fire the seven demo clips concurrently many times; all must be valid 200s."""
    _need_model(model_ok)
    samples = ["dog", "siren", "jack", "horn", "kids", "gun", "ac"]
    tasks = (samples * 4)[:24]

    def one(s):
        r = client.post("/api/predict", json={"sample": s})
        return s, r.status_code, r.text

    with ThreadPoolExecutor(max_workers=8) as ex:
        out = list(ex.map(one, tasks))

    for s, code, text in out:
        # a missing demo clip is a clean 503, not a defect of this test
        if code == 503:
            continue
        assert code != 500, f"{s}: 500 {text[:200]}"
        assert code == 200, f"{s}: {code} {text[:200]}"
        body = json.loads(
            text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite {c}")))
        assert all(np.isfinite(p) for p in body["probs"])


def test_concurrent_same_clip_identical(client, model_ok):
    """The SAME bytes fired concurrently must all yield the identical top label — a shared-state
    race would make some diverge even though inference is deterministic."""
    _need_model(model_ok)
    data = enc(chirp(secs=2.0))

    def one(_):
        r = upload(client, data, "same.wav", "audio/wav")
        assert r.status_code == 200, r.text
        return tuple(r.json()["order"][0])   # (idx, prob)

    with ThreadPoolExecutor(max_workers=8) as ex:
        labels = list(ex.map(one, range(24)))

    top_idxs = {lab[0] for lab in labels}
    assert len(top_idxs) == 1, f"same clip gave divergent labels under concurrency: {top_idxs}"


# ---- determinism -----------------------------------------------------------

def test_determinism_20x_identical(client, model_ok):
    """Same clip predicted 20x (sequential) returns identical label AND identical probs — the
    head is eval() (dropout off, frozen BN) and sliders are 0, so it must be bit-reproducible."""
    _need_model(model_ok)
    data = enc(chirp(secs=2.0))
    first = None
    for _ in range(20):
        r = upload(client, data, "d.wav", "audio/wav")
        body = _assert_valid_prediction(r)
        if first is None:
            first = body
            continue
        assert body["order"][0][0] == first["order"][0][0], "top label drifted across repeats"
        assert body["probs"] == first["probs"], "probs not bit-identical across repeats"


# ---- sequential stability + latency ----------------------------------------

def test_sequential_60_stable_and_bounded(client, model_ok):
    """60 sequential uploads: none error, and latency stays bounded (no unbounded growth that
    would signal a per-request leak). Bounds are generous so this is not flaky on a busy CPU."""
    _need_model(model_ok)
    N = 60
    bank = _clip_bank(N)
    lat = []
    for i in range(N):
        t0 = time.perf_counter()
        r = upload(client, bank[i], name=f"s{i}.wav", ctype="audio/wav")
        lat.append(time.perf_counter() - t0)
        _assert_valid_prediction(r)

    lat = np.array(lat)
    # absolute ceiling: a single CPU inference of a <=2 s clip is well under this
    assert lat.max() < 10.0, f"a request took {lat.max():.2f}s (latency spike)"
    # leak/growth guard: the last 20 must not be dramatically slower than the first 20
    first20, last20 = lat[:20].mean(), lat[-20:].mean()
    assert last20 < max(first20 * 5.0, 2.0), (
        f"latency grew from {first20*1000:.0f}ms (first 20) to {last20*1000:.0f}ms (last 20)")


# ---- security / validation edges -------------------------------------------

@pytest.mark.parametrize("bad_sample", [
    "../../etc/passwd",
    "..\\..\\windows\\system32",
    "/etc/passwd",
    "C:\\Windows\\win.ini",
    "dog/../siren",
    "samples/dog",
    "",
    "dog.wav",
])
def test_path_traversal_sample_rejected_422(client, bad_sample):
    """The demo 'sample' is a Literal of seven ids; anything else — traversal, absolute path,
    empty — is rejected by validation with 422 before any file path is built. Never 500/200."""
    r = client.post("/api/predict", json={"sample": bad_sample})
    assert r.status_code == 422, f"{bad_sample!r} -> {r.status_code} {r.text[:200]}"
    assert r.status_code != 500
    assert "detail" in r.json()


def test_long_filename_is_handled_cleanly(client, model_ok):
    """A 10k-char filename is metadata, never a filesystem path, so it cannot break anything.

    The outcome is parser-version dependent and both outcomes are acceptable: it either classifies
    (200), or a hardened multipart parser rejects the oversized part header with a clean 400 —
    python-multipart >= 0.0.30 caps part-header size as a denial-of-service mitigation (observed:
    0.0.20 accepts it, 0.0.32 returns 400 'There was an error parsing the body'). What must never
    happen is a 500 or a fabricated answer.
    """
    _need_model(model_ok)
    name = ("a" * 10000) + ".wav"
    r = upload(client, enc(tone(secs=1.0)), name=name, ctype="audio/wav")
    assert r.status_code != 500, f"500 (uncaught server error): {r.text[:300]}"
    if r.status_code == 200:
        _assert_valid_prediction(r)
    else:
        assert 400 <= r.status_code < 500 and "detail" in r.json(), r.text[:300]


def test_unicode_emoji_filename_classifies(client, model_ok):
    """Unicode/emoji filename with valid WAV bytes -> 200 (no encoding crash)."""
    _need_model(model_ok)
    r = upload(client, enc(tone(secs=1.0)), name="🎵_ç̃_名前_🔊.wav", ctype="audio/wav")
    _assert_valid_prediction(r)


def test_many_multipart_fields_handled(client):
    """A multipart body with a huge number of fields is HANDLED cleanly (starlette caps parts and
    returns 400 'Too many fields'), never a 500 or a hang."""
    data = {f"k{i}": "v" for i in range(2000)}
    r = client.post(
        "/api/predict/upload",
        files={"file": ("a.wav", enc(tone(secs=0.5)), "audio/wav")},
        data=data,
    )
    assert r.status_code != 500, r.text[:300]
    assert r.status_code in {200, 400, 413, 422}, f"unexpected {r.status_code}: {r.text[:200]}"
    assert "detail" in r.json() or r.status_code == 200


def test_moderate_extra_fields_still_classify(client, model_ok):
    """A reasonable number of extra/ignored form fields alongside a valid file still -> 200."""
    _need_model(model_ok)
    data = {f"k{i}": "v" for i in range(50)}
    data.update({"noise": "0", "volume": "0", "clip": "0"})
    r = client.post(
        "/api/predict/upload",
        files={"file": ("a.wav", enc(tone(secs=0.5)), "audio/wav")},
        data=data,
    )
    _assert_valid_prediction(r)


def test_health_never_leaks_stack_trace(client):
    """/api/health is a monitoring probe: 200, the three documented keys, and never a traceback."""
    for _ in range(5):
        r = client.get("/api/health")
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {"model_available", "note", "missing_demo_clips"}
        assert "Traceback" not in r.text and "File \"" not in r.text
        assert isinstance(body["missing_demo_clips"], list)


# ---- record gap: webm/m4a undecodable -> clean 400 (documented, not a 500) --

@pytest.mark.parametrize("name,ctype,head", [
    ("rec.webm", "audio/webm", b"\x1aE\xdf\xa3" + b"\x00" * 64),   # EBML/Matroska magic
    ("rec.m4a", "audio/mp4", b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 48),  # ISO-BMFF ftyp
])
def test_webm_m4a_record_gap_clean_400(client, name, ctype, head):
    """Known record gap: libsndfile 1.2.2 does not read webm/m4a/aac, so such uploads decode-fail
    with a CLEAN 400 {detail} (not a 500). Documents the gap rather than asserting support."""
    r = client.post("/api/predict/upload", files={"file": (name, head, ctype)})
    assert r.status_code != 500, r.text[:300]
    assert r.status_code == 400, f"{name} -> {r.status_code} {r.text[:200]}"
    assert "detail" in r.json()
