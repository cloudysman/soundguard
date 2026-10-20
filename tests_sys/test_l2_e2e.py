"""L2 — end-to-end flows: the stack agrees with itself.

Not endpoint validation (tests_be), not browser behaviour (tests_fe), not metric formulas (tests_ai).
What is asserted here is that the HTTP layer is a faithful transport for the library underneath:

  * every demo chip returns exactly what predictor.predict_wav() returns on samples/<id>.wav
  * samples/dog.wav uploaded == the dog chip
  * a 16-bit PCM mono WAV built byte-for-byte the way static/soundguard-app.js encodeWav() builds it
    is accepted and classified
  * the corruption sliders move the API result exactly as apply_slider_corruption + predict_wav do
  * GET /api/dashboard serves the content of results/dashboard.json unaltered
  * the response schema is identical across all seven chips, and identical requests -> identical bodies

Everything is checked with a strict JSON parser (no NaN/Infinity tokens) and finite numbers.
"""
import io
import json
import math
import os
import struct

import numpy as np
import pytest

from helpers_sys import CLASSES, DEMO_IDS

TOL = 1e-6

RESPONSE_KEYS = {"probs", "order", "conf", "entropy", "snrDb", "levelDb", "clipPct", "status", "cause",
                 "mel"}
STATUSES = {"reliable", "uncertain", "reject"}
CAUSES = {"noise", "quiet", "clip", "uncert", "clean"}

# The slider settings exercised end-to-end. apply_slider_corruption seeds its RNG (seed=0),
# so the noisy cases are deterministic and comparable to a direct call.
SLIDERS = [
    {"noise": 0.0, "volume": 0.0, "clip": 0.0},
    {"noise": 0.6, "volume": 0.0, "clip": 0.0},
    {"noise": 0.0, "volume": 0.8, "clip": 0.0},
    {"noise": 0.0, "volume": 0.0, "clip": 0.7},
    {"noise": 0.35, "volume": 0.25, "clip": 0.4},
    {"noise": 1.0, "volume": 1.0, "clip": 1.0},
]


# ---------------------------------------------------------------- helpers
def strict_json(r):
    """Parse a response body rejecting NaN/Infinity/-Infinity tokens, and require finite numbers."""
    body = json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-JSON constant {c}")))
    _all_finite(body)
    return body


def _all_finite(node, path="$"):
    if isinstance(node, bool):
        return
    if isinstance(node, float):
        assert math.isfinite(node), f"non-finite number at {path}: {node}"
    elif isinstance(node, dict):
        for k, v in node.items():
            _all_finite(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _all_finite(v, f"{path}[{i}]")


def load_sample(root, sid):
    """Read samples/<sid>.wav the way app._load_sample_wav does: mono float32, finite."""
    import soundfile as sf

    wav, sr = sf.read(os.path.join(root, "samples", f"{sid}.wav"))
    wav = np.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0), sr


def sample_bytes(root, sid):
    with open(os.path.join(root, "samples", f"{sid}.wav"), "rb") as f:
        return f.read()


def browser_encode_wav(mono, sample_rate):
    """Byte-for-byte port of encodeWav() in static/soundguard-app.js (the recorder path):
    44-byte canonical RIFF header, PCM fmt tag 1, 1 channel, 16 bits, then
    x < 0 ? x * 0x8000 : x * 0x7FFF, little-endian."""
    mono = np.asarray(mono, dtype=np.float64)
    n = len(mono)
    out = io.BytesIO()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + n * 2))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    out.write(b"data")
    out.write(struct.pack("<I", n * 2))
    for x in mono:
        x = max(-1.0, min(1.0, float(x)))
        v = int(x * 0x8000) if x < 0 else int(x * 0x7FFF)   # JS setInt16 truncates toward zero
        out.write(struct.pack("<h", max(-32768, min(32767, v))))
    return out.getvalue()


def api_predict(client, sid, **corr):
    payload = {"sample": sid}
    if corr:
        payload["corr"] = corr
    return client.post("/api/predict", json=payload)


def assert_same_result(api, direct, label=""):
    """The API body must be the predictor's dict, undistorted."""
    assert api["order"][0][0] == direct["order"][0][0], (
        f"{label}: top class differs — API {CLASSES[api['order'][0][0]]} vs direct {CLASSES[direct['order'][0][0]]}")
    assert abs(api["conf"] - direct["conf"]) <= TOL, f"{label}: conf {api['conf']} vs {direct['conf']}"
    assert len(api["probs"]) == len(direct["probs"]) == 10
    for i, (a, b) in enumerate(zip(api["probs"], direct["probs"])):
        assert abs(a - b) <= TOL, f"{label}: probs[{i}] {a} vs {b}"
    assert abs(api["entropy"] - direct["entropy"]) <= TOL, f"{label}: entropy"
    for k in ("snrDb", "levelDb", "clipPct"):
        assert abs(api[k] - direct[k]) <= TOL, f"{label}: {k} {api[k]} vs {direct[k]}"
    assert api["status"] == direct["status"], f"{label}: status"
    assert api["cause"] == direct["cause"], f"{label}: cause"


def assert_schema(body, label=""):
    assert set(body) == RESPONSE_KEYS, f"{label}: keys {sorted(body)}"
    probs = body["probs"]
    assert isinstance(probs, list) and len(probs) == 10, f"{label}: probs"
    assert all(isinstance(p, float) and 0.0 <= p <= 1.0 for p in probs), f"{label}: probs range"
    assert abs(sum(probs) - 1.0) <= 1e-5, f"{label}: probs sum {sum(probs)}"

    order = body["order"]
    assert isinstance(order, list) and len(order) == 10, f"{label}: order length"
    idxs = [o[0] for o in order]
    assert sorted(idxs) == list(range(10)), f"{label}: order is not a permutation: {idxs}"
    for i, p in order:
        assert isinstance(i, int) and abs(p - probs[i]) <= 1e-12, f"{label}: order pairs disagree with probs"
    vals = [o[1] for o in order]
    assert vals == sorted(vals, reverse=True), f"{label}: order not sorted descending"

    assert abs(body["conf"] - max(probs)) <= 1e-12, f"{label}: conf != max(probs)"
    assert abs(body["conf"] - order[0][1]) <= 1e-12, f"{label}: conf != order[0]"
    assert 0.0 <= body["entropy"] <= 1.0 + 1e-9, f"{label}: entropy {body['entropy']}"

    mel = body["mel"]
    assert isinstance(mel, list) and mel, f"{label}: mel is not a non-empty list"
    assert all(isinstance(row, list) and len(row) == len(mel[0]) for row in mel), f"{label}: mel is ragged"
    assert all(isinstance(v, int) and 0 <= v <= 255 for row in mel for v in row), f"{label}: mel range"
    # A constant spectrogram would mean the panel is showing nothing, which is the failure this
    # field exists to prevent: the interface used to draw a synthetic image and call it the signal.
    assert len({v for row in mel for v in row}) > 1, f"{label}: mel is constant"
    for k in ("snrDb", "levelDb", "clipPct"):
        assert isinstance(body[k], float), f"{label}: {k} type {type(body[k])}"
    assert 0.0 <= body["clipPct"] <= 100.0, f"{label}: clipPct"
    assert body["status"] in STATUSES, f"{label}: status {body['status']}"
    assert body["cause"] in CAUSES, f"{label}: cause {body['cause']}"


@pytest.fixture(scope="session")
def need_model(model_ok, predictor):
    if not model_ok:
        pytest.skip(f"model unavailable: {predictor.note}")
    return True


@pytest.fixture(scope="session")
def chip_api(client, need_model):
    """One clean API call per demo chip, reused by several tests (keeps runtime modest)."""
    out = {}
    for sid in DEMO_IDS:
        r = api_predict(client, sid)
        assert r.status_code == 200, f"{sid}: {r.status_code} {r.text}"
        out[sid] = strict_json(r)
    return out


# ---------------------------------------------------------------- 1. HTTP adds no distortion
@pytest.mark.parametrize("sid", DEMO_IDS)
def test_chip_equals_direct_predict_wav(client, predictor, need_model, sid):
    """POST /api/predict == predictor.predict_wav(samples/<sid>.wav) — same top class, conf within 1e-6."""
    wav, sr = load_sample(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sid)
    direct = predictor.predict_wav(wav, sr)

    r = api_predict(client, sid)
    assert r.status_code == 200, r.text
    api = strict_json(r)
    assert_schema(api, sid)
    assert_same_result(api, direct, sid)


def test_every_chip_answers_200_with_valid_json(chip_api):
    assert set(chip_api) == set(DEMO_IDS)
    for sid, body in chip_api.items():
        assert_schema(body, sid)


# ---------------------------------------------------------------- 2. upload == chip
def test_upload_dog_wav_matches_dog_chip(client, need_model, chip_api):
    """The same file through the multipart path lands on the same class as the chip."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = client.post("/api/predict/upload",
                    files={"file": ("dog.wav", sample_bytes(root, "dog"), "audio/wav")})
    assert r.status_code == 200, r.text
    up = strict_json(r)
    assert_schema(up, "upload dog.wav")
    chip = chip_api["dog"]
    assert up["order"][0][0] == chip["order"][0][0], (
        f"upload {CLASSES[up['order'][0][0]]} vs chip {CLASSES[chip['order'][0][0]]}")
    assert abs(up["conf"] - chip["conf"]) <= TOL
    for a, b in zip(up["probs"], chip["probs"]):
        assert abs(a - b) <= TOL
    assert (up["status"], up["cause"]) == (chip["status"], chip["cause"])


@pytest.mark.parametrize("sid", ["siren", "gun"])
def test_upload_of_demo_file_matches_its_chip(client, need_model, chip_api, sid):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = client.post("/api/predict/upload",
                    files={"file": (f"{sid}.wav", sample_bytes(root, sid), "audio/wav")})
    assert r.status_code == 200, r.text
    up = strict_json(r)
    assert up["order"][0][0] == chip_api[sid]["order"][0][0]
    assert abs(up["conf"] - chip_api[sid]["conf"]) <= TOL


# ---------------------------------------------------------------- 3. the browser recorder's WAV
def _recorder_signal(sr=48000, secs=2.0):
    """A deterministic mono float32 signal, the shape WebAudio hands to encodeWav()."""
    t = np.arange(int(sr * secs)) / sr
    sig = (0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1320 * t)
           + 0.05 * np.sin(2 * np.pi * 90 * t))
    return sig.astype(np.float32)


def test_browser_recorded_wav_is_accepted_and_classified(client, need_model):
    """A 16-bit PCM mono WAV encoded exactly like static/soundguard-app.js encodeWav()
    (48 kHz, the usual AudioContext rate) must classify, not 4xx and not 500."""
    data = browser_encode_wav(_recorder_signal(), 48000)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE" and len(data) == 44 + 48000 * 2 * 2
    r = client.post("/api/predict/upload", files={"file": ("mic_recording.wav", data, "audio/wav")})
    assert r.status_code == 200, f"recorder WAV rejected: {r.status_code} {r.text}"
    body = strict_json(r)
    assert_schema(body, "recorder wav")


@pytest.mark.parametrize("sr", [16000, 44100, 48000])
def test_recorder_wav_at_browser_sample_rates(client, need_model, sr):
    """AudioContext rates vary by device; the resampler in features.py must absorb all of them."""
    r = client.post("/api/predict/upload",
                    files={"file": ("mic_recording.wav", browser_encode_wav(_recorder_signal(sr, 1.5), sr),
                                    "audio/wav")})
    assert r.status_code == 200, r.text
    assert_schema(strict_json(r), f"recorder {sr}")


def test_recorder_wav_matches_direct_predict(client, predictor, need_model):
    """The hand-built recorder WAV decodes to what the predictor sees: the API result equals a
    direct predict_wav() on the same decoded samples."""
    import soundfile as sf

    data = browser_encode_wav(_recorder_signal(48000, 1.5), 48000)
    wav, sr = sf.read(io.BytesIO(data))
    wav = np.asarray(wav, dtype="float32")
    assert sr == 48000
    direct = predictor.predict_wav(predictor.apply_slider_corruption(wav, sr, 0.0, 0.0, 0.0), sr)

    r = client.post("/api/predict/upload", files={"file": ("mic_recording.wav", data, "audio/wav")})
    assert r.status_code == 200, r.text
    assert_same_result(strict_json(r), direct, "recorder wav")


def test_recorder_encoder_agnostic_on_class(client, need_model):
    """The browser's encoder and soundfile's PCM_16 writer differ only in the last quantisation step
    (JS setInt16 truncates x*0x7FFF; libsndfile rounds x*0x8000), i.e. by up to 1 LSB. The decision the
    system reports must not depend on that. Confidence is NOT compared: the head is measurably
    sensitive to 1-LSB requantisation (~0.07 conf on this tone), which is a model-quality property
    (tests_ai), not a stack-consistency one."""
    sig = _recorder_signal(16000, 1.5)
    hand = browser_encode_wav(sig, 16000)
    buf = io.BytesIO()
    sf_mod = pytest.importorskip("soundfile")
    sf_mod.write(buf, sig, 16000, format="WAV", subtype="PCM_16")
    a = client.post("/api/predict/upload", files={"file": ("a.wav", hand, "audio/wav")})
    b = client.post("/api/predict/upload", files={"file": ("b.wav", buf.getvalue(), "audio/wav")})
    assert a.status_code == b.status_code == 200, (a.text, b.text)
    ja, jb = strict_json(a), strict_json(b)
    assert_schema(ja, "hand-encoded")
    assert_schema(jb, "soundfile-encoded")
    assert ja["order"][0][0] == jb["order"][0][0], (
        f"encoder changed the class: {CLASSES[ja['order'][0][0]]} vs {CLASSES[jb['order'][0][0]]}")
    assert ja["status"] == jb["status"]


# ---------------------------------------------------------------- 4. sliders through the API
@pytest.mark.parametrize("corr", SLIDERS, ids=lambda c: f"n{c['noise']}_v{c['volume']}_c{c['clip']}")
def test_sliders_match_direct_corruption_pipeline(client, predictor, need_model, corr):
    """POST /api/predict with sliders == apply_slider_corruption(...) -> predict_wav(...) directly."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav, sr = load_sample(root, "dog")
    cw = predictor.apply_slider_corruption(wav, sr, corr["noise"], corr["volume"], corr["clip"])
    direct = predictor.predict_wav(cw, sr)

    r = api_predict(client, "dog", **corr)
    assert r.status_code == 200, r.text
    api = strict_json(r)
    assert_schema(api, str(corr))
    assert_same_result(api, direct, str(corr))


def test_sliders_are_not_a_no_op(client, need_model, chip_api):
    """A heavy corruption must visibly change the served result — otherwise the sliders are decorative."""
    clean = chip_api["dog"]
    r = api_predict(client, "dog", noise=1.0, volume=1.0, clip=1.0)
    assert r.status_code == 200, r.text
    dirty = strict_json(r)
    assert dirty["probs"] != clean["probs"], "max corruption produced an identical distribution"
    assert (dirty["conf"] < clean["conf"]) or (dirty["status"] != clean["status"]), (
        f"heavy corruption did not degrade the answer: conf {clean['conf']}->{dirty['conf']}, "
        f"status {clean['status']}->{dirty['status']}")


def test_slider_corruption_reaches_the_quality_report(client, need_model, chip_api):
    """The corrupted waveform — not the original — is what the quality heuristics measure."""
    clean = chip_api["dog"]
    quiet = strict_json(api_predict(client, "dog", volume=1.0))
    assert quiet["levelDb"] < clean["levelDb"] - 10, (clean["levelDb"], quiet["levelDb"])
    clipped = strict_json(api_predict(client, "dog", clip=1.0))
    assert clipped["clipPct"] > clean["clipPct"], (clean["clipPct"], clipped["clipPct"])


def test_upload_sliders_match_direct_pipeline(client, predictor, need_model):
    """The multipart path applies the same corruption as the JSON path (Form floats, same math)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav, sr = load_sample(root, "dog")
    direct = predictor.predict_wav(predictor.apply_slider_corruption(wav, sr, 0.5, 0.2, 0.3), sr)
    r = client.post("/api/predict/upload",
                    files={"file": ("dog.wav", sample_bytes(root, "dog"), "audio/wav")},
                    data={"noise": "0.5", "volume": "0.2", "clip": "0.3"})
    assert r.status_code == 200, r.text
    assert_same_result(strict_json(r), direct, "upload+sliders")


# ---------------------------------------------------------------- 5. dashboard passthrough
def test_dashboard_serves_the_file_unaltered(client, dashboard):
    """GET /api/dashboard is a passthrough of results/dashboard.json: same content, same numbers,
    same key order. (The wire bytes are compact JSON; the file is pretty-printed, so the comparison
    is over canonical serialisation, not raw whitespace.)"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "results", "dashboard.json"), "rb") as f:
        raw = f.read()
    on_disk = json.loads(raw.decode("utf-8"),
                         parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text
    served = strict_json(r)
    assert served == on_disk == dashboard
    # key order and float repr survive the round trip
    assert json.dumps(served, separators=(",", ":")) == json.dumps(on_disk, separators=(",", ":"))
    assert list(served) == list(on_disk)
    assert "NaN" not in r.text and "Infinity" not in r.text
    assert r.headers["content-type"].startswith("application/json")


def test_dashboard_is_stable_and_idempotent(client):
    a, b = client.get("/api/dashboard"), client.get("/api/dashboard")
    assert a.status_code == b.status_code == 200
    assert a.text == b.text


# ---------------------------------------------------------------- 6. schema stability
def test_schema_identical_across_all_seven_chips(chip_api):
    """Every chip returns the same keys with the same types — the frontend renders one shape."""
    shapes = {}
    for sid, body in chip_api.items():
        shapes[sid] = {k: type(v).__name__ for k, v in sorted(body.items())}
    ref_id = DEMO_IDS[0]
    for sid, shape in shapes.items():
        assert shape == shapes[ref_id], f"{sid} shape {shape} != {ref_id} shape {shapes[ref_id]}"


def test_schema_stable_under_corruption(client, need_model, chip_api):
    """Corruption must not change the response shape, only its numbers."""
    ref = {k: type(v).__name__ for k, v in sorted(chip_api["dog"].items())}
    for corr in SLIDERS:
        body = strict_json(api_predict(client, "dog", **corr))
        assert {k: type(v).__name__ for k, v in sorted(body.items())} == ref, corr


# ---------------------------------------------------------------- 7. determinism
@pytest.mark.parametrize("sid", ["dog", "ac"])
def test_two_identical_requests_identical_bodies(client, need_model, sid):
    a = api_predict(client, sid)
    b = api_predict(client, sid)
    assert a.status_code == b.status_code == 200
    assert a.text == b.text, f"{sid}: non-deterministic body"


def test_identical_noisy_requests_identical_bodies(client, need_model):
    """apply_slider_corruption seeds its RNG, so even the noise slider is reproducible."""
    corr = {"noise": 0.7, "volume": 0.1, "clip": 0.2}
    a = api_predict(client, "siren", **corr)
    b = api_predict(client, "siren", **corr)
    assert a.status_code == b.status_code == 200
    assert a.text == b.text, "noisy request is not reproducible"


def test_identical_uploads_identical_bodies(client, need_model):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = sample_bytes(root, "horn")
    a = client.post("/api/predict/upload", files={"file": ("horn.wav", data, "audio/wav")})
    b = client.post("/api/predict/upload", files={"file": ("horn.wav", data, "audio/wav")})
    assert a.status_code == b.status_code == 200
    assert a.text == b.text


def test_chip_result_survives_an_interleaved_upload(client, need_model, chip_api):
    """Serving another request in between must not perturb a chip's answer (no shared mutable state)."""
    client.post("/api/predict/upload", files={"file": ("t.wav", browser_encode_wav(_recorder_signal(16000, 1.0), 16000), "audio/wav")})
    again = strict_json(api_predict(client, "kids"))
    assert again == chip_api["kids"]
