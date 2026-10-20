"""L3 (error handling): malformed / hostile requests must fail CLEANLY.

Every hostile input to the upload / predict endpoints must resolve to a clean 4xx carrying a JSON
{detail} (or, for an input the pipeline actually accepts, a 200 with a strictly-valid finite body) —
never a 500, and never a NaN/Infinity token that a browser's JSON.parse would choke on.

Verified against app.py:
  predict_upload order of checks:
    1. _looks_like_audio(filename, content_type)  -> 415   (ct prefix "audio/" OR ext in AUDIO_EXTS)
    2. read up to MAX+1 bytes; len==0             -> 400   (empty)
    3. len > 10 MB                                -> 413   (too large)
    4. _require_model()                           -> 503   (only if the model failed to load)
    5. _decode(data) via soundfile               -> 400   (corrupt / unsupported codec)
  So decode-failure (400) cases sit AFTER the model gate: if the model is unavailable they surface as
  503 instead. Those cases are guarded to accept {400} when the model is up, {400,503} otherwise.
"""
import json

import numpy as np
import pytest

from helpers_be import enc, tone, upload

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# --- helpers ----------------------------------------------------------------

def _strict_json(r):
    """Parse the body rejecting NaN/Infinity; returns the object (or None for an empty body)."""
    if not r.text:
        return None
    return json.loads(
        r.text,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite token {c!r}")),
    )


def _assert_clean(r):
    """No 500, and any JSON body is strictly valid (no NaN/Inf)."""
    assert r.status_code != 500, f"500 is a defect: {r.text[:300]}"
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        _strict_json(r)  # raises on NaN/Infinity


def _garbage_pcm16_wav():
    """A structurally valid 16-bit PCM WAV whose data chunk is random bytes (still decodes)."""
    w = bytearray(enc(tone(secs=1.0), subtype="PCM_16"))
    rng = np.random.default_rng(0)
    w[44:] = bytes(rng.integers(0, 256, size=len(w) - 44, dtype="uint8"))
    return bytes(w)


# --- content-type / extension gate (415, pre-model) -------------------------

def test_valid_audio_bytes_but_nonaudio_ctype_and_ext_is_415(client):
    """Correct WAV bytes, but neither an audio/* content-type nor an audio extension -> 415."""
    r = upload(client, enc(tone()), "clip.bin", "application/octet-stream")
    assert r.status_code == 415, r.text
    _assert_clean(r)
    assert "detail" in _strict_json(r)


def test_nonaudio_ctype_and_ext_variants_are_415(client):
    for name, ctype in [("data.dat", "application/octet-stream"),
                        ("note.json", "application/json"),
                        ("x", "")]:
        r = upload(client, enc(tone()), name, ctype)
        assert r.status_code == 415, (name, ctype, r.text)
        _assert_clean(r)


# --- audio-looking wrappers around non-audio payloads (400, post-model) -----

def _assert_decode_reject(r, model_ok):
    """A payload that looks like audio but can't be decoded: 400 (model up) / {400,503} (model down)."""
    _assert_clean(r)
    if model_ok:
        assert r.status_code == 400, r.text
        assert "detail" in _strict_json(r)
    else:
        assert r.status_code in (400, 503), r.text


def test_truncated_wav_header_is_400(client, model_ok):
    r = upload(client, enc(tone())[:20], "trunc.wav", "audio/wav")
    _assert_decode_reject(r, model_ok)


def test_wav_name_carrying_nonaudio_bytes_is_400(client, model_ok):
    """A .wav filename (passes the ext gate) wrapping bytes that are not audio at all -> 400."""
    r = upload(client, b"this is definitely not a waveform, just prose", "clip.wav", "audio/wav")
    _assert_decode_reject(r, model_ok)


def test_audio_ctype_but_text_bytes_is_400(client, model_ok):
    """audio/* content-type (passes the ct gate) but the body is plain text -> 400."""
    r = upload(client, b"hello world, this is text not audio\n" * 8, "note.txt", "audio/wav")
    _assert_decode_reject(r, model_ok)


def test_webm_recording_blob_is_400_with_wav_hint(client, model_ok):
    """libsndfile 1.2.2 cannot read WebM; the record-gap message must mention WAV."""
    webm = b"\x1a\x45\xdf\xa3" + b"\x00\xff" * 64  # EBML magic + junk
    r = upload(client, webm, "recording.webm", "audio/webm")
    _assert_decode_reject(r, model_ok)
    if model_ok:
        detail = _strict_json(r)["detail"]
        assert "WAV" in detail, detail


def test_m4a_aac_blob_is_400(client, model_ok):
    """m4a/aac extensions pass the gate but libsndfile can't decode them -> 400."""
    m4a = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 32
    r = upload(client, m4a, "song.m4a", "audio/mp4")
    _assert_decode_reject(r, model_ok)
    aac = b"\xff\xf1" + b"\x00" * 64  # ADTS-ish header + junk
    r2 = upload(client, aac, "song.aac", "audio/aac")
    _assert_decode_reject(r2, model_ok)


# --- valid header, garbage PCM: this one DOES decode -------------------------

def test_valid_header_garbage_pcm_decodes_to_finite_200(client, model_ok):
    """A valid PCM16 header over random data is valid PCM — it decodes to noise, so 200 with a
    strictly-valid, all-finite body is the CORRECT outcome (not an error)."""
    r = upload(client, _garbage_pcm16_wav(), "garbage.wav", "audio/wav")
    _assert_clean(r)
    if not model_ok:
        pytest.skip("model unavailable")
    assert r.status_code == 200, r.text
    body = _strict_json(r)
    assert len(body["probs"]) == 10
    assert all(np.isfinite(body["probs"]))
    assert np.isfinite(body["conf"]) and np.isfinite(body["entropy"])
    for k in ("snrDb", "levelDb", "clipPct"):
        assert np.isfinite(body[k])


# --- size boundaries --------------------------------------------------------

def test_zero_length_is_400(client):
    r = upload(client, b"", "empty.wav", "audio/wav")
    assert r.status_code == 400, r.text
    _assert_clean(r)
    assert "detail" in _strict_json(r)


def test_exactly_at_limit_is_not_413(client):
    """Exactly 10 MB is at the ceiling, not over it — the size gate must NOT fire (no 413)."""
    r = upload(client, b"\x00" * MAX_UPLOAD_BYTES, "big.wav", "audio/wav")
    _assert_clean(r)
    assert r.status_code != 413, r.text  # boundary is inclusive; this must pass the size check


def test_just_over_limit_is_413(client):
    r = upload(client, b"\x00" * (MAX_UPLOAD_BYTES + 1), "big.wav", "audio/wav")
    assert r.status_code == 413, r.text
    _assert_clean(r)
    assert "detail" in _strict_json(r)


# --- multipart / schema faults ----------------------------------------------

def test_missing_file_part_is_422(client):
    r = client.post("/api/predict/upload")
    assert r.status_code == 422, r.text
    _assert_clean(r)
    assert "detail" in _strict_json(r)


def test_upload_slider_out_of_range_is_422(client):
    r = upload(client, enc(tone()), "a.wav", "audio/wav", noise="2.0")
    assert r.status_code == 422, r.text
    _assert_clean(r)


def test_predict_malformed_json_body_is_422(client):
    r = client.post("/api/predict", content="{ not valid json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    _assert_clean(r)
    assert "detail" in _strict_json(r)


def test_predict_empty_body_is_422_or_ok(client, model_ok):
    """No body at all: FastAPI applies the model defaults (sample='dog'). Either a 422 or the
    default prediction is acceptable; a 500 is not."""
    r = client.post("/api/predict")
    _assert_clean(r)
    assert r.status_code in (200, 422, 503), r.text


def test_predict_unknown_json_fields_ignored(client, model_ok):
    """Extra/unknown top-level and nested fields must be ignored by pydantic, not rejected."""
    r = client.post("/api/predict", json={
        "sample": "dog",
        "corr": {"noise": 0.0, "volume": 0.0, "clip": 0.0, "bogus": 7},
        "unexpected_field": "ignore me",
    })
    _assert_clean(r)
    assert r.status_code != 422, r.text  # unknown fields did not trigger validation error
    if not model_ok:
        pytest.skip("model unavailable")
    assert r.status_code == 200, r.text
    assert len(_strict_json(r)["probs"]) == 10


def test_predict_invalid_sample_literal_is_422(client):
    """A path-traversal / unknown sample id is rejected by the Literal before any file access."""
    for bad in ["../etc/passwd", "", "unknown", "DOG"]:
        r = client.post("/api/predict", json={"sample": bad})
        assert r.status_code == 422, (bad, r.text)
        _assert_clean(r)


# --- routing edge: GET on a POST-only API path ------------------------------

def test_get_on_predict_path_is_404(client):
    """GET /api/predict (POST-only) falls through to the static mount -> 404 JSON, never a 500."""
    r = client.get("/api/predict")
    assert r.status_code == 404, r.text
    _assert_clean(r)


def test_get_on_upload_path_is_404(client):
    r = client.get("/api/predict/upload")
    assert r.status_code == 404, r.text
    _assert_clean(r)
