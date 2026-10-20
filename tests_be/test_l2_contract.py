"""L2 — contract / formats / determinism for the SoundGuard backend.

Covers, against the REAL model + endpoints (in-process TestClient):
  * response-schema integrity for BOTH predict endpoints,
  * input validation (422/415/400/413 with the right bodies),
  * real container formats decode & classify (WAV PCM_16/PCM_24/FLOAT, FLAC, OGG, AIFF, MP3),
  * mono/stereo downmix and odd sample rates,
  * determinism (identical bytes -> identical JSON),
  * physical slider-corruption monotonicity (noise->snr, volume->level, clip->clipPct).

Verified against app.py / predictor.py / features.py / corruption.py / rejection.py.
A known defect (FLOAT-subtype WAV of NaN -> json.dumps raises -> 500) is documented with xfail.
"""
import json

import numpy as np
import pytest
import soundfile as sf

from helpers_be import enc, tone, upload


STATUS_SET = {"reliable", "uncertain", "reject"}
CAUSE_SET = {"clean", "uncert", "noise", "quiet", "clip"}
REQUIRED_KEYS = {"probs", "order", "conf", "entropy", "snrDb", "levelDb", "clipPct", "status", "cause",
                 "mel"}


# --- helpers ---------------------------------------------------------------

def _strict_json(text):
    """Parse JSON rejecting NaN/Infinity tokens (which break a browser JSON.parse)."""
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite token: {c}")))


def _finite(x):
    return isinstance(x, (int, float)) and np.isfinite(x)


def assert_valid_prediction(text):
    """Full contract check on a prediction response body (given as raw text)."""
    body = _strict_json(text)                       # rejects NaN/Infinity
    assert REQUIRED_KEYS <= set(body), f"missing keys: {REQUIRED_KEYS - set(body)}"

    probs = body["probs"]
    assert isinstance(probs, list) and len(probs) == 10
    assert all(_finite(p) and 0.0 <= p <= 1.0 for p in probs), probs
    assert abs(sum(probs) - 1.0) < 1e-4, sum(probs)

    order = body["order"]
    assert isinstance(order, list) and len(order) == 10
    idxs = [pair[0] for pair in order]
    assert sorted(idxs) == list(range(10)), f"order is not a permutation: {idxs}"
    order_probs = [pair[1] for pair in order]
    # desc permutation, and each listed prob matches probs[idx]
    assert order_probs == sorted(order_probs, reverse=True), order_probs
    for idx, p in order:
        assert abs(p - probs[idx]) < 1e-9

    assert body["conf"] == order[0][1], "conf must equal the top-order probability"
    assert _finite(body["entropy"]) and 0.0 <= body["entropy"] <= 1.0, body["entropy"]

    for k in ("snrDb", "levelDb", "clipPct"):
        assert _finite(body[k]), (k, body[k])

    assert body["status"] in STATUS_SET, body["status"]
    assert body["cause"] in CAUSE_SET, body["cause"]

    mel = body["mel"]
    assert isinstance(mel, list) and mel, "mel must be a non-empty list of rows"
    assert all(isinstance(row, list) and len(row) == len(mel[0]) for row in mel), "mel is ragged"
    assert all(isinstance(v, int) and 0 <= v <= 255 for row in mel for v in row), "mel out of range"
    return body


def _bursty(secs=2.0, sr=16000, freq=440.0, amp=0.5):
    """Loud tone bursts separated by silence -> a meaningful SNR-contrast (high when clean)."""
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    sig = (amp * np.sin(2 * np.pi * freq * t)).astype("float32")
    env = np.ones_like(sig)
    for k in range(0, len(sig), 4000):
        env[k + 2000:k + 4000] = 0.0
    return (sig * env).astype("float32")


def _predict_sample(client, sample="dog", corr=None):
    payload = {"sample": sample}
    if corr is not None:
        payload["corr"] = corr
    return client.post("/api/predict", json=payload)


# --- schema integrity: /api/predict (demo clip) ----------------------------

def test_predict_demo_schema(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    r = _predict_sample(client, "dog")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


@pytest.mark.parametrize("sample", ["dog", "siren", "jack", "horn", "kids", "gun", "ac"])
def test_predict_all_demo_ids_valid(client, model_ok, sample):
    if not model_ok:
        pytest.skip("model unavailable")
    r = _predict_sample(client, sample)
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


# --- schema integrity: /api/predict/upload ---------------------------------

def test_predict_upload_schema(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    r = upload(client, enc(_bursty()), "a.wav", "audio/wav")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


# --- input validation: /api/predict ----------------------------------------

@pytest.mark.parametrize("sample", ["unknown", "", "../../etc/passwd", "dog.wav", "AC"])
def test_predict_bad_sample_is_422(client, sample):
    """Literal type rejects any non-member sample (incl. traversal) with 422, before any file path."""
    r = _predict_sample(client, sample)
    assert r.status_code == 422, r.text
    assert "detail" in r.json()


@pytest.mark.parametrize("corr", [
    {"noise": 1.5}, {"noise": -0.1}, {"volume": 2.0}, {"volume": -0.5},
    {"clip": 1.0001}, {"clip": -1e-6},
])
def test_predict_corr_out_of_range_is_422(client, corr):
    r = _predict_sample(client, "dog", corr=corr)
    assert r.status_code == 422, r.text
    assert "detail" in r.json()


def test_predict_corr_in_range_ok(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    r = _predict_sample(client, "dog", corr={"noise": 0.0, "volume": 1.0, "clip": 0.5})
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


# --- input validation: /api/predict/upload ---------------------------------

@pytest.mark.parametrize("noise", [1.5, -0.1])
def test_upload_slider_out_of_range_is_422(client, noise):
    r = upload(client, enc(tone(secs=0.5)), "a.wav", "audio/wav", noise=noise)
    assert r.status_code == 422, r.text
    assert "detail" in r.json()


def test_upload_non_audio_is_415(client):
    """Neither an audio/* content-type nor an audio extension -> 415, before any decode."""
    r = upload(client, b"this is plainly not audio", "notes.txt", "text/plain")
    assert r.status_code == 415, r.text
    assert "detail" in r.json()


def test_upload_empty_is_400(client):
    """Passes the audio-type gate (.wav) but has zero bytes -> 400."""
    r = upload(client, b"", "a.wav", "audio/wav")
    assert r.status_code == 400, r.text
    assert "detail" in r.json()


def test_upload_oversize_is_413(client):
    """Just over the 10 MB ceiling (audio ext so it clears the type gate) -> 413."""
    data = b"\x00" * (10 * 1024 * 1024 + 16)
    r = upload(client, data, "big.wav", "audio/wav")
    assert r.status_code == 413, r.text
    assert "detail" in r.json()


def test_upload_undecodable_is_400(client, model_ok):
    """Audio-typed but garbage bytes -> soundfile fails -> clean 400 (not a 500)."""
    if not model_ok:
        pytest.skip("model unavailable")
    r = upload(client, b"RIFFxxxxNOTAWAVE" + b"\x01" * 4096, "a.wav", "audio/wav")
    assert r.status_code == 400, r.text
    assert r.status_code != 500
    assert "detail" in r.json()


def test_upload_audio_by_extension_only(client, model_ok):
    """A generic content-type but a .wav extension still counts as audio (ext branch)."""
    if not model_ok:
        pytest.skip("model unavailable")
    r = upload(client, enc(_bursty()), "clip.wav", "application/octet-stream")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


# --- real container formats decode & classify (200) ------------------------

_FMT_CASES = [
    ("WAV", "PCM_16"),
    ("WAV", "PCM_24"),
    ("WAV", "FLOAT"),
    ("FLAC", None),
    ("OGG", None),      # default subtype VORBIS
    ("AIFF", None),
]
_EXT = {"WAV": "wav", "FLAC": "flac", "OGG": "ogg", "AIFF": "aiff", "MP3": "mp3"}


@pytest.mark.parametrize("fmt,subtype", _FMT_CASES)
def test_formats_decode_and_classify(client, model_ok, fmt, subtype):
    if not model_ok:
        pytest.skip("model unavailable")
    if fmt not in sf.available_formats():
        pytest.skip(f"{fmt} not supported by this libsndfile build")
    data = enc(_bursty(), fmt=fmt, subtype=subtype)
    r = upload(client, data, f"clip.{_EXT[fmt]}", f"audio/{_EXT[fmt]}")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


def test_mp3_decode_and_classify(client, model_ok):
    """libsndfile 1.2.2 can both write and read MP3; test a real sf-written MP3, else skip."""
    if not model_ok:
        pytest.skip("model unavailable")
    if "MP3" not in sf.available_formats():
        pytest.skip("MP3 not supported by this libsndfile build")
    try:
        data = enc(_bursty(), fmt="MP3")
    except Exception:
        pytest.skip("MP3 encode unavailable in this libsndfile build")
    r = upload(client, data, "clip.mp3", "audio/mpeg")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


# --- mono/stereo downmix + odd sample rates --------------------------------

def test_stereo_downmix_classifies(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    mono = _bursty()
    stereo = np.stack([mono, np.roll(mono, 500)], axis=1)   # (frames, 2)
    r = upload(client, enc(stereo), "st.wav", "audio/wav")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


@pytest.mark.parametrize("sr", [8000, 22050, 44100, 48000])
def test_odd_sample_rates_resampled(client, model_ok, sr):
    if not model_ok:
        pytest.skip("model unavailable")
    r = upload(client, enc(_bursty(sr=sr), sr=sr), "sr.wav", "audio/wav")
    assert r.status_code == 200, r.text
    assert_valid_prediction(r.text)


# --- determinism -----------------------------------------------------------

def test_determinism_upload_identical_bytes(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    data = enc(_bursty())
    a = upload(client, data, "a.wav", "audio/wav")
    b = upload(client, data, "a.wav", "audio/wav")
    assert a.status_code == 200 and b.status_code == 200
    assert a.text == b.text, "identical bytes must yield an identical response body"


def test_determinism_demo_predict(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    a = _predict_sample(client, "siren", corr={"noise": 0.3})
    b = _predict_sample(client, "siren", corr={"noise": 0.3})
    assert a.status_code == 200 and b.status_code == 200
    assert a.text == b.text


# --- slider-corruption monotonicity (physical) -----------------------------

def _quality(client, data, **form):
    r = upload(client, data, "a.wav", "audio/wav", **form)
    assert r.status_code == 200, r.text
    return r.json()


def test_noise_slider_lowers_snr(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    data = enc(_bursty())
    snr = [_quality(client, data, noise=v)["snrDb"] for v in (0.0, 0.5, 1.0)]
    assert snr[0] > snr[1] > snr[2], snr


def test_volume_slider_lowers_level(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    data = enc(_bursty())
    lvl = [_quality(client, data, volume=v)["levelDb"] for v in (0.0, 0.5, 1.0)]
    assert lvl[0] > lvl[1] > lvl[2], lvl


def test_clip_slider_raises_clippct(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    data = enc(_bursty())
    clp = [_quality(client, data, clip=v)["clipPct"] for v in (0.0, 0.5, 1.0)]
    assert clp[0] == 0.0
    assert clp[0] < clp[1] <= clp[2], clp


# --- non-finite audio is sanitised at the decode boundary, never a 500 -----

def test_float_wav_nan_should_not_500(client, model_ok):
    if not model_ok:
        pytest.skip("model unavailable")
    sig = np.full(16000, np.nan, dtype="float32")
    data = enc(sig, fmt="WAV", subtype="FLOAT")
    r = upload(client, data, "nan.wav", "audio/wav")
    assert r.status_code != 500, "uncaught 500 on NaN audio"
    if r.status_code == 200:
        assert_valid_prediction(r.text)          # must be strictly-valid, finite JSON
    else:
        assert 400 <= r.status_code < 500 and "detail" in r.json()
