"""L4 — adversarial / pathological but *decodable* audio through /api/predict/upload.

The audit's heart: bytes that libsndfile happily decodes but that could make the numeric
pipeline (resample -> log-mel -> ResNet -> softmax -> quality heuristics) blow up, emit a 500,
or serialize a JSON body containing `NaN`/`Infinity` tokens (which break a browser's JSON.parse).

Correctness bar for EVERY case (from BACKEND_CONTRACT.md):
    (a) status_code != 500, and
    (b) if 200 -> strictly-valid JSON (no NaN/Infinity tokens) whose numeric fields are ALL finite
        (probs finite and sum ~= 1), else a clean 4xx carrying a JSON {detail}.

Verified against the real source (app.py / predictor.py / features.py / corruption.py):
  - apply_slider_corruption() ends with `np.clip(w, -1, 1)`, so +Inf/-Inf decoded samples are
    mapped to +/-1 *before* inference -> the +-Inf FLOAT case is fully finite and returns 200.
  - NaN is NOT tamed by np.clip (np.clip(nan,-1,1) is nan); it propagates through mel/ResNet/softmax
    into probs/conf/quality, and FastAPI's json.dumps then raises on the NaN -> 500 with a
    non-JSON "Internal Server Error" body. That is a GENUINE defect -> xfail(strict=False) + finding.
"""
import io
import json
import math

import numpy as np
import pytest

from helpers_be import enc, upload


# ---- correctness oracle -----------------------------------------------------

def _reject_const(tok):
    # parse_constant fires for NaN / Infinity / -Infinity tokens -> make json.loads reject them.
    raise ValueError(f"non-JSON constant token in body: {tok!r}")


PROB_N = 10
STATUS = {"reliable", "uncertain", "reject"}
CAUSE = {"clean", "uncert", "noise", "quiet", "clip"}


def assert_contract(r):
    """Assert the universal prediction-response contract on a TestClient Response."""
    # (a) never a hard 500
    assert r.status_code != 500, f"500 crash: {r.text[:200]!r}"

    # strictly-valid JSON either way (no NaN/Infinity tokens allowed)
    body = json.loads(r.text, parse_constant=_reject_const)

    if r.status_code == 200:
        probs = body["probs"]
        assert isinstance(probs, list) and len(probs) == PROB_N
        assert all(isinstance(p, (int, float)) and math.isfinite(p) for p in probs)
        assert all(p >= -1e-6 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-3, f"probs sum={sum(probs)}"
        for k in ("conf", "entropy", "snrDb", "levelDb", "clipPct"):
            v = body[k]
            assert isinstance(v, (int, float)) and math.isfinite(v), f"{k}={v!r}"
        assert len(body["order"]) == PROB_N
        assert body["status"] in STATUS
        assert body["cause"] in CAUSE
    else:
        # the only acceptable non-200 is a clean client error with a JSON detail
        assert 400 <= r.status_code < 500, f"unexpected status {r.status_code}"
        assert "detail" in body


# ---- pathological signal builders (all in-memory, deterministic) ------------

N = 16000  # 1 s @ 16 kHz


def _float_wav(signal, sr=16000, subtype="FLOAT"):
    """WAV whose data subtype is 32-bit FLOAT (so NaN/Inf sample values survive the round-trip)."""
    return enc(signal, sr=sr, fmt="WAV", subtype=subtype)


def _double_wav(signal, sr=16000):
    """WAV whose data subtype is 64-bit DOUBLE, written from a genuine float64 array."""
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(signal, dtype="float64"), sr, format="WAV", subtype="DOUBLE")
    return buf.getvalue()


def _four_channel():
    base = 0.3 * np.sin(np.linspace(0, 50.0, N))
    return np.stack([base, base * 0.5, -base, base * 0.25], axis=1)  # [frames, 4]


# --- Cases that MUST already satisfy the contract (200, finite, or clean 4xx) ---
GOOD_CASES = [
    pytest.param(_float_wav(np.concatenate([np.full(N // 2, np.inf),
                                            np.full(N // 2, -np.inf)]).astype("float32")),
                 id="float_wav_plus_minus_inf"),          # clipped to +-1 before inference
    pytest.param(enc(np.zeros(N, dtype="float32")), id="all_zero_silence"),
    pytest.param(enc(np.full(N, 0.5, dtype="float32")), id="pure_dc_offset"),
    pytest.param(enc(np.ones(N, dtype="float32")), id="full_scale_ones_square"),
    pytest.param(enc(np.array([0.5], dtype="float32")), id="single_sample"),
    pytest.param(enc(np.array([0.5, -0.5], dtype="float32")), id="two_samples"),
    pytest.param(enc(np.linspace(-0.5, 0.5, 10).astype("float32")), id="ten_samples"),
    pytest.param(enc(np.linspace(-0.5, 0.5, 100).astype("float32")), id="hundred_samples"),
    pytest.param(enc(np.array([0.1, 0.2, -0.1, 0.3], dtype="float32"), sr=1), id="header_sr_1"),
    pytest.param(enc(np.sin(np.linspace(0, 100, 8000)).astype("float32"), sr=8000), id="header_sr_8000"),
    pytest.param(enc(np.sin(np.linspace(0, 100, 96000)).astype("float32"), sr=96000), id="header_sr_96000"),
    pytest.param(enc(np.sin(np.linspace(0, 100, 192000)).astype("float32"), sr=192000), id="header_sr_192000"),
    pytest.param(_float_wav(np.full(N, 1e6, dtype="float32")), id="huge_amplitude_1e6"),
    pytest.param(_float_wav(np.full(N, 1e-30, dtype="float32")), id="denormal_1e-30"),
    pytest.param(enc(np.tile([1.0, -1.0], N // 2).astype("float32")), id="alternating_plus_minus_1"),
    pytest.param(enc(_four_channel().astype("float32")), id="four_channel_downmix"),
    pytest.param(enc(np.sin(np.linspace(0, 1000, 16000 * 60)).astype("float32")), id="sixty_second_clip"),
]

# --- Non-finite samples: neutralised at the decode boundary (_to_mono_finite), must NOT crash ---
NAN_CASES = [
    pytest.param(_float_wav(np.full(N, np.nan, dtype="float32")), id="float_wav_nan"),
    pytest.param(_double_wav(np.full(N, np.nan)), id="double_wav_nan"),
]


@pytest.fixture(scope="module", autouse=True)
def _need_model(model_ok):
    if not model_ok:
        pytest.skip("predictor unavailable; L4 exercises the real inference pipeline")


@pytest.mark.parametrize("data", GOOD_CASES)
def test_pathological_audio_is_handled(client, data):
    """Decodable-but-nasty audio must give 200 with finite JSON, or a clean 4xx — never a 500."""
    assert_contract(upload(client, data, name="edge.wav", ctype="audio/wav"))


@pytest.mark.parametrize("data", NAN_CASES)
def test_nan_samples_should_not_crash(client, data):
    """A FLOAT/DOUBLE WAV of NaN must not crash: _to_mono_finite replaces non-finite samples at decode,
    so the response is 200 with a strictly-valid, finite JSON body — never a 500."""
    assert_contract(upload(client, data, name="edge.wav", ctype="audio/wav"))


def test_determinism_same_bytes_same_answer(client):
    """Same pathological bytes twice -> byte-identical response (no hidden nondeterminism)."""
    data = enc(np.ones(N, dtype="float32"))
    r1 = upload(client, data, name="edge.wav", ctype="audio/wav")
    r2 = upload(client, data, name="edge.wav", ctype="audio/wav")
    assert r1.status_code == r2.status_code == 200
    assert r1.text == r2.text
