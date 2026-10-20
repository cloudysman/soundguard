"""SoundGuard backend — test suite (real-inference contract).

Run from the project root (where app.py lives):
    pip install -r requirements.txt pytest httpx
    pytest -v

These tests exercise the real model: the seven demo chips are real UrbanSound8K clips in samples/,
and /api/predict runs the frozen-ResNet + head over them. There is no mock path, so a test that
would have documented a "mock quirk" is gone; validation that used to be a known gap is now enforced.

Sections
    1. Routing & endpoints
    2. /api/predict — classification (each clean clip predicts its own class)
    3. /api/predict — response-schema integrity
    4. /api/predict — corruption behaviour (physical monotonicity + cause)
    5. /api/predict — input validation (Literal sample, [0,1] sliders)
    6. /api/dashboard — real aggregate results
    7. Determinism & robustness
    8. /api/predict/upload — validation layer
    9. /api/health

If the model or the demo clips are missing the whole suite skips with a clear message, rather than
failing on every case — install requirements.txt and ensure samples/ is present.
"""
import os

import pytest
from fastapi.testclient import TestClient

import app
from predictor import get_predictor

client = TestClient(app.app)

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]
ALL_SAMPLES = ["dog", "siren", "jack", "horn", "kids", "gun", "ac"]

# The real model and the demo clips must be present, or the classification tests are meaningless.
_missing_clips = [s for s in ALL_SAMPLES
                  if not os.path.exists(os.path.join(app.SAMPLES_DIR, f"{s}.wav"))]
needs_model = pytest.mark.skipif(
    not get_predictor().available or _missing_clips,
    reason="real model unavailable or demo clips missing — install requirements.txt / restore samples/",
)


def predict(sample="dog", noise=0.0, volume=0.0, clip=0.0):
    """Helper: POST /api/predict, assert 200, return the parsed JSON body."""
    r = client.post("/api/predict", json={
        "sample": sample, "corr": {"noise": noise, "volume": volume, "clip": clip},
    })
    assert r.status_code == 200, r.text
    return r.json()


# ===========================================================================
# 1. ROUTING & ENDPOINTS
# ===========================================================================

def test_root_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]

def test_openapi_schema_available():
    assert client.get("/openapi.json").status_code == 200

def test_unknown_route_returns_404():
    assert client.get("/api/does-not-exist").status_code == 404

def test_dashboard_rejects_post_method_405():
    assert client.post("/api/dashboard").status_code == 405


# ===========================================================================
# 2. /api/predict — CLASSIFICATION
# ===========================================================================

# (sample id, expected top-1 class index, expected class name)
SAMPLE_TOPCLASS = [
    ("dog", 3, "dog_bark"),
    ("siren", 8, "siren"),
    ("jack", 7, "jackhammer"),
    ("horn", 1, "car_horn"),
    ("gun", 6, "gun_shot"),
    ("ac", 0, "air_conditioner"),
    ("kids", 2, "children_playing"),
]

@needs_model
@pytest.mark.parametrize("sample,top_idx,top_name", SAMPLE_TOPCLASS)
def test_each_clean_sample_predicts_its_own_class(sample, top_idx, top_name):
    d = predict(sample)
    assert d["order"][0][0] == top_idx
    assert CLASSES[d["order"][0][0]] == top_name


# ===========================================================================
# 3. /api/predict — RESPONSE-SCHEMA INTEGRITY
# ===========================================================================

REQUIRED_KEYS = ["probs", "order", "conf", "entropy",
                 "snrDb", "levelDb", "clipPct", "status", "cause"]

@needs_model
def test_all_required_keys_present():
    d = predict()
    for k in REQUIRED_KEYS:
        assert k in d, f"missing key: {k}"

@needs_model
def test_probs_length_is_10():
    assert len(predict()["probs"]) == 10

@needs_model
def test_probs_sum_to_one():
    assert abs(sum(predict()["probs"]) - 1.0) < 1e-6

@needs_model
def test_probs_all_in_unit_range():
    assert all(0.0 <= p <= 1.0 for p in predict()["probs"])

@needs_model
def test_order_is_a_permutation_of_all_classes():
    idxs = [o[0] for o in predict()["order"]]
    assert sorted(idxs) == list(range(10))

@needs_model
def test_order_is_sorted_descending():
    probs_in_order = [o[1] for o in predict()["order"]]
    assert probs_in_order == sorted(probs_in_order, reverse=True)

@needs_model
def test_conf_equals_top_prob():
    d = predict()
    assert d["conf"] == d["order"][0][1]

@needs_model
def test_entropy_in_unit_range():
    assert 0.0 <= predict()["entropy"] <= 1.0 + 1e-9

@needs_model
def test_status_is_one_of_three_values():
    assert predict()["status"] in {"reliable", "uncertain", "reject"}

@needs_model
def test_cause_is_a_known_value():
    assert predict()["cause"] in {"clean", "uncert", "noise", "quiet", "clip"}


# ===========================================================================
# 4. /api/predict — CORRUPTION BEHAVIOUR (physical monotonicity + cause)
# ===========================================================================
# The input-quality numbers are computed by physical heuristics on the corrupted waveform, so they
# move monotonically with the sliders. Model confidence does NOT — a small noise jitter can raise it
# — so it is deliberately not asserted here.

@needs_model
def test_snr_decreases_as_noise_increases():
    snrs = [predict("dog", noise=n)["snrDb"] for n in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert snrs == sorted(snrs, reverse=True), snrs

@needs_model
def test_level_decreases_as_volume_increases():
    levels = [predict("dog", volume=v)["levelDb"] for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert levels == sorted(levels, reverse=True), levels

@needs_model
def test_clip_percentage_increases_with_clip():
    clips = [predict("dog", clip=c)["clipPct"] for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert clips == sorted(clips), clips

@needs_model
def test_clean_input_is_not_rejected():
    assert predict("dog", 0, 0, 0)["status"] != "reject"

@needs_model
def test_heavy_noise_drives_rejection():
    assert predict("dog", noise=1.0)["status"] == "reject"

@needs_model
def test_heavy_noise_reports_noise_cause():
    assert predict("dog", noise=1.0)["cause"] == "noise"

@needs_model
def test_heavy_volume_reports_quiet_cause():
    assert predict("dog", volume=1.0)["cause"] == "quiet"

@needs_model
def test_entropy_rises_under_heavy_noise():
    assert predict("dog", noise=0.0)["entropy"] < predict("dog", noise=1.0)["entropy"]


# ===========================================================================
# 5. /api/predict — INPUT VALIDATION
# ===========================================================================

@needs_model
def test_empty_body_uses_defaults():
    r = client.post("/api/predict", json={})
    assert r.status_code == 200

@needs_model
def test_missing_sample_defaults_to_dog():
    r = client.post("/api/predict", json={"corr": {"noise": 0}})
    assert r.status_code == 200
    assert r.json()["order"][0][0] == 3          # dog_bark

@needs_model
def test_partial_corr_is_accepted():
    assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": 0.2}}).status_code == 200

def test_unknown_sample_is_rejected_422():
    # a Literal type means an unlisted id can never reach a file path (no traversal, no silent 'dog')
    assert client.post("/api/predict", json={"sample": "no_such_sample"}).status_code == 422

def test_empty_string_sample_is_rejected_422():
    assert client.post("/api/predict", json={"sample": ""}).status_code == 422

def test_path_traversal_sample_is_rejected_422():
    assert client.post("/api/predict", json={"sample": "../checkpoints/mlp_prod"}).status_code == 422

def test_slider_above_one_is_rejected_422():
    assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": 5}}).status_code == 422

def test_slider_below_zero_is_rejected_422():
    assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": -1}}).status_code == 422

def test_string_corr_value_is_rejected_422():
    assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": "loud"}}).status_code == 422

def test_null_corr_value_is_rejected_422():
    assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": None}}).status_code == 422

def test_malformed_json_is_rejected_422():
    r = client.post("/api/predict", content="{not valid json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422

@needs_model
def test_extra_unknown_fields_are_ignored():
    assert client.post("/api/predict", json={"sample": "dog", "corr": {}, "junk": 123}).status_code == 200


# ===========================================================================
# 6. /api/dashboard — REAL AGGREGATE RESULTS
# ===========================================================================

def test_dashboard_returns_200():
    assert client.get("/api/dashboard").status_code == 200

def test_dashboard_has_four_sections():
    d = client.get("/api/dashboard").json()
    assert {"reliability", "riskCoverage", "classAcc", "metrics"} <= set(d.keys())

def test_dashboard_is_not_pending():
    # the corrupted evaluation is present, not a placeholder
    assert client.get("/api/dashboard").json().get("_corrupted_pending") is False

def test_reliability_arrays_are_aligned():
    r = client.get("/api/dashboard").json()["reliability"]
    assert len(r["bins"]) == len(r["clean"]) == len(r["corrupted"])

def test_classacc_covers_the_ten_known_classes():
    ca = client.get("/api/dashboard").json()["classAcc"]
    assert len(ca) == 10
    assert {c["k"] for c in ca} == set(CLASSES)

def test_metrics_has_the_four_expected_entries():
    m = client.get("/api/dashboard").json()["metrics"]
    assert {x["k"] for x in m} == {"ECE", "NLL", "Brier", "Accuracy"}

def test_robustness_gap_is_nonnegative_sanity():
    # corruption should not IMPROVE per-class accuracy
    ca = client.get("/api/dashboard").json()["classAcc"]
    assert all(c["clean"] >= c["corrupt"] for c in ca)

def test_calibration_degrades_under_corruption_sanity():
    m = {x["k"]: x for x in client.get("/api/dashboard").json()["metrics"]}
    assert m["ECE"]["corrupt"] > m["ECE"]["clean"]


# ===========================================================================
# 7. DETERMINISM & ROBUSTNESS
# ===========================================================================

@needs_model
def test_same_request_is_deterministic():
    assert predict("dog", 0.3, 0.1, 0.05) == predict("dog", 0.3, 0.1, 0.05)

@needs_model
def test_zero_corruption_equals_clean_call():
    assert predict("dog", 0, 0, 0) == predict("dog")

@needs_model
def test_many_requests_do_not_crash():
    for i in range(30):
        n = (i % 11) / 10
        assert client.post("/api/predict", json={"sample": "dog", "corr": {"noise": n}}).status_code == 200


# ===========================================================================
# 8. /api/predict/upload — VALIDATION LAYER
# ===========================================================================

def test_upload_requires_a_file():
    assert client.post("/api/predict/upload", data={"noise": 0}).status_code == 422

def test_upload_rejects_non_audio_415():
    r = client.post("/api/predict/upload",
                    files={"file": ("notes.txt", b"hello world", "text/plain")})
    assert r.status_code == 415

def test_upload_rejects_empty_file_400():
    r = client.post("/api/predict/upload", files={"file": ("clip.wav", b"", "audio/wav")})
    assert r.status_code == 400

def test_upload_rejects_oversized_file_413():
    big = b"\x00" * (10 * 1024 * 1024 + 1)
    r = client.post("/api/predict/upload", files={"file": ("clip.wav", big, "audio/wav")})
    assert r.status_code == 413

def test_upload_rejects_undecodable_audio_400():
    # passes the type gate but is not valid audio -> loud 400, never a fabricated prediction
    r = client.post("/api/predict/upload",
                    files={"file": ("clip.wav", b"RIFFfakewavbytes", "audio/wav")},
                    data={"noise": 0.0})
    assert r.status_code == 400

def test_upload_rejects_out_of_range_slider_422():
    r = client.post("/api/predict/upload",
                    files={"file": ("clip.wav", b"RIFFfakewavbytes", "audio/wav")},
                    data={"noise": 5.0})
    assert r.status_code == 422

@needs_model
def test_upload_classifies_a_real_wav():
    # the real dog clip, uploaded through the file path, must classify as dog_bark
    with open(os.path.join(app.SAMPLES_DIR, "dog.wav"), "rb") as f:
        data = f.read()
    r = client.post("/api/predict/upload", files={"file": ("dog.wav", data, "audio/wav")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["order"][0][0] == 3                 # dog_bark
    assert len(d["probs"]) == 10


# ===========================================================================
# 9. /api/health
# ===========================================================================

def test_health_reports_model_and_clips():
    d = client.get("/api/health").json()
    assert "model_available" in d and "missing_demo_clips" in d
