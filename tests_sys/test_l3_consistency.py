"""L3 — cross-layer number consistency: the same number everywhere.

The chain under test:
    rerun_results/mlp_fold*.pt  ->  metrics.py  ->  results/dashboard.json  ->  GET /api/dashboard  ->  report

This layer does NOT re-test metric formulas (tests_ai), endpoint contracts (tests_be) or rendering
(tests_fe). It tests that the numbers those layers each hold agree with each other, and that the
published headline figures are the honest out-of-fold ones stated in the report:

    clean:      accuracy 0.700   ECE 0.186   NLL 1.466   Brier 0.476
    corrupted:  accuracy 0.527   ECE 0.323

Any drift in that chain is a HIGH finding: a number that disagrees with its own source is exactly
the fabricated-number failure this project exists to prevent.
"""
import json
import os
import re

import numpy as np
import pytest

import eval as EV
import metrics as M
from helpers_sys import CLASSES, ROOT

STRICT = dict(parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-JSON token: {c}")))

# The report's headline table. round(x, 3) of the honest out-of-fold numbers.
HEADLINE_CLEAN = {"Accuracy": 0.700, "ECE": 0.186, "NLL": 1.466, "Brier": 0.476}
HEADLINE_CORRUPT = {"Accuracy": 0.527, "ECE": 0.323}


# --------------------------------------------------------------------------- helpers
def _metric(dash, key):
    hits = [m for m in dash["metrics"] if m["k"] == key]
    assert len(hits) == 1, f"expected exactly one {key!r} row, got {len(hits)}"
    return hits[0]


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


@pytest.fixture(scope="session")
def recomputed(rerun_logits):
    """metrics.py applied to the honest out-of-fold logits — the single source of truth."""
    logits, labels = rerun_logits
    return {
        "Accuracy": M.accuracy(logits, labels),
        "ECE": M.ece(logits, labels, n_bins=15),
        "NLL": M.nll(logits, labels),
        "Brier": M.brier(logits, labels),
        "reliability": M.reliability_curve(logits, labels, n_bins=15),
        "riskCoverage": M.risk_coverage_curve(logits, labels),
        "per_class": M.per_class_accuracy(logits, labels, num_classes=len(CLASSES)),
        "n": int(len(labels)),
    }


@pytest.fixture(scope="session")
def data_js_text():
    p = os.path.join(ROOT, "static", "soundguard-data.js")
    if not os.path.exists(p):
        pytest.skip("static/soundguard-data.js absent")
    with open(p, encoding="utf-8") as f:
        return f.read()


# ============================ 1. logits -> metrics.py -> dashboard.json ============================
def test_dashboard_clean_metrics_are_recomputable_from_rerun_logits(dashboard, recomputed):
    """Every clean headline in dashboard.json must fall out of metrics.py over rerun_results."""
    # eval.py publishes ECE at 4 dp and the rest at 3 dp; allow only that rounding.
    for key, dp in (("Accuracy", 3), ("ECE", 4), ("NLL", 3), ("Brier", 3)):
        published = _metric(dashboard, key)["clean"]
        expected = round(recomputed[key], dp)
        assert published == pytest.approx(expected, abs=10 ** -dp / 2 + 1e-12), (
            f"{key}: dashboard.json says {published}, metrics.py over rerun_results says "
            f"{recomputed[key]!r} (-> {expected})"
        )


def test_dashboard_clean_metrics_match_a_full_eval_rebuild(dashboard, rerun_logits):
    """Rebuild the metrics block the way eval.py does and compare field for field."""
    logits, labels = rerun_logits
    block = EV._block(logits, labels)
    got = {m["k"]: m["clean"] for m in dashboard["metrics"]}
    assert got["Accuracy"] == pytest.approx(round(block["accuracy"], 3), abs=1e-9)
    assert got["ECE"] == pytest.approx(round(block["ece"], 4), abs=1e-9)
    assert got["NLL"] == pytest.approx(round(block["nll"], 3), abs=1e-9)
    assert got["Brier"] == pytest.approx(round(block["brier"], 3), abs=1e-9)


def test_dashboard_reliability_clean_matches_recomputed_curve(dashboard, recomputed):
    rel = recomputed["reliability"]
    assert dashboard["reliability"]["bins"] == [round(float(x), 3) for x in rel["centers"]]
    assert dashboard["reliability"]["clean"] == [round(float(x), 3) for x in rel["accuracy"]]


def test_dashboard_risk_coverage_clean_matches_recomputed_curve(dashboard, recomputed):
    rc = recomputed["riskCoverage"]
    assert dashboard["riskCoverage"]["coverage"] == [round(float(x), 3) for x in rc["coverage"]]
    assert dashboard["riskCoverage"]["clean"] == [round(float(x), 3) for x in rc["risk"]]


def test_dashboard_class_acc_clean_matches_recomputed_per_class(dashboard, recomputed):
    expected = {CLASSES[p["class"]]: round(p["accuracy"] * 100, 1) for p in recomputed["per_class"]}
    got = {e["k"]: e["clean"] for e in dashboard["classAcc"]}
    assert got == expected


def test_class_acc_ordering_is_the_published_descending_order(dashboard):
    """eval.py sorts classAcc by descending clean accuracy; the report reads it top-down."""
    clean = [e["clean"] for e in dashboard["classAcc"]]
    assert clean == sorted(clean, reverse=True), f"classAcc is not sorted descending: {clean}"


def test_published_accuracy_is_the_pooled_out_of_fold_number(dashboard, rerun_logits):
    """What is published is the accuracy over the pooled out-of-fold predictions. Check it is not
    silently a mean-of-per-fold-means instead (the two differ once folds are unequal in size)."""
    logits, labels = rerun_logits
    pooled = M.accuracy(logits, labels)
    assert 0.0 <= pooled <= 1.0
    assert _metric(dashboard, "Accuracy")["clean"] == pytest.approx(round(pooled, 3), abs=1e-9)


# ============================ 2. the published headline numbers ============================
@pytest.mark.parametrize("key, want", sorted(HEADLINE_CLEAN.items()))
def test_published_clean_headline_is_the_honest_number(dashboard, key, want):
    got = round(float(_metric(dashboard, key)["clean"]), 3)
    assert got == want, f"clean {key}: report states {want}, dashboard.json publishes {got}"


@pytest.mark.parametrize("key, want", sorted(HEADLINE_CORRUPT.items()))
def test_published_corrupt_headline_is_the_honest_number(dashboard, key, want):
    got = round(float(_metric(dashboard, key)["corrupt"]), 3)
    assert got == want, f"corrupted {key}: report states {want}, dashboard.json publishes {got}"


def test_clean_headline_matches_report_when_recomputed_from_logits(recomputed):
    """The report's numbers must survive a recomputation from the raw logits, not just from the file."""
    for key, want in HEADLINE_CLEAN.items():
        assert round(recomputed[key], 3) == want, (
            f"{key}: report states {want}, metrics.py over rerun_results gives {recomputed[key]!r}"
        )


def test_accuracy_row_is_flagged_pct_and_higher_is_better(dashboard):
    """The frontend renders Accuracy as a percentage and every other row as lower-is-better;
    a wrong flag silently inverts the published comparison."""
    acc = _metric(dashboard, "Accuracy")
    assert acc["lowerBetter"] is False and acc.get("pct") is True
    for key in ("ECE", "NLL", "Brier"):
        row = _metric(dashboard, key)
        assert row["lowerBetter"] is True
        assert not row.get("pct", False)


def test_corruption_degrades_every_headline_in_the_published_direction(dashboard):
    """Corrupted audio must be worse on every metric, in the direction lowerBetter declares."""
    for row in dashboard["metrics"]:
        clean, corrupt = row["clean"], row["corrupt"]
        assert _finite(clean) and _finite(corrupt), row
        if row["lowerBetter"]:
            assert corrupt >= clean, f"{row['k']}: corrupt {corrupt} better than clean {clean}"
        else:
            assert corrupt <= clean, f"{row['k']}: corrupt {corrupt} better than clean {clean}"


# ============================ 3. dashboard.json -> GET /api/dashboard ============================
def test_served_dashboard_is_byte_for_byte_the_file(client):
    r = client.get("/api/dashboard")
    assert r.status_code == 200, r.text
    served = json.loads(r.text, **STRICT)
    with open(os.path.join(ROOT, "results", "dashboard.json"), encoding="utf-8") as f:
        on_disk = json.load(f)
    assert served == on_disk


def test_served_dashboard_carries_the_headline_unchanged(client, dashboard):
    served = json.loads(client.get("/api/dashboard").text, **STRICT)
    for key, want in HEADLINE_CLEAN.items():
        assert round(float(_metric(served, key)["clean"]), 3) == want
    for key, want in HEADLINE_CORRUPT.items():
        assert round(float(_metric(served, key)["corrupt"]), 3) == want


def test_served_dashboard_contains_only_finite_json_numbers(client):
    served = json.loads(client.get("/api/dashboard").text, **STRICT)

    def walk(node, path="$"):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, float):
            assert np.isfinite(node), f"non-finite number at {path}: {node}"

    walk(served)
    assert not re.search(r"\b(NaN|Infinity|-Infinity)\b", client.get("/api/dashboard").text)


def test_served_dashboard_is_stable_across_repeated_reads(client):
    a = json.loads(client.get("/api/dashboard").text, **STRICT)
    b = json.loads(client.get("/api/dashboard").text, **STRICT)
    assert a == b


# ============================ 4. dashboard structural invariants ============================
def test_corrupted_is_not_pending(dashboard):
    assert dashboard["_corrupted_pending"] is False, (
        "_corrupted_pending is True — the UI would hide the corrupted column the report publishes"
    )


def test_no_corrupted_field_is_null_when_not_pending(dashboard):
    assert dashboard["reliability"]["corrupted"] is not None
    assert dashboard["riskCoverage"]["corrupted"] is not None
    for row in dashboard["metrics"]:
        assert row["corrupt"] is not None, row["k"]
    for e in dashboard["classAcc"]:
        assert e["corrupt"] is not None, e["k"]


def test_class_acc_covers_exactly_the_ten_classes(dashboard):
    keys = [e["k"] for e in dashboard["classAcc"]]
    assert len(keys) == 10 and len(set(keys)) == 10
    assert sorted(keys) == sorted(CLASSES)


def test_class_acc_clean_beats_corrupt_for_every_class(dashboard):
    bad = [(e["k"], e["clean"], e["corrupt"]) for e in dashboard["classAcc"] if e["clean"] < e["corrupt"]]
    assert not bad, f"corruption improved these classes: {bad}"


def test_class_acc_values_are_percentages_in_range(dashboard):
    for e in dashboard["classAcc"]:
        for f in ("clean", "corrupt"):
            assert _finite(e[f]) and 0.0 <= e[f] <= 100.0, e


def test_class_acc_clean_mean_is_consistent_with_headline_accuracy(dashboard, recomputed):
    """Per-class accuracies are support-weighted into the headline; check the weighted mean matches."""
    counts = {CLASSES[p["class"]]: p["count"] for p in recomputed["per_class"]}
    total = sum(counts.values())
    weighted = sum(e["clean"] / 100.0 * counts[e["k"]] for e in dashboard["classAcc"]) / total
    assert weighted == pytest.approx(_metric(dashboard, "Accuracy")["clean"], abs=5e-3)


def test_reliability_and_risk_coverage_arrays_are_well_formed(dashboard):
    rel = dashboard["reliability"]
    assert len(rel["bins"]) == 15 and len(rel["clean"]) == 15 and len(rel["corrupted"]) == 15
    rc = dashboard["riskCoverage"]
    assert len(rc["coverage"]) == len(rc["clean"]) == len(rc["corrupted"]) == 10
    assert rc["coverage"] == sorted(rc["coverage"])
    for series in ("clean", "corrupted"):
        assert all(0.0 <= x <= 1.0 for x in rel[series]), rel[series]
        assert all(0.0 <= x <= 1.0 for x in rc[series]), rc[series]


def test_risk_at_full_coverage_equals_one_minus_headline_accuracy(dashboard):
    """Coverage 1.0 accepts everything, so risk must be exactly the published error rate."""
    rc = dashboard["riskCoverage"]
    assert rc["coverage"][-1] == 1.0
    acc = _metric(dashboard, "Accuracy")["clean"]
    assert rc["clean"][-1] == pytest.approx(1.0 - acc, abs=1.5e-3), (
        f"risk@1.0={rc['clean'][-1]} vs 1-accuracy={1 - acc}"
    )


def test_corrupt_risk_at_full_coverage_equals_one_minus_corrupt_accuracy(dashboard):
    rc = dashboard["riskCoverage"]
    acc = _metric(dashboard, "Accuracy")["corrupt"]
    assert rc["corrupted"][-1] == pytest.approx(1.0 - acc, abs=1.5e-3)


def test_selective_prediction_beats_full_coverage_on_both_series(dashboard):
    """The paper's claim: confidence ranks usefully — risk at 10% coverage < risk at 100%."""
    rc = dashboard["riskCoverage"]
    for series in ("clean", "corrupted"):
        assert rc[series][0] < rc[series][-1], series


def test_dashboard_head_is_the_served_head(dashboard, predictor, model_ok):
    assert dashboard["_head"] == "mlp"
    if model_ok:
        assert predictor.head_name == dashboard["_head"], (
            f"dashboard reports head={dashboard['_head']} but the server runs {predictor.head_name}"
        )


# ============================ 5. the corrupted numbers trace to their source ============================
def test_corrupted_headline_is_recomputable_from_its_source_dump(dashboard):
    """`_source` names rerun_results/corrupted_mlp.pt; the published corrupted numbers must come out of it."""
    import torch

    p = os.path.join(ROOT, "rerun_results", "corrupted_mlp.pt")
    if not os.path.exists(p):
        pytest.skip("rerun_results/corrupted_mlp.pt absent")
    d = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(d, dict):
        pytest.skip(f"unexpected dump type {type(d)}")
    lk = next((k for k in ("logits", "last_logits") if k in d), None)
    yk = next((k for k in ("labels", "y", "targets") if k in d), None)
    if lk is None or yk is None:
        pytest.skip(f"dump keys {sorted(d)} not recognised")
    logits = np.asarray(d[lk], dtype=np.float64)
    labels = np.asarray(d[yk]).astype(int)
    for key, fn, dp in (("Accuracy", M.accuracy, 3), ("ECE", lambda a, b: M.ece(a, b, 15), 4),
                        ("NLL", M.nll, 3), ("Brier", M.brier, 3)):
        published = _metric(dashboard, key)["corrupt"]
        assert published == pytest.approx(round(fn(logits, labels), dp), abs=10 ** -dp / 2 + 1e-12), key


def test_rerun_logits_cover_the_whole_fold_pool(rerun_logits):
    logits, labels = rerun_logits
    assert logits.ndim == 2 and logits.shape[1] == len(CLASSES)
    assert len(labels) == len(logits) and len(labels) > 8000
    assert np.isfinite(logits).all()
    assert set(np.unique(labels).tolist()) == set(range(len(CLASSES)))


# ============================ 6. one class list, everywhere ============================
def _classes_from_js(text):
    m = re.search(r"SG\.classes\s*=\s*\[(.*?)\]", text, re.S)
    assert m, "SG.classes array not found in static/soundguard-data.js"
    return re.findall(r"['\"]([A-Za-z_]+)['\"]", m.group(1))


def test_class_list_identical_across_config_predictor_metrics_and_frontend(
    predictor, model_ok, data_js_text
):
    with open(os.path.join(ROOT, "checkpoints", "feature_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    sources = {
        "checkpoints/feature_config.json": list(cfg["classes"]),
        "eval.CLASSES": list(EV.CLASSES),
        "conftest.CLASSES": list(CLASSES),
        "static/soundguard-data.js": _classes_from_js(data_js_text),
    }
    if model_ok:
        sources["predictor.classes"] = list(predictor.classes)

    ref = sources["checkpoints/feature_config.json"]
    for name, lst in sources.items():
        assert lst == ref, f"class list in {name} differs from feature_config.json:\n{lst}\n{ref}"
    assert ref == CLASSES
    assert cfg["num_classes"] == len(ref) == 10


def test_class_indices_line_up_between_config_and_dashboard(dashboard, data_js_text):
    """classAcc keys are class NAMES; the frontend indexes predictions by POSITION. A name that
    exists but at a different index would mislabel every prediction while every test still passes."""
    js = _classes_from_js(data_js_text)
    for e in dashboard["classAcc"]:
        assert e["k"] in js, e["k"]
    with open(os.path.join(ROOT, "checkpoints", "feature_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for i, name in enumerate(cfg["classes"]):
        assert js[i] == name, f"index {i}: config says {name}, frontend says {js[i]}"


def test_predictor_order_vector_indexes_the_same_class_list(client, model_ok, data_js_text):
    """End of the chain: a live prediction's `order` indices must address the frontend class list."""
    if not model_ok:
        pytest.skip("model unavailable")
    r = client.post("/api/predict", json={"sample": "dog"})
    assert r.status_code == 200, r.text
    body = json.loads(r.text, **STRICT)
    js = _classes_from_js(data_js_text)
    idx = [o[0] for o in body["order"]]
    assert sorted(idx) == list(range(len(js))), "order does not cover the class list exactly once"
    assert js[body["order"][0][0]] == "dog_bark"


# ============================ 7. no fabricated numbers in the frontend ============================
@pytest.mark.parametrize("ident", ["metrics", "reliability", "riskCoverage", "classAcc"])
def test_frontend_data_file_defines_no_hardcoded_result_arrays(data_js_text, ident):
    """soundguard-data.js is presentation constants only. A hard-coded metrics/reliability/
    riskCoverage/classAcc literal here is a fabricated number that looks real — the incident."""
    m = re.search(rf"SG\.{ident}\s*=", data_js_text)
    assert m is None, f"SG.{ident} is defined in static/soundguard-data.js: {data_js_text[m.start():m.start()+120]!r}"


def test_frontend_data_file_carries_no_published_metric_literals(data_js_text):
    """The exact headline values must never appear as literals in the data file."""
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", data_js_text, flags=re.S)   # strip comments
    for value in ("0.186", "0.1859", "1.466", "0.476", "0.527", "0.323", "0.3232", "91.7", "70.1"):
        assert value not in code, f"published metric literal {value} hard-coded in soundguard-data.js"


def test_frontend_data_file_has_no_prediction_or_confidence_literals(data_js_text):
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", data_js_text, flags=re.S)
    for banned in ("SG.predictions", "SG.mock", "SG.fake", "SG.dashboard", "SG.results"):
        assert banned not in code, f"{banned} present — the mock path must stay gone"


def test_only_seed_and_duration_numbers_survive_in_the_data_file(data_js_text):
    """Whitelist the numeric literals that are allowed: sample durations and canvas seeds."""
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", data_js_text, flags=re.S)
    allowed = {"4.0", "1337", "7", "99", "2024", "451", "5150", "60", "100", "0"}
    found = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", code))
    unexpected = found - allowed
    assert not unexpected, f"unexplained numeric literals in soundguard-data.js: {sorted(unexpected)}"
