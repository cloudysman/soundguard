"""L1 — deployment integrity: can this be installed and started from scratch, is anything missing?

This layer does not re-test endpoint semantics (tests_be), browser behaviour (tests_fe) or model
quality (tests_ai). It asks the question that comes *before* all of those: given a clean machine,
`pip install -r requirements.txt && uvicorn app:app`, does the deployment actually contain
everything it needs and expose exactly what it claims?

Checks, in order:
  1. dependency closure  — every third-party module imported by the top-level source is installable
     from requirements.txt (directly, or as a declared hard dependency of something in it).
  2. artifacts           — checkpoint, feature config, the seven demo clips, dashboard.json, the
     static bundle: present, non-empty, well-formed.
  3. config              — feature_config.json carries the keys features.py/predictor.py read, and
     its class list is the canonical taxonomy.
  4. surface             — /openapi.json exposes exactly the four intended API routes, GET / serves
     index.html, every asset the page references is fetchable with a sane content-type.
  5. self-report         — /api/health says the model is loaded and no demo clip is missing.
  6. hygiene             — no unused imports (reported, not failed), no leftover mock symbols.

Everything here is filesystem + already-loaded-app introspection; no inference is run.
"""
import ast
import json
import os
import re
import sys

import pytest

from helpers_sys import CLASSES, DEMO_IDS, ROOT

# ---------------------------------------------------------------- discovery helpers

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "_old", "node_modules", "uploads",
             "tests_fe", "tests_be", "tests_ai", "tests_sys", "sg", "rerun_cache",
             "rerun_results", "results_paper", "checkpoints", "samples", "results", "static"}

# The modules that constitute the served deployment (app.py's import closure + the offline
# aggregation step that produces the dashboard it serves).
DEPLOY_MODULES = ["app", "predictor", "features", "heads", "corruption", "rejection",
                  "metrics", "eval"]

# Working-tree research scripts: not part of the deployed subset, scanned only if present.
RESEARCH_SCRIPTS = ["pipeline.py", "summarise_all.py", "summarise_paper.py"]

STATIC_DIR = os.path.join(ROOT, "static")


def _top_level_sources():
    """Top-level .py files that are project source (tests excluded)."""
    out = []
    for name in sorted(os.listdir(ROOT)):
        if not name.endswith(".py") or name.startswith("test_") or name == "conftest.py":
            continue
        p = os.path.join(ROOT, name)
        if os.path.isfile(p):
            out.append(p)
    return out


def _local_module_names():
    """Names that resolve to project-local modules/packages, not PyPI distributions."""
    names = set()
    for name in os.listdir(ROOT):
        p = os.path.join(ROOT, name)
        if name.endswith(".py"):
            names.add(name[:-3])
        elif os.path.isdir(p) and os.path.exists(os.path.join(p, "__init__.py")):
            names.add(name)
    return names


def _imported_roots(path):
    """Root module name of every import in `path`, incl. function-local/lazy imports."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                roots.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative import -> local by definition
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def _third_party_imports(paths):
    """{root module -> [files importing it]} for imports that are neither stdlib nor local."""
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    local = _local_module_names()
    out = {}
    for p in paths:
        for r in _imported_roots(p):
            if r in stdlib or r in local:
                continue
            out.setdefault(r, []).append(os.path.basename(p))
    return out


# ---------------------------------------------------------------- requirements parsing

_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _declared_dists():
    """Distribution names listed in requirements.txt (extras/pins/comments stripped)."""
    path = os.path.join(ROOT, "requirements.txt")
    assert os.path.exists(path), "requirements.txt is missing from the deployment"
    dists = set()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#")[0].strip()
            if not line or line.startswith("-"):
                continue
            m = _REQ_LINE.match(line)
            if m:
                dists.add(m.group(1).lower().replace("_", "-"))
    return dists


def _dist_closure(dists):
    """Declared dists + their *unconditional* hard runtime dependencies (one level of transitivity
    is enough here and is read from installed metadata, not guessed). pip installs these too, so a
    module they provide cannot be missing after `pip install -r requirements.txt`."""
    from importlib import metadata

    closure = set(dists)
    for d in list(dists):
        try:
            reqs = metadata.requires(d) or []
        except metadata.PackageNotFoundError:
            continue
        for r in reqs:
            if ";" in r:                          # conditional / extra -> not guaranteed
                continue
            m = _REQ_LINE.match(r)
            if m:
                closure.add(m.group(1).lower().replace("_", "-"))
    return closure


def _providing_dists(module_root):
    """Distributions that actually provide the importable top-level `module_root`, from the
    installed environment's packages_distributions() map."""
    from importlib import metadata

    mapping = metadata.packages_distributions()
    return {d.lower().replace("_", "-") for d in mapping.get(module_root, [])}


# ---------------------------------------------------------------- 1. dependency closure

def test_requirements_is_parseable_and_pins_the_serving_stack():
    dists = _declared_dists()
    assert dists, "requirements.txt declares nothing"
    # The stack the README/app docstring promises: web layer + inference layer.
    for must in ("fastapi", "uvicorn", "python-multipart", "torch", "torchaudio",
                 "torchvision", "numpy", "soundfile"):
        assert must in dists, f"requirements.txt does not declare {must!r} (has {sorted(dists)})"


def test_every_thirdparty_import_in_the_deployment_is_installable():
    """Each third-party root imported by the top-level source must be provided by a distribution
    that `pip install -r requirements.txt` brings in — directly, or as a declared hard dependency
    of a declared distribution."""
    imports = _third_party_imports(_top_level_sources())
    assert imports, "no third-party imports found — the scan is broken, not the deployment"

    closure = _dist_closure(_declared_dists())
    missing = {}
    for root, files in sorted(imports.items()):
        providers = _providing_dists(root) or {root.lower().replace("_", "-")}
        if not (providers & closure):
            missing[root] = (sorted(providers), files)
    assert not missing, (
        "modules imported by the deployment but not installable from requirements.txt: "
        + "; ".join(f"{r} (provided by {p}, imported in {f})" for r, (p, f) in missing.items())
    )


def test_pydantic_is_imported_directly_but_only_declared_transitively():
    """app.py does `from pydantic import BaseModel, Field` while requirements.txt names only
    fastapi. It installs today because fastapi hard-depends on pydantic, so this is hygiene, not an
    outage: the assertion below is the *direct* declaration, and it is the thing that is missing."""
    from importlib import metadata

    imports = _third_party_imports(_top_level_sources())
    if "pydantic" not in imports:
        pytest.skip("pydantic is not imported directly by the top-level source")

    # The safety net that keeps this from being a real break — assert it really is there.
    fastapi_reqs = " ".join(metadata.requires("fastapi") or [])
    assert "pydantic" in fastapi_reqs, "fastapi no longer guarantees pydantic — now it IS a break"

    assert "pydantic" in _declared_dists(), (
        "app.py imports pydantic directly, so requirements.txt must declare it rather than rely on "
        "fastapi happening to pull it in")


def test_research_scripts_dependencies_are_declared():
    present = [os.path.join(ROOT, f) for f in RESEARCH_SCRIPTS
               if os.path.exists(os.path.join(ROOT, f))]
    if not present:
        pytest.skip("no research scripts in this deployment (served subset only)")
    closure = _dist_closure(_declared_dists())
    undeclared = {}
    for root, files in sorted(_third_party_imports(present).items()):
        providers = _providing_dists(root) or {root.lower().replace("_", "-")}
        if not (providers & closure):
            undeclared[root] = files
    assert not undeclared, f"undeclared: {undeclared}"


# ---------------------------------------------------------------- 2. artifacts

def _nonempty(rel, min_bytes=1):
    p = os.path.join(ROOT, rel)
    assert os.path.exists(p), f"missing artifact: {rel}"
    size = os.path.getsize(p)
    assert size >= min_bytes, f"artifact {rel} is empty/truncated ({size} bytes)"
    return p


def test_model_artifacts_present_and_non_empty():
    _nonempty(os.path.join("checkpoints", "mlp_prod.pth"), min_bytes=1024)
    _nonempty(os.path.join("checkpoints", "feature_config.json"), min_bytes=2)


def test_all_seven_demo_clips_present_and_non_empty():
    for sid in DEMO_IDS:
        _nonempty(os.path.join("samples", f"{sid}.wav"), min_bytes=44)   # >= a WAV header


def test_demo_clips_are_decodable_audio_with_sane_shape():
    """A present-but-corrupt clip is a 503 at runtime, so presence alone is not integrity.
    Metadata only — no decode of the full signal, no model."""
    import soundfile as sf

    for sid in DEMO_IDS:
        info = sf.info(os.path.join(ROOT, "samples", f"{sid}.wav"))
        assert info.frames > 0, f"{sid}.wav has no audio frames"
        assert info.samplerate > 0, f"{sid}.wav reports samplerate {info.samplerate}"
        assert 0.2 <= info.frames / info.samplerate <= 60.0, (
            f"{sid}.wav duration {info.frames / info.samplerate:.2f}s is implausible for a demo clip")


def test_dashboard_artifact_present_and_strictly_valid_json():
    p = _nonempty(os.path.join("results", "dashboard.json"), min_bytes=2)
    with open(p, encoding="utf-8") as f:
        text = f.read()
    d = json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    for key in ("reliability", "riskCoverage", "classAcc", "metrics"):
        assert key in d, f"dashboard.json is missing the {key!r} block the frontend reads"


def test_static_bundle_present_and_non_empty():
    _nonempty(os.path.join("static", "index.html"), min_bytes=100)
    for asset in _referenced_assets():
        _nonempty(os.path.join("static", asset), min_bytes=1)


def test_checkpoint_loads_and_its_shapes_match_feature_config():
    """The deployment fails at startup, not at request time, if the checkpoint disagrees with the
    config — so the two are an integrity pair. state_dict only; the model is not run."""
    import torch

    with open(os.path.join(ROOT, "checkpoints", "feature_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    state = torch.load(os.path.join(ROOT, "checkpoints", "mlp_prod.pth"),
                       map_location="cpu", weights_only=True)
    assert state, "mlp_prod.pth holds an empty state_dict"
    shapes = {k: tuple(v.shape) for k, v in state.items() if hasattr(v, "shape")}
    first = shapes["classifier.0.weight"]
    last = shapes["classifier.4.weight"]
    assert first[1] == cfg["in_features"], (
        f"checkpoint expects {first[1]} input features, feature_config says {cfg['in_features']}")
    assert last[0] == cfg["num_classes"] == len(cfg["classes"]), (
        f"checkpoint emits {last[0]} classes; config says {cfg['num_classes']}/{len(cfg['classes'])}")


# ---------------------------------------------------------------- 3. config

def test_feature_config_has_every_key_the_code_reads():
    import features

    with open(os.path.join(ROOT, "checkpoints", "feature_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    for k in features.DEFAULT_CONFIG:
        assert k in cfg, f"feature_config.json lacks {k!r}; serving would silently fall back to " \
                         f"the DEFAULT_CONFIG value {features.DEFAULT_CONFIG[k]!r}"
    assert cfg["sample_rate"] == 16000 and cfg["max_length_s"] == 4
    assert cfg["n_mels"] == 64 and cfg["n_fft"] == 1024 and cfg["hop_length"] == 512
    assert cfg["in_features"] == 512 and cfg["num_classes"] == 10
    assert cfg["normalize"] == "minmax01_then_imagenet"


def test_feature_config_classes_are_the_canonical_taxonomy():
    with open(os.path.join(ROOT, "checkpoints", "feature_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["classes"] == CLASSES, "feature_config.json class order differs from the taxonomy — " \
                                      "every served label would be permuted"


def test_class_list_is_identical_in_every_place_it_is_hard_coded():
    """predictor.py, eval.py and the frontend each carry their own copy; a deployment where they
    drift renders correct probabilities under wrong names."""
    import eval as E
    import predictor as P

    assert P.CLASSES == CLASSES, "predictor.CLASSES drifted from the taxonomy"
    assert E.CLASSES == CLASSES, "eval.CLASSES drifted from the taxonomy"

    js = open(os.path.join(STATIC_DIR, "soundguard-data.js"), encoding="utf-8").read()
    block = re.search(r"SG\.classes\s*=\s*\[(.*?)\]", js, re.S)
    assert block, "static/soundguard-data.js no longer declares SG.classes"
    assert re.findall(r"'([a-z_]+)'", block.group(1)) == CLASSES, \
        "the frontend's class list drifted from the backend taxonomy"


# ---------------------------------------------------------------- 4. served surface

EXPECTED_ROUTES = {
    ("/api/predict", "post"),
    ("/api/predict/upload", "post"),
    ("/api/dashboard", "get"),
    ("/api/health", "get"),
}


def test_openapi_exposes_exactly_the_intended_routes(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200, r.text
    spec = json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    got = {(p, m) for p, ops in spec["paths"].items()
           for m in ops if m in ("get", "post", "put", "patch", "delete")}
    assert got == EXPECTED_ROUTES, f"route surface drifted: extra={got - EXPECTED_ROUTES}, " \
                                   f"missing={EXPECTED_ROUTES - got}"


def test_root_serves_the_real_index_html(client):
    r = client.get("/")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")
    disk = open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    assert r.text == disk, "GET / does not serve static/index.html verbatim"
    assert "<title>" in r.text and "SoundGuard" in r.text


def _referenced_assets():
    """Same-origin assets index.html pulls in (href=/src=), excluding absolute URLs."""
    html = open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8").read()
    refs = re.findall(r'(?:href|src)\s*=\s*"([^"]+)"', html)
    return [r for r in refs
            if not r.startswith(("http://", "https://", "//", "#", "data:", "mailto:"))]


def test_index_references_at_least_the_expected_bundle():
    assert set(_referenced_assets()) >= {"soundguard.css", "soundguard-data.js",
                                         "soundguard-app.js"}, _referenced_assets()


CTYPE_OK = {
    ".css": ("text/css",),
    ".js": ("text/javascript", "application/javascript", "application/x-javascript"),
    ".html": ("text/html",),
    ".json": ("application/json",),
    ".ico": ("image/",),
    ".png": ("image/png",),
    ".svg": ("image/svg",),
}


@pytest.mark.parametrize("asset", sorted(set(_referenced_assets())))
def test_every_referenced_asset_is_served_with_a_sane_content_type(client, asset):
    r = client.get("/" + asset.lstrip("/"))
    assert r.status_code == 200, f"{asset} is referenced by index.html but returns {r.status_code}"
    assert len(r.content) > 0, f"{asset} is served empty"
    ext = os.path.splitext(asset)[1].lower()
    ctype = r.headers.get("content-type", "")
    if ext in CTYPE_OK:
        assert any(ctype.startswith(c) for c in CTYPE_OK[ext]), \
            f"{asset} served as {ctype!r}; browsers require one of {CTYPE_OK[ext]}"


def test_a_missing_static_path_is_a_clean_404_not_a_crash(client):
    r = client.get("/definitely-not-here.js")
    assert r.status_code == 404, r.status_code
    assert r.status_code != 500


# ---------------------------------------------------------------- 5. self-report

def test_health_reports_a_loaded_model_and_no_missing_clips(client):
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    h = json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert set(h) == {"model_available", "note", "missing_demo_clips"}
    assert h["model_available"] is True, f"the deployment cannot serve: note={h['note']!r}"
    assert h["missing_demo_clips"] == [], f"demo clips missing: {h['missing_demo_clips']}"
    assert h["note"] is None, f"model loaded but reported a note: {h['note']!r}"


def test_health_agrees_with_the_filesystem(client):
    """The health endpoint must be a measurement, not an assertion of its own."""
    h = client.get("/api/health").json()
    on_disk = [s for s in DEMO_IDS
               if not os.path.exists(os.path.join(ROOT, "samples", f"{s}.wav"))]
    assert h["missing_demo_clips"] == on_disk


def test_health_and_predictor_singleton_agree(client, predictor):
    assert client.get("/api/health").json()["model_available"] is predictor.available


def test_every_deployment_module_imports_cleanly():
    """A stale module that cannot even be imported is a deployment defect: it will only be found
    when someone touches it. Imports only — nothing is executed beyond module scope."""
    import importlib

    broken = {}
    for name in DEPLOY_MODULES:
        if not os.path.exists(os.path.join(ROOT, f"{name}.py")):
            broken[name] = "file missing from the deployment"
            continue
        try:
            importlib.import_module(name)
        except Exception as e:
            broken[name] = f"{type(e).__name__}: {e}"
    assert not broken, f"modules that fail to import: {broken}"


def test_working_tree_modules_import_cleanly():
    """Guards against a stale module creeping back in. runtime.py was exactly that — it imported
    FeatureConfig from features.py and build_head from heads.py, neither of which exists, so it could
    not be imported at all; it was dead code superseded by predictor.py and has been removed."""
    import importlib

    broken = {}
    for f in RESEARCH_SCRIPTS:
        if not os.path.exists(os.path.join(ROOT, f)):
            continue
        name = f[:-3]
        if name in ("pipeline", "summarise_all", "summarise_paper"):
            continue                 # heavy: argparse/pandas research entry points, not imported here
        try:
            importlib.import_module(name)
        except Exception as e:
            broken[name] = f"{type(e).__name__}: {e}"
    if not broken and not any(os.path.exists(os.path.join(ROOT, f)) for f in RESEARCH_SCRIPTS):
        pytest.skip("served subset only — no working-tree research modules present")
    assert not broken, f"modules that fail to import: {broken}"


# ---------------------------------------------------------------- 6. hygiene

def _bound_names(tree):
    """{bound name -> lineno} for every import in a module."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "*":
                    continue
                out[a.asname or a.name] = node.lineno
    return out


def _used_names(tree):
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            n = node
            while isinstance(n, ast.Attribute):
                n = n.value
            if isinstance(n, ast.Name):
                used.add(n.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            used.update(re.findall(r"[A-Za-z_]\w*", node.value))   # string annotations
    return used


def test_no_source_file_imports_a_module_it_never_uses(capsys):
    """Reported, not enforced: an unused import is dead weight in the install, but failing the
    build over one would be noise. The list is printed so it can be cleaned up."""
    dead = []
    for p in _top_level_sources():
        with open(p, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=p)
        used = _used_names(tree)
        for name, line in _bound_names(tree).items():
            if name not in used:
                dead.append(f"{os.path.basename(p)}:{line} imports {name!r} unused")
    with capsys.disabled():
        print("\n[unused imports] " + ("none" if not dead else "\n  " + "\n  ".join(dead)))


MOCK_SYMBOLS = ("run_model", "DASHBOARD_MOCK", "UNIFORM", "SAMPLES")


def test_no_leftover_mock_symbols_are_defined_in_the_python_source():
    """The incident this project is written around was a silent mock fallback. No mock symbol may
    survive as a *definition* anywhere (mentions in prose/comments are the postmortem, and fine)."""
    found = []
    for p in _top_level_sources():
        with open(p, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=p)
        for node in ast.walk(tree):
            name = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in MOCK_SYMBOLS:
                        name = t.id
            if name in MOCK_SYMBOLS:
                found.append(f"{os.path.basename(p)}:{node.lineno} defines {name}")
    assert not found, f"leftover mock definitions: {found}"


def test_no_leftover_mock_symbols_are_defined_in_the_frontend():
    found = []
    for name in sorted(os.listdir(STATIC_DIR)):
        if not name.endswith(".js"):
            continue
        src = open(os.path.join(STATIC_DIR, name), encoding="utf-8").read()
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"(?m)^\s*//.*$", "", src)
        for sym in MOCK_SYMBOLS:
            pat = rf"(?:\b(?:const|let|var|function)\s+{sym}\b|\bSG\.{sym}\b|\b{sym}\s*=[^=])"
            if re.search(pat, src):
                found.append(f"static/{name} defines {sym}")
    assert not found, f"leftover mock definitions in the frontend: {found}"


def test_the_frontend_hard_codes_no_predictions_or_metrics():
    """soundguard-data.js is presentation constants only — the incident was fabricated numbers that
    looked real. Guard the property, not the wording: no probability/metric-shaped literals."""
    src = open(os.path.join(STATIC_DIR, "soundguard-data.js"), encoding="utf-8").read()
    for banned in ("probs", "ece", "ECE", "brier", "accuracy", "reliability"):
        assert not re.search(rf"\b{banned}\s*[:=]", src), \
            f"soundguard-data.js appears to hard-code {banned!r} instead of reading the backend"


def test_no_source_file_reads_paths_relative_to_the_cwd():
    """app.py resolves artifacts off BASE_DIR precisely so `uvicorn app:app` works from any cwd.
    A bare relative open() would reintroduce the 'works on my machine' failure."""
    import app as app_module

    assert os.path.isabs(app_module.BASE_DIR)
    assert os.path.isabs(app_module.SAMPLES_DIR)
    assert app_module.SAMPLES_DIR.startswith(app_module.BASE_DIR)


def test_the_app_can_serve_from_an_unrelated_working_directory(client, tmp_path, monkeypatch):
    """The strongest form of 'can it be started from scratch': chdir somewhere empty, and every
    artifact-backed endpoint must still resolve its files."""
    monkeypatch.chdir(tmp_path)
    assert client.get("/api/health").json()["missing_demo_clips"] == []
    assert client.get("/api/dashboard").status_code == 200
    assert client.get("/").status_code == 200


def test_an_import_of_the_app_does_not_require_a_gpu():
    """CPU-forced deployment: the predictor must have loaded on CPU without a CUDA device."""
    import predictor as P

    p = P.get_predictor()
    assert p.available, f"predictor unavailable: {p.note}"
    assert str(p.extractor.device) == "cpu", f"extractor bound to {p.extractor.device}"


def test_the_deployment_carries_no_import_of_a_dev_only_module():
    """pytest/httpx must never be imported by shipped source: a production install that omits the
    test extras would then fail at import time."""
    imports = _third_party_imports([p for p in _top_level_sources()
                                    if os.path.basename(p)[:-3] in DEPLOY_MODULES])
    for dev in ("pytest", "httpx", "IPython", "matplotlib"):
        assert dev not in imports, f"shipped source imports {dev}: {imports[dev]}"
