# SoundGuard — system-level contract for test authors

This layer tests the whole stack together. Do NOT duplicate what already exists:

- `tests_fe/` (175 tests): browser behaviour, rendering, XSS, a11y, responsive, races.
- `tests_be/` (123 tests) + `test_app.py` (60): endpoint contract, validation, malformed input, adversarial audio, concurrency.
- `tests_ai/` (106 tests): metric formulas, train/serve preprocessing parity, model quality, reproducibility.

**What this layer adds**: deployment integrity, end-to-end flows through the real stack, cross-layer
consistency of the numbers, performance, and graceful degradation.

## The system
- `app.py` — FastAPI: `POST /api/predict` (Literal sample + corr sliders 0..1), `POST /api/predict/upload`
  (multipart, ≤10 MB), `GET /api/dashboard` (serves `results/dashboard.json`), `GET /api/health`,
  static mount at `/` serving `static/index.html`.
- `predictor.py` → `features.py` (resample→16k, pad/crop→4 s, log-mel 64/1024/512, 1→3ch, min-max, ImageNet norm,
  frozen ResNet-18 → [1,512]) → `heads.py` (MLP head) → softmax; `corruption.py` quality report; `rejection.py` verdict.
- Artifacts: `checkpoints/{mlp_prod.pth,feature_config.json}`, `samples/*.wav` (7 demo clips),
  `results/dashboard.json`, `rerun_results/mlp_fold*.pt`, `static/*`, `requirements.txt`.

## The numbers that must agree everywhere (honest, out-of-fold)
accuracy 0.700, ECE 0.186, NLL 1.466, Brier 0.476; corrupted accuracy 0.527, ECE 0.323.
Chain: `rerun_results` logits → `metrics.py` → `results/dashboard.json` → `GET /api/dashboard` → the report.
A disagreement anywhere in that chain is a HIGH finding.

## Fixtures (conftest, do not edit it or any source)
`client` (TestClient, raise_server_exceptions=False so a 500 is an assertable Response), `predictor`,
`model_ok`, `dashboard`, `rerun_logits`, `ROOT`, `DEMO_IDS`, `CLASSES`, and helpers `enc`, `tone`, `upload`.
Model is CPU-forced.

## Correctness bar
Every response: 200 with strictly-valid JSON (no NaN/Infinity tokens) and finite numbers, or a clean 4xx
with `{detail}`. A 500 is a defect. Validate JSON with
`json.loads(r.text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))`.

## Author rules
- Write ONLY your module. Assert the CORRECT behaviour. Verify against the real source before calling
  something a bug. On a GENUINE defect keep an `@pytest.mark.xfail(reason=..., strict=False)` test AND
  report it in `findings` with a precise repro and the measured number.
- Performance work must REPORT measured numbers (latency, memory, throughput) in `notes`/`findings`, not just
  pass a threshold — those numbers drive the optimisation that follows.
- Deterministic, self-contained, no network. Keep runtime modest.
