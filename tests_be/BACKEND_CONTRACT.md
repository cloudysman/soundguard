# SoundGuard backend — contract for test authors

In-process tests via `client` (TestClient). Import helpers from conftest: `enc, tone, noise, silence, chirp, upload`.
Model is forced to CPU. If `model_ok` is False, skip model-dependent assertions.

## Endpoints (app.py)
- `POST /api/predict`  JSON `{sample: Literal["dog","siren","jack","horn","kids","gun","ac"], corr:{noise,volume,clip}}`
  - `sample` unknown/empty/traversal -> 422 (Literal). corr each `Field(ge=0,le=1)` -> 422 out of range.
  - runs the real model on samples/<id>.wav; missing clip -> 503.
- `POST /api/predict/upload`  multipart `file`, optional form `noise/volume/clip` (each 0..1).
  - not audio (by content-type prefix `audio/` OR extension in {.wav .mp3 .flac .ogg .m4a .aac}) -> 415
  - empty -> 400 ; > 10 MB -> 413 ; undecodable bytes -> 400 ; slider out of [0,1] -> 422
  - decodes via `soundfile` (libsndfile 1.2.2: WAV/FLAC/OGG/AIFF/MP3/... but NOT webm/m4a/aac).
- `GET /api/dashboard` -> results/dashboard.json or 503 if absent. GET-only (POST -> 405 via static mount).
- `GET /api/health` -> `{model_available, note, missing_demo_clips}`.
- `GET /` -> index.html (static mount at "/").

## Prediction response dict (both predict endpoints)
`probs[10]` (sum 1), `order` [[idx,prob]]×10 desc, `conf`, `entropy`, `snrDb`, `levelDb`, `clipPct`,
`status ∈ {reliable,uncertain,reject}`, `cause ∈ {clean,uncert,noise,quiet,clip}`. All must be FINITE numbers
and the JSON body must be strictly valid (no `NaN`/`Infinity` tokens — those break a browser's JSON.parse).

## Pipeline internals (predictor.py / features.py / corruption.py / rejection.py)
- upload -> app._decode (soundfile) -> mono float32 -> predictor.apply_slider_corruption -> predict_wav.
- features: resample to 16 kHz -> pad/crop to 4 s -> log-mel(64) -> 3ch -> minmax[0,1] -> ImageNet norm -> frozen ResNet-18 -> [1,512] -> head -> softmax.
- corruption.quality_report: snrDb (energy-contrast proxy), clipPct, levelDb (dBFS).
- rejection.decide thresholds (recalibrated to real audio): snr_poor 2 / snr_weak 4 ; level_poor -50 / level_weak -40 ; clip_poor 30 / clip_weak 12 ; conf_reject .45 / conf_uncertain .70.

## Things worth probing (do NOT assume these are handled)
- Adversarial audio: NaN / +-Inf samples, all-silence, DC offset, full-scale/clipping, 1 sample, sub-frame length,
  extreme sample rates (1, 8k, 96k, 192k), 8/16/24-bit & float subtypes, stereo/multichannel, denormals, huge amplitude.
  For each: response must be 200 with a strictly-valid JSON body whose numbers are FINITE, OR a clean 4xx — never a 500
  and never a `NaN`/`Infinity` in the JSON.
- Formats: real WAV/FLAC/OGG should classify (200). webm/m4a/aac currently -> 400 (record gap).
- Concurrency/thread-safety: many simultaneous uploads through the shared model (TestClient + threads).
- Determinism: same bytes -> identical response. Memory/latency stability over many requests.
- Validation & security: path-traversal sample, oversize, wrong content-type vs extension, unicode filename, header/JSON edge cases.

## Author rules
- Assert the CORRECT behaviour. When you find a GENUINE defect, keep a test that documents it with
  `@pytest.mark.xfail(reason=..., strict=False)` AND report it in `findings` with a precise repro.
- Deterministic, self-contained, no network. Run: `python -m pytest tests_be/<yourfile> -q` until green (pass + intended xfail).
