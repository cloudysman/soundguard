# SoundGuard AI / model — contract for test authors

Tests the **science**, not HTTP. Import fixtures from conftest: `predictor, model_ok, extractor, config,
rerun_logits, rerun_best, us8k`, and `CLASSES`. Model is CPU-forced. If `model_ok` is False, skip
model-dependent tests; if `us8k` is None, skip audio-behaviour tests.

## What is under test
- `features.py` FeatureExtractor.wav_to_features(wav, sr) -> torch [1, 512]: resample→16k, pad/crop→4s,
  log-mel(64, n_fft 1024, hop 512), 1→3ch, per-sample min-max[0,1], ImageNet norm, frozen ResNet-18.
- `predictor.py` Predictor.predict_wav(wav, sr) -> dict (probs/order/conf/entropy/snr/level/clip/status/cause);
  apply_slider_corruption(wav, sr, noise, volume, clip).
- `metrics.py` softmax, accuracy, nll, brier, ece(n_bins=15), reliability_curve, risk_coverage_curve, per_class_accuracy.
- checkpoints/feature_config.json (the preprocessing spec) and checkpoints/mlp_prod.pth (the trained head).

## Ground truth / references (READ these before asserting)
- `sg/backbone_resnet18.py`: `load()` -> bundle, `embed(bundle, waves_np, batch)` -> features. This is the
  TRAINING-side feature pipeline. **Parity target**: features.py must reproduce it numerically on identical audio.
- `sg/reliability.py`, `sg/stats.py`: reference metric / bootstrap-CI implementations to cross-check metrics.py.
- `rerun_results/mlp_fold*.pt`: dict with best_logits/last_logits/labels/files/fold per fold — real logits.
- `results/dashboard.json`: the published numbers computed by eval.py from the LAST-epoch logits
  (accuracy 0.700, ECE 0.186, NLL 1.466, Brier 0.476; corrupted acc 0.527, ECE 0.323). Reproduce these from
  `rerun_logits` through metrics.py to prove the eval pipeline is correct.
- Local UrbanSound8K at C:\Users\trong\Downloads\us8k\UrbanSound8K (via the `us8k` fixture).

## Correctness bars worth probing
- **Metric formulas** (analytical): softmax rows sum to 1 & match a reference; NLL/Brier/accuracy on tiny
  hand-computable inputs; ECE≈0 for a perfectly-calibrated construction and ≈|conf-acc| for a known miscalibration;
  ECE/reliability/risk-coverage/ per-class on inputs where the answer is known by hand; numerical stability with
  extreme logits (±1e4) — no overflow/NaN; risk at coverage=1 equals 1-accuracy; risk-coverage is non-decreasing
  in a well-ordered case; all metrics finite and in their valid ranges.
- **Reproduce dashboard.json** from rerun_logits via metrics.py within tight tolerance (accuracy/ECE/NLL/Brier).
- **Train/serve parity**: features.py vs sg/backbone_resnet18.embed on the SAME waveforms -> feature vectors
  close (report max abs diff; they should match to a small tolerance). Also assert the serving mel/normalisation
  match feature_config (sr, n_fft, hop, n_mels), 3 channels, ImageNet mean/std applied, output shape [1,512].
- **Model behaviour** on a held-out us8k sample: top-1 accuracy in a sane band (the model is ~70%); the 7 bundled
  demo clips classify to their own class; confidence is higher on correct than wrong predictions on average
  (error-detection AUROC > 0.5 — the project's thesis); per-class accuracy all > chance; determinism (same wav ->
  identical logits); trivial invariance (a tiny gain change or a few samples of pad shift does not flip a
  confident prediction); accuracy degrades monotonically-ish as slider/corruption severity rises.
- **Reproducibility**: predict_wav deterministic across repeated calls and across a fresh Predictor; loading the
  checkpoint yields real trained weights (head is far better than chance on us8k; BatchNorm running stats are not
  the untrained defaults).

## Author rules
- Assert the CORRECT behaviour. On a GENUINE defect keep an xfail(strict=False) test + a `findings` entry with
  a precise repro (and, for numerical parity/quality, the measured number and the tolerance). Verify against the
  reference source before calling something a bug — a metric "disagreement" may be a definition choice, not a bug.
- Deterministic, self-contained, no network. Keep audio samples small (n_per_class 1-3) so runtime stays modest.
- Run: `python -m pytest tests_ai/<yourfile> -q` until green (pass + intended xfail). Report real numbers.
