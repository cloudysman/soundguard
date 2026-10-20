"""L4 — train/serve preprocessing parity.

THE key AI test: the serving feature pipeline (features.py FeatureExtractor) must numerically
reproduce the training-side pipeline (sg/backbone_resnet18.load()/embed()) on identical waveforms.
If they diverge, the deployed model silently underperforms (train/serve skew).

Strategy for a FAIR, controlled comparison
------------------------------------------
`embed()` hard-wires a 32 kHz -> 16 kHz resample (SR_IN=32000) and does NOT pad/crop; features.py
resamples native_sr -> 16 kHz and then pads/crops to exactly 4 s * 16 kHz = 64000 samples. To feed
BOTH pipelines the *identical* waveform and land on the *identical* 16 kHz length (so the global-avg
pool sees the same number of frames), every input is shaped to exactly 128000 samples (4 s at the
nominal 32 kHz that embed assumes). Then:
    - embed(w[None])            resamples 128000 -> 64000, no crop needed
    - extractor.wav_to_features(w, sr=32000) resamples 128000 -> 64000, pad/crop is a no-op
Both therefore exercise the resample branch (sr != 16000) on identical audio and must produce the
same 512-d embedding. Real UrbanSound8K clips supply complex, realistic content.

Everything is CPU-forced (conftest sets CUDA_VISIBLE_DEVICES=-1), so backbone_resnet18.DEVICE == cpu
and the two backbones run the identical kernels; parity is expected to ~1e-4.
"""
import numpy as np
import pytest

TARGET_LEN = 128000          # 4 s at the nominal 32 kHz embed() assumes -> 64000 samples at 16 kHz
NOMINAL_SR = 32000           # feed features.py the same rate embed() uses, so both resample 32k->16k
SERVE_SR = 16000
SERVE_LEN = 64000            # 4 s * 16 kHz

PARITY_TOL = 1e-3            # max abs diff on the 512-d vector; CPU-vs-CPU identical ops -> ~1e-5


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _fix_len(w, n=TARGET_LEN):
    """Crop or zero-pad a 1-D float32 array to exactly n samples."""
    w = np.asarray(w, dtype=np.float32).ravel()
    if w.shape[0] >= n:
        return np.ascontiguousarray(w[:n])
    return np.concatenate([w, np.zeros(n - w.shape[0], dtype=np.float32)])


def _tone(freq=440.0, sr=NOMINAL_SR, n=TARGET_LEN):
    t = np.arange(n, dtype=np.float64) / sr
    return (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seed=0, n=TARGET_LEN):
    return (0.3 * np.random.default_rng(seed).standard_normal(n)).astype(np.float32)


def _chirp(f0=200.0, f1=6000.0, sr=NOMINAL_SR, n=TARGET_LEN):
    t = np.arange(n, dtype=np.float64) / sr
    dur = n / sr
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
    return (0.6 * np.sin(phase)).astype(np.float32)


@pytest.fixture(scope="module")
def bundle():
    """Training-side reference pipeline (sg/backbone_resnet18)."""
    try:
        from sg import backbone_resnet18 as bb
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"sg.backbone_resnet18 unavailable: {type(e).__name__}: {e}")
    import torch
    # Parity is only meaningful CPU-vs-CPU; skip if the reference is stuck on cuda.
    if str(bb.DEVICE) != "cpu":
        pytest.skip(f"backbone_resnet18.DEVICE={bb.DEVICE}, expected cpu for a fair CPU parity check")
    return bb, bb.load(verbose=False)


def _embed_ref(bundle, w):
    """Reference 512-d vector as float64 numpy."""
    bb, handle = bundle
    ref = bb.embed(handle, _fix_len(w)[None, :].astype(np.float32))   # torch [1,512] cpu
    return ref.detach().cpu().numpy().astype(np.float64).ravel()


def _serve_vec(extractor, w):
    """Serving 512-d vector as float64 numpy (features.py)."""
    feats = extractor.wav_to_features(_fix_len(w), NOMINAL_SR)        # torch [1,512]
    return feats.detach().cpu().numpy().astype(np.float64).ravel()


# --------------------------------------------------------------------------------------------------
# 1) Serving preprocessing spec matches feature_config.json
# --------------------------------------------------------------------------------------------------
def test_config_matches_spec(config):
    assert config["sample_rate"] == SERVE_SR
    assert config["n_fft"] == 1024
    assert config["hop_length"] == 512
    assert config["n_mels"] == 64
    assert config["max_length_s"] == 4
    assert config["in_features"] == 512
    assert config["num_classes"] == 10
    assert config["normalize"] == "minmax01_then_imagenet"


def test_extractor_params(extractor):
    import torch
    assert extractor.sr == SERVE_SR
    assert extractor.num_samples == SERVE_LEN                  # 4 s * 16 kHz pad/crop target
    # ImageNet normalisation constants applied on the serving side
    assert torch.allclose(extractor.mean.flatten(),
                          torch.tensor([0.485, 0.456, 0.406]), atol=1e-6)
    assert torch.allclose(extractor.std.flatten(),
                          torch.tensor([0.229, 0.224, 0.225]), atol=1e-6)
    # mel front-end configured 16k / 1024 / 512 / 64
    mel = extractor.mel
    for attr, want in [("sample_rate", SERVE_SR), ("n_fft", 1024),
                       ("hop_length", 512), ("n_mels", 64)]:
        if hasattr(mel, attr):
            assert getattr(mel, attr) == want, f"mel.{attr}={getattr(mel, attr)} != {want}"


def test_pad_crop_to_4s(extractor):
    import torch
    # shorter than 4 s -> zero-padded up to exactly 64000
    short = torch.zeros(1, 30000)
    out = extractor._pad_or_crop(short)
    assert out.shape[1] == SERVE_LEN
    assert torch.count_nonzero(out[:, 30000:]) == 0           # tail is zeros
    # longer than 4 s -> cropped to exactly 64000
    long = torch.randn(1, 100000)
    out2 = extractor._pad_or_crop(long)
    assert out2.shape[1] == SERVE_LEN
    assert torch.equal(out2, long[:, :SERVE_LEN])


def test_output_shape_and_channels(extractor):
    """Final serving embedding is [1, 512], finite; 1->3 channel replication feeds a 3-ch backbone."""
    vec = _serve_vec(extractor, _tone())
    assert vec.shape == (512,)
    assert np.all(np.isfinite(vec))
    feats = extractor.wav_to_features(_tone(), NOMINAL_SR)
    assert tuple(feats.shape) == (1, 512)


def test_minmax_range_before_imagenet(extractor):
    """The per-recording min-max maps the (3-channel) log-mel into [0,1] (+1e-6 denominator guard)
    BEFORE ImageNet normalisation. Reproduce the pre-backbone tensor and check the [0,1] window."""
    import torch
    w = _fix_len(_chirp())
    wav = torch.tensor(w).unsqueeze(0)
    wav = extractor._torchaudio.transforms.Resample(NOMINAL_SR, extractor.sr)(wav)
    wav = extractor._pad_or_crop(wav)
    log_mel = extractor.amp_to_db(extractor.mel(wav))
    feats = log_mel.expand(3, -1, -1).clone()
    fmin, fmax = feats.min(), feats.max()
    normed = (feats - fmin) / (fmax - fmin + 1e-6)
    assert float(normed.min()) >= 0.0
    assert float(normed.max()) <= 1.0
    # with a non-degenerate spectrum the window is nearly full [0,1]
    assert float(normed.max()) > 0.99
    assert float(normed.min()) < 0.01


# --------------------------------------------------------------------------------------------------
# 2) Numerical parity: features.py vs sg/backbone_resnet18.embed on identical waveforms
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,wav_fn", [
    ("tone", _tone),
    ("noise", _noise),
    ("chirp", _chirp),
])
def test_parity_synthetic(name, wav_fn, extractor, bundle):
    w = wav_fn()
    ref = _embed_ref(bundle, w)
    serv = _serve_vec(extractor, w)
    max_abs = float(np.max(np.abs(ref - serv)))
    print(f"[parity/{name}] max|ref-serve|={max_abs:.3e} (tol {PARITY_TOL:.1e})")
    assert max_abs < PARITY_TOL, (
        f"train/serve feature skew on '{name}': max|ref-serve|={max_abs:.3e} >= tol {PARITY_TOL:.1e}")


def test_parity_us8k(extractor, bundle, us8k):
    if us8k is None:
        pytest.skip("UrbanSound8K dataset not present")
    clips = us8k(n_per_class=1, seed=0)
    if not clips:
        pytest.skip("no us8k clips loaded")
    worst = 0.0
    worst_cls = None
    for wav, sr, idx, cls in clips:
        ref = _embed_ref(bundle, wav)
        serv = _serve_vec(extractor, wav)
        m = float(np.max(np.abs(ref - serv)))
        if m > worst:
            worst, worst_cls = m, cls
    print(f"[parity/us8k] worst max|ref-serve|={worst:.3e} (class {worst_cls}, tol {PARITY_TOL:.1e})")
    assert worst < PARITY_TOL, (
        f"train/serve feature skew on real us8k clips: worst max|ref-serve|={worst:.3e} "
        f"(class {worst_cls}) >= tol {PARITY_TOL:.1e}")


# --------------------------------------------------------------------------------------------------
# 3) End-to-end: same head over reference vs serving features predicts the same class
# --------------------------------------------------------------------------------------------------
def test_head_agreement_us8k(predictor, model_ok, extractor, bundle, us8k):
    if not model_ok:
        pytest.skip(f"model unavailable: {getattr(predictor, 'note', None)}")
    if us8k is None:
        pytest.skip("UrbanSound8K dataset not present")
    clips = us8k(n_per_class=1, seed=0)
    if not clips:
        pytest.skip("no us8k clips loaded")
    torch = predictor._torch
    head = predictor.head
    disagree = []
    for wav, sr, idx, cls in clips:
        ref = _embed_ref(bundle, wav)
        serv = _serve_vec(extractor, wav)
        with torch.no_grad():
            a = int(torch.as_tensor(head(torch.tensor(ref, dtype=torch.float32)
                                         .unsqueeze(0))).argmax(dim=1).item())
            b = int(torch.as_tensor(head(torch.tensor(serv, dtype=torch.float32)
                                         .unsqueeze(0))).argmax(dim=1).item())
        if a != b:
            disagree.append((cls, a, b))
    assert not disagree, f"head predicts different classes on ref vs serving features: {disagree}"
