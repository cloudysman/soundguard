"""SoundGuard backend — FastAPI.

Serves the static frontend and three endpoints, all backed by the real model:
    POST /api/predict          classify one of the seven bundled demo clips (samples/<id>.wav)
    POST /api/predict/upload   classify an uploaded or recorded clip
    GET  /api/dashboard        the aggregate evaluation results (results/dashboard.json)

There is no mock path. If a dependency, the checkpoint, or a demo clip is missing, the endpoint
fails with an explicit error rather than returning a fabricated answer, because a fabricated answer
is indistinguishable from a real one to the caller and was the cause of a real incident (appendix F
of the report).

Run:
    pip install -r requirements.txt
    uvicorn app:app --port 8000
    # open http://localhost:8000
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predictor import get_predictor

log = logging.getLogger("soundguard")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # resolve paths off the source, not the CWD
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

DEMO_IDS = ("dog", "siren", "jack", "horn", "kids", "gun", "ac")


@asynccontextmanager
async def lifespan(app: FastAPI):
    pred = get_predictor()
    if not pred.available:
        log.warning("model is NOT available at startup: %s", pred.note)
    if not os.path.isdir(SAMPLES_DIR):
        log.warning("samples/ directory is missing; the demo chips will return 503")
    yield


app = FastAPI(title="SoundGuard", lifespan=lifespan)

# The seven demo chips, mapped to the UrbanSound8K class each represents. A Literal type means
# FastAPI rejects any other value with a 422 before the handler runs, so an unvalidated string can
# never reach a file path.
SampleId = Literal["dog", "siren", "jack", "horn", "kids", "gun", "ac"]


class Corr(BaseModel):
    """The three corruption sliders. Bounded to [0, 1]; out-of-range input is a 422, not garbage."""
    noise: float = Field(0.0, ge=0.0, le=1.0)
    volume: float = Field(0.0, ge=0.0, le=1.0)
    clip: float = Field(0.0, ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    sample: SampleId = "dog"
    corr: Corr = Corr()


def _to_mono_finite(wav):
    """Downmix to mono float32 and neutralise any non-finite samples. A float-encoded (WAV FLOAT/DOUBLE)
    file can carry NaN/±Inf; NaN survives np.clip and would propagate through the model into the response,
    where json.dumps rejects it as non-JSON-compliant and the request 500s. Replacing it here, at the
    decode boundary, guarantees the model and the serialiser only ever see finite audio in [-1, 1]."""
    import numpy as np
    wav = np.asarray(wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)


def _load_sample_wav(name: str):
    """Load samples/<name>.wav -> (mono waveform, sample_rate). Raises 503 if the decoder, the file
    or its contents are unusable, so a missing clip is a loud failure and never a silent fallback."""
    try:
        import soundfile as sf
    except ImportError as e:
        # Same failure the upload path guards in _decode. Left unguarded here it escapes as a 500,
        # and a missing decoder is exactly what caused the incident in appendix F of the report.
        raise HTTPException(status_code=503,
                            detail="Audio decoder unavailable — `soundfile` is not installed.") from e

    path = os.path.join(SAMPLES_DIR, f"{name}.wav")
    if not os.path.exists(path):
        raise HTTPException(status_code=503,
                            detail=f"Demo clip '{name}' is missing from the deployment (samples/).")
    try:
        wav, sr = sf.read(path)
    except Exception as e:
        log.exception("failed to decode demo clip %s", path)
        raise HTTPException(status_code=503, detail=f"Demo clip '{name}' could not be decoded.") from e
    return _to_mono_finite(wav), sr


def _require_model():
    pred = get_predictor()
    if not pred.available:
        log.error("predictor unavailable: %s", pred.note)
        raise HTTPException(status_code=503,
                            detail="Model unavailable — run: pip install -r requirements.txt")
    return pred


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Classify a bundled demo clip with the real model."""
    pred = _require_model()
    wav, sr = _load_sample_wav(req.sample)
    wav = pred.apply_slider_corruption(wav, sr, req.corr.noise, req.corr.volume, req.corr.clip)
    return pred.predict_wav(wav, sr)


# --- File upload ------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024              # 10 MB
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _looks_like_audio(filename: str, content_type: str) -> bool:
    if content_type and content_type.startswith("audio/"):
        return True
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in AUDIO_EXTS


def _decode(data: bytes):
    """Decode uploaded bytes to (mono waveform, sample_rate). Raises 400 on bytes that are not
    valid audio, 503 if no decoder is installed."""
    try:
        import io
        import soundfile as sf
    except ImportError as e:
        raise HTTPException(status_code=503,
                            detail="Audio decoder unavailable — `soundfile` is not installed.") from e
    try:
        wav, sr = sf.read(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail="Could not decode audio (corrupt or unsupported codec). "
                                   "Recorded clips must be WAV; some browsers record WebM, which "
                                   "this decoder does not read.") from e
    return _to_mono_finite(wav), sr


@app.post("/api/predict/upload")
async def predict_upload(
    file: UploadFile = File(...),
    noise: float = Form(0.0, ge=0.0, le=1.0),
    volume: float = Form(0.0, ge=0.0, le=1.0),
    clip: float = Form(0.0, ge=0.0, le=1.0),
):
    """Classify an uploaded or recorded audio clip with the real model."""
    if not _looks_like_audio(file.filename, file.content_type):
        raise HTTPException(status_code=415, detail="Unsupported file type — please upload an audio clip.")

    # Read with an explicit ceiling so a large upload cannot be fully buffered before the size check.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    pred = _require_model()
    wav, sr = _decode(data)
    wav = pred.apply_slider_corruption(wav, sr, noise, volume, clip)
    return pred.predict_wav(wav, sr)


# --- Dashboard --------------------------------------------------------------
def _reject_non_finite(token: str):
    """json.load calls this for the non-standard NaN/Infinity tokens. They must not pass: they parse
    fine but then fail at serialisation, which would surface as a 500 instead of a clean error."""
    raise ValueError(f"non-finite token {token!r} in the evaluation results")


@app.get("/api/dashboard")
def dashboard():
    """Serve the aggregate evaluation results. There is no mock: if the file is absent, unreadable, or
    carries non-finite numbers, the endpoint fails explicitly, so the charts never render fabricated
    numbers and the response is never invalid JSON."""
    path = os.path.join(BASE_DIR, "results", "dashboard.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=503,
                            detail="No evaluation results — run eval.py to produce results/dashboard.json.")
    try:
        with open(path) as f:
            return json.load(f, parse_constant=_reject_non_finite)
    except Exception as e:
        log.exception("results/dashboard.json is unreadable: %s", path)
        raise HTTPException(status_code=503,
                            detail="Evaluation results are unreadable — regenerate "
                                   "results/dashboard.json with eval.py.") from e


@app.get("/api/health")
def health():
    """Report whether the model and the demo clips are available, for monitoring."""
    pred = get_predictor()
    missing = [s for s in DEMO_IDS if not os.path.exists(os.path.join(SAMPLES_DIR, f"{s}.wav"))]
    return {"model_available": pred.available, "note": pred.note, "missing_demo_clips": missing}


# Serve the frontend. Mounted LAST so /api/* routes take precedence.
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
