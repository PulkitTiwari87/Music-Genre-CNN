"""FastAPI service: POST /api/predict (audio upload) -> genre prediction."""

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .predict import load_model, predict_song
from .preprocessing import GENRES, SAMPLE_RATE, SEGMENT_SAMPLES, load_audio

log = logging.getLogger("music_genre_cnn")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "music_genre_cnn_final_v3.keras"
FRONTEND_DIST = ROOT / "frontend" / "dist"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# The model was trained on 30 s clips; analysing the first minute also bounds memory
# (a 5-minute track exhausted Render's 512 MB free tier).
MAX_AUDIO_SECONDS = 60


def self_test(model):
    """Run decode -> features -> model once, so a broken environment fails startup, not requests."""
    wav = io.BytesIO()
    sf.write(wav, np.zeros(SEGMENT_SAMPLES, np.float32), SAMPLE_RATE, format="WAV")
    wav.seek(0)
    predict_song(model, load_audio(wav))


def create_app(model_path=None):
    model_path = Path(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    state = {"model": None}
    predict_lock = threading.Lock()  # one Keras predict at a time

    @asynccontextmanager
    async def lifespan(_app):
        try:
            model = load_model(model_path)
            self_test(model)
            state["model"] = model
            log.info("Loaded model from %s", model_path)
        except Exception:
            log.exception("Could not load model from %s", model_path)
        yield

    app = FastAPI(title="Music Genre CNN", lifespan=lifespan)

    # Only needed when the UI is hosted on a different origin (e.g. Vercel + Render).
    origins = [
        o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"]
        )

    def classify(data: bytes):
        # Decode + predict under one lock: concurrent requests would otherwise stack
        # their audio buffers and activations on a small-memory host.
        with predict_lock:
            try:
                audio = load_audio(io.BytesIO(data), max_seconds=MAX_AUDIO_SECONDS)
            except Exception as exc:
                log.warning("Audio decode failed", exc_info=True)
                raise HTTPException(415, "Could not decode the file as audio.") from exc
            try:
                return predict_song(state["model"], audio)
            except ValueError as exc:  # too short
                raise HTTPException(422, str(exc)) from exc

    @app.get("/api/health")
    def health():
        if state["model"] is None:
            raise HTTPException(503, "Model is not loaded.")
        return {"status": "ok", "genres": list(GENRES)}

    @app.post("/api/predict")
    async def predict(file: Annotated[UploadFile, File()]):
        if state["model"] is None:
            raise HTTPException(503, "Model is not loaded.")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)."
            )
        if not data:
            raise HTTPException(422, "Empty file.")
        return await run_in_threadpool(classify, data)

    # Mounted last so it never shadows /api routes.
    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app


app = create_app()
