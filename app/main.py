import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import database
from app.config import settings
from app.schemas import Health, Transcription
from app.transcriber import Transcriber

APP_DIR = Path(__file__).resolve().parent

ALLOWED_CONTENT_TYPES = ("audio/", "video/webm", "video/mp4", "video/ogg", "application/octet-stream")
CHUNK_SIZE = 1024 * 1024

# WhisperX models are not safe to call from multiple threads at once
_model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    app.state.transcriber = Transcriber(
        model_name=settings.model_name,
        device=settings.device,
        compute_type=settings.compute_type,
        diarize=settings.diarize,
        hf_token=settings.hf_token,
    )
    yield


app = FastAPI(title="Voice Transcription API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health", response_model=Health)
async def health():
    return Health(status="ok", model=settings.model_name, diarization=settings.diarize)


def _run_transcription(transcriber, path, min_speakers=None, max_speakers=None):
    with _model_lock:
        return transcriber.transcribe(
            path, min_speakers=min_speakers, max_speakers=max_speakers
        )


@app.post("/api/transcribe", response_model=Transcription, status_code=201)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None, ge=1, le=20),
    min_speakers: int | None = Form(None, ge=1, le=20),
    max_speakers: int | None = Form(None, ge=1, le=20),
):
    # num_speakers is shorthand for an exact count
    if num_speakers is not None:
        min_speakers = max_speakers = num_speakers
    if min_speakers and max_speakers and min_speakers > max_speakers:
        raise HTTPException(400, "min_speakers cannot be greater than max_speakers")

    content_type = (file.content_type or "").lower()
    if not content_type.startswith(ALLOWED_CONTENT_TYPES):
        raise HTTPException(400, f"Unsupported file type: {content_type or 'unknown'}")

    suffix = Path(file.filename or "").suffix or ".webm"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    size = 0

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with tmp:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
                tmp.write(chunk)

        if size == 0:
            raise HTTPException(400, "Uploaded file is empty")

        try:
            result = await run_in_threadpool(
                _run_transcription,
                request.app.state.transcriber,
                tmp.name,
                min_speakers,
                max_speakers,
            )
        except Exception as exc:
            raise HTTPException(500, f"Transcription failed: {exc}") from exc
    finally:
        os.unlink(tmp.name)

    return database.save_transcription(
        filename=file.filename,
        language=result["language"],
        duration=result["duration"],
        text=result["text"],
        segments=result["segments"],
    )


@app.get("/api/transcriptions", response_model=list[Transcription])
async def list_transcriptions(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return database.list_transcriptions(limit=limit, offset=offset)


@app.get("/api/transcriptions/{transcription_id}", response_model=Transcription)
async def get_transcription(transcription_id: int):
    record = database.get_transcription(transcription_id)
    if record is None:
        raise HTTPException(404, "Transcription not found")
    return record


@app.delete("/api/transcriptions/{transcription_id}", status_code=204)
async def delete_transcription(transcription_id: int):
    if not database.delete_transcription(transcription_id):
        raise HTTPException(404, "Transcription not found")
