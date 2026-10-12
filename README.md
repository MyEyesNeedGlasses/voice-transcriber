# Voice Transcriber

Record your voice in the browser and transcribe it with [WhisperX](https://github.com/m-bain/whisperX).
The app has a FastAPI backend, and transcripts are stored in SQLite.

## Project layout

```
app/
  main.py          FastAPI app and routes
  config.py        settings loaded from .env
  transcriber.py   WhisperX wrapper (optional speaker diarization)
  database.py      SQLite storage
  schemas.py       API response models
  templates/       index.html (recording UI)
  static/          app.js, style.css
scripts/
  transcribe_file.py   command-line transcription of a local file
samples/           local test audio (gitignored)
data/              SQLite database (gitignored, created at runtime)
```

## Setup (Windows / PowerShell)

1. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Audio is decoded with PyAV, which bundles FFmpeg, so you don't need a separate ffmpeg install.
   (Speaker diarization through pyannote may still want system FFmpeg.)
2. Copy `.env.example` to `.env` and adjust it. `HF_TOKEN` is only needed if `ENABLE_DIARIZATION=true`.

## Run

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000, click **Start recording**, then **Stop recording**, then **Transcribe**.
The first start downloads the Whisper model, so it takes a while.

## API

| Method | Path                         | Description                              |
|--------|------------------------------|------------------------------------------|
| GET    | `/api/health`                | Health check, model info                 |
| POST   | `/api/transcribe`            | Multipart `file` field, returns the transcript |
| GET    | `/api/transcriptions`        | List saved transcripts (`limit`, `offset`) |
| GET    | `/api/transcriptions/{id}`   | Get one transcript                       |
| DELETE | `/api/transcriptions/{id}`   | Delete a transcript                      |

Interactive docs are at http://127.0.0.1:8000/docs.

```powershell
curl.exe -F "file=@samples/Test.mp3" http://127.0.0.1:8000/api/transcribe
```

## CLI

```powershell
python scripts/transcribe_file.py samples/Test.mp3
python scripts/transcribe_file.py samples/Test.mp3 --diarize --speakers 2
```
