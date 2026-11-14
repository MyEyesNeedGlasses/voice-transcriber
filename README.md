# Voice Transcriber

Record your voice in the browser and transcribe it with [WhisperX](https://github.com/m-bain/whisperX).
The app has a FastAPI backend, and transcripts are stored in SQLite.

## Quick start with Docker (recommended)

This is the easiest way to run the app on any machine (Windows, macOS or Linux). You don't need to
install Python or FFmpeg.

### 1. Install Docker

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.

### 2. Get the code

```bash
git clone https://github.com/VinayAnupNair/voice-transcriber.git
cd voice-transcriber
```

### 3. Create your `.env`

```powershell
copy .env.example .env      # Windows (PowerShell)
```
```bash
cp .env.example .env        # macOS / Linux
```

The defaults work as they are. To label who is speaking, also set up speaker diarization (step 4).

### 4. (Optional) Speaker diarization: Hugging Face token

Diarization uses the pyannote models from Hugging Face. They are free, but you have to accept their
terms first.

1. Create a free account at https://huggingface.co.
2. While logged in, open
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1),
   fill in the short form and accept the conditions. Any company/website values are fine, for
   example "Personal" and your GitHub profile URL.
3. Create a **Read** token at https://huggingface.co/settings/tokens.
4. Put it in `.env`:
   ```
   ENABLE_DIARIZATION=true
   HF_TOKEN=hf_your_token_here
   ```

Every person needs their own token, and their account must accept the terms. A `403 ... gated repo`
error means the account hasn't been granted access yet.

### 5. Start it

```bash
docker compose up --build
```

The first build takes several minutes, and the image is about 3.5 GB. The first start also
downloads the Whisper model, plus the pyannote models when diarization is on. When you see
`Uvicorn running on http://0.0.0.0:8000`, open **http://localhost:8000**.

### Everyday use

| Task | Command |
|------|---------|
| Start | `docker compose up` |
| Start in the background | `docker compose up -d` |
| Stop | `Ctrl+C`, or `docker compose down` |
| View logs | `docker compose logs -f` |
| After pulling new code | `docker compose up --build` |
| After editing `.env` | `docker compose up` (no rebuild needed) |
| Transcribe a file from the CLI | `docker compose run --rm transcriber python scripts/transcribe_file.py samples/Test.mp3 --diarize --speakers 2` |

Notes:
- **Persistent data:** transcripts are saved to `./data` on your machine. Downloaded models are kept in
  the `model-cache` Docker volume, so later starts are fast. `docker compose down -v` deletes that cache.
- **Samples:** put audio files in `./samples` to use them from the CLI. That folder is mounted read-only.
- **Your token stays local:** `.env` is never copied into the image or committed to Git. It's
  passed to the container when it starts.
- **Microphone:** browsers only allow recording on `localhost` or HTTPS. If you open the app from
  another device using an IP address, recording is blocked, but file upload still works.
- **Hardware:** the image runs on CPU only. Transcription is slower than with a GPU, and the `small` model
  is a good balance. Change `WHISPER_MODEL` in `.env` to trade speed for accuracy.

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
Dockerfile         container image (CPU-only PyTorch + FFmpeg)
docker-compose.yml runs the app with data and model-cache volumes
samples/           local test audio (gitignored)
data/              SQLite database (gitignored, created at runtime)
```

## Running without Docker (Windows / PowerShell)

1. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   Audio is decoded with PyAV, which bundles FFmpeg, so you don't need a separate ffmpeg install.
   (Speaker diarization through pyannote may still want system FFmpeg.)
2. Copy `.env.example` to `.env` and adjust it (see step 4 above for diarization).
3. Run:
   ```powershell
   uvicorn app.main:app --reload
   ```

Open http://127.0.0.1:8000, click **Start recording**, then **Stop recording**, then **Transcribe**.
The first start downloads the Whisper model, so it takes a while.

## API

| Method | Path                         | Description                              |
|--------|------------------------------|------------------------------------------|
| GET    | `/api/health`                | Health check, model info                 |
| POST   | `/api/transcribe`            | Multipart `file` field (optional `num_speakers`, or `min_speakers`/`max_speakers`), returns the transcript |
| GET    | `/api/transcriptions`        | List saved transcripts (`limit`, `offset`) |
| GET    | `/api/transcriptions/{id}`   | Get one transcript                       |
| DELETE | `/api/transcriptions/{id}`   | Delete a transcript                      |

Interactive docs are at http://127.0.0.1:8000/docs.

```powershell
curl.exe -F "file=@samples/Test.mp3" http://127.0.0.1:8000/api/transcribe
```

## Speaker diarization

Set `ENABLE_DIARIZATION=true` and `HF_TOKEN` in `.env`, and accept the terms for
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
See step 4 of the Docker quick start.

When diarization is on, the transcript is aligned to get word timestamps. It is then split into
speaker turns, and each segment's `speaker` is set (`SPEAKER_00`, `SPEAKER_01`, ...). The web UI
shows a **Speakers** field. Leave it blank to detect the number automatically.

```powershell
curl.exe -F "file=@samples/Test.mp3" -F "num_speakers=2" http://127.0.0.1:8000/api/transcribe
```

## CLI

```powershell
python scripts/transcribe_file.py samples/Test.mp3
python scripts/transcribe_file.py samples/Test.mp3 --diarize --speakers 2
```
