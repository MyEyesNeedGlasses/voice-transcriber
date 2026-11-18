FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    DB_PATH=/app/data/transcriptions.db

# FFmpeg is used by pyannote/torchcodec for audio decoding
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Use the CPU wheels by default. The GPU Compose override selects CUDA wheels.
ARG PYTORCH_VARIANT=cpu
RUN pip install --index-url https://download.pytorch.org/whl/${PYTORCH_VARIANT} \
        torch==2.8.0 \
        torchaudio==2.8.0 \
        torchvision==0.23.0

COPY requirements.txt .
# Keep Triton available in CUDA builds.
RUN pip install -r requirements.txt whisperx==3.8.6 \
    && if [ "$PYTORCH_VARIANT" = cpu ]; then pip uninstall -y triton; fi

COPY app ./app
COPY scripts ./scripts

RUN useradd --create-home appuser \
    && mkdir -p /app/data /cache \
    && chown -R appuser:appuser /app/data /cache
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
