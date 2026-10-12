import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _bool(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model_name: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "small"))
    device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8"))
    diarize: bool = field(default_factory=lambda: _bool(os.getenv("ENABLE_DIARIZATION")))
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("DB_PATH", BASE_DIR / "data" / "transcriptions.db"))
    )
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "25")))


settings = Settings()
