from pydantic import BaseModel


class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class Transcription(BaseModel):
    id: int
    filename: str | None
    language: str | None
    duration: float | None
    text: str
    segments: list[Segment]
    created_at: str


class Health(BaseModel):
    status: str
    model: str
    diarization: bool
