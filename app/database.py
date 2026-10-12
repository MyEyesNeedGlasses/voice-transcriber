import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import settings


@contextmanager
def get_connection():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                language TEXT,
                duration REAL,
                text TEXT NOT NULL,
                segments_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def _row_to_dict(row):
    return {
        "id": row["id"],
        "filename": row["filename"],
        "language": row["language"],
        "duration": row["duration"],
        "text": row["text"],
        "segments": json.loads(row["segments_json"]),
        "created_at": row["created_at"],
    }


def save_transcription(filename, language, duration, text, segments):
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO transcriptions
                (filename, language, duration, text, segments_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, language, duration, text, json.dumps(segments), created_at),
        )
        new_id = cursor.lastrowid

    return get_transcription(new_id)


def list_transcriptions(limit=50, offset=0):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM transcriptions ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_transcription(transcription_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?",
            (transcription_id,),
        ).fetchone()

    return _row_to_dict(row) if row else None


def delete_transcription(transcription_id):
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM transcriptions WHERE id = ?",
            (transcription_id,),
        )

    return cursor.rowcount > 0
