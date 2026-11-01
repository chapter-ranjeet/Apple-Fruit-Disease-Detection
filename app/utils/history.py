from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .validation import safe_filename

FIELDS = ("timestamp", "filename", "predicted_class", "confidence", "device")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            device TEXT NOT NULL,
            signature TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.commit()
    return connection


def append_prediction(path: Path, record: dict[str, object], signature: str) -> bool:
    try:
        with _connect(path) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO predictions
                (timestamp, filename, predicted_class, confidence, device, signature)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    safe_filename(str(record["filename"])),
                    str(record["predicted_class"]),
                    float(record["confidence"]),
                    str(record["device"]),
                    signature,
                ),
            )
            return cursor.rowcount == 1
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        raise ValueError("Prediction history could not be updated.") from exc


def read_history(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with _connect(path) as connection:
            rows = connection.execute(
                """
                SELECT timestamp, filename, predicted_class, confidence, device
                FROM predictions ORDER BY id
                """
            ).fetchall()
        return [
            {
                "timestamp": str(row[0]),
                "filename": str(row[1]),
                "predicted_class": str(row[2]),
                "confidence": f"{float(row[3]):.6f}",
                "device": str(row[4]),
            }
            for row in rows
        ]
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        raise ValueError("Prediction history could not be read.") from exc


def clear_history(path: Path) -> None:
    if not path.exists():
        return
    try:
        with _connect(path) as connection:
            connection.execute("DELETE FROM predictions")
            connection.commit()
    except (OSError, sqlite3.Error) as exc:
        raise OSError("Prediction history could not be cleared.") from exc
