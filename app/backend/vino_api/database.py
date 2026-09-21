"""SQLite persistence for recognition runs and calibration feedback."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class EventDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS recognitions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK (mode IN ('user', 'calib')),
                    client_host TEXT,
                    user_agent TEXT,
                    original_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    predicted_slug TEXT,
                    candidate_slug TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    total_ms REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    pipeline_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_recognitions_created_at
                    ON recognitions(created_at DESC);
                CREATE INDEX IF NOT EXISTS ix_recognitions_session_id
                    ON recognitions(session_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recognition_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    verdict TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect', 'not_in_catalog')),
                    correct_slug TEXT,
                    note TEXT,
                    FOREIGN KEY (recognition_id) REFERENCES recognitions(id) ON DELETE CASCADE
                );
                """
            )

    def insert_recognition(self, record: dict[str, Any]) -> None:
        columns = tuple(record)
        values = [record[column] for column in columns]
        placeholders = ", ".join("?" for _ in columns)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO recognitions ({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )

    def upsert_feedback(
        self, recognition_id: str, verdict: str, correct_slug: str | None, note: str | None
    ) -> dict[str, Any]:
        created_at = utc_now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback (recognition_id, created_at, verdict, correct_slug, note)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(recognition_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    verdict=excluded.verdict,
                    correct_slug=excluded.correct_slug,
                    note=excluded.note
                """,
                (recognition_id, created_at, verdict, correct_slug, note),
            )
        return {
            "recognition_id": recognition_id,
            "created_at": created_at,
            "verdict": verdict,
            "correct_slug": correct_slug,
            "note": note,
        }

    def get_recognition(self, recognition_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT r.*, f.created_at AS feedback_created_at, f.verdict,
                       f.correct_slug, f.note
                FROM recognitions r
                LEFT JOIN feedback f ON f.recognition_id = r.id
                WHERE r.id = ?
                """,
                (recognition_id,),
            ).fetchone()
        return self._expand(row) if row else None

    def list_recognitions(
        self,
        limit: int = 50,
        offset: int = 0,
        mode: str | None = None,
        decision: str | None = None,
    ) -> dict[str, Any]:
        clauses: list[str] = []
        values: list[Any] = []
        if mode:
            clauses.append("r.mode = ?")
            values.append(mode)
        if decision:
            clauses.append("r.decision = ?")
            values.append(decision)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM recognitions r {where}", values
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT r.*, f.created_at AS feedback_created_at, f.verdict,
                       f.correct_slug, f.note
                FROM recognitions r
                LEFT JOIN feedback f ON f.recognition_id = r.id
                {where}
                ORDER BY r.created_at DESC
                LIMIT ? OFFSET ?
                """,
                [*values, limit, offset],
            ).fetchall()
        return {"total": total, "items": [self._expand(row) for row in rows]}

    def delete_recognition(self, recognition_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM recognitions WHERE id = ?",
                (recognition_id,),
            )
        return cursor.rowcount > 0

    def clear_recognitions(self) -> int:
        with self.connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM recognitions").fetchone()[0])
            connection.execute("DELETE FROM recognitions")
        return total

    def stats(self) -> dict[str, Any]:
        with self.connect() as connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total,
                    SUM(decision = 'match') AS matches,
                    SUM(decision = 'alternatives') AS alternatives,
                    SUM(decision = 'not_found') AS not_found,
                    AVG(total_ms) AS mean_ms
                FROM recognitions
                """
            ).fetchone()
            timings = [
                float(row[0])
                for row in connection.execute(
                    "SELECT total_ms FROM recognitions ORDER BY total_ms"
                ).fetchall()
            ]
            feedback = connection.execute(
                """
                SELECT COUNT(*) AS total,
                    SUM(verdict = 'correct') AS correct,
                    SUM(verdict = 'incorrect') AS incorrect,
                    SUM(verdict = 'not_in_catalog') AS not_in_catalog
                FROM feedback
                """
            ).fetchone()

        def percentile(values: list[float], fraction: float) -> float | None:
            if not values:
                return None
            index = round((len(values) - 1) * fraction)
            return values[index]

        return {
            "recognitions": {key: int(totals[key] or 0) for key in ("total", "matches", "alternatives", "not_found")},
            "timing_ms": {
                "mean": float(totals["mean_ms"]) if totals["mean_ms"] is not None else None,
                "p50": percentile(timings, 0.50),
                "p95": percentile(timings, 0.95),
            },
            "feedback": {key: int(feedback[key] or 0) for key in ("total", "correct", "incorrect", "not_in_catalog")},
        }

    @staticmethod
    def _expand(row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["result"] = json.loads(payload.pop("result_json"))
        payload["pipeline"] = json.loads(payload.pop("pipeline_json"))
        if payload.pop("feedback_created_at"):
            payload["feedback"] = {
                "verdict": payload.pop("verdict"),
                "correct_slug": payload.pop("correct_slug"),
                "note": payload.pop("note"),
            }
        else:
            payload.pop("verdict")
            payload.pop("correct_slug")
            payload.pop("note")
            payload["feedback"] = None
        return payload
