"""
SQLite-backed session store -- replaces the earlier in-memory dict so
anomaly history actually survives a process restart. Swap for
Postgres/Redis at higher scale (this is a single-file WAL-mode SQLite DB,
fine for a single-instance deployment or local dev, not for multi-worker
horizontal scaling without a shared DB).
"""
import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    reference_signature BLOB,
    frames_analyzed INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    anomaly_type TEXT,
    ts REAL DEFAULT (strftime('%s','now')),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.executescript(SCHEMA)


def get_or_create_session(session_id: str):
    with _conn() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,))


def set_reference(session_id: str, signature_bytes: bytes):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, reference_signature) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET reference_signature=excluded.reference_signature",
            (session_id, signature_bytes),
        )


def get_reference(session_id: str) -> bytes | None:
    with _conn() as conn:
        row = conn.execute("SELECT reference_signature FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return row[0] if row else None


def record_frame(session_id: str, anomalies: list[str]):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, frames_analyzed) VALUES (?, 1) "
            "ON CONFLICT(session_id) DO UPDATE SET frames_analyzed = frames_analyzed + 1",
            (session_id,),
        )
        for a in anomalies:
            conn.execute(
                "INSERT INTO anomalies (session_id, anomaly_type) VALUES (?, ?)", (session_id, a)
            )


def get_summary(session_id: str) -> dict:
    with _conn() as conn:
        row = conn.execute(
            "SELECT frames_analyzed, reference_signature FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        frames, ref = row if row else (0, None)
        counts = conn.execute(
            "SELECT anomaly_type, COUNT(*) FROM anomalies WHERE session_id=? GROUP BY anomaly_type",
            (session_id,),
        ).fetchall()
        return {
            "frames_analyzed": frames,
            "has_reference": ref is not None,
            "anomaly_counts": dict(counts),
        }
