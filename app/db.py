"""SQLite database for storing parsed IGC track data."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    pilot TEXT,
    glider TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    bbox TEXT NOT NULL,
    geojson TEXT NOT NULL,
    owner_token TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_tracks_dedup ON tracks(filename, start_time, owner_token);
"""


def _db_path() -> str:
    return settings.db_path


def init_db() -> None:
    """Create database file and tables if they don't exist."""
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)


@contextmanager
def get_db():
    """Yield a sqlite3 connection with row_factory set."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
