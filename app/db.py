"""SQLite database for storing parsed IGC track data."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from app.config import settings

# Bump this when schema changes. DB will be dropped and recreated.
_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS _meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
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
CREATE TABLE IF NOT EXISTS bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,                      -- soft ref (nullable; track may be deleted)
    track_filename TEXT NOT NULL,          -- denormalized, survives track deletion
    name TEXT NOT NULL DEFAULT '',         -- user-editable
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    start_idx INTEGER NOT NULL,            -- provenance only
    end_idx INTEGER NOT NULL,
    fixes TEXT NOT NULL,                   -- JSON {"coords":[[lon,lat,alt],...],"times":[epoch,...]}
    weather TEXT,                          -- JSON {"lat","lon","when","wind":{...}} or NULL
    altitude_gain REAL,                    -- denormalized summary for fast grid
    n_turns REAL,
    avg_climb_rate REAL,
    method TEXT NOT NULL DEFAULT 'linreg',  -- 'linreg' or 'ekf'
    owner_token TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_bookmarks_owner ON bookmarks(owner_token, created_at);
"""


def _db_path() -> str:
    return settings.db_path


def init_db() -> None:
    """Create database file and tables if they don't exist.

    If the schema version doesn't match, the DB is dropped and recreated.
    """
    path = _db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with sqlite3.connect(path) as conn:
            try:
                row = conn.execute("SELECT value FROM _meta WHERE key = 'schema_version'").fetchone()
                existing_version = int(row[0]) if row else 0
            except sqlite3.OperationalError:
                existing_version = 0

        if existing_version != _SCHEMA_VERSION:
            try:
                os.remove(path)
            except OSError:
                # File locked (Windows) — drop user tables in-place instead
                with sqlite3.connect(path) as conn:
                    tables = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                    for (name,) in tables:
                        conn.execute(f"DROP TABLE IF EXISTS [{name}]")
                    conn.commit()

    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES ('schema_version', ?)",
            (str(_SCHEMA_VERSION),),
        )
        # Non-destructive migration: add method column to existing databases.
        try:
            conn.execute("ALTER TABLE bookmarks ADD COLUMN method TEXT NOT NULL DEFAULT 'linreg'")
        except sqlite3.OperationalError:
            pass  # column already present
        conn.commit()


@contextmanager
def get_db():
    """Yield a sqlite3 connection with row_factory set."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
