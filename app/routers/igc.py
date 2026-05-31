from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, UploadFile

from app.db import get_db
from app.services.igc_analysis import parse_igc

router = APIRouter(prefix="/igc", tags=["igc"])

_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post("/upload")
async def upload_igc(file: UploadFile):
    """Upload and parse an IGC file. Stores processed track data in SQLite."""
    if not file.filename or not file.filename.lower().endswith(".igc"):
        raise HTTPException(status_code=400, detail="Only .igc files are accepted.")

    data = await file.read()
    if len(data) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 5 MB).")

    try:
        parsed = parse_igc(data)
    except (ValueError, Exception) as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse IGC file: {e}")

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO tracks (filename, pilot, glider, start_time, end_time, bbox, geojson)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                file.filename,
                parsed["pilot"],
                parsed["glider"],
                parsed["start_time"],
                parsed["end_time"],
                parsed["bbox"],
                parsed["geojson"],
            ),
        )
        conn.commit()
        track_id = cur.lastrowid

    return {
        "id": track_id,
        "filename": file.filename,
        "pilot": parsed["pilot"],
        "glider": parsed["glider"],
        "start_time": parsed["start_time"],
        "end_time": parsed["end_time"],
        "bbox": json.loads(parsed["bbox"]),
        "geojson": json.loads(parsed["geojson"]),
    }


@router.get("/tracks")
async def list_tracks():
    """List all stored tracks (without full GeoJSON)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, filename, pilot, glider, start_time, end_time, bbox, created_at FROM tracks ORDER BY created_at DESC"
        ).fetchall()

    return [
        {
            "id": r["id"],
            "filename": r["filename"],
            "pilot": r["pilot"],
            "glider": r["glider"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "bbox": json.loads(r["bbox"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.get("/tracks/{track_id}")
async def get_track(track_id: int):
    """Get a single track with full GeoJSON."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Track not found.")

    return {
        "id": row["id"],
        "filename": row["filename"],
        "pilot": row["pilot"],
        "glider": row["glider"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "bbox": json.loads(row["bbox"]),
        "geojson": json.loads(row["geojson"]),
        "created_at": row["created_at"],
    }


@router.delete("/tracks/{track_id}")
async def delete_track(track_id: int):
    """Delete a track."""
    with get_db() as conn:
        cur = conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
        conn.commit()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Track not found.")

    return {"deleted": track_id}
