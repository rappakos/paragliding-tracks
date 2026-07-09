from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, UploadFile
from pydantic import BaseModel

from app.db import get_db
from app.services.igc_analysis import parse_igc
from app.services.thermal_analysis import analyze_thermal_segment
from app.services.thermal_ekf import estimate_centerline_ekf

router = APIRouter(prefix="/igc", tags=["igc"])

_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


@router.post("/upload")
async def upload_igc(file: UploadFile, x_owner_token: str = Header(default="")):
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
        # Deduplication: same filename + start_time + owner → return existing
        existing = conn.execute(
            "SELECT id, filename, pilot, glider, start_time, end_time, bbox, geojson FROM tracks WHERE filename = ? AND start_time = ? AND owner_token = ?",
            (file.filename, parsed["start_time"], x_owner_token),
        ).fetchone()

        if existing:
            return {
                "id": existing["id"],
                "filename": existing["filename"],
                "pilot": existing["pilot"],
                "glider": existing["glider"],
                "start_time": existing["start_time"],
                "end_time": existing["end_time"],
                "bbox": json.loads(existing["bbox"]),
                "geojson": json.loads(existing["geojson"]),
                "duplicate": True,
            }

        cur = conn.execute(
            """INSERT INTO tracks (filename, pilot, glider, start_time, end_time, bbox, geojson, owner_token)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file.filename,
                parsed["pilot"],
                parsed["glider"],
                parsed["start_time"],
                parsed["end_time"],
                parsed["bbox"],
                parsed["geojson"],
                x_owner_token,
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
async def list_tracks(x_owner_token: str = Header(default="")):
    """List tracks belonging to this owner token."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, filename, pilot, glider, start_time, end_time, bbox, created_at FROM tracks WHERE owner_token = ? ORDER BY created_at DESC",
            (x_owner_token,),
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
async def get_track(track_id: int, x_owner_token: str = Header(default="")):
    """Get a single track with full GeoJSON (must match owner)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tracks WHERE id = ? AND owner_token = ?", (track_id, x_owner_token)
        ).fetchone()

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
async def delete_track(track_id: int, x_owner_token: str = Header(default="")):
    """Delete a track (must match owner)."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM tracks WHERE id = ? AND owner_token = ?", (track_id, x_owner_token)
        )
        conn.commit()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Track not found.")

    return {"deleted": track_id}


class AnalyzeRequest(BaseModel):
    start_idx: int
    end_idx: int
    method: Literal["linreg", "ekf"] = "linreg"


@router.post("/tracks/{track_id}/analyze")
async def analyze_track(track_id: int, body: AnalyzeRequest, x_owner_token: str = Header(default="")):
    """Analyze a thermal segment of a track (linear regression or EKF core-position estimate)."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT geojson FROM tracks WHERE id = ? AND owner_token = ?", (track_id, x_owner_token)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Track not found.")

    geojson = json.loads(row["geojson"])
    coords = geojson.get("geometry", {}).get("coordinates", [])
    properties = geojson.get("properties", {})
    times = properties.get("times", [])
    pressure_alts = properties.get("pressure_alts")

    if not times:
        raise HTTPException(status_code=400, detail="Track has no timestamp data. Please re-upload.")

    if body.start_idx < 0 or body.end_idx >= len(coords) or body.start_idx >= body.end_idx:
        raise HTTPException(status_code=400, detail="Invalid index range.")

    try:
        if body.method == "ekf":
            result = estimate_centerline_ekf(coords, times, pressure_alts, body.start_idx, body.end_idx)
        else:
            result = analyze_thermal_segment(coords, times, body.start_idx, body.end_idx)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result
