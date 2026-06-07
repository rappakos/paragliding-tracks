from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import anyio
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.db import get_db
from app.services.thermal_analysis import analyze_thermal_segment, thermal_cross_section
from app.services.wind import wind_state

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bookmarks", tags=["bookmarks"])


class CreateBookmarkRequest(BaseModel):
    track_id: int
    start_idx: int
    end_idx: int
    name: str | None = None
    capture_weather: bool = True


class RenameBookmarkRequest(BaseModel):
    name: str


def _epoch_to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


@router.post("", status_code=201)
async def create_bookmark(body: CreateBookmarkRequest, x_owner_token: str = Header(default="")):
    """Snapshot a thermal segment (fixes, timestamps, summary, wind) as a bookmark."""
    with get_db() as conn:
        track = conn.execute(
            "SELECT geojson, filename FROM tracks WHERE id = ? AND owner_token = ?",
            (body.track_id, x_owner_token),
        ).fetchone()

    if not track:
        raise HTTPException(status_code=404, detail="Track not found.")

    geojson = json.loads(track["geojson"])
    coords = geojson.get("geometry", {}).get("coordinates", [])
    times = geojson.get("properties", {}).get("times", [])

    if not times:
        raise HTTPException(status_code=400, detail="Track has no timestamp data. Please re-upload.")

    if body.start_idx < 0 or body.end_idx >= len(coords) or body.start_idx >= body.end_idx:
        raise HTTPException(status_code=400, detail="Invalid index range.")

    seg_coords = coords[body.start_idx:body.end_idx + 1]
    seg_times = times[body.start_idx:body.end_idx + 1]

    if len(seg_coords) < 10:
        raise HTTPException(status_code=400, detail="Segment too short (need at least 10 fixes).")

    start_time = _epoch_to_iso(seg_times[0])
    end_time = _epoch_to_iso(seg_times[-1])

    # Denormalized summary for the grid (best-effort — a too-short/flat segment is still bookmarkable)
    altitude_gain = n_turns = avg_climb_rate = None
    try:
        summary = analyze_thermal_segment(coords, times, body.start_idx, body.end_idx)
        altitude_gain = summary["altitude_gain"]
        n_turns = summary["n_turns"]
        avg_climb_rate = summary["avg_climb_rate"]
    except ValueError as e:
        logger.info("Bookmark summary skipped (%s)", e)

    # Wind snapshot at the segment's mean location and middle timestamp (frozen at create time)
    weather_json = None
    if body.capture_weather:
        mid_idx = (body.start_idx + body.end_idx) // 2
        lat_mid = sum(c[1] for c in seg_coords) / len(seg_coords)
        lon_mid = sum(c[0] for c in seg_coords) / len(seg_coords)
        when_utc = datetime.fromtimestamp(float(times[mid_idx]), tz=timezone.utc)
        try:
            wind = await anyio.to_thread.run_sync(wind_state, lat_mid, lon_mid, when_utc)
            weather_json = json.dumps({
                "lat": lat_mid,
                "lon": lon_mid,
                "when": when_utc.isoformat(),
                "wind": wind.model_dump(),
            })
        except Exception as e:  # never fail the bookmark over a weather hiccup
            logger.warning("Wind snapshot failed for bookmark: %s", e)

    fixes_json = json.dumps({
        "coords": seg_coords,
        "times": seg_times,
    })

    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO bookmarks
               (track_id, track_filename, name, start_time, end_time, start_idx, end_idx,
                fixes, weather, altitude_gain, n_turns, avg_climb_rate, owner_token)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                body.track_id,
                track["filename"],
                body.name or "",
                start_time,
                end_time,
                body.start_idx,
                body.end_idx,
                fixes_json,
                weather_json,
                altitude_gain,
                n_turns,
                avg_climb_rate,
                x_owner_token,
            ),
        )
        conn.commit()
        bookmark_id = cur.lastrowid

    return {
        "id": bookmark_id,
        "track_id": body.track_id,
        "track_filename": track["filename"],
        "name": body.name or "",
        "start_time": start_time,
        "end_time": end_time,
    }


@router.get("")
async def list_bookmarks(x_owner_token: str = Header(default="")):
    """List bookmarks for this owner, newest first (grid data — no heavy columns)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, track_id, track_filename, name, start_time, end_time,
                      altitude_gain, n_turns, avg_climb_rate, created_at
               FROM bookmarks WHERE owner_token = ? ORDER BY created_at DESC""",
            (x_owner_token,),
        ).fetchall()

    return [
        {
            "id": r["id"],
            "track_id": r["track_id"],
            "track_filename": r["track_filename"],
            "name": r["name"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "altitude_gain": r["altitude_gain"],
            "n_turns": r["n_turns"],
            "avg_climb_rate": r["avg_climb_rate"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.get("/{bookmark_id}")
async def get_bookmark(bookmark_id: int, x_owner_token: str = Header(default="")):
    """Full bookmark detail: fixes, recomputed analysis, cross-section, weather snapshot."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM bookmarks WHERE id = ? AND owner_token = ?",
            (bookmark_id, x_owner_token),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Bookmark not found.")

    fixes = json.loads(row["fixes"])
    coords = fixes.get("coords", [])
    times = fixes.get("times", [])
    last = len(coords) - 1

    # Recompute on the stored subset (indices 0..N-1, NOT the original track indices)
    analysis = None
    cross_section = None
    try:
        analysis = analyze_thermal_segment(coords, times, 0, last)
        cross_section = thermal_cross_section(coords, times, 0, last)
    except ValueError as e:
        logger.info("Bookmark %d analysis unavailable (%s)", bookmark_id, e)

    return {
        "id": row["id"],
        "track_id": row["track_id"],
        "track_filename": row["track_filename"],
        "name": row["name"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "created_at": row["created_at"],
        "fixes": fixes,
        "analysis": analysis,
        "cross_section": cross_section,
        "weather": json.loads(row["weather"]) if row["weather"] else None,
    }


@router.patch("/{bookmark_id}")
async def rename_bookmark(bookmark_id: int, body: RenameBookmarkRequest, x_owner_token: str = Header(default="")):
    """Update a bookmark's custom name."""
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE bookmarks SET name = ? WHERE id = ? AND owner_token = ?",
            (body.name, bookmark_id, x_owner_token),
        )
        conn.commit()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Bookmark not found.")

    return {"id": bookmark_id, "name": body.name}


@router.delete("/{bookmark_id}")
async def delete_bookmark(bookmark_id: int, x_owner_token: str = Header(default="")):
    """Delete a bookmark (must match owner)."""
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE id = ? AND owner_token = ?",
            (bookmark_id, x_owner_token),
        )
        conn.commit()

    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Bookmark not found.")

    return {"deleted": bookmark_id}
