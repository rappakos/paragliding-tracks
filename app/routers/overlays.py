from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.cache import overlay_cache
from app.config import settings
from app.services.dem_source import get_dem_array, _quantise_bbox
from app.services.triangulation import get_normals_cached
from app.services.sun import sun_state
from app.services.wind import wind_state
from app.services.overlay import thermal_overlay, wind_overlay

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/overlay", tags=["overlays"])


def _parse_bbox(bbox_str: str):
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)
    except Exception:
        raise HTTPException(status_code=422, detail="bbox must be W,S,E,N")


def _bbox_center(bb):
    west, south, east, north = bb
    return (south + north) / 2, (west + east) / 2


def _parse_dt(dt_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {dt_str!r}")


async def _get_normals(bb):
    dem_arr, transform, _ = get_dem_array(bb)
    qbbox = _quantise_bbox(bb)
    return get_normals_cached(("normals", qbbox), dem_arr, transform)


@router.get("/thermal")
async def get_thermal_overlay(
    bbox: str = Query(default=None),
    datetime_: str = Query(default=None, alias="datetime"),
):
    if bbox is None:
        bbox = settings.default_bbox
    bb = _parse_bbox(bbox)
    if datetime_ is None:
        when = datetime.now(timezone.utc)
    else:
        when = _parse_dt(datetime_)

    cache_key = ("thermal", tuple(bb), when.replace(minute=0, second=0, microsecond=0).isoformat())
    cached = overlay_cache.get(cache_key)
    if cached is not None:
        return _png_response(cached, bb)

    try:
        normals = await _get_normals(bb)
        lat, lon = _bbox_center(bb)
        sun = sun_state(lat, lon, when)
        png = thermal_overlay(normals, sun.sun_vec, sun.ghi)
    except Exception as exc:
        logger.error("thermal overlay failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    overlay_cache.set(cache_key, png, expire=settings.overlay_ttl)
    return _png_response(png, bb)


@router.get("/wind")
async def get_wind_overlay(
    bbox: str = Query(default=None),
    datetime_: str = Query(default=None, alias="datetime"),
):
    if bbox is None:
        bbox = settings.default_bbox
    bb = _parse_bbox(bbox)
    if datetime_ is None:
        when = datetime.now(timezone.utc)
    else:
        when = _parse_dt(datetime_)

    cache_key = ("wind_overlay", tuple(bb), when.replace(minute=0, second=0, microsecond=0).isoformat())
    cached = overlay_cache.get(cache_key)
    if cached is not None:
        return _png_response(cached, bb)

    try:
        normals = await _get_normals(bb)
        lat, lon = _bbox_center(bb)
        wind = wind_state(lat, lon, when)
        png = wind_overlay(normals, wind.mean_vec)
    except Exception as exc:
        logger.error("wind overlay failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    overlay_cache.set(cache_key, png, expire=settings.overlay_ttl)
    return _png_response(png, bb)


def _png_response(png_bytes: bytes, bb) -> Response:
    west, south, east, north = bb
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-BBox": f"{west},{south},{east},{north}",
            "Access-Control-Expose-Headers": "X-BBox",
        },
    )
