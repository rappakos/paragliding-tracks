from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.models.schemas import SunState, WindState
from app.services.sun import sun_state
from app.services.wind import wind_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["weather"])


def _bbox_center(bbox_str: str):
    parts = [float(x) for x in bbox_str.split(",")]
    west, south, east, north = parts
    return (south + north) / 2, (west + east) / 2


def _parse_dt(dt_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except Exception:
        raise HTTPException(status_code=422, detail=f"Invalid datetime: {dt_str!r}")


@router.get("/sun", response_model=SunState)
async def get_sun(
    bbox: str = Query(default=None),
    datetime_: str = Query(default=None, alias="datetime"),
):
    if bbox is None:
        bbox = settings.default_bbox
    lat, lon = _bbox_center(bbox)
    if datetime_ is None:
        when = datetime.now(timezone.utc)
    else:
        when = _parse_dt(datetime_)

    try:
        return sun_state(lat, lon, when)
    except Exception as exc:
        logger.error("sun_state failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/wind", response_model=WindState)
async def get_wind(
    bbox: str = Query(default=None),
    datetime_: str = Query(default=None, alias="datetime"),
):
    if bbox is None:
        bbox = settings.default_bbox
    lat, lon = _bbox_center(bbox)
    if datetime_ is None:
        when = datetime.now(timezone.utc)
    else:
        when = _parse_dt(datetime_)

    try:
        return wind_state(lat, lon, when)
    except Exception as exc:
        logger.error("wind_state failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
