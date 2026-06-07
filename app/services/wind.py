"""
Wind data from open-meteo API.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import List

import httpx

from app.cache import wind_cache
from app.config import settings
from app.models.schemas import WindLevel, WindState

logger = logging.getLogger(__name__)

_PRESSURE_LEVELS = [925, 850, 700]  # hPa


def _quantise(lat: float, lon: float, dt: datetime, step: float = 0.25) -> tuple:
    qlat = round(lat / step) * step
    qlon = round(lon / step) * step
    t = dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    return (round(qlat, 4), round(qlon, 4), t.isoformat())


def _polar_to_uv(speed: float, direction_from: float):
    """Convert met wind (direction from, speed m/s) to (u, v) ENU components."""
    # direction_from: where wind comes FROM, in degrees from north
    # wind blows TOWARD opposite direction
    toward = math.radians(direction_from + 180.0)
    u = speed * math.sin(toward)   # east
    v = speed * math.cos(toward)   # north
    return u, v


def _unit_vec(u: float, v: float) -> List[float]:
    mag = math.sqrt(u * u + v * v)
    if mag < 1e-6:
        return [0.0, 0.0, 0.0]
    return [u / mag, v / mag, 0.0]


def wind_state(lat: float, lon: float, when_utc: datetime) -> WindState:
    key = _quantise(lat, lon, when_utc)
    cached = wind_cache.get(key)
    if cached is not None:
        return cached

    try:
        result = _fetch_wind(lat, lon, when_utc)
    except Exception as exc:
        logger.warning("open-meteo fetch failed (%s); using synthetic wind.", exc)
        result = _synthetic_wind()

    wind_cache.set(key, result)
    return result


def _fetch_wind(lat: float, lon: float, when_utc: datetime) -> WindState:
    """Fetch real wind data from open-meteo (forecast or historical archive)."""
    date_str = when_utc.strftime("%Y-%m-%d")
    hour = when_utc.hour

    # Build variable lists
    hourly_vars = [
        "wind_speed_10m",
        "wind_direction_10m",
    ]
    for lv in _PRESSURE_LEVELS:
        hourly_vars += [f"wind_speed_{lv}hPa", f"wind_direction_{lv}hPa"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "start_date": date_str,
        "end_date": date_str,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }

    # For dates older than 5 days, use the historical-forecast archive rather
    # than the ERA5 archive: ERA5 (archive-api) only serves surface (10 m) wind
    # and returns null for pressure levels, whereas the historical-forecast API
    # provides the 925/850/700 hPa winds we need (data from 2022 onwards).
    now = datetime.now(timezone.utc)
    days_ago = (now - when_utc).days
    if days_ago > 5:
        api_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        api_url = f"{settings.openmeteo_base}/forecast"

    with httpx.Client(timeout=15.0) as client:
        resp = client.get(api_url, params=params)
        resp.raise_for_status()
        data = resp.json()

    hourly = data["hourly"]
    time_list = hourly["time"]

    # Find closest hour index
    idx = 0
    target_str = when_utc.strftime("%Y-%m-%dT%H:00")
    for i, t in enumerate(time_list):
        if t == target_str:
            idx = i
            break

    levels: List[WindLevel] = []

    # 10m
    spd_10 = hourly["wind_speed_10m"][idx] or 0.0
    dir_10 = hourly["wind_direction_10m"][idx] or 0.0
    u10, v10 = _polar_to_uv(spd_10, dir_10)
    levels.append(WindLevel(level="10m", speed=spd_10, direction=dir_10, u=u10, v=v10))

    # Pressure levels
    for lv in _PRESSURE_LEVELS:
        spd = hourly.get(f"wind_speed_{lv}hPa", [0.0] * len(time_list))[idx] or 0.0
        drn = hourly.get(f"wind_direction_{lv}hPa", [0.0] * len(time_list))[idx] or 0.0
        u, v = _polar_to_uv(spd, drn)
        levels.append(WindLevel(level=f"{lv}hPa", speed=spd, direction=drn, u=u, v=v))

    weights = settings.wind_weights
    mean_u = sum(w * lv.u for w, lv in zip(weights, levels))
    mean_v = sum(w * lv.v for w, lv in zip(weights, levels))

    return WindState(
        levels=levels,
        mean_u=mean_u,
        mean_v=mean_v,
        mean_vec=_unit_vec(mean_u, mean_v),
    )


def _synthetic_wind() -> WindState:
    """Return a synthetic 5 m/s NW wind when the API is unavailable."""
    spd = 5.0
    drn = 315.0  # from NW
    u, v = _polar_to_uv(spd, drn)
    lv = WindLevel(level="10m", speed=spd, direction=drn, u=u, v=v)
    return WindState(
        levels=[lv],
        mean_u=u,
        mean_v=v,
        mean_vec=_unit_vec(u, v),
    )
