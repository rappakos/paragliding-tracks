"""
Solar position and clear-sky GHI via pvlib.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
import pvlib

from app.cache import sun_cache
from app.models.schemas import SunState

logger = logging.getLogger(__name__)


def _quantise(lat: float, lon: float, dt: datetime, step_deg: float = 0.1, step_min: int = 1) -> tuple:
    qlat = round(lat / step_deg) * step_deg
    qlon = round(lon / step_deg) * step_deg
    # truncate to minute
    t = dt.replace(second=0, microsecond=0, tzinfo=timezone.utc)
    return (round(qlat, 4), round(qlon, 4), t.isoformat())


def _az_alt_to_enu(azimuth_deg: float, elevation_deg: float) -> List[float]:
    """Convert (azimuth, elevation) to ENU unit vector."""
    az = math.radians(azimuth_deg)
    alt = math.radians(elevation_deg)
    east = math.cos(alt) * math.sin(az)
    north = math.cos(alt) * math.cos(az)
    up = math.sin(alt)
    return [east, north, up]


def sun_state(lat: float, lon: float, when_utc: datetime) -> SunState:
    """Return solar position + clear-sky GHI for the given location and time."""
    key = _quantise(lat, lon, when_utc)
    cached = sun_cache.get(key)
    if cached is not None:
        return cached

    t = pd.DatetimeIndex([when_utc])
    pos = pvlib.solarposition.get_solarposition(t, lat, lon)

    az = float(pos["azimuth"].iloc[0])
    el = float(pos["elevation"].iloc[0])

    # Clear-sky GHI
    cs = pvlib.clearsky.haurwitz(pos["apparent_zenith"])
    ghi = float(cs["ghi"].iloc[0])

    sun_vec = _az_alt_to_enu(az, el)

    result = SunState(azimuth=az, elevation=el, ghi=ghi, sun_vec=sun_vec)
    sun_cache.set(key, result)
    return result
