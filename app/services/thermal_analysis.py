"""
Thermal core analysis – linear regression approximation.

Given a segment of a paraglider track (circling in a thermal),
estimates the thermal core position as a function of altitude
using linear regression on the pilot's GPS positions.
Extrapolates the core line to ground level using DEM data.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def analyze_thermal_segment(
    coords: list[list[float]],
    times: list[float],
    start_idx: int,
    end_idx: int,
) -> dict:
    """Analyze a thermal segment of a track.

    Args:
        coords: Full track coordinates [[lon, lat, alt], ...]
        times: Full track timestamps (epoch seconds)
        start_idx: Start index of the thermal segment
        end_idx: End index of the thermal segment (inclusive)

    Returns:
        dict with keys:
            core_line: GeoJSON Feature (LineString) from ground to top
            avg_climb_rate: average climb rate in m/s
            altitude_gain: total altitude gained in m
            n_turns: estimated number of full turns
            drift_bearing: drift direction in degrees (0=N, clockwise)
            drift_speed: drift speed in m/s (horizontal)
    """
    seg_coords = coords[start_idx:end_idx + 1]
    seg_times = times[start_idx:end_idx + 1]

    if len(seg_coords) < 10:
        raise ValueError("Segment too short for analysis (need at least 10 fixes).")

    lons = np.array([c[0] for c in seg_coords])
    lats = np.array([c[1] for c in seg_coords])
    alts = np.array([c[2] for c in seg_coords])
    ts = np.array(seg_times)

    # Convert to local meters (approximate, for analysis purposes)
    lat_ref = lats.mean()
    lon_ref = lons.mean()
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))

    x = (lons - lon_ref) * m_per_deg_lon  # east in meters
    y = (lats - lat_ref) * m_per_deg_lat  # north in meters

    # ── Linear regression: x,y as function of altitude ──
    # This gives the thermal core drift as altitude increases
    alt_min = alts.min()
    alt_max = alts.max()
    altitude_gain = float(alt_max - alt_min)

    # Fit x = a_x * alt + b_x, y = a_y * alt + b_y
    # Using numpy polyfit (degree 1)
    if altitude_gain < 10:
        raise ValueError("Altitude gain too small for meaningful analysis.")

    coeff_x = np.polyfit(alts, x, 1)  # [slope, intercept]
    coeff_y = np.polyfit(alts, y, 1)  # [slope, intercept]

    # Core position at min and max altitude
    x_bottom = np.polyval(coeff_x, alt_min)
    y_bottom = np.polyval(coeff_y, alt_min)
    x_top = np.polyval(coeff_x, alt_max)
    y_top = np.polyval(coeff_y, alt_max)

    # Convert back to lon/lat
    lon_bottom = lon_ref + x_bottom / m_per_deg_lon
    lat_bottom = lat_ref + y_bottom / m_per_deg_lat
    lon_top = lon_ref + x_top / m_per_deg_lon
    lat_top = lat_ref + y_top / m_per_deg_lat

    # Extrapolate core line to ground level using DEM
    ground_elevation = _get_ground_elevation(float(lon_bottom), float(lat_bottom))
    if ground_elevation is not None and ground_elevation < alt_min:
        # Extrapolate the regression line down to ground level
        x_ground = np.polyval(coeff_x, ground_elevation)
        y_ground = np.polyval(coeff_y, ground_elevation)
        lon_ground = lon_ref + x_ground / m_per_deg_lon
        lat_ground = lat_ref + y_ground / m_per_deg_lat
    else:
        # Fallback: use bottom of track segment
        lon_ground, lat_ground, ground_elevation = lon_bottom, lat_bottom, float(alt_min)

    # Core line GeoJSON (from ground to top)
    core_line = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [float(lon_ground), float(lat_ground), float(ground_elevation)],
                [float(lon_bottom), float(lat_bottom), float(alt_min)],
                [float(lon_top), float(lat_top), float(alt_max)],
            ],
        },
        "properties": {
            "type": "thermal_core",
            "ground_elevation": float(ground_elevation),
            "alt_min": float(alt_min),
            "alt_max": float(alt_max),
            "trigger_point": [float(lon_ground), float(lat_ground)],
        },
    }

    # ── Climb rate ──
    duration = ts[-1] - ts[0]
    avg_climb_rate = altitude_gain / duration if duration > 0 else 0.0

    # ── Number of turns ──
    # Estimate by counting sign changes in the cross product of successive vectors
    # relative to the regression center at each point
    cx = np.polyval(coeff_x, alts)  # center x at each altitude
    cy = np.polyval(coeff_y, alts)  # center y at each altitude
    dx = x - cx  # relative position to core
    dy = y - cy

    # Angle from center
    angles = np.arctan2(dy, dx)
    # Unwrap to count total rotation
    unwrapped = np.unwrap(angles)
    total_rotation = abs(unwrapped[-1] - unwrapped[0])
    n_turns = total_rotation / (2 * math.pi)

    # ── Drift vector ──
    drift_x = x_top - x_bottom  # meters east
    drift_y = y_top - y_bottom  # meters north
    drift_dist = math.sqrt(drift_x**2 + drift_y**2)
    drift_speed = drift_dist / duration if duration > 0 else 0.0
    drift_bearing = (math.degrees(math.atan2(drift_x, drift_y)) + 360) % 360

    return {
        "core_line": core_line,
        "avg_climb_rate": float(avg_climb_rate),
        "altitude_gain": float(altitude_gain),
        "n_turns": round(float(n_turns), 1),
        "drift_bearing": float(drift_bearing),
        "drift_speed": float(drift_speed),
        "ground_elevation": float(ground_elevation) if ground_elevation is not None else None,
        "trigger_point": [float(lon_ground), float(lat_ground)],
    }


def _get_ground_elevation(lon: float, lat: float) -> float | None:
    """Look up terrain elevation at a point using the DEM service."""
    try:
        from app.services.dem_source import get_dem_array

        # Bbox must be larger than the quantisation step (0.01) in dem_source
        delta = 0.015
        bbox = (lon - delta, lat - delta, lon + delta, lat + delta)
        arr, transform, _crs = get_dem_array(bbox, res_m=30)

        # Sample elevation at the point using the affine transform
        col, row = ~transform * (lon, lat)
        row, col = int(round(row)), int(round(col))
        row = max(0, min(row, arr.shape[0] - 1))
        col = max(0, min(col, arr.shape[1] - 1))
        elev = float(arr[row, col])
        if elev <= 0:
            return None
        return elev
    except Exception as exc:
        logger.warning("DEM lookup failed at (%.5f, %.5f): %s", lon, lat, exc)
        return None
