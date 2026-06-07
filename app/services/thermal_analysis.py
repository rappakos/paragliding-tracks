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


def thermal_cross_section(
    coords: list[list[float]],
    times: list[float],
    start_idx: int,
    end_idx: int,
) -> dict:
    """Project a thermal segment onto the vertical plane spanned by (0,0,1)
    and the regression (drift) line.

    Mirrors the local-meter frame and degree-1 polyfit of
    ``analyze_thermal_segment`` so the result stays consistent with
    ``core_line`` / ``drift_bearing``. For each fix it returns the signed
    horizontal distance ``s`` along the horizontal drift direction together
    with the altitude, plus the regression line endpoints in that plane.

    Args:
        coords: Track coordinates [[lon, lat, alt], ...]
        times: Track timestamps (epoch seconds) — accepted for signature
            symmetry with ``analyze_thermal_segment``; not used in the math.
        start_idx: Start index of the thermal segment
        end_idx: End index of the thermal segment (inclusive)

    Returns:
        dict with keys:
            s: per-fix signed horizontal distance along drift (m), centered on
               the regression bottom
            alt: per-fix altitude (m)
            regression: {"s": [...], "alt": [...]} core line, extended down to
               the DEM ground beneath the core bottom when terrain is available
            ground: {"s": [...], "alt": [...]} DEM terrain profile sampled along
               the transect, or None if unavailable
            drift_unit: [u_x, u_y] horizontal drift unit vector
            drift_bearing: drift direction in degrees (0=N, clockwise)
    """
    seg_coords = coords[start_idx:end_idx + 1]

    if len(seg_coords) < 10:
        raise ValueError("Segment too short for analysis (need at least 10 fixes).")

    lons = np.array([c[0] for c in seg_coords])
    lats = np.array([c[1] for c in seg_coords])
    alts = np.array([c[2] for c in seg_coords])

    # Local-meter frame (identical to analyze_thermal_segment)
    lat_ref = lats.mean()
    lon_ref = lons.mean()
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))

    x = (lons - lon_ref) * m_per_deg_lon  # east in meters
    y = (lats - lat_ref) * m_per_deg_lat  # north in meters

    alt_min = float(alts.min())
    alt_max = float(alts.max())
    if alt_max - alt_min < 10:
        raise ValueError("Altitude gain too small for meaningful analysis.")

    coeff_x = np.polyfit(alts, x, 1)  # [slope, intercept]
    coeff_y = np.polyfit(alts, y, 1)

    # Horizontal drift direction = regression slopes per metre of altitude
    a_x, a_y = float(coeff_x[0]), float(coeff_y[0])
    mag = math.hypot(a_x, a_y)
    if mag < 1e-9:
        u_x, u_y = 1.0, 0.0  # degenerate / near-vertical core
    else:
        u_x, u_y = a_x / mag, a_y / mag

    # Regression endpoints projected onto the drift axis
    s_bottom = float(np.polyval(coeff_x, alt_min) * u_x + np.polyval(coeff_y, alt_min) * u_y)
    s_top = float(np.polyval(coeff_x, alt_max) * u_x + np.polyval(coeff_y, alt_max) * u_y)

    # Per-fix signed horizontal distance along the drift axis, centered on bottom
    s_fixes = (x * u_x + y * u_y) - s_bottom

    drift_bearing = (math.degrees(math.atan2(u_x, u_y)) + 360) % 360

    # DEM terrain profile along the transect (centered s coordinate)
    s_fix_lo, s_fix_hi = float(s_fixes.min()), float(s_fixes.max())
    ground = _ground_profile(
        lon_ref, lat_ref, m_per_deg_lon, m_per_deg_lat,
        u_x, u_y, s_bottom, s_fix_lo, s_fix_hi,
    )

    # Core line in the plane; extend it down to the DEM ground beneath the
    # core bottom (s=0), consistent with the trigger point in analyze_thermal_segment.
    reg_s = [0.0, float(s_top - s_bottom)]
    reg_alt = [alt_min, alt_max]
    if ground is not None and alt_max > alt_min:
        g0 = float(np.interp(0.0, ground["s"], ground["alt"]))
        if g0 < alt_min:
            slope = (reg_s[1] - reg_s[0]) / (alt_max - alt_min)  # d s / d alt
            reg_s = [slope * (g0 - alt_min), reg_s[1]]
            reg_alt = [g0, alt_max]
            # Re-sample terrain across the full plot width so the ground spans
            # the whole x-axis (including the ground-extended core line).
            plot_lo = min(s_fix_lo, reg_s[0], reg_s[1])
            plot_hi = max(s_fix_hi, reg_s[0], reg_s[1])
            wider = _ground_profile(
                lon_ref, lat_ref, m_per_deg_lon, m_per_deg_lat,
                u_x, u_y, s_bottom, plot_lo, plot_hi,
            )
            if wider is not None:
                ground = wider

    return {
        "s": [float(v) for v in s_fixes],
        "alt": [float(v) for v in alts],
        "regression": {"s": reg_s, "alt": reg_alt},
        "ground": ground,
        "drift_unit": [u_x, u_y],
        "drift_bearing": float(drift_bearing),
    }


def _ground_profile(
    lon_ref: float,
    lat_ref: float,
    m_per_deg_lon: float,
    m_per_deg_lat: float,
    u_x: float,
    u_y: float,
    s_bottom: float,
    s_min: float,
    s_max: float,
    n: int = 48,
) -> dict | None:
    """Sample the DEM terrain along the cross-section transect.

    Walks the horizontal drift axis ``u`` over the centered ``s`` range, maps
    each sample back to lon/lat, fetches a single DEM tile covering them all,
    and samples the elevation at each point.

    Returns {"s": [...centered...], "alt": [...metres...]} or None if no DEM
    is available / the tile reads as nodata.
    """
    try:
        from app.services.dem_source import get_dem_array

        span = s_max - s_min
        if span < 1.0:  # degenerate / tiny horizontal extent
            s_min, s_max = s_min - 50.0, s_max + 50.0
            span = s_max - s_min
        margin = 0.1 * span
        s_samples = np.linspace(s_min - margin, s_max + margin, n)

        # Map centered s back to lon/lat along the u-axis (s_raw = s + s_bottom)
        x = (s_samples + s_bottom) * u_x
        y = (s_samples + s_bottom) * u_y
        lons = lon_ref + x / m_per_deg_lon
        lats = lat_ref + y / m_per_deg_lat

        pad = 0.02  # keep bbox comfortably larger than the 0.01° DEM quantisation
        bbox = (float(lons.min()) - pad, float(lats.min()) - pad,
                float(lons.max()) + pad, float(lats.max()) + pad)
        arr, transform, _crs = get_dem_array(bbox, res_m=30)
        inv = ~transform

        elevs = []
        for lon, lat in zip(lons, lats):
            col, row = inv * (float(lon), float(lat))
            row = max(0, min(int(round(row)), arr.shape[0] - 1))
            col = max(0, min(int(round(col)), arr.shape[1] - 1))
            elevs.append(float(arr[row, col]))

        if not any(e > 0 for e in elevs):
            return None
        return {"s": [float(v) for v in s_samples], "alt": elevs}
    except Exception as exc:
        logger.warning("Ground profile sampling failed: %s", exc)
        return None


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
