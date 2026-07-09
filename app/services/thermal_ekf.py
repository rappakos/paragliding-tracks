"""
Thermal centerline estimation via Extended Kalman Filter — Phase 1.

Fits a stationary thermal plume to a manually selected climb segment of a
paraglider IGC track using GPS position (H1) and baro-vario (H2)
measurements, running in reverse time from the top of the segment, then
extrapolates the plume centerline down to DEM terrain to locate the ground
trigger point.

See ../../EKF_MODEL.md for the full model derivation. This implements
sections 1-2, 4, 6-8, with the H3 wind-advection regularizer, RTS smoothing,
and uncertainty-ellipse reporting deferred to a later pass — the filter here
is already fully wind-independent and self-contained.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# State vector index (EKF_MODEL.md §2): [C_x, C_y, s_x, s_y, phi, R, w, h]
C_X, C_Y, S_X, S_Y, PHI, R, W, H = range(8)
N_STATES = 8

# Model constants (EKF_MODEL.md §3)
V_TAN = 9.0    # air-relative tangential speed, m/s
W_SINK = 1.0   # air-relative sink rate, m/s

# Process noise (diagonal of Q, scaled by |dt| each step). Tune q_s/q_R/q_w
# first if the filter tracks too stiffly or too loosely.
Q_DIAG = np.array([0.01, 0.01, 0.01, 0.01, 0.001, 0.05, 0.02, 0.001])

# Measurement noise
SIGMA_XY = 5.0      # GPS horizontal accuracy, m
SIGMA_Z = 10.0       # GPS altitude accuracy, m (worse than horizontal)
SIGMA_VARIO = 0.3    # baro-derived vario noise, m/s

R1_MAT = np.diag([SIGMA_XY**2, SIGMA_XY**2, SIGMA_Z**2])
R2_MAT = np.array([[SIGMA_VARIO**2]])

_MIN_FIXES = 10
_INIT_WINDOW = 40   # fixes near the top of the segment used for circle-fit init


# ── Local frame (matches thermal_analysis.py's convention) ──

def _to_local_xy(lons: np.ndarray, lats: np.ndarray):
    lat_ref = float(lats.mean())
    lon_ref = float(lons.mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref))
    x = (lons - lon_ref) * m_per_deg_lon
    y = (lats - lat_ref) * m_per_deg_lat
    return x, y, lon_ref, lat_ref, m_per_deg_lon, m_per_deg_lat


def _from_local_xy(x, y, lon_ref, lat_ref, m_per_deg_lon, m_per_deg_lat):
    return lon_ref + x / m_per_deg_lon, lat_ref + y / m_per_deg_lat


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


# ── Circle-fit initializer (EKF_MODEL.md §7) ──

def _fit_circle(x: np.ndarray, y: np.ndarray):
    """Kasa least-squares circle fit. Returns (cx, cy, radius)."""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    d, e, f = sol
    cx, cy = d / 2.0, e / 2.0
    radius = math.sqrt(max(f + cx**2 + cy**2, 1e-6))
    return cx, cy, radius


def _turn_direction(x: np.ndarray, y: np.ndarray, cx: float, cy: float) -> float:
    """+1 for counterclockwise (increasing phi), -1 for clockwise."""
    angles = np.arctan2(y - cy, x - cx)
    diffs = np.diff(np.unwrap(angles))
    return 1.0 if diffs.mean() >= 0 else -1.0


def _init_state(x_local: np.ndarray, y_local: np.ndarray, alts: np.ndarray, times: np.ndarray):
    """Fit a circle to the top of the segment. Returns (x0, P0, sigma)."""
    n = len(x_local)
    window = min(_INIT_WINDOW, n)
    xi, yi = x_local[n - window:], y_local[n - window:]

    cx, cy, radius = _fit_circle(xi, yi)
    sigma = _turn_direction(xi, yi, cx, cy)
    phi0 = math.atan2(y_local[-1] - cy, x_local[-1] - cx)

    dt_span = times[-1] - times[n - window]
    climb0 = (alts[-1] - alts[n - window]) / dt_span if dt_span else 0.0
    w0 = climb0 + W_SINK

    x0 = np.zeros(N_STATES)
    x0[C_X], x0[C_Y] = cx, cy
    x0[PHI] = phi0
    x0[R] = radius
    x0[W] = w0
    x0[H] = alts[-1]

    # Small on C, R, phi; larger on s, w (EKF_MODEL.md §7).
    P0 = np.diag([25.0, 25.0, 1.0, 1.0, 0.05, 25.0, 4.0, 4.0])
    return x0, P0, sigma


# ── Predict / update (EKF_MODEL.md §6) ──

def _predict(x: np.ndarray, P: np.ndarray, dt: float, sigma: float):
    climb = x[W] - W_SINK
    f = np.zeros(N_STATES)
    f[C_X] = x[S_X] * climb
    f[C_Y] = x[S_Y] * climb
    f[PHI] = sigma * V_TAN / x[R]
    f[H] = climb

    x_pred = x + f * dt
    x_pred[PHI] = _wrap_to_pi(x_pred[PHI])

    F = np.zeros((N_STATES, N_STATES))
    F[C_X, S_X] = climb
    F[C_X, W] = x[S_X]
    F[C_Y, S_Y] = climb
    F[C_Y, W] = x[S_Y]
    F[PHI, R] = -sigma * V_TAN / x[R]**2
    F[H, W] = 1.0

    Phi = np.eye(N_STATES) + F * dt
    P_pred = Phi @ P @ Phi.T + np.diag(Q_DIAG) * abs(dt)
    return x_pred, P_pred


def _update(x: np.ndarray, P: np.ndarray, z: np.ndarray, h_pred: np.ndarray,
            H_mat: np.ndarray, R_mat: np.ndarray):
    y = z - h_pred
    S = H_mat @ P @ H_mat.T + R_mat
    K = P @ H_mat.T @ np.linalg.inv(S)
    x = x + K @ y
    x[PHI] = _wrap_to_pi(x[PHI])
    A = np.eye(N_STATES) - K @ H_mat
    P = A @ P @ A.T + K @ R_mat @ K.T
    return x, P


def _h1_predict(x: np.ndarray) -> np.ndarray:
    return np.array([x[C_X] + x[R] * math.cos(x[PHI]),
                      x[C_Y] + x[R] * math.sin(x[PHI]),
                      x[H]])


def _h1_jacobian(x: np.ndarray) -> np.ndarray:
    H_mat = np.zeros((3, N_STATES))
    H_mat[0, C_X] = 1.0
    H_mat[0, PHI] = -x[R] * math.sin(x[PHI])
    H_mat[0, R] = math.cos(x[PHI])
    H_mat[1, C_Y] = 1.0
    H_mat[1, PHI] = x[R] * math.cos(x[PHI])
    H_mat[1, R] = math.sin(x[PHI])
    H_mat[2, H] = 1.0
    return H_mat


def _h2_predict(x: np.ndarray) -> np.ndarray:
    return np.array([x[W] - W_SINK])


def _h2_jacobian(x: np.ndarray) -> np.ndarray:
    H_mat = np.zeros((1, N_STATES))
    H_mat[0, W] = 1.0
    return H_mat


def _step(x: np.ndarray, P: np.ndarray, dt: float, sigma: float,
          gps_z: np.ndarray, vario_z: float | None):
    x, P = _predict(x, P, dt, sigma)
    x, P = _update(x, P, gps_z, _h1_predict(x), _h1_jacobian(x), R1_MAT)
    if vario_z is not None:
        x, P = _update(x, P, np.array([vario_z]), _h2_predict(x), _h2_jacobian(x), R2_MAT)
    return x, P


def _run_filter(x_local: np.ndarray, y_local: np.ndarray, alts: np.ndarray,
                 times: np.ndarray, pressure_alts: np.ndarray,
                 x0: np.ndarray, P0: np.ndarray, sigma: float):
    """Reverse-time run: top (index n-1) down to bottom (index 0)."""
    n = len(times)
    order = list(range(n - 1, -1, -1))
    use_vario = bool(np.any(pressure_alts > 0))

    xs = [x0]
    Ps = [P0]
    heights = [float(alts[order[0]])]

    x, P = x0, P0
    for k in range(1, n):
        a, b = order[k - 1], order[k]
        dt = float(times[b] - times[a])
        if dt == 0:
            continue

        vario_z = (float(pressure_alts[b]) - float(pressure_alts[a])) / dt if use_vario else None
        gps_z = np.array([x_local[b], y_local[b], alts[b]])

        x, P = _step(x, P, dt, sigma, gps_z, vario_z)
        xs.append(x.copy())
        Ps.append(P.copy())
        heights.append(float(alts[b]))

    return xs, Ps, heights


# ── Ground trigger (EKF_MODEL.md §8, without covariance propagation) ──

def _sample_elevation(arr: np.ndarray, transform, lon: float, lat: float) -> float | None:
    """Sample elevation from a pre-loaded DEM array at (lon, lat)."""
    col, row = ~transform * (lon, lat)
    row = max(0, min(int(round(row)), arr.shape[0] - 1))
    col = max(0, min(int(round(col)), arr.shape[1] - 1))
    elev = float(arr[row, col])
    return elev if elev > 0 else None


def _find_ground_trigger(cx_bottom: float, cy_bottom: float, s_x: float, s_y: float,
                          h_bottom: float, lon_ref: float, lat_ref: float,
                          m_per_deg_lon: float, m_per_deg_lat: float,
                          arr: np.ndarray, transform,
                          max_drop: float = 800.0, tol: float = 1.0, max_iter: int = 40):
    """Extrapolate C(h) = C_bottom + s*(h - h_bottom) below h_bottom and bisect
    for h_g such that h_g == DEM(C(h_g)). Returns (lon, lat, ground_elev) or None.

    ``arr`` and ``transform`` are a pre-loaded DEM tile covering the bisection
    path; pass the result of a single ``get_dem_array`` call from the caller."""

    def center_at(h: float):
        dh = h - h_bottom
        x = cx_bottom + s_x * dh
        y = cy_bottom + s_y * dh
        return lon_ref + x / m_per_deg_lon, lat_ref + y / m_per_deg_lat

    lon_hi, lat_hi = center_at(h_bottom)
    elev_hi = _sample_elevation(arr, transform, lon_hi, lat_hi)
    if elev_hi is None:
        return None

    if h_bottom <= elev_hi:
        # Segment bottom is already at/below terrain — use it directly.
        return lon_hi, lat_hi, float(elev_hi)

    h_lo = h_bottom - max_drop
    lon_lo, lat_lo = center_at(h_lo)
    elev_lo = _sample_elevation(arr, transform, lon_lo, lat_lo)
    if elev_lo is None or h_lo > elev_lo:
        return None  # terrain doesn't rise into the extrapolated line within max_drop

    lo, hi = h_lo, h_bottom
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        lon_m, lat_m = center_at(mid)
        elev_m = _sample_elevation(arr, transform, lon_m, lat_m)
        if elev_m is None:
            return None
        g_mid = mid - elev_m
        if abs(g_mid) < tol:
            lo = hi = mid
            break
        if g_mid > 0:
            hi = mid
        else:
            lo = mid

    h_g = 0.5 * (lo + hi)
    lon_g, lat_g = center_at(h_g)
    elev_g = _sample_elevation(arr, transform, lon_g, lat_g)
    return lon_g, lat_g, float(elev_g if elev_g is not None else h_g)


# ── Public entry point ──

def estimate_centerline_ekf(
    coords: list[list[float]],
    times: list[float],
    pressure_alts: list[float] | None,
    start_idx: int,
    end_idx: int,
) -> dict:
    """Estimate a thermal's centerline via EKF (Phase 1: H1 GPS + H2 vario).

    Args mirror analyze_thermal_segment(): coords are the full track's
    [[lon, lat, alt], ...], times the full track's epoch-second timestamps,
    pressure_alts the full track's baro altitudes (may be all-zero/None if
    unavailable, in which case the vario update is skipped).

    Returns the same core_line / trigger_point / ground_elevation contract as
    analyze_thermal_segment, plus EKF-derived avg_climb_rate, altitude_gain,
    n_turns, drift_bearing, drift_speed.
    """
    seg_coords = coords[start_idx:end_idx + 1]
    seg_times = times[start_idx:end_idx + 1]
    seg_p_alts = (pressure_alts if pressure_alts is not None else [0.0] * len(coords))[start_idx:end_idx + 1]

    if len(seg_coords) < _MIN_FIXES:
        raise ValueError("Segment too short for analysis (need at least 10 fixes).")

    lons = np.array([c[0] for c in seg_coords])
    lats = np.array([c[1] for c in seg_coords])
    alts = np.array([c[2] for c in seg_coords], dtype=float)
    ts = np.array(seg_times, dtype=float)
    p_alts = np.array(seg_p_alts, dtype=float)

    altitude_gain = float(alts.max() - alts.min())
    if altitude_gain < 10:
        raise ValueError("Altitude gain too small for meaningful analysis.")

    x_local, y_local, lon_ref, lat_ref, m_per_deg_lon, m_per_deg_lat = _to_local_xy(lons, lats)

    x0, P0, sigma = _init_state(x_local, y_local, alts, ts)
    xs, _Ps, heights = _run_filter(x_local, y_local, alts, ts, p_alts, x0, P0, sigma)

    xs_arr = np.array(xs)
    heights_arr = np.array(heights)

    core_lons, core_lats = _from_local_xy(xs_arr[:, C_X], xs_arr[:, C_Y], lon_ref, lat_ref,
                                           m_per_deg_lon, m_per_deg_lat)

    x_bottom, h_bottom = xs_arr[-1], heights_arr[-1]

    # Fetch DEM once for the whole segment (covers the bisection path without
    # re-fetching per bisection step; reuses the cache key from the overlay).
    dem_arr = dem_transform = None
    try:
        from app.services.dem_source import get_dem_array
        _pad = 0.02
        seg_bbox = (
            float(lons.min()) - _pad, float(lats.min()) - _pad,
            float(lons.max()) + _pad, float(lats.max()) + _pad,
        )
        dem_arr, dem_transform, _ = get_dem_array(seg_bbox, res_m=30)
    except Exception as _exc:
        logger.warning("DEM fetch failed for segment bbox: %s", _exc)

    trigger = None
    if dem_arr is not None:
        trigger = _find_ground_trigger(
            x_bottom[C_X], x_bottom[C_Y], x_bottom[S_X], x_bottom[S_Y], h_bottom,
            lon_ref, lat_ref, m_per_deg_lon, m_per_deg_lat,
            dem_arr, dem_transform,
        )

    core_coords = []
    if trigger is not None:
        lon_g, lat_g, ground_elev = trigger
        core_coords.append([float(lon_g), float(lat_g), float(ground_elev)])
    for lon, lat, h in zip(core_lons[::-1], core_lats[::-1], heights_arr[::-1]):
        core_coords.append([float(lon), float(lat), float(h)])

    core_line = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": core_coords},
        "properties": {
            "type": "thermal_core_ekf",
            "ground_elevation": float(trigger[2]) if trigger else None,
            "alt_min": float(alts.min()),
            "alt_max": float(alts.max()),
            "trigger_point": [float(trigger[0]), float(trigger[1])] if trigger else None,
        },
    }

    duration = float(ts[-1] - ts[0])
    avg_climb_rate = altitude_gain / duration if duration > 0 else 0.0

    phi_unwrapped = np.unwrap(xs_arr[:, PHI])
    n_turns = abs(phi_unwrapped[-1] - phi_unwrapped[0]) / (2 * math.pi)

    drift_x = float(x_bottom[S_X] * altitude_gain)
    drift_y = float(x_bottom[S_Y] * altitude_gain)
    drift_dist = math.hypot(drift_x, drift_y)
    drift_speed = drift_dist / duration if duration > 0 else 0.0
    drift_bearing = (math.degrees(math.atan2(drift_x, drift_y)) + 360) % 360

    return {
        "core_line": core_line,
        "avg_climb_rate": float(avg_climb_rate),
        "altitude_gain": altitude_gain,
        "n_turns": round(float(n_turns), 1),
        "drift_bearing": float(drift_bearing),
        "drift_speed": float(drift_speed),
        "ground_elevation": float(trigger[2]) if trigger else None,
        "trigger_point": [float(trigger[0]), float(trigger[1])] if trigger else None,
    }
