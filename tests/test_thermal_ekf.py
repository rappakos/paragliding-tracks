"""Tests for the EKF thermal-centerline estimator (EKF_MODEL.md, Phase 1)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.services import thermal_ekf
from app.services.thermal_ekf import (
    V_TAN,
    W_SINK,
    _fit_circle,
    _turn_direction,
    _find_ground_trigger,
    estimate_centerline_ekf,
)


def _make_synthetic_thermal(
    n=180,
    dt=1.0,
    lon_ref0=9.5,
    lat_ref0=51.9,
    cx0=0.0,
    cy0=0.0,
    radius=60.0,
    sigma=1.0,
    s_x=0.3,
    s_y=-0.1,
    true_w=3.0,
    w_sink=W_SINK,
    h_bottom=800.0,
    phi0=0.0,
):
    """Noiseless circling-climb segment: constant R/w/lean, so the process
    model is satisfied exactly and Euler integration introduces no error."""
    net_climb = true_w - w_sink
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_ref0))

    t = np.arange(n) * dt
    h = h_bottom + net_climb * t
    phi = phi0 + sigma * V_TAN / radius * t
    cx = cx0 + s_x * (h - h_bottom)
    cy = cy0 + s_y * (h - h_bottom)
    x = cx + radius * np.cos(phi)
    y = cy + radius * np.sin(phi)

    lons = lon_ref0 + x / m_per_deg_lon
    lats = lat_ref0 + y / m_per_deg_lat

    coords = [[float(lo), float(la), float(al)] for lo, la, al in zip(lons, lats, h)]
    times = [float(tt) for tt in t]
    pressure_alts = [float(al) for al in h]

    truth = dict(
        lon_ref0=lon_ref0, lat_ref0=lat_ref0,
        m_per_deg_lon=m_per_deg_lon, m_per_deg_lat=m_per_deg_lat,
        cx0=cx0, cy0=cy0, s_x=s_x, s_y=s_y, h_bottom=h_bottom, h_top=float(h[-1]),
    )
    return coords, times, pressure_alts, truth


def _true_center_lonlat(truth, h):
    cx = truth["cx0"] + truth["s_x"] * (h - truth["h_bottom"])
    cy = truth["cy0"] + truth["s_y"] * (h - truth["h_bottom"])
    lon = truth["lon_ref0"] + cx / truth["m_per_deg_lon"]
    lat = truth["lat_ref0"] + cy / truth["m_per_deg_lat"]
    return lon, lat


def _lonlat_dist_m(lon1, lat1, lon2, lat2, m_per_deg_lon, m_per_deg_lat):
    return math.hypot((lon1 - lon2) * m_per_deg_lon, (lat1 - lat2) * m_per_deg_lat)


class TestCircleFit:
    def test_recovers_exact_circle(self):
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        cx_true, cy_true, r_true = 12.0, -7.0, 55.0
        x = cx_true + r_true * np.cos(theta)
        y = cy_true + r_true * np.sin(theta)

        cx, cy, r = _fit_circle(x, y)
        assert abs(cx - cx_true) < 1e-6
        assert abs(cy - cy_true) < 1e-6
        assert abs(r - r_true) < 1e-6

    def test_turn_direction_ccw(self):
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        x, y = 50 * np.cos(theta), 50 * np.sin(theta)
        assert _turn_direction(x, y, 0.0, 0.0) == 1.0

    def test_turn_direction_cw(self):
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        x, y = 50 * np.cos(-theta), 50 * np.sin(-theta)
        assert _turn_direction(x, y, 0.0, 0.0) == -1.0


class TestGroundTrigger:
    def test_flat_terrain_bisection(self, monkeypatch):
        flat_elev = 650.0
        monkeypatch.setattr(thermal_ekf, "_ground_elevation", lambda lon, lat: flat_elev)

        lon, lat, elev = _find_ground_trigger(
            cx_bottom=0.0, cy_bottom=0.0, s_x=0.3, s_y=-0.1, h_bottom=800.0,
            lon_ref=9.5, lat_ref=51.9, m_per_deg_lon=68000.0, m_per_deg_lat=111320.0,
        )
        assert abs(elev - flat_elev) < 1.0

        dh = flat_elev - 800.0  # negative
        expected_lon = 9.5 + (0.3 * dh) / 68000.0
        expected_lat = 51.9 + (-0.1 * dh) / 111320.0
        assert abs(lon - expected_lon) < 1e-5
        assert abs(lat - expected_lat) < 1e-5

    def test_no_dem_coverage_returns_none(self, monkeypatch):
        monkeypatch.setattr(thermal_ekf, "_ground_elevation", lambda lon, lat: None)
        result = _find_ground_trigger(
            cx_bottom=0.0, cy_bottom=0.0, s_x=0.0, s_y=0.0, h_bottom=800.0,
            lon_ref=9.5, lat_ref=51.9, m_per_deg_lon=68000.0, m_per_deg_lat=111320.0,
        )
        assert result is None

    def test_terrain_never_reached_returns_none(self, monkeypatch):
        # Terrain sits below the segment bottom minus max_drop everywhere.
        monkeypatch.setattr(thermal_ekf, "_ground_elevation", lambda lon, lat: -10000.0)
        result = _find_ground_trigger(
            cx_bottom=0.0, cy_bottom=0.0, s_x=0.0, s_y=0.0, h_bottom=800.0,
            lon_ref=9.5, lat_ref=51.9, m_per_deg_lon=68000.0, m_per_deg_lat=111320.0,
        )
        assert result is None


class TestEstimateCenterlineEkf:
    def test_recovers_trigger_point_on_flat_terrain(self, monkeypatch):
        coords, times, pressure_alts, truth = _make_synthetic_thermal()
        flat_elev = truth["h_bottom"] - 150.0
        monkeypatch.setattr(thermal_ekf, "_ground_elevation", lambda lon, lat: flat_elev)

        result = estimate_centerline_ekf(coords, times, pressure_alts, 0, len(coords) - 1)

        assert result["ground_elevation"] == pytest.approx(flat_elev, abs=2.0)

        true_lon, true_lat = _true_center_lonlat(truth, flat_elev)
        dist = _lonlat_dist_m(
            result["trigger_point"][0], result["trigger_point"][1],
            true_lon, true_lat,
            truth["m_per_deg_lon"], truth["m_per_deg_lat"],
        )
        assert dist < 15.0

    def test_core_line_bottom_matches_truth(self, monkeypatch):
        coords, times, pressure_alts, truth = _make_synthetic_thermal()
        flat_elev = truth["h_bottom"] - 150.0
        monkeypatch.setattr(thermal_ekf, "_ground_elevation", lambda lon, lat: flat_elev)

        result = estimate_centerline_ekf(coords, times, pressure_alts, 0, len(coords) - 1)

        # core_line goes [ground_trigger, bottom, ..., top]; index 1 is the
        # filtered center at the segment's lowest fix, where the reverse-time
        # filter has had the whole segment to converge.
        coords_out = result["core_line"]["geometry"]["coordinates"]
        bottom_lon, bottom_lat, bottom_alt = coords_out[1]
        assert bottom_alt == pytest.approx(truth["h_bottom"], abs=1.0)

        true_lon, true_lat = _true_center_lonlat(truth, truth["h_bottom"])
        dist = _lonlat_dist_m(
            bottom_lon, bottom_lat, true_lon, true_lat,
            truth["m_per_deg_lon"], truth["m_per_deg_lat"],
        )
        assert dist < 15.0

    def test_raises_on_short_segment(self):
        coords, times, pressure_alts, _truth = _make_synthetic_thermal(n=180)
        with pytest.raises(ValueError, match="too short"):
            estimate_centerline_ekf(coords, times, pressure_alts, 0, 3)

    def test_raises_on_small_altitude_gain(self):
        coords, times, pressure_alts, _truth = _make_synthetic_thermal(true_w=1.0)  # net climb ~0
        with pytest.raises(ValueError, match="Altitude gain"):
            estimate_centerline_ekf(coords, times, pressure_alts, 0, len(coords) - 1)

    def test_works_without_pressure_alts(self, monkeypatch):
        """Vario update should be skippable when baro data is unavailable."""
        coords, times, _pressure_alts, truth = _make_synthetic_thermal()
        flat_elev = truth["h_bottom"] - 150.0
        monkeypatch.setattr(thermal_ekf, "_ground_elevation", lambda lon, lat: flat_elev)

        zero_pressure_alts = [0.0] * len(coords)
        result = estimate_centerline_ekf(coords, times, zero_pressure_alts, 0, len(coords) - 1)
        assert result["trigger_point"] is not None
