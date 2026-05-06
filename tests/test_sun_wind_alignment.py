"""Tests for sun/wind vector math and polar-to-ENU conversions."""
from __future__ import annotations

import math
import pytest

from app.services.sun import _az_alt_to_enu
from app.services.wind import _polar_to_uv, _unit_vec


class TestSunVector:
    def test_sun_due_south_horizon(self):
        """Azimuth 180 (south), elevation 0 → ENU (0, -1, 0)."""
        e, n, u = _az_alt_to_enu(180.0, 0.0)
        assert abs(e) < 1e-6
        assert n < -0.99
        assert abs(u) < 1e-6

    def test_sun_overhead(self):
        """Azimuth any, elevation 90 → ENU (0, 0, 1)."""
        e, n, u = _az_alt_to_enu(45.0, 90.0)
        assert abs(e) < 1e-6
        assert abs(n) < 1e-6
        assert abs(u - 1.0) < 1e-6

    def test_sun_due_east_horizon(self):
        """Azimuth 90 (east), elevation 0 → ENU (1, 0, 0)."""
        e, n, u = _az_alt_to_enu(90.0, 0.0)
        assert abs(e - 1.0) < 1e-6
        assert abs(n) < 1e-6
        assert abs(u) < 1e-6

    def test_sun_below_horizon_negative_up(self):
        """Elevation < 0 → up component < 0."""
        _, _, u = _az_alt_to_enu(180.0, -10.0)
        assert u < 0

    def test_unit_length(self):
        for az in range(0, 360, 30):
            for el in range(-10, 91, 10):
                e, n, u = _az_alt_to_enu(float(az), float(el))
                length = math.sqrt(e**2 + n**2 + u**2)
                assert abs(length - 1.0) < 1e-5, f"az={az} el={el} length={length}"


class TestWindVector:
    def test_north_wind_blows_south(self):
        """Wind FROM north (dir=0) blows toward south → u=0, v < 0."""
        u, v = _polar_to_uv(10.0, 0.0)
        assert abs(u) < 1e-6
        assert v < 0

    def test_south_wind_blows_north(self):
        """Wind FROM south (dir=180) blows toward north → u=0, v > 0."""
        u, v = _polar_to_uv(10.0, 180.0)
        assert abs(u) < 1e-6
        assert v > 0

    def test_east_wind_blows_west(self):
        """Wind FROM east (dir=90) blows toward west → u < 0, v ≈ 0."""
        u, v = _polar_to_uv(10.0, 90.0)
        assert u < 0
        assert abs(v) < 1e-6

    def test_west_wind_blows_east(self):
        """Wind FROM west (dir=270) blows toward east → u > 0, v ≈ 0."""
        u, v = _polar_to_uv(10.0, 270.0)
        assert u > 0
        assert abs(v) < 1e-6

    def test_speed_preserved(self):
        """Speed should equal sqrt(u²+v²)."""
        for spd in [0.0, 1.0, 5.0, 15.3]:
            u, v = _polar_to_uv(spd, 45.0)
            reconstructed = math.sqrt(u**2 + v**2)
            assert abs(reconstructed - spd) < 1e-5

    def test_unit_vec_zero(self):
        """Zero wind → zero vector."""
        vec = _unit_vec(0.0, 0.0)
        assert vec == [0.0, 0.0, 0.0]

    def test_unit_vec_length_one(self):
        """Non-zero wind → unit vector."""
        vec = _unit_vec(3.0, 4.0)
        length = math.sqrt(sum(x**2 for x in vec))
        assert abs(length - 1.0) < 1e-6
