"""Tests for overlay rendering math."""
from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image

from app.services.overlay import thermal_overlay, wind_overlay, _colormap_yellow_red, _colormap_diverging


def _flat_normals(h: int = 20, w: int = 20) -> np.ndarray:
    """Return (h, w, 3) array of upward-pointing normals."""
    n = np.zeros((h, w, 3), dtype=np.float32)
    n[..., 2] = 1.0
    return n


def _tilted_normals(h: int = 20, w: int = 20, tilt_east: float = 0.5) -> np.ndarray:
    """Return normals tilted in the east direction."""
    nz = np.sqrt(1.0 - tilt_east ** 2)
    n = np.zeros((h, w, 3), dtype=np.float32)
    n[..., 0] = tilt_east
    n[..., 2] = nz
    return n


class TestThermalOverlay:
    def test_returns_bytes(self):
        n = _flat_normals()
        sun_vec = [0.0, 0.0, 1.0]  # sun directly overhead
        result = thermal_overlay(n, sun_vec, ghi=800.0)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_valid_png(self):
        n = _flat_normals()
        result = thermal_overlay(n, [0.0, 0.0, 1.0], ghi=800.0)
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"
        assert img.size == (20, 20)

    def test_overhead_sun_maximum_drive(self):
        """With sun directly overhead and flat terrain, all cells should have max drive."""
        n = _flat_normals()
        result = thermal_overlay(n, [0.0, 0.0, 1.0], ghi=1000.0)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # Red channel should be 255 (max) for full thermal drive
        assert arr[..., 0].mean() > 200

    def test_night_no_drive(self):
        """With GHI=0 (night), overlay should be fully transparent."""
        n = _flat_normals()
        result = thermal_overlay(n, [0.0, 0.0, 1.0], ghi=0.0)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # Alpha channel should be 0 (transparent)
        assert arr[..., 3].max() == 0

    def test_below_horizon_no_drive(self):
        """With sun below horizon (sun_vec pointing down), thermal drive should be zero."""
        n = _flat_normals()
        result = thermal_overlay(n, [0.0, 0.0, -1.0], ghi=800.0)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        assert arr[..., 3].max() == 0


class TestWindOverlay:
    def test_returns_bytes(self):
        n = _flat_normals()
        result = wind_overlay(n, [1.0, 0.0, 0.0])
        assert isinstance(result, bytes)

    def test_valid_png(self):
        n = _flat_normals()
        result = wind_overlay(n, [1.0, 0.0, 0.0])
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"
        assert img.size == (20, 20)

    def test_zero_wind_transparent(self):
        """Zero wind vector → no exposure → transparent overlay."""
        n = _flat_normals()
        result = wind_overlay(n, [0.0, 0.0, 0.0])
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        assert arr[..., 3].max() == 0

    def test_windward_red_channel(self):
        """Wind hitting a face head-on → windward → red."""
        # Normal pointing east, wind from east (unit vec pointing west into face)
        n = np.zeros((5, 5, 3), dtype=np.float32)
        n[..., 0] = 1.0   # normal pointing east
        # Wind coming from the east → unit vec is (-1, 0, 0) (wind goes west)
        # But we store "toward" direction: wind FROM east → blowing WEST = (-1, 0, 0)
        wind_vec = [-1.0, 0.0, 0.0]
        result = wind_overlay(n, wind_vec)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # Windward: red > blue
        assert arr[..., 0].mean() > arr[..., 2].mean()


class TestColormaps:
    def test_yellow_red_zero(self):
        t = np.array([0.0])
        rgba = _colormap_yellow_red(t)
        np.testing.assert_allclose(rgba[0], [1.0, 1.0, 0.0, 0.0], atol=1e-5)  # yellow, transparent

    def test_yellow_red_one(self):
        t = np.array([1.0])
        rgba = _colormap_yellow_red(t)
        np.testing.assert_allclose(rgba[0], [1.0, 0.0, 0.0, 1.0], atol=1e-5)  # red, opaque

    def test_diverging_windward(self):
        t = np.array([-1.0])  # windward
        rgba = _colormap_diverging(t)
        assert rgba[0, 0] > rgba[0, 2]  # red > blue

    def test_diverging_lee(self):
        t = np.array([1.0])   # lee
        rgba = _colormap_diverging(t)
        assert rgba[0, 2] > rgba[0, 0]  # blue > red
