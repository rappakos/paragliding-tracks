"""Tests for triangulation / per-cell normal computation."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio.transform

from app.services.triangulation import compute_normals


def _flat_dem(h: int = 10, w: int = 10, elev: float = 100.0):
    dem = np.full((h, w), elev, dtype=np.float32)
    transform = rasterio.transform.from_origin(400000, 5800000, 30, 30)
    return dem, transform


def _slope_dem_east(h: int = 10, w: int = 15, rise_per_cell: float = 1.0):
    """DEM that slopes upward toward the east (+x direction)."""
    dem = np.zeros((h, w), dtype=np.float32)
    for j in range(w):
        dem[:, j] = j * rise_per_cell
    transform = rasterio.transform.from_origin(400000, 5800000, 30, 30)
    return dem, transform


def _slope_dem_north(h: int = 15, w: int = 10, rise_per_cell: float = 1.0):
    """DEM that slopes upward toward the north (decreasing row index)."""
    dem = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        dem[i, :] = (h - 1 - i) * rise_per_cell
    transform = rasterio.transform.from_origin(400000, 5800000, 30, 30)
    return dem, transform


class TestFlatDEM:
    def test_shape(self):
        dem, transform = _flat_dem(10, 12)
        normals = compute_normals(dem, transform)
        assert normals.shape == (9, 11, 3)

    def test_flat_normals_point_up(self):
        dem, transform = _flat_dem(10, 10)
        n = compute_normals(dem, transform)
        # All normals should be ~(0, 0, 1)
        np.testing.assert_allclose(n[..., 0], 0.0, atol=1e-5)
        np.testing.assert_allclose(n[..., 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(n[..., 2], 1.0, atol=1e-5)

    def test_normals_unit_length(self):
        dem, transform = _flat_dem(8, 8)
        n = compute_normals(dem, transform)
        lengths = np.linalg.norm(n, axis=-1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


class TestSlopedDEM:
    def test_east_slope_negative_east_component(self):
        """East-sloping terrain: normal tilts westward (negative east component)."""
        dem, transform = _slope_dem_east(rise_per_cell=5.0)
        n = compute_normals(dem, transform)
        # The normal should have a non-zero east component (negative = tilts west)
        assert n[..., 0].mean() < 0, "East-facing slope should have negative east normal component"

    def test_north_slope_positive_north_component(self):
        """North-sloping terrain (high elevation north): normal tilts southward (negative north component)."""
        dem, transform = _slope_dem_north(rise_per_cell=5.0)
        n = compute_normals(dem, transform)
        # Slope rising to north → surface normal tilts south (negative north component)
        assert n[..., 1].mean() < 0, "North-rising slope should have negative north normal component"

    def test_normals_always_upward(self):
        dem, transform = _slope_dem_east(rise_per_cell=10.0)
        n = compute_normals(dem, transform)
        assert np.all(n[..., 2] > 0), "All normals should point upward (z > 0)"

    def test_normals_unit_length_sloped(self):
        dem, transform = _slope_dem_east(rise_per_cell=3.0)
        n = compute_normals(dem, transform)
        lengths = np.linalg.norm(n, axis=-1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)
