"""
Per-cell triangle normals from a regular DEM grid.

Each cell (i, j) is split into two triangles:
    upper-left  = [(i,j), (i+1,j), (i,j+1)]
    lower-right = [(i+1,j+1), (i,j+1), (i+1,j)]

Returns averaged per-cell normal (H-1, W-1, 3).

Supports both UTM (metric) and WGS84 (degree) grids.
For WGS84, pixel spacing is converted to metres using latitude correction.
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter
from typing import Tuple

from app.cache import normals_cache

# Earth radius for degree→metre conversion
_DEG_TO_M_LAT = 111_320.0  # metres per degree latitude (approx)


def compute_normals(
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    crs=None,
) -> np.ndarray:
    """
    Compute per-cell averaged face normals.

    Parameters
    ----------
    dem : (H, W) float array, elevation in metres.
    transform : rasterio Affine for the grid.
    crs : optional CRS. If geographic (EPSG:4326), pixel spacing is
          converted to metres using latitude correction.

    Returns
    -------
    normals : (H-1, W-1, 3) float32 array, unit vectors in ENU.
    """
    H, W = dem.shape
    # Smooth DEM to remove per-pixel noise and staircase artifacts
    dem_smooth = gaussian_filter(dem.astype(np.float64), sigma=1.5)

    is_geographic = crs is not None and crs.is_geographic

    if is_geographic:
        # WGS84 grid: transform.a = pixel width in degrees (longitude)
        #             transform.e = pixel height in degrees (negative for north-up)
        dlon_deg = transform.a
        dlat_deg = -transform.e  # positive

        # Compute latitude for each row (center of pixel)
        # Row 0 is north (transform.f = north edge latitude)
        row_indices = np.arange(H, dtype=np.float64)
        lat_per_row = transform.f + transform.e * (row_indices + 0.5)  # degrees

        # Metric spacing per row
        cos_lat = np.cos(np.radians(lat_per_row))
        dx_per_row = dlon_deg * cos_lat * _DEG_TO_M_LAT  # metres per pixel in x
        dy = dlat_deg * _DEG_TO_M_LAT  # metres per pixel in y (constant)
    else:
        # UTM or other metric CRS: direct metric spacing
        dx_per_row = None
        dx = transform.a  # positive east
        dy = -transform.e  # positive north

    rows = H - 1
    cols = W - 1

    # Build vertex grids: shape (H, W, 3)
    xi = np.arange(W, dtype=np.float64)
    yi = np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(xi, yi)

    if is_geographic:
        # x spacing varies by row
        # vx[row, col] = col * dx_per_row[row]
        vx = xx * dx_per_row[:, np.newaxis]
    else:
        vx = xx * dx

    vy = (H - 1 - yy) * dy  # flip so row 0 = north
    vz = dem_smooth

    # Triangle A: (i,j), (i+1,j), (i,j+1)  (upper-left)
    p0 = np.stack([vx[:rows, :cols], vy[:rows, :cols], vz[:rows, :cols]], axis=-1)
    p1 = np.stack([vx[1:rows+1, :cols], vy[1:rows+1, :cols], vz[1:rows+1, :cols]], axis=-1)
    p2 = np.stack([vx[:rows, 1:cols+1], vy[:rows, 1:cols+1], vz[:rows, 1:cols+1]], axis=-1)
    nA = np.cross(p1 - p0, p2 - p0)

    # Triangle B: (i+1,j+1), (i,j+1), (i+1,j)  (lower-right)
    q0 = np.stack([vx[1:rows+1, 1:cols+1], vy[1:rows+1, 1:cols+1], vz[1:rows+1, 1:cols+1]], axis=-1)
    q1 = np.stack([vx[:rows, 1:cols+1], vy[:rows, 1:cols+1], vz[:rows, 1:cols+1]], axis=-1)
    q2 = np.stack([vx[1:rows+1, :cols], vy[1:rows+1, :cols], vz[1:rows+1, :cols]], axis=-1)
    nB = np.cross(q1 - q0, q2 - q0)

    # Average and ensure upward-pointing
    n = nA + nB
    # Flip if pointing downward
    flip = n[..., 2] < 0
    n[flip] *= -1

    # Normalise
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    mag = np.where(mag < 1e-10, 1.0, mag)
    n = (n / mag).astype(np.float32)

    return n


def get_normals_cached(
    bbox_key: tuple,
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
    crs=None,
) -> np.ndarray:
    cached = normals_cache.get(bbox_key)
    if cached is not None:
        return cached
    n = compute_normals(dem, transform, crs)
    normals_cache.set(bbox_key, n)
    return n
