"""
Per-cell triangle normals from a regular DEM grid.

Each cell (i, j) is split into two triangles:
    upper-left  = [(i,j), (i+1,j), (i,j+1)]
    lower-right = [(i+1,j+1), (i,j+1), (i+1,j)]

Returns averaged per-cell normal (H-1, W-1, 3).
"""
from __future__ import annotations

import numpy as np
import rasterio
from typing import Tuple

from app.cache import normals_cache


def compute_normals(
    dem: np.ndarray,
    transform: rasterio.transform.Affine,
) -> np.ndarray:
    """
    Compute per-cell averaged face normals.

    Parameters
    ----------
    dem : (H, W) float array, elevation in metres (UTM grid).
    transform : rasterio Affine for the UTM grid.

    Returns
    -------
    normals : (H-1, W-1, 3) float32 array, unit vectors in ENU.
    """
    H, W = dem.shape
    # pixel spacing in metres (UTM grid, so direct metric)
    dx = transform.a  # positive east
    dy = -transform.e  # positive north (rasterio uses negative e for north-up)

    rows = H - 1
    cols = W - 1

    # Build vertex grids: shape (H, W, 3)
    xi = np.arange(W, dtype=np.float64)
    yi = np.arange(H, dtype=np.float64)
    xx, yy = np.meshgrid(xi, yi)
    # UTM x increases east, y increases north (row index increases south)
    vx = xx * dx
    vy = (H - 1 - yy) * dy  # flip so row 0 = north
    vz = dem.astype(np.float64)

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
) -> np.ndarray:
    cached = normals_cache.get(bbox_key)
    if cached is not None:
        return cached
    n = compute_normals(dem, transform)
    normals_cache.set(bbox_key, n)
    return n
