"""
Numpy → RGBA PNG overlay rendering.
"""
from __future__ import annotations

import io
import numpy as np
from PIL import Image
from typing import List


def _colormap_yellow_red(t: np.ndarray) -> np.ndarray:
    """Map [0,1] → RGBA (orange=low, magenta=high, good contrast vs green OSM)."""
    t = np.clip(t, 0, 1)
    r = np.ones_like(t, dtype=np.float32)
    g = (0.4 * (1.0 - t)).astype(np.float32)    # orange (1,0.4,0) → magenta (1,0,0.4)
    b = (0.4 * t).astype(np.float32)
    # Minimum alpha 0.4 when t>0 so overlay is clearly visible over map
    a = np.where(t < 0.01, 0.0, 0.4 + 0.6 * t).astype(np.float32)
    return np.stack([r, g, b, a], axis=-1)


def _colormap_diverging(t: np.ndarray) -> np.ndarray:
    """Map [-1,1] → RGBA (red=windward, blue=lee)."""
    t = np.clip(t, -1, 1)
    # positive t → blue (lee), negative t → red (windward)
    r = np.clip(-t, 0, 1).astype(np.float32)
    g = np.zeros_like(t, dtype=np.float32)
    b = np.clip(t, 0, 1).astype(np.float32)
    a = np.abs(t).astype(np.float32)
    return np.stack([r, g, b, a], axis=-1)


def thermal_overlay(
    normals: np.ndarray,   # (H, W, 3) ENU unit vectors
    sun_vec: List[float],  # [east, north, up]
    ghi: float,            # W/m²
) -> bytes:
    """
    Compute thermal driving and encode as RGBA PNG bytes.

    Returns PNG bytes.
    """
    sv = np.array(sun_vec, dtype=np.float32)
    drive = np.einsum("...i,i->...", normals, sv)
    drive = np.nan_to_num(drive, nan=0.0)
    drive = np.maximum(0.0, drive) * max(ghi, 0.0)
    # Remove flat-terrain bias: only show deviation from horizontal surface
    flat_drive = max(float(sv[2]), 0.0) * max(ghi, 0.0)
    drive_rel = drive - flat_drive
    # Normalize: positive = better than flat (thermal trigger)
    max_abs = np.abs(drive_rel).max()
    if max_abs < 1e-6:
        drive_norm = np.zeros_like(drive_rel)
    else:
        drive_norm = np.clip(drive_rel / max_abs, 0, 1)
    rgba_f = _colormap_yellow_red(drive_norm)
    return _to_png(rgba_f)


def wind_overlay(
    normals: np.ndarray,    # (H, W, 3) ENU unit vectors
    wind_vec: List[float],  # [east, north, 0] (unit or scaled)
) -> bytes:
    """
    Compute wind exposure and encode as RGBA PNG bytes.

    n · w < 0 → windward (red), n · w > 0 → lee (blue).
    Returns PNG bytes.
    """
    wv = np.array(wind_vec, dtype=np.float32)
    dot = np.einsum("...i,i->...", normals, wv)
    max_abs = np.abs(dot).max()
    if max_abs < 1e-6:
        dot_norm = np.zeros_like(dot)
    else:
        dot_norm = dot / max_abs
    # dot<0 → windward → negative t → red in _colormap_diverging
    rgba_f = _colormap_diverging(dot_norm)
    return _to_png(rgba_f)


def _to_png(rgba_f: np.ndarray) -> bytes:
    """Convert float32 RGBA [0,1] array to PNG bytes."""
    rgba_u8 = (rgba_f * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgba_u8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
