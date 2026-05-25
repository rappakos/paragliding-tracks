from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from app.config import settings
from app.services.dem_source import get_dem_array, _quantise_bbox
from app.services.triangulation import get_normals_cached
from app.models.schemas import NormalsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dem", tags=["dem"])


def _parse_bbox(bbox_str: str):
    try:
        parts = [float(x) for x in bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return tuple(parts)  # W, S, E, N
    except Exception:
        raise HTTPException(status_code=422, detail="bbox must be W,S,E,N floats")


@router.get("")
async def get_dem_endpoint(
    bbox: str = Query(default=None, description="W,S,E,N in WGS84"),
    res: int = Query(default=30, description="DEM resolution in metres"),
):
    """Return hillshade PNG for the given bbox."""
    if bbox is None:
        bbox = settings.default_bbox
    bb = _parse_bbox(bbox)

    try:
        dem_arr, transform, crs = get_dem_array(bb, res)
    except Exception as exc:
        logger.error("DEM fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Simple hillshade for visual confirmation
    import numpy as np
    from app.services.overlay import _to_png

    # Normalise to 0-255 greyscale
    mn, mx = dem_arr.min(), dem_arr.max()
    if mx > mn:
        grey = ((dem_arr - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        grey = np.full_like(dem_arr, 128, dtype=np.uint8)

    import io
    from PIL import Image
    img = Image.fromarray(grey, mode="L").convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    west, south, east, north = bb
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-BBox": f"{west},{south},{east},{north}",
            "X-Width": str(dem_arr.shape[1]),
            "X-Height": str(dem_arr.shape[0]),
        },
    )


@router.get("/normals")
async def get_normals_endpoint(
    bbox: str = Query(default=None, description="W,S,E,N in WGS84"),
):
    """Return per-cell normals as JSON."""
    if bbox is None:
        bbox = settings.default_bbox
    bb = _parse_bbox(bbox)

    try:
        dem_arr, transform, crs = get_dem_array(bb)
    except Exception as exc:
        logger.error("DEM fetch failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    qbbox = _quantise_bbox(bb)
    normals = get_normals_cached(("normals", qbbox), dem_arr, transform)

    H, W, _ = normals.shape
    flat = normals.reshape(-1, 3).tolist()

    return NormalsResponse(
        bbox=list(bb),
        shape=[H, W],
        normals=flat,
    )
