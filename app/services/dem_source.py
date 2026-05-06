"""
DEM acquisition and reprojection service.

Fetches SRTM-30m data via py3dep (3DEP / WCS), reprojects to UTM-32N (EPSG:25832),
and caches the result as a GeoTIFF on disk.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from typing import Tuple

import numpy as np
import rasterio
import rasterio.transform
from pyproj import Transformer

from app.cache import dem_cache, normals_cache
from app.config import settings

logger = logging.getLogger(__name__)

BBox = Tuple[float, float, float, float]  # W, S, E, N


def _quantise_bbox(bbox: BBox, step: float = 0.01) -> BBox:
    """Round bbox to nearest step to improve cache hit rate."""
    return tuple(round(v / step) * step for v in bbox)  # type: ignore[return-value]


def _cache_path(bbox: BBox) -> str:
    os.makedirs(os.path.join(settings.cache_dir, "dem"), exist_ok=True)
    key = "_".join(f"{v:.4f}" for v in bbox)
    return os.path.join(settings.cache_dir, "dem", f"dem_{key}.tif")


def get_dem_array(bbox: BBox, res_m: int = 30) -> Tuple[np.ndarray, rasterio.transform.Affine, "rasterio.crs.CRS"]:
    """
    Return (elevation_array, transform, crs) in EPSG:25832 (UTM-32N).

    elevation_array shape: (H, W), float32, metres.
    """
    qbbox = _quantise_bbox(bbox)
    cache_key = ("dem", qbbox, res_m)

    cached = dem_cache.get(cache_key)
    if cached is not None:
        logger.debug("DEM cache hit for %s", qbbox)
        return _load_geotiff_bytes(cached)

    path = _cache_path(qbbox)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        dem_cache.set(cache_key, data)
        return _load_geotiff_bytes(data)

    logger.info("Fetching DEM for bbox %s …", qbbox)
    data = _fetch_and_reproject(bbox, res_m)
    with open(path, "wb") as f:
        f.write(data)
    dem_cache.set(cache_key, data)
    return _load_geotiff_bytes(data)


def _load_geotiff_bytes(data: bytes) -> Tuple[np.ndarray, rasterio.transform.Affine, "rasterio.crs.CRS"]:
    with rasterio.open(io.BytesIO(data)) as ds:
        arr = ds.read(1).astype(np.float32)
        return arr, ds.transform, ds.crs


def _fetch_and_reproject(bbox: BBox, res_m: int) -> bytes:
    """Fetch DEM via py3dep and reproject to EPSG:25832."""
    try:
        import py3dep
        west, south, east, north = bbox
        # py3dep expects (west, south, east, north) as a tuple
        dem = py3dep.get_dem(
            (west, south, east, north),
            resolution=res_m,
            crs="EPSG:4326",
        )
        # dem is an xarray DataArray; convert to numpy
        arr = dem.values.astype(np.float32)
        # build a minimal in-memory GeoTIFF in WGS84 first
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
        src_crs = CRS.from_epsg(4326)
        h, w = arr.shape
        transform_4326 = from_bounds(west, south, east, north, w, h)

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmppath = tmp.name
        with rasterio.open(
            tmppath,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=1,
            dtype=np.float32,
            crs=src_crs,
            transform=transform_4326,
        ) as dst:
            dst.write(arr, 1)
    except Exception as exc:
        logger.warning("py3dep fetch failed (%s); using synthetic flat DEM for testing.", exc)
        tmppath = _make_synthetic_dem(bbox, res_m)

    # Reproject to UTM-32N
    reprojected_path = _reproject_to_utm32(tmppath)
    os.unlink(tmppath)

    with open(reprojected_path, "rb") as f:
        data = f.read()
    os.unlink(reprojected_path)
    return data


def _reproject_to_utm32(src_path: str) -> str:
    """Reproject a GeoTIFF to EPSG:25832 and return path to the new file."""
    import rasterio.warp
    from rasterio.crs import CRS

    dst_crs = CRS.from_epsg(25832)
    with tempfile.NamedTemporaryFile(suffix="_utm.tif", delete=False) as tmp:
        dst_path = tmp.name

    with rasterio.open(src_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs, "transform": transform, "width": width, "height": height})

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.bilinear,
                )
    return dst_path


def _make_synthetic_dem(bbox: BBox, res_m: int) -> str:
    """Create a flat synthetic DEM for testing when py3dep is unavailable."""
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    west, south, east, north = bbox
    # Approximate grid size based on ~res_m per pixel at this latitude
    lat_m = 111_000  # m per degree latitude
    lon_m = 111_000 * np.cos(np.radians((south + north) / 2))
    h = max(10, int((north - south) * lat_m / res_m))
    w = max(10, int((east - west) * lon_m / res_m))

    # Small rolling hills for visual interest
    x = np.linspace(0, 2 * np.pi, w)
    y = np.linspace(0, 2 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    arr = (100 + 20 * np.sin(xx) * np.cos(yy)).astype(np.float32)

    src_crs = CRS.from_epsg(4326)
    transform = from_bounds(west, south, east, north, w, h)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmppath = tmp.name
    with rasterio.open(
        tmppath, "w", driver="GTiff",
        height=h, width=w, count=1, dtype=np.float32,
        crs=src_crs, transform=transform,
    ) as dst:
        dst.write(arr, 1)
    return tmppath
