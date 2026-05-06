#!/usr/bin/env python3
"""
One-off script to pre-populate the DEM cache for the default bbox.
Run: python -m scripts.fetch_dem
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.config import settings
from app.services.dem_source import get_dem_array
from app.services.triangulation import get_normals_cached, compute_normals
from app.services.dem_source import _quantise_bbox


def main():
    bbox = settings.bbox_tuple()
    print(f"Fetching DEM for bbox {bbox} …")
    dem_arr, transform, crs = get_dem_array(bbox, settings.dem_res_m)
    print(f"DEM shape: {dem_arr.shape}, CRS: {crs}")

    qbbox = _quantise_bbox(bbox)
    normals = get_normals_cached(("normals", qbbox), dem_arr, transform)
    print(f"Normals shape: {normals.shape}")
    print("Done. Cache populated.")


if __name__ == "__main__":
    main()
