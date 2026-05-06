from __future__ import annotations
from pydantic import BaseModel
from typing import List, Tuple, Optional


BBox = Tuple[float, float, float, float]  # W, S, E, N


class SunState(BaseModel):
    azimuth: float          # degrees, 0 = N, clockwise
    elevation: float        # degrees above horizon
    ghi: float              # W/m², clear-sky global horizontal irradiance
    sun_vec: List[float]    # ENU unit vector [east, north, up]


class WindLevel(BaseModel):
    level: str              # "10m", "925hPa", etc.
    speed: float            # m/s
    direction: float        # degrees from North, "direction from"
    u: float                # m/s east component
    v: float                # m/s north component


class WindState(BaseModel):
    levels: List[WindLevel]
    mean_u: float           # weighted mean east component
    mean_v: float           # weighted mean north component
    mean_vec: List[float]   # ENU unit vector [east, north, 0]


class NormalsResponse(BaseModel):
    bbox: List[float]
    shape: List[int]        # [H, W]
    # normals flattened row-major: each element is [nx, ny, nz]
    normals: List[List[float]]


class OverlayInfo(BaseModel):
    bbox: List[float]
    width: int
    height: int
