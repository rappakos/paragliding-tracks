"""
IGC parsing and thermal detection (future).
"""
from __future__ import annotations

import json
from io import StringIO

from aerofiles.igc import Reader as IgcReader


def parse_igc(data: bytes) -> dict:
    """Parse an IGC file and return structured track data.

    Returns dict with keys:
        pilot, glider, start_time, end_time, bbox, geojson
    """
    # aerofiles Reader expects a text-mode file object
    text = data.decode("latin-1")
    reader = IgcReader()
    result = reader.read(StringIO(text))

    # header is [errors_list, header_dict]
    header_dict = result.get("header", [[], {}])[1]

    pilot = header_dict.get("pilot", "")
    glider = header_dict.get("glider_model", "")

    # fix_records is [errors_list, fixes_list]
    fixes = result.get("fix_records", [[], []])[1]
    if not fixes:
        raise ValueError("IGC file contains no GPS fixes (B-records).")

    # Build coordinate list and compute bbox
    coords: list[list[float]] = []
    lats: list[float] = []
    lons: list[float] = []

    for fix in fixes:
        lat = fix.get("lat")
        lon = fix.get("lon")
        alt = fix.get("gps_alt") or fix.get("pressure_alt") or 0
        if lat is None or lon is None:
            continue
        coords.append([lon, lat, alt])
        lats.append(lat)
        lons.append(lon)

    if len(coords) < 2:
        raise ValueError("IGC file contains fewer than 2 valid GPS fixes.")

    # Timestamps — aerofiles provides a combined datetime on each fix
    start_dt = fixes[0].get("datetime")
    end_dt = fixes[-1].get("datetime")

    start_time = start_dt.isoformat() if start_dt else ""
    end_time = end_dt.isoformat() if end_dt else ""

    # Bounding box [W, S, E, N]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    # GeoJSON LineString
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": {
            "pilot": pilot,
            "glider": glider,
            "start_time": start_time,
            "end_time": end_time,
        },
    }

    return {
        "pilot": pilot,
        "glider": glider,
        "start_time": start_time,
        "end_time": end_time,
        "bbox": json.dumps(bbox),
        "geojson": json.dumps(geojson),
    }


def detect_thermals(track: dict) -> list:
    """Detect thermal circles in a track (future implementation)."""
    raise NotImplementedError("Thermal detection is not yet implemented.")
