"""
Disk-based caches using diskcache.
"""
from __future__ import annotations

import os

from diskcache import Cache

from app.config import settings

_base = settings.cache_dir

dem_cache = Cache(os.path.join(_base, "dem"))
normals_cache = Cache(os.path.join(_base, "normals"))
overlay_cache = Cache(os.path.join(_base, "overlay"))
sun_cache = Cache(os.path.join(_base, "sun"))
wind_cache = Cache(os.path.join(_base, "wind"))
