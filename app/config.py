from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Tuple


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openmeteo_base: str = "https://api.open-meteo.com/v1"
    cache_dir: str = "./cache"
    # W, S, E, N
    default_bbox: str = "9.36,51.83,9.65,52.03"
    log_level: str = "INFO"

    # Wind level weights: [10m, 925hPa, 850hPa, 700hPa]
    wind_weights: list[float] = Field(default=[0.4, 0.3, 0.2, 0.1])

    # DEM resolution in metres
    dem_res_m: int = 30

    # Overlay cache TTL in seconds
    overlay_ttl: int = 300

    def bbox_tuple(self) -> Tuple[float, float, float, float]:
        parts = [float(x) for x in self.default_bbox.split(",")]
        return (parts[0], parts[1], parts[2], parts[3])


settings = Settings()
