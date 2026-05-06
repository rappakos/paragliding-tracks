from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import settings
from app.routers import dem, weather, overlays, igc

logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Thermal Trigger Visualizer",
    description="Paragliding thermal trigger visualizer for German flatlands.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-BBox", "X-Width", "X-Height", "X-Pixel-Bounds"],
)

app.include_router(dem.router)
app.include_router(weather.router)
app.include_router(overlays.router)
app.include_router(igc.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# Serve frontend static files if built
_WEB_PUBLIC = os.path.join(os.path.dirname(__file__), "..", "web", "public")
if os.path.isdir(_WEB_PUBLIC):
    app.mount("/static", StaticFiles(directory=_WEB_PUBLIC), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(_WEB_PUBLIC, "index.html"))
