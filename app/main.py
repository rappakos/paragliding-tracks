from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import settings
from app.db import init_db
from app.routers import dem, weather, overlays, igc, bookmarks

logging.basicConfig(level=settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Thermal Trigger Visualizer",
    description="Paragliding thermal trigger visualizer for German flatlands.",
    version="0.2.0",
    lifespan=lifespan,
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
app.include_router(bookmarks.router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


# Serve frontend static files
_WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(_WEB_DIR, "index.html"))
