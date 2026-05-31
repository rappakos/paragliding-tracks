# paragliding-tracks

Thermal Trigger Visualizer for paragliding in the German flatlands (Weserbergland, Harz).

## What it does

Interactive map overlay showing:
- **Thermal driving** – sun-angle-weighted slope heating relative to flat terrain
- **Wind exposure** – windward/lee classification based on terrain normals and forecast wind

Uses real Copernicus GLO-30 DEM data, pvlib solar position, and Open-Meteo wind forecasts.

## Stack

- **Backend**: Python 3.11+ / FastAPI / uvicorn
- **DEM**: Copernicus GLO-30 via OpenTopography API
- **Solar**: pvlib
- **Wind**: Open-Meteo forecast API
- **Frontend**: MapLibre GL JS with raster image overlays
- **Caching**: diskcache (DEM, normals, overlays, sun, wind)

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -e .
```

Create a `.env` file:
```
OPENTOPOGRAPHY_API_KEY=your_key_here
```

Get a free API key at https://opentopography.org.

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 in browser.

## Configuration

Environment variables (or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_BBOX` | `9.21,51.74,9.80,52.12` | W,S,E,N bounding box |
| `OPENTOPOGRAPHY_API_KEY` | – | Required for DEM download |
| `OVERLAY_TTL` | `300` | Cache TTL for overlays (seconds) |
| `DEM_RES_M` | `30` | DEM resolution in metres |
