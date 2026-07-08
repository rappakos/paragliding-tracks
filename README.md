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

## Thermal Core Analysis

### IGC Track Upload

Upload `.igc` files to visualize paraglider flight tracks on the map. The system extracts per-fix GPS coordinates, timestamps, and altitudes. An interactive altitude-vs-time chart (uPlot) lets you select thermal segments for analysis.

### Linear Regression (zeroth approximation)

For a selected climbing segment, a linear regression estimates the thermal core position `(x_c, y_c)` as a function of altitude:

```
x_core(z) = a_x · z + b_x
y_core(z) = a_y · z + b_y
```

This gives:
- **Core line**: projected path of the thermal center from ground to top
- **Drift vector**: direction and speed of horizontal thermal drift (driven by wind profile)
- **Climb rate**: average vertical speed during the segment
- **Turn count**: estimated full circles from unwrapped angular position

### Extended Kalman Filter Model (planned)

See [Extended Kalman Filter](./EKF_MODEL.md)

### API Endpoint

```
POST /igc/tracks/{id}/analyze
Body: { "start_idx": 120, "end_idx": 450 }
```

Returns: `core_line` (GeoJSON), `avg_climb_rate`, `altitude_gain`, `n_turns`, `drift_bearing`, `drift_speed`.
