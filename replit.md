# TraceCarbon dMRV Registry

A production-ready digital Measurement, Reporting, and Verification (dMRV) backend for a carbon credit registry. Validates aerial tree-crown imagery through a 4-step async pipeline and emits an immutable Web3 JSON payload for ERC-20 carbon credit minting on Polygon Amoy.

## Run & Operate

- `cd /home/runner/workspace && python main.py` — run the FastAPI dMRV server (port 8080)
- `curl http://localhost:80/api/healthz` — subsystem readiness check
- `curl http://localhost:80/api/docs` — interactive Swagger UI

## Required Asset Files (place in project root before starting)

| File | Purpose |
|------|---------|
| `best.pt` | Custom-trained YOLOv8 tree crown detection model weights |
| `earth_engine_key.json` | Google Cloud service account key for Earth Engine API |

Both files must be present in `/home/runner/workspace/` before the server starts. The server boots cleanly without them but returns HTTP 503 on `/api/verify-audit` until they are uploaded and the server is restarted.

## Stack

- Python 3.11
- FastAPI + Uvicorn (async ASGI)
- YOLOv8 via `ultralytics` — tree crown object detection
- Google Earth Engine API — Sentinel-2 NDVI satellite delta
- OpenWeatherMap REST API — real-time weather cross-verification
- Pillow — EXIF GPS metadata extraction and fraud check
- `opencv-python-headless` — OpenCV without libGL (server-safe)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENWEATHERMAP_API_KEY` | `MOCK_OWM_KEY_REPLACE_IN_PRODUCTION` | OWM API key |
| `YOLO_MODEL_PATH` | `best.pt` | Path to YOLOv8 weights |
| `EE_KEY_PATH` | `earth_engine_key.json` | Path to GEE service account JSON |
| `PORT` | `8080` | Server listen port |

## dMRV Pipeline (4 Steps)

1. **EXIF GPS Fraud Prevention** — Haversine distance check between image EXIF GPS and submitted coordinates. Rejects if variance > 100 m or metadata absent.
2. **OpenWeatherMap Cross-Verification** — Captures real-time temperature, weather condition, and UTC timestamp as permanent audit telemetry.
3. **Google Earth Engine NDVI Delta** — Queries Sentinel-2 SR imagery for two windows (2 years ago and last 60 days) over a 1 km² ROI. Compares mean NDVI. Flags `VEGETATION_DECLINE_DETECTED` if greenness has dropped.
4. **YOLOv8 Carbon Accounting** — Detects tree crowns; each bounding box = 1 tree. Applies ARR metrics: 22 kg CO2/tree/yr, 15% buffer pool deduction.

## Carbon Credit Math

```
Gross CO2 (t/yr) = (tree_count × 22 kg) / 1000
Net Credits      = Gross × 0.85   (after 15% buffer pool deduction)
Buffer Retained  = Gross × 0.15
```

## Key Design Decisions

- **Lazy subsystem init** — YOLO and GEE load at startup via FastAPI lifespan, but failures only warn; the server always starts. `/api/verify-audit` returns HTTP 503 if either subsystem is unavailable.
- **NDVI gates minting** — `VEGETATION_DECLINE_DETECTED` sets `status: FLAGGED_FOR_REVIEW` and zeroes `net_carbon_credits_mintable`, blocking the blockchain queue.
- **CPU-bound offloading** — YOLO inference and GEE computation run in `asyncio.run_in_executor` to avoid blocking the event loop.
- **opencv-python-headless** — Used instead of `opencv-python` to avoid `libGL.so.1` dependency missing in headless server environments.
- **CORS** — `allow_origins=["*"]` with `allow_credentials=False` (spec-compliant). Swap to explicit origin list + `allow_credentials=True` for production credentialed flows.

## Where Things Live

- `main.py` — entire FastAPI application (pipeline steps, models, endpoints)
- `requirements.txt` — Python dependencies
- `best.pt` — YOLOv8 model weights (user-supplied, not in repo)
- `earth_engine_key.json` — GEE service account key (user-supplied, not in repo)
- `artifacts/api-server/.replit-artifact/artifact.toml` — service config (runs `python main.py` from workspace root)

## User Preferences

- All backend logic in a single `main.py` at the workspace root.
- No mocking of pipeline steps in production — all four steps must clear for `APPROVED`.
- NDVI vegetation decline must block minting (not just log a warning).

## Gotchas

- Run `pip install opencv-python-headless` not `opencv-python` — the full version requires `libGL.so.1` which is absent in the Nix environment.
- The artifact workflow CWD is `artifacts/api-server/`, so the run command must `cd /home/runner/workspace` first.
- `allow_origins=["*"]` + `allow_credentials=True` is a CORS spec violation; keep credentials False when using wildcard origins.
- GEE `ServiceAccountCredentials(email=None, key_file=...)` — the `email` field is parsed from the JSON key automatically; do not hardcode it.
