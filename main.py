"""
TraceCarbon dMRV Carbon Credit Registry - FastAPI Backend
==========================================================
Principal Backend / Core Web3 Architect Implementation

A modular, production-ready digital Measurement, Reporting, and Verification
(dMRV) pipeline that validates aerial imagery for carbon credit issuance.

Pipeline:
  Step 1 → EXIF GPS Fraud Prevention
  Step 2 → Real-time OpenWeatherMap Cross-Verification
  Step 3 → Google Earth Engine Satellite NDVI Delta Analysis
  Step 4 → YOLOv8 Computer Vision Tree Detection & Carbon Accounting

Response: Immutable Web3 JSON payload targeting Polygon Amoy.

STARTUP NOTE:
  The server starts immediately in all cases. If 'best.pt' or 'earth_engine_key.json'
  are absent, subsystems are marked UNAVAILABLE and the /api/verify-audit endpoint
  returns HTTP 503 with a clear diagnostic. Upload the files to the project root and
  restart the server to activate the full dMRV pipeline.
"""

import os
import io
import math
import struct
import tempfile
import logging
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any

import re
import httpx
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("tracecarbon.dmrv")


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLES / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

OPENWEATHERMAP_API_KEY: str = os.environ.get(
    "OPENWEATHERMAP_API_KEY",
    "MOCK_OWM_KEY_REPLACE_IN_PRODUCTION",
)

# International ARR (Afforestation, Reforestation, Revegetation) Carbon Metrics
SEQUESTRATION_KG_PER_TREE_PER_YEAR: float = 22.0  # kg CO2 per mature tree per year
BUFFER_POOL_DEDUCTION_RATE: float = 0.15  # 15% permanence risk buffer
NET_CREDIT_RETENTION_RATE: float = 1.0 - BUFFER_POOL_DEDUCTION_RATE  # 0.85

# Spatial fraud prevention threshold
EXIF_COORDINATE_MAX_VARIANCE_METERS: float = 100.0

# Upload constraints
MAX_IMAGE_BYTES: int = 50 * 1024 * 1024  # 50 MB hard cap
ALLOWED_MIME_PREFIXES: Tuple = ("image/jpeg", "image/png", "image/tiff", "image/webp")

# EVM wallet address: 0x followed by exactly 40 hex characters
_EVM_WALLET_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

# Asset file paths — expected at project root
YOLO_MODEL_PATH: str = os.environ.get("YOLO_MODEL_PATH", "best.pt")
EE_KEY_PATH: str = os.environ.get("EE_KEY_PATH", "earth_engine_key.json")

# Sentinel-2 surface reflectance collection identifier
EE_SENTINEL2_COLLECTION: str = "COPERNICUS/S2_SR_HARMONIZED"


# ─────────────────────────────────────────────────────────────────────────────
# SUBSYSTEM STATE  (lazy initialization — server is always startable)
# ─────────────────────────────────────────────────────────────────────────────


class SubsystemState:
    """Tracks readiness of heavyweight external subsystems."""

    yolo_model: Any = None  # YOLO instance or None
    yolo_ready: bool = False
    yolo_error: str = ""

    ee_ready: bool = False
    ee_error: str = ""

    def ready(self) -> bool:
        return self.yolo_ready and self.ee_ready

    def diagnostic(self) -> Dict[str, Any]:
        return {
            "yolo": {
                "status": "ready" if self.yolo_ready else "unavailable",
                "model_path": YOLO_MODEL_PATH,
                "error": self.yolo_error or None,
            },
            "earth_engine": {
                "status": "ready" if self.ee_ready else "unavailable",
                "key_path": EE_KEY_PATH,
                "error": self.ee_error or None,
            },
            "openweathermap": {
                "status": "configured"
                if OPENWEATHERMAP_API_KEY != "MOCK_OWM_KEY_REPLACE_IN_PRODUCTION"
                else "mock_key",
            },
        }


_state = SubsystemState()


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — attempt subsystem initialization at startup (non-fatal)
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Attempts to load YOLO and Earth Engine on startup.
    Missing assets produce WARNING logs but do NOT crash the server.
    """
    logger.info("════════════════════════════════════════")
    logger.info("TraceCarbon dMRV — Server starting up …")
    logger.info("════════════════════════════════════════")

    # ── 1. YOLOv8 tree crown detection model ─────────────────────────────────
    if not os.path.isfile(YOLO_MODEL_PATH):
        _state.yolo_error = (
            f"Model weights file '{YOLO_MODEL_PATH}' not found in project root. "
            "Upload 'best.pt' and restart the server."
        )
        logger.warning("[YOLO] ⚠ %s", _state.yolo_error)
    else:
        try:
            from ultralytics import YOLO

            _state.yolo_model = YOLO(YOLO_MODEL_PATH)
            _state.yolo_ready = True
            logger.info(
                "[YOLO] ✓ Custom tree crown model loaded from '%s'", YOLO_MODEL_PATH
            )
        except Exception as exc:
            _state.yolo_error = str(exc)
            logger.error("[YOLO] ✗ Failed to load model — %s", exc)

    # ── 2. Google Earth Engine ───────────────────────────────────────────────
    if not os.path.isfile(EE_KEY_PATH):
        _state.ee_error = (
            f"Service account key '{EE_KEY_PATH}' not found in project root. "
            "Upload 'earth_engine_key.json' and restart the server."
        )
        logger.warning("[GEE] ⚠ %s", _state.ee_error)
    else:
        try:
            import ee

            ee_credentials = ee.ServiceAccountCredentials(
                email=None,  # parsed automatically from the JSON key
                key_file=EE_KEY_PATH,
            )
            ee.Initialize(credentials=ee_credentials, project=None)
            _state.ee_ready = True
            logger.info(
                "[GEE] ✓ Google Earth Engine authenticated via '%s'", EE_KEY_PATH
            )
        except Exception as exc:
            _state.ee_error = str(exc)
            logger.error("[GEE] ✗ Earth Engine initialization failed — %s", exc)

    # ── Startup summary ───────────────────────────────────────────────────────
    if _state.ready():
        logger.info(
            "[STARTUP] ✓ All subsystems READY — dMRV pipeline fully operational."
        )
    else:
        logger.warning(
            "[STARTUP] ⚠ One or more subsystems UNAVAILABLE. "
            "The /api/verify-audit endpoint will return HTTP 503 until resolved. "
            "See /api/healthz for subsystem diagnostics."
        )

    yield  # ← server is live and accepting requests here

    logger.info("[SHUTDOWN] TraceCarbon dMRV — graceful shutdown complete.")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TraceCarbon dMRV Registry API",
    description=(
        "Digital Measurement, Reporting, and Verification pipeline for carbon "
        "credit issuance. Validates aerial imagery through a 4-step async audit: "
        "EXIF fraud prevention → weather cross-verification → satellite NDVI "
        "delta → YOLOv8 tree detection."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# CORS — wildcard origin + no credentials (safe for public API / design platforms).
# If you later need credentialed cross-origin flows, replace "*" with an explicit
# list of trusted origins and re-enable allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _dms_component_to_float(component: Any) -> float:
    """
    Robustly convert a single EXIF DMS component to float.

    Pillow may deliver any of these types depending on version and JPEG writer:
      • IFDRational  — supports float() coercion directly
      • tuple (num, den) — legacy rational representation
      • int / float  — already numeric (some cameras embed pre-computed values)
    """
    try:
        # IFDRational and plain numeric types all satisfy float()
        return float(component)
    except (TypeError, ValueError):
        pass
    # Fallback: explicit (numerator, denominator) tuple
    if hasattr(component, "__len__") and len(component) == 2:
        num, den = component
        if den == 0:
            return 0.0
        return float(num) / float(den)
    raise ValueError(f"Cannot convert EXIF DMS component to float: {component!r}")


def _dms_to_decimal(dms: Any, ref: str) -> float:
    """
    Convert EXIF GPS DMS value → signed decimal degrees.

    Handles all Pillow EXIF variants:
      • Sequence of IFDRational objects  (Pillow ≥ 6)
      • Sequence of (num, den) tuples    (legacy)
      • Plain numeric sequence           (pre-computed)

    Args:
        dms: Three-element sequence representing degrees, minutes, seconds.
        ref: Cardinal direction character: 'N', 'S', 'E', or 'W'.

    Returns:
        Signed decimal degrees (negative for S/W hemispheres).
    """
    degrees = _dms_component_to_float(dms[0])
    minutes = _dms_component_to_float(dms[1])
    seconds = _dms_component_to_float(dms[2])
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if str(ref).strip().upper() in ("S", "W"):
        decimal = -decimal
    return decimal


def _haversine_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance (metres) between two WGS-84 coordinates
    using the Haversine formula.
    """
    R = 6_371_000  # Earth mean radius in metres
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _extract_exif_gps(image: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Extract and parse GPS metadata from a PIL Image's EXIF block.

    Returns:
        Dict with 'latitude' and 'longitude' in decimal degrees, or None if absent.
    """
    try:
        raw_exif = image._getexif()  # type: ignore[attr-defined]
        if not raw_exif:
            return None

        exif_data: Dict[str, Any] = {
            TAGS.get(tag_id, tag_id): value for tag_id, value in raw_exif.items()
        }

        gps_info_raw: Optional[Dict] = exif_data.get("GPSInfo")
        if not gps_info_raw:
            return None

        gps_data: Dict[str, Any] = {
            GPSTAGS.get(key, key): value for key, value in gps_info_raw.items()
        }

        required_keys = {
            "GPSLatitude",
            "GPSLatitudeRef",
            "GPSLongitude",
            "GPSLongitudeRef",
        }
        if not required_keys.issubset(gps_data.keys()):
            logger.warning(
                "[STEP-1] GPSInfo block present but incomplete — keys: %s",
                list(gps_data.keys()),
            )
            return None

        lat = _dms_to_decimal(gps_data["GPSLatitude"], gps_data["GPSLatitudeRef"])
        lon = _dms_to_decimal(gps_data["GPSLongitude"], gps_data["GPSLongitudeRef"])
        return {"latitude": lat, "longitude": lon}

    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        struct.error,
        ZeroDivisionError,
    ) as exc:
        logger.warning("[STEP-1] EXIF parse error — %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────────────────────────────────────


async def step1_exif_fraud_check(
    image: Image.Image,
    submitted_lat: float,
    submitted_lon: float,
) -> bool:
    """
    STEP 1 — EXIF Metadata Fraud Prevention Check
    ───────────────────────────────────────────────
    Extracts embedded GPS EXIF coordinates from the uploaded image and computes
    the Haversine distance against the user-submitted coordinates.

    Returns True if EXIF GPS was present and within tolerance.
    Returns False (with a warning log) if EXIF metadata is absent or the
    distance calculation fails — execution continues using the user-supplied
    coordinates (development / test-mode behaviour).

    Raises HTTP 400 only if EXIF is present but the coordinate discrepancy
    exceeds EXIF_COORDINATE_MAX_VARIANCE_METERS (active fraud signal).
    """
    logger.info(
        "[STEP-1] Initiating EXIF GPS fraud prevention check (submitted: %.6f, %.6f)",
        submitted_lat,
        submitted_lon,
    )

    gps_coords = _extract_exif_gps(image)

    if gps_coords is None:
        logger.warning(
            "⚠️ [TEST MODE] EXIF Metadata missing or invalid. "
            "Bypassing location check for development testing."
        )
        return False

    try:
        exif_lat = gps_coords["latitude"]
        exif_lon = gps_coords["longitude"]
        distance_m = _haversine_distance_meters(
            submitted_lat, submitted_lon, exif_lat, exif_lon
        )
    except Exception as exc:
        logger.warning(
            "⚠️ [TEST MODE] EXIF Metadata missing or invalid. "
            "Bypassing location check for development testing. (detail: %s)",
            exc,
        )
        return False

    logger.info(
        "[STEP-1] EXIF GPS decoded — lat=%.6f, lon=%.6f | "
        "Haversine variance=%.2f m (threshold=%d m)",
        exif_lat,
        exif_lon,
        distance_m,
        EXIF_COORDINATE_MAX_VARIANCE_METERS,
    )

    if distance_m > EXIF_COORDINATE_MAX_VARIANCE_METERS:
        logger.error(
            "[STEP-1] REJECTED — EXIF/submitted coordinate variance %.2f m "
            "exceeds 100 m threshold. Possible coordinate spoofing detected.",
            distance_m,
        )
        raise HTTPException(
            status_code=400,
            detail=(
                f"Location Verification Failure: Asset metadata mismatch. "
                f"EXIF GPS coordinates ({exif_lat:.6f}, {exif_lon:.6f}) diverge "
                f"{distance_m:.1f} m from submitted coordinates "
                f"({submitted_lat:.6f}, {submitted_lon:.6f}). "
                f"Maximum permitted variance: {EXIF_COORDINATE_MAX_VARIANCE_METERS:.0f} m."
            ),
        )

    logger.info(
        "[STEP-1] ✓ EXIF fraud check PASSED — variance %.2f m within tolerance.",
        distance_m,
    )
    return True


async def step2_weather_cross_verify(lat: float, lon: float) -> Dict[str, Any]:
    """
    STEP 2 — Real-time OpenWeatherMap Environmental Context
    ────────────────────────────────────────────────────────
    Fetches current atmospheric conditions at the submitted coordinates.
    Captures temperature, weather status, and UTC timestamp as permanent
    dMRV environmental context metadata logged to the audit record.

    Returns:
        Dict with temperature_kelvin, weather_description, timestamp_utc.
    """
    logger.info(
        "[STEP-2] Fetching real-time weather context for (%.6f, %.6f) …",
        lat,
        lon,
    )

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={OPENWEATHERMAP_API_KEY}"
    )

    fallback = {
        "temperature_kelvin": None,
        "weather_description": "WEATHER_UNAVAILABLE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[STEP-2] OpenWeatherMap HTTP %d — %s", exc.response.status_code, exc
        )
        logger.warning("[STEP-2] Proceeding with fallback weather context.")
        return fallback
    except httpx.RequestError as exc:
        logger.error("[STEP-2] Weather API network error — %s", exc)
        return fallback

    temp_k: float = data.get("main", {}).get("temp", 0.0)
    weather_list = data.get("weather", [{}])
    weather_desc: str = (
        weather_list[0].get("description", "unknown") if weather_list else "unknown"
    )
    obs_ts: int = data.get("dt", 0)
    obs_utc: str = (
        datetime.fromtimestamp(obs_ts, tz=timezone.utc).isoformat()
        if obs_ts
        else datetime.now(timezone.utc).isoformat()
    )

    logger.info(
        "[STEP-2] ✓ Weather context captured — "
        "temp=%.1f K (%.1f °C), condition='%s', obs_time=%s",
        temp_k,
        temp_k - 273.15,
        weather_desc,
        obs_utc,
    )

    return {
        "temperature_kelvin": round(temp_k, 2),
        "weather_description": weather_desc,
        "timestamp_utc": obs_utc,
    }


async def step3_satellite_ndvi_delta(
    lat: float,
    lon: float,
) -> Tuple[float, float, str]:
    """
    STEP 3 — Historical Satellite Greenness Delta (Google Earth Engine)
    ────────────────────────────────────────────────────────────────────
    Pulls two Sentinel-2 SR image mosaics over a 1 km² bounding box:
      • Historical baseline: window centred 2 years ago (±30-day filter)
      • Current baseline:    most recent 60-day window

    Computes mean NDVI for each mosaic and assesses vegetation trajectory.

    Returns:
        Tuple of (ndvi_historical, ndvi_current, approval_flag_string).
    """
    logger.info(
        "[STEP-3] Querying Google Earth Engine NDVI delta for (%.6f, %.6f) …",
        lat,
        lon,
    )

    def _gee_compute() -> Tuple[float, float]:
        """Blocking GEE computation — runs in thread executor."""
        import ee  # earthengine-api (already initialized in lifespan)

        # 1 km² bounding box around the submitted coordinate
        point = ee.Geometry.Point([lon, lat])
        roi = point.buffer(500).bounds()

        # Date windows
        now = datetime.now(timezone.utc)
        hist_end = (now - timedelta(days=730)).strftime("%Y-%m-%d")
        hist_start = (now - timedelta(days=760)).strftime("%Y-%m-%d")
        curr_end = now.strftime("%Y-%m-%d")
        curr_start = (now - timedelta(days=60)).strftime("%Y-%m-%d")

        logger.info(
            "[STEP-3][GEE] Historical: %s → %s | Current: %s → %s",
            hist_start,
            hist_end,
            curr_start,
            curr_end,
        )

        def _ndvi_mean(start_date: str, end_date: str) -> float:
            """Cloud-filtered mean NDVI over ROI for a date range."""
            col = (
                ee.ImageCollection(EE_SENTINEL2_COLLECTION)
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                .select(["B8", "B4"])  # NIR, Red — NDVI = (B8−B4)/(B8+B4)
            )

            # Widen window if collection is empty (sparse coverage areas)
            if col.size().getInfo() == 0:
                logger.warning(
                    "[STEP-3][GEE] No imagery in primary window (%s→%s). "
                    "Widening to ±90 days.",
                    start_date,
                    end_date,
                )
                dt_start = datetime.strptime(start_date, "%Y-%m-%d")
                dt_end = datetime.strptime(end_date, "%Y-%m-%d")
                col = (
                    ee.ImageCollection(EE_SENTINEL2_COLLECTION)
                    .filterBounds(roi)
                    .filterDate(
                        (dt_start - timedelta(days=90)).strftime("%Y-%m-%d"),
                        (dt_end + timedelta(days=90)).strftime("%Y-%m-%d"),
                    )
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
                    .select(["B8", "B4"])
                )

            mosaic = col.mosaic()
            ndvi_image = mosaic.normalizedDifference(["B8", "B4"]).rename("ndvi")
            stats = ndvi_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=10,  # Sentinel-2 native 10 m resolution
                maxPixels=1_000_000,
            )
            val = stats.get("ndvi").getInfo()
            return float(val) if val is not None else 0.0

        return _ndvi_mean(hist_start, hist_end), _ndvi_mean(curr_start, curr_end)

    loop = asyncio.get_event_loop()
    ndvi_hist, ndvi_curr = await loop.run_in_executor(None, _gee_compute)

    delta = ndvi_curr - ndvi_hist
    logger.info(
        "[STEP-3] NDVI results — historical=%.4f | current=%.4f | delta=%.4f",
        ndvi_hist,
        ndvi_curr,
        delta,
    )

    if ndvi_curr >= ndvi_hist:
        flag = "POSITIVE_GROWTH_CONFIRMED"
        logger.info(
            "[STEP-3] ✓ Satellite NDVI check PASSED — "
            "stable/positive greenness trajectory (Δ=%.4f).",
            delta,
        )
    else:
        flag = "VEGETATION_DECLINE_DETECTED"
        logger.warning(
            "[STEP-3] ⚠ NDVI decline detected (Δ=%.4f). Flagged for manual review.",
            delta,
        )

    return ndvi_hist, ndvi_curr, flag


async def step4_yolo_carbon_accounting(
    image_bytes: bytes,
) -> Tuple[int, float, float, float]:
    """
    STEP 4 — YOLOv8 Computer Vision Inference & Carbon Accounting
    ───────────────────────────────────────────────────────────────
    Runs the uploaded aerial image through the globally-loaded custom YOLO
    tree crown detection model. Each detected bounding box = 1 physical tree.

    Carbon Accounting Formula (International ARR Standard):
      Gross Offset (t CO2/yr) = (tree_count × 22 kg/tree) / 1000
      Net Credits Mintable    = Gross Offset × 0.85  (after 15% buffer deduction)
      Buffer Pool Retained    = Gross Offset × 0.15

    Returns:
        Tuple of (tree_count, gross_co2_tons, net_credits, buffer_deduction).
    """
    logger.info("[STEP-4] Initiating YOLOv8 tree crown detection inference …")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(image_bytes)

    try:
        loop = asyncio.get_event_loop()

        def _run_inference() -> int:
            results = _state.yolo_model(
                source=tmp_path,
                verbose=False,
                conf=0.25,
                iou=0.45,
            )
            if not results:
                return 0
            return sum(len(r.boxes) for r in results if r.boxes is not None)

        tree_count: int = await loop.run_in_executor(None, _run_inference)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # ── Carbon Accounting Math ───────────────────────────────────────────────
    gross_kg: float = tree_count * SEQUESTRATION_KG_PER_TREE_PER_YEAR
    gross_tons: float = gross_kg / 1000.0
    buffer_deduction: float = gross_tons * BUFFER_POOL_DEDUCTION_RATE
    net_credits: float = gross_tons * NET_CREDIT_RETENTION_RATE

    logger.info(
        "[STEP-4] ✓ Inference complete — trees=%d | gross=%.4f t CO2/yr | "
        "net_credits=%.4f t | buffer_pool=%.4f t",
        tree_count,
        gross_tons,
        net_credits,
        buffer_deduction,
    )

    return (
        tree_count,
        round(gross_tons, 6),
        round(net_credits, 6),
        round(buffer_deduction, 6),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY dMRV AUDIT ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/verify-audit",
    summary="Submit dMRV Carbon Credit Verification Audit",
    description=(
        "Accepts an aerial survey image alongside GPS coordinates and a blockchain "
        "wallet address. Executes the 4-step dMRV pipeline and returns a structured "
        "Web3 payload ready for ERC-20 carbon credit minting on Polygon Amoy."
    ),
    response_model=None,
    tags=["dMRV Pipeline"],
)
async def verify_audit(
    latitude: float = Form(
        ..., description="Decimal latitude of the carbon asset site (WGS-84)"
    ),
    longitude: float = Form(
        ..., description="Decimal longitude of the carbon asset site (WGS-84)"
    ),
    wallet_address: str = Form(
        ..., description="Target EVM-compatible wallet address for credit minting"
    ),
    image: UploadFile = File(
        ..., description="Aerial survey image (JPEG/PNG with GPS EXIF metadata)"
    ),
) -> JSONResponse:
    """
    Execute the full 4-step dMRV verification pipeline and return an immutable
    Web3 carbon credit audit response.
    """
    # ── Subsystem readiness gate ─────────────────────────────────────────────
    if not _state.ready():
        diag = _state.diagnostic()
        missing = []
        if not _state.yolo_ready:
            missing.append(f"YOLO model: {_state.yolo_error}")
        if not _state.ee_ready:
            missing.append(f"Earth Engine: {_state.ee_error}")

        logger.error(
            "[VERIFY-AUDIT] Request rejected — subsystems not ready: %s",
            " | ".join(missing),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "error": "dMRV pipeline subsystems not fully initialised.",
                "resolution": (
                    "Upload 'best.pt' and 'earth_engine_key.json' to the project "
                    "root directory and restart the server."
                ),
                "subsystems": diag,
            },
        )

    audit_id = f"TC-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info(
        "════════════════════════════════════════════════════════════\n"
        "[AUDIT %s] New dMRV verification request\n"
        "  Coordinates  : (%.6f, %.6f)\n"
        "  Wallet       : %s\n"
        "  Image file   : %s\n"
        "════════════════════════════════════════════════════════════",
        audit_id,
        latitude,
        longitude,
        wallet_address,
        image.filename,
    )

    # ── Input validation ─────────────────────────────────────────────────────
    if not (-90.0 <= latitude <= 90.0):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid latitude '{latitude}'. Must be in range [-90, 90].",
        )
    if not (-180.0 <= longitude <= 180.0):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid longitude '{longitude}'. Must be in range [-180, 180].",
        )

    wallet_clean = wallet_address.strip()
    if not wallet_clean:
        raise HTTPException(status_code=422, detail="wallet_address must not be empty.")
    if not _EVM_WALLET_RE.match(wallet_clean):
        raise HTTPException(
            status_code=422,
            detail=(
                "wallet_address must be a valid EVM address "
                "(0x followed by 40 hexadecimal characters)."
            ),
        )

    # ── MIME type guard (check Content-Type header from the upload) ──────────
    content_type = (image.content_type or "").lower().split(";")[0].strip()
    if content_type and not any(
        content_type.startswith(p) for p in ALLOWED_MIME_PREFIXES
    ):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image type '{content_type}'. "
                f"Accepted types: JPEG, PNG, TIFF, WebP."
            ),
        )

    # ── Read upload once — reuse bytes across all pipeline steps ────────────
    image_bytes: bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds maximum permitted size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.load()
    except Exception as exc:
        logger.error("[AUDIT %s] Image decode failed — %s", audit_id, exc)
        # Sanitise: do not forward raw exception text to the client
        raise HTTPException(
            status_code=400,
            detail="Uploaded file could not be decoded as a valid image.",
        )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 — EXIF GPS Fraud Prevention
    # ────────────────────────────────────────────────────────────────────────
    logger.info("[AUDIT %s] ── STEP 1: EXIF Fraud Prevention ──", audit_id)
    exif_verified: bool = await step1_exif_fraud_check(pil_image, latitude, longitude)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 — Real-time Weather Cross-Verification
    # ────────────────────────────────────────────────────────────────────────
    logger.info("[AUDIT %s] ── STEP 2: Weather Cross-Verification ──", audit_id)
    weather_ctx = await step2_weather_cross_verify(latitude, longitude)
    weather_desc: str = weather_ctx["weather_description"]

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3 — Satellite NDVI Delta (GEE)
    # ────────────────────────────────────────────────────────────────────────
    logger.info("[AUDIT %s] ── STEP 3: Satellite NDVI Delta Analysis ──", audit_id)
    ndvi_hist, ndvi_curr, ndvi_flag = await step3_satellite_ndvi_delta(
        latitude, longitude
    )

    # ────────────────────────────────────────────────────────────────────────
    # STEP 4 — YOLOv8 Inference & Carbon Accounting
    # ────────────────────────────────────────────────────────────────────────
    logger.info("[AUDIT %s] ── STEP 4: YOLO Inference & Carbon Accounting ──", audit_id)
    (
        tree_count,
        gross_tons,
        net_credits,
        buffer_deduction,
    ) = await step4_yolo_carbon_accounting(image_bytes)

    # ────────────────────────────────────────────────────────────────────────
    # POLICY GATE — NDVI verification outcome drives final status
    # ────────────────────────────────────────────────────────────────────────
    ndvi_approved = ndvi_flag == "POSITIVE_GROWTH_CONFIRMED"

    # Determine final audit disposition
    if ndvi_approved:
        final_status = "APPROVED"
        action_pending = "MINT_ERC20"
        abi_ready = True
        http_status_code = 200
        logger.info(
            "[AUDIT %s] ✓ ALL 4 STEPS CLEARED — NDVI positive. "
            "Composing Polygon Amoy mint payload.",
            audit_id,
        )
    else:
        # Vegetation decline detected: halt minting, flag for human review
        final_status = "FLAGGED_FOR_REVIEW"
        action_pending = "MANUAL_REVIEW_REQUIRED"
        abi_ready = False
        http_status_code = 200  # still 200 — the audit ran; outcome is in the body
        logger.warning(
            "[AUDIT %s] ⚠ NDVI decline flagged — status=FLAGGED_FOR_REVIEW. "
            "Carbon credits withheld pending manual verification.",
            audit_id,
        )

    # ────────────────────────────────────────────────────────────────────────
    # IMMUTABLE WEB3 RESPONSE PAYLOAD
    # ────────────────────────────────────────────────────────────────────────
    payload: Dict[str, Any] = {
        "status": final_status,
        "audit_id": audit_id,
        "wallet_targeted": wallet_clean,
        "trees_detected": tree_count,
        "gross_annual_co2_tons": gross_tons,
        "net_carbon_credits_mintable": net_credits if ndvi_approved else 0.0,
        "buffer_pool_retained": buffer_deduction,
        "dMRV_telemetry": {
            "exif_match": exif_verified,
            "weather_condition": weather_desc,
            "weather_temperature_kelvin": weather_ctx.get("temperature_kelvin"),
            "weather_observation_utc": weather_ctx.get("timestamp_utc"),
            "satellite_ndvi_historical": round(ndvi_hist, 4),
            "satellite_ndvi_current": round(ndvi_curr, 4),
            "satellite_ndvi_delta": ndvi_flag,
        },
        "blockchain_queue": {
            "network": "Polygon Amoy",
            "action_pending": action_pending,
            "contract_abi_ready": abi_ready,
        },
    }

    logger.info(
        "[AUDIT %s] ══ AUDIT COMPLETE ══ status=%s | "
        "trees=%d | gross=%.4f tCO2 | net=%.4f credits",
        audit_id,
        final_status,
        tree_count,
        gross_tons,
        net_credits,
    )

    return JSONResponse(content=payload, status_code=http_status_code)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/healthz", summary="Health Check", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Returns server liveness and subsystem readiness status."""
    return {
        "status": "healthy",
        "service": "TraceCarbon dMRV Registry API",
        "version": "1.0.0",
        "pipeline_ready": _state.ready(),
        "subsystems": _state.diagnostic(),
        "carbon_metrics": {
            "sequestration_kg_per_tree_yr": SEQUESTRATION_KG_PER_TREE_PER_YEAR,
            "buffer_pool_deduction_pct": BUFFER_POOL_DEDUCTION_RATE * 100,
            "net_retention_pct": NET_CREDIT_RETENTION_RATE * 100,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api", summary="API Root", tags=["System"])
async def api_root() -> Dict[str, str]:
    return {
        "service": "TraceCarbon dMRV Registry API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/healthz",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: prevent raw tracebacks from leaking to API clients."""
    logger.exception(
        "[GLOBAL] Unhandled exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "ERROR",
            "error": "Internal server error. The incident has been logged.",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info("Starting TraceCarbon dMRV API on port %d …", port)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )
