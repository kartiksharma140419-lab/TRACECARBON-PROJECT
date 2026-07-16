# ─────────────────────────────────────────────────────────────────────────────
# TraceCarbon dMRV Registry — Dockerfile
# Target: Google Cloud Run
# Runtime: Python 3.11-slim (matches production interpreter)
#
# IMPORTANT: best.pt is NOT baked into this image.
# The backend fetches it at startup from HuggingFace (HF_TOKEN + HF_REPO_ID).
# earth_engine_key.json is injected at runtime via Cloud Run Secret Manager.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── OS-level system dependencies ──────────────────────────────────────────────
# build-essential : compiles C/C++ extensions required by ultralytics, numpy, etc.
# libgomp1        : OpenMP runtime needed by PyTorch / ultralytics inference
# libgl1-mesa-glx : headless OpenGL stub (opencv-python-headless still links it)
# libglib2.0-0    : required by opencv at import time on Debian-based images
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (separate layer for Docker cache efficiency) ───────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy only the entrypoint; all secrets / model weights are injected at runtime.
COPY main.py .

# ── Port exposure ─────────────────────────────────────────────────────────────
# Cloud Run injects PORT at container startup (default 8080).
EXPOSE ${PORT:-8080}

# ── Start command ─────────────────────────────────────────────────────────────
# uvicorn reads $PORT at launch; shell form required so the env var expands.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
