---
name: Python FastAPI on Replit
description: Environment quirks and decisions for running a Python FastAPI server alongside the pnpm monorepo in this workspace.
---

## Key Rules

**opencv-python vs headless:** Always install `opencv-python-headless`, never `opencv-python`. The full package requires `libGL.so.1` which is absent in the Nix environment and causes an `ImportError` at startup.

**artifact.toml run command must cd to workspace root:** The artifact workflow CWD is `artifacts/<slug>/`, not the workspace root. If `main.py` lives at the workspace root (alongside model files like `best.pt`), the run command must be `cd /home/runner/workspace && python main.py`.

**uv resolver blocks ultralytics on Linux:** The `installLanguagePackages` callback (which uses uv) cannot resolve `ultralytics` due to a platform marker issue. Use `pip install` via ShellExec instead.

**CORS spec rule:** `allow_origins=["*"]` + `allow_credentials=True` is a CORS spec violation. Use `allow_credentials=False` with wildcard, or an explicit origin list with credentials.

**Why:** These were all discovered as runtime failures during the TraceCarbon dMRV build.

**How to apply:** Any future Python FastAPI artifact in this workspace should follow all four rules above before first boot.
