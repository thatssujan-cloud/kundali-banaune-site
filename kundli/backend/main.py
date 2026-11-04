"""
main.py — FastAPI backend for the Vedic Kundali (natal chart) module.
=====================================================================

Endpoints
---------
GET  /api/health                    Liveness + ephemeris provenance.
POST /api/generate-kundali          PRIMARY endpoint used by the GitHub-Pages
                                    frontend (index.html) — JSON birth data in,
                                    complete kundali JSON out.
POST /api/v1/kundali/chart          Alias of the above (versioned route).
GET  /api/v1/kundali/chart          Same, via query parameters (curl-friendly).
GET  /                              Serves the single-file frontend if present.

CORS is configured so that *.github.io origins (GitHub Pages) can call the API
without cross-origin issues; set KUNDALI_CORS_ORIGINS to lock it down in prod.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Deploy free on Render/Railway — see DEPLOYMENT.md at repo root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from vedic_core import VedicCalculator, AYANAMSA_MODES

BASE_DIR = Path(__file__).resolve().parent
EPHE_PATH = os.environ.get("KUNDALI_EPHE_PATH", str(BASE_DIR / "ephe"))
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"
# Single-file GitHub-Pages frontend.  Kept at repo root; when the backend is run
# from `kundli/backend`, it lives two levels up (see path probe below).
SINGLE_FILE_INDEX_CANDIDATES = [
    BASE_DIR.parent.parent / "index.html",          # <repo>/index.html
    BASE_DIR.parent / "frontend" / "index.html",    # legacy location
]

app = FastAPI(
    title="Vedic Kundali API",
    description=("Authentic Jyotish natal-chart engine: Swiss Ephemeris "
                 "(NASA JPL DE431), Lahiri ayanamsa, Whole-Sign houses, "
                 "Vimshottari dasha."),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — GitHub Pages friendly.
# Default "*" lets any *.github.io page fetch the API.  For production, set
# KUNDALI_CORS_ORIGINS="https://<user>.github.io/kundali-banaune-site" to pin it.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("KUNDALI_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# One engine instance per process; Swiss Ephemeris is thread-safe after setup.
CALC = VedicCalculator(ephe_path=EPHE_PATH, ayanamsa="LAHIRI")


# ---------------------------------------------------------------------------
# Request model + validation
# ---------------------------------------------------------------------------

class BirthData(BaseModel):
    """Birth input contract — everything the engine needs."""

    date_of_birth: str = Field(..., description="Local date 'YYYY-MM-DD'",
                               examples=["1990-05-15"])
    time_of_birth: str = Field(..., description="Local wall-clock 'HH:MM:SS'",
                               examples=["10:00:00"])
    latitude: float = Field(..., ge=-90.0, le=90.0,
                            description="Decimal degrees, North positive")
    longitude: float = Field(..., ge=-180.0, le=180.0,
                             description="Decimal degrees, East positive")
    timezone_offset_hours: float = Field(
        ..., ge=-12.0, le=14.0,
        description="Local offset from UTC in hours (e.g. India = 5.5)",
        examples=[5.5])
    ayanamsa: Optional[str] = Field("LAHIRI",
                                    description="LAHIRI|RAMAN|YUKTESHWAR|"
                                                "KRISHNAMURTI|DJWHALKHUL|TRUE_CHITRA")
    node_type: Optional[str] = Field("MEAN", description="MEAN (Parashari) or TRUE (KP)")

    @field_validator("date_of_birth")
    @classmethod
    def _check_date(cls, v: str) -> str:
        parts = v.split("-")
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            raise ValueError("date_of_birth must be 'YYYY-MM-DD'")
        if not (1 <= m <= 12 and 1 <= d <= 31):
            raise ValueError("date_of_birth out of range")
        return v

    @field_validator("time_of_birth")
    @classmethod
    def _check_time(cls, v: str) -> str:
        parts = v.split(":")
        try:
            hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            raise ValueError("time_of_birth must be 'HH:MM:SS' (24 h)")
        if not (0 <= hh < 24 and 0 <= mm < 60 and 0 <= ss < 60):
            raise ValueError("time_of_birth out of range")
        return v

    @field_validator("ayanamsa")
    @classmethod
    def _check_ayanamsa(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "LAHIRI"
        v = v.upper()
        if v not in AYANAMSA_MODES:
            raise ValueError(f"unknown ayanamsa '{v}'")
        return v

    @field_validator("node_type")
    @classmethod
    def _check_node(cls, v: Optional[str]) -> Optional[str]:
        v = (v or "MEAN").upper()
        if v not in {"MEAN", "TRUE"}:
            raise ValueError("node_type must be MEAN or TRUE")
        return v


def _compute(b: BirthData) -> dict:
    """Dispatch to a calculator configured with the request's options."""
    calc = CALC
    if (b.ayanamsa != CALC.ayanamsa_name) or (b.node_type != CALC.node_type):
        calc = VedicCalculator(ephe_path=EPHE_PATH, ayanamsa=b.ayanamsa,
                               node_type=b.node_type)
    try:
        return calc.compute_chart(
            date=b.date_of_birth, time=b.time_of_birth,
            tz_offset_hours=b.timezone_offset_hours,
            latitude=b.latitude, longitude=b.longitude)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500,
                            detail=f"ephemeris error: {exc}") from exc


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": "pyswisseph",
        "ephemeris_source": CALC.ephemeris_source,
        "ephe_path": EPHE_PATH,
        "default_ayanamsa": CALC.ayanamsa_name,
        "house_system": "Whole Sign (Rasi)",
    }


@app.post("/api/generate-kundali")
def generate_kundali(body: BirthData) -> JSONResponse:
    """PRIMARY endpoint consumed by the GitHub-Pages frontend (index.html).

    Request body (JSON)::

        {
          "date_of_birth":         "1990-05-15",   # YYYY-MM-DD (local)
          "time_of_birth":         "10:00:00",     # HH:MM:SS   (local wall clock)
          "latitude":              28.6139,        # decimal deg, N positive
          "longitude":             77.2090,        # decimal deg, E positive
          "timezone_offset_hours": 5.5             # local offset from UTC
        }

    Response: full kundali JSON — see API_SCHEMA.md / the module docstring of
    ``vedic_core.py`` for the contract (meta, input, lagna, planets[9],
    vimshottari_dasha).
    """
    return JSONResponse(_compute(body))


@app.post("/api/v1/kundali/chart")
def chart_post(body: BirthData) -> JSONResponse:
    """Versioned alias of :func:`generate_kundali` (kept for API consumers)."""
    return JSONResponse(_compute(body))


@app.get("/api/v1/kundali/chart")
def chart_get(
    date: str = Query(..., description="YYYY-MM-DD"),
    time: str = Query(..., description="HH:MM:SS"),
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    tz: float = Query(..., ge=-12, le=14, description="UTC offset hours"),
    ayanamsa: str = Query("LAHIRI"),
    node: str = Query("MEAN"),
) -> JSONResponse:
    """Convenience GET mirror of the POST endpoint (curl-friendly)."""
    b = BirthData(date_of_birth=date, time_of_birth=time, latitude=lat,
                  longitude=lon, timezone_offset_hours=tz,
                  ayanamsa=ayanamsa, node_type=node)
    return JSONResponse(_compute(b))


# ---------------------------------------------------------------------------
# Optional static hosting (useful for local one-box demos; GitHub Pages serves
# index.html itself, so these routes are only a convenience).
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def index():
    # 1) built React app, 2) single-file GitHub-Pages frontend.
    idx = FRONTEND_DIST / "index.html"
    if idx.exists():
        return FileResponse(idx)
    for cand in SINGLE_FILE_INDEX_CANDIDATES:
        if cand.exists():
            return FileResponse(cand)
    raise HTTPException(404, "frontend not found")


@app.get("/assets/{path}", include_in_schema=False)
def assets(path: str):
    f = (FRONTEND_DIST / "assets" / path).resolve()
    if not str(f).startswith(str(FRONTEND_DIST)) or not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


if __name__ == "__main__":  # `python main.py` → uvicorn on $PORT (Render style)
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0",
                port=int(os.environ.get("PORT", 8000)), reload=False)
