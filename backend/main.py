"""
SolarMind AI — FastAPI Backend Server
Provides REST API endpoints for the Solar Twin Dashboard.
Integrates PV Panel Defect Dataset (Kaggle) and Sarvam AI for analysis.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from data.simulator import (
    generate_site_data, generate_telemetry,
    generate_defect_history, generate_progression_forecast,
    generate_weather_forecast,
)
from engine.recommendation import generate_recommendations, calculate_cps
from engine.forecasting import forecast_progression, generate_panel_history
from engine.classifier import classify_image_bytes, get_dataset_info
from engine.sarvam_client import generate_analysis, get_api_status
from models.vit_classifier import simulate_inference, get_model_info, simulate_batch_inference

# Global state (simulated database)
SITE_DATA: Dict[str, Any] = {}


def _find_panel(panel_id: str) -> Dict[str, Any]:
    """Find a panel by ID or raise HTTPException."""
    panels: List[Any] = SITE_DATA.get("panels", [])
    for i in range(len(panels)):
        p: Dict[str, Any] = panels[i]
        if p["id"] == panel_id:
            return p
    raise HTTPException(status_code=404, detail=f"Panel {panel_id} not found")


@asynccontextmanager
async def lifespan(app: Any) -> AsyncGenerator[None, None]:
    """Initialize simulated data on startup."""
    global SITE_DATA
    SITE_DATA = generate_site_data()
    panels: List[Any] = SITE_DATA["panels"]
    print(f"SolarMind AI Backend initialized with {len(panels)} panels")
    yield
    print("SolarMind AI Backend shutting down")


app = FastAPI(
    title="SolarMind AI API",
    description="Decision-Intelligent Predictive Maintenance for Solar Farms",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# IMAGE ANALYSIS ENDPOINTS (Kaggle + Sarvam AI)
# ──────────────────────────────────────────────

@app.post("/api/analyze")
async def analyze_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a solar panel image for defect analysis.
    Uses PV Panel Defect Dataset model + Sarvam AI for recommendations.
    """
    # Validate file type
    filename: str = file.filename or "upload.jpg"
    valid_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]
    ext: str = ""
    for e in valid_extensions:
        if filename.lower().endswith(e):
            ext = e
            break
    if not ext:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPG, PNG, or BMP image.",
        )

    # Read file content
    content: bytes = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(content) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    # Classify the image
    classification: Dict[str, Any] = classify_image_bytes(content, filename)

    # Generate AI analysis via Sarvam AI
    predicted_class: str = str(classification["predicted_class"])
    confidence: float = float(classification["confidence"])
    probabilities: Dict[str, float] = classification.get("probabilities", {})

    analysis: Dict[str, Any] = generate_analysis(
        predicted_class, confidence, probabilities
    )

    return {
        "classification": classification,
        "analysis": analysis,
        "filename": filename,
        "file_size_bytes": len(content),
    }


@app.get("/api/dataset/info")
async def dataset_info() -> Dict[str, Any]:
    """Get PV Panel Defect Dataset information."""
    return get_dataset_info()


@app.get("/api/sarvam/status")
async def sarvam_status() -> Dict[str, Any]:
    """Check Sarvam AI API status."""
    return get_api_status()


# ──────────────────────────────────────────────
# SITE & PANEL ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "SolarMind AI API v1.0", "status": "online"}


@app.get("/api/site")
async def get_site_overview() -> Dict[str, Any]:
    """Get site-level overview with KPIs and zone health."""
    return {
        "site_id": SITE_DATA["site_id"],
        "site_name": SITE_DATA["site_name"],
        "location": SITE_DATA["location"],
        "capacity_mw": SITE_DATA["capacity_mw"],
        "kpis": SITE_DATA["kpis"],
        "zone_health": SITE_DATA["zone_health"],
        "last_updated": SITE_DATA["last_updated"],
    }


@app.get("/api/panels")
async def get_panels(
    zone: Optional[str] = None,
    defect: Optional[str] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Get all panels with optional filtering."""
    panels: List[Dict[str, Any]] = list(SITE_DATA["panels"])

    if zone:
        panels = [p for p in panels if p["zone"] == zone]
    if defect:
        panels = [p for p in panels if p["defect"] == defect]
    if status:
        panels = [p for p in panels if p["status"] == status]

    return {"total": len(panels), "panels": panels}


@app.get("/api/panels/{panel_id}")
async def get_panel_detail(panel_id: str) -> Dict[str, Any]:
    """Get detailed information for a specific panel."""
    panel: Dict[str, Any] = _find_panel(panel_id)
    history = generate_panel_history(panel)
    forecast = forecast_progression(panel)
    weather_forecast: Optional[List[Dict[str, Any]]] = SITE_DATA.get("weather_forecast")
    recommendation = calculate_cps(panel, weather_forecast)
    return {
        "panel": panel, "history": history,
        "forecast": forecast, "recommendation": recommendation,
    }


# ──────────────────────────────────────────────
# RECOMMENDATION, FORECAST, DETECTION ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/api/recommendations")
async def get_recommendations(limit: int = 20) -> Dict[str, Any]:
    weather_forecast: Optional[List[Dict[str, Any]]] = SITE_DATA.get("weather_forecast")
    all_recs: List[Dict[str, Any]] = generate_recommendations(SITE_DATA["panels"], weather_forecast)
    limited: List[Dict[str, Any]] = []
    for i in range(min(limit, len(all_recs))):
        limited.append(all_recs[i])
    return {"total": len(all_recs), "recommendations": limited}


@app.get("/api/forecast/{panel_id}")
async def get_forecast(panel_id: str, days: int = 90) -> Dict[str, Any]:
    panel: Dict[str, Any] = _find_panel(panel_id)
    return forecast_progression(panel, days)


@app.get("/api/detect/{panel_id}")
async def detect_defect(panel_id: str) -> Dict[str, Any]:
    _find_panel(panel_id)
    return simulate_inference(panel_id)


@app.get("/api/detect/batch/{count}")
async def batch_detect(count: int = 10) -> Dict[str, Any]:
    results = simulate_batch_inference(min(count, 50))
    return {"total": len(results), "results": results}


@app.get("/api/kpis")
async def get_kpis() -> Dict[str, Any]:
    return dict(SITE_DATA["kpis"])


@app.get("/api/weather")
async def get_weather() -> Dict[str, Any]:
    return {"forecast": SITE_DATA.get("weather_forecast", [])}


@app.get("/api/model/info")
async def get_model() -> Dict[str, Any]:
    return get_model_info()


@app.get("/api/zones")
async def get_zones() -> Any:
    return SITE_DATA["zone_health"]


@app.get("/api/telemetry/{panel_id}")
async def get_telemetry(panel_id: str, days: int = 7) -> Dict[str, Any]:
    _find_panel(panel_id)
    telemetry: List[Any] = generate_telemetry(panel_id, min(days, 30))
    start_idx: int = max(0, len(telemetry) - 100)
    last_entries: List[Any] = []
    for i in range(start_idx, len(telemetry)):
        last_entries.append(telemetry[i])
    return {
        "panel_id": panel_id, "days": days,
        "data_points": len(telemetry), "telemetry": last_entries,
    }


if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
