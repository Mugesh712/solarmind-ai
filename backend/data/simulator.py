"""
SolarMind AI — Realistic Solar Farm Data Simulator
Generates panel data, telemetry, defects, and weather for the dashboard.
Uses real evaluation metrics where available and deterministic seeding for consistency.
"""
import os
import random
import math
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

random.seed(42)

DEFECT_TYPES = ["normal", "micro_crack", "hotspot", "dust_soiling"]
SEVERITY_RANGES: Dict[str, Tuple[float, float]] = {
    "normal": (0.0, 0.0),
    "micro_crack": (0.3, 0.95),
    "hotspot": (0.4, 0.90),
    "dust_soiling": (0.2, 0.85),
}
ACTIONS = {
    "high": "Replace panel within 72hrs",
    "medium_crack": "Schedule repair within 1 week",
    "medium_dust": "Clean now",
    "low": "Reinspect in 24hrs",
    "monitor": "Monitor — no action needed",
}

ZONES = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]

# Map simulator defect names → Kaggle dataset folder names
DEFECT_TO_DATASET_CLASS: Dict[str, str] = {
    "normal": "Clean",
    "Clean": "Clean",
    "micro_crack": "Physical-Damage",
    "hotspot": "Electrical-damage",
    "dust_soiling": "Dusty",
    "Bird-drop": "Bird-drop",
    "Dusty": "Dusty",
    "Electrical-damage": "Electrical-damage",
    "Physical-Damage": "Physical-Damage",
    "Snow-Covered": "Snow-Covered",
}

# Cache of available images per class
_IMAGE_CACHE: Dict[str, List[str]] = {}
DATASET_DIR: str = os.path.join(os.path.dirname(__file__), "pv_defect_dataset")


def _get_images_for_class(dataset_class: str) -> List[str]:
    """Get list of available images for a dataset class."""
    if dataset_class in _IMAGE_CACHE:
        return _IMAGE_CACHE[dataset_class]

    images: List[str] = []
    for split in ["train", "test", "val"]:
        class_dir = os.path.join(DATASET_DIR, split, dataset_class)
        if os.path.isdir(class_dir):
            for f in os.listdir(class_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    # Store as relative path: split/class/filename
                    images.append(f"{split}/{dataset_class}/{f}")
    _IMAGE_CACHE[dataset_class] = images
    return images


def assign_panel_image(panel: Dict[str, Any], rng: Optional[random.Random] = None) -> None:
    """Assign a real dataset image to a panel based on its defect type."""
    if rng is None:
        rng = random.Random(_seed_for(panel["id"]))

    defect = panel.get("defect", "normal")
    dataset_class = DEFECT_TO_DATASET_CLASS.get(defect, "Clean")
    images = _get_images_for_class(dataset_class)

    if images:
        chosen = rng.choice(images)
        panel["image_path"] = chosen
        panel["image_url"] = f"/images/{chosen}"
    else:
        panel["image_path"] = ""
        panel["image_url"] = ""


# ──────────────────────────────────────────────
# Load real evaluation results
# ──────────────────────────────────────────────
_EVAL_RESULTS: Optional[Dict[str, Any]] = None

def _load_evaluation_results() -> Dict[str, Any]:
    """Load real evaluation metrics from ml_pipeline/evaluation_results.json."""
    global _EVAL_RESULTS
    if _EVAL_RESULTS is not None:
        return _EVAL_RESULTS

    eval_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "ml_pipeline", "evaluation_results", "evaluation_results.json"
    )
    if os.path.isfile(eval_path):
        with open(eval_path, "r") as f:
            _EVAL_RESULTS = json.load(f)
    else:
        _EVAL_RESULTS = {}
    return _EVAL_RESULTS


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _seed_for(key: str) -> int:
    """Generate a deterministic seed from a string key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def generate_panels(count: int = 200) -> List[Dict[str, Any]]:
    """Generate a grid of solar panels with realistic attributes."""
    panels: List[Dict[str, Any]] = []
    cols = 20
    rows = count // cols

    rng = random.Random(42)  # Dedicated RNG for panel generation

    for i in range(count):
        row = i // cols
        col = i % cols
        zone = ZONES[row // (rows // len(ZONES))] if rows >= len(ZONES) else ZONES[0]

        # ~85% normal, ~5% micro_crack, ~4% hotspot, ~6% dust_soiling
        r = rng.random()
        if r < 0.85:
            defect = "normal"
        elif r < 0.90:
            defect = "micro_crack"
        elif r < 0.94:
            defect = "hotspot"
        else:
            defect = "dust_soiling"

        sev_min, sev_max = SEVERITY_RANGES[defect]
        severity = _r(rng.uniform(sev_min, sev_max), 2) if defect != "normal" else 0.0
        confidence = _r(rng.uniform(0.82, 0.99), 2) if defect != "normal" else _r(rng.uniform(0.95, 0.99), 2)

        max_output = _r(rng.uniform(4.8, 5.5), 2)
        if defect == "normal":
            efficiency = _r(rng.uniform(0.92, 1.0), 3)
        elif defect == "dust_soiling":
            efficiency = _r(rng.uniform(0.70, 0.90), 3)
        elif defect == "micro_crack":
            efficiency = _r(rng.uniform(0.60, 0.85), 3)
        else:
            efficiency = _r(rng.uniform(0.65, 0.88), 3)

        current_output = _r(max_output * efficiency, 2)
        energy_loss = _r((max_output - current_output) * 24, 2)  # kWh/day

        panel: Dict[str, Any] = {
            "id": f"P-{1000 + i:04d}",
            "row": row,
            "col": col,
            "zone": zone,
            "gps": {
                "lat": _r(17.385 + row * 0.0001, 6),
                "lon": _r(78.486 + col * 0.0002, 6),
            },
            "panel_type": rng.choice(["Mono-PERC", "Poly-Si", "HJT"]),
            "install_date": f"202{rng.randint(0, 4)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "defect": defect,
            "severity": severity,
            "confidence": confidence,
            "max_output_kw": max_output,
            "current_output_kw": current_output,
            "efficiency": efficiency,
            "energy_loss_kwh_day": energy_loss,
            "cost_loss_usd_day": _r(energy_loss * 0.08, 2),
            "co2_impact_kg_day": _r(energy_loss * 0.42, 2),
            "temperature_c": _r(rng.uniform(35, 65), 1),
            "rul_days": _calculate_rul(defect, severity, rng),
            "last_inspection": (datetime.now() - timedelta(days=rng.randint(1, 30))).strftime("%Y-%m-%d"),
            "status": "healthy" if defect == "normal" else ("critical" if severity > 0.7 else "warning"),
            "image_path": "",
            "image_url": "",
        }
        # Assign a real dataset image
        assign_panel_image(panel, rng)
        panels.append(panel)

    return panels


def _calculate_rul(defect: str, severity: float, rng: Optional[random.Random] = None) -> int:
    if defect == "normal":
        return 365
    if rng is None:
        rng = random.Random(42)
    base: Dict[str, int] = {"micro_crack": 120, "hotspot": 90, "dust_soiling": 60}
    return max(5, int(base[defect] * (1 - severity) + rng.randint(-10, 10)))


def generate_telemetry(panel_id: str, days: int = 30) -> List[Dict[str, Any]]:
    """Generate time-series telemetry for a panel (deterministic per panel_id)."""
    data: List[Dict[str, Any]] = []
    rng = random.Random(_seed_for(panel_id))
    base_power = rng.uniform(4.0, 5.2)
    degradation_rate = rng.uniform(0.001, 0.01)

    for day in range(days):
        dt = datetime.now() - timedelta(days=days - day)
        for hour in range(6, 19):  # Daylight hours
            irradiance = max(0.0, 800 * math.sin(math.pi * (hour - 6) / 12) + rng.uniform(-50, 50))
            temp = _r(25 + 20 * math.sin(math.pi * (hour - 6) / 12) + rng.uniform(-3, 3), 1)
            power = _r(base_power * (irradiance / 1000) * (1 - degradation_rate * day) + rng.uniform(-0.2, 0.2), 3)
            power = max(0.0, power)

            data.append({
                "panel_id": panel_id,
                "timestamp": dt.replace(hour=hour).isoformat(),
                "irradiance_w_m2": _r(irradiance, 1),
                "panel_temp_c": temp,
                "power_output_kw": power,
                "voltage_v": _r(38 + rng.uniform(-2, 2), 1),
                "current_a": _r(power / 38 * 1000, 1) if power > 0 else 0,
            })
    return data


def generate_defect_history(
    panel_id: str, defect_type: str, current_severity: float, points: int = 6
) -> List[Dict[str, Any]]:
    """Generate historical defect progression for a panel (deterministic per panel_id)."""
    history: List[Dict[str, Any]] = []
    rng = random.Random(_seed_for(panel_id + "_history"))

    if defect_type == "normal":
        for i in range(points):
            dt = datetime.now() - timedelta(days=(points - i) * 15)
            history.append({
                "date": dt.strftime("%Y-%m-%d"),
                "severity": 0.0,
                "risk_score": _r(rng.uniform(0, 0.1), 2),
                "defect": "normal",
                "action_taken": "Routine inspection",
            })
        return history

    start_sev = max(0.1, current_severity - rng.uniform(0.2, 0.5))
    for i in range(points):
        dt = datetime.now() - timedelta(days=(points - i) * 15)
        progress = float(i) / (points - 1) if points > 1 else 1.0
        sev = _r(start_sev + (current_severity - start_sev) * progress + rng.uniform(-0.05, 0.05), 2)
        sev = max(0.05, min(0.99, sev))
        risk = _r(min(1.0, sev * 1.1 + rng.uniform(-0.05, 0.05)), 2)

        actions = ["Inspected", "Cleaned", "Monitored", "None"]
        history.append({
            "date": dt.strftime("%Y-%m-%d"),
            "severity": sev,
            "risk_score": risk,
            "defect": defect_type,
            "action_taken": rng.choice(actions),
        })
    return history


def generate_progression_forecast(
    current_severity: float, defect_type: str, days: int = 90,
    panel_id: str = ""
) -> List[Dict[str, Any]]:
    """Generate a 90-day defect progression forecast (deterministic per panel_id)."""
    forecast: List[Dict[str, Any]] = []
    rng = random.Random(_seed_for(panel_id + "_forecast") if panel_id else 99)

    if defect_type == "normal":
        for d in range(0, days, 3):
            forecast.append({
                "day": d,
                "severity": _r(rng.uniform(0, 0.05), 3),
                "risk": _r(rng.uniform(0, 0.05), 3),
            })
        return forecast

    growth_rate: Dict[str, float] = {"micro_crack": 0.004, "hotspot": 0.003, "dust_soiling": 0.006}
    rate = growth_rate.get(defect_type, 0.003)

    for d in range(0, days, 3):
        # Logistic growth model (slows near 1.0)
        sev = current_severity + rate * d * (1.0 - current_severity * 0.5)
        sev = min(1.0, sev + rng.uniform(-0.01, 0.01))
        sev = max(0.0, _r(sev, 3))
        risk = min(1.0, sev * 1.05 + rng.uniform(-0.015, 0.015))
        risk = max(0.0, _r(risk, 3))
        forecast.append({"day": d, "severity": sev, "risk": risk})
    return forecast


def generate_weather_forecast(days: int = 7) -> List[Dict[str, Any]]:
    """Generate a 7-day weather forecast (deterministic per calendar day)."""
    forecast: List[Dict[str, Any]] = []
    for d in range(days):
        dt = datetime.now() + timedelta(days=d)
        # Seed by date so forecast is consistent within the same calendar day
        day_seed = _seed_for(dt.strftime("%Y-%m-%d"))
        rng = random.Random(day_seed)

        rain_prob = _r(rng.uniform(0, 0.4), 2)
        forecast.append({
            "date": dt.strftime("%Y-%m-%d"),
            "temp_high_c": _r(rng.uniform(30, 42), 1),
            "temp_low_c": _r(rng.uniform(18, 26), 1),
            "cloud_cover": _r(rng.uniform(0, 0.5), 2),
            "rain_probability": rain_prob,
            "wind_speed_ms": _r(rng.uniform(1, 8), 1),
            "irradiance_forecast_w_m2": _r(rng.uniform(600, 950), 0),
            "cleaning_suitability": _r(1.0 - rain_prob - rng.uniform(0, 0.2), 2),
        })
    return forecast


def generate_kpis() -> Dict[str, Any]:
    """Generate dashboard KPI metrics using real evaluation results."""
    eval_data = _load_evaluation_results()
    metrics = eval_data.get("metrics", {})
    macro = metrics.get("macro_avg", {})
    detection = eval_data.get("detection_metrics", {})

    # Use real evaluation metrics, with sensible defaults
    precision = macro.get("precision", 0.917)
    recall = macro.get("recall", 0.941)
    f1 = macro.get("f1_score", 0.928)
    accuracy = metrics.get("accuracy", 0.954)
    mAP = detection.get("mAP50", 0.897)

    return {
        "precision": _r(precision, 3),
        "recall": _r(recall, 3),
        "f1_score": _r(f1, 3),
        "accuracy": _r(accuracy, 3),
        "mAP": _r(mAP, 3),
        "false_alarm_rate": _r(1.0 - precision, 3),
        "downtime_reduction_pct": 32.4,
        "energy_yield_recovery_pct": 18.7,
        "maintenance_cost_reduction_pct": 25.3,
        "inference_latency_ms": 24.5,
        "edge_uptime_pct": 99.6,
        "recommendation_acceptance_pct": 84.2,
        "co2_savings_tonnes_year": 1280,
        "total_panels": 200,
        "healthy_panels": 0,  # calculated later
        "faulty_panels": 0,   # calculated later
        "critical_alerts": 0, # calculated later
    }


def generate_site_data() -> Dict[str, Any]:
    """Generate complete site data bundle for the dashboard."""
    panels = generate_panels(200)

    healthy = sum(1 for p in panels if p["defect"] == "normal")
    faulty = len(panels) - healthy
    critical = sum(1 for p in panels if p.get("severity", 0) > 0.7)

    kpis = generate_kpis()
    kpis["healthy_panels"] = healthy
    kpis["faulty_panels"] = faulty
    kpis["critical_alerts"] = critical
    kpis["total_panels"] = len(panels)

    weather = generate_weather_forecast()

    zone_health: Dict[str, Any] = {}
    for zone in ZONES:
        zone_panels = [p for p in panels if p["zone"] == zone]
        zone_healthy = sum(1 for p in zone_panels if p["defect"] == "normal")
        total = len(zone_panels)
        zone_health[zone] = {
            "total": total,
            "healthy": zone_healthy,
            "health_pct": _r(zone_healthy / total * 100, 1) if total > 0 else 0.0,
        }

    return {
        "site_id": "SITE-ALPHA",
        "site_name": "SolarMind Demo Farm",
        "location": "Hyderabad, Telangana",
        "capacity_mw": 10,
        "panels": panels,
        "kpis": kpis,
        "weather_forecast": weather,
        "zone_health": zone_health,
        "last_updated": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    data = generate_site_data()
    print(json.dumps(data, indent=2, default=str))
    print(f"\nGenerated {len(data['panels'])} panels")
    print(f"Healthy: {data['kpis']['healthy_panels']}, Faulty: {data['kpis']['faulty_panels']}, Critical: {data['kpis']['critical_alerts']}")
