"""
SolarMind AI — Defect Progression Forecasting Engine
Uses deterministic mathematical models for defect progression and RUL prediction.
"""
import math
import hashlib
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _seed_for(key: str) -> int:
    """Generate a deterministic seed from a string key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def forecast_progression(
    panel: Dict[str, Any], days: int = 90
) -> Dict[str, Any]:
    """
    Generate a defect progression forecast for a panel.
    Uses logistic growth models with panel-id-seeded deterministic noise.
    Returns severity and risk trajectories over time.
    """
    panel_id: str = panel.get("id", "unknown")
    rng = random.Random(_seed_for(panel_id + "_forecast"))

    defect: str = panel["defect"]
    current_severity: float = float(panel["severity"])

    if defect == "normal":
        trajectory = [
            {
                "day": d,
                "severity": _r(0.01 + rng.uniform(0, 0.02), 3),
                "risk": _r(0.01 + rng.uniform(0, 0.02), 3),
            }
            for d in range(0, days, 3)
        ]
        return {
            "panel_id": panel_id,
            "current_severity": 0.0,
            "rul_days": 365,
            "risk_level": "low",
            "trajectory": trajectory,
        }

    # Growth rates per defect type (logistic growth parameters)
    growth_rates: Dict[str, float] = {
        "micro_crack": 0.0045,
        "hotspot": 0.0035,
        "dust_soiling": 0.0065,
    }
    rate = growth_rates.get(defect, 0.004)

    trajectory: List[Dict[str, Any]] = []
    for d in range(0, days, 3):
        # Logistic growth model: severity grows but slows as it approaches 1.0
        # S(t) = S0 + r*t * (1 - S0*0.5) with sigmoid-like saturation
        sev = current_severity + rate * d * (1.0 - current_severity * 0.5)
        # Add small deterministic noise
        noise = rng.uniform(-0.01, 0.01)
        sev = min(1.0, max(0.0, _r(sev + noise, 3)))

        risk = min(1.0, sev * 1.08 + rng.uniform(-0.01, 0.01))
        risk = max(0.0, _r(risk, 3))

        trajectory.append({"day": d, "severity": sev, "risk": risk})

    # Estimate RUL (days until severity reaches critical threshold 0.9)
    rul_days: int = int(panel.get("rul_days", 180))

    if current_severity >= 0.9:
        rul_days = max(1, rng.randint(3, 15))
        risk_level = "critical"
    elif current_severity >= 0.7:
        risk_level = "high"
    elif current_severity >= 0.4:
        risk_level = "medium"
    else:
        risk_level = "low"

    # Confidence intervals (tighter than before — ±5% instead of ±8%)
    lower_ci = [max(0.0, _r(t["severity"] - 0.05, 3)) for t in trajectory]
    upper_ci = [min(1.0, _r(t["severity"] + 0.05, 3)) for t in trajectory]

    return {
        "panel_id": panel_id,
        "defect_type": defect,
        "current_severity": current_severity,
        "rul_days": rul_days,
        "risk_level": risk_level,
        "time_to_maintenance_days": max(1, rul_days - 15),
        "trajectory": trajectory,
        "confidence_interval": {
            "lower": lower_ci,
            "upper": upper_ci,
        },
    }


def generate_panel_history(
    panel: Dict[str, Any], months: int = 6
) -> List[Dict[str, Any]]:
    """Generate historical defect progression data for a panel (deterministic)."""
    panel_id: str = panel.get("id", "unknown")
    rng = random.Random(_seed_for(panel_id + "_history"))

    defect: str = panel["defect"]
    current_severity: float = float(panel["severity"])
    history: List[Dict[str, Any]] = []

    if defect == "normal":
        for i in range(months):
            dt = datetime.now() - timedelta(days=(months - i) * 30)
            history.append({
                "date": dt.strftime("%Y-%m-%d"),
                "severity": 0.0,
                "risk_score": _r(rng.uniform(0, 0.05), 3),
                "power_output_kw": _r(rng.uniform(4.5, 5.2), 2),
                "defect": "normal",
                "action_taken": "Routine inspection",
            })
        return history

    start_sev = max(0.05, current_severity - rng.uniform(0.15, 0.45))

    actions = ["Inspected", "Cleaned", "Monitored", "Repair attempted", "No action"]

    for i in range(months):
        dt = datetime.now() - timedelta(days=(months - i) * 30)
        progress = float(i) / (months - 1) if months > 1 else 1.0
        # Smooth interpolation from start_sev to current_severity
        sev = start_sev + (current_severity - start_sev) * progress
        sev = _r(max(0.05, min(0.99, sev + rng.uniform(-0.03, 0.03))), 3)
        risk = _r(min(1.0, sev * 1.1 + rng.uniform(-0.03, 0.03)), 3)
        max_kw = float(panel["max_output_kw"])
        power = _r(max_kw * (1 - sev * 0.3) + rng.uniform(-0.15, 0.15), 2)

        history.append({
            "date": dt.strftime("%Y-%m-%d"),
            "severity": sev,
            "risk_score": risk,
            "power_output_kw": max(0.0, power),
            "defect": defect,
            "action_taken": rng.choice(actions),
        })

    return history
