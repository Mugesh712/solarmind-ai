"""
SolarMind AI — CPS-Based Maintenance Recommendation Engine
Generates prioritized maintenance actions based on Composite Priority Scoring.
"""
import math
from datetime import datetime
from typing import Any, Dict, List, Optional


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _calculate_crew_availability() -> float:
    """
    Calculate crew availability based on current day and time.
    Weekday daytime (8-18): 0.90
    Weekday evening (18-22): 0.50
    Weekend daytime (8-18): 0.35
    Night hours (22-8): 0.15
    """
    now = datetime.now()
    hour = now.hour
    is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6

    if 8 <= hour < 18:
        return 0.35 if is_weekend else 0.90
    elif 18 <= hour < 22:
        return 0.25 if is_weekend else 0.50
    else:
        return 0.15


def calculate_cps(
    panel: Dict[str, Any], weather_forecast: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Calculate Composite Priority Score (CPS) for a panel.

    CPS = w1×Severity + w2×EnergyLoss + w3×WeatherSuitability + w4×CrewAvailability + w5×Urgency
    """
    if panel["defect"] == "normal":
        return {"cps": 0.0, "action": "No action needed", "priority": "P4", "priority_label": "Low"}

    w1, w2, w3, w4, w5 = 0.25, 0.25, 0.15, 0.10, 0.25

    # Severity score
    defect_weights: Dict[str, float] = {"micro_crack": 0.9, "hotspot": 0.85, "dust_soiling": 0.6}
    confidence: float = float(panel["confidence"])
    f_severity: float = confidence * defect_weights.get(panel["defect"], 0.5)

    # Energy loss score (normalized)
    max_kw: float = float(panel["max_output_kw"])
    max_possible_loss: float = max_kw * 24
    energy_loss: float = float(panel["energy_loss_kwh_day"])
    f_energy: float = min(1.0, energy_loss / max_possible_loss) if max_possible_loss > 0 else 0.0

    # Weather suitability (from forecast)
    f_weather: float = 0.7
    if weather_forecast and len(weather_forecast) > 0:
        total_cleaning: float = 0.0
        count: int = min(2, len(weather_forecast))
        for i in range(count):
            w_item: Dict[str, Any] = weather_forecast[i]
            total_cleaning += float(w_item.get("cleaning_suitability", 0.5))
        avg_cleaning: float = total_cleaning / count
        f_weather = max(0.0, min(1.0, avg_cleaning))

    # Crew availability (based on current day/time)
    f_crew: float = _calculate_crew_availability()

    # Urgency (based on RUL)
    max_rul: float = 365.0
    rul: float = float(panel.get("rul_days", 180))
    f_urgency: float = max(0.0, 1.0 - (rul / max_rul))

    cps = _r(w1 * f_severity + w2 * f_energy + w3 * f_weather + w4 * f_crew + w5 * f_urgency, 3)

    # Determine action and priority
    action: str
    priority: str
    priority_label: str

    if cps >= 0.8:
        action = "Replace panel within 72hrs"
        priority = "P1"
        priority_label = "Critical"
    elif cps >= 0.6:
        if panel["defect"] == "dust_soiling":
            action = "Clean now — yield recovery expected"
        else:
            action = "Schedule repair within 1 week"
        priority = "P2"
        priority_label = "High"
    elif cps >= 0.4:
        action = "Inspect within 48hrs"
        priority = "P3"
        priority_label = "Medium"
    else:
        action = "Recheck in 24hrs — monitor"
        priority = "P4"
        priority_label = "Low"

    yield_recovery = _r(energy_loss * 0.82, 2) if energy_loss > 0 else 0.0
    cost_recovery = _r(yield_recovery * 0.08, 2)

    return {
        "cps": cps,
        "action": action,
        "priority": priority,
        "priority_label": priority_label,
        "scores": {
            "severity": _r(f_severity, 3),
            "energy_loss": _r(f_energy, 3),
            "weather": _r(f_weather, 3),
            "crew": _r(f_crew, 3),
            "urgency": _r(f_urgency, 3),
        },
        "expected_yield_recovery_kwh": yield_recovery,
        "expected_cost_recovery_usd": cost_recovery,
        "rationale": _generate_rationale(panel, cps, f_weather),
    }


def _generate_rationale(panel: Dict[str, Any], cps: float, weather_score: float) -> str:
    """Generate human-readable rationale for the recommendation."""
    defect = panel["defect"].replace("_", " ").title()
    sev: float = float(panel["severity"])
    loss: float = float(panel["energy_loss_kwh_day"])
    conf: float = float(panel["confidence"])

    parts: List[str] = [f"{defect} detected with {conf*100:.0f}% confidence."]

    if sev > 0.7:
        parts.append(f"High severity ({sev:.0%}) — immediate attention required.")
    elif sev > 0.4:
        parts.append(f"Moderate severity ({sev:.0%}) — schedule maintenance soon.")
    else:
        parts.append(f"Low severity ({sev:.0%}) — continue monitoring.")

    if loss > 10:
        parts.append(f"Significant energy loss of {loss:.1f} kWh/day.")
    elif loss > 5:
        parts.append(f"Notable energy loss of {loss:.1f} kWh/day.")

    if weather_score > 0.7:
        parts.append("Weather conditions favorable for maintenance.")
    elif weather_score < 0.3:
        parts.append("Poor weather forecast — reschedule if non-critical.")

    rul_val = int(panel.get("rul_days", 999))
    if rul_val < 30:
        parts.append(f"⚠️ Estimated RUL only {rul_val} days — urgent action needed.")

    return " ".join(parts)


def generate_recommendations(
    panels: List[Dict[str, Any]],
    weather_forecast: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Generate recommendations for all faulty panels, sorted by priority."""
    recommendations: List[Dict[str, Any]] = []

    for panel in panels:
        if panel["defect"] == "normal":
            continue

        rec = calculate_cps(panel, weather_forecast)
        rec["panel_id"] = panel["id"]
        rec["defect"] = panel["defect"]
        rec["severity"] = panel["severity"]
        rec["zone"] = panel["zone"]
        rec["energy_loss_kwh_day"] = panel["energy_loss_kwh_day"]
        rec["rul_days"] = panel.get("rul_days", 999)
        recommendations.append(rec)

    # Sort by CPS descending (highest priority first)
    recommendations.sort(key=lambda x: float(x["cps"]), reverse=True)

    return recommendations
