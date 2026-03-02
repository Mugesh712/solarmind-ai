"""
SolarMind AI — Sarvam AI Client
Integrates Sarvam AI's chat completion API for generating
AI-powered maintenance analysis and recommendations.

Sarvam AI: https://sarvam.ai
API Docs: https://docs.sarvam.ai
"""
import os
import json
from typing import Any, Dict, List, Optional

# API Configuration
def _get_api_key() -> str:
    """Read API key dynamically so it picks up changes after import."""
    return os.environ.get("SARVAM_API_KEY", "")
SARVAM_API_URL: str = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_MODEL: str = "sarvam-m"


def _has_requests() -> bool:
    """Check if requests library is available."""
    try:
        import requests  # type: ignore
        return True
    except ImportError:
        return False


def _has_sarvamai() -> bool:
    """Check if sarvamai SDK is available."""
    try:
        import sarvamai  # type: ignore
        return True
    except ImportError:
        return False


def get_api_status() -> Dict[str, Any]:
    """Check Sarvam AI API configuration status."""
    return {
        "api_configured": bool(_get_api_key()),
        "model": SARVAM_MODEL,
        "sdk_available": _has_sarvamai(),
        "requests_available": _has_requests(),
    }


def generate_analysis(
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    panel_id: str = "",
) -> Dict[str, Any]:
    """
    Generate an AI-powered analysis report for a classified solar panel image.

    Uses Sarvam AI's chat completion API to generate natural language
    maintenance recommendations based on the classification result.
    """
    # Build the analysis prompt
    prob_str: str = ", ".join(
        f"{cls}: {prob:.1%}" for cls, prob in probabilities.items()
    )

    prompt: str = f"""You are SolarMind AI, an expert solar panel maintenance advisor. 
Analyze the following solar panel defect classification result and provide a brief, actionable maintenance report.

**Classification Result:**
- Predicted Defect: {predicted_class}
- Confidence: {confidence:.1%}
- All Probabilities: {prob_str}
{f'- Panel ID: {panel_id}' if panel_id else ''}

Provide your analysis in this format:
1. **Diagnosis**: Brief description of what "{predicted_class}" means
2. **Severity**: Low/Medium/High/Critical
3. **Recommended Action**: What maintenance action to take
4. **Timeline**: When this should be addressed
5. **Impact**: Estimated impact on panel efficiency if not addressed

Keep your response concise (under 200 words)."""

    # Try Sarvam AI API first
    if _get_api_key() and _has_requests():
        api_result: Optional[str] = _call_sarvam_api(prompt)
        if api_result is not None:
            return {
                "analysis": api_result,
                "source": "sarvam-ai",
                "model": SARVAM_MODEL,
                "panel_id": panel_id,
            }

    # Fallback: generate a template-based analysis
    return {
        "analysis": _generate_fallback_analysis(predicted_class, confidence),
        "source": "template",
        "model": "local-template",
        "panel_id": panel_id,
        "note": "Install sarvamai and set SARVAM_API_KEY for AI-powered analysis",
    }


def _call_sarvam_api(prompt: str) -> Optional[str]:
    """Call Sarvam AI chat completion API."""
    import requests  # type: ignore

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_api_key()}",
    }

    payload: Dict[str, Any] = {
        "model": SARVAM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are SolarMind AI, an expert solar panel maintenance advisor specializing in defect analysis and predictive maintenance.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 500,
    }

    try:
        response = requests.post(
            SARVAM_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        choices: List[Any] = data.get("choices", [])
        if len(choices) > 0:
            message: Dict[str, Any] = choices[0].get("message", {})
            content: str = str(message.get("content", ""))
            if content:
                return content
        return None
    except Exception as e:
        print(f"Sarvam AI API error: {e}")
        return None


def _generate_fallback_analysis(predicted_class: str, confidence: float) -> str:
    """Generate template-based analysis when API is not available."""
    analyses: Dict[str, Dict[str, str]] = {
        "Bird-drop": {
            "diagnosis": "Bird droppings detected on panel surface, causing localized shading and potential hotspot formation.",
            "severity": "Medium",
            "action": "Schedule panel cleaning within 1 week. Inspect for any corrosion damage underneath.",
            "timeline": "Within 7 days",
            "impact": "5-15% efficiency reduction in affected cells. Risk of hotspot damage if left unaddressed.",
        },
        "Clean": {
            "diagnosis": "Panel appears clean and free of visible defects. Normal operating condition.",
            "severity": "Low",
            "action": "No immediate action required. Continue regular monitoring schedule.",
            "timeline": "Next scheduled inspection",
            "impact": "No efficiency impact. Panel operating at expected capacity.",
        },
        "Dusty": {
            "diagnosis": "Significant dust/soiling accumulation detected on panel surface.",
            "severity": "Medium",
            "action": "Schedule panel cleaning. Consider installing automated cleaning system for recurring issues.",
            "timeline": "Within 3-5 days",
            "impact": "10-25% efficiency reduction depending on dust density. Cumulative degradation risk.",
        },
        "Electrical-damage": {
            "diagnosis": "Electrical damage detected — potential burnt cells, bypass diode failure, or wiring issues.",
            "severity": "Critical",
            "action": "Immediate isolation required. Dispatch technician for electrical inspection and repair/replacement.",
            "timeline": "Within 24 hours",
            "impact": "50-100% panel output loss. Fire hazard risk if not addressed immediately.",
        },
        "Physical-Damage": {
            "diagnosis": "Physical damage detected — cracks, chips, or broken glass on panel surface.",
            "severity": "High",
            "action": "Schedule panel replacement. Apply temporary protective covering to prevent water ingress.",
            "timeline": "Within 48-72 hours",
            "impact": "20-80% efficiency reduction. Risk of moisture damage and complete panel failure.",
        },
        "Snow-Covered": {
            "diagnosis": "Snow/ice coverage detected on panel surface, blocking sunlight completely.",
            "severity": "Medium",
            "action": "If persistent, use snow removal tools or activate de-icing system. Monitor weather forecast.",
            "timeline": "Within 1-2 days (weather dependent)",
            "impact": "90-100% temporary output loss. Risk of micro-cracks from thermal stress during thaw cycle.",
        },
    }

    info: Dict[str, str] = analyses.get(predicted_class, {
        "diagnosis": f"Defect type '{predicted_class}' detected.",
        "severity": "Medium",
        "action": "Schedule inspection for detailed assessment.",
        "timeline": "Within 1 week",
        "impact": "Potential efficiency reduction. Further analysis needed.",
    })

    conf_pct: str = f"{confidence:.1%}"

    report: str = f"""**Diagnosis**: {info['diagnosis']}
**Confidence**: {conf_pct}
**Severity**: {info['severity']}
**Recommended Action**: {info['action']}
**Timeline**: {info['timeline']}
**Impact**: {info['impact']}

_Note: This analysis was generated using built-in templates. Set up your Sarvam AI API key (`SARVAM_API_KEY`) for enhanced AI-powered analysis._"""

    return report
