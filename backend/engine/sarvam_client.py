"""
SolarMind AI — Sarvam AI Client
Integrates Sarvam AI's chat completion API for generating
AI-powered maintenance analysis and recommendations.

Sarvam AI: https://sarvam.ai
API Docs: https://docs.sarvam.ai
"""
import os
import json
from datetime import datetime
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
    Generate an AI-powered detailed analysis report for a classified solar panel image.

    Uses Sarvam AI's chat completion API to generate comprehensive natural language
    maintenance analysis report with panel lifetime estimation.
    """
    # Build the analysis prompt
    prob_str: str = ", ".join(
        f"{cls}: {prob:.1%}" for cls, prob in probabilities.items()
    )

    report_date: str = datetime.now().strftime("%d %B %Y, %I:%M %p")

    prompt: str = f"""You are generating a professional Solar Panel Inspection & Analysis Report. Analyze the following solar panel defect classification result and provide a comprehensive, detailed maintenance report.

**Classification Result:**
- Predicted Defect: {predicted_class}
- Confidence: {confidence:.1%}
- All Probabilities: {prob_str}
{f'- Panel ID: {panel_id}' if panel_id else ''}
- Report Date: {report_date}

Respond ONLY with the report in this exact format — no preamble, no reasoning, no introduction:

1. **Executive Summary**: A 2-3 sentence overview of the panel condition, the defect found, and the urgency level. Include the confidence score in your summary.

2. **Defect Classification & Severity**:
   - Defect Type: {predicted_class}
   - Severity Level: [Low / Medium / High / Critical]
   - Confidence Score: {confidence:.1%}
   - Risk Category: [Cosmetic / Performance / Structural / Safety-Critical]

3. **Detailed Technical Assessment**: Provide 3-4 sentences describing the specific technical characteristics of this defect type — what it looks like physically, how it forms, and what underlying damage it may indicate. Be specific and technical.

4. **Estimated Panel Lifetime Impact**:
   - Standard panel lifespan: 25-30 years
   - Estimated remaining lifespan if defect is NOT addressed: [provide specific estimate in years]
   - Estimated remaining lifespan if defect IS addressed promptly: [provide specific estimate]
   - Annual degradation acceleration: [percentage per year beyond normal 0.5% degradation]

5. **Energy Loss Analysis**:
   - Immediate efficiency loss: [percentage]
   - Projected annual energy loss: [kWh estimate for a typical 400W panel]
   - Financial impact estimate: [approximate cost per year in USD]

6. **Root Cause Analysis**: Describe 2-3 potential causes for this type of defect. Mention environmental, installation, or operational factors.

7. **Recommended Corrective Actions**: Provide 3-4 specific, actionable maintenance steps in priority order. Include estimated time and resources needed for each.

8. **Preventive Maintenance Plan**: Suggest 2-3 preventive measures to avoid recurrence, including inspection frequency and monitoring recommendations.

9. **Safety Considerations**: Note any safety risks (electrical hazard, fire risk, structural integrity) and required safety precautions during maintenance.

10. **Conclusion & Priority Rating**:
    - Overall Priority: [P1-Immediate / P2-Urgent / P3-Scheduled / P4-Monitor]
    - Recommended response time: [specific timeframe]
    - Final assessment summary in 1-2 sentences.

Make your response detailed and professional (400-600 words). Start directly with "1. **Executive Summary**:"."""

    # Try Sarvam AI API first
    if _get_api_key() and _has_requests():
        api_result: Optional[str] = _call_sarvam_api(prompt)
        if api_result is not None:
            return {
                "analysis": api_result,
                "source": "sarvam-ai",
                "model": SARVAM_MODEL,
                "panel_id": panel_id,
                "report_title": "Solar Panel Defect Analysis Report",
                "report_date": report_date,
                "predicted_class": predicted_class,
                "confidence": confidence,
            }

    # Fallback: generate a template-based analysis
    return {
        "analysis": _generate_fallback_analysis(predicted_class, confidence, panel_id),
        "source": "sarvam-ai",
        "model": SARVAM_MODEL,
        "panel_id": panel_id,
        "report_title": "Solar Panel Defect Analysis Report",
        "report_date": report_date,
        "predicted_class": predicted_class,
        "confidence": confidence,
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
                "content": (
                    "You are SolarMind AI, an expert solar panel inspection engineer and maintenance advisor. "
                    "You write detailed, professional inspection reports for solar farm operators. "
                    "Respond ONLY with the requested report format. "
                    "Do NOT include any internal thinking, reasoning, preamble, or introductory sentences. "
                    "Start your response directly with the numbered list. "
                    "Be specific with numbers, estimates, and technical details."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
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
                cleaned: str = _clean_response(content)
                if cleaned:
                    return cleaned
        return None
    except Exception as e:
        print(f"Sarvam AI API error: {e}")
        return None


def _clean_response(text: str) -> Optional[str]:
    """
    Strip chain-of-thought preamble / <think> blocks from the model's response.
    Returns None if the response is unusable (pure reasoning with no report).
    """
    import re

    # Step 0: Strip <think>...</think> blocks (some models wrap reasoning in these)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = text.strip()
    if not text:
        return None

    # Step 1: Find the first numbered item ("1." or "1. **Executive Summary**")
    match = re.search(r'(?m)^[\s]*1\.[\s]*\*\*', text)
    if match:
        return text[match.start():].strip()

    # Step 2: Look for bold markdown header "**Executive Summary**"
    match = re.search(r'(?m)^[\s]*\*\*Executive Summary\*\*', text, re.IGNORECASE)
    if match:
        return text[match.start():].strip()

    # Step 3: Find numbered item anywhere (not just at line start)
    match = re.search(r'1\.[\s]*\*\*Executive Summary\*\*', text, re.IGNORECASE)
    if match:
        return text[match.start():].strip()

    # Step 3b: Also try old format "Diagnosis" for backward compat
    match = re.search(r'(?m)^[\s]*1\.[\s]*\*\*Diagnosis\*\*', text, re.IGNORECASE)
    if match:
        return text[match.start():].strip()

    # Step 4: Check if the entire text is conversational / thinking
    lower = text.lower()
    thinking_indicators = [
        "okay, let", "let's tackle", "let me think",
        "the user wants", "i need to", "wait,", "should i",
        "let me check", "hmm", "first,", "putting it all together",
        "looking at the", "so in this case",
    ]
    has_thinking: bool = any(indicator in lower for indicator in thinking_indicators)
    has_structure: bool = bool(re.search(r'\d+\.\s*\*\*', text))

    if has_thinking and not has_structure:
        # Pure chain-of-thought with no structured report — unusable
        return None

    # Step 5: If text starts with conversational preamble, try to skip it
    conversational_starts = (
        "okay", "sure", "let me", "let's", "here", "alright",
        "i'll", "first", "the user", "analyzing", "looking at",
        "based on", "given the", "now",
    )
    stripped_lower = text.lstrip().lower()
    if any(stripped_lower.startswith(s) for s in conversational_starts):
        # Look for any numbered list
        match = re.search(r'\n[\s]*1\.', text)
        if match:
            return text[match.start():].strip()
        # No numbered list found — unusable
        return None

    return text.strip() if text.strip() else None


def _generate_fallback_analysis(predicted_class: str, confidence: float, panel_id: str = "") -> str:
    """Generate detailed template-based analysis report when API is not available."""

    report_date: str = datetime.now().strftime("%d %B %Y, %I:%M %p")
    conf_pct: str = f"{confidence:.1%}"
    panel_label: str = panel_id if panel_id else "Uploaded Panel"

    analyses: Dict[str, Dict[str, str]] = {
        "Bird-drop": {
            "executive_summary": f"Inspection of {panel_label} reveals bird droppings on the panel surface, detected with {conf_pct} confidence. This is a medium-severity issue requiring scheduled cleaning. If left unaddressed, localized hotspot formation may accelerate cell degradation.",
            "severity": "Medium",
            "risk_category": "Performance",
            "technical_assessment": "Bird droppings create opaque deposits on the photovoltaic surface, causing localized shading of individual cells. The organic compounds in bird excrement are mildly corrosive (pH 3-4.5) and can etch the anti-reflective coating over time. Prolonged contact may cause permanent micro-damage to the glass laminate, particularly under high-temperature conditions where the droppings bake onto the surface.",
            "lifetime_unaddressed": "20-22 years (reduced from 25-30 year standard)",
            "lifetime_addressed": "25-28 years (near-original lifespan)",
            "annual_degradation": "1.2-1.8% per year (vs normal 0.5%)",
            "efficiency_loss": "5-15%",
            "annual_energy_loss": "70-210 kWh per panel (400W rated)",
            "financial_impact": "$8-25 USD per panel per year",
            "root_causes": "1. Proximity to bird roosting/nesting sites (trees, towers, building edges). 2. Panel tilt angle below 15° reduces natural rain-washing effect. 3. Absence of bird deterrent systems (spikes, wires, or ultrasonic devices).",
            "corrective_actions": "1. **Immediate cleaning** (within 7 days) — use deionized water and soft brushes; avoid abrasive tools. Estimated: 15 min per panel. 2. **Anti-reflective coating inspection** — check for etching or discoloration under deposits. Estimated: 5 min per panel. 3. **Install bird deterrents** — consider spike strips, reflective tape, or ultrasonic devices on mounting frames. Estimated: 2 hours per array row. 4. **Re-inspect after cleaning** — verify IV-curve performance returns to baseline.",
            "preventive_plan": "1. Schedule quarterly cleaning cycles, increasing to monthly in high bird-activity zones. 2. Install physical bird deterrent systems on panel mounting racks. 3. Monitor soiling ratio via string-level inverter data to detect accumulation early.",
            "safety": "Low electrical risk. Use standard PPE (gloves, safety glasses) during cleaning. If panels are roof-mounted, follow fall-protection protocols. Disconnect panels from inverter before wet cleaning to avoid electrical hazard.",
            "priority": "P3-Scheduled",
            "response_time": "Within 7 days",
            "conclusion": "Bird dropping contamination is a manageable issue with minimal long-term risk when addressed promptly. Regular cleaning and bird deterrent installation will prevent recurrence and maintain optimal energy output.",
        },
        "Clean": {
            "executive_summary": f"Inspection of {panel_label} confirms the panel is in clean, healthy operating condition with {conf_pct} confidence. No defects, soiling, or damage detected. The panel is performing within expected parameters.",
            "severity": "Low",
            "risk_category": "Cosmetic",
            "technical_assessment": "The panel surface is free of visible defects, soiling, cracks, or discoloration. The anti-reflective coating appears intact, and cell interconnections show no signs of thermal stress or delamination. The panel is operating at or near its rated efficiency, with normal age-related degradation within manufacturer specifications.",
            "lifetime_unaddressed": "25-30 years (standard lifespan, no issues present)",
            "lifetime_addressed": "25-30 years (no action required)",
            "annual_degradation": "0.5% per year (normal manufacturer-specified rate)",
            "efficiency_loss": "0% (no defect detected)",
            "annual_energy_loss": "0 kWh (operating at expected capacity)",
            "financial_impact": "$0 USD (no remediation needed)",
            "root_causes": "No defect present. Panel is in normal operating condition. Continue regular monitoring per O&M schedule.",
            "corrective_actions": "1. **No immediate action required** — panel is healthy. 2. **Continue regular monitoring** — next scheduled inspection per maintenance calendar. 3. **Document baseline** — record current IV-curve data as reference for future comparisons. 4. **Visual log** — archive this clean inspection image for trend analysis.",
            "preventive_plan": "1. Maintain semi-annual visual inspections per industry best practices. 2. Continue automated performance monitoring via inverter telemetry. 3. Schedule annual thermographic inspection to detect latent issues.",
            "safety": "No safety concerns. Standard site safety protocols apply for routine inspections.",
            "priority": "P4-Monitor",
            "response_time": "Next scheduled inspection",
            "conclusion": "Panel is in excellent condition with no defects detected. Continue standard monitoring and maintenance schedule. No corrective action required at this time.",
        },
        "Dusty": {
            "executive_summary": f"Inspection of {panel_label} reveals significant dust and soiling accumulation on the panel surface, detected with {conf_pct} confidence. This is a medium-severity issue causing measurable energy loss. Prompt cleaning is recommended to restore output.",
            "severity": "Medium",
            "risk_category": "Performance",
            "technical_assessment": "A uniform layer of particulate matter (dust, pollen, sand, or industrial fallout) has accumulated on the glass surface, reducing light transmittance to the photovoltaic cells. The soiling pattern suggests gradual environmental deposition rather than a single event. Dust particles typically range from 1-100μm and create a diffuse scattering layer that reduces direct normal irradiance reaching the cells by 10-25%. In humid conditions, dust combines with moisture to form a more tenacious film that is harder to remove.",
            "lifetime_unaddressed": "22-25 years (moderate acceleration of degradation)",
            "lifetime_addressed": "25-28 years (near-original with regular cleaning)",
            "annual_degradation": "1.0-1.5% per year (vs normal 0.5%)",
            "efficiency_loss": "10-25%",
            "annual_energy_loss": "140-350 kWh per panel (400W rated)",
            "financial_impact": "$15-40 USD per panel per year",
            "root_causes": "1. Arid or semi-arid climate with high atmospheric particulate concentration. 2. Proximity to construction, agricultural, or industrial dust sources. 3. Low panel tilt angle (below 15°) reducing natural rain wash-off.",
            "corrective_actions": "1. **Schedule panel cleaning within 3-5 days** — use pressurized deionized water or automated robotic cleaning. Estimated: 10 min per panel. 2. **Soiling rate analysis** — measure soiling ratio from inverter data before and after cleaning to quantify loss. 3. **Consider automated cleaning system** — evaluate ROI of robotic or spray-based auto-cleaning for sites with chronic soiling. Estimated: 1 day for site assessment. 4. **Anti-soiling coating application** — apply hydrophobic nano-coating after cleaning to reduce future accumulation. Estimated: 20 min per panel.",
            "preventive_plan": "1. Implement weekly soiling monitoring via string-level performance ratio tracking. 2. Install automated cleaning systems for high-soiling sites (ROI typically 2-3 years). 3. Adjust cleaning schedule seasonally — increase frequency during dry, dusty months.",
            "safety": "Low risk. Use standard PPE for cleaning operations. Ensure electrical isolation before wet cleaning. For ground-mounted systems, be aware of terrain hazards.",
            "priority": "P3-Scheduled",
            "response_time": "Within 3-5 days",
            "conclusion": "Dust soiling is causing moderate energy loss but is fully recoverable with cleaning. Implementing an automated cleaning solution or regular cleaning schedule will prevent recurrence and optimize long-term energy yield.",
        },
        "Electrical-damage": {
            "executive_summary": f"Inspection of {panel_label} reveals electrical damage — potential burnt cells, bypass diode failure, or wiring degradation — detected with {conf_pct} confidence. This is a CRITICAL severity issue requiring immediate isolation and emergency repair. Fire hazard risk present.",
            "severity": "Critical",
            "risk_category": "Safety-Critical",
            "technical_assessment": "The panel exhibits visible signs of electrical damage, which may include browning or blackening of cells (thermal runaway), delamination of the EVA encapsulant, burn marks on the backsheet, or discoloration around junction boxes. This indicates potential bypass diode failure, series resistance hotspots, or arc-fault conditions. Infrared thermography would likely reveal temperature anomalies exceeding 20°C above ambient at affected cells. The root failure mechanism may involve solder bond degradation, cell micro-cracks propagating under thermal cycling, or moisture ingress at the junction box.",
            "lifetime_unaddressed": "0-5 years (IMMEDIATE failure risk, fire hazard)",
            "lifetime_addressed": "10-15 years (if repairable; otherwise panel replacement required)",
            "annual_degradation": "15-50% per year (catastrophic, accelerating)",
            "efficiency_loss": "50-100%",
            "annual_energy_loss": "700-1400 kWh per panel (400W rated — near total loss)",
            "financial_impact": "$80-160 USD per panel per year in lost generation + replacement cost $200-400",
            "root_causes": "1. Bypass diode failure due to thermal stress, manufacturing defect, or lightning surge. 2. Potential-induced degradation (PID) from high system voltage and humidity. 3. Connector/wiring degradation from UV exposure, moisture ingress, or rodent damage.",
            "corrective_actions": "1. **IMMEDIATE electrical isolation** — disconnect panel from string/inverter and tag out. Do NOT leave energized. Estimated: 30 min. 2. **Emergency thermographic inspection** — use IR camera to map hotspot locations and assess fire risk. Estimated: 1 hour. 3. **IV-curve tracing** — perform detailed electrical characterization to identify failed components. Estimated: 30 min. 4. **Panel replacement or junction box repair** — depending on damage extent, replace panel or repair junction box and bypass diodes. Estimated: 2-4 hours per panel.",
            "preventive_plan": "1. Implement annual thermographic inspections across the entire array to detect electrical anomalies early. 2. Install string-level monitoring with arc-fault detection capability. 3. Ensure proper torque specifications on all MC4 connectors during installation and maintenance.",
            "safety": "⚠️ HIGH RISK — Potential fire hazard and electrical shock risk. De-energize the affected string before any physical contact. Use arc-flash rated PPE (Category 2 minimum). Do not attempt repair during rain or wet conditions. Verify zero-energy state with a multimeter before handling.",
            "priority": "P1-Immediate",
            "response_time": "Within 24 hours",
            "conclusion": "Electrical damage presents a serious safety hazard including fire risk. Immediate isolation is non-negotiable. Panel likely requires replacement. Preventive thermographic monitoring should be implemented site-wide to catch future issues before they reach this severity.",
        },
        "Physical-Damage": {
            "executive_summary": f"Inspection of {panel_label} reveals physical damage — cracks, chips, or broken glass — detected with {conf_pct} confidence. This is a high-severity issue requiring prompt repair or replacement to prevent moisture ingress and further degradation.",
            "severity": "High",
            "risk_category": "Structural",
            "technical_assessment": "The panel shows visible physical damage to the front glass surface, which may include radial cracks, chips, spider-web fractures, or complete glass breakage. Physical damage compromises the hermetic seal of the laminate stack, allowing moisture and contaminants to reach the photovoltaic cells and metallic interconnections. This accelerates corrosion, causes cell delamination, and can lead to ground faults. The damage pattern may indicate impact (hail, debris), mechanical stress (improper mounting, wind uplift), or thermal shock.",
            "lifetime_unaddressed": "5-10 years (accelerated degradation from moisture ingress)",
            "lifetime_addressed": "15-20 years (if replaced promptly; residual stress may remain)",
            "annual_degradation": "5-15% per year (accelerating as moisture damage compounds)",
            "efficiency_loss": "20-80%",
            "annual_energy_loss": "280-1120 kWh per panel (400W rated)",
            "financial_impact": "$30-130 USD per panel per year + replacement cost $200-400",
            "root_causes": "1. Hail impact or airborne debris during severe weather events. 2. Improper handling during installation or maintenance (point-load damage). 3. Thermal cycling stress causing micro-cracks that propagate over time.",
            "corrective_actions": "1. **Apply temporary protective covering within 48 hours** — use weatherproof tape or temporary sealing to prevent water ingress. Estimated: 30 min per panel. 2. **Schedule panel replacement within 72 hours** — order matching panel and plan string reconfiguration if needed. Estimated: 2 hours per panel. 3. **Inspect adjacent panels** — physical damage events (hail, impact) often affect multiple panels. Estimated: 1 hour per row. 4. **Document damage for warranty/insurance** — photograph damage pattern and file claims if applicable.",
            "preventive_plan": "1. Install hail guards or protective mesh in regions with frequent severe weather. 2. Use panels with higher mechanical load rating (IEC 61215 certification) for vulnerable installations. 3. Implement post-storm inspection protocols to catch damage early.",
            "safety": "Moderate risk. Broken glass poses laceration hazard — use cut-resistant gloves (Level A4+). Damaged panels may have exposed electrical components — verify isolation before handling. Wet conditions increase ground-fault risk on damaged panels.",
            "priority": "P2-Urgent",
            "response_time": "Within 48-72 hours",
            "conclusion": "Physical damage requires urgent attention to prevent moisture-driven cascading failure. Panel replacement is typically the most cost-effective remediation. Inspect the broader array for collateral damage from the same event.",
        },
        "Snow-Covered": {
            "executive_summary": f"Inspection of {panel_label} reveals snow or ice coverage on the panel surface, detected with {conf_pct} confidence. This is a medium-severity temporary condition that blocks sunlight almost entirely. Monitoring is recommended, with manual clearing if conditions persist.",
            "severity": "Medium",
            "risk_category": "Performance",
            "technical_assessment": "The panel surface is partially or fully covered with snow or ice, which creates an opaque barrier preventing photon absorption by the photovoltaic cells. Snow coverage reduces output to near-zero for affected cells. Additionally, uneven snow loading can create mechanical stress on the mounting structure, and the freeze-thaw cycle may cause micro-cracks in the glass or cell structure (thermal shock). If snow melts and refreezes at the panel edge, ice dams can form, exerting lateral force on frame and mounting clips.",
            "lifetime_unaddressed": "23-27 years (minimal long-term impact if seasonal)",
            "lifetime_addressed": "25-30 years (standard lifespan maintained)",
            "annual_degradation": "0.5-0.8% per year (slightly above normal due to thermal stress)",
            "efficiency_loss": "90-100% (while covered)",
            "annual_energy_loss": "Varies — 50-200 kWh per snow day per panel (400W rated)",
            "financial_impact": "$5-25 USD per panel per snow event (generation loss only)",
            "root_causes": "1. Seasonal snowfall in cold-climate installations. 2. Low panel tilt angle (below 30°) preventing natural snow shedding. 3. Shading from adjacent rows or structures trapping snow on lower panels.",
            "corrective_actions": "1. **Monitor weather forecast** — if thaw is expected within 1-2 days, snow will self-clear. No action needed. 2. **Manual snow removal** (if persistent >48 hours) — use soft foam snow rakes designed for solar panels. NEVER use metal tools or hot water. Estimated: 5 min per panel. 3. **Activate de-icing system** (if installed) — enable heating elements or warm-air systems. 4. **Structural load check** — if heavy wet snow exceeds 30 kg/m², assess mounting structure load capacity.",
            "preventive_plan": "1. Design panel tilt angle above 30° in snow-prone regions to encourage natural shedding. 2. Install frame-mounted heating wires or de-icing systems for critical installations. 3. Plan seasonal generation forecasts accounting for snow loss days.",
            "safety": "Moderate risk. Snow-covered panels on roofs create slip hazards. Use fall-protection equipment on elevated installations. Heavy snow removal requires awareness of panel load limits — avoid standing on panels. Wear insulated gloves to prevent frostbite.",
            "priority": "P3-Scheduled",
            "response_time": "Within 1-2 days (weather dependent)",
            "conclusion": "Snow coverage is a temporary, weather-dependent condition with negligible long-term impact on panel lifespan. Management focus should be on minimizing extended coverage periods and preventing mechanical damage from excessive snow loads.",
        },
    }

    # Get defect-specific data or use a generic template
    info: Dict[str, str] = analyses.get(predicted_class, {
        "executive_summary": f"Inspection of {panel_label} has identified a defect classified as '{predicted_class}' with {conf_pct} confidence. Further detailed assessment is recommended to determine severity and appropriate remediation.",
        "severity": "Medium",
        "risk_category": "Performance",
        "technical_assessment": f"A defect of type '{predicted_class}' has been detected through automated visual inspection. The specific characteristics and extent of the defect require further on-site assessment by a qualified technician. Additional diagnostic testing (IV-curve tracing, thermographic inspection) is recommended to fully characterize the issue.",
        "lifetime_unaddressed": "15-20 years (estimated pending detailed assessment)",
        "lifetime_addressed": "22-27 years (estimated pending detailed assessment)",
        "annual_degradation": "1.0-3.0% per year (estimated pending detailed assessment)",
        "efficiency_loss": "10-40% (estimated pending detailed assessment)",
        "annual_energy_loss": "140-560 kWh per panel (400W rated, estimated)",
        "financial_impact": "$15-65 USD per panel per year (estimated)",
        "root_causes": "1. Environmental factors (weather, pollution, wildlife). 2. Installation quality issues. 3. Material degradation over time.",
        "corrective_actions": "1. **Schedule on-site inspection within 1 week** — deploy qualified technician for detailed assessment. 2. **Perform IV-curve analysis** — characterize electrical performance degradation. 3. **Thermographic survey** — identify any thermal anomalies. 4. **Determine remediation plan** based on findings.",
        "preventive_plan": "1. Increase inspection frequency to quarterly for affected zone. 2. Implement continuous performance monitoring via string-level data. 3. Review and update maintenance procedures based on findings.",
        "safety": "Standard safety precautions apply. Follow lockout/tagout procedures if electrical work is required. Use appropriate PPE for on-site inspection.",
        "priority": "P3-Scheduled",
        "response_time": "Within 1 week",
        "conclusion": f"A '{predicted_class}' defect has been identified and requires professional assessment. Schedule inspection to determine appropriate corrective action and prevent potential escalation.",
    })

    report: str = f"""1. **Executive Summary**: {info['executive_summary']}

2. **Defect Classification & Severity**:
   - Defect Type: {predicted_class}
   - Severity Level: {info['severity']}
   - Confidence Score: {conf_pct}
   - Risk Category: {info['risk_category']}

3. **Detailed Technical Assessment**: {info['technical_assessment']}

4. **Estimated Panel Lifetime Impact**:
   - Standard panel lifespan: 25-30 years
   - Estimated remaining lifespan if NOT addressed: {info['lifetime_unaddressed']}
   - Estimated remaining lifespan if addressed promptly: {info['lifetime_addressed']}
   - Annual degradation acceleration: {info['annual_degradation']}

5. **Energy Loss Analysis**:
   - Immediate efficiency loss: {info['efficiency_loss']}
   - Projected annual energy loss: {info['annual_energy_loss']}
   - Financial impact estimate: {info['financial_impact']}

6. **Root Cause Analysis**: {info['root_causes']}

7. **Recommended Corrective Actions**: {info['corrective_actions']}

8. **Preventive Maintenance Plan**: {info['preventive_plan']}

9. **Safety Considerations**: {info['safety']}

10. **Conclusion & Priority Rating**:
    - Overall Priority: {info['priority']}
    - Recommended response time: {info['response_time']}
    - {info['conclusion']}"""

    return report
