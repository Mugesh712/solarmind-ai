"""
SolarMind AI — Multimodal Fusion Model
Combines ViT (visual) + Telemetry (time-series) features for enhanced defect classification.
"""
import random
import math
from typing import Any, Dict, List, Optional, Tuple

# Fusion model configuration
FUSION_CONFIG: Dict[str, Any] = {
    "visual_backbone": "vit_base_patch16_224",
    "visual_embed_dim": 768,
    "telemetry_features": [
        "irradiance", "temperature", "power_output",
        "voltage", "current", "humidity"
    ],
    "telemetry_embed_dim": 128,
    "fusion_dim": 512,
    "fusion_method": "cross_attention",
    "num_classes": 4,
    "class_names": ["normal", "micro_crack", "hotspot", "dust_soiling"],
    "dropout": 0.2,
    "num_attention_heads": 8,
}

# Telemetry stats stored as separate typed dicts
TELEMETRY_UNITS: Dict[str, str] = {
    "irradiance": "W/m2",
    "temperature": "C",
    "power_output": "kW",
    "voltage": "V",
    "current": "A",
    "humidity": "%",
}

TELEMETRY_RANGES: Dict[str, Tuple[float, float]] = {
    "irradiance": (200.0, 1100.0),
    "temperature": (25.0, 85.0),
    "power_output": (0.5, 5.5),
    "voltage": (28.0, 42.0),
    "current": (2.0, 14.0),
    "humidity": (15.0, 90.0),
}

TELEMETRY_MEANS: Dict[str, float] = {
    "irradiance": 750.0,
    "temperature": 45.0,
    "power_output": 4.8,
    "voltage": 38.0,
    "current": 9.5,
    "humidity": 45.0,
}

CLASS_NAMES: List[str] = ["normal", "micro_crack", "hotspot", "dust_soiling"]
FEATURE_NAMES: List[str] = [
    "irradiance", "temperature", "power_output",
    "voltage", "current", "humidity"
]


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def simulate_telemetry_features(
    panel_id: str = "", defect_type: str = "normal"
) -> Dict[str, Dict[str, Any]]:
    """
    Simulate telemetry feature extraction for a panel.
    In production, this reads real sensor data and encodes it via a 1D-CNN or LSTM.
    """
    features: Dict[str, Dict[str, Any]] = {}
    anomaly_factor: float = 1.0

    if defect_type == "hotspot":
        anomaly_factor = 1.4
    elif defect_type == "micro_crack":
        anomaly_factor = 1.15
    elif defect_type == "dust_soiling":
        anomaly_factor = 1.25

    for feat_name in FEATURE_NAMES:
        base: float = TELEMETRY_MEANS[feat_name]
        low: float = TELEMETRY_RANGES[feat_name][0]
        high: float = TELEMETRY_RANGES[feat_name][1]

        value: float
        if feat_name == "temperature" and defect_type == "hotspot":
            value = base * anomaly_factor + random.uniform(-3, 8)
        elif feat_name == "power_output" and defect_type != "normal":
            value = base / anomaly_factor + random.uniform(-0.3, 0.1)
        elif feat_name == "current" and defect_type == "micro_crack":
            value = base * 0.85 + random.uniform(-0.5, 0.3)
        else:
            value = base + random.uniform(-base * 0.08, base * 0.08)

        value = max(low, min(high, _r(value, 2)))
        range_span: float = high - low
        anomaly_score: float = abs(value - base) / range_span if range_span > 0 else 0.0
        features[feat_name] = {
            "value": value,
            "unit": TELEMETRY_UNITS[feat_name],
            "anomaly_score": _r(anomaly_score, 3),
        }

    return features


def simulate_fusion_inference(
    panel_id: str = "", telemetry: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Simulate multimodal fusion inference.
    Combines visual (ViT) and telemetry features via cross-attention.
    """
    defect_type: str = random.choices(
        CLASS_NAMES,
        weights=[0.80, 0.08, 0.06, 0.06],
        k=1,
    )[0]

    tele_features: Dict[str, Dict[str, Any]]
    if telemetry is not None:
        tele_features = telemetry
    else:
        tele_features = simulate_telemetry_features(panel_id, defect_type)

    visual_weight: float = _r(random.uniform(0.55, 0.75), 3)
    telemetry_weight: float = _r(1.0 - random.uniform(0.55, 0.75), 3)

    probs: Dict[str, float] = {}

    if defect_type == "normal":
        probs = {
            "normal": _r(random.uniform(0.92, 0.99), 3),
            "micro_crack": _r(random.uniform(0.001, 0.03), 3),
            "hotspot": _r(random.uniform(0.001, 0.025), 3),
            "dust_soiling": _r(random.uniform(0.001, 0.025), 3),
        }
    else:
        dominant_raw: float = random.uniform(0.80, 0.98)
        probs[defect_type] = _r(dominant_raw, 3)
        remaining: float = 1.0 - dominant_raw
        other_classes: List[str] = [c for c in CLASS_NAMES if c != defect_type]
        for i in range(len(other_classes)):
            cls: str = other_classes[i]
            if i == len(other_classes) - 1:
                probs[cls] = _r(max(0.001, remaining), 3)
            else:
                p_raw: float = random.uniform(0.001, remaining * 0.45)
                probs[cls] = _r(p_raw, 3)
                remaining = remaining - p_raw

    # Find class with highest probability
    predicted_class: str = CLASS_NAMES[0]
    best_prob: float = -1.0
    for k, v in probs.items():
        if v > best_prob:
            best_prob = v
            predicted_class = k

    cross_attention_scores: List[Dict[str, Any]] = []
    for feat_name in FEATURE_NAMES:
        score: float = _r(random.uniform(0.1, 0.95), 3)
        if defect_type == "hotspot" and feat_name == "temperature":
            score = _r(random.uniform(0.75, 0.98), 3)
        elif defect_type == "micro_crack" and feat_name == "current":
            score = _r(random.uniform(0.70, 0.95), 3)
        elif defect_type == "dust_soiling" and feat_name == "irradiance":
            score = _r(random.uniform(0.65, 0.92), 3)
        cross_attention_scores.append({
            "feature": feat_name,
            "attention_weight": score,
        })

    result_panel_id: str = panel_id if panel_id else f"P-{random.randint(1000, 1199):04d}"

    return {
        "panel_id": result_panel_id,
        "predicted_class": predicted_class,
        "confidence": probs[predicted_class],
        "probabilities": probs,
        "model": "SolarMind-Fusion-v1",
        "fusion_method": str(FUSION_CONFIG["fusion_method"]),
        "modality_weights": {
            "visual": visual_weight,
            "telemetry": telemetry_weight,
        },
        "telemetry_features": tele_features,
        "cross_attention": cross_attention_scores,
        "inference_time_ms": _r(random.uniform(12, 28), 1),
    }


def get_fusion_model_info() -> Dict[str, Any]:
    """Return fusion model architecture information."""
    features_list: List[str] = list(FUSION_CONFIG["telemetry_features"])
    return {
        "architecture": "SolarMind Multimodal Fusion (ViT + Telemetry CNN)",
        "config": FUSION_CONFIG,
        "visual_branch": {
            "backbone": "ViT-Base/16",
            "input": "224x224 RGB/Thermal image",
            "output_dim": int(FUSION_CONFIG["visual_embed_dim"]),
        },
        "telemetry_branch": {
            "backbone": "1D-CNN + GRU",
            "input": f"{len(features_list)} sensor channels x 24h window",
            "output_dim": int(FUSION_CONFIG["telemetry_embed_dim"]),
        },
        "fusion": {
            "method": "Cross-Attention (visual queries, telemetry keys/values)",
            "dimension": int(FUSION_CONFIG["fusion_dim"]),
            "heads": int(FUSION_CONFIG["num_attention_heads"]),
        },
        "total_parameters": "92M",
        "inference_latency": "~20ms (GPU) / ~85ms (edge TPU)",
    }
