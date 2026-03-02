"""
SolarMind AI — ViT Defect Classifier Model Definition
Vision Transformer for solar panel defect classification.
"""
import random
import math
from typing import Any, Dict, List

# Model configuration
VIT_CONFIG: Dict[str, Any] = {
    "model_name": "vit_base_patch16_224",
    "pretrained": True,
    "num_classes": 4,
    "class_names": ["normal", "micro_crack", "hotspot", "dust_soiling"],
    "input_size": 224,
    "patch_size": 16,
    "num_patches": 196,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "dropout": 0.1,
}

TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingWarmRestarts",
    "loss_function": "FocalLoss",
    "focal_gamma": 2.0,
    "focal_alpha": [0.15, 0.35, 0.25, 0.25],
    "label_smoothing": 0.1,
    "mixed_precision": True,
    "early_stopping_patience": 10,
}

CLASS_NAMES: List[str] = ["normal", "micro_crack", "hotspot", "dust_soiling"]


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def simulate_inference(panel_id: str = "") -> Dict[str, Any]:
    """
    Simulate ViT inference result for a panel.
    In production, this would load a trained model and run actual inference.
    """
    # Simulate class probabilities
    defect_type: str = random.choices(
        CLASS_NAMES,
        weights=[0.82, 0.07, 0.05, 0.06],
        k=1,
    )[0]

    probs: Dict[str, float] = {}

    if defect_type == "normal":
        probs = {
            "normal": _r(random.uniform(0.90, 0.99), 3),
            "micro_crack": _r(random.uniform(0.001, 0.04), 3),
            "hotspot": _r(random.uniform(0.001, 0.03), 3),
            "dust_soiling": _r(random.uniform(0.001, 0.03), 3),
        }
    else:
        # Dominant class gets high probability
        dominant_raw: float = random.uniform(0.75, 0.97)
        probs[defect_type] = _r(dominant_raw, 3)
        remaining: float = 1.0 - dominant_raw
        other_classes: List[str] = [c for c in CLASS_NAMES if c != defect_type]
        for i in range(len(other_classes)):
            cls: str = other_classes[i]
            if i == len(other_classes) - 1:
                probs[cls] = _r(remaining, 3)
            else:
                p_raw: float = random.uniform(0.001, remaining * 0.5)
                probs[cls] = _r(p_raw, 3)
                remaining = remaining - p_raw

    # Find class with highest probability
    predicted_class: str = CLASS_NAMES[0]
    best_prob: float = -1.0
    for k, v in probs.items():
        if v > best_prob:
            best_prob = v
            predicted_class = k

    result_panel_id: str = panel_id if panel_id else f"P-{random.randint(1000, 1199):04d}"

    return {
        "panel_id": result_panel_id,
        "predicted_class": predicted_class,
        "confidence": probs[predicted_class],
        "probabilities": probs,
        "model": str(VIT_CONFIG["model_name"]),
        "inference_time_ms": _r(random.uniform(8, 15), 1),
        "attention_highlights": _simulate_attention_regions(),
    }


def _simulate_attention_regions() -> List[Dict[str, Any]]:
    """Simulate attention map highlights (regions the model focused on)."""
    num_regions: int = random.randint(1, 4)
    regions: List[Dict[str, Any]] = []
    for _ in range(num_regions):
        regions.append({
            "x": random.randint(20, 180),
            "y": random.randint(20, 180),
            "radius": random.randint(15, 45),
            "intensity": _r(random.uniform(0.5, 1.0), 2),
        })
    return regions


def simulate_batch_inference(num_panels: int = 10) -> List[Dict[str, Any]]:
    """Simulate batch inference on multiple panels."""
    results: List[Dict[str, Any]] = []
    for i in range(num_panels):
        result: Dict[str, Any] = simulate_inference(f"P-{1000 + i:04d}")
        results.append(result)
    return results


def get_model_info() -> Dict[str, Any]:
    """Return model architecture information."""
    return {
        "architecture": "Vision Transformer (ViT-Base/16)",
        "config": VIT_CONFIG,
        "training": TRAINING_CONFIG,
        "total_parameters": "86M",
        "trainable_parameters": "86M",
        "input_format": "224x224 RGB/Thermal image",
        "output": "4-class probability distribution",
        "classes": CLASS_NAMES,
        "pretrained_on": "ImageNet-21k (14M images)",
        "fine_tuned_on": "PV Defect Dataset + Drone Thermal Images",
    }
