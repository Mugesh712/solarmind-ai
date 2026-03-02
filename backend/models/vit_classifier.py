"""
SolarMind AI — ViT Defect Classifier Model Definition
Vision Transformer for solar panel defect classification.

Uses real model inference when PyTorch + trained checkpoint are available.
Falls back to deterministic simulation otherwise.
"""
import os
import math
import hashlib
import random
from typing import Any, Dict, List, Optional

# Model configuration
VIT_CONFIG: Dict[str, Any] = {
    "model_name": "vit_small_patch16_224",
    "pretrained": True,
    "num_classes": 6,
    "class_names": ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"],
    "input_size": 224,
    "patch_size": 16,
    "num_patches": 196,
    "embed_dim": 384,
    "num_heads": 6,
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
    "label_smoothing": 0.1,
    "mixed_precision": True,
    "early_stopping_patience": 10,
}

CLASS_NAMES: List[str] = sorted(["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"])

MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "classifier_model.pth")

# ──────────────────────────────────────────────
# Dependency checks
# ──────────────────────────────────────────────

def _has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # type: ignore
        return True
    except ImportError:
        return False


def _has_timm() -> bool:
    """Check if timm library is available."""
    try:
        import timm  # type: ignore
        return True
    except ImportError:
        return False


def _has_pil() -> bool:
    """Check if PIL/Pillow is available."""
    try:
        from PIL import Image  # type: ignore
        return True
    except ImportError:
        return False


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _seed_for(key: str) -> int:
    """Generate a deterministic seed from a string key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ──────────────────────────────────────────────
# Cached model loading
# ──────────────────────────────────────────────

_cached_model: Any = None
_cached_device: Any = None


def _load_model() -> Any:
    """Load and cache the trained ViT model. Returns (model, device) or (None, None)."""
    global _cached_model, _cached_device

    if _cached_model is not None:
        return _cached_model, _cached_device

    if not _has_torch() or not _has_timm() or not os.path.isfile(MODEL_PATH):
        return None, None

    import torch  # type: ignore
    import timm  # type: ignore

    device = torch.device("cpu")
    try:
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        _cached_model = model
        _cached_device = device
        print(f"[SolarMind] ViT model loaded from {MODEL_PATH}")
        return model, device
    except Exception as e:
        print(f"[SolarMind] Failed to load ViT model: {e}")
        return None, None


# ──────────────────────────────────────────────
# Real inference functions
# ──────────────────────────────────────────────

def run_inference(image_path: str = "", panel_id: str = "") -> Dict[str, Any]:
    """
    Run ViT inference on an image.

    If PyTorch + trained model are available and image_path is provided,
    runs real model inference. Otherwise falls back to deterministic simulation.
    """
    if image_path and os.path.isfile(image_path):
        model, device = _load_model()
        if model is not None and _has_pil():
            return _real_inference(image_path, model, device, panel_id)

    # Deterministic fallback simulation
    return _simulated_inference(panel_id)


def _real_inference(image_path: str, model: Any, device: Any, panel_id: str = "") -> Dict[str, Any]:
    """Classify using the real trained ViT model."""
    import torch  # type: ignore
    from torchvision import transforms  # type: ignore
    from PIL import Image  # type: ignore
    import time

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs_list: List[float] = [float(x) for x in probabilities[0].tolist()]
    inference_ms = (time.time() - start_time) * 1000

    probs_dict: Dict[str, float] = {}
    for i in range(len(CLASS_NAMES)):
        probs_dict[CLASS_NAMES[i]] = _r(probs_list[i], 4)

    predicted_idx: int = probs_list.index(max(probs_list))
    predicted_class: str = CLASS_NAMES[predicted_idx]
    confidence: float = probs_list[predicted_idx]

    result_panel_id: str = panel_id if panel_id else f"P-{hash(image_path) % 200 + 1000:04d}"

    return {
        "panel_id": result_panel_id,
        "predicted_class": predicted_class,
        "confidence": _r(confidence, 4),
        "probabilities": probs_dict,
        "model": str(VIT_CONFIG["model_name"]),
        "inference_time_ms": _r(inference_ms, 1),
        "attention_highlights": extract_attention_regions(image_path),
        "mode": "real",
    }


def _simulated_inference(panel_id: str = "") -> Dict[str, Any]:
    """
    Deterministic simulated inference for when the model is unavailable.
    Uses panel_id as seed for consistent results.
    """
    rng = random.Random(_seed_for(panel_id) if panel_id else 42)

    defect_type: str = rng.choices(
        CLASS_NAMES,
        weights=[0.05, 0.55, 0.12, 0.08, 0.10, 0.10],
        k=1,
    )[0]

    probs: Dict[str, float] = {}

    if defect_type == "Clean":
        dominant_raw: float = rng.uniform(0.90, 0.99)
        probs["Clean"] = _r(dominant_raw, 3)
        remaining: float = 1.0 - dominant_raw
        other_classes: List[str] = [c for c in CLASS_NAMES if c != "Clean"]
        for i in range(len(other_classes)):
            cls: str = other_classes[i]
            if i == len(other_classes) - 1:
                probs[cls] = _r(max(0.001, remaining), 3)
            else:
                p_raw: float = rng.uniform(0.001, remaining * 0.4)
                probs[cls] = _r(p_raw, 3)
                remaining = remaining - p_raw
    else:
        dominant_raw = rng.uniform(0.75, 0.97)
        probs[defect_type] = _r(dominant_raw, 3)
        remaining = 1.0 - dominant_raw
        other_classes = [c for c in CLASS_NAMES if c != defect_type]
        for i in range(len(other_classes)):
            cls = other_classes[i]
            if i == len(other_classes) - 1:
                probs[cls] = _r(max(0.001, remaining), 3)
            else:
                p_raw = rng.uniform(0.001, remaining * 0.4)
                probs[cls] = _r(p_raw, 3)
                remaining = remaining - p_raw

    predicted_class: str = max(probs, key=lambda k: probs[k])
    result_panel_id: str = panel_id if panel_id else f"P-{rng.randint(1000, 1199):04d}"

    return {
        "panel_id": result_panel_id,
        "predicted_class": predicted_class,
        "confidence": probs[predicted_class],
        "probabilities": probs,
        "model": str(VIT_CONFIG["model_name"]),
        "inference_time_ms": _r(rng.uniform(8, 15), 1),
        "attention_highlights": _simulated_attention_regions(panel_id),
        "mode": "simulated",
        "note": "Install PyTorch + timm and place trained model at backend/checkpoints/classifier_model.pth for real inference",
    }


# ──────────────────────────────────────────────
# Attention map extraction
# ──────────────────────────────────────────────

def extract_attention_regions(image_path: str = "") -> List[Dict[str, Any]]:
    """
    Extract attention regions from the ViT model.

    When the model is available, uses forward hooks on attention layers
    to get real attention weights. Otherwise returns deterministic simulated regions.
    """
    if image_path and os.path.isfile(image_path):
        model, device = _load_model()
        if model is not None and _has_pil():
            return _real_attention_regions(image_path, model, device)

    return _simulated_attention_regions(image_path)


def _real_attention_regions(image_path: str, model: Any, device: Any) -> List[Dict[str, Any]]:
    """Extract real attention map highlights from the ViT model's attention layers."""
    import torch  # type: ignore
    from torchvision import transforms  # type: ignore
    from PIL import Image  # type: ignore

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Register hook on the last attention block
    attention_weights: List[Any] = []

    def hook_fn(module: Any, input: Any, output: Any) -> None:
        if hasattr(module, 'attn_drop'):
            # timm ViT stores attention in the Attention module
            pass
        attention_weights.append(output)

    # Try to get attention from the last transformer block
    hooks = []
    try:
        # timm ViT structure: model.blocks[-1].attn
        last_block = model.blocks[-1].attn
        hook = last_block.register_forward_hook(hook_fn)
        hooks.append(hook)
    except (AttributeError, IndexError):
        pass

    # Forward pass to capture attention
    with torch.no_grad():
        _ = model(img_tensor)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Convert attention to regions
    regions: List[Dict[str, Any]] = []
    try:
        if attention_weights:
            # Get attention output and compute spatial attention
            attn_output = attention_weights[-1]
            if isinstance(attn_output, torch.Tensor):
                # Average across heads if multi-head
                if attn_output.dim() >= 3:
                    spatial = attn_output.mean(dim=1) if attn_output.dim() == 4 else attn_output
                    # Get spatial attention values
                    if spatial.dim() >= 2:
                        attn_map = spatial[0].cpu().numpy()
                        # Find top attention patches
                        num_patches_side = 14  # 224/16
                        if len(attn_map) >= num_patches_side * num_patches_side:
                            patch_attn = attn_map[:num_patches_side * num_patches_side]
                            # Find top 3 attention regions
                            import numpy as np  # type: ignore
                            top_indices = np.argsort(patch_attn.flatten())[-3:]
                            for idx in top_indices:
                                row = int(idx) // num_patches_side
                                col = int(idx) % num_patches_side
                                intensity = float(patch_attn.flatten()[idx])
                                max_val = float(patch_attn.max())
                                if max_val > 0:
                                    intensity = intensity / max_val
                                regions.append({
                                    "x": int(col * 16 + 8),
                                    "y": int(row * 16 + 8),
                                    "radius": 25,
                                    "intensity": _r(intensity, 2),
                                    "source": "real",
                                })
    except Exception as e:
        print(f"[SolarMind] Attention extraction error: {e}")

    if not regions:
        # Fallback to simulated if extraction failed
        return _simulated_attention_regions(image_path)

    return regions


def _simulated_attention_regions(panel_id: str = "") -> List[Dict[str, Any]]:
    """Deterministic simulated attention map highlights."""
    rng = random.Random(_seed_for(panel_id + "_attn") if panel_id else 7)
    num_regions: int = rng.randint(1, 4)
    regions: List[Dict[str, Any]] = []
    for _ in range(num_regions):
        regions.append({
            "x": rng.randint(20, 180),
            "y": rng.randint(20, 180),
            "radius": rng.randint(15, 45),
            "intensity": _r(rng.uniform(0.5, 1.0), 2),
            "source": "simulated",
        })
    return regions


# ──────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────

def simulate_batch_inference(num_panels: int = 10) -> List[Dict[str, Any]]:
    """Run inference on multiple panels."""
    results: List[Dict[str, Any]] = []
    for i in range(num_panels):
        result: Dict[str, Any] = run_inference(panel_id=f"P-{1000 + i:04d}")
        results.append(result)
    return results


# Keep backward compatibility
def simulate_inference(panel_id: str = "") -> Dict[str, Any]:
    """Backward-compatible wrapper for run_inference."""
    return run_inference(panel_id=panel_id)


# ──────────────────────────────────────────────
# Model information
# ──────────────────────────────────────────────

def get_model_info() -> Dict[str, Any]:
    """Return model architecture information."""
    model_available = os.path.isfile(MODEL_PATH)
    pytorch_available = _has_torch()
    timm_available = _has_timm()

    return {
        "architecture": "Vision Transformer (ViT-Small/16)",
        "config": VIT_CONFIG,
        "training": TRAINING_CONFIG,
        "total_parameters": "22M",
        "trainable_parameters": "22M",
        "input_format": "224x224 RGB image",
        "output": f"{len(CLASS_NAMES)}-class probability distribution",
        "classes": CLASS_NAMES,
        "pretrained_on": "ImageNet-21k (14M images)",
        "fine_tuned_on": "PV Defect Dataset (Kaggle)",
        "model_checkpoint": MODEL_PATH,
        "model_available": model_available,
        "pytorch_available": pytorch_available,
        "timm_available": timm_available,
        "inference_mode": "real" if (model_available and pytorch_available and timm_available) else "simulated",
    }
