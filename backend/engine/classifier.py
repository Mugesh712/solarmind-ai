"""
SolarMind AI — Image Classifier
Classifies solar panel images using a pre-trained CNN model.
Falls back to simulated classification when model/dependencies unavailable.

Dataset: PV Panel Defect Dataset (Kaggle)
https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset
"""
import os
import math
import random
from typing import Any, Dict, List, Optional

CLASS_NAMES: List[str] = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]

DATASET_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pv_defect_dataset")
MODEL_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "classifier_model.pth")


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _has_torch() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # type: ignore
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


def get_dataset_info() -> Dict[str, Any]:
    """Get information about the PV Panel Defect Dataset."""
    dataset_exists: bool = os.path.isdir(DATASET_DIR)
    model_exists: bool = os.path.isfile(MODEL_PATH)

    class_counts: Dict[str, int] = {}
    total_images: int = 0

    if dataset_exists:
        for cls in CLASS_NAMES:
            cls_dir: str = os.path.join(DATASET_DIR, cls)
            if os.path.isdir(cls_dir):
                count: int = len([
                    f for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ])
                class_counts[cls] = count
                total_images = total_images + count

    return {
        "dataset_name": "PV Panel Defect Dataset",
        "source": "https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset",
        "dataset_available": dataset_exists,
        "dataset_path": DATASET_DIR,
        "model_available": model_exists,
        "model_path": MODEL_PATH,
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "total_images": total_images,
        "class_counts": class_counts,
        "description": "Dataset for PV panel defect classification using hybrid CNN and ML methods",
    }


def classify_image(image_path: str) -> Dict[str, Any]:
    """
    Classify a solar panel image for defects.

    If PyTorch + trained model are available, uses the real model.
    Otherwise, uses image pixel analysis for classification.
    """
    if _has_torch() and _has_pil() and os.path.isfile(MODEL_PATH):
        result: Dict[str, Any] = _real_classify(image_path)
        return result
    # Use pixel-based analysis
    if _has_pil():
        result = _pixel_classify(image_path)
        return result
    result = _fallback_classify(image_path)
    return result


def classify_image_bytes(image_bytes: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
    """
    Classify an uploaded image from raw bytes.
    Saves temporarily, classifies, then removes temp file.
    """
    import tempfile
    suffix: str = ".jpg"
    if filename.lower().endswith(".png"):
        suffix = ".png"

    temp_fd: int
    temp_path: str
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        result: Dict[str, Any] = classify_image(temp_path)
        result["original_filename"] = filename
        return result
    finally:
        os.close(temp_fd)
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _real_classify(image_path: str) -> Dict[str, Any]:
    """Classify using the real trained ViT model."""
    import torch  # type: ignore
    from torchvision import transforms  # type: ignore
    from PIL import Image  # type: ignore
    import timm  # type: ignore

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Load model
    device = torch.device("cpu")
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        probs_list: List[float] = [float(x) for x in probabilities[0].tolist()]

    # Map indices to class names (ImageFolder sorts alphabetically)
    sorted_classes: List[str] = sorted(CLASS_NAMES)
    probs_dict: Dict[str, float] = {}
    for i in range(len(sorted_classes)):
        probs_dict[sorted_classes[i]] = _r(probs_list[i], 4)

    predicted_idx: int = probs_list.index(max(probs_list))
    predicted_class: str = sorted_classes[predicted_idx]
    confidence: float = probs_list[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": _r(confidence, 4),
        "probabilities": probs_dict,
        "model_type": "ViT-Small/16 (fine-tuned on PV Defect Dataset)",
        "image_path": image_path,
        "mode": "real",
    }


def _pixel_classify(image_path: str) -> Dict[str, Any]:
    """
    Classify using image pixel analysis (brightness, color, texture).
    Analyzes the actual image to produce meaningful per-image results.
    """
    from PIL import Image, ImageFilter, ImageStat  # type: ignore

    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((224, 224))

    stat = ImageStat.Stat(img_resized)
    r_mean: float = stat.mean[0]
    g_mean: float = stat.mean[1]
    b_mean: float = stat.mean[2]
    r_std: float = stat.stddev[0]
    g_std: float = stat.stddev[1]
    b_std: float = stat.stddev[2]

    brightness: float = (r_mean + g_mean + b_mean) / 3.0 / 255.0
    color_std: float = (r_std + g_std + b_std) / 3.0

    # Compute saturation — how colorful vs grey the image is
    max_c: float = max(r_mean, g_mean, b_mean)
    min_c: float = min(r_mean, g_mean, b_mean)
    saturation: float = 0.0
    if max_c > 0:
        saturation = (max_c - min_c) / max_c

    # Compute texture/edge intensity
    grey = img_resized.convert("L")
    edges = grey.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_mean: float = edge_stat.mean[0] / 255.0
    edge_std: float = edge_stat.stddev[0] / 255.0

    # Brown/yellow ratio (dust/bird-drop detection)
    brown_ratio: float = 0.0
    if b_mean > 0:
        brown_ratio = (r_mean * 0.6 + g_mean * 0.4) / (b_mean + 1.0)

    # Warmth: how warm (red/yellow) vs cool (blue) the image is
    warmth: float = (r_mean - b_mean) / 255.0  # positive = warm, negative = cool

    # Uniformity: low color_std means even coverage (dust, snow)
    uniformity: float = 1.0 - min(1.0, color_std / 80.0)

    # Compute scores for each class based on image features
    # CLASS_NAMES: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered
    scores: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Snow-Covered: bright images (>0.6), low saturation, uniform
    snow_score: float = 0.01
    if brightness > 0.6:
        snow_score = (brightness - 0.6) * 12.0 - saturation * 1.5 + uniformity * 1.0
    scores[5] = max(0.01, snow_score)

    # Clean: medium-low brightness, low edge, bluish/cool tones (solar cells are dark blue)
    cool_bonus: float = max(0.0, -warmth * 3.0)  # blue-ish images score higher
    clean_score: float = (1.0 - edge_mean) * 1.0 + cool_bonus + saturation * 1.5 - abs(brightness - 0.35) * 2.0
    scores[1] = max(0.01, clean_score)

    # Dusty: warm/brown tones, MID brightness (0.3-0.65), lower saturation
    dusty_mid: float = 1.0 - abs(brightness - 0.45) * 3.0  # peaks at 0.45, drops at extremes
    dusty_warm: float = max(0.0, warmth) * 2.5  # only warm images
    dusty_score: float = dusty_warm + brown_ratio * 0.8 + uniformity * 0.5 - saturation * 0.5 + max(0.0, dusty_mid)
    # Hard penalize if bright (snow territory) or dark (electrical territory)
    if brightness > 0.65:
        dusty_score = dusty_score * max(0.0, 1.0 - (brightness - 0.65) * 6.0)
    if brightness < 0.2:
        dusty_score = dusty_score * max(0.0, 1.0 - (0.2 - brightness) * 6.0)
    scores[2] = max(0.01, dusty_score)

    # Bird-drop: localized contrast spots → high edge variance but moderate edge mean
    bird_score: float = edge_std * 3.0 - edge_mean * 1.0 + color_std / 60.0 - 0.5
    scores[0] = max(0.01, bird_score)

    # Electrical-damage: very dark regions (<0.25 brightness)
    elec_score: float = 0.01
    if brightness < 0.3:
        elec_score = (0.3 - brightness) * 6.0 - saturation * 1.0
    scores[3] = max(0.01, elec_score)

    # Physical-Damage: extremely high edge intensity + high contrast (real cracks/breaks)
    phys_score: float = edge_mean * 4.0 + edge_std * 3.0 - uniformity * 2.0 - 1.5
    scores[4] = max(0.01, phys_score)

    # Normalize scores to probabilities using softmax
    temperature: float = 2.0
    total_score: float = 0.0
    for s in scores:
        total_score = total_score + math.exp(s * temperature)

    probs: Dict[str, float] = {}
    prob_list: List[float] = []
    for i in range(len(CLASS_NAMES)):
        p: float = math.exp(scores[i] * temperature) / total_score
        prob_list.append(p)
        probs[CLASS_NAMES[i]] = _r(p, 4)

    # Find predicted class
    max_prob: float = -1.0
    predicted_idx: int = 0
    for i in range(len(prob_list)):
        if prob_list[i] > max_prob:
            max_prob = prob_list[i]
            predicted_idx = i

    predicted_class: str = CLASS_NAMES[predicted_idx]

    return {
        "predicted_class": predicted_class,
        "confidence": _r(max_prob, 4),
        "probabilities": probs,
        "model_type": "Pixel Analysis (CNN-inspired feature extraction)",
        "image_path": image_path,
        "mode": "analysis",
        "features": {
            "brightness": _r(brightness, 3),
            "saturation": _r(saturation, 3),
            "edge_intensity": _r(edge_mean, 3),
            "edge_variance": _r(edge_std, 3),
            "color_variance": _r(color_std, 1),
            "brown_ratio": _r(brown_ratio, 3),
        },
    }


def _fallback_classify(image_path: str) -> Dict[str, Any]:
    """Fallback classification when PIL is not available."""
    # Use file size as a simple feature to vary results
    file_size: int = 0
    if os.path.exists(image_path):
        file_size = os.path.getsize(image_path)

    idx: int = file_size % len(CLASS_NAMES)
    defect_type: str = CLASS_NAMES[idx]

    probs: Dict[str, float] = {}
    dominant_raw: float = 0.65 + (file_size % 30) / 100.0

    probs[defect_type] = _r(dominant_raw, 4)
    remaining: float = 1.0 - dominant_raw
    other_classes: List[str] = [c for c in CLASS_NAMES if c != defect_type]

    for i in range(len(other_classes)):
        cls: str = other_classes[i]
        if i == len(other_classes) - 1:
            probs[cls] = _r(max(0.001, remaining), 4)
        else:
            share: float = remaining / (len(other_classes) - i + 1)
            probs[cls] = _r(share, 4)
            remaining = remaining - share

    return {
        "predicted_class": defect_type,
        "confidence": probs[defect_type],
        "probabilities": probs,
        "model_type": "Fallback (install Pillow for pixel analysis)",
        "image_path": image_path,
        "mode": "fallback",
    }
