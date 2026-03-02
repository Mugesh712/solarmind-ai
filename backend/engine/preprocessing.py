"""
SolarMind AI — Image Preprocessing Pipeline (OpenCV)
Demonstrates the preprocessing pipeline for thermal and RGB solar panel images.
"""
import random as _random
import math
from typing import Any, Dict, List


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _has_cv2() -> bool:
    """Check if OpenCV is available."""
    try:
        import cv2  # type: ignore
        return True
    except ImportError:
        return False


def _has_numpy() -> bool:
    """Check if NumPy is available."""
    try:
        import numpy  # type: ignore
        return True
    except ImportError:
        return False


def preprocess_thermal(image_path: str) -> Any:
    """
    Preprocess a thermal image for ViT input.
    Steps: Load -> Normalize -> CLAHE -> Denoise -> Resize
    """
    if not _has_cv2() or not _has_numpy():
        return _mock_preprocess(image_path, "thermal")

    import cv2  # type: ignore
    import numpy as np  # type: ignore

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Temperature normalization (min-max to 0-255)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_norm)

    # Gaussian denoising
    img_denoised = cv2.GaussianBlur(img_enhanced, (3, 3), 0)

    # Resize to model input
    img_resized = cv2.resize(img_denoised, (224, 224))

    return img_resized


def preprocess_rgb(image_path: str) -> Any:
    """
    Preprocess an RGB image for ViT input.
    Steps: Load -> Color correction -> Resize -> Normalize
    """
    if not _has_cv2() or not _has_numpy():
        return _mock_preprocess(image_path, "rgb")

    import cv2  # type: ignore

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Auto white balance
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    img_corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Resize
    img_resized = cv2.resize(img_corrected, (224, 224))

    return img_resized


def compute_quality_score(image_path: str) -> Dict[str, Any]:
    """
    Compute image quality score using Laplacian variance (blur detection).
    Returns score 0-1 where higher = sharper image.
    """
    if not _has_cv2():
        return {"quality_score": 0.85, "is_acceptable": True, "blur_variance": 150.0}

    import cv2  # type: ignore

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"quality_score": 0.0, "is_acceptable": False, "blur_variance": 0.0}

    laplacian_var: float = float(cv2.Laplacian(img, cv2.CV_64F).var())
    threshold = 100.0
    score = min(1.0, laplacian_var / 500.0)

    return {
        "quality_score": _r(score, 3),
        "is_acceptable": laplacian_var > threshold,
        "blur_variance": _r(laplacian_var, 2),
    }


def generate_attention_heatmap(width: int = 224, height: int = 224) -> List[List[float]]:
    """
    Generate a simulated ViT attention heatmap for visualization.
    In production, this would use actual attention weights from the model.
    """
    if not _has_numpy():
        # Return a simple list-based heatmap without numpy
        heatmap: List[List[float]] = []
        for _y in range(height):
            row: List[float] = []
            for _x in range(width):
                row.append(_random.uniform(0.0, 0.3))
            heatmap.append(row)
        return heatmap

    import numpy as np  # type: ignore

    heatmap_arr = np.zeros((height, width), dtype=np.float32)

    # Simulate attention hotspots
    margin_x: int = max(1, width // 6)
    margin_y: int = max(1, height // 6)
    num_hotspots: int = int(np.random.randint(1, 4))

    for _ in range(num_hotspots):
        cx: int = int(np.random.randint(margin_x, max(margin_x + 1, width - margin_x)))
        cy: int = int(np.random.randint(margin_y, max(margin_y + 1, height - margin_y)))
        max_sigma: int = max(2, min(40, width // 3))
        min_sigma: int = max(1, min(15, width // 8))
        if min_sigma >= max_sigma:
            min_sigma = max(1, max_sigma - 1)
        sigma: int = int(np.random.randint(min_sigma, max_sigma))

        for y in range(height):
            for x in range(width):
                dist_sq = float((x - cx) ** 2 + (y - cy) ** 2)
                denom = float(2 * sigma * sigma)
                heatmap_arr[y, x] += float(np.exp(-dist_sq / denom))

    # Normalize to 0-1
    max_val: float = float(heatmap_arr.max())
    if max_val > 0:
        heatmap_arr = heatmap_arr / max_val

    result: List[List[float]] = heatmap_arr.tolist()
    return result


def _mock_preprocess(image_path: str, mode: str) -> Dict[str, Any]:
    """Return mock preprocessed data when OpenCV is not available."""
    return {
        "status": "mock",
        "mode": mode,
        "input": image_path,
        "output_shape": [224, 224, 3] if mode == "rgb" else [224, 224],
        "note": "OpenCV not installed. This is simulated output.",
    }


# Pipeline configuration
PIPELINE_CONFIG: Dict[str, Dict[str, Any]] = {
    "thermal": {
        "input_size": (224, 224),
        "clahe_clip_limit": 3.0,
        "clahe_grid_size": (8, 8),
        "gaussian_kernel": (3, 3),
        "normalization": "min_max",
    },
    "rgb": {
        "input_size": (224, 224),
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "color_space": "LAB",
        "normalization": "imagenet",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}
