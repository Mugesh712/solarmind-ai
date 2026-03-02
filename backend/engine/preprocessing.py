"""
SolarMind AI — Image Preprocessing Pipeline (OpenCV)
Preprocessing pipeline for thermal and RGB solar panel images.
Uses real OpenCV processing when available, falls back to mock data otherwise.
"""
import os
import math
import hashlib
import random as _random
from typing import Any, Dict, List


def _r(value: float, ndigits: int = 0) -> float:
    """Type-safe rounding helper."""
    multiplier: float = 10.0 ** ndigits
    return math.floor(value * multiplier + 0.5) / multiplier


def _seed_for(key: str) -> int:
    """Generate a deterministic seed from a string key."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


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
        return {
            "quality_score": 0.85,
            "is_acceptable": True,
            "blur_variance": 150.0,
            "note": "OpenCV not installed — using estimate",
        }

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


def generate_attention_heatmap(
    width: int = 224, height: int = 224, image_path: str = ""
) -> List[List[float]]:
    """
    Generate a ViT attention heatmap for visualization.

    When a trained ViT model and image_path are available, extracts real
    attention weights from the model. Otherwise generates a deterministic
    simulated heatmap.
    """
    # Try real attention extraction
    if image_path and os.path.isfile(image_path):
        heatmap = _try_real_attention_heatmap(image_path, width, height)
        if heatmap is not None:
            return heatmap

    # Deterministic simulated heatmap
    return _simulated_attention_heatmap(width, height, image_path)


def _try_real_attention_heatmap(
    image_path: str, width: int, height: int
) -> "List[List[float]] | None":
    """Try to extract real attention heatmap from ViT model."""
    try:
        from models.vit_classifier import _load_model, _has_pil  # type: ignore

        model, device = _load_model()
        if model is None or not _has_pil():
            return None

        import torch  # type: ignore
        from torchvision import transforms  # type: ignore
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Extract attention from last block
        attention_data: List[Any] = []

        def hook_fn(module: Any, input: Any, output: Any) -> None:
            attention_data.append(output)

        hook = model.blocks[-1].attn.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = model(img_tensor)

        hook.remove()

        if attention_data:
            attn_output = attention_data[-1]
            if isinstance(attn_output, torch.Tensor) and attn_output.dim() >= 2:
                # Average attention across spatial dims
                attn_map = attn_output[0].mean(dim=0) if attn_output.dim() >= 3 else attn_output[0]
                attn_np = attn_map.cpu().numpy()

                # Reshape to spatial grid if possible (14x14 for ViT with 224/16 patches)
                num_patches = 14
                if attn_np.size >= num_patches * num_patches:
                    patch_attn = attn_np[:num_patches * num_patches].reshape(num_patches, num_patches)
                else:
                    patch_attn = attn_np.reshape(
                        int(np.sqrt(attn_np.size)), -1
                    )

                # Normalize to 0-1
                max_val = patch_attn.max()
                if max_val > 0:
                    patch_attn = patch_attn / max_val

                # Resize to requested dimensions
                from PIL import Image as PILImage
                attn_img = PILImage.fromarray((patch_attn * 255).astype(np.uint8))
                attn_resized = attn_img.resize((width, height), PILImage.BILINEAR)
                attn_array = np.array(attn_resized, dtype=np.float32) / 255.0

                return attn_array.tolist()

    except Exception as e:
        print(f"[SolarMind] Real attention heatmap extraction failed: {e}")

    return None


def _simulated_attention_heatmap(
    width: int, height: int, image_path: str = ""
) -> List[List[float]]:
    """Generate a deterministic simulated attention heatmap using gaussian blobs."""
    if not _has_numpy():
        rng = _random.Random(_seed_for(image_path) if image_path else 7)
        heatmap: List[List[float]] = []
        for _y in range(height):
            row: List[float] = []
            for _x in range(width):
                row.append(_r(rng.uniform(0.0, 0.3), 2))
            heatmap.append(row)
        return heatmap

    import numpy as np  # type: ignore

    rng = np.random.RandomState(_seed_for(image_path) if image_path else 7)
    heatmap_arr = np.zeros((height, width), dtype=np.float32)

    # Deterministic attention hotspots
    margin_x: int = max(1, width // 6)
    margin_y: int = max(1, height // 6)
    num_hotspots: int = int(rng.randint(1, 4))

    for _ in range(num_hotspots):
        cx: int = int(rng.randint(margin_x, max(margin_x + 1, width - margin_x)))
        cy: int = int(rng.randint(margin_y, max(margin_y + 1, height - margin_y)))
        max_sigma: int = max(2, min(40, width // 3))
        min_sigma: int = max(1, min(15, width // 8))
        if min_sigma >= max_sigma:
            min_sigma = max(1, max_sigma - 1)
        sigma: int = int(rng.randint(min_sigma, max_sigma))

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
        "note": "OpenCV not installed. Install with: pip install opencv-python",
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
