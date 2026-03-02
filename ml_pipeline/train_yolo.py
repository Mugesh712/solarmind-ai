"""
SolarMind AI — YOLOv8 Training Script
Trains YOLOv8 for solar panel defect detection and localization.

Usage:
    pip install ultralytics
    python train_yolo.py

Note: Requires a YOLO-format labeled dataset.
"""
import os
import random
from typing import Any, Dict, List


CONFIG: Dict[str, Any] = {
    "model_variant": "yolov8m.pt",  # medium variant
    "data_yaml": "./data/solar_defects.yaml",
    "epochs": 100,
    "img_size": 640,
    "batch_size": 16,
    "patience": 20,
    "output_dir": "./runs/detect/solarmind",
    "classes": ["micro_crack", "hotspot", "dust_soiling", "panel"],
}


def check_ultralytics() -> bool:
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def create_sample_data_yaml() -> None:
    """Create a sample data.yaml for reference."""
    yaml_content = """# SolarMind AI — YOLOv8 Dataset Configuration
# Place this file alongside your dataset

path: ./data/solar_defects
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: micro_crack
  1: hotspot
  2: dust_soiling
  3: panel

# Dataset structure:
# data/solar_defects/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
"""
    data_yaml = str(CONFIG["data_yaml"])
    parent_dir = os.path.dirname(data_yaml) or "."
    os.makedirs(parent_dir, exist_ok=True)
    with open(data_yaml, "w") as f:
        f.write(yaml_content)
    print(f"✅ Sample data.yaml created at {data_yaml}")


def train_yolo() -> None:
    """Train YOLOv8 model."""
    from ultralytics import YOLO  # type: ignore[import-untyped]

    print("\n" + "=" * 60)
    print("  SOLARMIND AI — YOLOv8 Training Pipeline")
    print("=" * 60 + "\n")

    data_yaml = str(CONFIG["data_yaml"])
    if not os.path.exists(data_yaml):
        print("⚠️  Dataset config not found. Creating sample data.yaml...")
        create_sample_data_yaml()
        print("   Please prepare your dataset and update data.yaml\n")
        run_demo()
        return

    model = YOLO(str(CONFIG["model_variant"]))

    _results = model.train(
        data=data_yaml,
        epochs=int(CONFIG["epochs"]),
        imgsz=int(CONFIG["img_size"]),
        batch=int(CONFIG["batch_size"]),
        patience=int(CONFIG["patience"]),
        project=str(CONFIG["output_dir"]),
        name="train",
        device="0",  # GPU 0, or "cpu"
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.3,
        fliplr=0.5,
    )

    print("\n✅ Training complete!")
    print(f"   Results saved to: {CONFIG['output_dir']}")


def run_demo() -> None:
    """Simulate YOLOv8 training output."""
    print("\n" + "=" * 60)
    print("  SOLARMIND AI — YOLOv8 Training (Demo Mode)")
    print("=" * 60 + "\n")
    print(f"Model: {CONFIG['model_variant']}")
    print(f"Image Size: {CONFIG['img_size']}")
    print(f"Classes: {CONFIG['classes']}")
    print()

    for epoch in range(1, 11):
        box_loss = 3.5 * (0.8 ** epoch) + random.uniform(-0.1, 0.1)
        cls_loss = 2.0 * (0.78 ** epoch) + random.uniform(-0.05, 0.05)
        mAP50 = min(0.95, 0.4 + epoch * 0.055 + random.uniform(-0.02, 0.02))
        mAP5095 = min(0.85, 0.25 + epoch * 0.05 + random.uniform(-0.02, 0.02))

        print(f"Epoch [{epoch:3d}/100]  "
              f"Box Loss: {box_loss:.4f}  Cls Loss: {cls_loss:.4f}  "
              f"mAP@50: {mAP50:.3f}  mAP@50:95: {mAP5095:.3f}")

    print("\n... (remaining epochs simulated)")
    print("\n✅ Training complete (demo)")
    print("   Best mAP@0.5: 0.891")
    print("   Best mAP@0.5:0.95: 0.724")


def main() -> None:
    if not check_ultralytics():
        print("⚠️  ultralytics not installed. Install with: pip install ultralytics")
        print("\nRunning demo mode...\n")
        run_demo()
        return

    train_yolo()


if __name__ == "__main__":
    main()
