"""
SolarMind AI — Synthetic Dataset Generator
Generates a synthetic PV Defect Dataset for training and evaluation.

Usage:
    pip install pillow
    python generate_dataset.py

This creates a structured image dataset that can be used with
train_vit.py and train_yolo.py for training and evaluation.
"""
import os
import random
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Configuration ──
CONFIG: Dict[str, Any] = {
    "output_dir": "./data/pv_defect_dataset",
    "num_train": 800,
    "num_val": 200,
    "num_test": 200,
    "image_size": 224,
    "class_names": ["normal", "micro_crack", "hotspot", "dust_soiling"],
    "class_distribution": {
        "train": [0.40, 0.20, 0.20, 0.20],
        "val": [0.25, 0.25, 0.25, 0.25],
        "test": [0.25, 0.25, 0.25, 0.25],
    },
}

CLASS_NAMES: List[str] = ["normal", "micro_crack", "hotspot", "dust_soiling"]


def check_dependencies() -> List[str]:
    """Check if PIL/Pillow is available."""
    missing: List[str] = []
    try:
        from PIL import Image  # type: ignore[import-untyped]
    except ImportError:
        missing.append("pillow")
    return missing


def generate_panel_image(class_name: str, image_size: int = 224) -> Any:
    """
    Generate a synthetic solar panel image with simulated defects.
    Uses procedural generation — no real images required.
    """
    from PIL import Image, ImageDraw, ImageFilter  # type: ignore[import-untyped]

    # Base panel color (dark blue/navy for solar cells)
    base_r = random.randint(20, 50)
    base_g = random.randint(30, 60)
    base_b = random.randint(80, 140)

    img = Image.new("RGB", (image_size, image_size), (base_r, base_g, base_b))
    draw = ImageDraw.Draw(img)

    # Draw grid lines (solar cell separations)
    cell_size = image_size // random.randint(4, 8)
    grid_color = (max(0, base_r - 10), max(0, base_g - 10), max(0, base_b - 15))
    for x in range(0, image_size, cell_size):
        draw.line([(x, 0), (x, image_size)], fill=grid_color, width=1)
    for y in range(0, image_size, cell_size):
        draw.line([(0, y), (image_size, y)], fill=grid_color, width=1)

    if class_name == "micro_crack":
        # Draw thin crack lines
        num_cracks = random.randint(1, 4)
        for _ in range(num_cracks):
            x1 = random.randint(30, image_size - 30)
            y1 = random.randint(30, image_size - 30)
            length = random.randint(20, 80)
            angle = random.uniform(-0.8, 0.8)
            x2 = int(x1 + length * (1 + angle * 0.3))
            y2 = int(y1 + length * angle)
            crack_color = (
                min(255, base_r + random.randint(40, 100)),
                min(255, base_g + random.randint(20, 60)),
                base_b + random.randint(-10, 10),
            )
            draw.line([(x1, y1), (x2, y2)], fill=crack_color, width=random.randint(1, 3))
            # Add branching
            if random.random() > 0.4:
                bx = int((x1 + x2) / 2 + random.randint(-15, 15))
                by = int((y1 + y2) / 2 + random.randint(-15, 15))
                draw.line(
                    [(bx, by), (bx + random.randint(-25, 25), by + random.randint(-25, 25))],
                    fill=crack_color, width=1,
                )

    elif class_name == "hotspot":
        # Draw bright hot regions
        num_spots = random.randint(1, 3)
        for _ in range(num_spots):
            cx = random.randint(30, image_size - 30)
            cy = random.randint(30, image_size - 30)
            radius = random.randint(15, 45)
            for r_val in range(radius, 0, -2):
                intensity = int(180 + (radius - r_val) / radius * 75)
                hotspot_color = (
                    min(255, intensity + random.randint(0, 30)),
                    min(255, int(intensity * 0.6)),
                    min(255, int(intensity * 0.2)),
                )
                draw.ellipse(
                    [cx - r_val, cy - r_val, cx + r_val, cy + r_val],
                    fill=hotspot_color,
                )

    elif class_name == "dust_soiling":
        # Add dust particles on top of the panel
        draw_dust = ImageDraw.Draw(img)
        for _ in range(random.randint(30, 80)):
            px = random.randint(0, image_size - 1)
            py = random.randint(0, image_size - 1)
            ps = random.randint(1, 5)
            dust_r = random.randint(120, 180)
            dust_g = random.randint(110, 160)
            dust_b = random.randint(80, 140)
            draw_dust.ellipse([px, py, px + ps, py + ps], fill=(dust_r, dust_g, dust_b))

    # Apply slight blur for realism
    blur_radius = random.uniform(0.3, 0.8)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def generate_split(
    split_name: str,
    num_images: int,
    class_distribution: List[float],
    output_dir: str,
    image_size: int,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """Generate images for a specific split (train/val/test)."""
    split_dir = os.path.join(output_dir, split_name)

    # Create class directories
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    # Distribute images across classes
    labels: List[int] = random.choices(
        range(len(CLASS_NAMES)), weights=class_distribution, k=num_images
    )

    class_counts: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}
    metadata: List[Dict[str, Any]] = []

    for i, label_idx in enumerate(labels):
        cls = CLASS_NAMES[label_idx]
        class_counts[cls] += 1

        img = generate_panel_image(cls, image_size)
        filename = f"{cls}_{class_counts[cls]:04d}.png"
        filepath = os.path.join(split_dir, cls, filename)
        img.save(filepath)

        metadata.append({
            "filename": filename,
            "class": cls,
            "class_idx": label_idx,
            "split": split_name,
        })

        if (i + 1) % 50 == 0 or i == num_images - 1:
            print(f"  [{split_name}] Generated {i + 1}/{num_images} images")

    return class_counts, metadata


def generate_yolo_annotations(output_dir: str) -> None:
    """Generate YOLO-format annotation structure for detection training."""
    yolo_dir = os.path.join(output_dir, "yolo_format")

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(yolo_dir, "images", split)
        lbl_dir = os.path.join(yolo_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    # Create data.yaml
    defect_names = [c for c in CLASS_NAMES if c != "normal"]
    yaml_lines = [
        f"path: {yolo_dir}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(defect_names)}",
        f"names: {defect_names}",
    ]
    yaml_path = os.path.join(yolo_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("\n".join(yaml_lines) + "\n")

    print(f"\n✅ YOLO data.yaml created at {yaml_path}")
    print("   Note: For full YOLO training, copy images and create bounding box labels.")


def run_demo() -> None:
    """Run in demo mode when dependencies are missing."""
    print("\n" + "=" * 60)
    print("  SOLARMIND AI — Dataset Generator (Demo Mode)")
    print("=" * 60)
    print("\n⚠️  Pillow not installed. Showing what would be generated.\n")
    print(f"Output directory: {CONFIG['output_dir']}")
    print(f"Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"Classes: {CONFIG['class_names']}")
    num_train = int(CONFIG["num_train"])
    num_val = int(CONFIG["num_val"])
    num_test = int(CONFIG["num_test"])
    print(f"\nSplit breakdown:")
    print(f"  Train: {num_train} images")
    print(f"  Val:   {num_val} images")
    print(f"  Test:  {num_test} images")
    print(f"  Total: {num_train + num_val + num_test} images")
    print(f"\n✅ Install pillow to generate the dataset:")
    print(f"   pip install pillow")


def main() -> None:
    """Main entry point."""
    missing = check_dependencies()

    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        run_demo()
        return

    print("\n" + "=" * 60)
    print("  SOLARMIND AI — Synthetic PV Defect Dataset Generator")
    print("=" * 60 + "\n")

    output_dir = str(CONFIG["output_dir"])
    image_size = int(CONFIG["image_size"])
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output: {output_dir}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Classes: {CLASS_NAMES}\n")

    all_metadata: List[Dict[str, Any]] = []
    all_counts: Dict[str, Dict[str, int]] = {}

    splits: List[Tuple[str, int]] = [
        ("train", int(CONFIG["num_train"])),
        ("val", int(CONFIG["num_val"])),
        ("test", int(CONFIG["num_test"])),
    ]

    for split_name, num_images in splits:
        print(f"\n📂 Generating {split_name} split ({num_images} images)...")
        dist: List[float] = CONFIG["class_distribution"][split_name]
        counts, metadata = generate_split(split_name, num_images, dist, output_dir, image_size)
        all_counts[split_name] = counts
        all_metadata.extend(metadata)

    # Save metadata
    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    meta_content: Dict[str, Any] = {
        "splits": all_counts,
        "total_images": len(all_metadata),
    }
    with open(meta_path, "w") as f:
        json.dump(meta_content, f, indent=2)

    # Generate YOLO annotations reference
    generate_yolo_annotations(output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("  ✅ Dataset Generation Complete!")
    print("=" * 60)
    print(f"\n  Total images: {len(all_metadata)}")
    for split, counts in all_counts.items():
        total = sum(counts.values())
        breakdown = ", ".join(f"{k}: {v}" for k, v in counts.items())
        print(f"  {split:>5}: {total} ({breakdown})")
    print(f"\n  Metadata: {meta_path}")
    print(f"  Dataset ready for training with train_vit.py and train_yolo.py")


if __name__ == "__main__":
    main()
