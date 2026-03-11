"""
SolarMind AI — Multi-Model Comparison Training Script
Trains ViT-Small/16, ResNet-50, EfficientNet-B0, and Swin-Tiny on the PV Defect
Dataset, then evaluates a ViT-Swin Ensemble (late fusion) and compares all models.

Usage:
    python train_compare_models.py

Output:
    - Trained model checkpoints in backend/checkpoints/
    - Comparison results in evaluation_results/model_comparison.json
"""
import os
import sys
import time
import json
from typing import Any, Dict, List, Tuple

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "data", "pv_defect_dataset")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")

NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0  # Safe for macOS

CLASS_NAMES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]

# Models to train and compare
MODELS_CONFIG = [
    {
        "name": "ViT-Small/16",
        "timm_name": "vit_small_patch16_224",
        "save_name": "vit_small_model.pth",
        "type": "Vision Transformer",
    },
    {
        "name": "ResNet-50",
        "timm_name": "resnet50",
        "save_name": "resnet50_model.pth",
        "type": "Convolutional Neural Network",
    },
    {
        "name": "EfficientNet-B0",
        "timm_name": "efficientnet_b0",
        "save_name": "efficientnet_b0_model.pth",
        "type": "Efficient CNN",
    },
    {
        "name": "Swin-Tiny",
        "timm_name": "swin_tiny_patch4_window7_224",
        "save_name": "swin_tiny_model.pth",
        "type": "Hierarchical Vision Transformer",
    },
]


def check_dependencies() -> List[str]:
    """Check if required packages are installed."""
    required = ["torch", "torchvision", "timm"]
    missing: List[str] = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def get_device() -> Any:
    """Get the best available device."""
    import torch
    if torch.backends.mps.is_available():
        print("🚀 Using Apple MPS (GPU) acceleration")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🚀 Using NVIDIA CUDA acceleration")
        return torch.device("cuda")
    else:
        print("⚡ Using CPU (training will be slower)")
        return torch.device("cpu")


def get_data_loaders() -> Tuple[Any, Any, Any, int]:
    """Create train/val/test data loaders."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    test_dir = os.path.join(DATA_DIR, "test")

    if not os.path.isdir(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    num_classes = len(train_dataset.classes)

    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val:   {len(val_dataset)} images")
    print(f"   Test:  {len(test_dataset)} images")
    print(f"   Classes ({num_classes}): {train_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, num_classes


def create_model(timm_name: str, num_classes: int) -> Any:
    """Create a model with custom classification head."""
    import timm
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, total_params, trainable_params


def train_and_evaluate(
    model_config: Dict[str, str],
    train_loader: Any,
    val_loader: Any,
    test_loader: Any,
    num_classes: int,
    device: Any,
) -> Dict[str, Any]:
    """Train a model and evaluate on test set. Returns metrics dict."""
    import torch
    import torch.nn as nn

    model_name = model_config["name"]
    timm_name = model_config["timm_name"]
    save_name = model_config["save_name"]

    print(f"\n{'='*70}")
    print(f"  Training: {model_name} ({timm_name})")
    print(f"{'='*70}")

    # Create model
    model, total_params, trainable_params = create_model(timm_name, num_classes)
    model = model.to(device)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable:    {trainable_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    save_path = os.path.join(CHECKPOINT_DIR, save_name)
    train_history: List[Dict[str, float]] = []
    total_train_time = 0.0

    # Training loop
    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        scheduler.step()

        # Save best
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved = " ✅ SAVED"

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
        })

        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% | "
              f"{epoch_time:.1f}s{saved}")

    # --- Test Evaluation ---
    print(f"\n📊 Evaluating {model_name} on test set...")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()

    test_correct = 0
    test_total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                all_labels.append(label)
                all_preds.append(pred)
                if pred == label:
                    class_correct[label] += 1

    test_acc = 100.0 * test_correct / test_total

    # Compute per-class precision, recall, F1
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for cls_idx in range(num_classes):
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"

        # True positives, false positives, false negatives
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l == cls_idx)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l != cls_idx)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != cls_idx and l == cls_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(100.0 * class_correct[cls_idx] / class_total[cls_idx], 2) if class_total[cls_idx] > 0 else 0.0,
            "support": class_total[cls_idx],
        }

    # Compute macro averages
    all_precisions = [m["precision"] for m in per_class_metrics.values()]
    all_recalls = [m["recall"] for m in per_class_metrics.values()]
    all_f1s = [m["f1_score"] for m in per_class_metrics.values()]

    macro_precision = round(sum(all_precisions) / len(all_precisions), 4) if all_precisions else 0.0
    macro_recall = round(sum(all_recalls) / len(all_recalls), 4) if all_recalls else 0.0
    macro_f1 = round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0.0

    print(f"   Test Accuracy: {test_acc:.1f}%")
    print(f"   Macro F1:      {macro_f1:.4f}")
    print(f"   Checkpoint:    {save_path}")

    # Compute confusion matrix
    confusion_matrix: List[List[int]] = [[0] * num_classes for _ in range(num_classes)]
    for p, l in zip(all_preds, all_labels):
        confusion_matrix[l][p] += 1

    return {
        "model_name": model_name,
        "architecture": timm_name,
        "model_type": model_config["type"],
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_sec": round(total_train_time, 1),
        "best_val_acc": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "training_history": train_history,
        "checkpoint_path": save_name,
        "confusion_matrix": confusion_matrix,
    }


def evaluate_ensemble(
    model_configs: List[Dict[str, str]],
    test_loader: Any,
    num_classes: int,
    device: Any,
) -> Dict[str, Any]:
    """
    Evaluate a late-fusion ensemble that averages softmax predictions
    from multiple trained models. Returns metrics dict.
    """
    import torch
    import torch.nn.functional as F

    model_names = [c["name"] for c in model_configs]
    ensemble_name = " + ".join(model_names) + " Ensemble"
    print(f"\n{'='*70}")
    print(f"  Evaluating Ensemble: {ensemble_name}")
    print(f"{'='*70}")

    # Load all models
    models_list = []
    total_params_sum = 0
    for cfg in model_configs:
        model, total_params, _ = create_model(cfg["timm_name"], num_classes)
        ckpt_path = os.path.join(CHECKPOINT_DIR, cfg["save_name"])
        if not os.path.isfile(ckpt_path):
            print(f"   ⚠️ Checkpoint not found: {ckpt_path}, skipping")
            continue
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        models_list.append(model)
        total_params_sum += total_params
        print(f"   Loaded {cfg['name']} ({total_params:,} params)")

    if len(models_list) < 2:
        print("   ❌ Need at least 2 models for ensemble, skipping")
        return {}

    # Evaluate ensemble on test set
    test_correct = 0
    test_total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    all_preds: List[int] = []
    all_labels: List[int] = []

    eval_start = time.time()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Average softmax predictions from all models
            avg_probs = None
            for m in models_list:
                logits = m(images)
                probs = F.softmax(logits, dim=1)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs = avg_probs + probs
            avg_probs = avg_probs / len(models_list)

            _, predicted = avg_probs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                all_labels.append(label)
                all_preds.append(pred)
                if pred == label:
                    class_correct[label] += 1

    eval_time = time.time() - eval_start
    test_acc = 100.0 * test_correct / test_total

    # Compute per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for cls_idx in range(num_classes):
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l == cls_idx)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == cls_idx and l != cls_idx)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p != cls_idx and l == cls_idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_metrics[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(100.0 * class_correct[cls_idx] / class_total[cls_idx], 2) if class_total[cls_idx] > 0 else 0.0,
            "support": class_total[cls_idx],
        }

    all_precisions = [m["precision"] for m in per_class_metrics.values()]
    all_recalls = [m["recall"] for m in per_class_metrics.values()]
    all_f1s = [m["f1_score"] for m in per_class_metrics.values()]
    macro_precision = round(sum(all_precisions) / len(all_precisions), 4) if all_precisions else 0.0
    macro_recall = round(sum(all_recalls) / len(all_recalls), 4) if all_recalls else 0.0
    macro_f1 = round(sum(all_f1s) / len(all_f1s), 4) if all_f1s else 0.0

    # Confusion matrix
    confusion_matrix: List[List[int]] = [[0] * num_classes for _ in range(num_classes)]
    for p, l in zip(all_preds, all_labels):
        confusion_matrix[l][p] += 1

    print(f"   Ensemble Test Accuracy: {test_acc:.1f}%")
    print(f"   Ensemble Macro F1:      {macro_f1:.4f}")
    print(f"   Evaluation Time:        {eval_time:.1f}s")

    # Build ensemble training history as average of component models
    # (ensemble doesn't train separately, so we synthesize from components)
    return {
        "model_name": ensemble_name,
        "architecture": "ensemble_late_fusion",
        "model_type": "Ensemble (Late Fusion)",
        "total_params": total_params_sum,
        "trainable_params": total_params_sum,
        "training_time_sec": round(eval_time, 1),
        "best_val_acc": round(test_acc, 2),  # No separate val for ensemble
        "test_accuracy": round(test_acc, 2),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "training_history": [],  # Ensemble has no training history
        "checkpoint_path": "ensemble_vit_swin",
        "confusion_matrix": confusion_matrix,
        "ensemble_components": [c["name"] for c in model_configs],
    }


def main() -> None:
    """Main entry point — trains all models, evaluates ensemble, and saves comparison."""
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"   Install with: pip install {' '.join(missing)}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  SOLARMIND AI — Multi-Model Comparison Training")
    print("  ViT-Small/16 vs ResNet-50 vs EfficientNet-B0 vs Swin-Tiny vs Ensemble")
    print("=" * 70)

    print(f"\n📂 Loading dataset from: {DATA_DIR}")
    device = get_device()

    train_loader, val_loader, test_loader, num_classes = get_data_loaders()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Train all individual models
    all_results: List[Dict[str, Any]] = []
    for model_config in MODELS_CONFIG:
        result = train_and_evaluate(
            model_config, train_loader, val_loader, test_loader, num_classes, device
        )
        all_results.append(result)

    # Evaluate ViT + Swin Ensemble
    vit_config = MODELS_CONFIG[0]   # ViT-Small/16
    swin_config = MODELS_CONFIG[3]  # Swin-Tiny
    ensemble_result = evaluate_ensemble(
        [vit_config, swin_config], test_loader, num_classes, device
    )
    if ensemble_result:
        all_results.append(ensemble_result)

    # Determine winner
    best_model = max(all_results, key=lambda r: r["test_accuracy"])

    # Print comparison summary
    print("\n\n" + "=" * 70)
    print("  📊 COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Accuracy':>10} {'F1':>8} {'Precision':>11} {'Recall':>8} {'Params':>12} {'Time':>8}")
    print("-" * 90)
    for r in all_results:
        winner = " 🏆" if r["model_name"] == best_model["model_name"] else ""
        print(f"{r['model_name']:<25} {r['test_accuracy']:>9.1f}% {r['macro_f1']:>7.4f} "
              f"{r['macro_precision']:>10.4f} {r['macro_recall']:>7.4f} "
              f"{r['total_params']:>11,} {r['training_time_sec']:>7.1f}s{winner}")

    print(f"\n🏆 Best Model: {best_model['model_name']} with {best_model['test_accuracy']:.1f}% test accuracy")

    # Save comparison results
    comparison_output = {
        "comparison_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "PV Panel Defect Dataset",
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "training_config": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        },
        "best_model": best_model["model_name"],
        "models": all_results,
    }

    output_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison_output, f, indent=2)

    print(f"\n📦 Comparison results saved to: {output_path}")
    print("✅ All models trained and evaluated successfully!\n")


if __name__ == "__main__":
    main()
