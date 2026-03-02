"""
SolarMind AI — Real ViT Training Script
Fine-tunes a Vision Transformer on the PV Panel Defect Dataset.
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "data", "pv_defect_dataset")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", "checkpoints")
MODEL_NAME = "vit_small_patch16_224"  # Lighter ViT for faster training on CPU/MPS
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 0  # Safe for macOS

CLASS_NAMES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]

# ─── Device ───────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple MPS (GPU) acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("🚀 Using NVIDIA CUDA acceleration")
else:
    device = torch.device("cpu")
    print("⚡ Using CPU (training will be slower)")

# ─── Data Transforms ─────────────────────────────────────────────────────────
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

# ─── Load Dataset ─────────────────────────────────────────────────────────────
print(f"\n📂 Loading dataset from: {DATA_DIR}")

train_dir = os.path.join(DATA_DIR, "train")
val_dir = os.path.join(DATA_DIR, "val")
test_dir = os.path.join(DATA_DIR, "test")

if not os.path.isdir(train_dir):
    print(f"❌ Training directory not found: {train_dir}")
    sys.exit(1)

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

# Map class indices to our class names
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
num_classes = len(train_dataset.classes)

print(f"   Train: {len(train_dataset)} images")
print(f"   Val:   {len(val_dataset)} images")
print(f"   Test:  {len(test_dataset)} images")
print(f"   Classes ({num_classes}): {train_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ─── Model ────────────────────────────────────────────────────────────────────
print(f"\n🧠 Loading pre-trained {MODEL_NAME}...")
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")

# ─── Training Setup ──────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ─── Training Loop ───────────────────────────────────────────────────────────
print(f"\n🔥 Starting training: {NUM_EPOCHS} epochs, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
print("=" * 70)

best_val_acc = 0.0
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
save_path = os.path.join(CHECKPOINT_DIR, "classifier_model.pth")

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

        # Progress
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} [{batch_idx+1}/{len(train_loader)}] "
                  f"loss={loss.item():.4f} acc={100.*train_correct/train_total:.1f}%", end="\r")

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

    scheduler.step()

    # Save best model
    saved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), save_path)
        saved = " ✅ SAVED"

    print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
          f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
          f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% | "
          f"{epoch_time:.1f}s{saved}")

print("=" * 70)
print(f"\n🏆 Best validation accuracy: {best_val_acc:.1f}%")
print(f"📦 Model saved to: {save_path}")

# ─── Test Evaluation ─────────────────────────────────────────────────────────
print(f"\n📊 Evaluating on test set ({len(test_dataset)} images)...")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

test_correct = 0
test_total = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)

        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1

test_acc = 100.0 * test_correct / test_total
print(f"\n   Overall Test Accuracy: {test_acc:.1f}%\n")

print("   Per-class accuracy:")
for i in range(num_classes):
    cls_name = idx_to_class[i]
    if class_total[i] > 0:
        cls_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"     {cls_name:20s}: {cls_acc:5.1f}% ({class_correct[i]}/{class_total[i]})")

print(f"\n✅ Training complete! Model ready at: {save_path}")
