"""
ResNet18 - Meilleure généralisation
FaceForensics++ c23 - 5 classes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import mlflow
import time
from sklearn.metrics import classification_report
import os

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "ResNet18-Final"
EXPERIMENT_NAME = "Deepfake-Detection"

EPOCHS = 20           # plus d'epochs
BATCH_SIZE = 32
LR = 5e-5             # LR plus bas pour meilleure généralisation
STEP_SIZE = 7
GAMMA = 0.3
NUM_CLASSES = 5
WEIGHT_DECAY = 1e-4   # régularisation

mlflow.set_tracking_uri("sqlite:///C:/Users/DELL/Desktop/IRM_2/Semestre2/deepfake-detection-system/mlflow.db")

DATA_DIR = r"C:\Users\DELL\Desktop\IRM_2\Semestre2\deepfake-detection-system\data\processed\FaceForensics_224"
SAVE_DIR = Path(r"C:\Users\DELL\Desktop\IRM_2\Semestre2\deepfake-detection-system\results\models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = SAVE_DIR / "resnet18_final_best.pth"

CATEGORIES = ['real', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures']
CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
DEVICE = torch.device("cpu")

# ============================================================
# AUGMENTATION AGRESSIVE pour meilleure généralisation
# ============================================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    ], p=0.3),
    transforms.RandomApply([
        transforms.RandomPerspective(distortion_scale=0.2)
    ], p=0.3),
    # Simuler compression JPEG smartphone
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))
    ], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # Simuler bruit
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# DATASET
# ============================================================
class FaceForensicsDataset(Dataset):
    def __init__(self, data_dir, categories, split='train',
                 train_ratio=0.7, val_ratio=0.2, transform=None, seed=42):
        self.transform = transform
        self.samples = []
        np.random.seed(seed)

        for category in categories:
            cat_path = Path(data_dir) / category
            if not cat_path.exists():
                continue
            images = sorted(list(cat_path.rglob('*.jpg')))
            n = len(images)
            indices = np.random.permutation(n)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            if split == 'train':
                selected = [images[i] for i in indices[:train_end]]
            elif split == 'val':
                selected = [images[i] for i in indices[train_end:val_end]]
            else:
                selected = [images[i] for i in indices[val_end:]]

            label = CLASS_TO_IDX[category]
            for img_path in selected:
                self.samples.append((img_path, label))

        print(f"✅ Split '{split}' : {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(batch_size=BATCH_SIZE):
    train_ds = FaceForensicsDataset(DATA_DIR, CATEGORIES, 'train', transform=train_transforms)
    val_ds   = FaceForensicsDataset(DATA_DIR, CATEGORIES, 'val',   transform=val_transforms)
    test_ds  = FaceForensicsDataset(DATA_DIR, CATEGORIES, 'test',  transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# ============================================================
# MODÈLE — Fine-tuning complet avec toutes les couches
# ============================================================
def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Dégeler TOUTES les couches — fine-tuning complet
    for param in model.parameters():
        param.requires_grad = True

    # Remplacer fc avec dropout fort
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total params     : {total:,}")
    print(f"📊 Trainable params : {trainable:,}")
    return model


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"🚀 Entraînement : {MODEL_NAME}")
    print(f"🎯 Objectif : Meilleure généralisation sur images réelles")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders()
    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing

    # Optimizer différentiel — LR plus bas pour les premières couches
    optimizer = optim.AdamW([
        {'params': model.layer1.parameters(), 'lr': LR * 0.1},
        {'params': model.layer2.parameters(), 'lr': LR * 0.3},
        {'params': model.layer3.parameters(), 'lr': LR * 0.7},
        {'params': model.layer4.parameters(), 'lr': LR},
        {'params': model.fc.parameters(),     'lr': LR * 2},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing — meilleur que StepLR pour la généralisation
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": 0.1,
            "augmentation": "aggressive",
            "fine_tuning": "full",
            "num_classes": NUM_CLASSES,
        })

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()
            epoch_time = time.time() - epoch_start

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            }, step=epoch)

            print(f"Epoch [{epoch:02d}/{EPOCHS}] "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                  f"Loss: {val_loss:.4f} | Time: {epoch_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'model_name': MODEL_NAME
                }, SAVE_PATH)
                print(f"   💾 Sauvegardé (val_acc={val_acc:.4f})")

        total_time = time.time() - start_time

        print("\n📊 Évaluation finale...")
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

        print(f"\n🎯 RÉSULTATS FINAUX")
        print("=" * 60)
        print(f"   Best Val Accuracy : {best_val_acc*100:.2f}%")
        print(f"   Test Accuracy     : {test_acc*100:.2f}%")
        print(f"   Temps total       : {total_time/60:.1f} minutes")
        print("=" * 60)
        print(classification_report(test_preds, test_labels, target_names=CATEGORIES))

        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "training_time_minutes": total_time / 60
        })


if __name__ == "__main__":
    main()