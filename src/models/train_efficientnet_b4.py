"""
Entraînement EfficientNet-B4 pour détection de deepfakes
Meilleur modèle pour précision + Grad-CAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from pathlib import Path
import mlflow
import time
from sklearn.metrics import classification_report
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, CATEGORIES

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "EfficientNet-B4"
EXPERIMENT_NAME = "Deepfake-Detection"

EPOCHS = 15
BATCH_SIZE = 8     
LR = 1e-4
STEP_SIZE = 5
GAMMA = 0.5
NUM_CLASSES = 5

mlflow.set_tracking_uri("sqlite:///C:/Users/DELL/Desktop/IRM_2/Semestre2/deepfake-detection-system/mlflow.db")

SAVE_DIR = Path(r"C:\Users\DELL\Desktop\IRM_2\Semestre2\deepfake-detection-system\results\models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = SAVE_DIR / "efficientnet_b4_best.pth"

DEVICE = torch.device("cpu")
print(f"🖥️  Device : {DEVICE}")


# ============================================================
# MODÈLE
# ============================================================
def build_model():
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

    # Geler toutes les couches
    for param in model.parameters():
        param.requires_grad = False

    # Dégeler les 3 derniers blocs — plus de couches que B0
    for param in model.features[6].parameters():
        param.requires_grad = True
    for param in model.features[7].parameters():
        param.requires_grad = True
    for param in model.features[8].parameters():
        param.requires_grad = True

    # Remplacer le classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Paramètres total     : {total:,}")
    print(f"📊 Paramètres entraînés : {trainable:,}")

    return model


# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
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
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=MODEL_NAME):
        mlflow.log_params({
            "model": MODEL_NAME,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "optimizer": "Adam",
            "scheduler": f"StepLR(step={STEP_SIZE}, gamma={GAMMA})",
            "num_classes": NUM_CLASSES,
            "unfrozen_blocks": "features[6,7,8], classifier",
            "classifier": "Linear(in→512→5) with ReLU+Dropout",
            "device": str(DEVICE)
        })

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()
            epoch_time = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "learning_rate": current_lr
            }, step=epoch)

            print(f"Epoch [{epoch:02d}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'model_name': MODEL_NAME
                }, SAVE_PATH)
                print(f"   💾 Meilleur modèle sauvegardé (val_acc={val_acc:.4f})")

        total_time = time.time() - start_time

        print("\n📊 Évaluation finale sur Test Set...")
        checkpoint = torch.load(SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)

        print(f"\n🎯 RÉSULTATS FINAUX - {MODEL_NAME}")
        print("=" * 60)
        print(f"   Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"   Test Accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   Temps total       : {total_time/60:.1f} minutes")
        print("=" * 60)
        print("\n📋 Classification Report :")
        print(classification_report(test_preds, test_labels, target_names=CATEGORIES))

        mlflow.log_metrics({
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "training_time_minutes": total_time / 60
        })

        print(f"\n✅ Run MLflow terminé !")


if __name__ == "__main__":
    main()