"""
DataLoader partagé pour FaceForensics++
Utilisé par les 3 scripts d'entraînement
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import Counter
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = "C:/Users/DELL/Desktop/IRM_2/Semestre2/deepfake-detection-system/data/processed/FaceForensics_224"

CATEGORIES = ['real', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures']

CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IDX_TO_CLASS = {idx: cat for cat, idx in CLASS_TO_IDX.items()}

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0  # CPU local → toujours 0


# ============================================================
# TRANSFORMS
# ============================================================
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# DATASET CLASS
# ============================================================
class FaceForensicsDataset(Dataset):
    def __init__(self, data_dir, categories, split='train',
                 train_ratio=0.7, val_ratio=0.2, transform=None, seed=42):
        """
        Args:
            data_dir   : Chemin vers les images preprocessées
            categories : Liste des classes
            split      : 'train', 'val', ou 'test'
            transform  : Transformations à appliquer
        """
        self.transform = transform
        self.samples = []  # [(image_path, label), ...]

        np.random.seed(seed)

        for category in categories:
            cat_path = Path(data_dir) / category
            if not cat_path.exists():
                print(f"⚠️  Catégorie introuvable : {cat_path}")
                continue

            images = sorted(list(cat_path.rglob('*.jpg')))
            if not images:
                print(f"⚠️  Aucune image dans : {cat_path}")
                continue

            # Split train/val/test
            n = len(images)
            indices = np.random.permutation(n)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            if split == 'train':
                selected = [images[i] for i in indices[:train_end]]
            elif split == 'val':
                selected = [images[i] for i in indices[train_end:val_end]]
            else:  # test
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

    def get_labels(self):
        return [label for _, label in self.samples]


# ============================================================
# WEIGHTED SAMPLER (corrige le déséquilibre des classes)
# ============================================================
def get_weighted_sampler(dataset):
    labels = dataset.get_labels()
    class_counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ============================================================
# FONCTION PRINCIPALE : get_dataloaders()
# ============================================================
def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Retourne train_loader, val_loader, test_loader
    """
    train_dataset = FaceForensicsDataset(
        data_dir, CATEGORIES, split='train', transform=train_transforms)
    val_dataset = FaceForensicsDataset(
        data_dir, CATEGORIES, split='val', transform=val_transforms)
    test_dataset = FaceForensicsDataset(
        data_dir, CATEGORIES, split='test', transform=val_transforms)

    # Weighted sampler pour équilibrer les classes au training
    sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=sampler, num_workers=NUM_WORKERS)

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS)

    print(f"\n📊 Dataset chargé :")
    print(f"   Train : {len(train_dataset)} images")
    print(f"   Val   : {len(val_dataset)} images")
    print(f"   Test  : {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


# ============================================================
# TEST RAPIDE
# ============================================================
if __name__ == "__main__":
    print("🔍 Test du DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Vérifier un batch
    images, labels = next(iter(train_loader))
    print(f"\n✅ Batch shape  : {images.shape}")
    print(f"✅ Labels shape : {labels.shape}")
    print(f"✅ Classes      : {[IDX_TO_CLASS[l.item()] for l in labels[:5]]}")
    print("\n🎉 DataLoader OK !")