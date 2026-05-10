"""
FastAPI Backend — Système de Détection de Deepfakes
Endpoints: /predict, /gradcam, /health, /model-info
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import base64
import io
import time
from pathlib import Path
from datetime import datetime
import os
from gradcam_plus_plus import GradCAMPlusPlus, generate_overlay_gradcampp, generate_heatmap_b64

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_PATH = Path(os.getenv("MODEL_PATH",
    r"C:\Users\DELL\Desktop\IRM_2\Semestre2\deepfake-detection-system\results\models\resnet18_final_best.pth"))

CATEGORIES = ['real', 'deepfakes', 'face2face', 'faceswap', 'neuraltextures']
IDX_TO_CLASS = {i: c for i, c in enumerate(CATEGORIES)}
NUM_CLASSES = 5
DEVICE = torch.device("cpu")
START_TIME = time.time()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="Deepfake Detection API",
    description="API de détection de deepfakes basée sur ResNet18 + GradCAM++",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODÈLE
# ============================================================
model = None
gradcam_instance = None
MODEL_VAL_ACC = 0.0


def load_model():
    """Charger ResNet18-Final avec le bon classifier"""
    m = models.resnet18(weights=None)
    # ← Même architecture que train_resnet18_final.py
    m.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(m.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.eval()
    print(f"✅ ResNet18-Final chargé (val_acc={checkpoint['val_acc']:.4f})")
    return m, checkpoint['val_acc']


@app.on_event("startup")
async def startup_event():
    global model, gradcam_instance, MODEL_VAL_ACC
    print("🚀 Démarrage de l'API...")
    model, MODEL_VAL_ACC = load_model()
    # ← GradCAM++ sur layer4[-1] — meilleure couche pour ResNet18
    target_layer = model.layer4[-1]
    gradcam_instance = GradCAMPlusPlus(model, target_layer)
    print("✅ API prête !")


# ============================================================
# HELPERS
# ============================================================
def preprocess_image(file_bytes: bytes) -> tuple:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        return image, tensor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image invalide : {str(e)}")


def image_to_b64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/health")
async def health():
    uptime_seconds = int(time.time() - START_TIME)
    uptime_str = f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m {uptime_seconds % 60}s"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "uptime": uptime_str,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/model-info")
async def model_info():
    return {
        "architecture": "ResNet18-Final",
        "task": "Deepfake Detection — Classification 5 classes",
        "dataset": "FaceForensics++ (compression c23)",
        "num_classes": NUM_CLASSES,
        "classes": CATEGORIES,
        "val_accuracy": round(float(MODEL_VAL_ACC) * 100, 2),
        "test_accuracy": 95.79,
        "f1_score_macro": 0.96,
        "input_size": "224x224 pixels",
        "training": {
            "epochs": 20,
            "batch_size": 32,
            "optimizer": "AdamW",
            "learning_rate": 0.00005,
            "scheduler": "CosineAnnealingLR",
            "augmentation": "Aggressive",
            "label_smoothing": 0.1,
            "device": "CPU"
        },
        "interpretability": "GradCAM++ (layer4[-1])"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400,
            detail=f"Format non supporté : {file.content_type}. Utilisez JPG ou PNG.")

    file_bytes = await file.read()
    original_img, input_tensor = preprocess_image(file_bytes)

    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0].numpy()
    inference_time = round((time.time() - start) * 1000, 2)

    pred_idx = int(np.argmax(probs))
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = float(probs[pred_idx])
    is_fake = pred_class != "real"

    return {
        "prediction": pred_class,
        "is_fake": is_fake,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            cat: round(float(prob) * 100, 2)
            for cat, prob in zip(CATEGORIES, probs)
        },
        "inference_time_ms": inference_time,
        "filename": file.filename
    }


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400,
            detail=f"Format non supporté : {file.content_type}.")

    file_bytes = await file.read()
    original_img, input_tensor = preprocess_image(file_bytes)

    start = time.time()
    heatmap, pred_idx, confidence, probs = gradcam_instance.generate(input_tensor)
    inference_time = round((time.time() - start) * 1000, 2)

    pred_class = IDX_TO_CLASS[pred_idx]
    is_fake = pred_class != "real"

    overlay_b64  = generate_overlay_gradcampp(original_img, heatmap)
    heatmap_b64  = generate_heatmap_b64(heatmap)
    original_b64 = image_to_b64(original_img.resize((224, 224)))

    return {
        "prediction": pred_class,
        "is_fake": is_fake,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            cat: round(float(prob) * 100, 2)
            for cat, prob in zip(CATEGORIES, probs)
        },
        "inference_time_ms": inference_time,
        "images": {
            "original": original_b64,
            "heatmap": heatmap_b64,
            "overlay": overlay_b64
        },
        "filename": file.filename
    }