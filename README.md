# 🔍 Deepfake Detection System
### Système de Détection de Deepfakes par Intelligence Artificielle

---

## 📋 Description

Système complet de détection et localisation de deepfakes basé sur le Deep Learning. Le système analyse des images faciales et détermine si elles sont authentiques ou manipulées, en identifiant précisément le type de manipulation parmi 5 catégories.

---

## 🎯 Résultats

| Modèle | Test Accuracy | F1 Macro |
|--------|-------------|---------|
| 🥇 **ResNet18** | **95.79%** | **0.96** |
| EfficientNet-B0 | 86.17% | 0.86 |
| MobileNetV3-Large | 76.83% | 0.76 |

---

## 🏗️ Architecture du Projet

```
deepfake-detection-system/
├── api/                          # Backend FastAPI
│   ├── main.py                   # API endpoints
│   ├── gradcam_plus_plus.py      # GradCAM++ implementation
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                     # Interface React
│   ├── src/
│   │   └── App.tsx               # Interface DeepGuard
│   ├── Dockerfile
│   └── nginx.conf
├── src/
│   ├── data/
│   │   └── preprocessing.py      # Pipeline preprocessing
│   └── models/
│       ├── dataset.py            # DataLoader PyTorch
│       ├── train_efficientnet.py # Entraînement EfficientNet-B0
│       ├── train_resnet18.py     # Entraînement ResNet18
│       ├── train_mobilenetv3.py  # Entraînement MobileNetV3
│       └── train_resnet18_final.py # Modèle final optimisé
├── FaceForensics/
│   ├── download.py               # Téléchargement dataset
│   └── extract_frames.py         # Extraction frames vidéos
├── notebooks/
│   └── 01_EDA_FaceForensics.ipynb # Analyse exploratoire
├── results/
│   └── gradcam/                  # Visualisations GradCAM++
├── docker-compose.yml
└── README.md
```

---

## 🔬 Dataset

**FaceForensics++** (compression c23) — Université Technique de Munich

| Paramètre | Valeur |
|-----------|--------|
| Total images | 9,041 |
| Train | 6,327 (70%) |
| Validation | 1,808 (20%) |
| Test | 906 (10%) |
| Classes | 5 |
| Résolution | 224×224 pixels |

### 5 Classes

| Classe | Description |
|--------|-------------|
| ✅ **Real** | Visage authentique non manipulé |
| ⚠️ **Deepfakes** | Synthèse faciale complète par auto-encodeur |
| ⚠️ **Face2Face** | Transfert d'expressions d'un visage à un autre |
| ⚠️ **FaceSwap** | Remplacement complet du visage |
| ⚠️ **NeuralTextures** | Modification subtile des textures de peau |

---

## 🚀 Installation et Lancement

### Prérequis
- Python 3.10+
- Node.js 20+
- Docker Desktop

### Option 1 — Docker (Recommandé)

```bash
# Cloner le repo
git clone https://github.com/cyrinesam20/deepfake-detection-system.git
cd deepfake-detection-system

# Lancer tous les services
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Interface Web | http://localhost:3000 |
| API Swagger | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

---

### Option 2 — Local

**1. Créer l'environnement virtuel**
```bash
python -m venv venv1
venv1\Scripts\activate
pip install -r api/requirements.txt
```

**2. Télécharger le dataset**
```bash
cd FaceForensics
python download.py ./dataset_sample -d original -c c23 -t videos -n 50 --server EU2
python download.py ./dataset_sample -d Deepfakes -c c23 -t videos -n 100 --server EU2
python download.py ./dataset_sample -d Face2Face -c c23 -t videos -n 100 --server EU2
python download.py ./dataset_sample -d FaceSwap -c c23 -t videos -n 100 --server EU2
python download.py ./dataset_sample -d NeuralTextures -c c23 -t videos -n 100 --server EU2
```

**3. Extraire les frames**
```bash
python extract_frames.py
```

**4. Preprocessing**
```bash
python src/data/preprocessing.py
```

**5. Entraîner le modèle**
```bash
cd src/models
python train_resnet18_final.py
```

**6. Lancer l'API**
```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**7. Lancer le Frontend**
```bash
cd frontend
npm install
npm run dev
```

**8. Lancer MLflow**
```bash
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

---

## 🧠 Modèle & Techniques

### ResNet18 — Architecture choisie

- **18 couches** avec connexions résiduelles (skip connections)
- **Fine-tuning complet** sur FaceForensics++
- **AdamW optimizer** avec learning rate différentiel par couche
- **CosineAnnealingLR** scheduler
- **Label smoothing 0.1** pour meilleure généralisation
- **Data augmentation agressive** : flip, rotation, ColorJitter, GaussianBlur, perspective

### GradCAM++ — Interprétabilité

Technique avancée de visualisation qui localise les zones du visage ayant influencé la décision du modèle. Plus précis que GradCAM standard grâce au calcul des coefficients alpha.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | État de l'API |
| GET | `/model-info` | Informations sur le modèle |
| POST | `/predict` | Classifier une image |
| POST | `/gradcam` | Classifier + visualisation GradCAM++ |

### Exemple de réponse `/predict`

```json
{
  "prediction": "deepfakes",
  "is_fake": true,
  "confidence": 98.5,
  "probabilities": {
    "real": 0.5,
    "deepfakes": 98.5,
    "face2face": 0.7,
    "faceswap": 0.2,
    "neuraltextures": 0.1
  },
  "inference_time_ms": 157.84
}
```

---

## 📊 MLflow Tracking

Les expériences sont trackées avec MLflow :
- Hyperparamètres de chaque run
- Courbes train/val accuracy et loss
- Métriques finales (test accuracy, F1-score)
- Comparaison des 3 modèles

---

## ⚠️ Limitations

- **Domain shift** : Le modèle est entraîné sur FaceForensics++ (vidéos YouTube compressées). Les performances peuvent varier sur des images externes avec des conditions très différentes (selfies, éclairage naturel, accessoires).
- **GANs** : Le système ne détecte pas les visages synthétiques générés par GAN (StyleGAN, DALL-E).
- **Deepfakes modernes** : Les outils de génération post-2022 produisent des fakes plus réalistes que ceux du dataset d'entraînement.

---

## 🛠️ Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Deep Learning | PyTorch 2.3 |
| Modèle | ResNet18 (pré-entraîné ImageNet) |
| Interprétabilité | GradCAM++ |
| Backend API | FastAPI |
| Frontend | React 18 + TypeScript + Tailwind |
| MLOps | MLflow |
| Déploiement | Docker + docker-compose |
| Animations | Framer Motion |

---

## 👩‍💻 Auteur

**Cyrine SAMMOUDA**  
Module Deep Learning - Computer Vision  


---

## 📄 Références

- [FaceForensics++](https://github.com/ondyari/FaceForensics) — Rössler et al., 2019
- [GradCAM++](https://arxiv.org/abs/1710.11063) — Chattopadhyay et al., 2018
- [ResNet](https://arxiv.org/abs/1512.03385) — He et al., 2015
- [EfficientNet](https://arxiv.org/abs/1905.11946) — Tan & Le, 2019