# 📋 Livrable 01 - Projet Deep Learning

---

## 📌 Informations du Projet

|     **Champ**     |                               **Détail**                        |
|-------------------|-----------------------------------------------------------------|
| **Nom du projet** | Système de Détection de Deepfakes par Intelligence Artificielle |
|  **Étudiant(e)**  | Cyrine SAMMOUDA                                                 |
|     **Module**    | Deep Learning - Computer Vision & Modèles Génératifs            |
|   **Enseignant**  | Haythem Ghazouani                                               |
|      **Date**     | 30/01/2026                                                      |

---

## 🎯 Sujet du Projet

### Thématique Choisie
**Détection de Deepfakes (Classification + Localisation)**

### Description
Développer un système intelligent capable de :
- **Classifier** les images en authentiques ou falsifiées (deepfakes)
- **Localiser** visuellement les zones manipulées dans l'image
- **Déployer** le système avec une API REST et une interface web

### Type de Tâches
- Classification multi-classe (5 classes)
- Détection/Localisation de zones manipulées
- Interprétabilité des résultats (heatmap)

---

## 📊 Dataset

### Dataset Principal
**FaceForensics++**

| **Caractéristique** |  **Détail**                               |
|---------------------|-------------------------------------------|
| **Source**          | Université Technique de Munich (TUM)      |
| **Type**            | Images de visages (frames vidéo extraits) |
| **Taille**          | ~10,000 images (extraction à 1 fps)       |
| **Résolution**      | 1920×1080 (Full HD)                       |
| **Format**          | PNG / JPG                                 |
| **Accès**           | Gratuit (formulaire académique)           |

### Classes du Dataset

| **Classe**     | **Description**                 | **Nombre d'images** |
|----------------|---------------------------------|---------------------|
| Real           | Images authentiques             | ~2,500              |
| DeepFakes      | Manipulation par DeepFakes      | ~2,500              |
| Face2Face      | Manipulation par Face2Face      | ~2,500              |
| FaceSwap       | Manipulation par FaceSwap       | ~2,500              |
| NeuralTextures | Manipulation par NeuralTextures | ~2,500              |
| **Total**      |                                 | **~10,000**         |


## 💻 Technologies Utilisées

### Deep Learning & Computer Vision

| **Composant**    | **Technologie**                         | **Version** |
|------------------|-----------------------------------------|-------------|
| Framework DL     | PyTorch                                 | 2.0+        |
| Computer Vision  | torchvision, OpenCV, Pillow             | Latest      |
| Modèle           | EfficientNet-B0 (pré-entraîné ImageNet) | -           |
| Interprétabilité | pytorch-grad-cam (Grad-CAM)             | Latest      |

### Backend & API

| **Composant** | **Technologie** | **Version** |
|---------------|-----------------|-------------|
| API Framework | FastAPI | 0.104+ |
| Serveur ASGI | Uvicorn | Latest |
| Validation | Pydantic | Latest |

### Frontend

| **Composant** | **Technologie** | **Version** |
|---------------|-----------------|-------------|
| Framework UI  | React           | 18+         |
| Langage       | TypeScript      | 5.0+        |
| Styling       | CSS             | Latest      |
| HTTP Client   | Axios           | Latest      |
| Routing       | React Router    | 6+          |

### MLOps & Déploiement

| **Composant**        | **Technologie** | **Version** |
|----------------------|-----------------|-------------|
| Tracking Expériences | MLflow          | 2.8+        |
| Conteneurisation     | Docker          | Latest      |
| Orchestration        | docker-compose  | Latest      |
| Tests                | pytest          | Latest      |
| Version Control      | Git + GitHub    | -           |

### Stack Technique Résumée
```
Backend:  PyTorch + FastAPI + MLflow
Frontend: React + TypeScript + CSS
Deploy:   Docker + docker-compose
```
---

## 🔗 Ressources

### Dataset
- **FaceForensics++ GitHub** : https://github.com/ondyari/FaceForensics
