# Livrable 2 : Exploration du Dataset (EDA)

**Durée :** ~2 heures

---

## 📊 Dataset : FaceForensics++

### Statistiques Globales

| Métrique              | Valeur        |
|-----------------------|---------------|
| **Total images**      | 10,900 images |
| **Catégories**        | 5             |
| **Moyenne/catégorie** | 2,180 images  |

### Distribution par Catégorie

| Catégorie      | Nombre | Pourcentage |
|----------------|--------|-------------|
| Real           | 1979   |    18.2%    |
| DeepFakes      | 2510   |    23.0%    |
| Face2Face      | 2509   |    23.0%    |
| FaceSwap       | 1951   |    17.9%    |
| NeuralTextures | 1951   |    17.9%    |

---

## ✅ Observations Clés

1. **Dataset équilibré** : Toutes catégories ≈  2,180 images
2. **Résolutions variables** : ~800-1200 pixels
3. **Qualité satisfaisante** : Aucune image corrompue
4. **Prêt pour preprocessing**

---

## 📈 Visualisations Générées

- `results/distribution_categories.png` : Bar chart de distribution
- `results/samples_grid.png` : Grille 6×5 d'échantillons
- `results/dataset_summary.csv` : Résumé tabulaire

---

## 🔧 Recommandations Techniques

### Preprocessing Nécessaire
1. Détection de visages (MTCNN)
2. Resize : 224×224 px
3. Normalisation ImageNet

### Split Proposé
- Train : 70% (7,629 images)
- Val : 20% (2,180 images)
- Test : 10% (1,090 images)

---

## 💾 Fichiers Créés
```
notebooks/01_EDA_FaceForensics.ipynb
results/distribution_categories.png
results/samples_grid.png
results/dataset_summary.csv
Livrables/Livrable_02_EDA.md
```

---

**Statut :** ✅ TERMINÉ