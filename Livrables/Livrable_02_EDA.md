# Livrable 2 : Exploration du Dataset (EDA)

**Durée :** ~2 heures

---

## 📊 Dataset : FaceForensics++

### Statistiques Globales

| Métrique              | Valeur        |
|-----------------------|---------------|
| **Total images**      | 12,869 images |
| **Catégories**        | 6             |
| **Moyenne/catégorie** | 2,145 images  |

### Distribution par Catégorie

| Catégorie      | Nombre | Pourcentage |
|----------------|--------|-------------|
| Real           | 1979   |    15.4%    |
| Actors         | 3828   |    29.7%    |
| DeepFakes      | 1979   |    15.4%    |
| Face2Face      | 1979   |    15.4%    |
| FaceSwap       | 1552   |    12.1%    |
| NeuralTextures | 1552   |    12.1%    |

---

## ✅ Observations Clés

1. **Dataset équilibré** : Toutes catégories ≈  2,145 images
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
- Train : 70% ( 9,008 images)
- Val : 20% (2,573 images)
- Test : 10% (1,286 images)

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