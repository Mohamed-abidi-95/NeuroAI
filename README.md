# 🧠 Système de Détection et Classification des Stades de Tumeurs
**Deep Learning Medical Imaging — CNN + Transfer Learning + Grad-CAM + FastAPI + Docker**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 📊 **Avancement du projet** : Voir [PROGRESS.md](PROGRESS.md) pour les résultats détaillés

---

## 🏗️ Structure du Projet

```
DeepLearningP1/
├── data/
│   ├── raw/brain-tumor-mri-dataset/   # Dataset brut Kaggle
│   └── staged/                        # Dataset remappé (5 stades)
│       ├── stage_0/  (no_tumor)
│       ├── stage_1/  (meningioma)
│       ├── stage_2/  (pituitary)
│       ├── stage_3/  (glioma 50%)
│       └── stage_4/  (glioma 50%)
├── models/                            # Poids sauvegardés (.keras)
├── reports/                           # Métriques, courbes, Grad-CAM
│   └── gradcam/
├── src/
│   ├── remap_dataset.py               # Remapping 4→5 classes
│   ├── data_pipeline.py               # Prétraitement + augmentation
│   ├── model_baseline.py              # CNN Sequential Keras
│   ├── model_advanced.py              # EfficientNetB0 / ResNet50V2
│   ├── losses.py                      # Focal Loss + Weighted CE
│   ├── train.py                       # Script d'entraînement
│   ├── evaluate.py                    # Métriques cliniques
│   ├── gradcam.py                     # XAI Grad-CAM
│   └── autoencoder.py                 # Détection d'anomalies
├── api/
│   └── main.py                        # API FastAPI async
├── dashboard/
│   └── app.py                         # Dashboard Streamlit
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── notebooks/
│   └── colab_training.ipynb           # Notebook Google Colab
├── requirements.txt
└── README.md
```

---

## 🗄️ Datasets

| Dataset | Lien | Usage |
|---------|------|-------|
| **Brain Tumor MRI** | [kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | Principal (~7023 IRM, ~150MB) |
| **Breast Histopathology** | [kaggle.com/datasets/paultimothymooney/breast-histopathology-images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) | Validation multi-modalité |

### Remapping vers 5 Stades (WHO)

| Dossier Original | Nb Images | Stade | Justification |
|---|---|---|---|
| `no_tumor/` | ~500 | **Stade 0** | Contrôle négatif |
| `meningioma/` | ~937 | **Stade I** | Bénin WHO Grade I |
| `pituitary/` | ~901 | **Stade II** | Extension locale |
| `glioma/` (50% premiers) | ~826 | **Stade III** | WHO Grade III |
| `glioma/` (50% derniers) | ~826 | **Stade IV** | GBM WHO Grade IV |

---

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.10 (TensorFlow 2.15 incompatible avec Python 3.14)
- 16 GB RAM minimum (recommandé : 32 GB)
- Espace disque : ~2 GB (dataset + modèles)

### 1. Installation

**Option A : Environnement virtuel Python 3.10**
```powershell
# Windows
py -3.10 -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
```

**Option B : Utiliser l'environnement existant**
```powershell
# Si venv310 existe déjà
.\venv310\Scripts\activate
```

### 2. Téléchargement du Dataset
```bash
pip install kaggle
# Placez kaggle.json dans C:\Users\<nom>\.kaggle\
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
# Décompressez dans data/Training/ et data/Testing/
```

### 3. Pipeline Complet (Recommandé)
```powershell
# Remapping + Entraînement CNN + Évaluation + Grad-CAM
python run_pipeline.py --epochs 50 --ae-epochs 30
```

**Ou étape par étape :**

### 3a. Remapping (4 classes → 5 stades)
```bash
python src/remap_dataset.py
# Génère data/staged/ avec 7200 images réparties en 5 stades
```

### 3b. Entraînement CNN Baseline
```bash
python src/train.py --loss focal --epochs 50
```

### 3c. Transfer Learning (Recommandé pour Recall > 95%)
```bash
python src/train_advanced.py --phase1-epochs 30 --phase2-epochs 20
```

### 3d. Auto-Encodeur (Détection anomalies)
```bash
python src/autoencoder.py --train --epochs 30
```

### 4. Évaluation
```bash
python src/evaluate.py --model models/cnn_baseline_best.keras
# Génère : matrice confusion, courbes ROC, rapport détaillé
```

### 5. Grad-CAM (Explicabilité XAI)
```bash
python src/gradcam.py --model models/cnn_baseline_best.keras --samples 5
# Génère : reports/gradcam/*.png
```

### 6. API FastAPI
```bash
cd api
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs (Swagger UI)
```

**Test avec cURL :**
```powershell
curl -X POST "http://localhost:8000/predict" -F "file=@test_image.jpg"
```

### 7. Dashboard Streamlit
```bash
streamlit run dashboard/app.py
# → http://localhost:8501
```

### 8. Docker (Production)
```bash
cd docker
docker-compose up --build
# API → http://localhost:8000
# Dashboard → http://localhost:8501
```

---

## 🖥️ Configurations IntelliJ IDEA

Le projet inclut des **Run Configurations** prêtes à l'emploi :

1. **1 - Remap Dataset** : Remapping 4→5 stades
2. **2 - Train CNN Baseline** : Entraînement CNN baseline
3. **3 - Train Advanced** : Transfer Learning
4. **4 - Train AutoEncoder** : Auto-encodeur
5. **5 - Evaluate Model** : Évaluation complète
6. **6 - GradCAM** : Génération Grad-CAM
7. **7 - Full Pipeline** : Pipeline complet
8. **8 - API FastAPI** : Démarrage API

**Utilisation** : Clic droit → Run '2 - Train CNN Baseline'

> ⚠️ **Important** : Configurez l'interpréteur Python sur `venv310/Scripts/python.exe`

---

## 📊 Architecture CNN Baseline

| Couche | Paramètres | Activation |
|--------|-----------|------------|
| Input | (128, 128, 3) | — |
| Conv2D | 32 filtres, 3×3 | ReLU |
| MaxPooling2D | (2, 2) | — |
| Conv2D | 64 filtres, 3×3 | ReLU |
| MaxPooling2D | (2, 2) | — |
| Conv2D | 128 filtres, 3×3 | ReLU |
| MaxPooling2D | (2, 2) | — |
| Flatten + Dense | 128 neurones | ReLU |
| Dense (sortie) | 5 neurones | Softmax |

**Compilation :** Adam + Focal Loss (γ=2) + métriques : Accuracy, Recall, AUC

---

## 🎯 Résultats Finaux (11 Mai 2026)

### Modèle Final — TL EfficientNetB0 4 Classes

| Métrique | Valeur | Cible | Statut |
|---|---|---|---|
| **Val Recall (seuil 0.70)** | **97.8%** | **> 95%** | ✅ **CIBLE ATTEINTE** |
| Val Accuracy | 92.4% | > 80% | ✅ Excellent |
| Val AUC | 98.7% | > 90% | ✅ Excellent |
| Val Precision | 93.4% | — | ✅ Très bon |

### Historique des Modèles

| Modèle | Recall | Accuracy | AUC |
|---|---|---|---|
| CNN Baseline 5 classes | 62.3% | 71.5% | 93.4% |
| TL EfficientNetB0 5 classes | 74.0% | 82.0% | 95.9% |
| CNN 4 classes | 72.0% | 77.3% | 92.3% |
| **TL EfficientNetB0 4 classes** | **97.8%*** | **92.4%** | **98.7%** |

*Seuil de décision optimisé à 0.70

📊 **Détails complets** : [PROGRESS.md](PROGRESS.md) | [SUMMARY.md](SUMMARY.md) | [QUESTIONS_PROF.md](QUESTIONS_PROF.md)

---

## 🎯 Métriques Cibles (Cahier des Charges)

- ✅ **Sensibilité (Recall)** > 95% (minimiser faux négatifs — risques vitaux)
- ✅ **F2-Score** (privilégie le Recall sur la Précision)
- ✅ **ROC-AUC** par stade (discrimination par classe pathologique)
- ✅ **Inférence** < 200ms par image

---

## 🔗 API Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/` | Santé de l'API |
| POST | `/predict` | Prédiction stade + Grad-CAM |
| GET | `/model/info` | Infos modèle chargé |

**Réponse `/predict` :**
```json
{
  "stage_id": 3,
  "stage_label": "Stade III — Anaplasique",
  "clinical_note": "Atteinte des ganglions...",
  "confidence": 0.8734,
  "probabilities": [0.02, 0.03, 0.05, 0.87, 0.03],
  "gradcam_base64": "...",
  "anomaly_score": 0.0023,
  "requires_review": false,
  "review_reason": "Aucune"
}
```

---

## 📁 Fichiers Importants

- **[PROGRESS.md](PROGRESS.md)** : Suivi détaillé de l'avancement et résultats d'entraînement
- **[plan_projet.tex](plan_projet.tex)** : Plan du projet (LaTeX)
- **[requirements.txt](requirements.txt)** : Dépendances Python
- **[run_pipeline.py](run_pipeline.py)** : Script maître (pipeline complet)

---

**Bonne chance — Inch'Allah le projet sera terminé dans les délais ! 🚀**

