"""
data_pipeline.py
Pipeline de prétraitement et d'augmentation — conforme au cahier des charges.

Spec :
  - Redimensionnement 128×128
  - Normalisation [0, 1]
  - Augmentation : rotations, flips horizontal/vertical, zoom
  - Split 70% train / 15% val / 15% test
  - Encodage catégoriel (5 classes : stage_0 à stage_4)
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join("data", "staged")
IMG_SIZE    = (128, 128)
BATCH_SIZE  = 32
NUM_CLASSES = 5
SEED        = 42


def get_generators(rescale=True):
    """
    Retourne (train_gen, val_gen) via ImageDataGenerator.
    Augmentation conforme au cahier des charges :
      - rotations aléatoires (SPEC)
      - flips horizontal et vertical (SPEC)
      - variations de zoom (SPEC)
      + luminosité, décalages (bonus pour robustesse IRM)

    Args:
        rescale (bool): True pour CNN baseline [0,1].
                        False pour Transfer Learning (EfficientNet a son propre preprocessing).
    """
    scale_factor = 1.0 / 255 if rescale else None
    train_datagen = ImageDataGenerator(
        # ── SPEC : normalisation [0, 1] pour CNN baseline ──────────────────
        rescale=scale_factor,

        # ── SPEC : rotations aléatoires ─────────────────────────────────────
        rotation_range=25,

        # ── SPEC : flips horizontal/vertical ────────────────────────────────
        horizontal_flip=True,
        vertical_flip=True,

        # ── SPEC : variations de zoom ───────────────────────────────────────
        zoom_range=0.15,

        # ── BONUS : robustesse supplémentaire ───────────────────────────────
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        brightness_range=[0.85, 1.15],   # simule variations capteur IRM
        fill_mode="reflect",

        # ── Split 80% train / 20% val (→ 70/15/15 avec test séparé) ─────────
        validation_split=0.2,
    )

    # Validation : normalisation uniquement (PAS d'augmentation)
    val_datagen = ImageDataGenerator(
        rescale=scale_factor,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False,
    )

    print(f"[✓] Classes trouvées : {train_gen.class_indices}")
    print(f"    Train   : {train_gen.samples} images (avec augmentation)")
    print(f"    Val     : {val_gen.samples} images")

    return train_gen, val_gen


def get_test_generator(test_dir=None):
    """Générateur de test sans augmentation."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    src = test_dir if test_dir else DATA_DIR
    test_gen = test_datagen.flow_from_directory(
        src,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    print(f"    Test    : {test_gen.samples} images")
    return test_gen


def load_dataset_as_arrays(rescale=True):
    """
    Charge l'intégralité du dataset en arrays NumPy.
    Split 70% / 15% / 15% (conforme cahier des charges).
    Utile pour l'auto-encodeur et l'évaluation fine.

    Args:
        rescale (bool): True → pixels [0,1] (CNN baseline).
                        False → pixels [0,255] (Transfer Learning).
    """
    from PIL import Image
    from tensorflow.keras.utils import to_categorical

    X, y = [], []
    class_names = sorted(os.listdir(DATA_DIR))

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize(IMG_SIZE)
                arr = np.array(img, dtype=np.float32)
                if rescale:
                    arr = arr / 255.0
                X.append(arr)
                y.append(label)
            except Exception:
                pass

    X = np.array(X)
    y = np.array(y)

    # Split 70 / 15 / 15 (SPEC)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_val   = to_categorical(y_val,   NUM_CLASSES)
    y_test  = to_categorical(y_test,  NUM_CLASSES)

    print(f"[OK] Split 70/15/15 -- Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train_raw):
    """Calcule les poids de classe pour compenser le déséquilibre."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_raw)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_raw)
    return dict(zip(classes.astype(int), weights))


if __name__ == "__main__":
    train_gen, val_gen = get_generators()
    print("[✓] Pipeline prêt.")
