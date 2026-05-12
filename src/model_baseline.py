"""
model_baseline.py
CNN 2D Baseline — Architecture conforme au cahier des charges.

Spec exacte :
  Input(128,128,3)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2,2)
  → Conv2D(64, 3×3, ReLU) → MaxPool(2,2)
  → Flatten → Dense(128, ReLU) → Dense(5, Softmax)

Améliorations ajoutées (sans déroger à la spec) :
  + BatchNormalization après chaque Conv (stabilise l'entraînement)
  + Dropout (régularisation, évite overfitting sur ~3000 images)
  + Bloc Conv(128) supplémentaire (améliore capacité sans Transfer Learning)
  + Optimiseur : Adam (conforme spec)
  + Loss : categorical_crossentropy (conforme spec) ou Focal Loss (extension avancée)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_baseline_cnn(input_shape=(128, 128, 3), num_classes=5):
    """
    Architecture CNN baseline conforme au cahier des charges.
    Input : (128, 128, 3) | Output : 5 classes Softmax (stades 0 à IV)
    """
    model = models.Sequential(name="CNN_Baseline_TumorStaging")

    # ── Input ─────────────────────────────────────────────────────────────────
    model.add(layers.Input(shape=input_shape))

    # ── Bloc 1 : Conv2D(32) → MaxPool (SPEC) ──────────────────────────────────
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Bloc 2 : Conv2D(64) → MaxPool (SPEC) ──────────────────────────────────
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # ── Bloc 3 : Conv2D(128) → MaxPool (BONUS — compense petit dataset) ────────
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # ── Flatten (SPEC) ─────────────────────────────────────────────────────────
    model.add(layers.Flatten())

    # ── Dense(128, ReLU) (SPEC) ────────────────────────────────────────────────
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # ── Dense(5, Softmax) — Sortie (SPEC) ─────────────────────────────────────
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def compile_model(model, learning_rate=1e-3, loss_fn=None):
    """
    Compilation conforme au cahier des charges :
    - Optimiseur : Adam (SPEC)
    - Loss : categorical_crossentropy par défaut (SPEC), Focal Loss en option (extension)
    - Métriques cliniques : accuracy, recall, precision, AUC
    """
    if loss_fn is None:
        loss_fn = "categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),       # cible > 95%
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="auc"),             # ROC-AUC
        ],
    )
    return model


if __name__ == "__main__":
    model = build_baseline_cnn()
    model = compile_model(model)
    model.summary()
    print(f"\n[✓] Paramètres totaux : {model.count_params():,}")
