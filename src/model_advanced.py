"""
model_advanced.py
Transfer Learning avec EfficientNetB0 (ou ResNet50V2).
Phase 1 : Feature Extraction (couches gelées)
Phase 2 : Fine-Tuning (dégel progressif)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2


def build_efficientnet(input_shape=(128, 128, 3), num_classes=5,
                       backbone="efficientnetb0"):
    """
    Construit le modèle de Transfer Learning.
    backbone : "efficientnetb0" ou "resnet50v2"

    NOTE : Le générateur de données doit envoyer des pixels [0, 255] (rescale=False)
    car EfficientNetB0 et ResNet50V2 ont leur propre couche de preprocessing intégrée.
    """
    inputs = layers.Input(shape=input_shape)

    # ── Prétraitement backbone ─────────────────────────────────────────────
    if backbone == "efficientnetb0":
        # EfficientNetB0 : preprocessing intégré (pixels [0,255] → normalisation interne)
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
        )
        print("[✓] Backbone : EfficientNetB0 (ImageNet) — preprocessing intégré")
    else:
        # ResNet50V2 : preprocessing explicite requis
        x_preprocess = tf.keras.applications.resnet_v2.preprocess_input(inputs)
        base_model = ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_tensor=x_preprocess,
        )
        print("[✓] Backbone : ResNet50V2 (ImageNet) — preprocessing intégré")

    # ── Phase 1 : Feature Extraction — tout geler ─────────────────────────
    base_model.trainable = False

    # Tête de classification
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs,
                         name=f"TL_{backbone}_TumorStaging")
    return model, base_model


def compile_phase1(model, learning_rate=1e-3, loss_fn=None):
    """Compilation Phase 1 (Feature Extraction)."""
    if loss_fn is None:
        loss_fn = "categorical_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    trainable = sum([tf.size(w).numpy() for w in model.trainable_variables])
    print(f"[Phase 1] Paramètres entraînables : {trainable:,}")
    return model


def unfreeze_for_finetuning(model, base_model, unfreeze_from_layer: int = -30,
                             learning_rate=1e-5, loss_fn=None):
    """
    Phase 2 : Fine-Tuning — dégel des `unfreeze_from_layer` dernières couches.
    """
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from_layer:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False   # BN figé pour stabilité

    if loss_fn is None:
        loss_fn = "categorical_crossentropy"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    trainable = sum([tf.size(w).numpy() for w in model.trainable_variables])
    print(f"[Phase 2] Fine-Tuning — couches dégelées depuis index {unfreeze_from_layer}")
    print(f"          Paramètres entraînables : {trainable:,}")
    return model


if __name__ == "__main__":
    model, base = build_efficientnet()
    model = compile_phase1(model)
    model.summary(line_length=100)

