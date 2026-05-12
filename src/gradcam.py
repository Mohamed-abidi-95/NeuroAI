"""
gradcam.py
Grad-CAM : Gradient-weighted Class Activation Mapping.
Génère des heatmaps XAI superposées aux images médicales.
Compatibles avec le CNN baseline et les modèles de Transfer Learning.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image

STAGE_NAMES_4 = [
    "Stage 0 - No Tumor",
    "Stage I - Meningioma",
    "Stage II - Pituitary",
    "Stage III/IV - Glioma",
]
STAGE_NAMES_5 = [
    "Stage 0 - No Tumor",
    "Stage I - Meningioma",
    "Stage II - Pituitary",
    "Stage III - Anaplastic",
    "Stage IV - GBM",
]

def get_stage_names(num_classes):
    return STAGE_NAMES_4 if num_classes <= 4 else STAGE_NAMES_5

# backward compat
STAGE_NAMES = STAGE_NAMES_5
GRADCAM_DIR = "reports/gradcam"
os.makedirs(GRADCAM_DIR, exist_ok=True)


def find_last_conv_layer(model) -> str:
    """
    Trouve automatiquement le nom de la derniere couche Conv2D.
    Cherche recursivement dans les sous-modeles (ex : EfficientNetB0).
    """
    # D'abord chercher dans les couches directes du modele (tete + backbone)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # Sous-modele (ex : backbone EfficientNetB0)
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    raise ValueError("No Conv2D layer found in model.")


def compute_gradcam(model, img_array: np.ndarray,
                    class_idx: int = None,
                    conv_layer_name: str = None) -> np.ndarray:
    """
    Calcule la heatmap Grad-CAM pour une image.

    Args:
        model       : modèle Keras chargé
        img_array   : image normalisée (128, 128, 3) ou (1, 128, 128, 3)
        class_idx   : classe cible (None = classe prédite)
        conv_layer_name : couche Conv2D cible (None = dernière auto-détectée)

    Returns:
        heatmap (ndarray) : heatmap float32 dans [0, 1], shape (H, W)
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    if conv_layer_name is None:
        conv_layer_name = find_last_conv_layer(model)

    # Modele intermediaire : input -> [activations conv, predictions]
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    # Gradient de la perte par rapport aux activations conv
    grads = tape.gradient(loss, conv_outputs)

    # Pondération globale (Global Average Pooling des gradients)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Combinaison linéaire
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), class_idx


def overlay_gradcam(img_array: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """
    Superpose la heatmap colorisée (jet) sur l'image originale.

    Returns:
        image superposée (uint8) shape (128, 128, 3)
    """
    # Resize heatmap à la taille de l'image
    h, w = img_array.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Colorisation
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # RGB
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Image originale en uint8
    if img_array.max() <= 1.0:
        img_uint8 = (img_array * 255).astype(np.uint8)
    else:
        img_uint8 = img_array.astype(np.uint8)

    # Superposition
    superimposed = cv2.addWeighted(img_uint8, 1 - alpha,
                                   heatmap_colored, alpha, 0)
    return superimposed


def visualize_gradcam(model, img_array: np.ndarray,
                      true_label: int = None,
                      save_name: str = "gradcam_output",
                      conv_layer_name: str = None):
    """
    Genere et sauvegarde la visualisation Grad-CAM complete :
    [Image originale | Heatmap | Superposition].
    """
    if img_array.ndim == 3:
        img_input = np.expand_dims(img_array, axis=0)
    else:
        img_input = img_array

    heatmap, pred_class = compute_gradcam(
        model, img_input, conv_layer_name=conv_layer_name)
    superimposed = overlay_gradcam(img_array, heatmap)

    confidence = float(model.predict(img_input, verbose=0)[0][pred_class])
    num_classes = int(model.output_shape[-1])
    snames = get_stage_names(num_classes)

    pred_lbl = snames[pred_class] if pred_class < len(snames) else str(pred_class)
    true_lbl = snames[true_label] if (true_label is not None and
                                       true_label < len(snames)) else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title = "Grad-CAM - Predicted: {} ({:.1%})".format(pred_lbl, confidence)
    if true_lbl:
        title += "\nTrue: {}".format(true_lbl)
    fig.suptitle(title, fontsize=12)

    axes[0].imshow(img_array if img_array.max() > 1 else img_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    path = os.path.join(GRADCAM_DIR, "{}.png".format(save_name))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Grad-CAM saved : {}".format(path))
    return path, pred_class, confidence, superimposed


def gradcam_batch(model, X_samples: np.ndarray, y_true: np.ndarray,
                  n_samples: int = 10, prefix: str = "batch"):
    """Applique Grad-CAM sur un echantillon aleatoire du jeu de test."""
    indices = np.random.choice(len(X_samples), size=min(n_samples, len(X_samples)),
                               replace=False)
    for i, idx in enumerate(indices):
        visualize_gradcam(
            model=model,
            img_array=X_samples[idx],
            true_label=int(y_true[idx]),
            save_name="{}_sample_{}_stage{}".format(prefix, i, int(y_true[idx])),
        )


if __name__ == "__main__":
    import argparse
    from data_pipeline import load_dataset_as_arrays
    from losses import FocalLoss

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="Chemin modèle .keras")
    parser.add_argument("--samples", type=int, default=5, help="Nb images à visualiser")
    args = parser.parse_args()

    model = tf.keras.models.load_model(
        args.model, custom_objects={"FocalLoss": FocalLoss})

    _, _, X_test, _, _, y_test = load_dataset_as_arrays()
    y_test_raw = np.argmax(y_test, axis=1)

    gradcam_batch(model, X_test, y_test_raw, n_samples=args.samples)
    print("[OK] Grad-CAM done.")

