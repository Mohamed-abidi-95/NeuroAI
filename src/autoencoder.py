"""
autoencoder.py
Auto-encodeur convolutif pour la détection d'anomalies.
Entraîné uniquement sur les images normales (stage_0).
Une erreur de reconstruction élevée signale un cas "Suspect/Inconnu".
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score

MODELS_DIR  = "models"
REPORTS_DIR = "reports"
IMG_SIZE    = (128, 128)
LATENT_DIM  = 256
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_autoencoder(input_shape=(128, 128, 3)):
    """
    Auto-encodeur convolutif symétrique.
    Encoder : Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(latent)
    Decoder : Dense → ConvT(128) → ConvT(64) → ConvT(32) → Conv2D(3)
    """
    # ── Encoder ───────────────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same", strides=2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)

    # Forme avant flatten pour le décodeur
    shape_before_flatten = x.shape[1:]
    x = layers.Flatten()(x)
    encoded = layers.Dense(LATENT_DIM, activation="relu", name="latent")(x)

    encoder = models.Model(inputs, encoded, name="Encoder")

    # ── Decoder ───────────────────────────────────────────────────────────
    latent_inputs = layers.Input(shape=(LATENT_DIM,), name="decoder_input")

    x = layers.Dense(
        shape_before_flatten[0] * shape_before_flatten[1] * shape_before_flatten[2],
        activation="relu"
    )(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same", strides=2)(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same",
                            name="reconstruction")(x)

    decoder = models.Model(latent_inputs, decoded, name="Decoder")

    # ── Auto-encodeur complet ─────────────────────────────────────────────
    autoencoder_output = decoder(encoder(inputs))
    autoencoder = models.Model(inputs, autoencoder_output, name="AutoEncoder_TumorAnomalyDetector")

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return autoencoder, encoder, decoder


def compute_reconstruction_error(autoencoder, X: np.ndarray) -> np.ndarray:
    """Calcule l'erreur MSE pixel-wise pour chaque image."""
    X_recon = autoencoder.predict(X, verbose=0)
    mse = np.mean((X - X_recon) ** 2, axis=(1, 2, 3))
    return mse, X_recon


def find_anomaly_threshold(mse_normal: np.ndarray, percentile: float = 95) -> float:
    """
    Seuil d'anomalie = percentile des erreurs sur données saines.
    Par défaut : 95e percentile.
    """
    threshold = float(np.percentile(mse_normal, percentile))
    print("[OK] Seuil d'anomalie (p{:.0f}) = {:.6f}".format(percentile, threshold))
    return threshold


def detect_anomalies(mse: np.ndarray, threshold: float):
    """Retourne un masque booléen : True = anomalie."""
    return mse > threshold


def train_autoencoder(X_normal: np.ndarray, epochs: int = 50, batch_size: int = 32):
    """
    Entraîne l'auto-encodeur uniquement sur les images saines (stage_0).
    """
    print("\n" + "=" * 60)
    print("  ENTRAÎNEMENT — AUTO-ENCODEUR (Détection d'Anomalies)")
    print("=" * 60)
    print(f"  Images saines pour l'entraînement : {len(X_normal)}")

    autoencoder, encoder, decoder = build_autoencoder()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "autoencoder_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1),
    ]

    history = autoencoder.fit(
        X_normal, X_normal,          # Input = Target (reconstruction)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    autoencoder.save(os.path.join(MODELS_DIR, "autoencoder_final.keras"))
    encoder.save(os.path.join(MODELS_DIR, "encoder_final.keras"))
    print("[OK] Auto-encodeur sauvegarde.")

    # Courbe de loss
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train Loss", color="royalblue")
    plt.plot(history.history["val_loss"], label="Val Loss",   color="tomato")
    plt.title("Auto-Encodeur — Loss de Reconstruction (MSE)")
    plt.xlabel("Époque")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "autoencoder_history.png"), dpi=150)
    plt.close()

    return autoencoder, encoder, decoder


def visualize_reconstruction(autoencoder, X_samples: np.ndarray,
                              mse: np.ndarray, threshold: float,
                              n: int = 8):
    """Visualise les reconstructions et les erreurs."""
    indices = np.random.choice(len(X_samples), size=min(n, len(X_samples)), replace=False)
    fig, axes = plt.subplots(3, len(indices), figsize=(2.5 * len(indices), 8))
    fig.suptitle("Auto-Encodeur : Originals | Reconstructions | Erreurs", fontsize=12)

    X_recon = autoencoder.predict(X_samples[indices], verbose=0)

    for col, idx in enumerate(range(len(indices))):
        orig   = X_samples[indices[idx]]
        recon  = X_recon[idx]
        err_map = np.abs(orig - recon).mean(axis=-1)
        color = "red" if mse[indices[idx]] > threshold else "green"

        axes[0, col].imshow(orig)
        axes[0, col].set_title(f"Orig.", fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(np.clip(recon, 0, 1))
        axes[1, col].set_title(f"Recon.", fontsize=9)
        axes[1, col].axis("off")

        axes[2, col].imshow(err_map, cmap="hot")
        axes[2, col].set_title(
            f"MSE={mse[indices[idx]]:.4f}", fontsize=8, color=color)
        axes[2, col].axis("off")

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "autoencoder_reconstructions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Visualisation reconstruction : {}".format(path))


if __name__ == "__main__":
    import argparse
    from data_pipeline import load_dataset_as_arrays

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  action="store_true", help="Entraîner l'auto-encodeur")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset_as_arrays()
    y_train_raw = np.argmax(y_train, axis=1)
    y_test_raw  = np.argmax(y_test, axis=1)

    # Images normales uniquement (stage_0)
    X_normal = X_train[y_train_raw == 0]
    print(f"Images saines (stage_0) : {len(X_normal)}")

    if args.train:
        ae, enc, dec = train_autoencoder(X_normal, epochs=args.epochs)
    else:
        from losses import FocalLoss
        ae = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "autoencoder_final.keras"),
            custom_objects={"FocalLoss": FocalLoss}
        )

    mse_normal, _ = compute_reconstruction_error(ae, X_normal)
    threshold = find_anomaly_threshold(mse_normal)

    mse_test, _ = compute_reconstruction_error(ae, X_test)
    anomalies = detect_anomalies(mse_test, threshold)
    n_anom = int(np.sum(anomalies))
    print(f"\n[•] Anomalies détectées sur test set : {n_anom}/{len(X_test)} "
          f"({n_anom/len(X_test)*100:.1f}%)")

    visualize_reconstruction(ae, X_test, mse_test, threshold)

