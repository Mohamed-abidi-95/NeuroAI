"""
train.py
Script d'entraînement principal — conforme au cahier des charges.

Usage :
  python src/train.py               # Focal Loss (recommandé)
  python src/train.py --loss ce     # categorical_crossentropy (spec de base)
  python src/train.py --epochs 70
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from data_pipeline import get_generators, get_class_weights
from model_baseline import build_baseline_cnn, compile_model
from losses import FocalLoss, compute_focal_alpha

# ─── Config ──────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


def get_callbacks(patience=12):
    """Callbacks pour un entraînement robuste."""
    return [
        # Sauvegarde le meilleur modèle (val_accuracy)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "cnn_baseline_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # Arrêt précoce si pas d'amélioration
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Réduction du learning rate si stagnation
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        # TensorBoard pour visualisation
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", "cnn_baseline"),
            histogram_freq=1,
        ),
    ]


def plot_history(history):
    """Sauvegarde les courbes loss/accuracy/recall en PNG."""
    metrics_to_plot = [
        ("loss",     "val_loss",     "Loss"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("recall",   "val_recall",   "Recall (Sensibilité)"),
        ("auc",      "val_auc",      "ROC-AUC"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Entraînement CNN Baseline — Détection Stades Tumoraux", fontsize=13)

    for ax, (train_k, val_k, title) in zip(axes, metrics_to_plot):
        if train_k in history.history:
            ax.plot(history.history[train_k], label="Train", color="royalblue")
        if val_k in history.history:
            ax.plot(history.history[val_k],   label="Val",   color="tomato")
        ax.set_title(title)
        ax.set_xlabel("Époque")
        ax.legend()
        ax.grid(True, alpha=0.3)

    path = os.path.join(MODELS_DIR, "cnn_baseline_history.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Courbes sauvegardées : {path}")


def main():
    parser = argparse.ArgumentParser(description="Entraînement CNN — Détection Tumeurs")
    parser.add_argument("--loss",   choices=["ce", "focal"], default="focal",
                        help="ce = categorical_crossentropy (spec), focal = Focal Loss (extension)")
    parser.add_argument("--epochs", type=int, default=70,
                        help="Nombre d'époques (default: 70)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ENTRAÎNEMENT — CNN BASELINE (5 stades tumoraux)")
    print(f"  Loss : {'Focal Loss' if args.loss == 'focal' else 'Categorical CrossEntropy'}")
    print(f"  Epochs : {args.epochs}")
    print("=" * 60)

    # ── Chargement des données ─────────────────────────────────────────────
    train_gen, val_gen = get_generators()
    labels = train_gen.classes
    class_weights = get_class_weights(labels)
    print(f"\n[✓] Poids des classes : {class_weights}")

    # ── Choix de la fonction de perte ─────────────────────────────────────
    if args.loss == "focal":
        # Extension avancée : Focal Loss (gestion déséquilibre classes)
        counts = {i: int(np.sum(labels == i)) for i in range(5)}
        alpha  = compute_focal_alpha(counts)
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        print(f"[✓] Focal Loss | gamma=2.0 | alpha={[round(a, 3) for a in alpha]}")
    else:
        # Spec baseline : categorical_crossentropy
        loss_fn = None
        print("[✓] Loss : categorical_crossentropy (spec baseline)")

    # ── Construction et compilation du modèle ─────────────────────────────
    model = build_baseline_cnn()
    model = compile_model(model, learning_rate=1e-3, loss_fn=loss_fn)
    model.summary()

    # ── Entraînement ──────────────────────────────────────────────────────
    print(f"\n[•] Début entraînement ({args.epochs} epochs max) ...\n")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,   # compense déséquilibre
        callbacks=get_callbacks(),
        verbose=1,
    )

    # ── Sauvegarde finale ─────────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, "cnn_baseline_final.keras")
    model.save(final_path)
    print(f"\n[✓] Modèle final sauvegardé : {final_path}")

    # ── Rapport rapide ────────────────────────────────────────────────────
    best_val_acc    = max(history.history.get("val_accuracy", [0]))
    best_val_recall = max(history.history.get("val_recall",   [0]))
    best_val_auc    = max(history.history.get("val_auc",      [0]))

    print(f"\n{'─'*50}")
    print(f"  Meilleure Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"  Meilleure Val Recall   : {best_val_recall:.4f} ({best_val_recall*100:.1f}%)")
    if best_val_recall >= 0.95:
        print("  ✅ Cible Recall > 95% ATTEINTE !")
    else:
        print(f"  ⚠️  Cible Recall > 95% non atteinte (actuel : {best_val_recall*100:.1f}%)")
    print(f"  Meilleure Val AUC      : {best_val_auc:.4f}")
    print(f"{'─'*50}")

    plot_history(history)

    print("\n[✓] Entraînement terminé. Prochaine étape :")
    print(f"    python src/evaluate.py --model {final_path}")


if __name__ == "__main__":
    main()

