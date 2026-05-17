import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import get_generators, get_class_weights
from model_advanced import build_efficientnet, compile_phase1, unfreeze_for_finetuning
from losses import FocalLoss, compute_focal_alpha

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def get_callbacks_tl(phase: str, patience: int = 10):
                                           
    monitor = "val_accuracy"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, f"tl_{phase}_best.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-8,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join("logs", f"tl_{phase}"),
            histogram_freq=0,
        ),
    ]

def plot_tl_history(h1, h2):
                                                             
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Transfer Learning EfficientNetB0 — Détection Stades Tumoraux", fontsize=13)

    def plot_metric(ax, key, title):
        offset = len(h1.history.get(key, []))
        if key in h1.history:
            ax.plot(h1.history[key],                  color="royalblue",   label="Phase1 Train")
        if f"val_{key}" in h1.history:
            ax.plot(h1.history[f"val_{key}"],         color="cornflowerblue", label="Phase1 Val", linestyle="--")
        x2 = range(offset, offset + len(h2.history.get(key, [])))
        if key in h2.history:
            ax.plot(x2, h2.history[key],              color="tomato",      label="Phase2 Train")
        if f"val_{key}" in h2.history:
            ax.plot(x2, h2.history[f"val_{key}"],     color="salmon",      label="Phase2 Val", linestyle="--")
        if offset > 0:
            ax.axvline(x=offset, color="gray", linestyle=":", alpha=0.7, label="Fine-Tuning →")
        ax.set_title(title)
        ax.set_xlabel("Époque")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plot_metric(axes[0], "loss",     "Loss")
    plot_metric(axes[1], "accuracy", "Accuracy")
    plot_metric(axes[2], "recall",   "Recall (Sensibilité)")

    path = os.path.join(MODELS_DIR, "tl_history.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[✓] Courbes TL sauvegardées : {path}")

def main():
    parser = argparse.ArgumentParser(description="Transfer Learning — Détection Tumeurs")
    parser.add_argument("--backbone",      choices=["efficientnetb0", "resnet50v2"],
                        default="efficientnetb0")
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--loss",          choices=["ce", "focal"], default="focal")
    parser.add_argument("--unfreeze",      type=int, default=-30,
                        help="Nb couches dégelées pour fine-tuning (négatif = dernières N)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TRANSFER LEARNING — EfficientNetB0 / ResNet50V2")
    print(f"  Backbone : {args.backbone}")
    print(f"  Phase 1  : {args.phase1_epochs} epochs (Feature Extraction)")
    print(f"  Phase 2  : {args.phase2_epochs} epochs (Fine-Tuning)")
    print("=" * 60)

    train_gen, val_gen = get_generators(rescale=False)
    labels = train_gen.classes
    class_weights = get_class_weights(labels)

    if args.loss == "focal":
        counts = {i: int(np.sum(labels == i)) for i in range(5)}
        alpha  = compute_focal_alpha(counts)
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        print(f"[✓] Focal Loss | alpha={[round(a, 3) for a in alpha]}")
    else:
        loss_fn = None

    model, base_model = build_efficientnet(backbone=args.backbone)
    model = compile_phase1(model, learning_rate=1e-3, loss_fn=loss_fn)
    print(f"\n[•] Paramètres totaux : {model.count_params():,}")

    print(f"\n[Phase 1] Feature Extraction — {args.phase1_epochs} epochs\n")
    h1 = model.fit(
        train_gen,
        epochs=args.phase1_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks_tl("phase1"),
        verbose=1,
    )

    print(f"\n[Phase 2] Fine-Tuning (dégel couches {args.unfreeze}:) — {args.phase2_epochs} epochs\n")
    model = unfreeze_for_finetuning(
        model, base_model,
        unfreeze_from_layer=args.unfreeze,
        learning_rate=5e-6,
        loss_fn=loss_fn,
    )

    h2 = model.fit(
        train_gen,
        epochs=args.phase2_epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks_tl("phase2", patience=8),
        verbose=1,
    )

    final_path = os.path.join(MODELS_DIR, f"tl_{args.backbone}_final.keras")
    model.save(final_path)
    print(f"\n[✓] Modèle TL sauvegardé : {final_path}")

    best_p2_acc    = max(h2.history.get("val_accuracy", [0]))
    best_p2_recall = max(h2.history.get("val_recall",   [0]))
    best_p2_auc    = max(h2.history.get("val_auc",      [0]))

    print(f"\n{'─'*50}")
    print(f"  Phase2 Best Val Accuracy : {best_p2_acc:.4f} ({best_p2_acc*100:.1f}%)")
    print(f"  Phase2 Best Val Recall   : {best_p2_recall:.4f} ({best_p2_recall*100:.1f}%)")
    if best_p2_recall >= 0.95:
        print("  ✅ Cible Recall > 95% ATTEINTE !")
    else:
        print(f"  ⚠️  Recall : {best_p2_recall*100:.1f}% (cible 95%)")
    print(f"  Phase2 Best Val AUC      : {best_p2_auc:.4f}")
    print(f"{'─'*50}")

    plot_tl_history(h1, h2)

    print(f"\n[✓] Évaluation : python src/evaluate.py --model {final_path}")

if __name__ == "__main__":
    main()
