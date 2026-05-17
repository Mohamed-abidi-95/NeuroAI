import os
import sys
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses import FocalLoss, compute_focal_alpha
import numpy as np

DATA_DIR = os.path.join("data", "staged_4classes")
MODELS_DIR = "models"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 4                            

os.makedirs(MODELS_DIR, exist_ok=True)

def get_generators_4classes():
                                     
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=25,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.10,
        brightness_range=[0.85, 1.15],
        fill_mode="reflect",
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=42,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=42,
        shuffle=False,
    )

    print(f"[OK] Classes trouvees : {train_gen.class_indices}")
    print(f"    Train   : {train_gen.samples} images")
    print(f"    Val     : {val_gen.samples} images")

    return train_gen, val_gen

def build_cnn_4classes():
                                                                                       
    reg = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=reg, input_shape=(128, 128, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                               kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.40),

        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),             
    ], name="CNN_4Classes_v2")

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["ce", "focal"], default="focal")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print("=" * 60)
    print("  ENTRAÎNEMENT — CNN BASELINE (4 classes)")
    print(f"  Loss : {'Focal Loss' if args.loss == 'focal' else 'Categorical CrossEntropy'}")
    print(f"  Epochs : {args.epochs}")
    print("=" * 60)

    train_gen, val_gen = get_generators_4classes()
    labels = train_gen.classes

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = dict(zip(classes.astype(int), weights))
    print(f"[OK] Poids des classes : {class_weights}")

    if args.loss == "focal":
        counts = {i: int(np.sum(labels == i)) for i in range(NUM_CLASSES)}
        alpha = compute_focal_alpha(counts)
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        print(f"[OK] Focal Loss | gamma=2.0 | alpha={[round(a, 3) for a in alpha]}")
    else:
        loss_fn = "categorical_crossentropy"

    model = build_cnn_4classes()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=["accuracy",
                 tf.keras.metrics.Recall(name="recall"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.AUC(name="auc")],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "cnn_4classes_best.keras"),
            monitor="val_recall",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            patience=15,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_recall",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            mode="max",
            verbose=1,
        ),
    ]

    print(f"\n[•] Début entraînement ({args.epochs} epochs max) ...\n")
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(MODELS_DIR, "cnn_4classes_final.keras")
    model.save(final_path)
    print(f"\n[OK] Modèle final sauvegardé : {final_path}")

    best_acc = max(history.history["val_accuracy"])
    best_recall = max(history.history.get("val_recall", [0]))
    best_auc = max(history.history.get("val_auc", [0]))

    print(f"\n{'─'*50}")
    print(f"  Meilleure Val Accuracy : {best_acc:.4f} ({best_acc*100:.1f}%)")
    print(f"  Meilleure Val Recall   : {best_recall:.4f} ({best_recall*100:.1f}%)")
    if best_recall >= 0.95:
        print("  ✅ Cible Recall > 95% ATTEINTE !")
    else:
        print(f"  Cible Recall 95% : manque {(0.95-best_recall)*100:.1f}%")
    print(f"  Meilleure Val AUC      : {best_auc:.4f}")
    print(f"{'─'*50}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CNN 4 Classes — Entraînement", fontsize=13)

    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train")
    axes[1].plot(history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if "recall" in history.history:
        axes[2].plot(history.history["recall"], label="Train")
        axes[2].plot(history.history["val_recall"], label="Val")
        axes[2].set_title("Recall")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "cnn_4classes_history.png"), dpi=150)
    print(f"[OK] Courbes sauvegardées : {os.path.join(MODELS_DIR, 'cnn_4classes_history.png')}")

    print(f"\n[OK] Entraînement terminé. Prochaine étape :")
    print(f"    python src/evaluate.py --model {final_path}")

if __name__ == "__main__":
    main()
