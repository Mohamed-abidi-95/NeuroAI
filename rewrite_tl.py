"""Reecrit train_4classes_tl.py de zero, sans BOM ni caracteres speciaux."""
import os

content = r'''"""
train_4classes_tl.py - Transfer Learning EfficientNetB0 sur 4 classes.
Objectif : val_recall > 95%
Phase 1 - Feature Extraction (backbone ImageNet gele)
Phase 2 - Fine-Tuning (dernieres couches degelees, lr tres faible)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from losses import FocalLoss, compute_focal_alpha

DATA_DIR    = os.path.join("data", "staged_4classes")
MODELS_DIR  = "models"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 4

os.makedirs(MODELS_DIR, exist_ok=True)


def get_generators_4classes_tl():
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.08,
        brightness_range=[0.85, 1.15],
        fill_mode="reflect",
        validation_split=0.2,
    )
    val_datagen = ImageDataGenerator(validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="training", seed=42, shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", subset="validation", seed=42, shuffle=False,
    )
    print("[OK] Classes : {}".format(train_gen.class_indices))
    print("     Train   : {} images".format(train_gen.samples))
    print("     Val     : {} images".format(val_gen.samples))
    return train_gen, val_gen


def build_tl_4classes(backbone="efficientnetb0"):
    inputs = tf.keras.layers.Input(shape=(*IMG_SIZE, 3))

    if backbone == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=inputs)
        print("[OK] Backbone : EfficientNetB0 (ImageNet)")
    elif backbone == "resnet50v2":
        x_pre = tf.keras.applications.resnet_v2.preprocess_input(inputs)
        base = tf.keras.applications.ResNet50V2(
            include_top=False, weights="imagenet", input_tensor=x_pre)
        print("[OK] Backbone : ResNet50V2 (ImageNet)")
    elif backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
        base.trainable = False
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.40)(x)
        outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs,
                               name="TL_MobileNetV2_4Classes")
        print("[OK] Backbone : MobileNetV2 (ImageNet)")
        return model, base
    else:
        raise ValueError("Backbone inconnu : {}".format(backbone))

    base.trainable = False
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs,
                           name="TL_{}_4Classes".format(backbone))
    return model, base


def compile_model(model, lr, loss_fn):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    trainable = sum([tf.size(w).numpy() for w in model.trainable_variables])
    print("     Params entrainables : {:,}".format(trainable))
    return model


def get_callbacks(phase, patience_es=12, patience_lr=5):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "tl_4classes_{}_best.keras".format(phase)),
            monitor="val_recall", save_best_only=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", patience=patience_es,
            restore_best_weights=True, mode="max", verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_recall", factor=0.5, patience=patience_lr,
            min_lr=1e-8, mode="max", verbose=1,
        ),
    ]


def optimize_threshold(model, val_gen):
    print("\n[.] Optimisation du seuil de decision...")
    val_gen.reset()
    y_true_all, y_pred_all = [], []
    for i in range(len(val_gen)):
        xb, yb = val_gen[i]
        preds = model.predict(xb, verbose=0)
        y_true_all.append(np.argmax(yb, axis=1))
        y_pred_all.append(preds)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    best_thresh, best_recall = 0.5, 0.0
    for thresh in np.arange(0.30, 0.71, 0.02):
        y_conf  = np.max(y_pred, axis=1)
        y_label = np.argmax(y_pred, axis=1)
        mask    = y_conf >= thresh
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_label[mask]
        recall_avg = float(np.mean([
            float(np.sum((yt == c) & (yp == c))) / float(max(int(np.sum(yt == c)), 1))
            for c in range(NUM_CLASSES)
        ]))
        if recall_avg > best_recall:
            best_recall = recall_avg
            best_thresh = thresh

    print("     Seuil optimal : {:.2f}  -> recall={:.1f}%".format(
        best_thresh, best_recall * 100))
    return best_thresh


def plot_history(h1, h2, backbone):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Transfer Learning {} -- 4 Classes".format(backbone.upper()), fontsize=13)

    for ax, key, title in zip(axes,
                               ["loss", "accuracy", "recall", "auc"],
                               ["Loss", "Accuracy", "Recall (cible >= 95%)", "AUC"]):
        off = len(h1.history.get(key, []))
        vkey = "val_{}".format(key)
        if key in h1.history:
            ax.plot(h1.history[key], color="royalblue", label="Ph1 Train")
        if vkey in h1.history:
            ax.plot(h1.history[vkey], color="cornflowerblue",
                    label="Ph1 Val", linestyle="--")
        x2 = range(off, off + len(h2.history.get(key, [])))
        if key in h2.history:
            ax.plot(x2, h2.history[key], color="tomato", label="Ph2 Train")
        if vkey in h2.history:
            ax.plot(x2, h2.history[vkey], color="salmon",
                    label="Ph2 Val", linestyle="--")
        if off > 0:
            ax.axvline(x=off, color="gray", linestyle=":", alpha=0.7)
        if key == "recall":
            ax.axhline(y=0.95, color="green", linestyle="--", alpha=0.7, label="Cible 95%")
        ax.set_title(title)
        ax.set_xlabel("Epoque")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    path = os.path.join(MODELS_DIR, "tl_4classes_{}_history.png".format(backbone))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("[OK] Courbes sauvegardees : {}".format(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["efficientnetb0", "resnet50v2", "mobilenetv2"],
                        default="efficientnetb0")
    parser.add_argument("--phase1-epochs", type=int, default=30)
    parser.add_argument("--phase2-epochs", type=int, default=25)
    parser.add_argument("--loss", choices=["ce", "focal"], default="focal")
    parser.add_argument("--unfreeze", type=int, default=-40)
    args = parser.parse_args()

    print("=" * 65)
    print("  TRANSFER LEARNING -- 4 CLASSES -- Cible Recall >= 95%")
    print("  Backbone : {}".format(args.backbone.upper()))
    print("  Phase 1  : {} epochs".format(args.phase1_epochs))
    print("  Phase 2  : {} epochs (couches {}:)".format(args.phase2_epochs, args.unfreeze))
    print("  Loss     : {}".format("Focal Loss" if args.loss == "focal" else "CrossEntropy"))
    print("=" * 65)

    train_gen, val_gen = get_generators_4classes_tl()
    labels = train_gen.classes

    classes      = np.unique(labels)
    weights_arr  = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = dict(zip(classes.astype(int), weights_arr))
    print("[OK] Class weights : {}".format(class_weights))

    if args.loss == "focal":
        counts  = {i: int(np.sum(labels == i)) for i in range(NUM_CLASSES)}
        alpha   = compute_focal_alpha(counts)
        loss_fn = FocalLoss(gamma=2.0, alpha=alpha)
        print("[OK] Focal Loss | gamma=2.0 | alpha={}".format([round(a, 3) for a in alpha]))
    else:
        loss_fn = "categorical_crossentropy"

    model, base_model = build_tl_4classes(backbone=args.backbone)
    model = compile_model(model, lr=1e-3, loss_fn=loss_fn)
    print("[OK] Params totaux : {:,}".format(model.count_params()))

    print("\n" + "-" * 65)
    print("  PHASE 1 -- Feature Extraction ({} epochs max)".format(args.phase1_epochs))
    print("-" * 65)
    h1 = model.fit(
        train_gen, epochs=args.phase1_epochs,
        validation_data=val_gen, class_weight=class_weights,
        callbacks=get_callbacks("phase1", patience_es=12, patience_lr=5),
        verbose=1,
    )
    best_p1 = max(h1.history.get("val_recall", [0]))
    print("  Phase 1 -- Meilleur val_recall : {:.1f}%".format(best_p1 * 100))

    print("\n" + "-" * 65)
    print("  PHASE 2 -- Fine-Tuning (couches {}:), lr=5e-6".format(args.unfreeze))
    print("-" * 65)

    base_model.trainable = True
    for layer in base_model.layers[:args.unfreeze]:
        layer.trainable = False
    for layer in base_model.layers[args.unfreeze:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)

    model = compile_model(model, lr=5e-6, loss_fn=loss_fn)
    h2 = model.fit(
        train_gen, epochs=args.phase2_epochs,
        validation_data=val_gen, class_weight=class_weights,
        callbacks=get_callbacks("phase2", patience_es=10, patience_lr=4),
        verbose=1,
    )

    final_path = os.path.join(MODELS_DIR, "tl_4classes_{}_final.keras".format(args.backbone))
    model.save(final_path)
    print("\n[OK] Modele sauvegarde : {}".format(final_path))

    best_recall = max(h2.history.get("val_recall", [0]))
    best_acc    = max(h2.history.get("val_accuracy", [0]))
    best_auc    = max(h2.history.get("val_auc", [0]))
    overall     = max(best_recall, best_p1)

    print("\n" + "=" * 65)
    print("  RESULTATS FINAUX")
    print("-" * 65)
    print("  Val Accuracy : {:.1f}%".format(best_acc * 100))
    print("  Val Recall   : {:.1f}% (overall best: {:.1f}%)".format(
        best_recall * 100, overall * 100))
    print("  Val AUC      : {:.4f}".format(best_auc))
    if overall >= 0.95:
        print("  [OK] Cible Recall >= 95% ATTEINTE !")
    else:
        print("  [!]  Recall : {:.1f}% -- manque {:.1f}%".format(
            overall * 100, (0.95 - overall) * 100))
    print("=" * 65)

    best_thresh = optimize_threshold(model, val_gen)
    thresh_path = os.path.join(MODELS_DIR, "tl_4classes_{}_threshold.txt".format(args.backbone))
    with open(thresh_path, "w") as f:
        f.write("{:.4f}\n".format(best_thresh))
    print("[OK] Seuil optimal : {} -> {}".format(best_thresh, thresh_path))

    plot_history(h1, h2, args.backbone)
    print("\n[OK] Prochaine etape :")
    print("     python src/evaluate.py --model {}".format(final_path))


if __name__ == "__main__":
    main()
'''

out_path = os.path.join("src", "train_4classes_tl.py")
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)

# Verification : pas de BOM
with open(out_path, "rb") as f:
    first = f.read(3)
assert first != b"\xef\xbb\xbf", "BOM detecte !"
print("Fichier reecrit proprement (sans BOM) : {}".format(out_path))

