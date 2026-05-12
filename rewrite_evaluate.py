"""Reecrit src/evaluate.py completement."""
import os

content = '''"""
evaluate.py
Evaluation clinique complete du modele - 4 ou 5 classes (auto-detection).
- Accuracy, Sensibilite (Recall), F2-Score
- ROC-AUC par stade
- Matrice de confusion
- Detection d incertitude probabiliste (seuil Softmax < 0.7)
- Optimisation du seuil de decision pour maximiser le recall
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, fbeta_score,
    recall_score, precision_score,
)
from sklearn.preprocessing import label_binarize

MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
CONFIDENCE_THRESHOLD = 0.7
os.makedirs(REPORTS_DIR, exist_ok=True)

STAGE_NAMES_4 = [
    "Stage 0\\n(No Tumor)",
    "Stage I\\n(Meningioma)",
    "Stage II\\n(Pituitary)",
    "Stage III/IV\\n(Glioma)",
]
STAGE_NAMES_5 = [
    "Stage 0\\n(No Tumor)",
    "Stage I\\n(Meningioma)",
    "Stage II\\n(Pituitary)",
    "Stage III\\n(Anaplastic)",
    "Stage IV\\n(GBM)",
]


def get_stage_names(num_classes):
    return STAGE_NAMES_4 if num_classes <= 4 else STAGE_NAMES_5


def predict_with_uncertainty(model, X_test):
    y_proba = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_proba, axis=1)
    confidence = np.max(y_proba, axis=1)
    requires_review = confidence < CONFIDENCE_THRESHOLD
    return y_pred, y_proba, confidence, requires_review


def plot_confusion_matrix(y_true, y_pred, model_name, stage_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    n = len(stage_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 2), max(6, n * 1.8)))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=stage_names, yticklabels=stage_names,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix (%) - {}".format(model_name), fontsize=13, pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "{}_confusion_matrix.png".format(model_name))
    plt.savefig(path, dpi=150)
    plt.close()
    print("[OK] Confusion matrix : {}".format(path))
    return path


def plot_roc_curves(y_true_bin, y_proba, model_name, stage_names):
    colors = ["navy", "cornflowerblue", "darkorange", "forestgreen", "crimson"]
    fig, ax = plt.subplots(figsize=(10, 7))
    num_classes = y_proba.shape[1]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        short = stage_names[i].split("\\n")[0]
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label="{} - AUC = {:.3f}".format(short, auc))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curves by Stage - {}".format(model_name), fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "{}_roc_curves.png".format(model_name))
    plt.savefig(path, dpi=150)
    plt.close()
    print("[OK] ROC curves : {}".format(path))
    return path


def find_optimal_threshold(y_true, y_proba, num_classes):
    best_thresh, best_recall = 0.5, 0.0
    for thresh in np.arange(0.30, 0.71, 0.02):
        conf   = np.max(y_proba, axis=1)
        labels = np.argmax(y_proba, axis=1)
        mask   = conf >= thresh
        if mask.sum() < 10:
            continue
        yt = y_true[mask]
        yp = labels[mask]
        r  = recall_score(yt, yp, average="macro", zero_division=0)
        if r > best_recall:
            best_recall = r
            best_thresh = thresh
    kept = int((np.max(y_proba, axis=1) >= best_thresh).sum())
    print("     Optimal threshold : {:.2f} -> recall {:.1f}% ({}/{} samples)".format(
        best_thresh, best_recall * 100, kept, len(y_true)))
    return float(best_thresh), float(best_recall)


def compute_all_metrics(y_true, y_pred, y_proba, model_name, num_classes):
    stage_names = get_stage_names(num_classes)
    print("\\n" + "=" * 62)
    print("  CLINICAL EVALUATION -- {}".format(model_name.upper()))
    print("=" * 62)

    acc = float(np.mean(y_true == y_pred))
    print("\\n  Accuracy              : {:.4f} ({:.1f}%)".format(acc, acc * 100))

    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    prec_macro   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    print("  Recall (macro)        : {:.4f} ({:.1f}%)".format(
        recall_macro, recall_macro * 100))
    if recall_macro >= 0.95:
        print("  [OK] Target Recall >= 95% ACHIEVED!")
    else:
        print("  [!]  Target Recall >= 95% : missing {:.1f}%".format(
            (0.95 - recall_macro) * 100))

    f2 = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)
    print("  F2-Score (macro)      : {:.4f}".format(f2))
    print("  Precision (macro)     : {:.4f} ({:.1f}%)".format(prec_macro, prec_macro * 100))

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
        print("  ROC-AUC (macro OvR)   : {:.4f}".format(auc_macro))
    except Exception as exc:
        print("  ROC-AUC : error ({})".format(exc))
        auc_macro = 0.0

    print("\\n" + classification_report(
        y_true, y_pred, target_names=stage_names, zero_division=0))

    opt_thresh, opt_recall = find_optimal_threshold(y_true, y_proba, num_classes)

    report_path = os.path.join(REPORTS_DIR, "{}_metrics.txt".format(model_name))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model : {}\\n".format(model_name))
        f.write("Num classes         : {}\\n".format(num_classes))
        f.write("Accuracy            : {:.4f} ({:.1f}%)\\n".format(acc, acc * 100))
        f.write("Recall (macro)      : {:.4f} ({:.1f}%)\\n".format(
            recall_macro, recall_macro * 100))
        f.write("Precision (macro)   : {:.4f} ({:.1f}%)\\n".format(
            prec_macro, prec_macro * 100))
        f.write("F2-Score (macro)    : {:.4f}\\n".format(f2))
        f.write("ROC-AUC (macro OvR) : {:.4f}\\n".format(auc_macro))
        f.write("Optimal threshold   : {:.2f} -> recall {:.1f}%\\n\\n".format(
            opt_thresh, opt_recall * 100))
        f.write(classification_report(y_true, y_pred, target_names=stage_names,
                                      zero_division=0))
    print("[OK] Report saved : {}".format(report_path))

    return {
        "accuracy": acc, "recall": recall_macro,
        "f2": f2, "auc": auc_macro,
        "optimal_threshold": opt_thresh,
    }


def log_uncertain_cases(filenames, confidence, y_pred, requires_review,
                         model_name, stage_names):
    import pandas as pd
    rows = []
    for i, (fn, conf, pred, rev) in enumerate(
            zip(filenames, confidence, y_pred, requires_review)):
        if rev:
            rows.append({
                "index":           i,
                "file":            fn,
                "predicted_stage": stage_names[pred].replace("\\n", " "),
                "confidence":      round(float(conf), 4),
                "action":          "MANUAL REVIEW REQUIRED",
            })
    if rows:
        import pandas as pd2
        df = pd2.DataFrame(rows)
        path = os.path.join(REPORTS_DIR, "{}_uncertain_cases.csv".format(model_name))
        df.to_csv(path, index=False, encoding="utf-8")
        print("[!]  {} uncertain cases logged : {}".format(len(rows), path))
    else:
        print("[OK] No uncertain cases (all confidence >= {}).".format(
            CONFIDENCE_THRESHOLD))


def load_test_data_from_dir(data_dir, img_h, img_w, rescale):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale=1.0 / 255 if rescale else None)

    # Chemin du dossier Testing
    test_path = os.path.join("data", "Testing")
    if not os.path.exists(test_path):
        # Fallback: val split du data_dir
        test_datagen2 = ImageDataGenerator(
            rescale=1.0 / 255 if rescale else None,
            validation_split=0.2)
        gen = test_datagen2.flow_from_directory(
            data_dir, target_size=(img_h, img_w), batch_size=32,
            class_mode="categorical", subset="validation", seed=42, shuffle=False)
    else:
        gen = test_datagen.flow_from_directory(
            test_path, target_size=(img_h, img_w), batch_size=32,
            class_mode="categorical", shuffle=False)

    print("[.] Loading test data ({} images from {})...".format(
        gen.samples, gen.directory))
    X_list, y_list = [], []
    for i in range(len(gen)):
        xb, yb = gen[i]
        X_list.append(xb)
        y_list.append(yb)
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    return X, y


def evaluate_model(model_path, X_test, y_test_raw, num_classes=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from losses import FocalLoss

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print("[.] Loading model : {}".format(model_path))
    model = tf.keras.models.load_model(
        model_path, custom_objects={"FocalLoss": FocalLoss})

    if num_classes is None:
        num_classes = int(model.output_shape[-1])
    stage_names = get_stage_names(num_classes)
    print("[OK] Model loaded | {} classes | input {}".format(
        num_classes, model.input_shape))

    y_pred, y_proba, confidence, requires_review = predict_with_uncertainty(
        model, X_test)

    metrics = compute_all_metrics(
        y_test_raw, y_pred, y_proba, model_name, num_classes)

    y_true_bin = label_binarize(y_test_raw, classes=list(range(num_classes)))
    plot_confusion_matrix(y_test_raw, y_pred, model_name, stage_names)
    try:
        plot_roc_curves(y_true_bin, y_proba, model_name, stage_names)
    except Exception as exc:
        print("[!]  ROC curves skipped : {}".format(exc))

    n_uncertain = int(np.sum(requires_review))
    pct = n_uncertain / len(y_test_raw) * 100
    print("\\n  Uncertain cases (conf < {}) : {}/{} ({:.1f}%)".format(
        CONFIDENCE_THRESHOLD, n_uncertain, len(y_test_raw), pct))

    return metrics


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set")
    parser.add_argument("--model", required=True, help="Path to .keras/.h5 model")
    parser.add_argument("--data-dir", default=None,
                        help="Override data directory")
    parser.add_argument("--no-rescale", action="store_true",
                        help="Skip rescaling (for TL models expecting [0,255])")
    args = parser.parse_args()

    model_basename = os.path.basename(args.model).lower()
    is_tl     = "tl_" in model_basename
    is_4class = "4class" in model_basename or "staged_4" in (args.data_dir or "")

    rescale = not (args.no_rescale or is_tl)

    print("[.] Model    : {}".format(args.model))
    print("[.] Rescale  : {}".format(rescale))
    print("[.] TL model : {}".format(is_tl))

    # Determiner img_size depuis le modele
    from losses import FocalLoss
    _tmp = tf.keras.models.load_model(args.model,
                                       custom_objects={"FocalLoss": FocalLoss})
    img_h, img_w = int(_tmp.input_shape[1]), int(_tmp.input_shape[2])
    num_cls = int(_tmp.output_shape[-1])
    del _tmp

    if args.data_dir:
        data_dir = args.data_dir
    elif is_4class:
        data_dir = os.path.join("data", "staged_4classes")
    else:
        data_dir = os.path.join("data", "staged")

    X_test, y_test = load_test_data_from_dir(data_dir, img_h, img_w, rescale)
    y_test_raw = np.argmax(y_test, axis=1)

    print("[OK] Test set : {} samples, {} classes".format(len(y_test_raw), num_cls))
    evaluate_model(args.model, X_test, y_test_raw, num_classes=num_cls)
'''

out_path = os.path.join("src", "evaluate.py")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)

# Verif BOM
with open(out_path, "rb") as f:
    first = f.read(3)
assert first != b"\\xef\\xbb\\xbf", "BOM detecte!"
print("evaluate.py reecrit proprement ({} lignes)".format(content.count("\\n")))

