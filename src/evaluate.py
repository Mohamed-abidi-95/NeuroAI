import os
import sys
import argparse
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
from tensorflow.keras.preprocessing import image as keras_image

MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
CONFIDENCE_THRESHOLD = 0.7
os.makedirs(REPORTS_DIR, exist_ok=True)

LABELS_4 = ["Stage0-NoTumor", "Stage1-Meningioma", "Stage2-Pituitary", "Stage3-4-Glioma"]
LABELS_5 = ["Stage0-NoTumor", "Stage1-Meningioma", "Stage2-Pituitary", "Stage3-Anaplastic", "Stage4-GBM"]

FOLDER_TO_IDX_4 = {
             :    0,
                : 1,
               :  2,
            :     3,
}
FOLDER_TO_IDX_5 = {
             :    0,
                : 1,
               :  2,
            :     3,
}

def find_best_model():
                                                                    
    priority = [
                                                ,
                                       ,
                                  ,
                                  ,
    ]
    for name in priority:
        path = os.path.join(MODELS_DIR, name)
        if os.path.exists(path):
            return path
    for f in os.listdir(MODELS_DIR):
        if f.endswith(".keras"):
            return os.path.join(MODELS_DIR, f)
    return None

def load_test_data(data_dir: str, img_size: tuple, num_classes: int, normalize: bool = True):
           
    folder_map = FOLDER_TO_IDX_4 if num_classes <= 4 else FOLDER_TO_IDX_5
    X, y = [], []
    for folder, idx in folder_map.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            print("[WARN] Dossier manquant : {}".format(folder_path))
            continue
        files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print("  {} : {} images".format(folder, len(files)))
        for fname in files:
            img_path = os.path.join(folder_path, fname)
            try:
                img = keras_image.load_img(img_path, target_size=img_size)
                arr = keras_image.img_to_array(img)
                if normalize:
                    arr = arr / 255.0
                X.append(arr)
                y.append(idx)
            except Exception as e:
                print("[WARN] Impossible de charger {} : {}".format(fname, e))
    return np.array(X), np.array(y)

def evaluate_model(model, X_test, y_true, class_labels, model_name="model"):
                                                                        
    print("\n" + "=" * 60)
    print("  EVALUATION : {}".format(model_name))
    print("=" * 60)
    n_classes = len(class_labels)

    proba = model.predict(X_test, verbose=1, batch_size=32)
    y_pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    acc    = float(np.mean(y_pred == y_true))
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    prec   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    f2     = fbeta_score(y_true, y_pred, beta=2, average="macro", zero_division=0)

    print("\n[METRIQUES]")
    print("  Accuracy         : {:.4f}".format(acc))
    print("  Recall (macro)   : {:.4f}".format(recall))
    print("  Precision (macro): {:.4f}".format(prec))
    print("  F2-Score (macro) : {:.4f}".format(f2))

    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if n_classes == 2:
        auc = roc_auc_score(y_true, proba[:, 1])
    else:
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
    print("  ROC-AUC (macro)  : {:.4f}".format(auc))

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, target_names=class_labels, zero_division=0))

    uncertain_mask = confidence < CONFIDENCE_THRESHOLD
    n_uncertain = int(uncertain_mask.sum())
    print("[INCERTAINS] {} cas avec confiance < {:.0%}  ({:.1f}%)".format(
        n_uncertain, CONFIDENCE_THRESHOLD, n_uncertain / len(y_true) * 100))

    metrics_path = os.path.join(REPORTS_DIR, "{}_metrics.txt".format(model_name))
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("MODEL: {}\n".format(model_name))
        f.write("Accuracy         : {:.4f}\n".format(acc))
        f.write("Recall (macro)   : {:.4f}\n".format(recall))
        f.write("Precision (macro): {:.4f}\n".format(prec))
        f.write("F2-Score (macro) : {:.4f}\n".format(f2))
        f.write("ROC-AUC (macro)  : {:.4f}\n".format(auc))
        f.write("Cas incertains   : {}/{}\n".format(n_uncertain, len(y_true)))
        f.write("\n")
        f.write(classification_report(y_true, y_pred,
                                       target_names=class_labels, zero_division=0))
    print("[OK] Metriques sauvegardees : {}".format(metrics_path))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Reel")
    ax.set_title("Matrice de Confusion - {}".format(model_name))
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "{}_confusion_matrix.png".format(model_name))
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Matrice de confusion : {}".format(cm_path))

    colors = ["steelblue", "tomato", "seagreen", "orange", "purple"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, label in enumerate(class_labels):
        if i < y_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc_i = roc_auc_score(y_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, color=colors[i % len(colors)],
                    label="{} (AUC={:.3f})".format(label, auc_i))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Taux Faux Positifs")
    ax.set_ylabel("Taux Vrais Positifs")
    ax.set_title("Courbes ROC - {}".format(model_name))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "{}_roc_curves.png".format(model_name))
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] Courbes ROC : {}".format(roc_path))

    return {
                  :    acc,
                :      recall,
            :          f2,
             :         auc,
                     : n_uncertain,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluation clinique du modele")
    parser.add_argument("--model",     type=str, default=None,
                        help="Chemin vers le modele .keras")
    parser.add_argument("--data",      type=str, default="data/Testing",
                        help="Dossier de test (defaut: data/Testing)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Seuil de confiance (defaut: 0.7)")
    args = parser.parse_args()

    global CONFIDENCE_THRESHOLD
    CONFIDENCE_THRESHOLD = args.threshold

    model_path = args.model if args.model else find_best_model()
    if model_path is None or not os.path.exists(model_path):
        print("[ERREUR] Aucun modele trouve. Specifiez --model <chemin>.")
        sys.exit(1)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print("\n[MODELE] {}".format(model_path))

    threshold_path = model_path.replace(".keras", "_threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            opt_threshold = float(f.read().strip())
        print("[SEUIL] Seuil optimal charge : {:.4f}".format(opt_threshold))
        CONFIDENCE_THRESHOLD = opt_threshold

    try:
        sys.path.insert(0, "src")
        from losses import FocalLoss
        custom_objs = {"FocalLoss": FocalLoss}
    except ImportError:
        custom_objs = {}

    print("[INFO] Chargement du modele...")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)

    img_height, img_width = model.input_shape[1:3]
    num_classes = int(model.output_shape[-1])
    img_size = (img_height, img_width)

    normalize = (img_height < 224)
    print("[INFO] Input shape : {}x{} | Classes : {} | Normalize: {}".format(
        img_height, img_width, num_classes, normalize))

    class_labels = LABELS_4 if num_classes <= 4 else LABELS_5

    print("\n[DATA] Chargement depuis : {}".format(args.data))
    if not os.path.isdir(args.data):
        print("[ERREUR] Dossier de test introuvable : {}".format(args.data))
        sys.exit(1)

    X_test, y_test = load_test_data(args.data, img_size, num_classes, normalize=normalize)
    print("[DATA] {} images chargees.".format(len(X_test)))

    if len(X_test) == 0:
        print("[ERREUR] Aucune image trouvee dans {}".format(args.data))
        sys.exit(1)

    metrics = evaluate_model(model, X_test, y_test, class_labels, model_name)

    print("\n" + "=" * 60)
    print("  RESUME FINAL")
    print("=" * 60)
    print("  Accuracy  : {:.2%}".format(metrics["accuracy"]))
    print("  Recall    : {:.2%}".format(metrics["recall"]))
    print("  F2-Score  : {:.4f}".format(metrics["f2"]))
    print("  ROC-AUC   : {:.4f}".format(metrics["auc"]))
    print("  Incertains: {}".format(metrics["n_uncertain"]))
    print("=" * 60)
    print("[OK] Evaluation terminee. Rapports dans : {}".format(REPORTS_DIR))

if __name__ == "__main__":
    main()
