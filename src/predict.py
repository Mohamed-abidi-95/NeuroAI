import os
import sys
import argparse
import glob
import csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from losses import FocalLoss
from gradcam import visualize_gradcam

LABELS_4 = {
    0: "Stage 0 - No Tumor (Negative Control)",
    1: "Stage I  - Meningioma (WHO Grade I — Benign)",
    2: "Stage II - Pituitary (Local Extension)",
    3: "Stage III/IV - Glioma (WHO Grade III/IV — Malignant)",
}
LABELS_5 = {
    0: "Stage 0 - No Tumor (Negative Control)",
    1: "Stage I  - Meningioma (WHO Grade I — Benign)",
    2: "Stage II - Pituitary (Local Extension)",
    3: "Stage III - Anaplastic Glioma (WHO Grade III)",
    4: "Stage IV - Glioblastoma GBM (WHO Grade IV)",
}
CLINICAL_4 = {
    0: "No mass detected. Routine follow-up recommended.",
    1: "Slow-growing benign tumor. Close monitoring advised.",
    2: "Localized tumor. Oncology consultation recommended.",
    3: "Malignant tumor detected. URGENT ONCOLOGICAL MANAGEMENT required.",
}
CLINICAL_5 = {
    0: "No mass detected. Routine follow-up recommended.",
    1: "Slow-growing benign tumor. Close monitoring advised.",
    2: "Localized tumor. Oncology consultation recommended.",
    3: "Regional lymph node involvement. Urgent treatment required.",
    4: "Multi-focal lesions detected. ONCOLOGICAL EMERGENCY.",
}

CONF_THRESHOLD = 0.70
MODELS_DIR     = "models"
GRADCAM_DIR    = "reports/gradcam"

def find_best_model():
                                                                     
    candidates = [
        os.path.join(MODELS_DIR, "tl_4classes_efficientnetb0_final.keras"),
        os.path.join(MODELS_DIR, "tl_4classes_resnet50v2_final.keras"),
        os.path.join(MODELS_DIR, "tl_4classes_mobilenetv2_final.keras"),
        os.path.join(MODELS_DIR, "cnn_4classes_best.keras"),
        os.path.join(MODELS_DIR, "cnn_4classes_final.keras"),
        os.path.join(MODELS_DIR, "tl_efficientnetb0_final.keras"),
        os.path.join(MODELS_DIR, "cnn_baseline_best.keras"),
        os.path.join(MODELS_DIR, "cnn_baseline_final.keras"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def load_model(model_path):
                                                       
    print("[.] Chargement du modèle : {}".format(model_path))
    model = tf.keras.models.load_model(
        model_path, custom_objects={"FocalLoss": FocalLoss})
    num_classes = int(model.output_shape[-1])
    img_h = int(model.input_shape[1])
    img_w = int(model.input_shape[2])
    print("[OK] Modèle chargé | {} classes | {}×{} px".format(num_classes, img_h, img_w))
    return model, num_classes, (img_h, img_w)

def preprocess_image(img_path, img_size, num_classes, is_tl=False):
                                                       
    img = Image.open(img_path).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32)

    if not is_tl:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0), arr

def predict_single(model, img_batch, num_classes, labels, clinical_notes):
                                                                  
    y_proba    = model.predict(img_batch, verbose=0)[0]
    stage_id   = int(np.argmax(y_proba))
    confidence = float(np.max(y_proba))
    requires_review = confidence < CONF_THRESHOLD

    return {
                  :        stage_id,
                     :     labels.get(stage_id, "Unknown"),
                       :   clinical_notes.get(stage_id, ""),
                    :      round(confidence, 4),
                       :   [round(float(p), 4) for p in y_proba],
                         : requires_review,
    }

def print_result(result, img_path):
                                                   
    sid  = result["stage_id"]
    lbl  = result["stage_label"]
    conf = result["confidence"]
    probs = result["probabilities"]
    note = result["clinical_note"]
    rev  = result["requires_review"]

    sep = "=" * 65
    print("\n" + sep)
    print("  PREDICTION : {}".format(os.path.basename(img_path)))
    print(sep)
    print("  Stage prédit    : {}".format(lbl))
    print("  Confiance       : {:.1%}".format(conf))
    print("  Note clinique   : {}".format(note))
    print("\n  Probabilités par stade :")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 30)
        marker = " <--- PRÉDIT" if i == sid else ""
        print("    Stade {:1d}  {:.1%}  {}{}".format(i, p, bar, marker))

    if rev:
        print("\n  [!] RÉVISION MANUELLE REQUISE  "
                                                 .format(conf, CONF_THRESHOLD))
    else:
        print("\n  [OK] Diagnostic confiant")
    print(sep)

def process_image(model, img_path, num_classes, is_tl, save_gradcam=False):
                                                   
    labels         = LABELS_4 if num_classes <= 4 else LABELS_5
    clinical_notes = CLINICAL_4 if num_classes <= 4 else CLINICAL_5

    img_h = int(model.input_shape[1])
    img_w = int(model.input_shape[2])
    img_size = (img_h, img_w)

    img_batch, img_arr = preprocess_image(img_path, img_size, num_classes, is_tl)
    result = predict_single(model, img_batch, num_classes, labels, clinical_notes)
    print_result(result, img_path)

    if save_gradcam:
        os.makedirs(GRADCAM_DIR, exist_ok=True)
        try:
            base = os.path.splitext(os.path.basename(img_path))[0]
            save_name = "{}_gradcam_stage{}".format(base, result["stage_id"])
            path, _, _, _ = visualize_gradcam(
                model, img_arr,
                true_label=None,
                save_name=save_name,
            )
            print("[OK] Grad-CAM sauvegardé : {}".format(path))
        except Exception as e:
            print("[!] Grad-CAM échoué : {}".format(e))

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Inférence CLI — Détection & Classification des Stades de Tumeurs")
    parser.add_argument("--image",   type=str, default=None,
                        help="Chemin d'une image unique (.jpg/.png)")
    parser.add_argument("--dir",     type=str, default=None,
                        help="Dossier contenant plusieurs images")
    parser.add_argument("--model",   type=str, default=None,
                        help="Chemin du modèle .keras (auto-détection si omis)")
    parser.add_argument("--gradcam", action="store_true",
                        help="Générer les visualisations Grad-CAM")
    parser.add_argument("--output",  type=str, default=None,
                        help="Fichier CSV de sortie pour le mode --dir")
    args = parser.parse_args()

    if args.image is None and args.dir is None:
        parser.print_help()
        print("\n[!] Spécifiez --image ou --dir.")
        sys.exit(1)

    model_path = args.model or find_best_model()
    if model_path is None or not os.path.exists(model_path):
        print("[!] Aucun modèle trouvé. Entraînez d'abord ou spécifiez --model.")
        sys.exit(1)

    model, num_classes, img_size = load_model(model_path)

    is_tl = (img_size[0] >= 224)
    print("     Type : {} | img_size : {}×{}".format(
                            if is_tl else "CNN Baseline",
        img_size[0], img_size[1]))

    if args.image:
        if not os.path.exists(args.image):
            print("[!] Fichier introuvable : {}".format(args.image))
            sys.exit(1)
        process_image(model, args.image, num_classes, is_tl,
                      save_gradcam=args.gradcam)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print("[!] Dossier introuvable : {}".format(args.dir))
            sys.exit(1)

        patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        images = []
        for p in patterns:
            images.extend(glob.glob(os.path.join(args.dir, p)))
        images = sorted(set(images))

        if not images:
            print("[!] Aucune image trouvée dans : {}".format(args.dir))
            sys.exit(1)

        print("\n[.] {} images trouvées dans '{}'".format(len(images), args.dir))

        rows = []
        labels_map = LABELS_4 if num_classes <= 4 else LABELS_5
        clinical    = CLINICAL_4 if num_classes <= 4 else CLINICAL_5

        for i, img_path in enumerate(images, 1):
            print("\n[{}/{}] {}".format(i, len(images), os.path.basename(img_path)))
            try:
                result = process_image(model, img_path, num_classes, is_tl,
                                       save_gradcam=args.gradcam)
                rows.append({
                          :            os.path.basename(img_path),
                              :        result["stage_id"],
                                 :     result["stage_label"],
                                :      "{:.1%}".format(result["confidence"]),
                                     : "YES" if result["requires_review"] else "no",
                                   :   result["clinical_note"],
                })
            except Exception as e:
                print("[!] Erreur sur {} : {}".format(img_path, e))
                rows.append({
                          :            os.path.basename(img_path),
                              :        -1,
                                 :     "ERROR",
                                :      "0%",
                                     : "YES",
                                   :   str(e),
                })

        print("\n" + "=" * 65)
        print("  RÉCAPITULATIF ({} images)".format(len(rows)))
        print("=" * 65)
        from collections import Counter
        stage_counts = Counter(r["stage_id"] for r in rows if r["stage_id"] >= 0)
        for sid, count in sorted(stage_counts.items()):
            lbl = labels_map.get(sid, str(sid)).split(" - ")[0]
            print("  {:8s} : {:3d} images".format(lbl, count))
        urgent = sum(1 for r in rows if r["requires_review"] == "YES")
        print("  Révision  : {:3d} / {} cas".format(urgent, len(rows)))

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["file", "stage_id", "stage_label",
                                               , "requires_review", "clinical_note"])
                writer.writeheader()
                writer.writerows(rows)
            print("\n[OK] Résultats exportés : {}".format(args.output))
        else:
            print("\n[INFO] Ajoutez --output results.csv pour exporter les résultats.")

if __name__ == "__main__":
    main()
