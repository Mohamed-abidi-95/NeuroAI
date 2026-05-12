"""
run_pipeline.py
Script maitre - Lance le pipeline complet en une commande.

Usage :
    python run_pipeline.py                         # Pipeline complet (4 classes TL)
    python run_pipeline.py --skip-remap            # Si staged_4classes/ deja pret
    python run_pipeline.py --baseline-only         # CNN baseline uniquement
    python run_pipeline.py --tl-only               # Transfer Learning 4 classes uniquement
    python run_pipeline.py --epochs 30             # Moins d'epoques pour test rapide
"""

import os
import sys
import argparse
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
PYTHON   = sys.executable


def run(cmd, label=""):
    """Execute une commande et affiche la progression."""
    print("\n" + "=" * 60)
    print("  >>  {}".format(label))
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print("\n[!] ERREUR dans : {} (code {})".format(label, result.returncode))
        sys.exit(result.returncode)
    print("\n[OK] {} -- {:.1f}s".format(label, elapsed))
    return result


def find_best_model():
    """Retourne le meilleur modele disponible (priorite TL > CNN)."""
    candidates = [
        os.path.join(BASE_DIR, "models", "tl_4classes_efficientnetb0_final.keras"),
        os.path.join(BASE_DIR, "models", "tl_4classes_resnet50v2_final.keras"),
        os.path.join(BASE_DIR, "models", "cnn_4classes_best.keras"),
        os.path.join(BASE_DIR, "models", "cnn_4classes_final.keras"),
        os.path.join(BASE_DIR, "models", "tl_efficientnetb0_final.keras"),
        os.path.join(BASE_DIR, "models", "cnn_baseline_best.keras"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline complet Detection Stades Tumoraux (4 classes)")
    parser.add_argument("--skip-remap",    action="store_true",
                        help="Sauter le remapping (staged_4classes/ deja pret)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Entrainer uniquement le CNN baseline 4 classes")
    parser.add_argument("--tl-only",       action="store_true",
                        help="Lancer uniquement le Transfer Learning 4 classes")
    parser.add_argument("--epochs",        type=int, default=80)
    parser.add_argument("--tl-p1-epochs",  type=int, default=30)
    parser.add_argument("--tl-p2-epochs",  type=int, default=25)
    parser.add_argument("--loss",          choices=["ce", "focal"], default="focal")
    parser.add_argument("--backbone",      choices=["efficientnetb0", "resnet50v2",
                                                     "mobilenetv2"],
                        default="efficientnetb0")
    parser.add_argument("--skip-ae",       action="store_true")
    parser.add_argument("--skip-gradcam",  action="store_true")
    parser.add_argument("--gradcam-samples", type=int, default=5)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLET -- DETECTION STADES TUMORAUX (4 classes)")
    print("  CNN Baseline + Transfer Learning EfficientNetB0")
    print("=" * 60)
    print("  Python   : {}".format(PYTHON))
    print("  Base     : {}".format(BASE_DIR))
    print("  Backbone : {}".format(args.backbone))
    print("  Loss     : {}".format(args.loss))

    # ETAPE 1 : Remapping
    if not args.skip_remap:
        remap_4 = os.path.join(SRC_DIR, "remap_dataset_4classes.py")
        if os.path.exists(remap_4):
            run([PYTHON, remap_4], "ETAPE 1 -- Remapping 4 classes (staged_4classes/)")
        else:
            run([PYTHON, os.path.join(SRC_DIR, "remap_dataset.py")],
                "ETAPE 1 -- Remapping dataset (staged/)")
    else:
        print("\n[SKIP] Remapping.")

    # ETAPE 2 : Entrainement
    if args.tl_only:
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes_tl.py"),
             "--backbone", args.backbone,
             "--phase1-epochs", str(args.tl_p1_epochs),
             "--phase2-epochs", str(args.tl_p2_epochs),
             "--loss", args.loss],
            "ETAPE 2 -- Transfer Learning 4 classes ({})".format(args.backbone))
    elif args.baseline_only:
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes.py"),
             "--loss", args.loss, "--epochs", str(args.epochs)],
            "ETAPE 2 -- CNN Baseline 4 classes ({} epochs)".format(args.epochs))
    else:
        # CNN baseline d'abord, puis TL
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes.py"),
             "--loss", args.loss, "--epochs", str(args.epochs)],
            "ETAPE 2a -- CNN Baseline 4 classes")
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes_tl.py"),
             "--backbone", args.backbone,
             "--phase1-epochs", str(args.tl_p1_epochs),
             "--phase2-epochs", str(args.tl_p2_epochs),
             "--loss", args.loss],
            "ETAPE 2b -- Transfer Learning 4 classes ({})".format(args.backbone))

    # ETAPE 3 : Auto-encodeur (optionnel)
    ae_script = os.path.join(SRC_DIR, "autoencoder.py")
    if not args.skip_ae and os.path.exists(ae_script):
        run([PYTHON, ae_script, "--train", "--epochs", "30"],
            "ETAPE 3 -- Auto-Encodeur (detection anomalies)")
    else:
        print("\n[SKIP] Auto-encodeur.")

    # ETAPE 4 : Evaluation
    best_model = find_best_model()
    if best_model:
        run([PYTHON, os.path.join(SRC_DIR, "evaluate.py"), "--model", best_model],
            "ETAPE 4 -- Evaluation Clinique (Accuracy, Recall, F2, ROC-AUC)")
    else:
        print("\n[WARN] Aucun modele trouve pour l'evaluation.")

    # ETAPE 5 : Grad-CAM
    if not args.skip_gradcam and best_model:
        run([PYTHON, os.path.join(SRC_DIR, "gradcam.py"),
             "--model", best_model, "--samples", str(args.gradcam_samples)],
            "ETAPE 5 -- Visualisations Grad-CAM ({} images)".format(
                args.gradcam_samples))

    # Résumé final
    print("\n" + "=" * 60)
    print("  PIPELINE TERMINE")
    print("=" * 60)
    artifacts = [
        "models/tl_4classes_{}_final.keras".format(args.backbone),
        "models/cnn_4classes_best.keras",
        "models/tl_4classes_{}_history.png".format(args.backbone),
        "reports/",
        "reports/gradcam/",
    ]
    for path in artifacts:
        full = os.path.join(BASE_DIR, path)
        mark = "[OK]" if os.path.exists(full) else "[ ] "
        print("  {}  {}".format(mark, path))

    print("\n  Demarrer les services :")
    print("  API       : cd api && uvicorn main:app --reload --port 8000")
    print("  Dashboard : streamlit run dashboard/app.py")
    print("  Docker    : cd docker && docker-compose up --build\n")


if __name__ == "__main__":
    main()



