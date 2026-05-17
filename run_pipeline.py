import os
import sys
import argparse
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
PYTHON   = sys.executable

def run(cmd, label=""):
                                                         
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
                                                                  ],
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

    if not args.skip_remap:
        remap_4 = os.path.join(SRC_DIR, "remap_dataset_4classes.py")
        if os.path.exists(remap_4):
            run([PYTHON, remap_4], "ETAPE 1 -- Remapping 4 classes (staged_4classes/)")
        else:
            run([PYTHON, os.path.join(SRC_DIR, "remap_dataset.py")],
                                                        )
    else:
        print("\n[SKIP] Remapping.")

    if args.tl_only:
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes_tl.py"),
                         , args.backbone,
                              , str(args.tl_p1_epochs),
                              , str(args.tl_p2_epochs),
                     , args.loss],
                                                         .format(args.backbone))
    elif args.baseline_only:
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes.py"),
                     , args.loss, "--epochs", str(args.epochs)],
                                                           .format(args.epochs))
    else:

        run([PYTHON, os.path.join(SRC_DIR, "train_4classes.py"),
                     , args.loss, "--epochs", str(args.epochs)],
                                                )
        run([PYTHON, os.path.join(SRC_DIR, "train_4classes_tl.py"),
                         , args.backbone,
                              , str(args.tl_p1_epochs),
                              , str(args.tl_p2_epochs),
                     , args.loss],
                                                          .format(args.backbone))

    ae_script = os.path.join(SRC_DIR, "autoencoder.py")
    if not args.skip_ae and os.path.exists(ae_script):
        run([PYTHON, ae_script, "--train", "--epochs", "30"],
                                                            )
    else:
        print("\n[SKIP] Auto-encodeur.")

    best_model = find_best_model()
    if best_model:
        run([PYTHON, os.path.join(SRC_DIR, "evaluate.py"), "--model", best_model],
                                                                            )
    else:
        print("\n[WARN] Aucun modele trouve pour l'evaluation.")

    if not args.skip_gradcam and best_model:
        run([PYTHON, os.path.join(SRC_DIR, "gradcam.py"),
                      , best_model, "--samples", str(args.gradcam_samples)],
                                                            .format(
                args.gradcam_samples))

    print("\n" + "=" * 60)
    print("  PIPELINE TERMINE")
    print("=" * 60)
    artifacts = [
                                           .format(args.backbone),
                                        ,
                                           .format(args.backbone),
                  ,
                          ,
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
