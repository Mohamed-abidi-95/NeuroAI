import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 60)
print("  VÉRIFICATION DE L'ENVIRONNEMENT")
print("=" * 60)

print(f"\n[•] Python : {sys.version}")

deps = {
                : "TensorFlow/Keras",
         :        "OpenCV",
             :    "Scikit-Learn",
                : "Matplotlib",
             :    "Seaborn",
           :      "NumPy",
         :        "Pillow",
             :    "FastAPI",
               :  "Streamlit",
            :     "Pytest",
              :   "Requests",
}

all_ok = True
for module, name in deps.items():
    try:
        m = __import__(module)
        version = getattr(m, "__version__", "?")
        print(f"  ✅ {name:20s} v{version}")
    except ImportError:
        print(f"  ❌ {name:20s} MANQUANT → pip install {module}")
        all_ok = False

print("\n[•] GPU :")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            print(f"  ✅ {gpu.name}")
    else:
        print("  ⚠️  Aucun GPU détecté — entraînement sur CPU (ou utilisez Google Colab)")
except Exception as e:
    print(f"  ❌ Erreur TF : {e}")

print("\n[•] Dataset (4 classes) :")
staged_4_path = os.path.join("data", "staged_4classes")
training_path = os.path.join("data", "Training")
testing_path  = os.path.join("data", "Testing")

if os.path.exists(staged_4_path):
    stages = [d for d in os.listdir(staged_4_path)
              if os.path.isdir(os.path.join(staged_4_path, d))]
    if len(stages) == 4:
        total = 0
        for s in sorted(stages):
            n = len(os.listdir(os.path.join(staged_4_path, s)))
            print(f"  ✅ {s}/ : {n} images")
            total += n
        print(f"     Total : {total} images (4 classes)")
    else:
        print(f"  ⚠️  {len(stages)} stades trouvés → Lancez : python src/remap_dataset_4classes.py")
        all_ok = False
elif os.path.exists(training_path):
    print(f"  ℹ️  Données brutes trouvées (Training/) → Lancez : python src/remap_dataset_4classes.py")
else:
    print(f"  ❌ Dataset introuvable → Téléchargez depuis Kaggle :")
    print("     kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset")
    all_ok = False

print("\n[•] Modèles entraînés :")
models_dir = "models"
model_files = [f for f in os.listdir(models_dir) if f.endswith((".keras", ".h5"))]    if os.path.exists(models_dir) else []
if model_files:
    for mf in sorted(model_files):
        size = os.path.getsize(os.path.join(models_dir, mf)) / 1e6
        best = "  ← MEILLEUR" if "tl_4classes_efficientnetb0_final" in mf else ""
        print(f"  ✅ {mf} ({size:.1f} MB){best}")
else:
    print("  ℹ️  Aucun modèle → Lancez : python run_pipeline.py --tl-only")

print("\n[•] Rapports d'évaluation :")
reports_dir = "reports"
if os.path.exists(reports_dir):
    txt_files = [f for f in os.listdir(reports_dir) if f.endswith(".txt")]
    png_files = [f for f in os.listdir(reports_dir) if f.endswith(".png")]
    if txt_files:
        for f in sorted(txt_files):
            print(f"  ✅ {f}")
    else:
        print("  ℹ️  Aucun rapport → Lancez : python src/evaluate.py --model <model_path>")
    if png_files:
        print(f"  ✅ {len(png_files)} visualisation(s) PNG")

print("\n[•] Services :")
print("  API       : cd api && uvicorn main:app --reload --port 8000")
print("  Dashboard : streamlit run dashboard/app.py")
print("  Docker    : cd docker && docker-compose up --build")
print("  Tests     : python -m pytest tests/ -v")

print("\n" + "=" * 60)
if all_ok:
    print("  ✅ Environnement prêt — Vous pouvez commencer l'entraînement !")
    print("  Commande recommandée : python run_pipeline.py --tl-only")
else:
    print("  ⚠️  Des problèmes ont été détectés — Corrigez-les avant de continuer.")
print("=" * 60)
