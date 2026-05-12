"""
remap_dataset.py
Restructure le dataset Brain Tumor MRI (4 classes) en 5 dossiers stades (0 à IV)

Dataset disponible :
  data/Training/{glioma, meningioma, notumor, pituitary}/
  data/Testing/{glioma, meningioma, notumor, pituitary}/

Mapping WHO :
  notumor    -> stage_0  (contrôle négatif)
  meningioma -> stage_1  (WHO Grade I - bénin)
  pituitary  -> stage_2  (extension locale)
  glioma 50% -> stage_3  (WHO Grade III - anaplasique)
  glioma 50% -> stage_4  (WHO Grade IV - glioblastome GBM)
"""

import os
import shutil
import glob
import random

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE_DIR = "data"
SPLITS   = ["Training", "Testing"]
DATA_OUT = os.path.join("data", "staged")
STAGES   = ["stage_0", "stage_1", "stage_2", "stage_3", "stage_4"]

MAPPING = {
    "notumor":    "stage_0",
    "meningioma": "stage_1",
    "pituitary":  "stage_2",
}

SEED = 42


def create_dirs():
    for stage in STAGES:
        os.makedirs(os.path.join(DATA_OUT, stage), exist_ok=True)
    print("[✓] Dossiers staged/ créés.")


def copy_mapped_classes():
    total = 0
    for src_class, dst_stage in MAPPING.items():
        count = 0
        for split in SPLITS:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                pattern = os.path.join(BASE_DIR, split, src_class, ext)
                for img in glob.glob(pattern):
                    fname = f"{split}_{os.path.basename(img)}"
                    shutil.copy(img, os.path.join(DATA_OUT, dst_stage, fname))
                    count += 1
        print(f"  {src_class:15s} -> {dst_stage}  ({count} images)")
        total += count
    return total


def split_glioma():
    all_glioma = []
    for split in SPLITS:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            pattern = os.path.join(BASE_DIR, split, "glioma", ext)
            for img in sorted(glob.glob(pattern)):
                all_glioma.append((split, img))

    random.seed(SEED)
    random.shuffle(all_glioma)
    mid = len(all_glioma) // 2

    count3, count4 = 0, 0
    for split, img in all_glioma[:mid]:
        fname = f"{split}_{os.path.basename(img)}"
        shutil.copy(img, os.path.join(DATA_OUT, "stage_3", fname))
        count3 += 1
    for split, img in all_glioma[mid:]:
        fname = f"{split}_{os.path.basename(img)}"
        shutil.copy(img, os.path.join(DATA_OUT, "stage_4", fname))
        count4 += 1

    print(f"  {'glioma (50%)':15s} -> stage_3  ({count3} images)")
    print(f"  {'glioma (50%)':15s} -> stage_4  ({count4} images)")
    return count3 + count4


def print_summary():
    print("\n─── Résumé du Dataset Remappé ───────────────────────────────")
    total = 0
    for stage in STAGES:
        path = os.path.join(DATA_OUT, stage)
        n = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        bar = "█" * (n // 50)
        print(f"  {stage}/  :  {n:5d} images  {bar}")
        total += n
    print(f"  {'TOTAL':10s}:  {total:5d} images")
    print("─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    print("=" * 60)
    print("  REMAPPING : 4 classes → 5 stades WHO")
    print("=" * 60)

    create_dirs()
    print("\n[•] Classes directes (stades 0, 1, 2) ...")
    copy_mapped_classes()
    print("\n[•] Subdivision glioma (stades 3 et 4) ...")
    split_glioma()
    print_summary()
    print("\n[✓] Remapping terminé !")
