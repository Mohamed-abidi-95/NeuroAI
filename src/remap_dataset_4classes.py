"""
remap_dataset_4classes.py
Remapping optimisé : 4 classes au lieu de 5
- Stage 0 : Contrôle (no tumor)
- Stage I : Bénin (meningioma)
- Stage II : Local (pituitary)
- Stage III : Gliome agressif (tous les gliomes III+IV fusionnés)
"""

import os
import shutil
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
STAGED_DIR = DATA_DIR / "staged_4classes"

# Mapping 4 classes → stades WHO
MAPPING = {
    "notumor": "stage_0",      # Contrôle négatif
    "meningioma": "stage_1",   # Grade I WHO (bénin)
    "pituitary": "stage_2",    # Extension locale
    "glioma": "stage_3",       # Gliome agressif (III+IV fusionnés)
}


def remap_dataset():
    print("=" * 60)
    print("  REMAPPING OPTIMISÉ : 4 classes (fusion III+IV)")
    print("=" * 60)

    # Nettoyer staged_4classes
    if STAGED_DIR.exists():
        shutil.rmtree(STAGED_DIR)
    
    STAGED_DIR.mkdir(parents=True)
    for stage in ["stage_0", "stage_1", "stage_2", "stage_3"]:
        (STAGED_DIR / stage).mkdir()
    
    print("[✓] Dossiers staged_4classes/ créés.\n")

    # Copier toutes les images
    total = 0
    for source_dir in [TRAIN_DIR, TEST_DIR]:
        for original_class, target_stage in MAPPING.items():
            src_path = source_dir / original_class
            if not src_path.exists():
                continue
            
            dst_path = STAGED_DIR / target_stage
            
            files = list(src_path.glob("*.jpg"))
            for img_file in files:
                shutil.copy(img_file, dst_path / img_file.name)
            
            total += len(files)
            print(f"  {original_class:15s} -> {target_stage:8s}  ({len(files):4d} images)")
    
    # Résumé
    print("\n" + "─" * 60)
    print("  Résumé du Dataset Remappé (4 classes)")
    print("─" * 60)
    
    max_count = 0
    for stage in ["stage_0", "stage_1", "stage_2", "stage_3"]:
        count = len(list((STAGED_DIR / stage).glob("*.jpg")))
        max_count = max(max_count, count)
    
    for stage in ["stage_0", "stage_1", "stage_2", "stage_3"]:
        count = len(list((STAGED_DIR / stage).glob("*.jpg")))
        bar = "█" * int(36 * count / max_count)
        print(f"  {stage}/  :   {count:4d} images  {bar}")
    
    print(f"  TOTAL     :   {total:4d} images")
    print("─" * 60)
    print("\n[✓] Remapping 4 classes terminé !")
    print(f"[INFO] Dataset disponible dans : {STAGED_DIR}")


if __name__ == "__main__":
    remap_dataset()
