"""Script de correction d'encodage pour train_4classes_tl.py"""
import os

path = os.path.join("src", "train_4classes_tl.py")

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    "\u2265": ">=",
    "\u2264": "<=",
    "\u2714": "[OK]",
    "\u2705": "[OK]",
    "\u26a0\ufe0f": "[!]",
    "\u26a0": "[!]",
    "\u2550": "=",
    "\u2500": "-",
    "\u2192": "->",
    "\u2190": "<-",
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Ajouter reconfiguration encodage au debut
header = "import sys\nif hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')\n"
if "sys.stdout.reconfigure" not in content:
    content = header + content

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Encodage corrige avec succes !")
