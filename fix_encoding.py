import os

path = os.path.join("src", "train_4classes_tl.py")

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
            : ">=",
            : "<=",
            : "[OK]",
            : "[OK]",
                  : "[!]",
            : "[!]",
            : "=",
            : "-",
            : "->",
            : "<-",
}

for old, new in replacements.items():
    content = content.replace(old, new)

header = "import sys\nif hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')\n"
if "sys.stdout.reconfigure" not in content:
    content = header + content

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Encodage corrige avec succes !")
