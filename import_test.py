import os
import sys
import traceback

BASE = "/content/neosr-arch-factory"
ARCH_DIR = os.path.join(BASE, "archs")

if BASE not in sys.path:
    sys.path.append(BASE)
if ARCH_DIR not in sys.path:
    sys.path.append(ARCH_DIR)

print("=== IMPORT TEST START ===")

for file in os.listdir(ARCH_DIR):
    if not file.endswith(".py"):
        continue

    module_name = file.replace(".py", "")
    print(f"{file:<20}", end=" ")

    try:
        __import__(module_name)
        print("✔ OK")
    except Exception as e:
        print("❌ FAILED")
        msg = str(e).split("\n")[0]
        print("   └─", msg)

print("=== IMPORT TEST END ===")
