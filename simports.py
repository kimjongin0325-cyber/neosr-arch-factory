import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(BASE, "archs")

# 반드시 append가 아닌 insert(0, …)!!!
sys.path.insert(0, BASE)
sys.path.insert(0, ARCH_DIR)

from registry import ARCH_REGISTRY

print("=== ARCH REGISTRY REBUILD START ===")

ARCH_REGISTRY._dict.clear()

loaded = []
failed = []

for file in os.listdir(ARCH_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        try:
            __import__(module_name)
            loaded.append(module_name)
        except Exception as e:
            failed.append((module_name, str(e).split("\n")[0]))

print(f"Imported modules: {len(loaded)}")
for m in loaded:
    print(f"  ✔ {m}")

print("\nRegistered models:")
print(list(ARCH_REGISTRY._dict.keys()))

print("=== ARCH REGISTRY REBUILD END ===")
