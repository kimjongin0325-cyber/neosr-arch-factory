import os
import importlib
from config_paths import setup_paths
from registry import ARCH_REGISTRY

BASE, ARCH_DIR = setup_paths()

print("=== ARCH REGISTRY REBUILD START ===")

ARCH_REGISTRY._dict.clear()

loaded = []
failed = []

for file in os.listdir(ARCH_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        try:
            importlib.import_module(module_name)
            loaded.append(module_name)
        except Exception as e:
            failed.append((module_name, str(e).split("\n")[0]))

print(f"Imported modules: {len(loaded)}")
for m in loaded:
    print("  ✔", m)

print("\nRegistered models:")
print(list(ARCH_REGISTRY._dict.keys()))

print("=== ARCH REGISTRY REBUILD END ===")
