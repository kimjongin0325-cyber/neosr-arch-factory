#!/usr/bin/env python3
import os
import sys

# ------------ PATH SETUP ------------
ROOT = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(ROOT, "archs")

# registry.py 와 arch/* 들을 최우선 import 경로로 만든다
sys.path.insert(0, ROOT)
sys.path.insert(0, ARCH_DIR)

from registry import ARCH_REGISTRY

print("=== ARCH REGISTRY REBUILD START ===")

# ------------ RESET REGISTRY ------------
ARCH_REGISTRY._dict.clear()

loaded = []
failed = []

# ------------ IMPORT ALL ARCH MODULES ------------
for file in os.listdir(ARCH_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module = file[:-3]
        try:
            __import__(module)
            loaded.append(module)
        except Exception as e:
            failed.append((module, str(e).split("\n")[0]))

# ------------ LOG RESULT ------------
print(f"Imported modules: {len(loaded)}")
for m in loaded:
    print("  ✔", m)

if failed:
    print("\n❌ Failed imports:")
    for name, msg in failed:
        print(f"  {name} → {msg}")

print("\nRegistered models:")
print(list(ARCH_REGISTRY._dict.keys()))

print("=== ARCH REGISTRY REBUILD END ===")
