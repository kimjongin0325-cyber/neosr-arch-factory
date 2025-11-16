import os, sys

BASE = "/content/neosr-arch-factory"
ARCH_DIR = os.path.join(BASE, "archs")

if BASE not in sys.path:
    sys.path.append(BASE)
if ARCH_DIR not in sys.path:
    sys.path.append(ARCH_DIR)

from registry import ARCH_REGISTRY

# registry 초기화
ARCH_REGISTRY._dict.clear()

# arch 파일 import
for file in os.listdir(ARCH_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        __import__(module_name)

print("Loaded Models:", list(ARCH_REGISTRY._dict.keys()))
