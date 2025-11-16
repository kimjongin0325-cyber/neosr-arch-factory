import os, sys

BASE = "/content/neosr-arch-factory"
ARCH_DIR = os.path.join(BASE, "archs")

# registry.py 와 arch/*.py 들을 최상위 모듈로 취급
if BASE not in sys.path:
    sys.path.append(BASE)
if ARCH_DIR not in sys.path:
    sys.path.append(ARCH_DIR)

from registry import ARCH_REGISTRY

ARCH_REGISTRY._dict.clear()

for file in os.listdir(ARCH_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]
        __import__(module_name)

print(ARCH_REGISTRY._dict.keys())
