import os
import sys

# BASE: import_test.py 가 위치한 폴더
BASE = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(BASE, "archs")

# registry.py 및 arch/*.py 들을 top-level에서 import 가능하게 설정
if BASE not in sys.path:
    sys.path.append(BASE)
if ARCH_DIR not in sys.path:
    sys.path.append(ARCH_DIR)

from registry import ARCH_REGISTRY

print("=== ARCH REGISTRY REBUILD START ===")

# 기존 registry 초기화
ARCH_REGISTRY._dict.clear()

# archs 폴더 내 모든 .py 파일 import → @ARCH_REGISTRY.register 실행됨
loaded = []
failed = []

for file in os.listdir(ARCH_DIR):
    if file.endswith(".py") and file != "__init__.py":
        module_name = file[:-3]  # "dat_arch" 형태
        try:
            __import__(module_name)
            loaded.append(module_name)
        except Exception as e:
            failed.append((module_name, str(e).split("\n")[0]))

# 결과 출력
print(f"Imported modules: {len(loaded)}")
for m in loaded:
    print(f"  ✔ {m}")

if failed:
    print("\nFailed imports:")
    for name, msg in failed:
        print(f"  ❌ {name} → {msg}")

print("\nRegistered models:")
print(list(ARCH_REGISTRY._dict.keys()))

print("=== ARCH REGISTRY REBUILD END ===")
