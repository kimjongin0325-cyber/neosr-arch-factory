import os
import sys

# 현재 파일(import_test.py)이 위치한 디렉토리를 BASE 로 사용
BASE = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(BASE, "archs")

# PYTHONPATH 등록
if BASE not in sys.path:
    sys.path.append(BASE)
if ARCH_DIR not in sys.path:
    sys.path.append(ARCH_DIR)

print("=== IMPORT TEST START ===")

# archs 폴더 내부의 모든 .py 파일 스캔
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
