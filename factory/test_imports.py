import os
import sys
import importlib.util
import traceback

# ------------------------
# sys.path 에 factory 상위 경로 추가 (중요)
# ------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("[INFO] sys.path 등록 완료:", ROOT)

def get_dist_dir():
    return os.path.join(ROOT, "dist")

def test_all_imports():
    dist = get_dist_dir()
    print(f"[INFO] 테스트 대상: {dist}")

    results = []

    for fname in sorted(os.listdir(dist)):
        if not fname.endswith("_arch.py"):
            continue

        path = os.path.join(dist, fname)
        modname = fname.replace(".py", "")

        try:
            spec = importlib.util.spec_from_file_location(modname, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            print(f"✔ OK: {fname}")
            results.append(f"OK: {fname}")

        except Exception as e:
            print(f"❌ FAIL: {fname} — {e}")
            results.append(f"FAIL: {fname} — {e}")
            traceback.print_exc()

    out = os.path.join(ROOT, "standalone_import_results.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\n[INFO] 테스트 결과 저장됨: {out}")

if __name__ == "__main__":
    test_all_imports()
