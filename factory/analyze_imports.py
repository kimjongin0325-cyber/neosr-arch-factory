import os

def get_dist_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    dist_dir = os.path.join(root, "dist")
    return dist_dir

def analyze():
    dist_dir = get_dist_dir()
    if not os.path.isdir(dist_dir):
        print("[ERROR] dist 디렉토리가 없습니다. 먼저 extract_arch.py를 실행하세요.")
        return

    print(f"[INFO] 분석 대상 dist 디렉토리: {dist_dir}")
    out_lines = []

    for fname in sorted(os.listdir(dist_dir)):
        if not fname.endswith("_arch.py"):
            continue
        path = os.path.join(dist_dir, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    out_lines.append(f"{fname}: {stripped}")

    # 화면에도 일부 출력
    print("\n[IMPORT LINES PREVIEW]")
    for line in out_lines[:80]:
        print(line)

    # 전체를 파일로 저장
    root = os.path.abspath(os.path.join(dist_dir, ".."))
    out_path = os.path.join(root, "arch_imports.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"\n[INFO] 전체 import 목록 저장 완료: {out_path}")

if __name__ == "__main__":
    analyze()
