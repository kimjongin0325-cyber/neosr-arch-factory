import os
import re

def get_dist_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(root, "dist")

def clean():
    dist_dir = get_dist_dir()
    print(f"[INFO] Cleaning imports in: {dist_dir}")

    for fname in os.listdir(dist_dir):
        if not fname.endswith("_arch.py"):
            continue

        fpath = os.path.join(dist_dir, fname)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            # 1) ARCH_REGISTRY 제거
            if "from neosr.utils.registry" in line:
                continue

            # 2) neosr.archs.arch_util → local utils로 대체 (임시)
            line = line.replace(
                "from neosr.archs.arch_util import",
                "from factory.local_utils import"
            )

            new_lines.append(line)

        # 저장
        with open(fpath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        print(f"[CLEANED] {fname}")

if __name__ == "__main__":
    clean()
