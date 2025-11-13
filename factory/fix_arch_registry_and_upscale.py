import os
import re

def get_dist_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(root, "dist")

def patch_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    new_lines = []
    has_upscale_global = False
    
    for line in lines:
        # 1) ARCH_REGISTRY decorator 제거
        if "@ARCH_REGISTRY" in line:
            continue

        # 2) upscale 전역 사용 탐지
        if "upscale" in line and "self.upscale" not in line:
            has_upscale_global = True

        new_lines.append(line)

    # upscale needed? → global default 삽입
    if has_upscale_global:
        new_lines.insert(0, "upscale = 4  # auto-added by arch factory\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[FIXED] {os.path.basename(path)} (registry/upscale patched)")

def run():
    dist = get_dist_dir()
    for fname in sorted(os.listdir(dist)):
        if fname.endswith("_arch.py"):
            patch_file(os.path.join(dist, fname))

if __name__ == "__main__":
    run()
