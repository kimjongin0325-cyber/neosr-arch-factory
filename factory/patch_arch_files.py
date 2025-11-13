import os
import re

def get_dist_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    return os.path.join(root, "dist")

def patch_arch_file(path):
    new_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        # 1) registry 제거
        if "from neosr.utils.registry" in line:
            continue

        # 2) arch_util import 제거 (clean_imports에서 1차 처리됨)
        if "from neosr.archs.arch_util" in line:
            continue

        # 3) neosr.utils.* 제거
        if "from neosr.utils" in line:
            continue

        # 4) neosr.losses.* 제거
        if "from neosr.losses" in line:
            continue

        # 5) neosr.metrics.* 제거
        if "from neosr.metrics" in line:
            continue

        # 6) 네오SR 옵션/설정 관련 제거
        if "parse_options" in line:
            continue
        if "net_opt" in line:
            continue

        # 7) arch_util 기반 함수 호출 → local_utils로 교체
        # DropPath, to_2tuple, DySample, conv*
        line = re.sub(r"DropPath", "DropPath", line)
        line = re.sub(r"to_2tuple", "to_2tuple", line)
        line = re.sub(r"DySample", "DySample", line)

        new_lines.append(line)

    # 8) 맨 위에 local_utils import 삽입
    import_header = "from factory.local_utils import DropPath, DySample, to_2tuple, conv1x1, conv3x3\n"
    if import_header not in new_lines[0]:
        new_lines.insert(0, import_header)

    # 저장
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[PATCHED] {os.path.basename(path)}")

def patch_all():
    dist_dir = get_dist_dir()
    print(f"[INFO] 패치 대상 디렉토리: {dist_dir}")

    for fname in sorted(os.listdir(dist_dir)):
        if fname.endswith("_arch.py"):
            patch_arch_file(os.path.join(dist_dir, fname))

if __name__ == "__main__":
    patch_all()
