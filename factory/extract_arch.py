import os
import shutil

def find_neosr_arch_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    arch_dir = os.path.join(root, "..", "neosr", "neosr", "archs")
    return arch_dir

def ensure_dist():
    here = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(os.path.abspath(os.path.join(here, "..")), "dist")
    os.makedirs(dist_dir, exist_ok=True)
    return dist_dir

def copy_arch_file(src_file, dist_dir):
    dst = os.path.join(dist_dir, os.path.basename(src_file))
    shutil.copy(src_file, dst)
    print(f"[COPIED] {src_file} -> {dst}")

def extract_all_archs():
    arch_dir = find_neosr_arch_dir()
    dist_dir = ensure_dist()

    print(f"[INFO] arch source : {arch_dir}")
    print(f"[INFO] dist target : {dist_dir}")

    for fname in sorted(os.listdir(arch_dir)):
        if fname.endswith("_arch.py"):
            src = os.path.join(arch_dir, fname)
            copy_arch_file(src, dist_dir)

if __name__ == "__main__":
    extract_all_archs()
