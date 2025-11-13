import os

def find_neosr_root():
    """
    기본 가정:
    - 이 스크립트는 /content/neosr-arch-factory 안에서 실행됨
    - 원본 neosr 리포는 ../neosr 에 위치함
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    neosr_path = os.path.join(root, "..", "neosr")
    return neosr_path

def list_arch_files():
    neosr_root = find_neosr_root()
    arch_dir = os.path.join(neosr_root, "neosr", "archs")
    print(f"[INFO] neosr root: {neosr_root}")
    print(f"[INFO] arch dir : {arch_dir}")

    if not os.path.isdir(arch_dir):
        print("[ERROR] arch 디렉토리가 존재하지 않습니다.")
        return

    print("\n[ARCH FILES]")
    for fname in sorted(os.listdir(arch_dir)):
        if fname.endswith("_arch.py"):
            print(" -", fname)

if __name__ == "__main__":
    list_arch_files()
