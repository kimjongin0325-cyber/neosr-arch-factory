import os, sys

def setup_paths():
    # --- Colab 여부 체크 ---
    try:
        BASE = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE = "/content/neosr-arch-factory"

    ARCH_DIR = os.path.join(BASE, "archs")

    # sys.path 최우선에 삽입 (append 절대 금지)
    sys.path.insert(0, BASE)
    sys.path.insert(0, ARCH_DIR)

    return BASE, ARCH_DIR
