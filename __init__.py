import os
import sys

# 이 파일이 있는 디렉토리 = 프로젝트 ROOT
ROOT = os.path.dirname(os.path.abspath(__file__))

ARCH_DIR = os.path.join(ROOT, "archs")

# ROOT 및 archs 디렉토리를 sys.path 최상단에 넣기
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if ARCH_DIR not in sys.path:
    sys.path.insert(0, ARCH_DIR)
