#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import importlib
import torch
import argparse
from registry import ARCH_REGISTRY

# =============================
# 기본 경로 설정
# =============================
ROOT = os.path.dirname(os.path.dirname(__file__))
ARCH_DIR = os.path.join(ROOT, "archs")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# archs/__init__.py 보장
init_file = os.path.join(ARCH_DIR, "__init__.py")
if not os.path.exists(init_file):
    with open(init_file, "w") as f:
        pass

# =============================
# 모든 _arch.py 자동 import
# =============================
for file in os.listdir(ARCH_DIR):
    if file.endswith("_arch.py"):
        module = file[:-3]
        importlib.import_module(f"archs.{module}")

# =============================
# 모델 제약 조건 자동 추출
# =============================
def get_model_constraints(model):
    window = getattr(model, "window_size", getattr(model, "window", 1))
    patch = getattr(model, "patch_size", getattr(model, "patch", 1))
    upscale = getattr(model, "upscale", 1)

    return {
        "window": int(window),
        "patch": int(patch),
        "upscale": int(upscale),
    }

def auto_align_size(rules, size):
    base = max(rules["window"], rules["patch"])
    aligned = ((size + base - 1) // base) * base
    return aligned

# =============================
# Dummy Input 자동 생성
# =============================
def make_dummy(model, size):
    rules = get_model_constraints(model)
    h = auto_align_size(rules, size)
    w = auto_align_size(rules, size)

    return torch.randn(1, 3, h, w).to(next(model.parameters()).device)

# =============================
# ONNX EXPORT
# =============================
def export_onnx(model_name, output, size, opset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ARCH_REGISTRY.get(model_name)().to(device)
    model.eval()

    dummy = make_dummy(model, size)

    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "h", 3: "w"},
            "output": {2: "H", 3: "W"},
        },
    )

# =============================
# CLI
# =============================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, default="model.onnx")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    export_onnx(args.model, args.output, args.size, args.opset)
