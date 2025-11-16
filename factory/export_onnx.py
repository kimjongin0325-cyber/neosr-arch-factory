#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import importlib
import argparse
import torch

# ------------ PATH SETUP ------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH_DIR = os.path.join(ROOT, "archs")

sys.path.insert(0, ROOT)
sys.path.insert(0, ARCH_DIR)

from registry import ARCH_REGISTRY

# archs/__init__.py 보장 (패키지 인식)
if not os.path.exists(os.path.join(ARCH_DIR, "__init__.py")):
    open(os.path.join(ARCH_DIR, "__init__.py"), "w").close()

# ------------ AUTO IMPORT ALL ARCHS ------------
for f in os.listdir(ARCH_DIR):
    if f.endswith("_arch.py"):
        module = f[:-3]
        importlib.import_module(module)  # f"archs.{module}" 가 아니라 module 자체

# ------------ MODEL INFO UTILS ------------
def get_model_constraints(model):
    window = getattr(model, "window_size", getattr(model, "window", 1))
    patch  = getattr(model, "patch_size", getattr(model, "patch", 1))
    upscale = getattr(model, "upscale", 1)
    return {"window": int(window), "patch": int(patch), "upscale": int(upscale)}

def auto_align_size(rules, size):
    base = max(rules["window"], rules["patch"])
    return ((size + base - 1) // base) * base

def make_dummy(model, size):
    rules = get_model_constraints(model)
    h = auto_align_size(rules, size)
    w = auto_align_size(rules, size)
    return torch.randn(1, 3, h, w).to(next(model.parameters()).device)

# ------------ ONNX EXPORT ------------
def export_onnx(model_name, output, size, opset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cls = ARCH_REGISTRY.get(model_name)
    model = model_cls().to(device).eval()

    dummy = make_dummy(model, size)

    torch.onnx.export(
        model,
        dummy,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"},
                      "output": {2: "H", 3: "W"}}
    )

# ------------ CLI ENTRY ------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, default="model.onnx")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    export_onnx(args.model, args.output, args.size, args.opset)
