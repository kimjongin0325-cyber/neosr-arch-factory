#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import argparse
from registry import ARCH_REGISTRY


# --------------------------------------------------------
# 1) 모델이 가진 window, patch, upscale 그대로 읽기
# --------------------------------------------------------
def get_model_constraints(model):
    # window_size
    if hasattr(model, "window_size"):
        window = int(model.window_size)
    elif hasattr(model, "window"):
        window = int(model.window)
    else:
        window = 1  # fallback (거의 없음)

    # patch size
    if hasattr(model, "patch_size"):
        patch = int(model.patch_size)
    elif hasattr(model, "patch"):
        patch = int(model.patch)
    else:
        patch = 1

    # upscale
    upscale = getattr(model, "upscale", 1)

    return {
        "window": window,
        "patch": patch,
        "upscale": upscale,
    }


# --------------------------------------------------------
# 2) 입력 크기를 자동으로 윈도우/패치에 맞게 정렬
# --------------------------------------------------------
def auto_align_size(model_constraints, size):
    w = model_constraints["window"]
    p = model_constraints["patch"]

    # patch, window 모두 고려한 최소 단위
    base = max(w, p)

    # size를 base 배수로 정렬
    aligned = ((size + base - 1) // base) * base
    return aligned


# --------------------------------------------------------
# 3) 더미 입력 자동 생성 (각 모델별 구조를 따름)
# --------------------------------------------------------
def make_dummy_input(model, user_size):
    rules = get_model_constraints(model)

    # 모델이 요구하는 최소 단위에 맞게 자동 정렬
    h = auto_align_size(rules, user_size)
    w = auto_align_size(rules, user_size)

    print(f"\n🔹 Dummy Input 자동 조정됨: {h} × {w}")
    print(f"   (window={rules['window']}, patch={rules['patch']}, upscale={rules['upscale']})")

    return torch.randn(1, 3, h, w).to(next(model.parameters()).device)


# --------------------------------------------------------
# 4) ONNX EXPORT
# --------------------------------------------------------
def export_onnx(model_name, output_path, input_size, opset):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n=== MODEL LOAD: {model_name} ===")
    model = ARCH_REGISTRY.get(model_name)().to(device)
    model.eval()

    # dummy input 자동 계산
    dummy = make_dummy_input(model, input_size)

    print("\n=== Exporting ONNX ===")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        verbose=False,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "h", 3: "w"},
            "output": {2: "H", 3: "W"},
        },
    )

    print(f"\n🎉 Export 완료: {output_path}\n")


# --------------------------------------------------------
# 5) CLI
# --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="모델 이름 (예: ATD)")
    parser.add_argument("--output", type=str, default="model.onnx")
    parser.add_argument("--size", type=int, default=128, help="입력 크기(자동 정렬됨)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(args.model, args.output, args.size, args.opset)
