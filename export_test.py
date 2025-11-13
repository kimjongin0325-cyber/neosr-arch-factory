import sys
sys.path.append("/content/neosr-arch-factory")

import torch
import importlib
import os

ARCH_FILE = "realplksr_arch"
CLASS_NAME = "realplksr"

MODULE = f"neosr_core.models.{ARCH_FILE}"
print(f"[TEST] Loading module: {MODULE}")

try:
    mod = importlib.import_module(MODULE)

    if not hasattr(mod, CLASS_NAME):
        raise RuntimeError(f"Class '{CLASS_NAME}' not found in {MODULE}")

    ModelClass = getattr(mod, CLASS_NAME)

    print(f"[TEST] Using class: {CLASS_NAME}")

    # Dummy input (CPU)
    dummy = torch.randn(1, 3, 32, 32)
    model = ModelClass().eval()

    out_dir = "/content/neosr-arch-factory/onnx_test"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{ARCH_FILE}.onnx"

    print("[TEST] Exporting to ONNX...")

    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print("✔ SUCCESS! ONNX exported:", out_path)

except Exception as e:
    print("❌ FAILED")
    print(e)
