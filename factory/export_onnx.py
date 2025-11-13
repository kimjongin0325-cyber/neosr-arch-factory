import os
import sys
import importlib.util
import torch
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DIST = os.path.join(ROOT, "dist")
ONNX_DIR = os.path.join(ROOT, "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

def extract_class(module):
    """arch.py 내부에서 nn.Module 클래스를 자동으로 찾는다."""
    import torch.nn as nn
    for name, obj in module.__dict__.items():
        try:
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                return name, obj
        except:
            pass
    return None, None

def export_single_arch(path, filename):
    modname = filename.replace(".py", "")

    try:
        # module load
        spec = importlib.util.spec_from_file_location(modname, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # class 찾기
        cls_name, cls_obj = extract_class(module)
        if cls_obj is None:
            print(f"❌ NO MODEL CLASS: {filename}")
            return False

        print(f"[INFO] Exporting {cls_name} from {filename}")

        # 모델 생성
        try:
            model = cls_obj().eval()
        except Exception as e:
            print(f"❌ INIT FAIL: {filename} — {e}")
            return False

        # 기본 dummy input
        dummy = torch.randn(1, 3, 256, 256)

        outpath = os.path.join(ONNX_DIR, f"{cls_name}.onnx")

        # ONNX export
        torch.onnx.export(
            model,
            dummy,
            outpath,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "N", 2: "H", 3: "W"},
                "output": {0: "N", 2: "H", 3: "W"}
            }
        )

        print(f"✔ ONNX SAVED: {outpath}")
        return True

    except Exception as e:
        print(f"❌ EXPORT FAIL: {filename}")
        traceback.print_exc()
        return False

def main():
    print("[INFO] ONNX 변환 시작")
    for fname in sorted(os.listdir(DIST)):
        if fname.endswith("_arch.py"):
            export_single_arch(os.path.join(DIST, fname), fname)

    print("\n[INFO] 변환 완료!")

if __name__ == "__main__":
    main()
