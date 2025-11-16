import os
import re

ARCH_DIR = "/content/neosr-arch-factory/archs"

DECORATOR = "@ARCH_REGISTRY.register()\n"

for file in os.listdir(ARCH_DIR):
    if file.endswith("_arch.py"):
        path = os.path.join(ARCH_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            code = f.readlines()

        # 이미 데코레이터 있으면_skip
        if any("@ARCH_REGISTRY.register" in line for line in code):
            continue
        
        # 클래스 정의 찾기
        new_code = []
        inserted = False

        for line in code:
            new_code.append(line)

            # class XXX(nn.Module):
            if line.strip().startswith("class ") and "(nn.Module)" in line:
                # 바로 위에 데코레이터 삽입
                new_code.insert(len(new_code)-1, DECORATOR)
                inserted = True
        
        if inserted:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_code)
            print(f"✔ patched: {file}")
        else:
            print(f"⚠ no class found in: {file}")
