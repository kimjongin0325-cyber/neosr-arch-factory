class Registry:
    def __init__(self):
        self._dict = {}

    def register(self, name=None):
        def decorator(obj):
            # 키를 항상 소문자로 정규화
            key = (name if name else obj.__name__).lower()
            self._dict[key] = obj
            return obj
        return decorator

    def get(self, name):
        # 입력 이름도 소문자로 변환해서 찾음
        key = name.lower()
        if key not in self._dict:
            raise KeyError(f"{name} not found in registry")
        return self._dict[key]


ARCH_REGISTRY = Registry()

