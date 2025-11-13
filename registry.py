class Registry:
    def __init__(self):
        self._dict = {}

    def register(self, name=None):
        def decorator(obj):
            key = name if name else obj.__name__
            self._dict[key] = obj
            return obj
        return decorator

    def get(self, name):
        if name not in self._dict:
            raise KeyError(f"{name} not found in registry")
        return self._dict[name]

ARCH_REGISTRY = Registry()
