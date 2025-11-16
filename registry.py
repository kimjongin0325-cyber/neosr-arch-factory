class Registry:
    def __init__(self):
        self._dict = {}

    def register(self, name=None):
        def decorator(obj):
            key = name if name else obj.__name__
            self._dict[key] = obj
            self._dict[key.lower()] = obj
            self._dict[key.upper()] = obj
            return obj
        return decorator

    def get(self, name):
        if name in self._dict:
            return self._dict[name]
        if name.lower() in self._dict:
            return self._dict[name.lower()]
        if name.upper() in self._dict:
            return self._dict[name.upper()]
        raise KeyError(f"{name} not found in registry")


ARCH_REGISTRY = Registry()
