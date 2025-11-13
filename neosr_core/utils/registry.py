# neosr_core/utils/registry.py

class Registry:
    """Simple registry for architecture and other modules."""

    def __init__(self, name: str):
        self._name = name
        self._registry = {}

    def register(self, name: str = None):
        """Decorator to register a function or class."""
        def decorator(func_or_class):
            key = name if name is not None else func_or_class.__name__
            if key in self._registry:
                raise KeyError(f"{self._name} registry: '{key}' already registered!")

            self._registry[key] = func_or_class
            return func_or_class

        return decorator

    def get(self, name: str):
        if name not in self._registry:
            raise KeyError(f"{self._name}: '{name}' not found")
        return self._registry[name]

    def list(self):
        return list(self._registry.keys())


# Global registry for arch modules
ARCH_REGISTRY = Registry("arch")
