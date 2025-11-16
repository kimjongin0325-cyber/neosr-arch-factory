# -*- coding: utf-8 -*-
# 완성본 Registry (대소문자 자동 매핑)

class Registry:
    def __init__(self):
        self._dict = {}

    def register(self, name=None):
        def decorator(obj):
            key = name if name else obj.__name__

            # 원본 키
            self._dict[key] = obj
            # 대소문자 자동 매핑
            self._dict[key.lower()] = obj
            self._dict[key.upper()] = obj

            return obj
        return decorator

    def get(self, name):
        if name in self._._dict:
            return self._dict[name]
        if name.lower() in self._dict:
            return self._dict[name.lower()]
        if name.upper() in self._dict:
            return self._dict[name.upper()]
        raise KeyError(f"{name} not found in registry")

ARCH_REGISTRY = Registry()
