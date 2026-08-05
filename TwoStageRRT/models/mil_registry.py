"""
MIL Registry — 可插拔MIL分类器注册表

Usage:
    from models.mil_registry import MIL_REGISTRY, register_mil

    @register_mil("abmil")
    class AttentionMIL(nn.Module):
        ...

    # 创建实例
    mil = MIL_REGISTRY.create("abmil", input_dim=512, num_classes=2)

    # 列出所有可用MIL
    print(MIL_REGISTRY.list_available())
"""

import torch.nn as nn


class MILRegistry:
    """MIL模型注册表 — 单例模式"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, name: str, model_class: type):
        """注册MIL模型类"""
        if not issubclass(model_class, nn.Module):
            raise TypeError(
                f"MIL model '{name}' must be a nn.Module subclass, "
                f"got {model_class.__name__}"
            )
        self._registry[name] = model_class

    def get(self, name: str) -> type:
        """获取已注册的MIL模型类"""
        if name not in self._registry:
            available = self.list_available()
            raise KeyError(
                f"MIL type '{name}' not found. "
                f"Available: {available}"
            )
        return self._registry[name]

    def create(self, name: str, **kwargs):
        """创建MIL模型实例"""
        model_class = self.get(name)
        return model_class(**kwargs)

    def list_available(self) -> list:
        """列出所有已注册的MIL模型名称"""
        return sorted(self._registry.keys())

    def is_registered(self, name: str) -> bool:
        """检查某个MIL名称是否已注册"""
        return name in self._registry


# 全局单例
MIL_REGISTRY = MILRegistry()


def register_mil(name: str):
    """装饰器：注册MIL模型到全局注册表

    Usage:
        @register_mil("transmil")
        class TransMIL(nn.Module):
            ...
    """
    def decorator(cls):
        MIL_REGISTRY.register(name, cls)
        return cls
    return decorator
