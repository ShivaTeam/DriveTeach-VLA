"""Dataset adapter registry."""

from .base import BaseDatasetAdapter

_ADAPTERS: dict[str, type[BaseDatasetAdapter]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset adapter class."""
    def wrapper(cls):
        cls.name = name
        _ADAPTERS[name] = cls
        return cls
    return wrapper


def get_adapter_class(name: str) -> type[BaseDatasetAdapter]:
    """Get a registered dataset adapter class by name."""
    if name not in _ADAPTERS:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Registered: {list(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[name]
