"""PromptBuilder base class and registry."""

from abc import ABC, abstractmethod

from ir.schema import SampleIR

_PROMPTS: dict[str, type["PromptBuilder"]] = {}


def register_prompt(name: str):
    """Decorator to register a PromptBuilder."""
    def wrapper(cls):
        cls.name = name
        _PROMPTS[name] = cls
        return cls
    return wrapper


def get_prompt(name: str) -> "PromptBuilder":
    """Instantiate a registered PromptBuilder by name."""
    if name not in _PROMPTS:
        raise KeyError(f"Unknown prompt '{name}'. Registered: {list(_PROMPTS.keys())}")
    return _PROMPTS[name]()


class PromptBuilder(ABC):
    """Base class for prompt builders.

    Each builder produces a llamafactory-compatible dict:
    {id, image, system, conversations}
    """

    name: str

    @abstractmethod
    def build(self, sample: SampleIR) -> dict:
        """Build prompt from a single SampleIR."""
        ...
