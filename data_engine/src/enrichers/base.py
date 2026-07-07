"""Enricher base class with reads/writes declarations and registry."""

from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import Optional

from ir.schema import SampleIR

_ENRICHERS: dict[str, type["Enricher"]] = {}


def register_enricher(name: str):
    """Decorator to register an enricher class."""
    def wrapper(cls):
        cls.name = name
        _ENRICHERS[name] = cls
        return cls
    return wrapper


def get_enricher(name: str, params: Optional[dict] = None) -> "Enricher":
    """Instantiate a registered enricher by name with optional params."""
    if name not in _ENRICHERS:
        raise KeyError(f"Unknown enricher '{name}'. Registered: {list(_ENRICHERS.keys())}")
    if params:
        return _ENRICHERS[name](**params)
    return _ENRICHERS[name]()


class Enricher(ABC):
    """Base class for all enrichers.

    An enricher reads IR fields, computes something, and writes new IR fields.
    Internal implementation (single-pass, two-pass, N-pass, cached) is opaque
    to the pipeline runner.

    reads/writes: top-level SampleIR field names for startup validation.
    """

    reads: list[str] = []
    writes: list[str] = []

    @abstractmethod
    def enrich_all(self, samples: list[SampleIR],
                   pool: Optional[Pool] = None) -> list[SampleIR]:
        """Process all samples and return the (possibly filtered) list."""
        ...
