"""filter_incomplete enricher: drop samples with missing pseudo_label."""

from multiprocessing import Pool
from typing import Optional

from tqdm import tqdm

from ir.schema import SampleIR
from .base import Enricher, register_enricher


@register_enricher("filter_incomplete_pseudo")
class FilterIncompletePseudoEnricher(Enricher):
    """Remove samples whose pseudo_label is missing or has empty critical_objects.

    reads:  pseudo_label
    writes: - (no new field, just filters)
    """

    reads = ["pseudo_label"]
    writes = []

    def enrich_all(self, samples: list[SampleIR],
                   pool: Optional[Pool] = None) -> list[SampleIR]:
        input_count = len(samples)
        result = [s for s in samples
                  if s.pseudo_label is not None
                  and s.pseudo_label.critical_objects is not None
                  and s.pseudo_label.explanation is not None]
        if input_count > len(result):
            print(f"filter_incomplete: {input_count - len(result)}/{input_count} "
                  f"samples removed")
        return result
