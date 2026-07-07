"""inject_bbox_path enricher: replace base path to construct bbox image path."""

from multiprocessing import Pool
from typing import Optional

from tqdm import tqdm

from ir.schema import SampleIR
from .base import Enricher, register_enricher


@register_enricher("inject_bbox_path")
class InjectBboxPathEnricher(Enricher):
    """Replace image base path to inject bbox-augmented image path.

    img_path.replace(old_base, new_base) — preserves subdirectory structure.

    reads:  images
    writes: bbox_image
    """

    reads = ["images"]
    writes = ["bbox_image"]

    def __init__(self, old_base: str, new_base: str):
        self.old_base = old_base
        self.new_base = new_base

    def enrich_all(self, samples: list[SampleIR],
                   pool: Optional[Pool] = None) -> list[SampleIR]:
        for s in tqdm(samples, desc="inject_bbox_path"):
            if s.images:
                s.bbox_image = s.images[0].replace(self.old_base, self.new_base)
        return samples
