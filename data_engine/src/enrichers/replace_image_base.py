"""replace_image_base enricher: replace image base path (in-place, unlike inject_bbox_path)."""

from multiprocessing import Pool
from typing import Optional

from tqdm import tqdm

from ir.schema import SampleIR
from .base import Enricher, register_enricher


@register_enricher("replace_image_base")
class ReplaceImageBaseEnricher(Enricher):
    """Replace base path prefix in SampleIR.images.

    Unlike inject_bbox_path, this modifies the primary image paths directly.

    reads:  images
    writes: images
    """

    reads = ["images"]
    writes = ["images"]

    def __init__(self, old_base: str, new_base: str):
        self.old_base = old_base
        self.new_base = new_base

    def enrich_all(self, samples: list[SampleIR],
                   pool: Optional[Pool] = None) -> list[SampleIR]:
        for s in tqdm(samples, desc="replace_image_base"):
            s.images = [
                p.replace(self.old_base, self.new_base) for p in s.images
            ]
        return samples
