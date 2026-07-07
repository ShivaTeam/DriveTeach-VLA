"""merge_pseudo_labels enricher: read JSONL and parse VLM outputs."""

import json

from tqdm import tqdm

from ir.schema import SampleIR, PseudoLabelIR
from utils.json_repair import extract_clean_json
from .base import Enricher, register_enricher


_PARSERS: dict[str, type["PseudoLabelParser"]] = {}


def register_parser(name: str):
    def wrapper(cls):
        cls.name = name
        _PARSERS[name] = cls
        return cls
    return wrapper


class PseudoLabelParser:
    name: str
    def parse(self, data: dict, sample: SampleIR) -> dict:
        raise NotImplementedError


@register_parser("poutine")
class PoutineParser(PseudoLabelParser):
    name = "poutine"

    def parse(self, data: dict, sample: SampleIR) -> dict:
        raw = data.get("predict", "")
        obj = extract_clean_json(raw, repair=True)
        if obj is None:
            return {}
        return {
            "critical_objects": obj.get("critical_objects"),
            "meta_behaviour": obj.get("meta_behaviour"),
            "explanation": obj.get("explanation"),
        }


@register_enricher("merge_pseudo_labels")
class MergePseudoLabelsEnricher(Enricher):
    reads = []
    writes = ["pseudo_label"]

    def __init__(self, source_path: str, parser: str = "poutine",
                 key_field: str = "id", json_repair: bool = True):
        self.source_path = source_path
        self.key_field = key_field
        self.json_repair = json_repair
        parser_cls = _PARSERS.get(parser)
        if parser_cls is None:
            raise KeyError(f"Unknown parser '{parser}'. Registered: {list(_PARSERS.keys())}")
        self.parser = parser_cls()

    def enrich_all(self, samples: list[SampleIR], pool=None) -> list[SampleIR]:
        # Load + parse JSONL in one pass (fast, no pickle overhead)
        index = {}
        with open(self.source_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading pseudo labels"):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                pid = record.get(self.key_field, record.get("id", ""))
                if not pid:
                    continue
                parsed = self.parser.parse(record, None)
                if parsed:
                    index[pid] = parsed

        # Apply to samples
        for s in tqdm(samples, desc="merge_pseudo_labels"):
            parsed = index.get(s.sample_id)
            if not parsed:
                continue
            s.pseudo_label = PseudoLabelIR(
                critical_objects=parsed.get("critical_objects"),
                explanation=parsed.get("explanation"),
                meta_behaviour=parsed.get("meta_behaviour"),
            )
        return samples
