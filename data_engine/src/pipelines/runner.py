"""Pipeline orchestrator with reads/writes validation and multiprocessing."""

import os
import pickle
import time
import multiprocessing as mp
from multiprocessing import Pool

from tqdm import tqdm

from ir.builder import build_sample_ir
from datasets.registry import get_adapter_class

# Trigger all @register_* decorators
import datasets.navsim.adapter
import enrichers.normalize_trajectory
import enrichers.project_2d_tgp
import enrichers.inject_bbox_path
import enrichers.merge_pseudo_labels
import enrichers.filter_incomplete
import enrichers.replace_image_base
from enrichers.base import get_enricher
from utils.trajectory import compute_sample_trajectory
from prompts.base import PromptBuilder
from writers.json_writer import write_output

# Use fork on Linux to avoid pickling overhead (copy-on-write shared memory)
if hasattr(mp, 'set_start_method'):
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass


def validate_enricher_chain(enricher_configs: list, base_fields: set):
    """Validate that each enricher's reads are satisfied by previous writes."""
    produced = set(base_fields)
    for ec in enricher_configs:
        if isinstance(ec, str):
            name, params = ec, {}
        else:
            name = ec["name"]
            params = ec.get("params", {})
        enricher = get_enricher(name, params)
        missing = set(enricher.reads) - produced
        if missing:
            raise ValueError(
                f"Enricher '{name}' needs fields {missing}, "
                f"but no preceding enricher produces them. "
                f"Check enricher ordering in pipeline config."
            )
        produced |= set(enricher.writes)


def _load_dataset_config(ds_cfg: dict) -> dict:
    """Resolve dataset config: if only {name, split}, load full YAML for paths."""
    if "splits" in ds_cfg or "data_path" in ds_cfg:
        return ds_cfg

    import yaml
    ds_yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "configs", "datasets", f"{ds_cfg['name']}.yaml"
    )
    if os.path.exists(ds_yaml_path):
        with open(ds_yaml_path) as f:
            full = yaml.safe_load(f)
        full["split"] = ds_cfg.get("split", "train")
        return full
    return ds_cfg


def run_pipeline(pipeline_cfg: dict, prompt_builder: PromptBuilder):
    """Run the full pipeline: extract -> enrich -> render -> write."""
    ds_cfg = _load_dataset_config(pipeline_cfg["dataset"])
    ds_name = ds_cfg['name']
    ds_split = ds_cfg.get('split', 'train')

    num_workers = pipeline_cfg.get("num_workers", 1)
    if num_workers <= 0:
        num_workers = 1

    # --- Cache key (computed before adapter to skip loading if cached) ---
    cache_cfg = pipeline_cfg.get("cache", {})
    cache_enabled = cache_cfg.get("enabled", True)
    cache_dir = pipeline_cfg.get("paths", {}).get("cache_dir", "./cache")
    cache_path = None
    if cache_enabled:
        import json, hashlib
        split_cfg = ds_cfg.get("splits", {}).get(ds_split, {})
        h = hashlib.md5(json.dumps(split_cfg, sort_keys=True).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir,
                                  f"{ds_name}_{ds_split}_{h}.pkl")

    pool = Pool(num_workers) if num_workers > 1 else None
    print(f"Workers: {num_workers} (pool={'enabled' if pool else 'disabled'})")
    t_start = time.time()
    try:
        # --- Extract (with IR cache — skips adapter loading if cached) ---
        t0 = time.time()
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached IRs from {cache_path} ...")
            with open(cache_path, 'rb') as f:
                samples = pickle.load(f)
            print(f"Loaded {len(samples)} IRs, {time.time() - t0:.1f}s")
        else:
            adapter_cls = get_adapter_class(ds_cfg["name"])
            adapter = adapter_cls(ds_cfg)
            total = adapter.get_sample_count()
            print(f"Extracting {total} samples from '{ds_name}' (split={ds_split})")
            raw_samples = [adapter.extract_sample(i) for i in tqdm(range(total), desc="Extracting")]
            samples = [s for s in (build_sample_ir(raw) for raw in raw_samples) if s is not None]
            # Always compute trajectory inline (CPU-bound, can use pool)
            print(f"Computing trajectory for {len(samples)} IRs...")
            if pool:
                samples = list(tqdm(
                    pool.imap(compute_sample_trajectory, samples, chunksize=50),
                    total=len(samples), desc="compute_trajectory"
                ))
            else:
                samples = [compute_sample_trajectory(s) for s in tqdm(samples, desc="compute_trajectory")]
            samples = [s for s in samples if s is not None]
            print(f"Extract + Trajectory: {len(samples)} IRs, {time.time() - t0:.1f}s")
            if cache_path:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(samples, f)
                print(f"Cached to {cache_path}")

        # Validate enricher chain
        enricher_configs = pipeline_cfg.get("enrichers", [])
        validate_enricher_chain(enricher_configs, {
            "sample_id", "metadata", "frames", "images", "trajectory",
        })

        # --- Enrich ---
        for ec in enricher_configs:
            if isinstance(ec, str):
                name, params = ec, {}
            else:
                name = ec["name"]
                params = ec.get("params", {})
            enricher = get_enricher(name, params)
            print(f"Running enricher: {name} "
                  f"(reads={enricher.reads}, writes={enricher.writes})")
            t0 = time.time()
            samples = enricher.enrich_all(samples, pool)
            samples = [s for s in samples if s is not None]
            print(f"  -> {len(samples)} samples, {time.time() - t0:.1f}s")

        # --- Render ---
        print(f"Rendering prompts with '{prompt_builder.name}'...")
        t0 = time.time()
        qa_pairs = []
        skipped = 0
        for s in tqdm(samples, desc="Rendering"):
            try:
                qa = prompt_builder.build(s)
                if qa is not None:
                    qa_pairs.append(qa)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if skipped:
            print(f"  Skipped {skipped} failed renders.")
        print(f"Generated {len(qa_pairs)} QA pairs, {time.time() - t0:.1f}s")

        # --- Write ---
        t0 = time.time()
        output_cfg = pipeline_cfg["output"]
        write_output(qa_pairs, output_cfg)
        print(f"[Write] {time.time() - t0:.1f}s")

    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"Done. Output saved to {pipeline_cfg['output']['path']}.")
    print(f"Total: {time.time() - t_start:.1f}s")
