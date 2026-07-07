"""CLI entry point for the DriveTeach-VLA data engine.

Usage:
    python main.py --config configs/pipelines/poutine.yaml
    python main.py --config configs/pipelines/prompter.yaml
    python main.py --config configs/pipelines/planner.yaml
"""

import argparse
import sys
import os

import yaml

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipelines.runner import run_pipeline
from prompts.base import get_prompt

# Trigger @register_prompt decorators
import prompts.poutine_label
import prompts.prompter
import prompts.planner


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="DriveTeach-VLA Data Engine"
    )
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Pipeline config YAML file"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Override number of workers (default: from defaults.yaml, fallback CPU count)"
    )
    args = parser.parse_args()

    # Load defaults + pipeline config (pipeline overrides defaults)
    defaults_path = os.path.join(os.path.dirname(__file__), "configs", "defaults.yaml")
    defaults_cfg = load_yaml(defaults_path) if os.path.exists(defaults_path) else {}
    pipeline_cfg = load_yaml(args.config)

    # Merge top-level keys from defaults (pipeline wins)
    for key in ("paths", "cache", "runtime"):
        if key in defaults_cfg and key not in pipeline_cfg:
            pipeline_cfg[key] = defaults_cfg[key]

    # num_workers fallback chain
    default_workers = defaults_cfg.get("runtime", {}).get("num_workers", 1)
    pipeline_cfg.setdefault("num_workers", default_workers)

    if args.num_workers is not None:
        pipeline_cfg["num_workers"] = args.num_workers

    # Resolve prompt builder
    prompt_builder = get_prompt(pipeline_cfg["prompt"])

    # Run
    run_pipeline(pipeline_cfg, prompt_builder)


if __name__ == "__main__":
    main()
