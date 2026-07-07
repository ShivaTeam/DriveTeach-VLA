# DriveTeach-VLA Data Engine

Configurable data engine for building training datasets for the [DriveTeach-VLA](https://github.com/ShivaTeam/DriveTeach-VLA) autonomous driving VLA framework (ECCV 2026).

## Overview

The data engine extracts raw sensor data from the [NAVSIM](https://github.com/autonomousvision/navsim) dataset, runs a chain of configurable **enrichers** to compute ego-centric trajectories, normalize coordinates, and project BEV trajectories to camera pixels, then renders training samples through **PromptBuilders** into LLaMA-Factory compatible format.

### Three Built-in Pipelines

| Pipeline | Purpose | PromptBuilder | Output |
|----------|---------|---------------|--------|
| `poutine_label` | Pseudo-labeling input | `PoutineLabelBuilder` | `{critical_objects, meta_behaviour, explanation}` |
| `prompter` | DVD pretraining (2D-TGP) | `PrompterBuilder` | `[PT, {"point_2d":[x,y],"heading":h},...]` |
| `planner` | CoT-SFT training | `PlannerBuilder` | CoT JSON + normalized `<answer>[PT,...]</answer>` |

## Quick Start

```bash
# Install dependencies
pip install pydantic pyquaternion numpy tqdm pyyaml json-repair

# Install navsim v2.0 (see https://github.com/autonomousvision/navsim)
pip install -e /path/to/navsim/

# Update paths in configs/datasets/navsim.yaml

# Run a pipeline
python data_engine/main.py --config data_engine/configs/pipelines/poutine_label.yaml
python data_engine/main.py --config data_engine/configs/pipelines/prompter.yaml
python data_engine/main.py --config data_engine/configs/pipelines/planner.yaml
```

## Architecture

```
raw navsim data  ──Adapter──▶  raw dict  ──Builder──▶  SampleIR
                                                          │
                              ┌───────────────────────────┘
                              │  Enricher chain
                              ├── inject_bbox_path       (bbox image path)
                              ├── replace_image_base     (path substitution)
                              ├── normalize_trajectory   (step-wise z-score)
                              ├── project_2d_tgp         (BEV→pixels)
                              ├── merge_pseudo_labels    (VLM output→IR)
                              └── filter_incomplete      (quality filter)
                                                          │
                                                          ▼
                                                     PromptBuilder
                                                          │
                                                          ▼
                                                   JSON output
```

### Key Concepts

**SampleIR** — Unified Intermediate Representation. All pipeline components read/write this Pydantic model. Fields are append-only: each enricher writes its own field, preserving original values.

**Enrichers** — Pluggable processing units. Each declares `reads`/`writes` (IR field names). The pipeline runner validates the dependency chain at startup. Enrichers can internally use any number of passes (two-pass normalization, N-pass filtering, etc.) — the framework is pass-agnostic.

**PromptBuilder** — Python class decorated with `@register_prompt(name)` that converts a SampleIR into a LLaMA-Factory training sample `{id, image, system, conversations}`.

**Adapter** — Dataset-specific extraction layer. Only the adapter touches the raw navsim API. Upgrading navsim or adding a new dataset only requires a new adapter — enrichers and prompts are dataset-agnostic.

## Configuration

All config is YAML. Three layers:

### 1. `configs/datasets/navsim.yaml` — Dataset registration

```yaml
name: navsim
splits:
  train:
    data_path: "/path/to/datasets/navsim/trainval_logs/trainval"
    sensor_blobs_path: "/path/to/datasets/navsim/trainval_sensor_blobs/trainval"
    log_names_file: "/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
  test:
    data_path: "/path/to/datasets/navsim/test_navsim_logs/test"
    sensor_blobs_path: "/path/to/datasets/navsim/test_sensor_blobs/test"
    log_names_file: "/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
```

### 2. `configs/pipelines/planner.yaml` — Pipeline definition

```yaml
dataset: {name: navsim, split: train}

enrichers:
  - name: replace_image_base
    params: {old_base: "/path/to/datasets/navsim", new_base: "navsim"}
  - name: normalize_trajectory
    params: {stats_file: "./stats/navsim_train_stats.json"}
  - name: project_2d_tgp
    params: {image_height: 1080, image_width: 1920, target_height: 448, target_width: 784}
  - name: merge_pseudo_labels
    params:
      parser: poutine
      source_path: "/path/to/vlm_output.jsonl"
      json_repair: true
  - filter_incomplete_pseudo

prompt: planner

output:
  format: json
  path: "./output/QA_navsim_train_planner.json"
```

### 3. PromptBuilder (Python code) — Prompt format

See `src/prompts/` for the three built-in builders. Each uses `@register_prompt(name)` decorator.

## Adding a Custom Enricher

```python
from enrichers.base import Enricher, register_enricher
from ir.schema import SampleIR

@register_enricher("my_enricher")
class MyEnricher(Enricher):
    reads = ["trajectory"]
    writes = ["my_new_field"]

    def enrich_all(self, samples, pool=None):
        for s in samples:
            s.my_new_field = ...
        return samples
```

Use it in a pipeline YAML:

```yaml
enrichers:
  - my_enricher
```

## Adding a Custom Prompt Format

```python
from prompts.base import PromptBuilder, register_prompt

@register_prompt("my_prompt")
class MyPromptBuilder(PromptBuilder):
    def build(self, sample):
        return {
            "id": sample.sample_id,
            "image": sample.images,
            "system": "You are an expert driver.",
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."},
            ]
        }
```

Import the module in `main.py` to trigger `@register_prompt`, then use `prompt: my_prompt` in a pipeline YAML.

## Adding a Custom Pseudo-Label Parser

```python
from enrichers.merge_pseudo_labels import PseudoLabelParser, register_parser

@register_parser("my_format")
class MyParser(PseudoLabelParser):
    def parse(self, data, sample):
        raw = data.get("predict", "")
        return {"critical_objects": ..., "explanation": ..., "meta_behaviour": ...}
```

Use in pipeline: `parser: "my_format"`.

## Inference

The data engine **does not** perform VLM inference. It produces:

1. **Poutine Label pipeline** → JSON for pseudo-labeling → feed to VLM → produces JSONL
2. **Prompter/Planner pipelines** → consume the JSONL via `merge_pseudo_labels` enricher → produce training data

The inference step (vLLM) is a separate component.

## Output Format

All output is LLaMA-Factory compatible:

```json
[
  {
    "id": "scene_token",
    "image": ["navsim/sensor_blobs/trainval/.../cam_f0.jpg"],
    "system": "You are an expert driver.",
    "conversations": [
      {"from": "human", "value": "<image>\nIntent: GO STRAIGHT\n..."},
      {"from": "gpt", "value": "<answer>[PT, (+2.50, -1.00, +0.15), ...]</answer>"}
    ]
  }
]
```

DVD (prompter) output additionally includes `"bbox_image": "/path/to/bbox.jpg"` for training-code distillation.
