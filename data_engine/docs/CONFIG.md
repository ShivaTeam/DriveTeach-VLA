# Configuration Guide

## File Structure

```
configs/
  defaults.yaml              # Global defaults (workers, paths)
  datasets/
    navsim.yaml              # Dataset registration (train/test splits)
  pipelines/
    poutine_label.yaml       # Pipeline: VLM pseudo-labeling input
    prompter.yaml            # Pipeline: DVD pretraining (TGP-Prompter)
    planner.yaml             # Pipeline: CoT-SFT training (TGP-Planner)
```

## Dataset Configuration

`configs/datasets/navsim.yaml` defines train/test splits. SceneFilter params (frame_interval, num_history_frames, log_names, has_route) are read from the referenced YAML files via `log_names_file`.

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

The adapter tries OmegaConf instantiate first, then falls back to manual param extraction.

## Pipeline Configuration

Each pipeline YAML defines: **dataset**, **enrichers** (ordered chain), **prompt** (registered `@register_prompt` name), **output** (format + path).

### Poutine Label (`poutine_label`)

VLM pseudo-labeling input. Includes expert future trajectory and velocity for the VLM to label.

```yaml
dataset: {name: navsim, split: train}

enrichers:
  - name: replace_image_base
    params: {old_base: "/path/to/datasets/navsim", new_base: "navsim"}

prompt: poutine_label

output:
  format: json
  path: "./output/QA_navsim_train_poutine_label.json"
```

### Prompter (`prompter`)

DVD pretraining — TGP-Prompter 2D-TGP regression.

```yaml
dataset: {name: navsim, split: train}

enrichers:
  - name: inject_bbox_path
    params: {old_base: "/path/to/datasets/navsim", new_base: "boxed_navsim"}
  - name: replace_image_base
    params: {old_base: "/path/to/datasets/navsim", new_base: "navsim"}
  - name: normalize_trajectory
    params: {stats_file: "./stats/navsim_train_stats.json"}
  - name: project_2d_tgp
    params: {image_height: 1080, image_width: 1920, target_height: 448, target_width: 784}

prompt: prompter

output:
  format: json
  path: "./output/QA_navsim_train_prompter.json"
```

### Planner (`planner`)

CoT-SFT — TGP-Planner with CoT reasoning and normalized BEV trajectory.

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
    params: {source_path: "/path/to/vlm_output.jsonl", parser: poutine, key_field: id, json_repair: true}
  - filter_incomplete_pseudo

prompt: planner

output:
  format: json
  path: "./output/QA_navsim_train_planner.json"
```

## Enricher Reference

| Enricher | reads | writes | Key params |
|----------|-------|--------|------------|
| `replace_image_base` | images | images | `old_base`, `new_base` |
| `normalize_trajectory` | trajectory | normalized_trajectory | `stats_file`, `num_timesteps: 8` |
| `inject_bbox_path` | images | bbox_image | `old_base`, `new_base` |
| `project_2d_tgp` | trajectory, normalized_trajectory, frames | trajectory_2d_tgp | `image_height`, `image_width`, `target_height`, `target_width` |
| `merge_pseudo_labels` | — | pseudo_label | `source_path`, `parser`, `key_field`, `json_repair` |
| `filter_incomplete_pseudo` | pseudo_label | — | (none) |

## PromptBuilder Reference

| Builder | Registered name | Purpose |
|---------|----------------|---------|
| `PoutineLabelBuilder` | `poutine_label` | VLM pseudo-labeling input |
| `PrompterBuilder` | `prompter` | DVD pretraining (2D-TGP regression) |
| `PlannerBuilder` | `planner` | CoT-SFT (CoT + normalized BEV) |

All builders use `@register_prompt(name)` decorator and follow the same pattern:

```python
@register_prompt("my_prompt")
class MyPromptBuilder(PromptBuilder):
    def build(self, sample: SampleIR) -> dict:
        return {"id": ..., "image": ..., "system": ..., "conversations": [...]}
```

## CLI

```bash
# Run with default config
python main.py --config configs/pipelines/planner.yaml

# Override number of workers
python main.py --config configs/pipelines/planner.yaml --num-workers 32
```
