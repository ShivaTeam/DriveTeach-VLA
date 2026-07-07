# Configurable Paths Reference

All paths are hardcoded in YAML configs. To deploy on a different machine, update the paths in the referenced files.

## Paths by Config File

### 1. `configs/datasets/navsim.yaml` — Dataset

| Line | Key | Value | Description |
|------|-----|-------|-------------|
| 9 | `splits.train.data_path` | `/path/to/datasets/navsim/trainval_logs/trainval` | Training set pickle log directory |
| 10 | `splits.train.sensor_blobs_path` | `/path/to/datasets/navsim/trainval_sensor_blobs/trainval` | Training set sensor blobs (images) directory |
| 11 | `splits.train.log_names_file` | `/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml` | Scene filter YAML |
| 13 | `splits.test.data_path` | `/path/to/datasets/navsim/test_navsim_logs/test` | Test set pickle log directory |
| 14 | `splits.test.sensor_blobs_path` | `/path/to/datasets/navsim/test_sensor_blobs/test` | Test set sensor blobs directory |
| 15 | `splits.test.log_names_file` | `/path/to/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml` | Scene filter YAML |

### 2. `configs/defaults.yaml` — Global defaults

| Line | Key | Value | Description |
|------|-----|-------|-------------|
| 4 | `paths.stats_dir` | `./stats` | Normalization stats cache directory |
| 5 | `paths.output_dir` | `./output` | Output JSON directory |
| 8 | `runtime.num_workers` | `1` | Default multiprocessing pool size |

### 3. `configs/pipelines/poutine_label.yaml` — Poutine Label

| Line | Key | Value | Description |
|------|-----|-------|-------------|
| 11 | `enrichers[0].params.old_base` | `/path/to/datasets/navsim` | Image path prefix to replace |
| 12 | `enrichers[0].params.new_base` | `navsim` | Replacement prefix |
| 19 | `output.path` | `./output/QA_navsim_train_poutine_label.json` | Output file |

### 4. `configs/pipelines/prompter.yaml` — Prompter (DVD)

| Line | Key | Value | Description |
|------|-----|-------|-------------|
| 11 | `enrichers[0].params.old_base` | `/path/to/datasets/navsim` | Bbox path prefix to replace (inject_bbox_path) |
| 12 | `enrichers[0].params.new_base` | `boxed_navsim` | Bbox path replacement |
| 15 | `enrichers[1].params.old_base` | `/path/to/datasets/navsim` | Image path prefix to replace (replace_image_base) |
| 16 | `enrichers[1].params.new_base` | `navsim` | Replacement prefix |
| 20 | `enrichers[3].params.stats_file` | `./stats/navsim_train_stats.json` | Normalization stats cache |
| 23 | `enrichers[4].params.image_height` | `1080` | Original image height |
| 24 | `enrichers[4].params.image_width` | `1920` | Original image width |
| 25 | `enrichers[4].params.target_height` | `448` | Target image height after scaling |
| 26 | `enrichers[4].params.target_width` | `784` | Target image width after scaling |
| 32 | `output.path` | `./output/QA_navsim_train_prompter.json` | Output file |

### 5. `configs/pipelines/planner.yaml` — Planner (CoT-SFT)

| Line | Key | Value | Description |
|------|-----|-------|-------------|
| 11 | `enrichers[0].params.old_base` | `/path/to/datasets/navsim` | Image path prefix to replace |
| 12 | `enrichers[0].params.new_base` | `navsim` | Replacement prefix |
| 17 | `enrichers[2].params.stats_file` | `./stats/navsim_train_stats.json` | Normalization stats cache |
| 20 | `enrichers[3].params.image_height` | `1080` | Original image height |
| 21 | `enrichers[3].params.image_width` | `1920` | Original image width |
| 22 | `enrichers[3].params.target_height` | `448` | Target image height after scaling |
| 23 | `enrichers[3].params.target_width` | `784` | Target image width after scaling |
| 26 | `enrichers[4].params.source_path` | `/path/to/vlm_output.jsonl` | **Must update** — VLM inference JSONL |
| 36 | `output.path` | `./output/QA_navsim_train_planner.json` | Output file |

## Paths That Must Be Updated for Deployment

| # | File | Key | Change |
|---|------|-----|--------|
| 1 | `configs/datasets/navsim.yaml` | `splits.train.data_path` | Set to actual trainval pickle directory |
| 2 | `configs/datasets/navsim.yaml` | `splits.train.sensor_blobs_path` | Set to actual trainval images directory |
| 3 | `configs/datasets/navsim.yaml` | `splits.test.data_path` | Set to actual test pickle directory |
| 4 | `configs/datasets/navsim.yaml` | `splits.test.sensor_blobs_path` | Set to actual test images directory |
| 5 | `configs/datasets/navsim.yaml` | `splits.*.log_names_file` | Set to actual navtrain/navtest.yaml paths |
| 6 | `configs/pipelines/*.yaml` | `replace_image_base.params` | Set old_base to actual dataset root |
| 7 | `configs/pipelines/prompter.yaml` | `inject_bbox_path.params` | Set old_base/new_base to actual paths |
| 8 | `configs/pipelines/planner.yaml` | `merge_pseudo_labels.source_path` | Set to actual VLM inference JSONL |
