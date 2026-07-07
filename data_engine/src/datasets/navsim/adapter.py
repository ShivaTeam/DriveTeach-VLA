"""Navsim v2.0.0 adapter — reads raw dicts, preserves image paths."""

import numpy as np
from pathlib import Path
from typing import Any

from pyquaternion import Quaternion

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

from ..base import BaseDatasetAdapter
from ..registry import register_dataset


def _quaternion_yaw(rotation: list[float]) -> float:
    """Extract yaw from [qw, qx, qy, qz] quaternion."""
    q = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    return q.yaw_pitch_roll[0]


def _decode_driving_command(cmd: np.ndarray) -> str:
    """Decode one-hot driving command to string."""
    navigation_commands = ['turn left', 'go straight', 'turn right', 'unknown']
    if hasattr(cmd, 'tolist'):
        cmd = cmd.tolist()
    for i in range(len(navigation_commands)):
        if i < len(cmd) and cmd[i] == 1:
            return navigation_commands[i]
    return "unknown"


@register_dataset("navsim")
class NavsimAdapter(BaseDatasetAdapter):
    """Extracts raw frame data from navsim v2.0.0 pickle logs.

    Reads raw frame dicts (NOT Scene objects) to preserve image paths
    and avoid eager image loading into memory.
    """

    def __init__(self, config: dict):
        self.split = config["split"]
        split_cfg = config.get("splits", {}).get(self.split, config)

        # Resolve log_names_file path relative to project root
        import yaml, os
        log_names = split_cfg.get("log_names")

        log_names_file = split_cfg.get("log_names_file")
        if log_names_file and log_names is None:
            if not os.path.isabs(log_names_file):
                log_names_file = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", log_names_file
                )
            if os.path.exists(log_names_file):
                scene_filter = self._build_scene_filter_from_yaml(log_names_file)
            else:
                raise FileNotFoundError(f"Scene filter YAML not found: {log_names_file}")
        else:
            # Fallback: manual params
            scene_filter = SceneFilter(
                num_history_frames=split_cfg.get("num_history_frames", 4),
                num_future_frames=split_cfg.get("num_future_frames", 10),
                frame_interval=split_cfg.get("frame_interval"),
                log_names=log_names,
                has_route=split_cfg.get("has_route", True),
            )

        sensor_config = SensorConfig.build_no_sensors()
        self._data_path = Path(split_cfg["data_path"])
        self._sensor_path = Path(split_cfg["sensor_blobs_path"])

        self._loader = SceneLoader(
            data_path=self._data_path,
            original_sensor_path=self._sensor_path,
            scene_filter=scene_filter,
            sensor_config=sensor_config,
        )

        # Cache scene_filter properties
        self._num_history = scene_filter.num_history_frames

    def _build_scene_filter_from_yaml(self, yaml_path: str) -> SceneFilter:
        """Build SceneFilter from YAML config, following the same pattern as
        the original dataset_navsim.py: prefer OmegaConf instantiate, fallback to
        manual param extraction."""
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        # Try Hydra/OmegaConf instantiate (matches old dataset_navsim.py behavior)
        try:
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            om_cfg = OmegaConf.load(yaml_path)
            scene_filter = instantiate(om_cfg)
            return scene_filter
        except ImportError:
            pass

        # Fallback: extract params manually from loaded dict
        log_names = cfg.get("log_names")
        return SceneFilter(
            num_history_frames=cfg.get("num_history_frames", 4),
            num_future_frames=cfg.get("num_future_frames", 10),
            frame_interval=cfg.get("frame_interval"),
            log_names=log_names,
            has_route=cfg.get("has_route", True),
        )

    def get_sample_count(self) -> int:
        return len(self._loader)

    def extract_sample(self, index: int) -> dict:
        token = self._loader[index]
        frames_dicts = self._loader.scene_frames_dicts[token]
        current_idx = self._num_history - 1

        # Extract image paths + calib from current frame only
        camera_dict = frames_dicts[current_idx].get("cams", {})
        images, camera_calib = {}, {}
        for cam_name, cam_data in camera_dict.items():
            cam_id = cam_name.lower()
            data_path = cam_data.get("data_path", "")
            if data_path:
                images[cam_id] = str(self._sensor_path / data_path)
            camera_calib[cam_id] = {
                "intrinsics": cam_data.get("cam_intrinsic"),
                "sensor2lidar_rotation": cam_data.get("sensor2lidar_rotation"),
                "sensor2lidar_translation": cam_data.get("sensor2lidar_translation"),
                "distortion": cam_data.get("distortion"),
            }

        # Build frame data for all frames (history + current + future)
        frames = []
        for i, fd in enumerate(frames_dicts):
            ego_trans = fd.get("ego2global_translation", [0, 0, 0])
            ego_rot = fd.get("ego2global_rotation", [1, 0, 0, 0])
            ego_dynamic = fd.get("ego_dynamic_state", [0, 0, 0, 0])
            driving_cmd = fd.get("driving_command", np.array([0, 0, 0, 1]))

            is_current = (i == current_idx)

            frames.append({
                "token": fd.get("token", ""),
                "timestamp": fd.get("timestamp", 0),
                "position": [ego_trans[0], ego_trans[1],
                             _quaternion_yaw(ego_rot)],
                "rotation": ego_rot,
                "velocity": list(ego_dynamic[:2]),
                "acceleration": list(ego_dynamic[2:]),
                "command": _decode_driving_command(driving_cmd),
                "images": images if is_current else {},
                "camera_calib": camera_calib if is_current else {},
            })

        return {
            "token": token,
            "metadata": {
                "log_name": frames_dicts[current_idx].get("log_name", ""),
                "scene_token": frames_dicts[current_idx].get("scene_token", token),
                "map_name": frames_dicts[current_idx].get("map_location", ""),
                "num_frames": len(frames_dicts),
            },
            "frames": frames,
        }
