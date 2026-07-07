"""project_2d_tgp enricher: BEV trajectory -> camera pixels -> inverse project.

Adapted from projection_human_agent.py.
Output format: [{"point_2d": [x:int, y:int], "heading": raw_yaw}, ...]
"""

import math
from typing import Optional

import numpy as np
from tqdm import tqdm

from ir.schema import SampleIR, CameraCalib
from .base import Enricher, register_enricher


@register_enricher("project_2d_tgp")
class Project2DTGPEnricher(Enricher):
    """Project raw BEV trajectory to 2D camera pixels, then inverse-project.

    Uses the camera calibration from the current frame (cam_f0).

    reads:  trajectory, frames
    writes: trajectory_2d_tgp
    """

    reads = ["trajectory", "normalized_trajectory", "frames"]
    writes = ["trajectory_2d_tgp"]

    def __init__(self, z_offset: float = 0.0, ground_z: float = 0.0,
                 quantize_factor: int = 28, max_pixels: int = 262144,
                 image_height: int = 1080, image_width: int = 1920,
                 target_height: int = 448, target_width: int = 784):
        self.z_offset = z_offset
        self.ground_z = ground_z
        self.quantize_factor = quantize_factor
        self.max_pixels = max_pixels
        self.image_height = image_height
        self.image_width = image_width
        self.target_height = target_height
        self.target_width = target_width

    def enrich_all(self, samples: list[SampleIR],
                   pool=None) -> list[SampleIR]:
        return [self._project_one(s) for s in tqdm(samples, desc="project_2d_tgp")]

    def _project_one(self, sample: SampleIR) -> SampleIR:
        if sample.trajectory is None:
            return sample

        try:
            calib = self._get_front_calib(sample)
            if calib is None:
                return sample

            future_pts = [p for p in sample.trajectory if p.t > 1e-9]
            if not future_pts:
                return sample

            bev_poses = np.array([[p.x, p.y, p.heading] for p in future_pts])
            projected = self._project_to_image(bev_poses, calib)
            # Scale to target image size
            projected = self._scale_pixels(
                projected, self.image_height, self.image_width,
            )
            projected = self._quantize_pixels(
                projected, self.target_height, self.target_width
            )

            result = []
            for i, pt in enumerate(future_pts):
                px, py = float(projected[i][0]), float(projected[i][1])
                if not (np.isfinite(px) and np.isfinite(py)):
                    px, py = 0.0, 0.0
                # Use normalized heading if available, else raw
                h = pt.heading
                if sample.normalized_trajectory and i < len(sample.normalized_trajectory):
                    h = sample.normalized_trajectory[i].heading
                result.append({
                    "point_2d": [int(round(px)), int(round(py))],
                    "heading": round(h, 2),
                })

            sample.trajectory_2d_tgp = result
        except Exception:
            pass  # failed sample, trajectory_2d_tgp stays None

        return sample

    def _get_front_calib(self, sample: SampleIR) -> Optional[dict]:
        """Get cam_f0 calibration as a plain dict for numpy operations."""
        for frame in sample.frames:
            if frame.camera_calib and "cam_f0" in frame.camera_calib:
                c = frame.camera_calib["cam_f0"]
                if isinstance(c, CameraCalib):
                    return {
                        "intrinsics": c.intrinsics,
                        "sensor2lidar_rotation": c.sensor2lidar_rotation,
                        "sensor2lidar_translation": c.sensor2lidar_translation,
                    }
                return c
        return None

    def _project_to_image(self, trajectory_poses: np.ndarray,
                          calib: dict) -> np.ndarray:
        num_points = trajectory_poses.shape[0]
        points_ego = np.hstack([
            trajectory_poses[:, :2],
            np.full((num_points, 1), self.z_offset),
            np.ones((num_points, 1)),
        ])

        R = np.array(calib["sensor2lidar_rotation"])
        t = np.array(calib["sensor2lidar_translation"])
        R_lidar2cam = R.T
        t_lidar2cam = -R_lidar2cam @ t

        T_lidar2cam = np.eye(4)
        T_lidar2cam[:3, :3] = R_lidar2cam
        T_lidar2cam[:3, 3] = t_lidar2cam

        points_cam = (T_lidar2cam @ points_ego.T).T
        K = np.array(calib["intrinsics"])
        projected = (K @ points_cam[:, :3].T).T

        valid = []
        for pt in projected:
            valid.append([pt[0] / pt[2], pt[1] / pt[2]])

        return np.array(valid)

    def _quantize_pixels(self, pixels: np.ndarray, height: int,
                         width: int) -> np.ndarray:
        factor = self.quantize_factor
        max_pixels = self.max_pixels

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor

        pixels[:, 0] = np.round(pixels[:, 0] / width * w_bar * 10) / 10
        pixels[:, 1] = np.round(pixels[:, 1] / height * h_bar * 10) / 10

        pixels[:, 0] = pixels[:, 0] / w_bar * width
        pixels[:, 1] = pixels[:, 1] / h_bar * height

        return pixels

    def _scale_pixels(self, pixels: np.ndarray, height: int,
                      width: int) -> np.ndarray:
        """Scale pixel coordinates to target image size.

        Width and height scaled independently to target_width / target_height.
        """
        pixels[:, 0] = pixels[:, 0] / width * self.target_width
        pixels[:, 1] = pixels[:, 1] / height * self.target_height
        return pixels

    def _pixel_to_ground(self, pixel: np.ndarray, calib: dict) -> Optional[np.ndarray]:
        """Inverse-project a pixel to ground plane via ray-plane intersection."""
        u, v = pixel[0], pixel[1]
        K = np.array(calib["intrinsics"])
        K_inv = np.linalg.inv(K)

        ray_cam = K_inv @ np.array([u, v, 1.0])

        R = np.array(calib["sensor2lidar_rotation"])
        cam_pos = np.array(calib["sensor2lidar_translation"])
        ray_lidar = R @ ray_cam

        z_ray = ray_lidar[2]
        if abs(z_ray) < 1e-6:
            return None

        t = (self.ground_z - cam_pos[2]) / z_ray
        pos_lidar = cam_pos + ray_lidar * t

        return pos_lidar[:2]
