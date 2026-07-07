"""Coordinate transformation and trajectory computation utilities."""

import math
from typing import Optional

import numpy as np
from pyquaternion import Quaternion

from ir.schema import SampleIR, EgoPoint


def global_to_ego(global_x, global_y, ego_x, ego_y, ego_yaw_rad):
    delta_x = global_x - ego_x
    delta_y = global_y - ego_y
    cos_theta = math.cos(ego_yaw_rad)
    sin_theta = math.sin(ego_yaw_rad)
    return (delta_x * cos_theta + delta_y * sin_theta,
            -delta_x * sin_theta + delta_y * cos_theta)


def quaternion_yaw(rotation):
    if len(rotation) == 4:
        return Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]).yaw_pitch_roll[0]
    return 0.0


def relative_yaw(start_quat, target_quat):
    start = Quaternion(start_quat[0], start_quat[1], start_quat[2], start_quat[3])
    target = Quaternion(target_quat[0], target_quat[1], target_quat[2], target_quat[3])
    return (start.inverse * target).yaw_pitch_roll[0]


def compute_sample_trajectory(sample: SampleIR, past_window=2.0,
                               future_window=4.0, interval=0.5) -> Optional[SampleIR]:
    """Compute ego-centric trajectory for a single IR sample.

    Returns the sample with trajectory populated, or None if no current frame found.
    """
    frames = sample.frames
    current_idx = None
    for i, f in enumerate(frames):
        if f.images:
            current_idx = i
            break
    if current_idx is None:
        return None

    current_ego_x = frames[current_idx].position[0]
    current_ego_y = frames[current_idx].position[1]
    current_yaw = frames[current_idx].position[2]
    start_quat = frames[current_idx].rotation

    trajectory = []
    # Past
    for i in range(current_idx, -1, -1):
        t_offset = (i - current_idx) * interval
        if t_offset < -past_window - 1e-9:
            break
        gx, gy = frames[i].position[0], frames[i].position[1]
        ego_x, ego_y = global_to_ego(gx, gy, current_ego_x, current_ego_y, current_yaw)
        rel_h = relative_yaw(start_quat, frames[i].rotation)
        trajectory.append(EgoPoint(t=t_offset, x=ego_x, y=ego_y, heading=rel_h))

    # Future
    for i in range(current_idx + 1, len(frames)):
        t_offset = (i - current_idx) * interval
        if t_offset > future_window + 1e-9:
            break
        gx, gy = frames[i].position[0], frames[i].position[1]
        ego_x, ego_y = global_to_ego(gx, gy, current_ego_x, current_ego_y, current_yaw)
        rel_h = relative_yaw(start_quat, frames[i].rotation)
        trajectory.append(EgoPoint(t=t_offset, x=ego_x, y=ego_y, heading=rel_h))

    trajectory.sort(key=lambda p: p.t)
    sample.trajectory = trajectory
    return sample

import math
import numpy as np
from pyquaternion import Quaternion


def global_to_ego(global_x: float, global_y: float,
                  ego_x: float, ego_y: float,
                  ego_yaw_rad: float) -> tuple[float, float]:
    """Convert global ENU coordinates to ego-centric coordinates."""
    delta_x = global_x - ego_x
    delta_y = global_y - ego_y
    cos_theta = math.cos(ego_yaw_rad)
    sin_theta = math.sin(ego_yaw_rad)
    return (
        delta_x * cos_theta + delta_y * sin_theta,
        -delta_x * sin_theta + delta_y * cos_theta,
    )


def quaternion_yaw(rotation: list[float]) -> float:
    """Extract yaw angle from quaternion [qx, qy, qz, qw] or [qw, qx, qy, qz]."""
    # navsim stores quaternion as [qw, qx, qy, qz] (scalar first)
    if len(rotation) == 4:
        return Quaternion(rotation[0], rotation[1], rotation[2], rotation[3]).yaw_pitch_roll[0]
    return 0.0


def relative_yaw(start_quat: list[float], target_quat: list[float]) -> float:
    """Compute relative yaw (radians) between two quaternions.

    start_quat, target_quat: [qw, qx, qy, qz] (scalar-first).
    """
    start = Quaternion(start_quat[0], start_quat[1], start_quat[2], start_quat[3])
    target = Quaternion(target_quat[0], target_quat[1], target_quat[2], target_quat[3])
    relative = start.inverse * target
    return relative.yaw_pitch_roll[0]