"""Intermediate Representation (IR) — the unified data model for all pipeline stages.

All fields are append-only: each enricher writes its own field, never overwriting another's.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CameraCalib(BaseModel):
    """Camera calibration data extracted from navsim raw dict."""
    intrinsics: list[list[float]]
    sensor2lidar_rotation: list[list[float]]
    sensor2lidar_translation: list[float]
    distortion: Optional[list[float]] = None


class FrameIR(BaseModel):
    """Single frame from the dataset (history, current, or future)."""
    token: str
    timestamp: int
    position: list[float]          # [x, y, heading] — global ENU
    rotation: list[float]          # [qx, qy, qz, qw] — quaternion
    velocity: list[float]          # [vx, vy]
    acceleration: list[float]      # [ax, ay]
    command: str                   # "turn left" | "go straight" | "turn right" | "unknown"
    images: dict[str, str] = Field(default_factory=dict)        # camera_name → image_path
    camera_calib: dict[str, CameraCalib] = Field(default_factory=dict)


class MetadataIR(BaseModel):
    log_name: str
    scene_token: str
    map_name: str
    num_frames: int
    command: str = "unknown"    # driving command of the current frame


class EgoPoint(BaseModel):
    """A single trajectory point in ego-centric coordinates."""
    t: float      # seconds from current frame (-2.0 → +4.0)
    x: float
    y: float
    heading: float  # relative yaw (radians)


class PseudoLabelIR(BaseModel):
    """CoT annotations from VLM pseudo-labeling output."""
    critical_objects: Optional[dict[str, str]] = None
    explanation: Optional[str] = None
    meta_behaviour: Optional[dict[str, str]] = None


class SampleIR(BaseModel):
    """Unified Intermediate Representation for a single scene/sample.

    Each enricher writes to its own field — fields are append-only.
    """
    sample_id: str
    metadata: MetadataIR
    frames: list[FrameIR]

    # Front-view image paths (length 1, raw only)
    images: list[str] = Field(default_factory=list)

    # Computed during extract phase (raw ego-centric)
    trajectory: Optional[list[EgoPoint]] = None

    # Written by normalize_trajectory
    normalized_trajectory: Optional[list[EgoPoint]] = None

    # Written by project_2d_tgp: [{"point_2d": [x, y], "heading": h}, ...]
    trajectory_2d_tgp: Optional[list[dict]] = None

    # Written by merge_pseudo_labels
    pseudo_label: Optional[PseudoLabelIR] = None

    # Written by inject_bbox_path
    bbox_image: Optional[str] = None
