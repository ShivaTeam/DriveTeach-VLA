"""Build SampleIR from raw adapter dicts."""

from .schema import (
    SampleIR, MetadataIR, FrameIR, CameraCalib, PseudoLabelIR
)


def build_sample_ir(raw: dict) -> SampleIR:
    """Convert a raw adapter dict into a validated SampleIR.

    Args:
        raw: dict from BaseDatasetAdapter.extract_sample()
             with keys: token, metadata, frames
    """
    metadata = MetadataIR(
        log_name=raw["metadata"]["log_name"],
        scene_token=raw["metadata"]["scene_token"],
        map_name=raw["metadata"]["map_name"],
        num_frames=raw["metadata"]["num_frames"],
        command=_extract_command(raw["frames"]),
    )

    frames = []
    images = []
    for i, fd in enumerate(raw["frames"]):
        calib = {}
        for cam_name, cam_data in fd.get("camera_calib", {}).items():
            calib[cam_name] = CameraCalib(
                intrinsics=cam_data["intrinsics"],
                sensor2lidar_rotation=cam_data["sensor2lidar_rotation"],
                sensor2lidar_translation=cam_data["sensor2lidar_translation"],
                distortion=cam_data.get("distortion"),
            )
        frame = FrameIR(
            token=fd["token"],
            timestamp=fd["timestamp"],
            position=fd["position"],
            rotation=fd["rotation"],
            velocity=fd["velocity"],
            acceleration=fd["acceleration"],
            command=fd["command"],
            images=fd.get("images", {}),
            camera_calib=calib,
        )
        frames.append(frame)
        # Collect front-view image paths from the current frame
        if fd.get("images") and "cam_f0" in fd["images"]:
            images.append(fd["images"]["cam_f0"])

    sample_id = raw["token"]

    return SampleIR(
        sample_id=sample_id,
        metadata=metadata,
        frames=frames,
        images=images,
    )


def _extract_command(frames: list[dict]) -> str:
    """Extract driving command from the current frame.

    The current frame is the one with camera images (non-empty 'images' dict).
    """
    for fd in frames:
        if fd.get("images"):
            return fd["command"]
    return frames[-1]["command"] if frames else "unknown"
