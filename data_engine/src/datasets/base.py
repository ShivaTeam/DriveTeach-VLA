"""Abstract base class for dataset adapters."""

from abc import ABC, abstractmethod


class BaseDatasetAdapter(ABC):
    """Interface for dataset-specific extraction logic.

    Each dataset (navsim, nuScenes, etc.) implements this to convert
    its native data into a standardized raw dict consumed by IR builder.
    """

    name: str

    @abstractmethod
    def get_sample_count(self) -> int:
        """Return the total number of samples for this dataset split."""
        ...

    @abstractmethod
    def extract_sample(self, index: int) -> dict:
        """Extract raw data for a single sample.

        Returns a dict with keys: token, metadata, frames.
        Each frame has: token, timestamp, position, rotation, velocity,
                        acceleration, command, images, camera_calib.
        """
        ...
