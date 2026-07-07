"""normalize_trajectory enricher: two-pass z-score normalization."""

import json
import os

import numpy as np
from tqdm import tqdm

from ir.schema import SampleIR, EgoPoint
from .base import Enricher, register_enricher


@register_enricher("normalize_trajectory")
class NormalizeTrajectoryEnricher(Enricher):
    """Normalize trajectory (x,y,heading) using z-score normalization.

    Two-pass: collect stats across all samples, then apply.
    If stats_file exists, skip collection pass (cached).

    reads:  trajectory
    writes: normalized_trajectory
    """

    reads = ["trajectory"]
    writes = ["normalized_trajectory"]

    def __init__(self, stats_file: str, num_timesteps: int = 8):
        self.stats_file = stats_file
        self.num_timesteps = num_timesteps

    def enrich_all(self, samples: list[SampleIR],
                   pool=None) -> list[SampleIR]:
        # Load or compute stats
        if os.path.exists(self.stats_file):
            with open(self.stats_file) as f:
                stats = json.load(f)
        else:
            stats = self._collect_stats(samples)
            os.makedirs(os.path.dirname(self.stats_file) or '.', exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f)

        mean = np.array(stats["mean"])   # (T, 3)
        std = np.array(stats["std"])

        # Pass 2: apply normalization (single-threaded — light scalar math)
        self._mean = mean
        self._std = std
        for s in tqdm(samples, desc="normalize_trajectory"):
            self._normalize_one(s)
        return samples

    def _normalize_one(self, s: SampleIR) -> SampleIR:
        if s.trajectory is None:
            return s
        future_pts = [p for p in s.trajectory if p.t > 1e-9]
        normalized = []
        for i, pt in enumerate(future_pts):
            if i >= self.num_timesteps:
                break
            nx = (pt.x - self._mean[i][0]) / self._std[i][0]
            ny = (pt.y - self._mean[i][1]) / self._std[i][1]
            nh = (pt.heading - self._mean[i][2]) / self._std[i][2]
            normalized.append(EgoPoint(t=pt.t, x=nx, y=ny, heading=nh))
        s.normalized_trajectory = normalized
        return s

    def _collect_stats(self, samples: list[SampleIR]) -> dict:
        """Collect all future trajectory points for per-timestep stats."""
        all_vals = [[] for _ in range(self.num_timesteps)]
        for s in tqdm(samples, desc="collecting trajectory stats"):
            if s.trajectory is None:
                continue
            future_pts = [p for p in s.trajectory if p.t > 1e-9]
            for i, pt in enumerate(future_pts):
                if i >= self.num_timesteps:
                    break
                all_vals[i].append([pt.x, pt.y, pt.heading])

        mean = []
        std = []
        for vals in all_vals:
            if not vals:
                mean.append([0.0, 0.0, 0.0])
                std.append([1e-6, 1e-6, 1e-6])
                continue
            arr = np.array(vals)
            m = arr.mean(axis=0)
            s = arr.std(axis=0)
            s[s < 1e-6] = 1e-6
            mean.append(m.tolist())
            std.append(s.tolist())

        return {"mean": mean, "std": std}
