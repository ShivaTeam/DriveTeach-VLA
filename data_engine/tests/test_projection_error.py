"""Test BEV → pixel → inverse-project round-trip error across the dataset.

Measures how much information is lost when projecting ground-truth BEV
trajectories to camera pixels (with quantization) and back to ground plane.
"""

import sys
import os
import json
import argparse
from pathlib import Path

from typing import Optional

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets.navsim.adapter import NavsimAdapter
from ir.builder import build_sample_ir
from utils.trajectory import compute_sample_trajectory
from enrichers.project_2d_tgp import Project2DTGPEnricher


def compute_roundtrip_errors(
    adapter: NavsimAdapter,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Compute projection round-trip errors for the dataset.

    For each sample:
    1. Extract raw data → build SampleIR
    2. Compute ego-centric trajectory
    3. Project to camera pixels (with quantization) → inverse-project back
    4. Compare: ||original (x,y) - recovered (x,y)|| for each timestep

    Returns:
        dict with per-timestep error stats and outlier list.
    """
    total = adapter.get_sample_count()
    if num_samples is not None and num_samples < total:
        rng = np.random.RandomState(seed)
        indices = sorted(rng.choice(total, size=num_samples, replace=False))
    else:
        indices = list(range(total))
        num_samples = total

    proj_enricher = Project2DTGPEnricher()

    # Per-timestep error accumulators
    errors_per_step = [[] for _ in range(8)]   # up to 8 future steps
    l2_errors = []                              # overall
    outliers = []                               # high-error samples

    skipped = 0
    for idx in tqdm(indices, desc="Testing projection round-trip"):
        raw = adapter.extract_sample(idx)
        sample = build_sample_ir(raw)

        # Compute trajectory
        sample = compute_sample_trajectory(sample)
        if sample is None or sample.trajectory is None:
            skipped += 1
            continue

        # Get original future BEV (x, y) — before projection
        future_orig = np.array([
            [p.x, p.y] for p in sample.trajectory if p.t > 1e-9
        ])

        # Run projection
        calib = proj_enricher._get_front_calib(sample)
        if calib is None:
            skipped += 1
            continue

        if len(future_orig) == 0:
            skipped += 1
            continue

        # Step 1: BEV → pixels (original 1080×1920)
        bev_poses = np.array([
            [p.x, p.y, p.heading] for p in sample.trajectory if p.t > 1e-9
        ])
        projected = proj_enricher._project_to_image(bev_poses, calib)

        # Step 2: Scale to target size (1080×1920 → 448×784)
        projected = proj_enricher._scale_pixels(
            projected, proj_enricher.image_height, proj_enricher.image_width,
        )

        # Step 3: Quantize in target space (what the model sees)
        projected_q = proj_enricher._quantize_pixels(
            projected.copy(),
            proj_enricher.target_height,
            proj_enricher.target_width,
        )

        # Step 4: Unscale back to original (448×784 → 1080×1920)
        projected_orig = projected_q.copy()
        projected_orig[:, 0] = projected_orig[:, 0] / proj_enricher.target_width * proj_enricher.image_width
        projected_orig[:, 1] = projected_orig[:, 1] / proj_enricher.target_height * proj_enricher.image_height

        # Step 5: Inverse project back to ground (with original intrinsics)
        recovered = []
        for px in projected_orig:
            pt = proj_enricher._pixel_to_ground(px, calib)
            if pt is not None:
                recovered.append(pt)
            else:
                recovered.append([np.nan, np.nan])

        recovered = np.array(recovered)

        # Compute per-point L2 errors
        n_steps = min(len(future_orig), len(recovered))
        for i in range(n_steps):
            if np.isnan(recovered[i][0]):
                continue
            err = np.linalg.norm(future_orig[i] - recovered[i])
            if i < len(errors_per_step):
                errors_per_step[i].append(err)
            l2_errors.append(err)

        # Track high-error samples
        max_err = max(l2_errors[-n_steps:]) if l2_errors else 0
        if max_err > 1.0:
            outliers.append({
                "sample_id": sample.sample_id,
                "max_error": float(max_err),
                "orig": future_orig[:n_steps].tolist(),
                "recovered": recovered[:n_steps].tolist(),
            })

    return {
        "total_samples": num_samples,
        "processed": num_samples - skipped,
        "skipped": skipped,
        "errors_per_step": errors_per_step,
        "l2_errors": l2_errors,
        "outliers": outliers,
    }


def print_stats(results: dict):
    """Print error statistics."""
    print(f"\n{'='*60}")
    print("Projection Round-Trip Error Analysis")
    print(f"{'='*60}")
    print(f"Samples: {results['total_samples']} total, "
          f"{results['processed']} processed, {results['skipped']} skipped")

    l2 = np.array(results["l2_errors"])
    if len(l2) == 0:
        print("No valid errors computed!")
        return

    print(f"\nOverall L2 Error (meters):")
    print(f"  Mean:   {np.mean(l2):.4f}")
    print(f"  Median: {np.median(l2):.4f}")
    print(f"  Std:    {np.std(l2):.4f}")
    print(f"  Max:    {np.max(l2):.4f}")
    print(f"  P90:    {np.percentile(l2, 90):.4f}")
    print(f"  P99:    {np.percentile(l2, 99):.4f}")
    print(f"  P99.9:  {np.percentile(l2, 99.9):.4f}")

    print(f"\nPer-timestep L2 Error (meters):")
    for i, errs in enumerate(results["errors_per_step"]):
        if errs:
            e = np.array(errs)
            print(f"  Step {i+1} (t={0.5*(i+1):.1f}s): "
                  f"mean={np.mean(e):.4f}, median={np.median(e):.4f}, "
                  f"p90={np.percentile(e, 90):.4f}, max={np.max(e):.4f}")
        else:
            print(f"  Step {i+1}: no data")

    print(f"\nOutliers (max error > 1.0m): {len(results['outliers'])}")
    if results["outliers"]:
        for o in results["outliers"][:5]:
            print(f"  {o['sample_id']}: max_error={o['max_error']:.2f}m")
        if len(results["outliers"]) > 5:
            print(f"  ... and {len(results['outliers']) - 5} more")


def plot_errors(results: dict, output_path: str = "projection_errors.png"):
    """Plot error distributions."""
    l2 = np.array(results["l2_errors"])
    if len(l2) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    ax = axes[0]
    ax.hist(np.clip(l2, 0, 5), bins=100, color="steelblue", edgecolor="white")
    ax.axvline(np.median(l2), color="red", linestyle="--", label=f"Median: {np.median(l2):.3f}m")
    ax.axvline(np.mean(l2), color="orange", linestyle="--", label=f"Mean: {np.mean(l2):.3f}m")
    ax.set_xlabel("L2 Error (m)")
    ax.set_ylabel("Count")
    ax.set_title("Round-Trip Projection Error Distribution")
    ax.legend()

    # Per-timestep box plot
    ax = axes[1]
    step_data = []
    step_labels = []
    for i, errs in enumerate(results["errors_per_step"]):
        if errs:
            step_data.append(errs)
            step_labels.append(f"t+{0.5*(i+1):.1f}s")
    ax.boxplot(step_data, labels=step_labels)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("L2 Error (m)")
    ax.set_title("Error by Timestep")

    # CDF
    ax = axes[2]
    sorted_errs = np.sort(l2)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
    ax.plot(sorted_errs, cdf, color="steelblue")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Median")
    ax.axhline(0.9, color="orange", linestyle="--", alpha=0.5, label="P90")
    ax.axhline(0.99, color="green", linestyle="--", alpha=0.5, label="P99")
    ax.set_xlabel("L2 Error (m)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Error CDF")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test projection round-trip error across the dataset"
    )
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to test (default: all)")
    parser.add_argument("--output", type=str, default="projection_errors.png",
                        help="Output plot path")
    parser.add_argument("--save-outliers", type=str, default=None,
                        help="Save outlier details to JSON file")
    args = parser.parse_args()

    # Load dataset config
    import yaml
    ds_yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "datasets", "navsim.yaml"
    )
    with open(ds_yaml_path) as f:
        adapter_cfg = yaml.safe_load(f)
    adapter_cfg["split"] = args.split

    print(f"Initializing navsim adapter (split={args.split})...")
    adapter = NavsimAdapter(adapter_cfg)
    print(f"Dataset size: {adapter.get_sample_count()} samples")

    results = compute_roundtrip_errors(
        adapter,
        num_samples=args.num_samples,
    )

    print_stats(results)
    plot_errors(results, args.output)

    if args.save_outliers:
        with open(args.save_outliers, "w") as f:
            json.dump(results["outliers"], f, indent=2)
        print(f"Outliers saved to: {args.save_outliers}")

    # Return exit code based on error quality
    l2 = np.array(results["l2_errors"])
    if len(l2) > 0 and np.median(l2) > 5.0:
        print("\nWARNING: Median error > 5.0m — projection may have systematic issues!")
        sys.exit(1)


if __name__ == "__main__":
    main()
