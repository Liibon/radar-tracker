"""Run the complete tracker and evaluation pipeline on real RadarScenes data."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from .csv_io import read_estimates, read_measurements, read_truth
from .import_radarscenes import import_sequence
from .plot_results import plot_results


def evaluate(truth_path: Path, measurement_path: Path, estimate_path: Path) -> dict:
    truth = read_truth(truth_path)
    measurements = read_measurements(measurement_path)
    estimates = read_estimates(estimate_path)
    truth_lookup = {(round(row["timestamp"], 9), row["target_id"]): row for row in truth}
    errors: list[float] = []
    target_tracks: dict[int, set[int]] = defaultdict(set)
    identities_by_track: dict[int, set[int]] = defaultdict(set)
    for estimate in estimates:
        identity = estimate.get("truth_target_id", -1)
        identities_by_track[estimate["track_id"]].add(identity)
        if identity < 0:
            continue
        target_tracks[identity].add(estimate["track_id"])
        expected = truth_lookup.get((round(estimate["timestamp"], 9), identity))
        if expected:
            errors.append(math.hypot(estimate["x"] - expected["x"], estimate["y"] - expected["y"]))
    measured_targets = {row["true_target_id"] for row in measurements if row["true_target_id"] >= 0}
    false_tracks = sum(1 for values in identities_by_track.values() if not any(value >= 0 for value in values))
    metrics = {
        "dataset": "RadarScenes sequence_158",
        "real_sensor_data": True,
        "truth_is_pointwise_annotation_centroid_not_independent_position_ground_truth": True,
        "raw_measurement_clusters": len(measurements),
        "annotated_targets_in_sequence": len({row["target_id"] for row in truth}),
        "targets_surviving_label_free_detection_preprocessing": len(measured_targets),
        "targets_with_confirmed_tracks": len(target_tracks),
        "confirmed_track_ids": len(identities_by_track),
        "false_confirmed_tracks": false_tracks,
        "track_fragmentations": sum(max(0, len(track_ids) - 1) for track_ids in target_tracks.values()),
        "matched_estimate_samples": len(errors),
    }
    if errors:
        metrics.update({
            "position_rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
            "position_median_m": statistics.median(errors),
            "position_p95_m": sorted(errors)[int(0.95 * (len(errors) - 1))],
        })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("real_output"))
    parser.add_argument("--tracker", type=Path, default=Path("build/radar_tracker"))
    parser.add_argument("--gate", type=float, default=3.0)
    parser.add_argument("--max-misses", type=int, default=12)
    parser.add_argument("--confirm-hits", type=int, default=3)
    parser.add_argument("--min-cluster-points", type=int, default=2)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    truth, measurements, metadata = import_sequence(
        args.sequence_dir, args.output_dir, min_cluster_points=args.min_cluster_points,
    )
    estimates = args.output_dir / "estimates.csv"
    subprocess.run([
        str(args.tracker), "--measurements", str(measurements), "--output", str(estimates),
        "--gate", str(args.gate), "--max-misses", str(args.max_misses),
        "--confirm-hits", str(args.confirm_hits),
    ], check=True)
    metrics = evaluate(truth, measurements, estimates)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not args.skip_plots:
        for path in plot_results(
            truth, measurements, estimates, metadata, args.output_dir / "plots",
        ):
            print(f"plot: {path}")


if __name__ == "__main__":
    main()
