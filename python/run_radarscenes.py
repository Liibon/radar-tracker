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


def _measurement_xy(row: dict) -> tuple[float, float]:
    if row["type"] == "polar":
        return (
            row["sensor_x"] + row["value1"] * math.cos(row["value2"]),
            row["sensor_y"] + row["value1"] * math.sin(row["value2"]),
        )
    return row["value1"], row["value2"]


def _summary(values: list[float], prefix: str) -> dict:
    if not values:
        return {}
    return {
        f"{prefix}_rmse_m": math.sqrt(sum(value * value for value in values) / len(values)),
        f"{prefix}_median_m": statistics.median(values),
        f"{prefix}_p95_m": sorted(values)[int(0.95 * (len(values) - 1))],
    }


def evaluate(
    truth_path: Path, measurement_path: Path, estimate_path: Path,
    match_radius: float = 2.0,
) -> dict:
    """Evaluate without consulting labels carried through the tracker.

    Measurements are independently matched to the nearest annotation centroid.
    Estimates use deterministic, one-to-one greedy matching within each frame.
    This prevents ``true_target_id`` copied from an update from deciding whether
    the resulting track is correct.
    """
    if not math.isfinite(match_radius) or match_radius <= 0:
        raise ValueError("match_radius must be positive and finite")
    truth = read_truth(truth_path)
    measurements = read_measurements(measurement_path)
    estimates = read_estimates(estimate_path)
    truth_at: dict[float, list[dict]] = defaultdict(list)
    for row in truth:
        truth_at[round(row["timestamp"], 9)].append(row)

    raw_errors: list[float] = []
    measured_targets: set[int] = set()
    for measurement in measurements:
        x, y = _measurement_xy(measurement)
        candidates = [
            (math.hypot(x - row["x"], y - row["y"]), row["target_id"])
            for row in truth_at.get(round(measurement["timestamp"], 9), [])
        ]
        if candidates:
            distance, target_id = min(candidates)
            if distance <= match_radius:
                raw_errors.append(distance)
                measured_targets.add(target_id)

    estimates_at: dict[float, list[dict]] = defaultdict(list)
    for row in estimates:
        estimates_at[round(row["timestamp"], 9)].append(row)
    errors: list[float] = []
    target_tracks: dict[int, set[int]] = defaultdict(set)
    assignments_by_track: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for timestamp, frame_estimates in estimates_at.items():
        candidates: list[tuple[float, int, int, int, int]] = []
        frame_truth = truth_at.get(timestamp, [])
        for estimate_index, estimate in enumerate(frame_estimates):
            for truth_index, expected in enumerate(frame_truth):
                distance = math.hypot(
                    estimate["x"] - expected["x"], estimate["y"] - expected["y"],
                )
                if distance <= match_radius:
                    candidates.append((
                        distance, estimate["track_id"], expected["target_id"],
                        estimate_index, truth_index,
                    ))
        used_estimates: set[int] = set()
        used_truth: set[int] = set()
        for distance, track_id, target_id, estimate_index, truth_index in sorted(candidates):
            if estimate_index in used_estimates or truth_index in used_truth:
                continue
            used_estimates.add(estimate_index)
            used_truth.add(truth_index)
            errors.append(distance)
            target_tracks[target_id].add(track_id)
            assignments_by_track[track_id].append((timestamp, target_id))

    confirmed_track_ids = {row["track_id"] for row in estimates}
    false_track_ids = confirmed_track_ids - assignments_by_track.keys()
    identity_switches = 0
    for assignments in assignments_by_track.values():
        identities = [target_id for _, target_id in sorted(assignments)]
        identity_switches += sum(
            previous != current for previous, current in zip(identities, identities[1:])
        )
    annotated_targets = len({row["target_id"] for row in truth})
    detected_targets = len(measured_targets)
    confirmed_targets = len(target_tracks)
    metrics = {
        "dataset": "RadarScenes sequence_158",
        "real_sensor_data": True,
        "truth_is_pointwise_annotation_centroid_not_independent_position_ground_truth": True,
        "evaluation_matching": "label-independent one-to-one nearest centroid per frame",
        "evaluation_match_radius_m": match_radius,
        "raw_measurement_clusters": len(measurements),
        "raw_measurement_matched_samples": len(raw_errors),
        "annotated_targets_in_sequence": annotated_targets,
        "targets_surviving_label_free_detection_preprocessing": detected_targets,
        "detection_target_recall": detected_targets / annotated_targets if annotated_targets else 0.0,
        "targets_with_confirmed_tracks": confirmed_targets,
        "confirmed_target_recall": confirmed_targets / annotated_targets if annotated_targets else 0.0,
        "confirmation_rate_given_detection": confirmed_targets / detected_targets if detected_targets else 0.0,
        "confirmed_track_ids": len(confirmed_track_ids),
        "false_confirmed_tracks": len(false_track_ids),
        "false_confirmed_track_rate": len(false_track_ids) / len(confirmed_track_ids) if confirmed_track_ids else 0.0,
        "track_fragmentations": sum(max(0, len(track_ids) - 1) for track_ids in target_tracks.values()),
        "track_identity_switches": identity_switches,
        "matched_estimate_samples": len(errors),
    }
    metrics.update(_summary(raw_errors, "raw_measurement_position"))
    metrics.update(_summary(errors, "position"))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("real_output"))
    parser.add_argument("--tracker", type=Path, default=Path("build/radar_tracker"))
    parser.add_argument("--gate", type=float, default=2.0)
    parser.add_argument("--max-misses", type=int, default=12)
    parser.add_argument("--confirm-hits", type=int, default=3)
    parser.add_argument("--min-cluster-points", type=int, default=2)
    parser.add_argument("--radial-velocity-threshold", type=float, default=0.2)
    parser.add_argument("--cluster-radius", type=float, default=1.8)
    parser.add_argument("--truth-match-radius", type=float, default=2.0)
    parser.add_argument("--evaluation-match-radius", type=float, default=2.0)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    truth, measurements, metadata = import_sequence(
        args.sequence_dir, args.output_dir,
        radial_velocity_threshold=args.radial_velocity_threshold,
        cluster_radius=args.cluster_radius,
        truth_match_radius=args.truth_match_radius,
        min_cluster_points=args.min_cluster_points,
    )
    estimates = args.output_dir / "estimates.csv"
    subprocess.run([
        str(args.tracker), "--measurements", str(measurements), "--output", str(estimates),
        "--gate", str(args.gate), "--max-misses", str(args.max_misses),
        "--confirm-hits", str(args.confirm_hits),
    ], check=True)
    metrics = evaluate(truth, measurements, estimates, args.evaluation_match_radius)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not args.skip_plots:
        from .plot_results import plot_results

        for path in plot_results(
            truth, measurements, estimates, metadata, args.output_dir / "plots",
        ):
            print(f"plot: {path}")


if __name__ == "__main__":
    main()
