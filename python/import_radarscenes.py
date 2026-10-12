"""Convert a RadarScenes sequence into this project's tracker CSV contract.

Radar detections are clustered without consulting labels.  RadarScenes track
IDs are used only to produce a separate truth table and evaluation-only IDs.
The C++ tracker never reads the truth table and never uses ``true_target_id``
when filtering or associating measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .simulator import MEASUREMENT_FIELDS, TRUTH_FIELDS


LABEL_NAMES = {
    0: "passenger_car", 1: "large_vehicle", 2: "truck", 3: "bus",
    4: "train", 5: "bicycle", 6: "motorized_two_wheeler",
    7: "pedestrian", 8: "pedestrian_group", 9: "animal",
    10: "other_dynamic", 11: "static_environment",
}


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def connected_clusters(points: np.ndarray, radius: float) -> list[list[int]]:
    """Return deterministic Euclidean connected components (label-free)."""
    if radius <= 0:
        raise ValueError("cluster radius must be positive")
    if len(points) == 0:
        return []
    seen = np.zeros(len(points), dtype=bool)
    result: list[list[int]] = []
    radius2 = radius * radius
    for root in range(len(points)):
        if seen[root]:
            continue
        seen[root] = True
        pending = [root]
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            distance2 = np.sum((points - points[current]) ** 2, axis=1)
            neighbors = np.flatnonzero((distance2 <= radius2) & ~seen)
            seen[neighbors] = True
            pending.extend(int(index) for index in neighbors)
        result.append(sorted(component))
    return result


def _truth_rows(radar: np.ndarray, scenes: dict, first_timestamp: int) -> tuple[list[dict], dict[bytes, int], dict]:
    track_ids = sorted(set(radar["track_id"]) - {b""})
    id_map = {track_id: index for index, track_id in enumerate(track_ids)}
    samples: dict[bytes, list[dict]] = defaultdict(list)
    classes: dict[int, set[int]] = defaultdict(set)
    for timestamp_text, scene in sorted(scenes.items(), key=lambda item: int(item[0])):
        begin, end = scene["radar_indices"]
        scan = radar[begin:end]
        timestamp = (int(timestamp_text) - first_timestamp) / 1e6
        for track_id in sorted(set(scan["track_id"]) - {b""}):
            values = scan[scan["track_id"] == track_id]
            finite = np.isfinite(values["x_seq"]) & np.isfinite(values["y_seq"])
            if not finite.any():
                continue
            values = values[finite]
            target_id = id_map[track_id]
            samples[track_id].append({
                "timestamp": timestamp, "target_id": target_id,
                "x": float(np.mean(values["x_seq"])),
                "y": float(np.mean(values["y_seq"])),
            })
            classes[target_id].update(int(value) for value in values["label_id"])

    rows: list[dict] = []
    for track_id, values in samples.items():
        for index, row in enumerate(values):
            if len(values) == 1:
                vx = vy = 0.0
            else:
                left = values[max(0, index - 1)]
                right = values[min(len(values) - 1, index + 1)]
                dt = right["timestamp"] - left["timestamp"]
                vx = (right["x"] - left["x"]) / dt if dt > 0 else 0.0
                vy = (right["y"] - left["y"]) / dt if dt > 0 else 0.0
            rows.append({**row, "vx": vx, "vy": vy})
    rows.sort(key=lambda row: (row["timestamp"], row["target_id"]))
    class_metadata = {
        str(target_id): [LABEL_NAMES.get(value, f"label_{value}") for value in sorted(values)]
        for target_id, values in classes.items()
    }
    return rows, id_map, class_metadata


def _measurement_rows(
    radar: np.ndarray, odometry: np.ndarray, scenes: dict, sensors: dict[int, dict], first_timestamp: int,
    id_map: dict[bytes, int], radial_velocity_threshold: float, cluster_radius: float,
    truth_match_radius: float, min_cluster_points: int,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    measurement_id = 0
    raw_count = dynamic_count = 0
    matched_count = 0
    for timestamp_text, scene in sorted(scenes.items(), key=lambda item: int(item[0])):
        begin, end = scene["radar_indices"]
        scan = radar[begin:end]
        raw_count += len(scan)
        finite = (
            np.isfinite(scan["vr_compensated"]) & np.isfinite(scan["x_seq"]) &
            np.isfinite(scan["y_seq"])
        )
        dynamic = finite & (np.abs(scan["vr_compensated"]) >= radial_velocity_threshold)
        selected = scan[dynamic]
        dynamic_count += len(selected)
        points = np.column_stack((selected["x_seq"], selected["y_seq"]))

        truth_centers: list[tuple[bytes, np.ndarray]] = []
        for track_id in sorted(set(scan["track_id"]) - {b""}):
            labeled = scan[scan["track_id"] == track_id]
            valid = np.isfinite(labeled["x_seq"]) & np.isfinite(labeled["y_seq"])
            if valid.any():
                truth_centers.append((track_id, np.array([
                    np.mean(labeled["x_seq"][valid]), np.mean(labeled["y_seq"][valid]),
                ])))

        pose = odometry[int(scene["odometry_index"])]
        timestamp = (int(timestamp_text) - first_timestamp) / 1e6
        sensor_id = int(scene["sensor_id"])
        mounting = sensors[sensor_id]
        ego_yaw = float(pose["yaw_seq"])
        cosine, sine = math.cos(ego_yaw), math.sin(ego_yaw)
        sensor_x = float(pose["x_seq"]) + cosine * mounting["x"] - sine * mounting["y"]
        sensor_y = float(pose["y_seq"]) + sine * mounting["x"] + cosine * mounting["y"]
        for indices in connected_clusters(points, cluster_radius):
            if len(indices) < min_cluster_points:
                continue
            cluster = points[indices]
            center = np.mean(cluster, axis=0)
            # RadarScenes x_seq/y_seq are already transformed with ego odometry.
            # Reconstruct a world-frame polar observation around the moving radar
            # origin so the existing offset-sensor EKF can consume it.
            dx, dy = float(center[0] - sensor_x), float(center[1] - sensor_y)
            distance = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx)
            spread = float(np.sqrt(np.mean(np.sum((cluster - center) ** 2, axis=1))))

            target_id = -1
            if truth_centers:
                track_id, nearest = min(truth_centers, key=lambda item: float(np.linalg.norm(center - item[1])))
                if float(np.linalg.norm(center - nearest)) <= truth_match_radius:
                    target_id = id_map[track_id]
                    matched_count += 1
            rows.append({
                "timestamp": timestamp, "sensor_id": str(sensor_id),
                "measurement_id": measurement_id, "type": "polar",
                "value1": distance, "value2": bearing,
                "true_target_id": target_id,
                "sensor_x": sensor_x, "sensor_y": sensor_y,
                "range_std": max(0.5, spread),
                "bearing_std": max(math.radians(0.5), math.atan2(max(0.25, spread), max(distance, 0.25))),
                "position_std": max(0.75, spread),
            })
            measurement_id += 1
    stats = {
        "raw_radar_returns": raw_count,
        "velocity_filtered_returns": dynamic_count,
        "unlabeled_clusters": len(rows),
        "evaluation_matched_clusters": matched_count,
    }
    return rows, stats


def import_sequence(
    sequence_dir: str | Path, output_dir: str | Path, *,
    radial_velocity_threshold: float = 0.5, cluster_radius: float = 1.8,
    truth_match_radius: float = 2.0, min_cluster_points: int = 2,
) -> tuple[Path, Path, Path]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - dependency diagnostic
        raise RuntimeError(
            "RadarScenes import requires h5py: pip install -r python/requirements.txt"
        ) from error
    source = Path(sequence_dir)
    output = Path(output_dir)
    scenes_document = json.loads((source / "scenes.json").read_text(encoding="utf-8"))
    sensor_document = json.loads((source / "sensors.json").read_text(encoding="utf-8"))
    sensors = {int(value["id"]): value for value in sensor_document.values()}
    scenes = scenes_document["scenes"]
    if min_cluster_points < 1:
        raise ValueError("min_cluster_points must be at least one")
    first_timestamp = int(scenes_document["first_timestamp"])
    with h5py.File(source / "radar_data.h5", "r") as handle:
        radar = handle["radar_data"][:]
        odometry = handle["odometry"][:]

    truth, id_map, classes = _truth_rows(radar, scenes, first_timestamp)
    measurements, stats = _measurement_rows(
        radar, odometry, scenes, sensors, first_timestamp, id_map,
        radial_velocity_threshold, cluster_radius, truth_match_radius, min_cluster_points,
    )
    truth_path = output / "truth.csv"
    measurement_path = output / "measurements.csv"
    metadata_path = output / "scenario.json"
    _write_csv(truth_path, TRUTH_FIELDS, truth)
    _write_csv(measurement_path, MEASUREMENT_FIELDS, measurements)
    metadata = {
        "schema_version": 2,
        "source": "RadarScenes real-world automotive radar data",
        "source_doi": "10.5281/zenodo.4559821",
        "source_url": "https://zenodo.org/records/4559821",
        "license": "CC BY-NC-SA 4.0",
        "sequence": scenes_document["sequence_name"],
        "category": scenes_document["category"],
        "duration_seconds": (int(scenes_document["last_timestamp"]) - first_timestamp) / 1e6,
        "processing": {
            "uses_labels_for_detection_or_clustering": False,
            "radial_velocity_threshold_mps": radial_velocity_threshold,
            "connected_component_radius_m": cluster_radius,
            "truth_match_radius_m": truth_match_radius,
            "minimum_returns_per_cluster": min_cluster_points,
        },
        "stats": stats,
        "truth_targets": len(id_map),
        "truth_velocity_reliable": False,
        "truth_note": "Positions are centroids of point-wise object annotations; velocities are finite-difference aids only.",
        "target_classes": classes,
        # Plotting expects fixed world-frame sensors. RadarScenes sensors move
        # with the ego vehicle, so preserve their vehicle-frame calibration
        # separately rather than drawing misleading stationary markers.
        "sensors": [],
        "sensor_mounts": [sensor_document[name] for name in sorted(sensor_document)],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return truth_path, measurement_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("real_output"))
    parser.add_argument("--radial-velocity-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-radius", type=float, default=1.8)
    parser.add_argument("--truth-match-radius", type=float, default=2.0)
    parser.add_argument("--min-cluster-points", type=int, default=2)
    args = parser.parse_args()
    paths = import_sequence(
        args.sequence_dir, args.output_dir,
        radial_velocity_threshold=args.radial_velocity_threshold,
        cluster_radius=args.cluster_radius,
        truth_match_radius=args.truth_match_radius,
        min_cluster_points=args.min_cluster_points,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
