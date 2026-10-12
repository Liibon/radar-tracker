"""Plot paths and estimation errors from simulator/C++ tracker CSV files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:  # Support both ``python -m python.plot_results`` and direct execution.
    from .csv_io import read_estimates, read_measurements, read_truth
except ImportError:  # pragma: no cover - exercised by command-line usage
    from csv_io import read_estimates, read_measurements, read_truth


def _group(rows: list[dict], key: str) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row[key])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: row["timestamp"])
    return grouped


def _measurement_xy(row: dict, sensors: dict[int, dict]) -> tuple[float, float]:
    if row["type"] == "cartesian":
        return row["value1"], row["value2"]
    sensor = sensors.get(
        row["sensor_id"],
        {"x": row.get("sensor_x", 0.0), "y": row.get("sensor_y", 0.0)},
    )
    return (
        float(sensor["x"]) + row["value1"] * math.cos(row["value2"]),
        float(sensor["y"]) + row["value1"] * math.sin(row["value2"]),
    )


def _matched_errors(truth: list[dict], estimates: list[dict]) -> dict[int, list[tuple]]:
    truth_at: dict[float, list[dict]] = defaultdict(list)
    for row in truth:
        truth_at[round(row["timestamp"], 9)].append(row)
    errors: dict[int, list[tuple]] = defaultdict(list)
    for estimate in estimates:
        candidates = truth_at.get(round(estimate["timestamp"], 9), [])
        if "truth_target_id" in estimate:
            candidates = [r for r in candidates if r["target_id"] == estimate["truth_target_id"]]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda r: (r["x"] - estimate["x"]) ** 2 + (r["y"] - estimate["y"]) ** 2)
        position_error = math.hypot(nearest["x"] - estimate["x"], nearest["y"] - estimate["y"])
        velocity_error = math.hypot(nearest["vx"] - estimate["vx"], nearest["vy"] - estimate["vy"])
        errors[estimate["track_id"]].append((estimate["timestamp"], position_error, velocity_error))
    return errors


def plot_results(
    truth_path: str | Path, measurement_path: str | Path,
    estimate_path: str | Path | None, metadata_path: str | Path | None,
    output_dir: str | Path,
) -> list[Path]:
    truth = read_truth(truth_path)
    measurements = read_measurements(measurement_path)
    estimates = read_estimates(estimate_path) if estimate_path and Path(estimate_path).exists() else []
    sensors: dict[int, dict] = {}
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        sensors = {int(sensor["sensor_id"]): sensor for sensor in metadata.get("sensors", [])}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps["tab10"]
    for target_id, rows in _group(truth, "target_id").items():
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], lw=2,
                color=colors(target_id % 10), label=f"truth {target_id}")
    detections = [r for r in measurements if r["true_target_id"] >= 0]
    clutter = [r for r in measurements if r["true_target_id"] < 0]
    if detections:
        points = np.array([_measurement_xy(r, sensors) for r in detections])
        ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.28, c="0.35", label="detections")
    if clutter:
        points = np.array([_measurement_xy(r, sensors) for r in clutter])
        ax.scatter(points[:, 0], points[:, 1], s=9, alpha=0.22, c="crimson", marker="x", label="clutter")
    for track_id, rows in _group(estimates, "track_id").items():
        ax.plot([r["x"] for r in rows], [r["y"] for r in rows], "--", lw=1.7,
                label=f"estimate {track_id}")
    for sensor_id, sensor in sensors.items():
        ax.scatter(sensor["x"], sensor["y"], marker="^", s=80, edgecolor="black", label=f"sensor {sensor_id}")
    ax.set(title="Target trajectories and radar observations", xlabel="x (m)", ylabel="y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    paths_path = output / "paths.png"
    fig.savefig(paths_path, dpi=150)
    plt.close(fig)

    errors_path = output / "errors.png"
    errors = _matched_errors(truth, estimates)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for track_id, values in errors.items():
        t = [v[0] for v in values]
        axes[0].plot(t, [v[1] for v in values], label=f"track {track_id}")
        axes[1].plot(t, [v[2] for v in values], label=f"track {track_id}")
    axes[0].set_ylabel("position error (m)")
    axes[1].set_ylabel("velocity error (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[0].set_title("Nearest-truth estimation error")
    for ax in axes:
        ax.grid(alpha=0.25)
    if errors:
        axes[0].legend(fontsize=8, ncol=2)
    else:
        axes[0].text(0.5, 0.5, "No estimates supplied", transform=axes[0].transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(errors_path, dpi=150)
    plt.close(fig)
    return [paths_path, errors_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--estimates")
    parser.add_argument("--metadata")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    for path in plot_results(args.truth, args.measurements, args.estimates, args.metadata, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
