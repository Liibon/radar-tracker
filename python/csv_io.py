"""Typed CSV readers shared by plotting scripts."""

from __future__ import annotations

import csv
from pathlib import Path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_truth(path: str | Path) -> list[dict]:
    return [{
        "timestamp": float(r["timestamp"]), "target_id": int(r["target_id"]),
        "x": float(r["x"]), "y": float(r["y"]),
        "vx": float(r["vx"]), "vy": float(r["vy"]),
    } for r in read_rows(path)]


def read_measurements(path: str | Path) -> list[dict]:
    return [{
        "timestamp": float(r["timestamp"]), "sensor_id": int(r["sensor_id"]),
        "measurement_id": int(r["measurement_id"]), "type": r["type"],
        "value1": float(r["value1"]), "value2": float(r["value2"]),
        "true_target_id": int(r.get("true_target_id", -1) or -1),
        # Defaults retain compatibility with early single-origin CSV files.
        "sensor_x": float(r.get("sensor_x", 0.0) or 0.0),
        "sensor_y": float(r.get("sensor_y", 0.0) or 0.0),
        "range_std": float(r.get("range_std", 8.0) or 8.0),
        "bearing_std": float(r.get("bearing_std", 0.006108652381980153) or 0.006108652381980153),
        "position_std": float(r.get("position_std", 10.0) or 10.0),
    } for r in read_rows(path)]


def read_estimates(path: str | Path) -> list[dict]:
    rows = []
    for raw in read_rows(path):
        row = {
            "timestamp": float(raw["timestamp"]), "track_id": int(raw["track_id"]),
            "x": float(raw["x"]), "y": float(raw["y"]),
            "vx": float(raw["vx"]), "vy": float(raw["vy"]),
        }
        if raw.get("truth_target_id", "") != "":
            row["truth_target_id"] = int(raw["truth_target_id"])
        rows.append(row)
    return rows
