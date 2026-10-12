"""Deterministic 2-D target and radar measurement simulation.

CSV interchange schema (also described by :func:`write_scenario`):

``truth.csv``
    timestamp,target_id,x,y,vx,vy

``measurements.csv``
    timestamp,sensor_id,measurement_id,type,value1,value2,true_target_id,
    sensor_x,sensor_y,range_std,bearing_std,position_std

For ``cartesian`` rows value1/value2 are global x/y in metres.  For ``polar``
rows they are range in metres and bearing in radians, measured counter-clockwise
from the global +x axis at the named sensor.  A true_target_id of -1 means the
measurement is clutter.  Rows are sorted by time, sensor, then measurement id.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TRUTH_FIELDS = ("timestamp", "target_id", "x", "y", "vx", "vy")
MEASUREMENT_FIELDS = (
    "timestamp", "sensor_id", "measurement_id", "type", "value1", "value2",
    "true_target_id", "sensor_x", "sensor_y", "range_std", "bearing_std",
    "position_std",
)


@dataclass(frozen=True)
class TargetSpec:
    target_id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    turn_rate: float = 0.0  # radians / second

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.vx, self.vy, self.ax, self.ay, self.turn_rate)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target state and motion parameters must be finite")


@dataclass(frozen=True)
class SensorSpec:
    sensor_id: int
    x: float
    y: float
    kind: str = "polar"
    range_std: float = 8.0
    bearing_std: float = math.radians(0.35)
    position_std: float = 10.0
    detection_probability: float = 0.96
    clutter_rate: float = 1.5
    max_range: float = 4000.0

    def __post_init__(self) -> None:
        if self.kind not in {"polar", "cartesian"}:
            raise ValueError("sensor kind must be 'polar' or 'cartesian'")
        values = (self.x, self.y, self.range_std, self.bearing_std,
                  self.position_std, self.clutter_rate, self.max_range)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sensor parameters must be finite")
        if min(self.range_std, self.bearing_std, self.position_std,
               self.clutter_rate) < 0.0:
            raise ValueError("sensor noise and clutter parameters must be non-negative")
        if self.max_range <= 0.0:
            raise ValueError("max_range must be positive")
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0, 1]")


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 60.0
    dt: float = 1.0
    seed: int = 7

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration) or self.duration < 0.0:
            raise ValueError("duration must be finite and non-negative")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")


def default_targets(count: int = 4) -> list[TargetSpec]:
    """Return repeatable, separated targets with mild maneuvers."""
    if count < 1:
        raise ValueError("target count must be at least one")
    base = [
        TargetSpec(0, -1250, -650, 32, 15, 0.08, 0.02, 0.0020),
        TargetSpec(1, -850, 1050, 24, -22, 0.02, -0.04, -0.0015),
        TargetSpec(2, 950, -1100, -20, 28, -0.04, 0.06, 0.0010),
        TargetSpec(3, 1350, 750, -31, -12, 0.03, -0.02, -0.0022),
    ]
    if count <= len(base):
        return base[:count]
    targets = list(base)
    for target_id in range(len(base), count):
        angle = 2.0 * math.pi * target_id / count
        targets.append(TargetSpec(
            target_id, 1450 * math.cos(angle), 1450 * math.sin(angle),
            -25 * math.sin(angle), 25 * math.cos(angle),
            turn_rate=(0.001 if target_id % 2 == 0 else -0.001),
        ))
    return targets


def default_sensors(second_sensor: bool = True, cartesian: bool = False) -> list[SensorSpec]:
    """Return a central radar and, optionally, a spatially separated radar."""
    kind = "cartesian" if cartesian else "polar"
    sensors = [SensorSpec(0, 0.0, 0.0, kind=kind)]
    if second_sensor:
        sensors.append(SensorSpec(
            1, 1800.0, -900.0, kind=kind, range_std=11.0,
            bearing_std=math.radians(0.45), position_std=13.0,
            detection_probability=0.93, clutter_rate=1.0,
        ))
    return sensors


def _advance(state: np.ndarray, spec: TargetSpec, dt: float) -> np.ndarray:
    x, y, vx, vy = state
    if spec.turn_rate:
        theta = spec.turn_rate * dt
        c, s = math.cos(theta), math.sin(theta)
        vx, vy = c * vx - s * vy, s * vx + c * vy
    x += vx * dt + 0.5 * spec.ax * dt * dt
    y += vy * dt + 0.5 * spec.ay * dt * dt
    vx += spec.ax * dt
    vy += spec.ay * dt
    return np.array((x, y, vx, vy), dtype=float)


def simulate_truth(targets: Sequence[TargetSpec], config: SimulationConfig) -> list[dict]:
    if len({target.target_id for target in targets}) != len(targets):
        raise ValueError("target_id values must be unique")
    states = {t.target_id: np.array((t.x, t.y, t.vx, t.vy), dtype=float) for t in targets}
    rows: list[dict] = []
    steps = int(math.floor(config.duration / config.dt + 1e-12))
    for step in range(steps + 1):
        timestamp = step * config.dt
        for spec in targets:
            state = states[spec.target_id]
            rows.append(dict(zip(TRUTH_FIELDS, (timestamp, spec.target_id, *state))))
            if step < steps:
                states[spec.target_id] = _advance(state, spec, config.dt)
    return rows


def simulate_measurements(
    truth: Sequence[dict], sensors: Sequence[SensorSpec], seed: int = 7,
) -> list[dict]:
    """Create noisy detections and Poisson clutter using a local seeded RNG."""
    if len({sensor.sensor_id for sensor in sensors}) != len(sensors):
        raise ValueError("sensor_id values must be unique")
    rng = np.random.default_rng(seed)
    by_time: dict[float, list[dict]] = {}
    for row in truth:
        by_time.setdefault(float(row["timestamp"]), []).append(row)
    rows: list[dict] = []
    measurement_id = 0
    for timestamp in sorted(by_time):
        for sensor in sorted(sensors, key=lambda item: item.sensor_id):
            scan: list[tuple[float, float, int]] = []
            for target in by_time[timestamp]:
                dx = float(target["x"]) - sensor.x
                dy = float(target["y"]) - sensor.y
                distance = math.hypot(dx, dy)
                if distance > sensor.max_range or rng.random() > sensor.detection_probability:
                    continue
                if sensor.kind == "polar":
                    scan.append((
                        max(0.0, distance + rng.normal(0, sensor.range_std)),
                        _wrap_angle(math.atan2(dy, dx) + rng.normal(0, sensor.bearing_std)),
                        int(target["target_id"]),
                    ))
                else:
                    scan.append((
                        float(target["x"]) + rng.normal(0, sensor.position_std),
                        float(target["y"]) + rng.normal(0, sensor.position_std),
                        int(target["target_id"]),
                    ))
            for _ in range(rng.poisson(sensor.clutter_rate)):
                # Uniform over sensor coverage area, not biased toward long range.
                clutter_range = sensor.max_range * math.sqrt(rng.random())
                clutter_bearing = rng.uniform(-math.pi, math.pi)
                if sensor.kind == "polar":
                    scan.append((clutter_range, clutter_bearing, -1))
                else:
                    scan.append((
                        sensor.x + clutter_range * math.cos(clutter_bearing),
                        sensor.y + clutter_range * math.sin(clutter_bearing), -1,
                    ))
            rng.shuffle(scan)
            for value1, value2, target_id in scan:
                rows.append({
                    "timestamp": timestamp, "sensor_id": sensor.sensor_id,
                    "measurement_id": measurement_id, "type": sensor.kind,
                    "value1": value1, "value2": value2,
                    "true_target_id": target_id,
                    "sensor_x": sensor.x, "sensor_y": sensor.y,
                    "range_std": sensor.range_std,
                    "bearing_std": sensor.bearing_std,
                    "position_std": sensor.position_std,
                })
                measurement_id += 1
    return rows


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Explicit LF keeps the final header name clean for simple C++ getline
        # readers on every platform (the csv module otherwise emits CRLF).
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_scenario(
    output_dir: str | Path, targets: Sequence[TargetSpec], sensors: Sequence[SensorSpec],
    config: SimulationConfig,
) -> tuple[Path, Path, Path]:
    """Generate and write truth, measurements, and scenario metadata."""
    output = Path(output_dir)
    truth = simulate_truth(targets, config)
    measurements = simulate_measurements(truth, sensors, seed=config.seed)
    truth_path = output / "truth.csv"
    measurement_path = output / "measurements.csv"
    metadata_path = output / "scenario.json"
    _write_csv(truth_path, TRUTH_FIELDS, truth)
    _write_csv(measurement_path, MEASUREMENT_FIELDS, measurements)
    metadata = {
        "schema_version": 2,
        "units": {"timestamp": "s", "position": "m", "velocity": "m/s", "bearing": "rad"},
        "seed": config.seed, "duration": config.duration, "dt": config.dt,
        "truth_columns": list(TRUTH_FIELDS),
        "measurement_columns": list(MEASUREMENT_FIELDS),
        "targets": [asdict(target) for target in targets],
        "sensors": [asdict(sensor) for sensor in sensors],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return truth_path, measurement_path, metadata_path
