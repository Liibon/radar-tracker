#!/usr/bin/env python3
"""Validate the self-contained radar measurement CSV interchange schema."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys


REQUIRED_COLUMNS = (
    "timestamp",
    "sensor_id",
    "measurement_id",
    "type",
    "value1",
    "value2",
    "true_target_id",
    "sensor_x",
    "sensor_y",
    "range_std",
    "bearing_std",
    "position_std",
)


class ValidationError(ValueError):
    """Raised when a measurement file violates the interchange contract."""


def _finite_float(row: dict[str, str], name: str, line_number: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"line {line_number}: {name} is not numeric") from exc
    if not math.isfinite(value):
        raise ValidationError(f"line {line_number}: {name} is not finite")
    return value


def _integer(row: dict[str, str], name: str, line_number: int) -> int:
    try:
        return int(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"line {line_number}: {name} is not an integer") from exc


def validate_measurements(path: pathlib.Path) -> dict[str, int | float]:
    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot open {path}: {exc}") from exc

    row_count = 0
    sensors: set[str] = set()
    measurement_keys: set[tuple[str, int]] = set()
    first_timestamp: float | None = None
    previous_timestamp = -math.inf

    with source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValidationError(f"missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            sensor_id = (row.get("sensor_id") or "").strip()
            if not sensor_id:
                raise ValidationError(f"line {line_number}: empty sensor_id")
            measurement_id = _integer(row, "measurement_id", line_number)
            _integer(row, "true_target_id", line_number)
            key = (sensor_id, measurement_id)
            if key in measurement_keys:
                raise ValidationError(
                    f"line {line_number}: duplicate measurement_id for sensor {sensor_id}"
                )
            measurement_keys.add(key)

            timestamp = _finite_float(row, "timestamp", line_number)
            if timestamp < previous_timestamp:
                raise ValidationError(f"line {line_number}: timestamp decreases")
            if first_timestamp is None:
                first_timestamp = timestamp
            previous_timestamp = timestamp

            measurement_type = (row.get("type") or "").strip().lower()
            if measurement_type not in {"cartesian", "polar"}:
                raise ValidationError(
                    f"line {line_number}: type must be cartesian or polar"
                )
            value1 = _finite_float(row, "value1", line_number)
            _finite_float(row, "value2", line_number)
            # Sensor origins are required even for an origin sensor. Requiring
            # explicit finite values prevents a missing pose from silently
            # changing the reference frame.
            _finite_float(row, "sensor_x", line_number)
            _finite_float(row, "sensor_y", line_number)
            for name in ("range_std", "bearing_std", "position_std"):
                if _finite_float(row, name, line_number) < 0.0:
                    raise ValidationError(f"line {line_number}: {name} is negative")
            if measurement_type == "polar" and value1 < 0.0:
                raise ValidationError(f"line {line_number}: polar range is negative")

            sensors.add(sensor_id)
            row_count += 1

    if row_count == 0:
        raise ValidationError("CSV contains no data rows")
    return {
        "rows": row_count,
        "sensors": len(sensors),
        "first_timestamp": first_timestamp if first_timestamp is not None else 0.0,
        "last_timestamp": previous_timestamp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_measurements(args.csv_path)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    for name, value in summary.items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
