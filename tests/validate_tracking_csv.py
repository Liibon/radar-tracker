#!/usr/bin/env python3
"""Validate tracker CSV output and print position-error summary statistics."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Iterable


REQUIRED_COLUMNS = (
    "timestamp",
    "track_id",
    "truth_x",
    "truth_y",
    "estimate_x",
    "estimate_y",
)


class ValidationError(ValueError):
    """Raised when a tracking result violates the documented CSV contract."""


def _percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for sorted finite values."""
    if not values:
        raise ValidationError("CSV contains no data rows")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_and_summarize(path: pathlib.Path) -> dict[str, float | int]:
    errors: list[float] = []
    last_timestamp: dict[str, float] = defaultdict(lambda: -math.inf)

    try:
        source = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot open {path}: {exc}") from exc

    with source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValidationError(f"missing required columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            track_id = (row.get("track_id") or "").strip()
            if not track_id:
                raise ValidationError(f"line {line_number}: empty track_id")

            parsed: dict[str, float] = {}
            for name in REQUIRED_COLUMNS:
                if name == "track_id":
                    continue
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValidationError(
                        f"line {line_number}: {name} is not numeric"
                    ) from exc
                if not math.isfinite(value):
                    raise ValidationError(f"line {line_number}: {name} is not finite")
                parsed[name] = value

            timestamp = parsed["timestamp"]
            if timestamp < last_timestamp[track_id]:
                raise ValidationError(
                    f"line {line_number}: timestamp decreases for track {track_id}"
                )
            last_timestamp[track_id] = timestamp

            dx = parsed["estimate_x"] - parsed["truth_x"]
            dy = parsed["estimate_y"] - parsed["truth_y"]
            errors.append(math.hypot(dx, dy))

    if not errors:
        raise ValidationError("CSV contains no data rows")

    return {
        "rows": len(errors),
        "tracks": len(last_timestamp),
        "position_rmse": math.sqrt(statistics.fmean(error * error for error in errors)),
        "position_median": statistics.median(errors),
        "position_p95": _percentile(errors, 0.95),
        "position_max": max(errors),
    }


def _format_summary(items: Iterable[tuple[str, float | int]]) -> str:
    lines = []
    for name, value in items:
        rendered = str(value) if isinstance(value, int) else f"{value:.9g}"
        lines.append(f"{name}={rendered}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_and_summarize(args.csv_path)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    print(_format_summary(summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
