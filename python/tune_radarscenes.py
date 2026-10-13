"""Tune the label-free RadarScenes front end and tracker on one sequence."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

from .import_radarscenes import import_sequence
from .run_radarscenes import evaluate


def _numbers(text: str, conversion):
    return [conversion(value) for value in text.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--tracker", type=Path, default=Path("build/radar_tracker"))
    parser.add_argument("--output", type=Path, default=Path("real_tuning.json"))
    parser.add_argument("--radial-thresholds", default="0.2,0.35,0.5,0.75")
    parser.add_argument("--cluster-radii", default="1.0,1.4,1.8,2.2")
    parser.add_argument("--minimum-cluster-points", default="1,2")
    parser.add_argument("--gates", default="2.0,3.0,4.5")
    parser.add_argument("--confirmation-hits", default="2,3")
    parser.add_argument("--max-misses", type=int, default=12)
    parser.add_argument("--evaluation-match-radius", type=float, default=2.0)
    args = parser.parse_args()

    tracker = args.tracker.resolve()
    results: list[dict] = []
    preprocessing = itertools.product(
        _numbers(args.radial_thresholds, float),
        _numbers(args.cluster_radii, float),
        _numbers(args.minimum_cluster_points, int),
    )
    with tempfile.TemporaryDirectory(prefix="radar-tuning-") as temporary:
        root = Path(temporary)
        for index, (radial_threshold, cluster_radius, minimum_points) in enumerate(preprocessing):
            run_dir = root / str(index)
            truth, measurements, _ = import_sequence(
                args.sequence_dir, run_dir,
                radial_velocity_threshold=radial_threshold,
                cluster_radius=cluster_radius,
                min_cluster_points=minimum_points,
            )
            for gate, confirmation_hits in itertools.product(
                _numbers(args.gates, float), _numbers(args.confirmation_hits, int),
            ):
                estimates = run_dir / f"estimates-{gate}-{confirmation_hits}.csv"
                subprocess.run([
                    str(tracker), "--measurements", str(measurements),
                    "--output", str(estimates), "--gate", str(gate),
                    "--max-misses", str(args.max_misses),
                    "--confirm-hits", str(confirmation_hits), "--serial",
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                metrics = evaluate(
                    truth, measurements, estimates, args.evaluation_match_radius,
                )
                results.append({
                    "radial_velocity_threshold_mps": radial_threshold,
                    "cluster_radius_m": cluster_radius,
                    "minimum_returns_per_cluster": minimum_points,
                    "association_gate_m": gate,
                    "confirmation_hits": confirmation_hits,
                    **metrics,
                })

    # Select a conservative operating point: avoid false tracks and identity
    # switches before increasing coverage. Fragmentation and p95 break ties.
    results.sort(key=lambda row: (
        row["false_confirmed_tracks"],
        row["track_identity_switches"],
        -row["targets_with_confirmed_tracks"],
        row["track_fragmentations"],
        row.get("position_p95_m", float("inf")),
        row["confirmed_track_ids"],
    ))
    document = {
        "runs": len(results),
        "selection_policy": (
            "minimize false tracks and identity switches, then maximize confirmed "
            "targets, then minimize fragmentation, p95 error, and track count"
        ),
        "best": results[0] if results else None,
        "top": results[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["best"], indent=2))


if __name__ == "__main__":
    main()
