"""Generate scenarios, invoke the C++ tracker, and create result plots."""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

try:  # Support both ``python -m python.runner`` and direct execution.
    from .simulator import SimulationConfig, default_sensors, default_targets, write_scenario
except ImportError:  # pragma: no cover - exercised by command-line usage
    from simulator import SimulationConfig, default_sensors, default_targets, write_scenario


def generate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    config = SimulationConfig(duration=args.duration, dt=args.dt, seed=args.seed)
    return write_scenario(
        args.output_dir, default_targets(args.targets),
        default_sensors(second_sensor=not args.one_sensor, cartesian=args.cartesian), config,
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least one")
    return parsed


def build_tracker_command(
    command: str, truth: Path, measurements: Path, estimates: Path, metadata: Path,
) -> list[str]:
    """Expand a tracker command and supply any omitted required path flags."""
    raw_tokens = shlex.split(command)
    substitutions = {
        "truth": truth, "measurements": measurements,
        "estimates": estimates, "metadata": metadata,
    }
    tokens = [
        token.format(**{name: str(path) for name, path in substitutions.items()})
        for token in raw_tokens
    ]
    if not any("{measurements}" in token for token in raw_tokens):
        tokens.extend(["--input", str(measurements)])
    if not any("{estimates}" in token for token in raw_tokens):
        tokens.extend(["--output", str(estimates)])
    return tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--duration", type=_nonnegative_float, default=60.0)
    parser.add_argument("--dt", type=_positive_float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--targets", type=_positive_int, default=4)
    parser.add_argument("--one-sensor", action="store_true")
    parser.add_argument("--cartesian", action="store_true", help="use Cartesian instead of range/bearing observations")
    parser.add_argument(
        "--tracker",
        help=("tracker command; {truth}, {measurements}, {estimates}, and "
              "{metadata} placeholders are supported"),
    )
    parser.add_argument("--estimates", type=Path, help="plot an existing estimates CSV instead of invoking tracker")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tracker and args.estimates:
        parser.error("--tracker and --estimates cannot be used together")
    if args.estimates and not args.estimates.is_file():
        parser.error(f"estimates CSV does not exist: {args.estimates}")

    truth, measurements, metadata = generate(args)
    estimates = args.estimates
    if args.tracker:
        estimates = args.output_dir / "estimates.csv"
        tokens = build_tracker_command(args.tracker, truth, measurements, estimates, metadata)
        subprocess.run(tokens, check=True)
    print(f"truth: {truth}")
    print(f"measurements: {measurements}")
    print(f"metadata: {metadata}")
    if not args.skip_plots:
        try:
            from .plot_results import plot_results
        except ImportError:  # pragma: no cover - direct script execution
            from plot_results import plot_results
        for path in plot_results(truth, measurements, estimates, metadata, args.output_dir / "plots"):
            print(f"plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
