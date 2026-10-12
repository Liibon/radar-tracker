"""Plot tracker benchmark results from a small, implementation-neutral CSV."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_performance(input_path: str | Path, output_path: str | Path) -> Path:
    """Plot CSV columns method,objects,elapsed_ms[,frames]."""
    with Path(input_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("performance CSV contains no rows")
    grouped: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["method"]][int(row["objects"])].append(row)
    method_names = sorted(grouped)
    baseline_name = next(
        (name for name in method_names if "serial" in name.lower()),
        next((name for name in method_names if "cpu" in name.lower()), method_names[0]),
    )
    baseline = {
        objects: median(float(row["elapsed_ms"]) for row in samples)
        for objects, samples in grouped[baseline_name].items()
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for method, by_objects in sorted(grouped.items()):
        objects = sorted(by_objects)
        elapsed = [median(float(r["elapsed_ms"]) for r in by_objects[n]) for n in objects]
        frames = [median(float(r.get("frames", 1) or 1) for r in by_objects[n]) for n in objects]
        axes[0].plot(objects, elapsed, marker="o", label=method)
        axes[1].plot(objects, [1000.0 * f / e for f, e in zip(frames, elapsed)], marker="o", label=method)
        common = [(n, baseline[n] / e) for n, e in zip(objects, elapsed) if n in baseline]
        axes[2].plot([item[0] for item in common], [item[1] for item in common], marker="o", label=method)
    axes[0].set(title="Tracker latency", xlabel="objects", ylabel="elapsed (ms)")
    axes[1].set(title="Tracker throughput", xlabel="objects", ylabel="frames / second")
    axes[2].set(title=f"Speedup vs {baseline_name}", xlabel="objects", ylabel="speedup (x)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="CSV: method,objects,elapsed_ms[,frames]")
    parser.add_argument("--output", default="performance.png")
    args = parser.parse_args()
    print(plot_performance(args.input, args.output))


if __name__ == "__main__":
    main()
