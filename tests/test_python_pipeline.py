from __future__ import annotations

import csv
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from python.csv_io import read_measurements
from python.plot_results import _matched_errors, _measurement_xy, plot_results
from python.runner import build_tracker_command, main
from python.simulator import (
    MEASUREMENT_FIELDS,
    SensorSpec,
    SimulationConfig,
    TargetSpec,
    default_sensors,
    default_targets,
    simulate_measurements,
    simulate_truth,
    write_scenario,
)


class SimulatorTests(unittest.TestCase):
    def test_truth_includes_both_endpoints(self) -> None:
        rows = simulate_truth(
            [TargetSpec(3, 1.0, 2.0, 4.0, -2.0)],
            SimulationConfig(duration=2.0, dt=0.5, seed=1),
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual([row["timestamp"] for row in rows], [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual((rows[-1]["x"], rows[-1]["y"]), (9.0, -2.0))

    def test_sensor_relative_polar_measurement_carries_pose_and_noise(self) -> None:
        truth = simulate_truth(
            [TargetSpec(9, 13.0, 14.0, 0.0, 0.0)],
            SimulationConfig(duration=0.0, dt=1.0),
        )
        sensor = SensorSpec(
            4, 10.0, 10.0, range_std=0.0, bearing_std=0.0,
            position_std=0.0, detection_probability=1.0, clutter_rate=0.0,
        )
        rows = simulate_measurements(truth, [sensor], seed=123)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(row["value1"], 5.0)
        self.assertAlmostEqual(row["value2"], math.atan2(4.0, 3.0))
        self.assertEqual((row["sensor_x"], row["sensor_y"]), (10.0, 10.0))
        self.assertEqual(row["range_std"], 0.0)
        self.assertEqual(_measurement_xy(row, {}), (13.0, 14.0))

    def test_generation_is_byte_deterministic_and_has_two_sensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            args = (default_targets(3), default_sensors(second_sensor=True),
                    SimulationConfig(duration=3.0, dt=1.0, seed=99))
            paths_a = write_scenario(first, *args)
            paths_b = write_scenario(second, *args)
            for path_a, path_b in zip(paths_a, paths_b):
                self.assertEqual(path_a.read_bytes(), path_b.read_bytes())
            measurements = read_measurements(paths_a[1])
            self.assertEqual({row["sensor_id"] for row in measurements}, {0, 1})
            with paths_a[1].open(newline="", encoding="utf-8") as handle:
                self.assertEqual(tuple(next(csv.reader(handle))), MEASUREMENT_FIELDS)
            metadata = json.loads(paths_a[2].read_text(encoding="utf-8"))
            self.assertEqual(len(metadata["targets"]), 3)
            self.assertEqual(len(metadata["sensors"]), 2)

    def test_invalid_configuration_and_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target count"):
            default_targets(0)
        with self.assertRaisesRegex(ValueError, "dt"):
            SimulationConfig(dt=0.0)
        duplicate_targets = [
            TargetSpec(1, 0, 0, 0, 0), TargetSpec(1, 1, 1, 0, 0),
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            simulate_truth(duplicate_targets, SimulationConfig())


class PlotAndRunnerTests(unittest.TestCase):
    def test_error_matching_uses_explicit_truth_identity(self) -> None:
        truth = [
            {"timestamp": 0.0, "target_id": 1, "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0},
            {"timestamp": 0.0, "target_id": 2, "x": 100.0, "y": 0.0, "vx": 0.0, "vy": 1.0},
        ]
        estimates = [{
            "timestamp": 0.0, "track_id": 7, "x": 99.0, "y": 0.0,
            "vx": 0.0, "vy": 0.0, "truth_target_id": 1,
        }]
        errors = _matched_errors(truth, estimates)
        self.assertAlmostEqual(errors[7][0][1], 99.0)
        self.assertAlmostEqual(errors[7][0][2], 1.0)

    def test_plot_results_creates_nonempty_path_and_error_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth, measurements, metadata = write_scenario(
                root, [TargetSpec(0, 100, 50, 1, 2)],
                [SensorSpec(0, 0, 0, detection_probability=1.0, clutter_rate=0.0)],
                SimulationConfig(duration=2.0, dt=1.0, seed=2),
            )
            estimates = root / "estimates.csv"
            estimates.write_text(
                "timestamp,track_id,x,y,vx,vy,truth_target_id\n"
                "0,5,101,50,1,2,0\n1,5,102,52,1,2,0\n",
                encoding="utf-8",
            )
            outputs = plot_results(truth, measurements, estimates, metadata, root / "plots")
            self.assertEqual({path.name for path in outputs}, {"paths.png", "errors.png"})
            self.assertTrue(all(path.stat().st_size > 1000 for path in outputs))

    def test_tracker_command_expands_paths_and_appends_only_missing_flags(self) -> None:
        paths = tuple(Path(f"/{name}.csv") for name in ("truth", "measurements", "estimates", "metadata"))
        command = build_tracker_command("tracker --measurements {measurements}", *paths)
        self.assertEqual(command, [
            "tracker", "--measurements", "/measurements.csv",
            "--output", "/estimates.csv",
        ])

    def test_runner_cli_generates_deterministic_scenario_without_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenario"
            with contextlib.redirect_stdout(io.StringIO()):
                result = main([
                    "--output-dir", str(output), "--duration", "1", "--dt", "0.5",
                    "--targets", "2", "--seed", "17", "--skip-plots",
                ])
            self.assertEqual(result, 0)
            self.assertTrue((output / "truth.csv").is_file())
            self.assertTrue((output / "measurements.csv").is_file())
            self.assertTrue((output / "scenario.json").is_file())


if __name__ == "__main__":
    unittest.main()
