from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from python.import_radarscenes import connected_clusters
from python.run_radarscenes import evaluate


class RadarScenesAdapterTests(unittest.TestCase):
    def test_connected_clusters_are_deterministic_and_transitive(self) -> None:
        points = np.array([[0.0, 0.0], [0.9, 0.0], [1.8, 0.0], [8.0, 1.0]])
        self.assertEqual(connected_clusters(points, 1.0), [[0, 1, 2], [3]])

    def test_empty_input_and_invalid_radius(self) -> None:
        self.assertEqual(connected_clusters(np.empty((0, 2)), 1.0), [])
        with self.assertRaisesRegex(ValueError, "positive"):
            connected_clusters(np.empty((0, 2)), 0.0)

    def test_evaluation_matches_geometry_instead_of_propagated_labels(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.csv"
            measurements = root / "measurements.csv"
            estimates = root / "estimates.csv"
            truth.write_text(
                "timestamp,target_id,x,y,vx,vy\n"
                "0,0,0,0,0,0\n0,1,10,0,0,0\n"
                "1,0,0,0,0,0\n1,1,10,0,0,0\n",
                encoding="utf-8",
            )
            measurements.write_text(
                "timestamp,sensor_id,measurement_id,type,value1,value2,true_target_id,"
                "sensor_x,sensor_y,range_std,bearing_std,position_std\n"
                "0,0,0,cartesian,0.1,0,-1,0,0,1,0.1,1\n"
                "0,0,1,cartesian,9.9,0,-1,0,0,1,0.1,1\n",
                encoding="utf-8",
            )
            estimates.write_text(
                "timestamp,track_id,x,y,vx,vy,truth_target_id\n"
                "0,5,0.2,0,0,0,1\n0,6,9.8,0,0,0,0\n"
                "1,5,0.1,0,0,0,1\n1,6,9.9,0,0,0,0\n",
                encoding="utf-8",
            )
            metrics = evaluate(truth, measurements, estimates, match_radius=1.0)
            self.assertEqual(metrics["targets_with_confirmed_tracks"], 2)
            self.assertEqual(metrics["false_confirmed_tracks"], 0)
            self.assertEqual(metrics["track_identity_switches"], 0)
            self.assertEqual(metrics["targets_surviving_label_free_detection_preprocessing"], 2)
            self.assertAlmostEqual(metrics["confirmed_target_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
