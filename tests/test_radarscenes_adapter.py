from __future__ import annotations

import unittest

import numpy as np

from python.import_radarscenes import connected_clusters


class RadarScenesAdapterTests(unittest.TestCase):
    def test_connected_clusters_are_deterministic_and_transitive(self) -> None:
        points = np.array([[0.0, 0.0], [0.9, 0.0], [1.8, 0.0], [8.0, 1.0]])
        self.assertEqual(connected_clusters(points, 1.0), [[0, 1, 2], [3]])

    def test_empty_input_and_invalid_radius(self) -> None:
        self.assertEqual(connected_clusters(np.empty((0, 2)), 1.0), [])
        with self.assertRaisesRegex(ValueError, "positive"):
            connected_clusters(np.empty((0, 2)), 0.0)


if __name__ == "__main__":
    unittest.main()
