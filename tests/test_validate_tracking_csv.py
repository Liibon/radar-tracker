from __future__ import annotations

import pathlib
import tempfile
import unittest

from validate_tracking_csv import ValidationError, validate_and_summarize


FIXTURES = pathlib.Path(__file__).with_name("fixtures")


class TrackingCsvValidationTests(unittest.TestCase):
    def test_valid_fixture_statistics(self) -> None:
        summary = validate_and_summarize(FIXTURES / "valid_tracking.csv")
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["tracks"], 2)
        self.assertAlmostEqual(summary["position_rmse"], 0.5)
        self.assertAlmostEqual(summary["position_median"], 0.5)

    def test_missing_column_is_rejected(self) -> None:
        self._assert_invalid(
            "timestamp,track_id,truth_x,truth_y,estimate_x\n0,a,0,0,0\n",
            "missing required columns",
        )

    def test_non_finite_value_is_rejected(self) -> None:
        self._assert_invalid(
            "timestamp,track_id,truth_x,truth_y,estimate_x,estimate_y\n"
            "0,a,0,0,nan,0\n",
            "not finite",
        )

    def test_decreasing_timestamp_per_track_is_rejected(self) -> None:
        self._assert_invalid(
            "timestamp,track_id,truth_x,truth_y,estimate_x,estimate_y\n"
            "1,a,0,0,0,0\n0,a,0,0,0,0\n",
            "timestamp decreases",
        )

    def test_interleaved_tracks_may_have_independent_timestamps(self) -> None:
        text = (
            "timestamp,track_id,truth_x,truth_y,estimate_x,estimate_y\n"
            "2,a,0,0,0,0\n0,b,0,0,0,0\n3,a,0,0,0,0\n1,b,0,0,0,0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tracking.csv"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(validate_and_summarize(path)["tracks"], 2)

    def _assert_invalid(self, text: str, message: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "tracking.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, message):
                validate_and_summarize(path)


if __name__ == "__main__":
    unittest.main()
