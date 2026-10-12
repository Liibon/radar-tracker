from __future__ import annotations

import pathlib
import tempfile
import unittest

from validate_measurements_csv import ValidationError, validate_measurements


FIXTURES = pathlib.Path(__file__).with_name("fixtures")


class MeasurementCsvValidationTests(unittest.TestCase):
    def test_valid_fixture_has_explicit_sensor_origins(self) -> None:
        summary = validate_measurements(FIXTURES / "valid_measurements.csv")
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["sensors"], 2)

    def test_sensor_x_is_required(self) -> None:
        self._assert_missing_pose_column("sensor_x")

    def test_sensor_y_is_required(self) -> None:
        self._assert_missing_pose_column("sensor_y")

    def test_blank_sensor_coordinate_is_rejected(self) -> None:
        text = self._valid_text().replace(",100,-20,", ",,-20,")
        self._assert_invalid(text, "sensor_x is not numeric")

    def test_nonfinite_sensor_coordinate_is_rejected(self) -> None:
        text = self._valid_text().replace(",100,-20,", ",100,nan,")
        self._assert_invalid(text, "sensor_y is not finite")

    def _assert_missing_pose_column(self, missing: str) -> None:
        lines = self._valid_text().splitlines()
        header = lines[0].split(",")
        index = header.index(missing)
        header.pop(index)
        values = lines[1].split(",")
        values.pop(index)
        self._assert_invalid(
            ",".join(header) + "\n" + ",".join(values) + "\n",
            f"missing required columns: {missing}",
        )

    def _valid_text(self) -> str:
        return (
            "timestamp,sensor_id,measurement_id,type,value1,value2,true_target_id,"
            "sensor_x,sensor_y,range_std,bearing_std,position_std\n"
            "1,b,9,polar,5,0,3,100,-20,2,0.1,7\n"
        )

    def _assert_invalid(self, text: str, message: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "measurements.csv"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, message):
                validate_measurements(path)


if __name__ == "__main__":
    unittest.main()
