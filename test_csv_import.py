"""
Regression tests for the CSV import path.

Every case here is a shape a real household or meter export arrives in. The
values and, just as importantly, the timestamps are checked: an importer that
silently rewrites 2024-03-05 into 2024-05-03 is worse than one that refuses
the file.
"""
import io
import unittest

import numpy as np
import pandas as pd

from app import read_uploaded_csv
from data_source import capabilities_of, normalize_uploaded


class FakeUpload:
    """Stands in for a Streamlit UploadedFile."""

    def __init__(self, payload, name="meter.csv"):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


def upload(frame, name="meter.csv"):
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False)
    return FakeUpload(buffer.getvalue(), name)


def load(frame, power_unit=None):
    """Read + normalize, returning (canonical frame, report)."""
    raw, error = read_uploaded_csv(upload(frame))
    assert raw is not None, "reader rejected the file: %s" % error
    return normalize_uploaded(raw, power_unit=power_unit)


def base_frame(n=48, freq="h"):
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq=freq),
        "Global_active_power": np.linspace(500, 1500, n),
        "Global_reactive_power": np.linspace(100, 200, n),
        "Global_intensity": np.linspace(200, 400, n),
        "sub_metering_1": np.full(n, 250.0),
    })


class TestAcceptedFormats(unittest.TestCase):
    def test_canonical_uci_columns_in_watts(self):
        df, rep = load(base_frame())
        self.assertEqual(len(df), 48)
        self.assertAlmostEqual(df["Global_active_power"].mean(), 1.0, places=3)
        self.assertEqual(rep["applied_unit"], "W")
        self.assertEqual(str(df["datetime"].min())[:10], "2024-01-01")

    def test_split_date_and_time_columns(self):
        frame = base_frame()
        frame["Date"] = frame["datetime"].dt.strftime("%m/%d/%Y")
        frame["Time"] = frame["datetime"].dt.strftime("%H:%M:%S")
        frame = frame.drop(columns=["datetime"])
        df, _ = load(frame)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-01-01")
        self.assertEqual(str(df["datetime"].max())[:10], "2024-01-02")
        self.assertAlmostEqual(df["Global_active_power"].mean(), 1.0, places=3)

    def test_messy_aliases_and_kw_units(self):
        frame = base_frame().drop(columns=["datetime"]).rename(
            columns={"sub_metering_1": "Sub Metering 1",
                     "Global_active_power": "total power"})
        frame["total power"] = frame["total power"] / 1000
        frame["datetime"] = base_frame()["datetime"]
        df, rep = load(frame)
        self.assertEqual(rep["applied_unit"], "kW")
        self.assertIn("Sub_metering_1", df.columns)
        self.assertAlmostEqual(df["Global_active_power"].mean(), 1.0, places=3)

    def test_wh_columns_become_kw(self):
        frame = base_frame()
        frame["Global_active_power"] = np.linspace(0.5, 1.5, 48)
        df, _ = load(frame)
        self.assertAlmostEqual(df["Global_active_power"].mean(), 1.0, places=3)

    def test_plain_total_kwh_column(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-05-01", periods=72, freq="h")
            .strftime("%Y-%m-%d"),
            "hour": np.tile(np.arange(24), 3),
            "total_kwh": np.full(72, 900.0),
        })
        df, _ = load(frame)
        self.assertEqual(len(df), 72)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-05-01")
        self.assertAlmostEqual(df["Global_active_power"].mean(), 0.9, places=3)

    def test_unnamed_index_column_is_dropped(self):
        payload = (b"Unnamed: 0,datetime,Global_active_power\n"
                   b"0,2024-01-01 00:00:00,1200\n"
                   b"1,2024-01-01 01:00:00,1300\n"
                   b"2,2024-01-01 02:00:00,1400\n"
                   b"3,2024-01-01 03:00:00,1500\n")
        raw, _ = read_uploaded_csv(FakeUpload(payload))
        self.assertIsNotNone(raw)
        df, rep = load(pd.DataFrame({
            "Unnamed: 0": [0, 1, 2, 3],
            "datetime": pd.date_range("2024-01-01", periods=4, freq="h"),
            "Global_active_power": [1200.0, 1300.0, 1400.0, 1500.0],
        }))
        self.assertNotIn("Unnamed: 0", df.columns)
        self.assertIn("Unnamed: 0", rep["ignored"])

    def test_semicolon_delimiter(self):
        payload = ("datetime;Global_active_power\n"
                   "2024-01-01 00:00:00;1200\n2024-01-01 01:00:00;1300\n"
                   "2024-01-01 02:00:00;1400\n2024-01-01 03:00:00;1500\n")
        raw, _ = read_uploaded_csv(FakeUpload(payload.encode()))
        self.assertIsNotNone(raw)
        self.assertEqual(len(raw.columns), 2)

    def test_latin1_encoding(self):
        payload = ("datetime,Global_active_power\n"
                   "2024-01-01 00:00:00,1200\n2024-01-01 01:00:00,1300\n"
                   "2024-01-01 02:00:00,1400\n"
                   "café\n").encode("latin-1")
        raw, _ = read_uploaded_csv(FakeUpload(payload))
        self.assertIsNotNone(raw)
        self.assertEqual(len(raw), 4)


class TestSubHourlyData(unittest.TestCase):
    def test_watts_are_averaged_not_summed(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-03-05", periods=192, freq="15min"),
            "Global_active_power": np.full(192, 600.0),
        })
        df, rep = load(frame)
        self.assertEqual(rep["interval"], 15.0)
        self.assertEqual(len(df), 48)
        self.assertAlmostEqual(df["Global_active_power"].mean(), 0.6, places=3)

    def test_watt_hours_are_summed(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-03-05", periods=192, freq="15min"),
            "Global_active_power": np.full(192, 150.0),
        })
        df, _ = load(frame)
        self.assertEqual(len(df), 48)
        self.assertAlmostEqual(df["Global_active_power"].mean(), 0.6, places=3)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-03-05")

    def test_ambiguous_unit_is_flagged_not_guessed_silently(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-03-05", periods=96, freq="15min"),
            "Global_active_power": np.full(96, 0.15),
        })
        _, rep = load(frame)
        self.assertFalse(rep["unit_confident"])
        self.assertIn("src_note_unit_unsure",
                      [code for code, _ in rep["note_codes"]])

    def test_explicit_unit_overrides_detection(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-03-05", periods=192, freq="15min"),
            "Global_active_power": np.full(192, 600.0),
        })
        df, rep = load(frame, power_unit="W")
        self.assertTrue(rep["unit_confident"])
        self.assertAlmostEqual(df["Global_active_power"].mean(), 0.6, places=3)


class TestTimestampFidelity(unittest.TestCase):
    def test_iso_dates_are_not_swapped_to_day_first(self):
        """day-first inference once turned 2024-03-05 into 2024-05-03."""
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-03-05", periods=10, freq="D"),
            "Global_active_power": np.full(10, 1.0),
        })
        df, _ = load(frame)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-03-05")
        self.assertEqual(str(df["datetime"].max())[:10], "2024-03-14")

    def test_row_index_column_is_not_treated_as_an_hour(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-05-01", periods=72, freq="h")
            .strftime("%Y-%m-%d"),
            "readout": range(72),
            "total_kwh": np.full(72, 900.0),
        })
        df, _ = load(frame)
        self.assertEqual(len(df), 72)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-05-01")

    def test_day_first_dates_still_parse(self):
        """A month-long export always has a day above 12, so the order is knowable."""
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-05-01", periods=20, freq="D")
            .strftime("%d-%m-%Y"),
            "Value": np.full(20, 1.0),
        })
        df, _ = load(frame)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-05-01")
        self.assertEqual(str(df["datetime"].max())[:10], "2024-05-20")

    def test_ambiguous_date_order_is_reported(self):
        frame = pd.DataFrame({
            "Date": ["03/01/2024", "04/01/2024", "05/01/2024"],
            "Value": [1.0, 2.0, 3.0],
        })
        _, rep = load(frame)
        self.assertEqual(rep["date_order"], "month-first")
        self.assertIn("src_note_date_order",
                      [code for code, _ in rep["note_codes"]])

    def test_iso_dates_report_no_ambiguity(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-03-05", periods=3, freq="D")
            .strftime("%Y-%m-%d"),
            "Value": [1.0, 2.0, 3.0],
        })
        _, rep = load(frame)
        self.assertEqual(rep["date_order"], "iso")
        self.assertNotIn("src_note_date_order",
                         [code for code, _ in rep["note_codes"]])

    def test_month_first_is_the_default_for_slash_dates(self):
        """A column with no field above 12 is read month-first, not day-first."""
        frame = pd.DataFrame({
            "Date": ["03/01/2024", "04/01/2024", "05/01/2024"],
            "Value": [1.0, 2.0, 3.0],
        })
        df, _ = load(frame)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-03-01")
        self.assertEqual(str(df["datetime"].max())[:10], "2024-05-01")

    def test_a_day_above_twelve_switches_the_whole_column_to_day_first(self):
        frame = pd.DataFrame({
            "Date": ["01/02/2024", "25/02/2024", "26/02/2024"],
            "Value": [1.0, 2.0, 3.0],
        })
        df, _ = load(frame)
        self.assertEqual(str(df["datetime"].min())[:10], "2024-02-01")
        self.assertEqual(str(df["datetime"].max())[:10], "2024-02-26")


class TestRejections(unittest.TestCase):
    def test_garbage_bytes(self):
        raw, error = read_uploaded_csv(FakeUpload(b"\x00\x01not a csv"))
        self.assertIsNone(raw)
        # A localized sentence, never a raw parser traceback.
        self.assertIn("could not be read", error)
        self.assertNotIn("Error", error)
        self.assertNotIn("Traceback", error)

    def test_empty_file(self):
        raw, error = read_uploaded_csv(FakeUpload(b""))
        self.assertIsNone(raw)
        self.assertTrue(error)

    def test_no_timestamp_column(self):
        frame = pd.DataFrame({"power": [1.0, 2.0], "voltage": [230.0, 231.0]})
        df, rep = load(frame)
        self.assertFalse(rep["ok"])
        self.assertIn("src_note_no_timestamp",
                      [code for code, _ in rep["note_codes"]])

    def test_no_power_column(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
            "Voltage": [230.0] * 5,
        })
        df, rep = load(frame)
        self.assertFalse(rep["ok"])
        self.assertIn("src_note_no_power",
                      [code for code, _ in rep["note_codes"]])

    def test_non_numeric_power_is_dropped(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
            "Global_active_power": ["a", "b", "c", "d", "e"],
        })
        df, rep = load(frame)
        self.assertEqual(len(df), 0)


class TestCapabilities(unittest.TestCase):
    def test_total_only_file_gates_models_but_keeps_costs(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=48, freq="h"),
            "Global_active_power": np.full(48, 1.0),
        })
        df, _ = load(frame)
        caps = capabilities_of(df)
        self.assertTrue(caps["costs"]["enabled"])
        self.assertTrue(caps["trends"]["enabled"])
        self.assertFalse(caps["prediction"]["enabled"])
        self.assertIn("Voltage", caps["prediction"]["missing"])

    def test_full_feature_file_enables_everything(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=48, freq="h"),
            "Global_active_power": np.full(48, 1.0),
            "Global_reactive_power": np.full(48, 0.2),
            "Voltage": np.full(48, 235.0),
            "Global_intensity": np.full(48, 5.0),
            "Sub_metering_1": np.full(48, 0.1),
            "Sub_metering_2": np.full(48, 0.2),
            "Sub_metering_3": np.full(48, 0.3),
        })
        df, _ = load(frame)
        caps = capabilities_of(df)
        for name, detail in caps.items():
            self.assertTrue(detail["enabled"], "%s disabled: %s"
                            % (name, detail["missing"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
