"""Every page reads the household's simulated meter (no sample data, nothing drawn per visit)."""
import numpy as np
import pandas as pd

from data_source import meter_hourly, source_from_meter
from replay import ReplaySimulator


def _minutes(days=3, end="2026-10-20 14:37"):
    t = pd.date_range(end=pd.Timestamp(end).floor("D") + pd.Timedelta(days=1), periods=days * 1440,
                      freq="min", inclusive="left")
    kw = 0.4 + 0.3 * np.sin(np.arange(len(t)) / 300.0) ** 2
    return pd.DataFrame({"datetime": t, "mains_kw": kw, "ac_kw": kw * 0.5, "fridge_kw": 0.1,
                         "microwave_kw": 0.0})


def test_hourly_frame_stops_at_the_last_finished_hour_and_has_every_column():
    h = meter_hourly(_minutes(), "2026-10-20 14:37")
    assert h["datetime"].max() == pd.Timestamp("2026-10-20 13:00")             # no future readings
    for col in ("Global_active_power", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "Global_intensity"):
        assert col in h.columns
    src = source_from_meter(h)
    assert src.kind == "meter" and not src.is_sample
    assert all(v["enabled"] for v in src.capabilities.values())
    assert abs(h["Sub_metering_2"].iloc[0] - 100.0) < 1e-9                     # fridge 0.1 kW = 100 Wh/h


def test_same_inputs_give_the_same_readings():
    a = meter_hourly(_minutes(), "2026-10-20 14:37")
    b = meter_hourly(_minutes(), "2026-10-20 14:37")
    pd.testing.assert_frame_equal(a, b)


def test_meter_readings_are_shown_as_already_known_not_replayed():
    sim = ReplaySimulator(delay=2)
    h = meter_hourly(_minutes(), "2026-10-20 14:37")
    sim.set_data(h, restart=False)
    sim.jump_to_end()
    played, total = sim.progress
    assert played == total == len(h) and not sim.is_running
    assert pd.Timestamp(sim.latest_row["datetime"]) == pd.Timestamp("2026-10-20 13:00")
