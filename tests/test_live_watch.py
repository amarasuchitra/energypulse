"""Live alerts checked every minute: high use and appliances left on (live_watch.py)."""
import numpy as np
import pandas as pd

import live_watch as lw
from db import DatabaseManager
from monitoring import deliver

DATE = "2026-10-19"


def _day(base=0.5, spikes=(), on=None):
    """A day of 1-minute readings at `base` kW; spikes = [(start, end, kw)]; on = {key: (start, end, kw)}."""
    kw = np.full(1440, base)
    for a, b, v in spikes:
        kw[a:b] = v
    df = pd.DataFrame({"datetime": pd.date_range(DATE, periods=1440, freq="min"), "mains_kw": kw})
    for key, (a, b, v) in (on or {}).items():
        col = np.zeros(1440)
        col[a:b] = v
        df[f"{key}_kw"], df[f"{key}_on"] = col, col > 0
        df["mains_kw"] += col
    return df


TYPICAL = [0.5] * 1440


def test_short_bursts_do_not_alert_but_ten_minutes_of_high_use_does():
    assert lw.high_use_events(_day(spikes=[(600, 606, 3.0)])["mains_kw"].tolist(), TYPICAL, 1439) == []
    ev = lw.high_use_events(_day(spikes=[(600, 640, 3.0)])["mains_kw"].tolist(), TYPICAL, 1439)
    assert len(ev) == 1 and ev[0]["start"] == 600 and ev[0]["end"] == 640 and ev[0]["alert_at"] == 609
    assert abs(ev[0]["extra_kwh"] - 40 * 2.5 / 60) < 1e-9


def test_event_is_ongoing_until_use_drops_back():
    ev = lw.high_use_events(_day(spikes=[(600, 700, 3.0)])["mains_kw"].tolist(), TYPICAL, 650)
    assert ev[0]["ongoing"] and ev[0]["end"] is None
    assert lw.high_use_events(_day(spikes=[(600, 700, 3.0)])["mains_kw"].tolist(), TYPICAL, 605) == []


def test_small_rise_on_a_low_base_is_not_high_use():
    # 3 x usual but only 0.4 kW more: below the 1 kW minimum
    assert lw.high_use_events(_day(base=0.2, spikes=[(600, 700, 0.6)])["mains_kw"].tolist(), [0.2] * 1440, 1439) == []


def test_appliance_left_on_and_culprit_named():
    df = _day(on={"geyser": (360, 500, 2.0)})
    events = lw.check(df, TYPICAL, 1439, ("geyser",), DATE)
    kinds = {e["kind"] for e in events}
    assert kinds == {"high_use", "left_on"}
    left = next(e for e in events if e["kind"] == "left_on")
    assert "left on" in left["subject"] and "140 minutes" in left["body"] and left["alert_at"] == 360 + 60
    high = next(e for e in events if e["kind"] == "high_use")
    assert "water heater" in high["body"].lower()


def test_each_event_pops_up_once_and_old_ones_are_not_replayed():
    df = _day(spikes=[(300, 330, 3.0), (900, 1000, 3.0)])
    events = lw.check(df, TYPICAL, 960, (), DATE)
    first = lw.to_announce(events, set(), 960)
    assert [e["start"] for e in first] == [900]               # the morning event is listed, not popped up
    assert lw.to_announce(events, {e["id"] for e in first}, 961) == []
    assert len(events) == 2 and events[0]["ongoing"] and not events[1]["ongoing"]


def test_disabled_and_settings_are_cleaned():
    assert lw.check(_day(spikes=[(600, 700, 3.0)]), TYPICAL, 1439, (), DATE, lw.LiveSettings(enabled=False)) == []
    s = lw.LiveSettings.clean({"ratio": "x", "extra_kw": 99, "min_minutes": 0})
    assert s.ratio == 1.5 and s.extra_kw == 5.0 and s.min_minutes == 2
    assert lw.LiveSettings.clean(None) == lw.LiveSettings()


def test_live_alert_is_stored_without_email_for_an_old_event(tmp_path):
    db = DatabaseManager(str(tmp_path / "t.db"))

    class Service:
        sent = 0

        def email_configured(self):
            return True

        def _compose(self, *a):
            return "x"

        def _deliver(self, *a):
            Service.sent += 1
            return True, None
    rep = deliver(db, "h1", [{"name": "A", "email": "a@example.com"}], "High energy use", "body", {"id": "1"},
                  Service(), notification_type=lw.LIVE_TYPE, triggered_by="live meter check", email=False)
    assert rep[0]["status"] == "in-app" and Service.sent == 0
    entry = db.get_notification_log("h1")[0]
    assert entry["notification_type"] == lw.LIVE_TYPE and entry["triggered_by"] == "live meter check"
