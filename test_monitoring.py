"""Overconsumption detection: thresholds, sustained excess, de-duplication, recovery, stale data."""
import pandas as pd
import pytest

from db import DatabaseManager
from monitoring import (ALERT_TYPE, SOURCE, STATE_KEY, AlertSettings, alert_text, check_overconsumption,
                        compare, deliver, next_state)

NOW = pd.Timestamp("2026-10-14 12:20")
S = AlertSettings()                       # 15%, 0.15 kWh, 2 hours, 3 h freshness


def _log(pairs, end=pd.Timestamp("2026-10-14 11:00")):
    """pairs of (predicted, actual) for consecutive hours ending at `end`."""
    hours = pd.date_range(end=end, periods=len(pairs), freq="h")
    return pd.DataFrame({"target_hour": hours, "predicted_kwh": [p for p, _ in pairs],
                         "actual_kwh": [a for _, a in pairs]})


def test_normal_use_raises_nothing():
    assert check_overconsumption(_log([(1.0, 1.05), (1.0, 0.9), (1.0, 1.1)]), S, NOW)["status"] == "ok"


def test_small_absolute_excess_is_ignored_even_if_the_percentage_is_high():
    # 0.10 -> 0.20 is +100% but only 0.1 units
    assert check_overconsumption(_log([(0.10, 0.20), (0.10, 0.20)]), S, NOW)["status"] == "ok"


def test_sustained_excess_triggers_and_values_are_right():
    r = check_overconsumption(_log([(1.0, 1.0), (1.4, 1.8), (1.0, 1.3)]), S, NOW)
    assert r["status"] == "over" and len(r["hours"]) == 2
    first = r["hours"][0]
    assert first["excess_kwh"] == pytest.approx(0.4) and first["excess_pct"] == pytest.approx(400 / 14)
    subject, body, data = alert_text(r, None)
    assert data["actual_kwh"] == pytest.approx(3.1) and data["predicted_kwh"] == pytest.approx(2.4)
    assert data["excess_pct"] == pytest.approx(29.2, abs=0.05)
    assert "3.10 units" in body and "2.40 units" in body and "29%" in body


def test_a_single_spike_is_not_an_event():
    assert check_overconsumption(_log([(1.0, 1.0), (1.0, 2.0)]), S, NOW)["status"] == "insufficient"


def test_gap_between_hours_is_not_sustained():
    log = _log([(1.0, 1.5), (1.0, 1.0), (1.0, 1.5)])
    log = log.drop(index=1)                                    # hours not back to back
    assert check_overconsumption(log, S, NOW)["status"] == "insufficient"


def test_near_zero_forecast_uses_absolute_excess_without_dividing_by_zero():
    row = compare(0.0, 0.4, S)
    assert row["excess_pct"] is None and row["over"]
    assert compare(0.01, 0.02, S)["over"] is False


def test_stale_readings_never_alert():
    old = _log([(1.0, 2.0), (1.0, 2.0)], end=pd.Timestamp("2026-10-14 06:00"))
    assert check_overconsumption(old, S, NOW)["status"] == "insufficient"


def test_one_alert_per_event_and_a_new_one_after_recovery():
    over = check_overconsumption(_log([(1.0, 1.5), (1.0, 1.5)]), S, NOW)
    ok = check_overconsumption(_log([(1.0, 1.5), (1.0, 1.0)]), S, NOW)
    state, alert = next_state(None, over)
    assert alert and state["open"]
    state, alert = next_state(state, over)
    assert not alert                                           # same event: suppressed
    state, alert = next_state(state, ok)
    assert not alert and not state["open"]                     # recovered
    state, alert = next_state(state, over)
    assert alert                                               # a new event alerts again


def test_settings_are_validated():
    s = AlertSettings.clean({"threshold_pct": "abc", "consecutive_hours": 99, "min_excess_kwh": -3})
    assert s.threshold_pct == 15.0 and s.consecutive_hours == 6 and s.min_excess_kwh == 0.05


def test_forecast_log_keeps_the_first_forecast_and_fills_actuals_once(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "m.db"))
    hour = pd.Timestamp("2026-10-14 10:00")
    assert db.log_forecasts("hh", SOURCE, [(hour, 1.0, 0.5, 1.5)]) == 1
    assert db.log_forecasts("hh", SOURCE, [(hour, 9.9, 0.0, 20.0)]) == 0      # not overwritten
    assert db.record_actuals("hh", SOURCE, [(hour, 1.3)]) == 1
    assert db.record_actuals("hh", SOURCE, [(hour, 5.0)]) == 0                # filled once
    log = db.forecast_log("hh", SOURCE)
    assert log["predicted_kwh"].iloc[0] == 1.0 and log["actual_kwh"].iloc[0] == 1.3
    assert db.forecast_log("other", SOURCE).empty                            # households are separate


class _FailingService:
    def email_configured(self):
        return True

    def _compose(self, name, body, lang):
        return body

    def _deliver(self, *a):
        raise ConnectionError("smtp down")


def test_delivery_failure_is_recorded_not_raised(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "d.db"))
    report = deliver(db, "hh", [{"name": "A", "email": "a@example.com"}], "s", "b", {"x": 1}, _FailingService())
    assert report[0]["status"] == "failed"
    entry = db.get_notification_log("hh")[0]
    assert entry["notification_type"] == ALERT_TYPE and entry["status"] == "failed"
    report = deliver(db, "hh", [{"name": "B", "email": "b@example.com"}], "s", "b", {}, None)
    assert report[0]["status"] == "in-app"                                    # never claimed as sent


def test_alert_state_is_per_household(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "s.db"))
    db.set_store("hh1", STATE_KEY, {"open": True})
    assert db.get_store("hh2", STATE_KEY) is None
