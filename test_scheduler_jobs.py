"""Background jobs: each runs end to end against a throwaway database."""
from datetime import datetime, timedelta

import pandas as pd

import scheduler_jobs as jobs
from db import DatabaseManager


class FakeService:
    """Records what would have been sent; never touches the network."""

    def __init__(self):
        self.weekly = []

    def email_configured(self):
        return False

    def sms_configured(self):
        return False

    def send_weekly_summary(self, **kw):
        self.weekly.append(kw)
        return [kw["primary_email"]]


def _db(tmp_path):
    db = DatabaseManager(db_path=str(tmp_path / "jobs.db"))
    db.create_user("asha@example.com", "Asha", "Str0ng!Passw0rd", household_id="hh1")
    db.create_user("guest-1@guest.energypulse.local", "Guest", "x", is_guest=True, household_id="hh2")
    db.save_home_details("hh1", {"tariff_rate": 7.5, "occupants": 4, "home_type": "Apartment",
                                 "appliances": [{"name": "Refrigerator"}, {"name": "Air Conditioner"}]})
    return db


def _history(days=14):
    idx = pd.date_range(datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=days),
                        periods=days * 24, freq="h")
    return pd.DataFrame({"datetime": idx, "Global_active_power": 1.0})


def test_only_registered_households_are_used(tmp_path):
    db = _db(tmp_path)
    assert [h["household_id"] for h in jobs.registered_households(db)] == ["hh1"]


def test_rate_comes_from_saved_home(tmp_path):
    db = _db(tmp_path)
    assert jobs.household_rate(db, "hh1") == 7.5
    assert jobs.household_rate(db, "unknown") == jobs.DEFAULT_RATE


def test_daily_brief_is_sent_once_per_day(tmp_path):
    db = _db(tmp_path)
    first = jobs.run_daily_briefs(db=db, service=FakeService())
    assert first["failed"] == 0 and first["sent"] == 1
    again = jobs.run_daily_briefs(db=db, service=FakeService())
    assert again["sent"] == 0 and again["skipped"] == 1


def test_weekly_summary_uses_saved_meter_readings(tmp_path):
    db = _db(tmp_path)
    db.save_simulation_history("hh1", _history())
    svc = FakeService()
    result = jobs.send_weekly_summaries(db=db, service=svc)
    assert result == {"sent": 1, "skipped": 0, "failed": 0}
    assert abs(svc.weekly[0]["week_kwh"] - 7 * 24) < 25      # about one kW for a week
    # A second run in the same week does not repeat it.
    assert jobs.send_weekly_summaries(db=db, service=svc)["sent"] == 0


def test_history_is_not_saved_twice(tmp_path):
    db = _db(tmp_path)
    hist = _history(3)
    assert db.save_simulation_history("hh1", hist) == len(hist)
    assert db.save_simulation_history("hh1", hist) == 0
    next_hour = pd.DataFrame({"datetime": [hist["datetime"].iloc[-1] + pd.Timedelta(hours=1)],
                              "Global_active_power": [2.0]})
    more = pd.concat([hist, next_hour], ignore_index=True)
    assert db.save_simulation_history("hh1", more) == 1
    assert len(db.get_simulation_history("hh1")) == len(hist) + 1


def test_bill_alerts_run_for_each_household(tmp_path, monkeypatch):
    db = _db(tmp_path)
    calls = []
    import notifications
    monkeypatch.setattr(notifications, "dispatch_household_alerts", lambda **kw: calls.append(kw))
    when = pd.date_range("2030-01-01", periods=24 * 30, freq="h")
    forecast = pd.DataFrame({"datetime": when, "predicted_kwh": 1.0, "day": when.date})
    result = jobs.check_bill_alerts(db=db, forecast_df=forecast)
    assert result == {"checked": 1, "failed": 0}
    assert calls[0]["tariff_rate"] == 7.5 and calls[0]["predicted_kwh"] > 0


def test_guest_household_is_removed_completely(tmp_path, monkeypatch):
    import integrity
    import nameplate
    monkeypatch.setattr(integrity, "BILL_DIR", str(tmp_path / "bills"))
    monkeypatch.setattr(nameplate, "PHOTO_DIR", str(tmp_path / "photos"))
    db = _db(tmp_path)
    db.save_home_details("hh2", {"tariff_rate": 8})
    db.save_simulation_history("hh2", _history(1))
    db.set_store("hh2", "goal", {"rupees": 900})
    (tmp_path / "bills" / "hh2").mkdir(parents=True)
    (tmp_path / "bills" / "hh2" / "b.pdf").write_bytes(b"x")
    (tmp_path / "photos" / "hh2").mkdir(parents=True)
    assert db.delete_household("hh2") > 0
    assert db.get_user("guest-1@guest.energypulse.local") is None
    assert db.get_home_details("hh2") is None
    assert db.get_simulation_history("hh2").empty
    assert db.get_store("hh2", "goal") is None
    assert not (tmp_path / "bills" / "hh2").exists() and not (tmp_path / "photos" / "hh2").exists()
    # The registered household is untouched.
    assert db.get_user("asha@example.com") and db.get_home_details("hh1")


def test_only_old_guests_are_purged(tmp_path):
    db = _db(tmp_path)
    assert db.purge_stale_guests(24) == 0                     # created just now
    with db.get_connection() as conn:
        conn.execute("UPDATE users SET created_at = datetime('now', '-2 days') WHERE is_guest = 1")
    assert jobs.purge_guests(db, 24) == 1
    assert db.get_user("asha@example.com") is not None
