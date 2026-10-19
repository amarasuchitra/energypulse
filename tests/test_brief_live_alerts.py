"""Live readings, freshness and overconsumption alerts (brief sections 5 and 8).

The pipeline tests drive monitoring.run_check end to end on a real SQLite database with a
stand-in forecast model, so the reading -> hourly -> forecast -> log -> alert path is exercised.
"""
import time

import numpy as np
import pandas as pd
import pytest

from db import DatabaseManager
from forecast import Forecaster
from forecast_features import hourly_features, to_hourly
from monitoring import (ALERT_TYPE, SETTINGS_KEY, STATE_KEY, AlertSettings, alert_text, check_overconsumption,
                        compare, deliver, latest_alert, next_state, run_check)
from readings import MAX_KW, STALE_AFTER, clean_readings, freshness
from sim_dataset import minute_to_hourly, power_features

NOW = pd.Timestamp("2026-10-14 12:20")
S = AlertSettings()


# ------------------------------------------------------------------ helpers
def _minutes(days=3, kw=1.0, end=pd.Timestamp("2026-10-14 12:20"), extra=None):
    """Steady 1-minute readings ending just before `end`; `extra` = {hour: added kW}."""
    t = pd.date_range(end=end - pd.Timedelta(minutes=1), periods=days * 1440, freq="min")
    mains = np.full(len(t), float(kw))
    for hour, add in (extra or {}).items():
        mains[(t >= hour) & (t < hour + pd.Timedelta(hours=1))] += add
    return pd.DataFrame({"datetime": t, "mains_kw": mains, "ac_kw": np.where(mains > kw, mains - kw, 0.0),
                         "other_kw": np.full(len(t), float(kw))})


class _Flat:
    """Forecasts `kw` for every hour, whatever the inputs."""
    def __init__(self, kw=1.0):
        self.kw = kw

    def predict(self, X):
        return np.full(len(X), self.kw)


def _fc(model=None):
    fc = object.__new__(Forecaster)
    hourly = minute_to_hourly(_minutes())
    fc.xgb_sim = model or _Flat()
    fc.sim_features = power_features(hourly_features(to_hourly(hourly)).columns)
    fc.sim_offset, fc.q05_sim, fc.q95_sim, fc.sim_error = 0.0, None, None, None
    return fc


class _Mail:
    """Records what would be emailed; never sends anything."""
    def __init__(self, ok=True, configured=True, fail_times=0):
        self.ok, self.configured, self.fail_times, self.sent = ok, configured, fail_times, []

    def email_configured(self):
        return self.configured

    def _compose(self, name, body, lang):
        return body

    def _deliver(self, to, subject, body, phone):
        if self.fail_times > 0:
            self.fail_times -= 1
            return False, "Connection unexpectedly closed"
        self.sent.append((to, subject))
        return (True, None) if self.ok else (False, "535 Username and Password not accepted")


PEOPLE = [{"name": "Amara", "email": "amara@example.com"}]


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "live.db"))


# ------------------------------------------------------------------ readings
@pytest.mark.brief("LIVE-001", "LIVE-002")
def test_simulated_meter_produces_valid_readings():
    from meter_sim import simulate_home
    df = simulate_home(days=1, seed=3)
    clean, report = clean_readings(df)
    assert len(df) == 1440 and report["kept"] == 1440
    assert (clean["mains_kw"] >= 0).all() and clean["datetime"].is_monotonic_increasing


@pytest.mark.brief("LIVE-007", "LIVE-008", "LIVE-009", "LIVE-010", "ML-005", "ML-017")
def test_bad_readings_are_dropped_and_counted():
    t = pd.date_range("2026-10-14 10:00", periods=8, freq="min")
    df = pd.DataFrame({"datetime": list(t[:7]) + [None],
                       "mains_kw": [1.0, None, "abc", -0.5, MAX_KW * 4, np.inf, 1.2, 1.1]})
    clean, r = clean_readings(df)
    assert clean["mains_kw"].tolist() == [1.0, 1.2]
    assert (r["missing"], r["not_numeric"], r["negative"], r["outlier"]) == (2, 2, 1, 1)
    assert r["received"] == 8 and r["kept"] == 2


@pytest.mark.brief("LIVE-011", "LIVE-012")
def test_duplicates_keep_the_first_and_out_of_order_readings_are_sorted():
    df = pd.DataFrame({"datetime": pd.to_datetime(["2026-10-14 10:02", "2026-10-14 10:00", "2026-10-14 10:01",
                                                   "2026-10-14 10:01"]),
                       "mains_kw": [3.0, 1.0, 2.0, 9.0]})
    clean, r = clean_readings(df)
    assert clean["mains_kw"].tolist() == [1.0, 2.0, 3.0]
    assert r["duplicate"] == 1 and r["reordered"] >= 1


@pytest.mark.brief("LIVE-007")
def test_readings_without_the_needed_columns_are_refused():
    with pytest.raises(ValueError):
        clean_readings(pd.DataFrame({"time": [1], "kw": [1.0]}))
    assert clean_readings(pd.DataFrame(columns=["datetime", "mains_kw"]))[1]["kept"] == 0


@pytest.mark.brief("LIVE-013", "ML-019", "UI-011")
def test_freshness_marks_each_stage_and_never_invents_a_time():
    now = pd.Timestamp("2026-10-14 12:00")
    f = freshness({"measurement": now - pd.Timedelta(minutes=1), "prediction": now - pd.Timedelta(hours=3),
                   "optimization": None, "alert_check": now - pd.Timedelta(minutes=30)}, now)
    assert not f["measurement"]["stale"] and f["prediction"]["stale"] and not f["alert_check"]["stale"]
    assert f["optimization"] == {"at": None, "age_min": None, "stale": True}
    assert f["prediction"]["age_min"] == 180.0
    later = freshness({"measurement": now}, now + STALE_AFTER["measurement"] + pd.Timedelta(seconds=1))
    assert later["measurement"]["stale"]                        # the simulation stopped: reported as stale


@pytest.mark.brief("LIVE-003")
def test_dashboard_refresh_intervals_are_as_documented():
    src = open("frontend/app.py", encoding="utf-8").read()
    assert "@st.fragment(run_every=1)\n    def live_tile" in src        # live tile: every second
    assert "@st.fragment(run_every=60)\n    def today_chart" in src      # chart: every minute
    assert "once an hour" in src                                     # forecast + alert check: hourly


# ------------------------------------------------------------------ the pipeline
@pytest.mark.brief("LIVE-004", "LIVE-005", "ALERT-001", "INT-001")
def test_normal_use_logs_forecasts_and_actuals_and_raises_nothing(db):
    out = run_check(db, _fc(), "hh1", _minutes(), NOW, owned=("ac",), people=PEOPLE, service=_Mail())
    assert out["logged"] > 0 and out["filled"] > 0
    assert out["result"]["status"] == "ok" and out["alert"] is None
    assert out["checked_at"] == NOW and out["readings"]["kept"] == 3 * 1440
    log = db.forecast_log("hh1", "simulated_meter")
    assert (log["predicted_kwh"] == 1.0).all() and log["actual_kwh"].dropna().between(0.99, 1.01).all()


@pytest.mark.brief("ALERT-002", "ALERT-004", "ALERT-014", "ALERT-015", "INT-002", "E2E-003")
def test_sustained_excess_raises_one_alert_with_all_figures(db):
    mail = _Mail()
    extra = {pd.Timestamp("2026-10-14 10:00"): 1.0, pd.Timestamp("2026-10-14 11:00"): 1.0}
    out = run_check(db, _fc(), "hh1", _minutes(extra=extra), NOW, owned=("ac",), people=PEOPLE, service=mail)
    alert = out["alert"]
    assert out["result"]["status"] == "over" and alert is not None
    d = alert["data"]
    assert d["actual_kwh"] == pytest.approx(4.0, abs=0.01) and d["predicted_kwh"] == pytest.approx(2.0)
    assert d["excess_kwh"] == pytest.approx(2.0, abs=0.01) and d["excess_pct"] == pytest.approx(100.0, abs=0.5)
    assert d["start"].startswith("2026-10-14 10:00") and d["appliance"] == "ac"
    assert "air conditioner" in alert["body"].lower() and "Raise the AC set temperature" in alert["body"]
    assert alert["delivery"][0]["status"] == "sent" and mail.sent == [("amara@example.com", alert["subject"])]
    entry = latest_alert(db, "hh1")                             # what the dashboard banner reads
    assert entry["notification_type"] == ALERT_TYPE and entry["status"] == "sent"


@pytest.mark.brief("ALERT-005", "ALERT-019", "LIVE-018")
def test_rechecking_the_same_event_does_not_alert_or_email_again(db):
    mail = _Mail()
    extra = {pd.Timestamp("2026-10-14 10:00"): 1.0, pd.Timestamp("2026-10-14 11:00"): 1.0}
    minutes = _minutes(extra=extra)
    first = run_check(db, _fc(), "hh1", minutes, NOW, owned=("ac",), people=PEOPLE, service=mail)
    again = run_check(db, _fc(), "hh1", minutes, NOW + pd.Timedelta(minutes=5), owned=("ac",), people=PEOPLE,
                      service=mail)
    assert first["alert"] is not None and again["alert"] is None
    assert again["logged"] == 0 and again["filled"] == 0          # nothing processed twice
    assert len(mail.sent) == 1 and len(db.get_notification_log("hh1")) == 1


@pytest.mark.brief("ALERT-003")
def test_one_high_hour_is_not_an_alert(db):
    out = run_check(db, _fc(), "hh1", _minutes(extra={pd.Timestamp("2026-10-14 11:00"): 1.5}), NOW,
                    owned=("ac",), people=PEOPLE, service=_Mail())
    assert out["alert"] is None and out["result"]["status"] == "insufficient"


@pytest.mark.brief("ALERT-006", "ALERT-007")
def test_recovery_is_recorded_and_a_later_event_alerts_again():
    over = {"status": "over", "hours": [{"target_hour": pd.Timestamp("2026-10-14 10:00")},
                                        {"target_hour": pd.Timestamp("2026-10-14 11:00")}]}
    ok = {"status": "ok", "hours": [{"target_hour": pd.Timestamp("2026-10-14 12:00")}]}
    state, alert = next_state(None, over)
    assert alert
    state, alert = next_state(state, ok)
    assert not alert and state == {"open": False, "recovered_at": "2026-10-14 12:00:00"}
    assert next_state(state, over)[1]


@pytest.mark.brief("ALERT-008", "ALERT-009")
def test_zero_and_near_zero_forecasts_do_not_divide_by_zero():
    for pred in (0.0, 0.01, 0.049):
        row = compare(pred, pred + 0.5, S)
        assert row["excess_pct"] is None and row["over"]            # judged on units, not a huge percentage
    assert compare(0.0, 0.0, S)["over"] is False
    _, body, data = alert_text({"hours": [dict(compare(0.0, 0.5, S), target_hour=pd.Timestamp("2026-10-14 10:00"))]},
                               None)
    assert data["excess_pct"] is None and "%" not in body


@pytest.mark.brief("ALERT-010")
def test_units_are_consistent_hourly_kwh_equals_mean_kw():
    hourly = minute_to_hourly(_minutes(days=1, kw=2.0))
    assert hourly["Global_active_power"].dropna().eq(2.0).all()   # 2 kW for an hour = 2 kWh in that hour


@pytest.mark.brief("ALERT-011", "ALERT-012", "LIVE-013")
def test_gaps_and_stale_or_missing_readings_never_alert(db):
    extra = {pd.Timestamp("2026-10-14 10:00"): 1.0, pd.Timestamp("2026-10-14 11:00"): 1.0}
    m = _minutes(extra=extra)
    m = m[~((m["datetime"] >= "2026-10-14 10:00") & (m["datetime"] < "2026-10-14 10:20"))]   # 40 readings left
    out = run_check(db, _fc(), "hh1", m, NOW, owned=("ac",), people=PEOPLE, service=_Mail())
    assert out["alert"] is None                                  # the 10:00 hour is incomplete: not judged
    stale = run_check(db, _fc(), "hh2", _minutes(extra=extra, end=pd.Timestamp("2026-10-14 08:00")),
                      NOW, owned=("ac",), people=PEOPLE, service=_Mail())
    assert stale["alert"] is None and stale["result"]["status"] == "insufficient"
    empty = run_check(db, _fc(), "hh3", pd.DataFrame(columns=["datetime", "mains_kw"]), NOW, owned=(),
                      people=PEOPLE)
    assert empty["alert"] is None and empty["logged"] == 0


@pytest.mark.brief("ALERT-013", "ALERT-016")
def test_email_failures_are_recorded_and_never_claimed_as_sent(db):
    bad_password = _Mail(ok=False)
    report = deliver(db, "hh1", PEOPLE, "s", "b", {"x": 1}, bad_password)
    assert report[0]["status"] == "failed" and "535" in report[0]["error"]
    unconfigured = deliver(db, "hh1", PEOPLE, "s", "b", {}, _Mail(configured=False))
    assert unconfigured[0]["status"] == "in-app"
    statuses = [e["status"] for e in db.get_notification_log("hh1")]
    assert sorted(statuses) == ["failed", "in-app"]


@pytest.mark.brief("ALERT-017", "ALERT-018", "SEC-012")
def test_alerts_belong_to_their_own_household(db):
    deliver(db, "hh1", PEOPLE, "s1", "b", {}, None)
    deliver(db, "hh2", [{"name": "B", "email": "b@example.com"}], "s2", "b", {}, None)
    assert [e["subject"] for e in db.get_notification_log("hh1")] == ["s1"]
    assert [e["subject"] for e in db.get_notification_log("hh2")] == ["s2"]
    db.set_store("hh1", STATE_KEY, {"open": True})
    assert db.get_store("hh2", STATE_KEY) is None


@pytest.mark.brief("ALERT-020")
def test_smtp_retries_once_on_a_dropped_connection_but_not_on_a_bad_password(monkeypatch):
    import smtplib
    import notifications as N
    monkeypatch.setattr(N, "SMTP_RETRY_WAIT", 0.0)
    calls = {"n": 0}

    class Flaky:
        def __init__(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise smtplib.SMTPServerDisconnected("Connection unexpectedly closed")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            if self.bad:
                raise smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")

        def sendmail(self, *a):
            pass
    svc = object.__new__(N.NotificationService)
    svc.SMTP_HOST, svc.SMTP_PORT, svc.SMTP_FROM = "smtp.test", 587, "from@example.com"
    svc.SMTP_USE_SSL, svc.SMTP_STARTTLS, svc.SMTP_USERNAME, svc.SMTP_PASSWORD = False, True, "u@example.com", "x"
    svc.SENDGRID_API_KEY = svc.TWILIO_AUTH_TOKEN = svc.TWILIO_ACCOUNT_SID = None
    monkeypatch.setattr(smtplib, "SMTP", Flaky)
    Flaky.bad = False
    assert svc._send_via_smtp("to@example.com", "s", "b") == (True, None) and calls["n"] == 2
    calls["n"], Flaky.bad = 1, True                          # connects fine, password rejected
    ok, err = svc._send_via_smtp("to@example.com", "s", "b")
    assert not ok and calls["n"] == 2                        # tried once, not retried


@pytest.mark.brief("LIVE-014", "LIVE-019", "ML-013", "OPT-011", "UI-005")
def test_a_failing_model_never_breaks_the_check(db):
    class Broken:
        def predict(self, X):
            raise RuntimeError("model crashed")
    with pytest.raises(RuntimeError):
        run_check(db, _fc(Broken()), "hh1", _minutes(), NOW, owned=(), people=PEOPLE)
    src = open("frontend/app.py", encoding="utf-8").read()
    # The dashboard wraps the check and the forecast; a failure shows a notice, not a crash.
    assert 'log.exception("Monitoring failed for %s", household_id)' in src
    assert 'log.exception("Meter forecast failed")' in src
    assert "The alert check could not run this hour" in src


@pytest.mark.brief("LIVE-015", "LIVE-020")
def test_monitoring_resumes_after_a_restart_from_what_was_stored(tmp_path):
    path = str(tmp_path / "r.db")
    extra = {pd.Timestamp("2026-10-14 10:00"): 1.0, pd.Timestamp("2026-10-14 11:00"): 1.0}
    first = run_check(DatabaseManager(db_path=path), _fc(), "hh1", _minutes(extra=extra), NOW, owned=("ac",),
                      people=PEOPLE)
    assert first["alert"] is not None
    restarted = DatabaseManager(db_path=path)                  # a new process: state comes from the database
    again = run_check(restarted, _fc(), "hh1", _minutes(extra=extra), NOW + pd.Timedelta(minutes=10), owned=("ac",),
                      people=PEOPLE)
    assert again["alert"] is None and restarted.get_store("hh1", STATE_KEY)["open"] is True


@pytest.mark.brief("ALERT-002")
def test_settings_change_the_threshold(db):
    extra = {pd.Timestamp("2026-10-14 10:00"): 0.3, pd.Timestamp("2026-10-14 11:00"): 0.3}
    out = run_check(db, _fc(), "hh1", _minutes(extra=extra), NOW, owned=("ac",), people=PEOPLE)
    assert out["alert"] is not None                             # +30% with the default 15%
    db.set_store("hh2", SETTINGS_KEY, {"threshold_pct": 50})
    out = run_check(db, _fc(), "hh2", _minutes(extra=extra), NOW, owned=("ac",), people=PEOPLE)
    assert out["alert"] is None                                 # not above a 50% threshold
    db.set_store("hh3", SETTINGS_KEY, {"enabled": False})
    assert run_check(db, _fc(), "hh3", _minutes(extra=extra), NOW, owned=("ac",), people=PEOPLE)["alert"] is None


@pytest.mark.brief("LIVE-016", "UI-017", "UI-018")
def test_repeated_checks_stay_fast_and_bounded(db):
    m = _minutes(days=14)
    t0 = time.perf_counter()
    for i in range(5):
        run_check(db, _fc(), "hh1", m, NOW + pd.Timedelta(minutes=i), owned=(), people=PEOPLE)
    per_check = (time.perf_counter() - t0) / 5
    assert per_check < 5.0                                      # measured, see the report
    assert len(db.forecast_log("hh1", "simulated_meter")) <= 49  # only the last 48 hours are logged
    print(f"run_check on 14 days of minute data: {per_check:.3f} s per call")
