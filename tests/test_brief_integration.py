"""Database, background jobs, meter API and deployment checks (brief sections 10 and 11)."""
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pandas as pd
import pytest

from db import DatabaseManager
from paths import MODEL_DIR, ROOT


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(db_path=str(tmp_path / "int.db"))


# ------------------------------------------------------------------ database
@pytest.mark.brief("INT-004")
def test_each_operation_reconnects_so_a_lost_connection_does_not_stick(tmp_path):
    path = str(tmp_path / "re.db")
    db = DatabaseManager(db_path=path)
    db.set_store("hh", "k", 1)
    good = db.db_path
    db.db_path = str(tmp_path / "missing" / "x.db")            # the file goes away for a moment
    with pytest.raises(sqlite3.OperationalError):
        db.get_store("hh", "k")
    db.db_path = good                                          # back again: the next call just works
    assert db.get_store("hh", "k") == 1


@pytest.mark.brief("INT-005")
def test_invalid_stored_records_are_ignored_not_crashed_on(db):
    with db.get_connection() as c:
        c.execute("INSERT INTO household_homes (household_id, details_json, updated_at) VALUES ('hh', '{broken', 0)")
        c.execute("INSERT INTO household_homes (household_id, details_json, updated_at) VALUES ('hh2', '[1,2]', 0)")
    assert db.get_home_details("hh") is None and db.get_home_details("hh2") is None
    assert db.get_home_details("missing") is None and db.get_store("missing", "k", default="d") == "d"


@pytest.mark.brief("INT-006", "LIVE-011")
def test_duplicate_forecast_and_actual_rows_are_not_stored_twice(db):
    hour = pd.Timestamp("2026-10-14 10:00")
    assert db.log_forecasts("hh", "simulated_meter", [(hour, 1.0, 0.5, 1.5), (hour, 2.0, 1.0, 3.0)]) == 1
    assert db.record_actuals("hh", "simulated_meter", [(hour, 1.1)]) == 1
    assert db.record_actuals("hh", "simulated_meter", [(hour, 9.9)]) == 0
    assert len(db.forecast_log("hh", "simulated_meter")) == 1


@pytest.mark.brief("INT-007", "APP-025")
def test_concurrent_writers_do_not_lose_or_corrupt_rows(db):
    errors = []

    def write(n):
        try:
            for i in range(20):
                db.log_notification(f"hh{n}", "a@example.com", "A", "test", "t", {"i": i}, "s", "b")
        except Exception as exc:                                # noqa: BLE001
            errors.append(exc)
    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert all(len(db.get_notification_log(f"hh{n}")) == 20 for n in range(4))


@pytest.mark.brief("INT-008", "SEC-011")
def test_deleting_a_household_removes_only_its_own_data(db):
    from auth import Accounts
    acc = Accounts(db_path=db.db_path)
    for email in ("a@example.com", "b@example.com"):
        acc.sign_up(email, "Tr4il-Mango-Kettle", email[0])
        db.ensure_user_login(email, email[0])
    a = db.get_user("a@example.com")["household_id"]
    b = db.get_user("b@example.com")["household_id"]
    for hid in (a, b):
        db.save_home_details(hid, {"appliances": [{"name": "Fan"}]})
        db.log_notification(hid, "x@example.com", "X", "t", "t", {}, "s", "b")
    acc.request_reset("a@example.com")
    db.delete_household(a, "a@example.com", remove_files=False)
    assert db.get_home_details(a) is None and not db.get_notification_log(a) and not acc.exists("a@example.com")
    with sqlite3.connect(db.db_path) as c:
        assert c.execute("SELECT COUNT(*) FROM password_resets WHERE email='a@example.com'").fetchone()[0] == 0
    assert db.get_home_details(b) is not None and len(db.get_notification_log(b)) == 1 and acc.exists("b@example.com")


@pytest.mark.brief("INT-003")
def test_hourly_background_check_runs_for_every_household_without_errors(db, monkeypatch):
    """Regression: the hourly job used an undefined name and failed for every household."""
    pytest.importorskip("streamlit")
    pytest.importorskip("apscheduler")
    import scheduler_jobs as jobs
    from forecast import Forecaster
    from forecast_features import hourly_features, to_hourly
    from sim_dataset import power_features

    class Flat:
        def predict(self, X):
            return np.full(len(X), 0.5)
    fc = object.__new__(Forecaster)
    fc.xgb_sim = Flat()
    fc.sim_features = power_features(hourly_features(to_hourly(pd.DataFrame(
        {"datetime": pd.date_range("2026-01-01", periods=200, freq="h"), "Global_active_power": 1.0}))).columns)
    fc.sim_offset, fc.q05_sim, fc.q95_sim, fc.sim_error = 0.0, None, None, None
    db.create_user("asha@example.com", "Asha", "Str0ng!Passw0rd", household_id="hh1")
    db.save_home_details("hh1", {"tariff_rate": 7.5, "occupants": 4, "home_type": "Apartment",
                                 "appliances": [{"name": "Refrigerator"}, {"name": "Air Conditioner"}]})

    class NoMail:
        def email_configured(self):
            return False
    counts = jobs.check_meter_usage(db=db, service=NoMail(), fc=fc)
    assert counts["failed"] == 0 and counts["checked"] == 1, counts


# ------------------------------------------------------------------ meter API
@pytest.mark.brief("INT-009")
def test_api_timeouts_and_connection_errors_give_a_plain_reason(monkeypatch):
    requests = pytest.importorskip("requests")
    import data_source as ds
    monkeypatch.setattr(ds, "check_api_url", lambda url, allow_private=None: ("meter.example.com", None))

    class Slow:
        def get(self, *a, **k):
            raise requests.exceptions.Timeout()

    class Down:
        def get(self, *a, **k):
            raise requests.exceptions.ConnectionError("refused")
    assert ds.fetch_meter_api("https://meter.example.com/x", session=Slow(), timeout=3)[1] == "No reply within 3 seconds."
    assert "Could not connect" in ds.fetch_meter_api("https://meter.example.com/x", session=Down())[1]


@pytest.mark.brief("INT-010", "UI-006", "UI-007")
@pytest.mark.parametrize("body, ctype", [(b"", ""), (b"{not json", "application/json"), (b'{"a": 1}', "application/json"),
                                         (b"<html>oops</html>", "text/html"), (b"[null, null]", "application/json"),
                                         (b'{"data": [{"time": null, "power": null}]}', "application/json")])
def test_unexpected_api_replies_are_refused_with_a_reason(body, ctype):
    import data_source as ds
    frame, reason = ds.parse_api_payload(body, ctype)
    if frame is not None:                                       # a table with only nulls: refused at the next step
        out, report = ds.normalize_uploaded(frame)
        assert out is None or out.empty or out["Global_active_power"].notna().sum() == 0
    else:
        assert reason


@pytest.mark.brief("INT-011", "SEC-012")
@pytest.mark.parametrize("url", ["http://169.254.169.254/latest/meta-data", "https://127.0.0.1/x", "ftp://example.com/x",
                                 "https://user:pass@example.com/x", "javascript:alert(1)", "https://10.0.0.5/x"])
def test_api_addresses_that_could_reach_internal_services_are_refused(url):
    import data_source as ds
    host, reason = ds.check_api_url(url, allow_private=False)
    assert host is None and reason


@pytest.mark.brief("INT-012")
def test_api_redirects_and_oversized_replies_are_refused(monkeypatch):
    import data_source as ds
    monkeypatch.setattr(ds, "check_api_url", lambda url, allow_private=None: ("meter.example.com", None))

    class Resp:
        def __init__(self, code, chunks=(), headers=None):
            self.status_code, self._chunks, self.headers = code, chunks, headers or {}

        def iter_content(self, n):
            return iter(self._chunks)

        def close(self):
            pass

    class S:
        def __init__(self, r):
            self.r = r

        def get(self, *a, **k):
            return self.r
    assert "redirect" in ds.fetch_meter_api("https://m.example.com", session=S(Resp(302)))[1]
    assert "refused access" in ds.fetch_meter_api("https://m.example.com", session=S(Resp(401)))[1]
    big = Resp(200, [b"x" * 1024] * 3)
    assert "larger than" in ds.fetch_meter_api("https://m.example.com", session=S(big), max_bytes=2048)[1]


# ------------------------------------------------------------------ deployment smoke
@pytest.mark.brief("SMOKE-001")
def test_requirements_cover_every_third_party_import():
    req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read().lower()
    for pkg in ("streamlit", "pandas", "numpy", "xgboost", "lightgbm", "scikit-learn", "plotly", "cryptography",
                "python-dotenv", "apscheduler", "requests"):
        assert pkg in req, pkg


@pytest.mark.brief("SMOKE-002")
def test_entry_point_and_model_files_are_where_the_deploy_expects():
    for f in ("frontend/app.py", "Procfile", "Dockerfile", "runtime.txt"):
        assert os.path.exists(os.path.join(ROOT, f)), f
    assert "frontend/app.py" in open(os.path.join(ROOT, "Procfile")).read()
    assert "frontend/app.py" in open(os.path.join(ROOT, "Dockerfile")).read()
    meta = json.load(open(os.path.join(MODEL_DIR, "forecast_meta.json")))
    needed = ["forecast_meta.json", "forecast_xgb_sim.json", "forecast_q05_sim.pkl", "forecast_q95_sim.pkl"]
    if meta["simulated_meter"].get("model") == "lightgbm":
        needed.append("forecast_lgb_sim.txt.gz")
    for f in needed:
        assert os.path.getsize(os.path.join(MODEL_DIR, f)) > 0, f


@pytest.mark.brief("SMOKE-004", "SEC-008")
def test_no_secrets_are_committed():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not tracked:
        pytest.skip("not a git checkout")
    assert ".env" not in tracked and ".streamlit/secrets.toml" not in tracked
    pattern = re.compile(r"(SMTP_PASSWORD|SENDGRID_API_KEY|TWILIO_AUTH_TOKEN|ENERGYPULSE_SECRET)\s*=\s*['\"]?[A-Za-z0-9_\-]{12,}")
    for f in tracked:
        if f.endswith((".py", ".toml", ".txt", ".md", ".yml", ".yaml", ".env.example", ".cfg", ".ini", ".json")) \
                and os.path.getsize(os.path.join(ROOT, f)) < 2_000_000:
            text = open(os.path.join(ROOT, f), encoding="utf-8", errors="ignore").read()
            placeholder = ("your", "example", "xxxx", "password", "here", "the-", "changeme", "placeholder")
            hits = [m.group(0) for m in pattern.finditer(text)
                    if not any(p in m.group(0).lower() for p in placeholder)
                    and not (f.endswith(".py") and not re.search(r"=\s*['\"]", m.group(0)))]   # code reads settings
            assert not hits, (f, hits)


@pytest.mark.brief("SMOKE-005", "ALERT-016")
def test_missing_email_settings_fail_safe(monkeypatch):
    for var in ("SMTP_USER", "SMTP_USERNAME", "SMTP_PASSWORD", "SENDGRID_API_KEY", "SMTP_HOST"):
        monkeypatch.delenv(var, raising=False)
    import notifications as N
    monkeypatch.setattr(N, "_secret_lookup", lambda name: None)
    svc = N.NotificationService()
    assert not svc.email_configured()
    ok, err = svc.send_reset_code("a@example.com", "ABCDEFGH")
    assert not ok and "not set up" in err


@pytest.mark.brief("SMOKE-006")
def test_backend_modules_import_cleanly():
    code = ("import sys; sys.path[:0] = ['frontend', 'backend']; "
            "import auth, db, home_config, readings, monitoring, scheduler, forecast, notifications, model")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    if "No module named 'streamlit'" in r.stderr:
        pytest.skip("streamlit not installed here")
    assert r.returncode == 0, r.stderr[-2000:]
