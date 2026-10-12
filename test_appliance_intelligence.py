"""Tests for main-meter appliance detection, sessions and recommendations."""
import numpy as np
import pandas as pd
import pytest

from appliance_profiles import (
    ALWAYS_ON, DEFAULT_PROFILES, HouseholdPrefs, allowed_advice, can_switch_off, get_profiles,
)
from disaggregate import Disaggregator, EdgeBaseline, evaluate, build_features
from meter_sim import simulate_from_switches, simulate_home, simulate_many
from scheduler import explain_refusal, feasible_starts, recommend
from sessions import extract_sessions, flag_long_runs, fridge_duty, totals
from tariff import Tariff


@pytest.fixture(scope="module")
def model():
    normal = simulate_many(6, 5, seed=3)
    # Homes where something is wrong, so abnormal runs are seen in training.
    faulty_homes = simulate_many(8, 2, seed=5,
                                 faults=["geyser_left_on", "pump_dry_run", "fridge_seal"])
    faulty_homes["home"] += 100
    return Disaggregator().fit(pd.concat([normal, faulty_homes], ignore_index=True))


@pytest.fixture(scope="module")
def faulty(model):
    df = simulate_home(days=10, seed=42,
                       faults=["evening_laundry", "geyser_left_on", "fridge_seal"])
    return df, model.predict(df[["datetime", "mains_kw"]])


def test_mains_is_sum_of_parts():
    df = simulate_home(days=2, seed=1, noise_kw=0.0)
    parts = df[list(DEFAULT_PROFILES)].sum(axis=1) + df["other_kw"]
    assert np.allclose(df["mains_kw"], parts, atol=1e-3)


def test_features_use_only_the_meter():
    df = simulate_home(days=1, seed=1)
    a = build_features(df[["datetime", "mains_kw"]])
    b = build_features(df)
    assert a.equals(b)


def test_model_beats_rule_baseline_on_unseen_home(model):
    test = simulate_home(days=6, seed=500)
    learned = evaluate(model.predict(test), test)
    base = evaluate(EdgeBaseline().predict(test), test)
    for key in ("ac", "geyser"):
        assert learned[key]["f1"] > 0.9
    assert np.mean([learned[k]["f1"] for k in learned]) > np.mean([base[k]["f1"] for k in base])


def test_estimates_never_exceed_meter(model):
    test = simulate_home(days=2, seed=7)
    pred = model.predict(test)
    total = sum(pred[f"{k}_kw"] for k in DEFAULT_PROFILES)
    assert (total <= pred["mains_kw"] + 1e-6).all()


def test_fridge_cannot_be_switched_off_in_test_mode():
    df = simulate_from_switches({"fridge": [], "geyser": [(400, 425)]})
    assert df["fridge"].sum() > 0
    assert (df["geyser"] > 1.0).sum() == 25


def test_always_on_has_no_sessions_and_only_health_advice(faulty):
    _, pred = faulty
    sessions = extract_sessions(pred)
    assert "fridge" not in set(sessions["appliance"])
    assert allowed_advice(DEFAULT_PROFILES["fridge"]) == {"health"}
    assert not can_switch_off(DEFAULT_PROFILES["fridge"])


def test_no_suggestion_breaks_its_category(faulty):
    _, pred = faulty
    result = recommend(pred, extract_sessions(pred), Tariff(8.0),
                       HouseholdPrefs(min_monthly_saving_rs=0, max_suggestions=20))
    assert result["suggestions"]
    for s in result["suggestions"]:
        profile = DEFAULT_PROFILES[s["appliance"]]
        assert s["kind"] in allowed_advice(profile)
        if profile.category == ALWAYS_ON:
            assert s["kind"] == "health"
    assert not [s for s in result["suggestions"] if s["appliance"] == "ac" and s["kind"] == "shift"]
    assert not [s for s in result["suggestions"] if s["appliance"] == "microwave"]


def test_flat_tariff_gives_no_timing_tips(faulty):
    _, pred = faulty
    result = recommend(pred, extract_sessions(pred), Tariff(8.0, tod_enabled=False),
                       HouseholdPrefs(min_monthly_saving_rs=0, max_suggestions=20))
    assert not [s for s in result["suggestions"] if s["kind"] == "shift"]
    assert any("same all day" in n for n in result["notes"])


def test_long_geyser_run_is_flagged(faulty):
    _, pred = faulty
    flags = flag_long_runs(extract_sessions(pred))
    assert any(f["appliance"] == "geyser" and f["minutes"] > 120 for f in flags)


def test_faulty_fridge_gets_health_tip_not_switch_off(faulty):
    _, pred = faulty
    assert fridge_duty(pred)["duty"].median() > 0.6
    result = recommend(pred, extract_sessions(pred))
    tips = [s for s in result["suggestions"] if s["appliance"] == "fridge"]
    assert tips and tips[0]["kind"] == "health"
    assert "Do not switch it off" in tips[0]["detail"]


def test_dismissed_and_small_tips_are_hidden(faulty):
    _, pred = faulty
    sessions = extract_sessions(pred)
    everything = recommend(pred, sessions, prefs=HouseholdPrefs(min_monthly_saving_rs=0, max_suggestions=20))
    first = everything["suggestions"][0]
    again = recommend(pred, sessions, prefs=HouseholdPrefs(
        min_monthly_saving_rs=0, max_suggestions=20, dismissed={first["id"]}))
    assert first["id"] not in [s["id"] for s in again["suggestions"]]
    strict = recommend(pred, sessions, prefs=HouseholdPrefs(min_monthly_saving_rs=10_000))
    assert strict["suggestions"] == []
    assert len(recommend(pred, sessions, prefs=HouseholdPrefs(
        min_monthly_saving_rs=0, max_suggestions=1))["suggestions"]) == 1


def test_scheduling_limits():
    prefs = HouseholdPrefs()
    washer = DEFAULT_PROFILES["washing_machine"]
    for s in feasible_starts(washer, 60, 19 * 60, prefs):
        assert washer.allowed_hours[0] * 60 <= s and s + 60 <= washer.allowed_hours[1] * 60
    geyser = DEFAULT_PROFILES["geyser"]
    morning = 6 * 60 + 30
    starts = feasible_starts(geyser, 25, morning, prefs)
    assert starts and all(morning - 180 <= s <= morning for s in starts)


def test_tariff_costs():
    t = Tariff(10.0)
    assert t.rate_at(12) == pytest.approx(8.0)
    assert t.rate_at(19) == pytest.approx(12.0)
    assert t.rate_at(2) == pytest.approx(10.0)
    assert t.run_cost(12 * 60, 60, 1.0) == pytest.approx(8.0)
    assert Tariff(10.0, tod_enabled=False).run_cost(19 * 60, 60, 1.0) == pytest.approx(10.0)


def test_totals_add_up_to_meter(faulty):
    _, pred = faulty
    assert totals(pred)["kwh"].sum() == pytest.approx(pred["mains_kw"].sum() / 60, rel=0.01)


def test_chatbot_refusals_and_overrides():
    assert "stay on" in explain_refusal("fridge", "switch_off")
    assert explain_refusal("ac", "shift")
    assert explain_refusal("geyser", "shift") is None
    custom = get_profiles(HouseholdPrefs(overrides={"fridge": {"category": "shiftable", "rated_kw": 0.2}}))
    assert custom["fridge"].category == ALWAYS_ON and custom["fridge"].rated_kw == 0.2


def test_home_only_reports_appliances_it_owns(model):
    from meter_source import owned_appliances, restrict_to_owned
    owned = owned_appliances({"appliances": [{"name": "Air Conditioner"}, {"name": "Refrigerator"},
                                             {"name": "Television"}]})
    assert owned == ("fridge", "ac")
    assert owned_appliances({"appliances": []}) == ()
    df = simulate_home(days=2, seed=5, include=owned)
    assert df["geyser"].sum() == 0 and df["ac"].sum() > 0
    pred = restrict_to_owned(model.predict(df[["datetime", "mains_kw"]]), owned)
    for key in DEFAULT_PROFILES:
        if key not in owned:
            assert not pred[f"{key}_on"].any() and pred[f"{key}_kw"].sum() == 0
    total = sum(pred[f"{k}_kw"] for k in DEFAULT_PROFILES) + pred["other_kw"]
    assert np.allclose(total, pred["mains_kw"], atol=1e-6)


def test_tampering_with_bills_and_reports_is_detected(tmp_path, monkeypatch):
    import sqlite3
    import integrity
    monkeypatch.setattr(integrity, "KEY_FILE", str(tmp_path / "key"))
    monkeypatch.delenv("ENERGYPULSE_SECRET", raising=False)
    ledger = integrity.Ledger(db_path=str(tmp_path / "t.db"))
    bill = b"%PDF bill for October, amount 1840"
    first = ledger.record("home1", "bill", "oct.pdf", bill, {"amount": "1840"})
    ledger.record("home1", "bill", "nov.pdf", b"november bill", {"amount": "2010"})
    assert ledger.verify_chain("home1") == []

    entry = ledger.entries("home1")[0]
    assert ledger.check_content(entry, bill, {"amount": "1840"})[0]
    assert not ledger.check_content(entry, bill.replace(b"1840", b"1340"))[0]      # edited file
    assert not ledger.check_content(entry, bill, {"amount": "1340"})[0]            # edited figure
    assert ledger.find_by_content("home1", bill)["id"] == first["id"]
    assert ledger.find_by_content("home1", b"some other file") is None

    # someone edits the stored record directly in the database
    with sqlite3.connect(str(tmp_path / "t.db")) as conn:
        conn.execute("UPDATE integrity_ledger SET payload_json = ? WHERE id = ?",
                     ('{"amount":"1340"}', first["id"]))
    assert any("changed after saving" in p for p in ledger.verify_chain("home1"))

    # ...or deletes an old record
    ledger2 = integrity.Ledger(db_path=str(tmp_path / "u.db"))
    a = ledger2.record("h", "bill", "a", b"a")
    ledger2.record("h", "bill", "b", b"b")
    with sqlite3.connect(str(tmp_path / "u.db")) as conn:
        conn.execute("DELETE FROM integrity_ledger WHERE id = ?", (a["id"],))
    assert any("removed or reordered" in p for p in ledger2.verify_chain("h"))

    report = integrity.sign_report("Total 386 kWh, Rs. 2342")
    assert integrity.verify_report(report)[0]
    assert not integrity.verify_report(report.replace("2342", "1342"))[0]
    assert not integrity.verify_report("Total 386 kWh")[0]


def test_accounts_only_let_the_owner_in(tmp_path, monkeypatch):
    import auth
    import integrity
    monkeypatch.setattr(integrity, "KEY_FILE", str(tmp_path / "key"))
    monkeypatch.delenv("ENERGYPULSE_SECRET", raising=False)
    accounts = auth.Accounts(db_path=str(tmp_path / "a.db"))
    assert not accounts.sign_up("not-an-email", "longenough1")[0]
    assert not accounts.sign_up("amara@example.com", "short")[0]
    assert accounts.sign_up("Amara@Example.com", "Correct-Horse 9", "Amara") == (True, "Amara")
    # a second person cannot claim the same email, however it is typed
    assert not accounts.sign_up(" amara@example.com ", "Another-Pass 77")[0]
    assert accounts.sign_in("amara@example.com", "Correct-Horse 9") == (True, "Amara")
    assert not accounts.sign_in("amara@example.com", "wrong")[0]
    assert not accounts.sign_in("nobody@example.com", "Correct-Horse 9")[0]

    # the password itself is not in the database
    import sqlite3
    with sqlite3.connect(str(tmp_path / "a.db")) as conn:
        stored = " ".join(map(str, conn.execute("SELECT * FROM credentials").fetchone()))
    assert "Correct-Horse 9" not in stored

    # repeated wrong passwords lock the account, even for the right password
    for _ in range(auth.MAX_FAILURES):
        accounts.sign_in("amara@example.com", "wrong", now=1000.0)
    assert "Too many" in accounts.sign_in("amara@example.com", "Correct-Horse 9", now=1001.0)[1]
    assert accounts.sign_in("amara@example.com", "Correct-Horse 9", now=1000.0 + auth.LOCK_SECONDS + 1)[0]

    # guests are private, sessions end, stored files are unreadable without the key
    assert auth.new_guest_email() != auth.new_guest_email()
    assert auth.session_expired(None) and not auth.session_expired(100.0, now=200.0)
    assert auth.session_expired(100.0, now=100.0 + auth.SESSION_SECONDS + 1)
    blob = auth.protect(b"consumer no. 12345, amount 1840")
    assert b"12345" not in blob and auth.unprotect(blob) == b"consumer no. 12345, amount 1840"
    assert auth.unprotect(blob[:-4] + b"abcd") is None


def test_weak_passwords_are_refused(tmp_path):
    import auth
    accounts = auth.Accounts(db_path=str(tmp_path / "p.db"))
    weak = ["short1!A", "alllowercase12!", "ALLUPPERCASE12!", "NoDigitsHere!!", "NoSymbols12345",
            "Password123!", "Amara-2026-xyz", "Aaaaaaaaa1!"]
    for password in weak:
        ok, why = accounts.sign_up("amara@example.com", password)
        assert not ok and "stronger password" in why, password
    assert auth.password_problems("Tr4il-Mango-Kettle") == []
    assert accounts.sign_up("amara@example.com", "Tr4il-Mango-Kettle")[0]


def test_every_listed_appliance_reaches_the_home():
    from devices import household_devices
    from appliance_profiles import APPLIANCE_KEYS
    listed = [{"name": "Air Conditioner", "type": "Cooling"}, {"name": "Refrigerator", "type": "Kitchen"},
              {"name": "Television", "type": "Electronics"}, {"name": "Lights & Fans", "type": "Lighting"},
              {"name": "Iron box", "type": "Other"}, {"name": "Thingamajig", "type": "Other"}]
    owned, switched, notes = household_devices({"appliances": listed})
    assert owned == ("fridge", "ac")
    names = " ".join(d["name"].lower() for d in switched)
    for word in ["television", "living room fan", "bedroom fan", "living room lights", "iron box", "thingamajig"]:
        assert word in names
    # every switched device can be drawn, described and switched; no two share a place
    assert all(d["kind"] in {"fan", "light", "tv", "generic"} and d["description"] and d["kw"] > 0 for d in switched)
    assert len({d["key"] for d in switched}) == len(switched)
    assert len({(d["kind"], tuple(d["pos"])) for d in switched}) == len(switched)
    # only what was listed is shown: no fridge-less home gets a fridge, an empty list gives an empty home
    assert household_devices(None) == ((), [], [])
    owned2, switched2, notes2 = household_devices({"appliances": [{"name": "Television", "type": "Electronics"}]})
    assert owned2 == () and [d["kind"] for d in switched2] == ["tv"] and notes2
    # a second unit of a detectable type is still shown, from its switch
    owned3, switched3, _ = household_devices({"appliances": [{"name": "Air Conditioner"}, {"name": "Air Conditioner"}]})
    assert owned3 == ("ac",) and len(switched3) == 1 and "air conditioner 2" in switched3[0]["name"].lower()


def test_home_setup_is_remembered(tmp_path):
    from db import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "h.db"))
    assert db.get_home_details("house-1") is None
    details = {"appliances": [{"name": "Microwave", "type": "Kitchen"}], "tariff_rate": 7.5, "peak_morning": (6, 10)}
    db.save_home_details("house-1", details)
    assert db.get_home_details("house-1") == details
    assert db.get_home_details("house-2") is None            # another household sees nothing
    db.save_home_details("house-1", {"appliances": []})
    assert db.get_home_details("house-1") == {"appliances": []}


def test_daily_forecast_is_measured_precise_and_sent_once(faulty, tmp_path):
    from daily_brief import NOTIFICATION_TYPE, brief_text, build_brief, send_daily_brief
    from db import DatabaseManager
    from appliance_profiles import APPLIANCE_KEYS
    pred, tariff = faulty[1], Tariff(8.0, tod_enabled=True)
    brief = build_brief(pred, APPLIANCE_KEYS, tariff)
    # the appliance figures and everything else add up to the day's total
    parts = sum(a["kwh"] for a in brief["appliances"]) + brief["other_kwh"]
    assert abs(parts - brief["total_kwh"]) < 0.15
    assert brief["low_kwh"] <= brief["total_kwh"] <= brief["high_kwh"]
    assert brief["backtest_days"] >= 2 and 0 <= brief["error_pct"] < 40
    # the forecast never looks at today: changing today's readings leaves it unchanged
    tampered = pred.copy()
    last_day = pd.to_datetime(tampered["datetime"]).dt.normalize() == pd.to_datetime(tampered["datetime"]).dt.normalize().iloc[-1]
    for col in [c for c in tampered.columns if c.endswith("_kw")]:
        tampered.loc[last_day, col] = tampered.loc[last_day, col] * 3
    assert build_brief(tampered, APPLIANCE_KEYS, tariff)["total_kwh"] == brief["total_kwh"]
    # every action carries its arithmetic and obeys the appliance category
    for action in brief["actions"]:
        assert action["calculation"] and action["saving_month"] >= 30
        assert "switch off" not in action["title"].lower() or action["appliance"] != "fridge"
    subject, body = brief_text(brief)
    assert f"{brief['total_kwh']:.1f} units" in subject and "By appliance" in body
    # sent once per household per day, to every member
    db = DatabaseManager(db_path=str(tmp_path / "n.db"))
    people = [{"name": "Amara", "email": "amara@example.com"}, {"name": "Ravi", "email": "ravi@example.com"}]
    assert send_daily_brief(db, "house-1", people, brief)["subject"] == subject
    assert send_daily_brief(db, "house-1", people, brief) is None
    log = [e for e in db.get_notification_log("house-1") if e["notification_type"] == NOTIFICATION_TYPE]
    assert len(log) == 2 and {e["recipient_email"] for e in log} == {p["email"] for p in people}
    assert send_daily_brief(db, "house-2", people[:1], brief) is not None
