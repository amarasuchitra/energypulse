"""Checks for the household toolkit sums (toolkit.py)."""
import numpy as np
import pandas as pd
import pytest

import toolkit as tk
from appliance_profiles import APPLIANCE_KEYS
from daily_brief import _daily
from meter_sim import simulate_home
from sessions import extract_sessions
from tariff import Tariff

DELHI = tk.TARIFF_PRESETS["Delhi (domestic)"]


@pytest.fixture(scope="module")
def home():
    """Ground-truth simulated home in the same column layout the detector produces."""
    df = simulate_home(days=24, seed=8, faults=["geyser_left_on", "fridge_seal"])
    pred = df[["datetime", "mains_kw"]].copy()
    for key in APPLIANCE_KEYS:
        pred[f"{key}_kw"] = df[key]
        pred[f"{key}_on"] = df[key] > 0.05
    pred["other_kw"] = (df["mains_kw"] - df[APPLIANCE_KEYS].sum(axis=1)).clip(lower=0)
    tariff = Tariff(8.0, tod_enabled=True)
    kwh, cost = _daily(pred, APPLIANCE_KEYS, tariff)
    return pred, tariff, kwh, cost, extract_sessions(pred, tariff)


def test_slab_bill_is_worked_out_block_by_block():
    energy, lines = tk.energy_charge(450, DELHI["slabs"])
    assert energy == 200 * 3.0 + 200 * 4.5 + 50 * 6.5 and len(lines) == 3
    assert tk.energy_charge(0, DELHI["slabs"])[0] == 0
    assert tk.fixed_charge(3, DELHI["fixed"]) == 150 and tk.fixed_charge(2, DELHI["fixed"]) == 40
    res = tk.bill_check(450, DELHI, 3)
    assert abs(res["total"] - (1825 + 150 + 1825 * 0.08)) < 1e-6
    assert tk.bill_check(450, DELHI, 3, billed=res["total"] * 1.02)["verdict"] == "matches"
    high = tk.bill_check(450, DELHI, 3, billed=res["total"] + 400)
    assert high["verdict"] == "higher" and abs(high["difference"] - 400) < 1e-6
    assert tk.bill_check(450, DELHI, 3, billed=res["total"] - 400)["verdict"] == "lower"
    flat = tk.bill_check(100, tk.TARIFF_PRESETS["Karnataka (BESCOM, domestic)"], 2)
    assert abs(flat["energy"] - 580) < 1e-6 and flat["fixed"] == 290


def test_slab_warning_says_when_and_what_it_costs():
    days = pd.date_range("2026-10-01", periods=11)
    position = tk.month_position(pd.Series([12.0] * 11, index=days))
    assert position["used"] == 120 and position["days_left"] == 21 and abs(position["projected"] - 372) < 1e-6
    warn = tk.slab_warning(position, DELHI["slabs"])
    assert warn["boundary"] == 200 and warn["will_cross"] and warn["next_rate"] == 4.5
    assert abs(warn["extra_cost"] - 172 * 1.5) < 1e-6 and abs(warn["allowance"] - 80 / 21) < 1e-6
    assert 16 <= warn["cross_day"] <= 18
    light = tk.slab_warning(tk.month_position(pd.Series([4.0] * 11, index=days)), DELHI["slabs"])
    assert not light["will_cross"]
    assert tk.slab_warning(position, [(None, 5.8)])["flat"]


def test_bill_change_is_split_by_appliance(home):
    _, _, kwh, cost, _ = home
    why = tk.explain_change(kwh, cost, span=10)
    assert abs(sum(r["delta_cost"] for r in why["rows"]) - why["delta_cost"]) < 0.01
    assert abs(sum(r["delta_kwh"] for r in why["rows"]) - why["delta_kwh"]) < 0.01


def test_faults_and_overload_are_found(home):
    pred, _, _, _, sessions = home
    events = tk.health_events(pred, sessions, APPLIANCE_KEYS)
    kinds = {(e["appliance"], e["kind"]) for e in events}
    assert ("geyser", "Long run") in kinds and ("fridge", "Compressor overworking") in kinds
    tight = tk.peak_demand(pred, 1.0, APPLIANCE_KEYS)
    roomy = tk.peak_demand(pred, 50.0, APPLIANCE_KEYS)
    assert tight["over"] and not roomy["near"] and tight["worst"]["kw"] == roomy["worst"]["kw"]
    assert tight["worst"]["kw"] <= pred["mains_kw"].max() + 1e-9


def test_left_on_is_notified_once(tmp_path, home):
    from db import DatabaseManager
    from sessions import flag_long_runs
    _, _, _, _, sessions = home
    flags = flag_long_runs(sessions)[:2]
    db = DatabaseManager(db_path=str(tmp_path / "l.db"))
    people = [{"name": "Amara", "email": "amara@example.com"}]
    assert len(tk.send_left_on(db, "h1", people, flags)) == len(flags) > 0
    assert tk.send_left_on(db, "h1", people, flags) == []


def test_upgrade_sums(home):
    pred = home[0]
    rows = tk.replacement_table({"fridge": 60.0, "ac": 200.0, "microwave": 5.0}, 8.0)
    by = {r["key"]: r for r in rows}
    assert set(by) == {"fridge", "ac"}
    assert abs(by["ac"]["saved_rs"] - 200 * 0.30 * 8) < 1e-6
    assert abs(by["ac"]["payback_months"] - 42000 / 480) < 1e-6
    assert tk.solar_subsidy(1) == 30000 and tk.solar_subsidy(2) == 60000 and tk.solar_subsidy(3) == 78000
    assert tk.solar_subsidy(8) == 78000
    with_net = tk.solar_estimate(pred, 8.0, kwp=3, net_metering=True)
    without = tk.solar_estimate(pred, 8.0, kwp=3, net_metering=False)
    assert with_net["generated"] == 360 and without["used"] <= with_net["used"] <= with_net["month_kwh"] + 1e-6
    assert with_net["net_cost"] == 3 * 60000 - 78000 and with_net["payback_years"] > 0
    size = tk.backup_size([{"name": "Fan", "kw": 0.07}, {"name": "TV", "kw": 0.1}, {"name": "Geyser", "kw": 2.0}], 3)
    assert abs(size["va"] - 2170 / 0.8 * 1.25) < 1e-6 and size["heavy"] == ["Geyser"]
    small = tk.backup_size([{"name": "Fan", "kw": 0.07}, {"name": "TV", "kw": 0.1}], 4)
    assert small["standard_va"] == 600 and abs(small["ah"] - 170 * 4 / 9.6) < 1e-6 and small["batteries"] == 1


def test_goal_schedule_and_carbon(home):
    _, _, _, cost, sessions = home
    days = pd.date_range("2026-10-01", periods=11)
    g = tk.goal_status(pd.Series([100.0] * 8 + [60.0, 50.0, 999.0], index=days), 3100.0)
    assert g["spent"] == 910 and g["streak"] == 10 and g["days_left"] == 21
    assert not tk.goal_status(pd.Series([200.0] * 11, index=days), 3100.0)["on_track"]
    assert tk.goal_status(pd.Series([200.0] * 11, index=days), 3100.0)["streak"] == 0
    report = tk.schedule_report(sessions, {"geyser": (0, 1440), "washing_machine": (0, 1)})
    by = {r["key"]: r for r in report}
    assert by["geyser"]["runs"] > 0 and not by["geyser"]["outside"]
    assert by["washing_machine"]["runs"] == len(by["washing_machine"]["outside"])
    assert abs(tk.carbon_kg(100) - 71) < 1e-6


def test_member_activity_pairs_on_with_off():
    import json
    def row(who, device, on, at):
        return {"notification_type": "appliance_switch", "triggered_by": who, "sent_at": at,
                "subject": f"{who} {device} {on}", "trigger_data_json": json.dumps({"device": device, "on": on})}
    log = [row("Amara", "tv", True, "2026-10-05 10:00:00"), row("Amara", "tv", True, "2026-10-05 10:00:00"),
           row("Ravi", "tv", False, "2026-10-05 12:00:00"), row("Ravi", "fan", True, "2026-10-05 12:30:00"),
           {"notification_type": "daily_forecast", "sent_at": "2026-10-05 08:00:00"}]
    rows = {r["name"]: r for r in tk.member_activity(log, {"tv": 0.1, "fan": 0.07}, {"tv": "Television"}, 8.0)}
    assert rows["Amara"]["switches"] == 1 and rows["Ravi"]["switches"] == 2      # the duplicate row is one event
    assert rows["Amara"]["minutes"] == 120 and abs(rows["Amara"]["cost"] - 0.2 * 8) < 1e-6
    assert rows["Amara"]["most_used"] == "Television"


def test_weekly_pdf_complaint_and_outages():
    pdf = tk.weekly_report_pdf({
        "household": "Amara", "prepared": "05 Oct 2026", "period": "28 Sep to 04 Oct 2026", "kwh": 98.4, "cost": 790.0,
        "change_pct": -4.0, "appliances": [{"name": "Air Conditioner", "kwh": 40.0, "cost": 320.0, "share": 41.0}],
        "actions": [{"title": "Raise the AC by 2 degrees", "saving_month": 150.0, "calculation": "12% of Rs. 1250"}],
        "events": [{"when": pd.Timestamp("2026-10-02"), "name": "Geyser", "kind": "Long run", "text": "Ran 95 min."}],
        "members": [{"name": "Amara", "switches": 3, "most_used": "Television"}], "co2": 69.9})
    assert pdf[:5] == b"%PDF-" and len(pdf) > 1500
    letter = tk.complaint_letter("Bill amount looks wrong", "Amara", "C-123", "12 Lake Road", "BESCOM",
                                 ["Units billed: 450.", "Difference: Rs. 400 more than expected."])
    for text in ["Consumer No. C-123", "BESCOM", "1. Units billed: 450.", "Yours faithfully,", "Amara"]:
        assert text in letter
    today = pd.Timestamp("2026-10-05")
    out = tk.outage_summary([{"date": "2026-10-02", "start": "18:00", "minutes": 90, "note": ""},
                             {"date": "2026-09-20", "start": "07:00", "minutes": 30, "note": ""},
                             {"date": "2026-07-01", "start": "07:00", "minutes": 600, "note": ""}], today)
    assert out["hours_month"] == 1.5 and out["hours_30"] == 2.0 and out["longest"] == 600 and out["count"] == 3
    assert tk.outage_summary([], today)["count"] == 0


def test_settings_store_is_private_to_a_household(tmp_path):
    from db import DatabaseManager
    db = DatabaseManager(db_path=str(tmp_path / "s.db"))
    assert db.get_store("h1", "toolkit", {}) == {}
    db.set_store("h1", "toolkit", {"goal": 2500})
    db.set_store("h1", "toolkit", {"goal": 2600})
    assert db.get_store("h1", "toolkit") == {"goal": 2600} and db.get_store("h2", "toolkit") is None


def test_usage_levels_and_household_size_change_the_simulated_home():
    from meter_source import habits_key, split_scenario, with_habits
    base = simulate_home(days=14, seed=3)
    heavy = simulate_home(days=14, seed=3, usage={"ac": 1.5})
    light = simulate_home(days=14, seed=3, usage={"ac": 0.6})
    assert light["ac"].sum() < base["ac"].sum() < heavy["ac"].sum()
    assert abs(heavy["fridge"].sum() - base["fridge"].sum()) < 1e-6          # the fridge is not a habit
    big, small = simulate_home(days=14, seed=3, people=7), simulate_home(days=14, seed=3, people=1)
    for col in ["geyser", "other_kw"]:
        assert small[col].sum() < base[col].sum() < big[col].sum()
    home = {"occupants": 6, "appliances": [{"name": "Air Conditioner", "usage": "High"},
                                           {"name": "Refrigerator", "usage": "Medium"},
                                           {"name": "Water Heater", "usage": "Low"}]}
    assert habits_key(home) == "ac=1.5,geyser=0.6;people=6"
    assert split_scenario(with_habits("Geyser left on by mistake", home)) == (
        "Geyser left on by mistake", {"ac": 1.5, "geyser": 0.6}, 6.0)
    assert split_scenario("Typical summer home") == ("Typical summer home", {}, 4.0)
    assert with_habits("Typical summer home", home) != with_habits("Typical summer home", {**home, "occupants": 2})


def test_right_now_only_counts_what_has_happened(home):
    from daily_brief import right_now
    pred, tariff = home[0], home[1]
    early, late = right_now(pred, APPLIANCE_KEYS, tariff, 6 * 60), right_now(pred, APPLIANCE_KEYS, tariff, 21 * 60)
    assert early["kwh"] < late["kwh"] and early["cost"] < late["cost"]
    assert len(early["trace"]) == 361 and len(late["typical"]) == 1440
    today = pred.tail(1440)
    assert abs(late["kwh"] - today["mains_kw"].iloc[:1261].sum() / 60) < 1e-6
    assert late["next_lo"] <= late["next_kw"] <= late["next_hi"] and late["next_days"] >= 7
    assert right_now(pred, APPLIANCE_KEYS, tariff, 23 * 60 + 50)["next_days"] >= 7     # hour that wraps midnight
