"""Appliance scheduling: joint planning under the sanctioned load, the combined tip, speed."""
import time

import numpy as np
import pandas as pd
import pytest

from appliance_profiles import APPLIANCE_KEYS, HouseholdPrefs
from meter_sim import simulate_home
from scheduler import _slots, day_loads, recommend, validate_plan
from sessions import extract_sessions
from tariff import Tariff


def _detected(days=21, seed=4):
    """Simulated home with perfect detection, in the detector's output format."""
    sim = simulate_home(days=days, seed=seed)
    pred = pd.DataFrame({"datetime": sim["datetime"], "mains_kw": sim["mains_kw"]})
    for key in APPLIANCE_KEYS:
        kw = sim[key].astype(float) if key in sim else pd.Series(0.0, index=sim.index)
        pred[f"{key}_kw"] = kw
        pred[f"{key}_on"] = kw > 0.02
    pred["other_kw"] = sim["other_kw"]
    return pred


def test_minute_rates_cached_and_correct():
    t = Tariff()
    ref = np.array([t.rate_at(m / 60.0) for m in range(1440)])
    assert np.allclose(t.minute_rates(), ref)
    assert Tariff(tod_enabled=False).minute_rates().max() == 8.0
    start = time.time()
    for _ in range(2000):
        t.run_cost(600, 45, 1.2)
    assert time.time() - start < 1.0


def test_slots_wrap_midnight():
    assert set(_slots(23 * 60 + 50, 30)) == {95, 0, 1}      # 23:50-00:20


def test_day_loads_cover_every_slot():
    loads = day_loads(_detected())
    assert all(len(v) == 96 for v in loads.values()) and len(loads) >= 20


def test_plans_respect_the_cap_on_every_real_day():
    pred = _detected()
    sessions = extract_sessions(pred)
    for cap in (1.5, 2.5, 4.0):
        prefs = HouseholdPrefs(sanctioned_kw=cap, load_margin=1.0, min_monthly_saving_rs=0, max_suggestions=10)
        result = recommend(pred, sessions, Tariff(), prefs)
        check = validate_plan(pred, sessions, result["suggestions"], Tariff(), prefs)
        assert check["all_constraints_ok"], (cap, check["load_cap_violations"])
        assert all(c["ok"] for c in check["checks"])


def test_plan_figures_are_consistent():
    pred = _detected(days=28, seed=7)
    sessions = extract_sessions(pred)
    prefs = HouseholdPrefs(min_monthly_saving_rs=0, max_suggestions=10)
    result = recommend(pred, sessions, Tariff(), prefs)
    v = validate_plan(pred, sessions, result["suggestions"], Tariff(), prefs)
    assert v["shift"]["energy_saving_kwh_month"] == 0.0              # moving a run does not change its units
    assert v["cost_saving_month"] == pytest.approx(v["baseline"]["cost_month"] - v["optimised"]["cost_month"], abs=0.02)
    if v["baseline"]["cost_month"] > 0:
        assert v["cost_saving_pct"] == pytest.approx(v["cost_saving_month"] / v["baseline"]["cost_month"] * 100, abs=0.02)
    assert v["energy_saving_kwh_month"] == pytest.approx(v["baseline"]["kwh_month"] - v["optimised"]["kwh_month"], abs=0.02)
    assert v["cost_saving_month"] >= 0 and v["optimised"]["avg_daily_peak_kw"] <= v["baseline"]["avg_daily_peak_kw"] + 0.05


def test_validator_catches_an_infeasible_plan():
    pred = _detected()
    sessions = extract_sessions(pred)
    day = pd.Timestamp(pred["datetime"].iloc[0]).normalize()
    bad = [{"kind": "shift", "appliance": "washing_machine", "monthly_saving_rs": 50,
            "moves": [{"day": day, "from": 600, "to": 23 * 60, "dur": 60, "kw": 0.5, "kwh": 0.5}]}]
    v = validate_plan(pred, sessions, bad, Tariff(), HouseholdPrefs())
    assert not v["all_constraints_ok"] and v["checks"][0]["problems"]


def test_no_improvement_means_no_invented_savings():
    pred = _detected()
    v = validate_plan(pred, extract_sessions(pred), [], Tariff(), HouseholdPrefs())
    assert v["cost_saving_month"] == 0 and v["energy_saving_kwh_month"] == 0 and v["all_constraints_ok"]


def test_small_moves_are_offered_together():
    pred = _detected(days=28, seed=7)
    sessions = extract_sessions(pred)
    alone = recommend(pred, sessions, Tariff(), HouseholdPrefs(min_monthly_saving_rs=0, max_suggestions=10))
    shifts = [s for s in alone["suggestions"] if s["kind"] == "shift"]
    if len(shifts) < 2:
        return                                                  # this home has nothing to combine
    small = sorted(s["monthly_saving_rs"] for s in shifts)
    threshold = max(small) + 1                                  # every single move is now too small
    result = recommend(pred, sessions, Tariff(), HouseholdPrefs(min_monthly_saving_rs=threshold, max_suggestions=10))
    combos = [s for s in result["suggestions"] if s["id"].startswith("shift:routine:")]
    if sum(small) >= threshold:
        assert combos and combos[0]["monthly_saving_rs"] >= threshold
        assert set(combos[0]["members"]) <= {s["appliance"] for s in shifts}
    else:
        assert not combos


def test_flat_tariff_still_gives_no_timing_tips():
    pred = _detected()
    result = recommend(pred, extract_sessions(pred), Tariff(tod_enabled=False),
                       HouseholdPrefs(min_monthly_saving_rs=0))
    assert not [s for s in result["suggestions"] if s["kind"] == "shift"]
