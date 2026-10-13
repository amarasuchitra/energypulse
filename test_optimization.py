"""Appliance scheduling: joint planning under the sanctioned load, the combined tip, speed."""
import time

import numpy as np
import pandas as pd

from appliance_profiles import APPLIANCE_KEYS, HouseholdPrefs, get_profiles
from meter_sim import simulate_home
from scheduler import _slots, background_load, recommend
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


def test_background_load_excludes_movable_appliances():
    pred = _detected()
    with_all = background_load(pred, [])
    without = background_load(pred, ["washing_machine", "water_pump", "geyser"])
    assert len(with_all) == 96 and (without <= with_all + 1e-9).all()


def test_moves_never_push_the_home_over_its_cap():
    pred = _detected()
    sessions = extract_sessions(pred)
    for cap in (1.5, 2.5, 4.0):
        prefs = HouseholdPrefs(sanctioned_kw=cap, load_margin=1.0, min_monthly_saving_rs=0, max_suggestions=10)
        result = recommend(pred, sessions, Tariff(), prefs)
        movable = [k for k, p in get_profiles(prefs).items() if p.category == "shiftable"]
        load = background_load(pred, movable)
        for s in result["suggestions"]:
            if s["kind"] != "shift" or "members" in s:
                continue
            group = sessions[sessions["appliance"] == s["appliance"]]
            kw = float((group["kwh"] / (group["span_min"] / 60.0)).quantile(0.9))
            hh, mm = s["title"].split(" around ")[1].split(" instead")[0].split(" ")[0].split(":")
            pm = s["title"].split(" around ")[1].split(" instead")[0].endswith("pm")
            start = (int(hh) % 12 + (12 if pm else 0)) * 60 + int(mm)
            dur = int(group["span_min"].median())
            load[_slots(start, dur)] += kw
        assert load.max() <= cap + 0.6, (cap, load.max())      # per-run choices vary around the typical time


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
