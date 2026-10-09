"""
Home tab
========
The 3D home console.  Python prepares one day of main-meter data plus what
the detection model made of it; home_component/ (HTML + three.js, bundled
locally, no internet needed) plays it on a model of the home.

The meter is simulated; no hardware is involved.  Two sources:
  simulated    a simulated day from meter_sim.py
  test mode    the user's own switches, turned into a meter signal

In every case the model is given the meter total only.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from appliance_profiles import (
    ALWAYS_ON, APPLIANCE_KEYS, CATEGORY_LABELS, COMFORT, DEFAULT_PROFILES, ON_DEMAND,
    HouseholdPrefs, can_switch_off,
)
from appliance_ui import meter_settings
from devices import household_devices
from meter_source import (
    HISTORY_DAYS, SCENARIOS, detected_history, get_model as _model, now_minute,
    restrict_to_owned, today_str,
)
from meter_sim import simulate_from_switches
from scheduler import recommend
from sessions import extract_sessions, flag_long_runs
from tariff import Tariff
from clock import local_now

_component = components.declare_component(
    "energy_home", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "home_component"))

COLORS = {"fridge": "#7bc8a4", "ac": "#6fa8dc", "geyser": "#f08a5d",
          "washing_machine": "#b39ddb", "water_pump": "#4fc3c7", "microwave": "#f28fb1"}
PERIOD_COLORS = {"Normal": "#6f7994", "Solar hours": "#5fcb8f", "Evening peak": "#ef6a5b"}
MINUTES = 1440

_NO_TIP = {
    ALWAYS_ON: "Nothing to change. This has to stay on, so it is never told to switch off or run at another time.",
    COMFORT: "No change suggested right now. Advice for this is only ever about temperature and hours, not timing.",
    ON_DEMAND: "Used when it is needed, so no advice is given for it.",
}
_SHIFT_NO_TIP = "No change worth making. Running it at another time would save too little to matter."
_ALERT_ADVICE = {
    "geyser": "Check whether it was left switched on.",
    "water_pump": "Check for a dry run or a leaking tank.",
    "washing_machine": "Check whether the cycle is stuck.",
    "ac": "Check whether the room is still in use.",
    "microwave": "Check that it is not running empty.",
}


METER_EXPLAINER = """
**This meter is a simulation. No physical meter or sensor is connected.**

1. **The home is simulated.** For each appliance you listed, the simulator writes a realistic power
   pattern for every minute of the day: a refrigerator compressor that cycles on and off, an air
   conditioner that runs for hours on warm evenings, a geyser that heats in the morning, and so on.
2. **The patterns are added together.** The sum, plus a background load for lights and standby and a
   little noise, is the **main-meter reading**: one number in kW for each minute, which is all a real
   single-phase household meter measures.
3. **The detection model sees only that one number.** It is never told which appliances were
   running. From the size and shape of the steps in the reading, it works out which appliance
   switched on, for how long, and how much power it drew.
4. **Units and cost come from the reading.** Units (kWh) are the kW values added up over time;
   cost is the units multiplied by the tariff for that hour.
5. **Live and Test.** *Live* replays the simulated day. In *Test* you switch the appliances
   yourself; the simulator turns your switches into a new meter reading, and the model has to find
   them from that reading alone.

The switches change the simulation only. Nothing in a real home is switched, and an appliance you
switch on stays on until you switch it off and confirm it.
"""


def _periods(tariff: Tariff) -> list:
    out, start = [], 0
    for m in range(1, MINUTES + 1):
        if m == MINUTES or tariff.label_at(m / 60) != tariff.label_at(start / 60):
            label = tariff.label_at(start / 60)
            out.append({"start": start, "end": m, "label": label,
                        "mult": tariff.multiplier_at(start / 60), "color": PERIOD_COLORS.get(label, "#6f7994")})
            start = m
    return out


def _pad(values, n=MINUTES) -> list:
    arr = np.zeros(n)
    arr[: len(values)] = np.asarray(values, dtype=float)[:n]
    return [round(float(v), 3) for v in arr]


def _switched_on(on, gap: int, always: bool) -> list:
    """
    Minutes the appliance is switched ON, whether or not it is drawing power:
    short rests inside a run (a thermostat cutting the compressor, a washing
    machine soaking) count as on.  An always-on appliance is on all day.
    """
    on = np.asarray(on, dtype=bool)
    if always:
        return [1] * len(on)
    out = on.copy()
    idx = np.flatnonzero(on)
    for a, g in zip(idx[:-1], np.diff(idx)):
        if 1 < g <= gap + 1:
            out[a:a + g] = True
    return [int(v) for v in out]


def _payload(day: pd.DataFrame, truth, tips: dict, alerts: list, tariff: Tariff,
             mode: str, source: dict, start_minute: int, switches: dict, version: str,
             owned=tuple(APPLIANCE_KEYS)) -> dict:
    appliances = []
    for key in [k for k in APPLIANCE_KEYS if k in owned]:
        prof = DEFAULT_PROFILES[key]
        appliances.append({
            "key": key, "name": prof.name, "color": COLORS[key],
            "category_label": CATEGORY_LABELS[prof.category], "note": prof.note,
            "locked": not can_switch_off(prof),
            "product": prof.product, "rated_kw": prof.rated_kw,
            "kw": _pad(day[f"{key}_kw"]),
            "on": [int(v) for v in _pad(day[f"{key}_on"].astype(int))],
            "run": _switched_on(_pad(day[f"{key}_on"].astype(int)), max(prof.merge_gap_min, 12),
                                prof.category == ALWAYS_ON),
            "truth": None if truth is None else
                     [int(v) for v in _pad((truth[key] > prof.on_threshold_kw).astype(int))],
            "tips": tips.get(key, []),
            "no_tip": "Test mode. Switch it on and watch whether the meter gives it away."
                      if mode == "test" else _NO_TIP.get(prof.category, _SHIFT_NO_TIP),
        })
    return {
        "version": version, "mode": mode, "source": source, "switches": switches,
        "date_label": pd.Timestamp(day["datetime"].iloc[0]).strftime("%a %-d %b")
        if os.name != "nt" else pd.Timestamp(day["datetime"].iloc[0]).strftime("%a %d %b"),
        "start_minute": start_minute, "live_edge": len(day),
        "mains": _pad(day["mains_kw"]), "rates": [round(float(r), 3) for r in tariff.minute_rates()],
        "periods": _periods(tariff), "appliances": appliances, "alerts": alerts,
    }


def _tips_and_alerts(pred: pd.DataFrame, today: pd.DataFrame, tariff: Tariff):
    sessions = extract_sessions(pred, tariff)
    result = recommend(pred, sessions, tariff, HouseholdPrefs(max_suggestions=10))
    tips = {}
    for s in result["suggestions"]:
        tips.setdefault(s["appliance"], []).append(
            {"title": s["title"], "detail": s["detail"], "saving": int(s["monthly_saving_rs"])})
    midnight = pd.Timestamp(today["datetime"].iloc[0])
    alerts = []
    for f in flag_long_runs(extract_sessions(today, tariff)):
        prof = DEFAULT_PROFILES[f["appliance"]]
        start = int((f["start"] - midnight).total_seconds() // 60)
        alerts.append({
            "key": f["appliance"], "name": prof.name, "start": start, "end": start + f["minutes"],
            "warn_at": start + prof.max_run_min,
            "normal": f"{prof.typical_run_min[0]}-{prof.typical_run_min[1]} min",
            "advice": _ALERT_ADVICE.get(f["appliance"], ""),
        })
    return tips, alerts


@st.cache_data(show_spinner=False)
def _simulated_day(date: str, scenario: str, rate: float, tod: bool, owned: tuple) -> dict:
    tariff = Tariff(rate, tod_enabled=tod)
    pred = detected_history(date, scenario, owned)          # same history every tab uses
    today = pred.tail(MINUTES).reset_index(drop=True)
    tips, alerts = _tips_and_alerts(pred, today, tariff)
    version = hashlib.md5(f"live|{date}|{scenario}|{rate}|{tod}|{owned}".encode()).hexdigest()
    return _payload(today, None, tips, alerts, tariff, "live",
                    {"label": "Simulated meter", "real": False}, 330, {}, version, owned)


@st.cache_data(show_spinner=False)
def _test_day(date: str, switches_json: str, rate: float, tod: bool, owned: tuple) -> dict:
    tariff = Tariff(rate, tod_enabled=tod)
    switches = {k: v for k, v in json.loads(switches_json).items() if k in owned}
    df = simulate_from_switches(switches, seed=11, start=date)
    if "fridge" not in owned:
        df["mains_kw"] = (df["mains_kw"] - df["fridge"]).clip(lower=0)
        df["fridge"] = 0.0
    pred = restrict_to_owned(_model().predict(df[["datetime", "mains_kw"]]), owned)
    version = hashlib.md5(f"test|{date}|{switches_json}|{rate}|{tod}|{owned}".encode()).hexdigest()
    return _payload(pred, df, {}, [], tariff, "test",
                    {"label": "Test signal", "real": False}, 360, switches, version, owned)


def build_payload(mode: str, switches: dict, rate: float, tod: bool, scenario: str,
                  owned=tuple(APPLIANCE_KEYS)) -> dict:
    today = today_str()
    if mode == "test":
        return _test_day(today, json.dumps(switches, sort_keys=True), float(rate), tod, owned)
    # Live: the day is known only up to this minute on the household's clock.
    live = dict(_simulated_day(today, scenario, float(rate), tod, owned))
    minute = now_minute()
    live.update({"follow": True, "now_minute": minute, "start_minute": minute, "live_edge": minute + 1,
                 "alerts": [a for a in live["alerts"] if a["warn_at"] <= minute],
                 "version": hashlib.md5(f"{live['version']}|{minute}".encode()).hexdigest()})
    return live


def _members(db, household_id: str, user_name: str, user_email: str) -> list:
    people = [{"name": user_name, "email": user_email}]
    for m in db.get_family_members(household_id):
        people.append({"name": m.get("name") or m.get("email"), "email": m.get("email"), "phone": m.get("phone")})
    return people


def _device_name(key: str, switched: list) -> str:
    dev = next((f for f in switched if f["key"] == key), None)
    return dev["name"] if dev else DEFAULT_PROFILES[key].name


def _log_switch(db, household_id: str, people: list, event: dict, language: str, switched: list) -> None:
    """Tell the whole household who switched what.  Shown in the app; no email is sent."""
    device = str(event.get("device", ""))
    if device not in APPLIANCE_KEYS and device not in [f["key"] for f in switched]:
        return
    who = str(event.get("member") or people[0]["name"])[:60]
    if who not in [p["name"] for p in people]:
        who = people[0]["name"]
    text = f"{who} switched the {_device_name(device, switched).lower()} {'on' if event.get('on') else 'off'}"
    for person in people:
        if person.get("email"):
            db.log_notification(household_id, person["email"], person["name"], "appliance_switch",
                                who, {"device": device, "on": bool(event.get("on"))},
                                text, text, language, status="in-app")


def _local_hhmm(stamp) -> str:
    """The database stores UTC; people read the household's own time."""
    try:
        from clock import to_local
        return to_local(stamp).strftime("%H:%M")
    except Exception:
        return str(stamp or "")[11:16]


def _recent_activity(db, household_id: str, limit: int = 6) -> list:
    seen, out = set(), []
    for entry in db.get_notification_log(household_id, limit=200):
        if entry.get("notification_type") != "appliance_switch":
            continue
        key = (entry.get("subject"), entry.get("sent_at"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": entry.get("subject") or "", "time": _local_hhmm(entry.get("sent_at"))})
        if len(out) >= limit:
            break
    return out


def render_home_tab(tariff_rate: float, home_details=None, db=None, household_id: str = "",
                    user_name: str = "You", user_email: str = "", theme: str = "dark",
                    language: str = "en"):
    state = st.session_state.setdefault(
        "home_state", {"mode": "live", "switches": {}, "nonce": None, "fans": {}})
    state.setdefault("fans", {})
    scenario, tod = meter_settings()
    owned, switched, device_notes = household_devices(home_details)
    if not owned and not switched:
        st.info("Your home has no appliances yet. Open Settings, choose Edit home and add the appliances "
                "you have. Only what you add is shown here.")
        return
    state["fans"] = {k: v for k, v in state["fans"].items() if k in {f["key"] for f in switched}}
    people = _members(db, household_id, user_name, user_email) if db else [{"name": user_name, "email": ""}]

    # No spinner: it would push the console down on every switch in test mode.
    # The console shows its own "Updating the meter signal..." note.
    payload = dict(build_payload(state["mode"], state["switches"], tariff_rate, tod, scenario, owned))
    activity = _recent_activity(db, household_id) if db else []
    from devices import appliance_specs
    spec_of = appliance_specs(home_details)
    payload["appliances"] = [{**a, "specs": spec_of.get(a["key"])} for a in payload["appliances"]]
    payload.update({
        "home_type": (home_details or {}).get("home_type", "Apartment"),
        "theme": theme, "fans": switched, "fan_state": state["fans"],
        "members": [p["name"] for p in people], "activity": activity, "ack": state.get("nonce"),
        "household_size": len(people),
    })
    payload["version"] = hashlib.md5(
        f"{payload['version']}|{theme}|{sorted(state['fans'].items())}|{activity[:1]}|{len(people)}|{state.get('nonce')}|{(home_details or {}).get('home_type')}|{sorted(spec_of)}|{[f['key'] for f in switched]}".encode()
    ).hexdigest()

    value = _component(data=payload, key="energy_home", default=None, height=1000)
    if value and value.get("nonce") != state["nonce"]:
        clean = {k: [[int(a), int(b)] for a, b in v][:20]
                 for k, v in (value.get("switches") or {}).items() if k in APPLIANCE_KEYS}
        fans = {f["key"]: bool((value.get("fans") or {}).get(f["key"])) for f in switched}
        if db and isinstance(value.get("event"), dict):
            _log_switch(db, household_id, people, value["event"], language, switched)
        st.session_state["home_state"] = {
            "mode": "test" if value.get("mode") == "test" else "live",
            "switches": clean, "nonce": value.get("nonce"), "fans": fans}
        st.rerun()

    with st.expander("How the meter works"):
        st.markdown(METER_EXPLAINER)
    st.caption("No meter is connected: this is a simulated meter, shown at the current time on your clock. "
               "Drag the timeline to look back over today. The house shows only what "
               f"you entered during setup: {len(owned)} appliance(s) found from the meter and "
               f"{len(switched)} device(s) shown from their switch. Add or remove items under Settings, Edit home.")
    for note in device_notes:
        st.caption(note)
