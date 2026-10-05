"""
Home tab
========
The 3D home console.  Python prepares one day of main-meter data plus what
the detection model made of it; home_component/ (HTML + three.js, bundled
locally, no internet needed) plays it on a model of the home.

Three data sources, picked automatically:
  real meter   data/live_meter.csv has fresh readings (see live_meter.py)
  simulated    otherwise, a simulated day from meter_sim.py
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
    FANS, ALWAYS_ON, APPLIANCE_KEYS, CATEGORY_LABELS, COMFORT, DEFAULT_PROFILES, ON_DEMAND,
    HouseholdPrefs, can_switch_off,
)
from appliance_ui import meter_settings
from meter_source import (
    HISTORY_DAYS, SCENARIOS, detected_history, get_model as _model, owned_appliances,
    restrict_to_owned, today_str,
)
from live_meter import load_today
from meter_sim import simulate_from_switches
from scheduler import recommend
from sessions import extract_sessions, flag_long_runs
from tariff import Tariff

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


def _real_day(feed: pd.DataFrame, device: str, rate: float, tod: bool,
              owned=tuple(APPLIANCE_KEYS)) -> dict:
    tariff = Tariff(rate, tod_enabled=tod)
    pred = restrict_to_owned(_model().predict(feed), owned)
    tips, alerts = _tips_and_alerts(pred, pred, tariff)
    version = hashlib.md5(f"real|{feed['datetime'].iloc[-1]}|{rate}|{tod}|{owned}".encode()).hexdigest()
    return _payload(pred, None, tips, alerts, tariff, "live",
                    {"label": f"Meter feed: {device}", "real": True}, len(pred) - 1, {}, version, owned)


def build_payload(mode: str, switches: dict, rate: float, tod: bool, scenario: str,
                  owned=tuple(APPLIANCE_KEYS)) -> dict:
    today = today_str()
    if mode == "test":
        return _test_day(today, json.dumps(switches, sort_keys=True), float(rate), tod, owned)
    feed, device = load_today()
    if feed is not None:
        return _real_day(feed, device, float(rate), tod, owned)
    return _simulated_day(today, scenario, float(rate), tod, owned)


def _members(db, household_id: str, user_name: str, user_email: str) -> list:
    people = [{"name": user_name, "email": user_email}]
    for m in db.get_family_members(household_id):
        people.append({"name": m.get("name") or m.get("email"), "email": m.get("email")})
    return people


def _device_name(key: str) -> str:
    fan = next((f for f in FANS if f["key"] == key), None)
    return fan["name"] if fan else DEFAULT_PROFILES[key].name


def _log_switch(db, household_id: str, people: list, event: dict, language: str) -> None:
    """Tell the whole household who switched what.  Shown in the app; no email is sent."""
    device = str(event.get("device", ""))
    if device not in APPLIANCE_KEYS and device not in [f["key"] for f in FANS]:
        return
    who = str(event.get("member") or people[0]["name"])[:60]
    if who not in [p["name"] for p in people]:
        who = people[0]["name"]
    text = f"{who} switched the {_device_name(device).lower()} {'on' if event.get('on') else 'off'}"
    for person in people:
        if person.get("email"):
            db.log_notification(household_id, person["email"], person["name"], "appliance_switch",
                                who, {"device": device, "on": bool(event.get("on"))},
                                text, text, language, status="in-app")


def _local_hhmm(stamp) -> str:
    """The database stores UTC; people read local time."""
    try:
        return pd.Timestamp(stamp, tz="UTC").tz_convert(pd.Timestamp.now().astimezone().tzinfo).strftime("%H:%M")
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
    owned = owned_appliances(home_details)
    people = _members(db, household_id, user_name, user_email) if db else [{"name": user_name, "email": ""}]

    # No spinner: it would push the console down on every switch in test mode.
    # The console shows its own "Updating the meter signal..." note.
    payload = dict(build_payload(state["mode"], state["switches"], tariff_rate, tod, scenario, owned))
    activity = _recent_activity(db, household_id) if db else []
    payload.update({
        "theme": theme, "fans": FANS, "fan_state": state["fans"],
        "members": [p["name"] for p in people], "activity": activity,
        "household_size": len(people),
    })
    payload["version"] = hashlib.md5(
        f"{payload['version']}|{theme}|{sorted(state['fans'].items())}|{activity[:1]}|{len(people)}".encode()
    ).hexdigest()

    value = _component(data=payload, key="energy_home", default=None, height=1000)
    if value and value.get("nonce") != state["nonce"]:
        clean = {k: [[int(a), int(b)] for a, b in v][:20]
                 for k, v in (value.get("switches") or {}).items() if k in APPLIANCE_KEYS}
        fans = {f["key"]: bool((value.get("fans") or {}).get(f["key"])) for f in FANS}
        if db and isinstance(value.get("event"), dict):
            _log_switch(db, household_id, people, value["event"], language)
        st.session_state["home_state"] = {
            "mode": "test" if value.get("mode") == "test" else "live",
            "switches": clean, "nonce": value.get("nonce"), "fans": fans}
        st.rerun()

    if payload["source"]["real"]:
        st.caption(f"Showing readings from {payload['source']['label']}. Detection runs on those readings.")
    else:
        st.caption("No meter is connected, so this is a replay of a simulated day. The house shows the "
                   "appliances you listed during setup. Change the simulated home on the Settings page.")
