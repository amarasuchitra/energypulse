"""
One meter history for the whole app
===================================
Home, Appliances, Analysis and Save Energy all read the SAME detected
history from here, so their numbers always agree.

    pred = detected_history(date, scenario, owned)   # 1-minute rows

`owned` is the set of appliances the household said it has during setup.
Appliances the home does not own are left out of the simulation and are
never reported by the detector.

The meter is simulated from the household's appliance list; there is no hardware.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from appliance_profiles import APPLIANCE_KEYS
from disaggregate import MODEL_PATH, Disaggregator, train_default
from meter_sim import simulate_home
from clock import local_now

HISTORY_DAYS = 30

SCENARIOS = {
    "Typical summer home": [],
    "Laundry always in the evening peak": ["evening_laundry"],
    "Geyser left on by mistake": ["geyser_left_on"],
    "Fridge with a worn door seal": ["fridge_seal"],
    "Water pump running dry": ["pump_dry_run"],
    "All of the above": ["evening_laundry", "geyser_left_on", "fridge_seal", "pump_dry_run"],
}

def owned_appliances(home_details: Optional[dict]) -> Tuple[str, ...]:
    """
    Detector keys for the appliances this household listed during setup.
    Only listed appliances are returned; the tuple can be empty.  (devices.household_devices also returns the switched devices.)
    """
    from devices import household_devices
    return household_devices(home_details)[0]


@st.cache_resource(show_spinner=False)
def get_model() -> Disaggregator:
    """Load the saved model; retrain (about a minute) if it cannot be loaded."""
    try:
        return Disaggregator.load(MODEL_PATH)
    except Exception:
        model, _ = train_default(verbose=False)
        return model


def restrict_to_owned(pred: pd.DataFrame, owned: Iterable[str]) -> pd.DataFrame:
    """Anything 'detected' for an appliance the home does not have is moved to other."""
    owned = set(owned)
    pred = pred.copy()
    for key in APPLIANCE_KEYS:
        if key not in owned:
            pred["other_kw"] = pred["other_kw"] + pred[f"{key}_kw"]
            pred[f"{key}_kw"] = 0.0
            pred[f"{key}_on"] = False
    return pred


USAGE_FACTOR = {"Low": 0.6, "Medium": 1.0, "High": 1.5}


def habits_key(home_details: Optional[dict]) -> str:
    """The household's usage levels and size as a short text, e.g. 'ac=1.5,geyser=0.6;people=5'."""
    from devices import detected_key
    hd = home_details or {}
    levels = {}
    for item in hd.get("appliances", []) or []:
        if not isinstance(item, dict):
            continue
        key = detected_key(str(item.get("name", "")))
        if key and key not in levels:
            levels[key] = USAGE_FACTOR.get(item.get("usage", "Medium"), 1.0)
    parts = ",".join(f"{k}={v}" for k, v in sorted(levels.items()) if v != 1.0)
    try:
        people = int(hd.get("occupants", 4) or 4)
    except (TypeError, ValueError):
        people = 4
    try:
        size = int(hd.get("home_size") or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:                                   # no size given: a typical one for the home type
        size = TYPE_AREA.get(str(hd.get("home_type") or ""), 0)
    from devices import household_ratings
    rated = ",".join(f"{k}:{v}" for k, v in sorted(household_ratings(hd).items()))
    city = "".join(ch for ch in str(hd.get("city") or "").strip().lower() if ch.isalpha() or ch == " ")[:30]
    return (f"{parts};people={people}" + (f";size={size}" if size > 0 else "") + (f";city={city}" if city else "")
            + (f";rated={rated}" if rated else ""))


def with_habits(scenario: str, home_details: Optional[dict]) -> str:
    return f"{scenario}||{habits_key(home_details)}"


def split_scenario(scenario: str):
    """'Typical summer home||ac=1.5;people=5' -> (name, {'ac': 1.5}, 5.0).  A bare name is the reference home."""
    name, _, habits = str(scenario).partition("||")
    usage, people = {}, 4.0
    if habits:
        levels, _, tail = habits.partition(";")
        for pair in filter(None, levels.split(",")):
            k, _, v = pair.partition("=")
            try:
                usage[k] = float(v)
            except ValueError:
                pass
        try:
            people = float(dict(p.partition("=")[::2] for p in tail.split(";")).get("people", 4) or 4)
        except ValueError:
            people = 4.0
    return (name if name in SCENARIOS else list(SCENARIOS)[0]), usage, people


TYPE_AREA = {"Studio": 450, "Apartment": 1000, "Independent House": 1400, "Villa": 2200}


def scenario_ratings(scenario: str) -> dict:
    """Running power (kW) of the household's chosen appliance models, from the scenario key."""
    fields = dict(p.partition("=")[::2] for p in str(scenario).partition("||")[2].split(";") if "=" in p)
    out = {}
    for pair in filter(None, fields.get("rated", "").split(",")):
        k, _, v = pair.partition(":")
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def scenario_place(scenario: str):
    """(city, floor area in sq ft) carried in the scenario key; ('', 1000.0) when not given."""
    fields = dict(p.partition("=")[::2] for p in str(scenario).partition("||")[2].split(";") if "=" in p)
    try:
        size = float(fields.get("size") or 1000.0)
    except ValueError:
        size = 1000.0
    return fields.get("city", ""), size


def temperatures(date: str, scenario: str, days: int = HISTORY_DAYS) -> pd.Series:
    """Simulated daily high for each day of the history that ends on `date` (see weather.py)."""
    from weather import daily_high
    start = pd.Timestamp(date) - pd.Timedelta(days=days - 1)
    return daily_high(pd.date_range(start, periods=days, freq="D"), scenario_place(scenario)[0])


@st.cache_data(show_spinner=False)
def detected_history(date: str, scenario: str, owned: Tuple[str, ...],
                     days: int = HISTORY_DAYS) -> pd.DataFrame:
    """`days` of 1-minute detection results, ending at the end of `date`."""
    start = pd.Timestamp(date) - pd.Timedelta(days=days - 1)
    name, usage, people = split_scenario(scenario)
    from meter_sim import random_ratings
    ratings = {**random_ratings(np.random.default_rng(777)), **scenario_ratings(scenario)}
    df = simulate_home(days=days, seed=777, start=str(start.date()), ratings=ratings,
                       faults=SCENARIOS[name], include=owned, usage=usage, people=people,
                       temps=temperatures(date, scenario, days).tolist(),
                       area_sqft=scenario_place(scenario)[1])
    pred = get_model().predict(df[["datetime", "mains_kw"]])
    return restrict_to_owned(pred, owned)


def last_days(pred: pd.DataFrame, days: int) -> pd.DataFrame:
    return pred.tail(days * 1440).reset_index(drop=True)


def now_minute() -> int:
    """Minutes since midnight on the household's clock."""
    now = local_now()
    return int(now.hour * 60 + now.minute)


def today_str() -> str:
    return str(local_now().date())


SOURCE_NOTE = ("Simulated main-meter feed. No physical meter is connected, so appliance "
               "figures come from a simulated 1-minute meter signal. The detection model "
               "reads only the meter total.")
