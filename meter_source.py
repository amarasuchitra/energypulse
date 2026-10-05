"""
One meter history for the whole app
===================================
Home, Appliances, Analysis and Save Energy all read the SAME detected
history from here, so their numbers always agree.

    pred = detected_history(date, scenario, owned)   # 1-minute rows

`owned` is the set of appliances the household said it has during setup.
Appliances the home does not own are left out of the simulation and are
never reported by the detector.

To move the app onto a real recording later, this is the one place to change.
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from appliance_profiles import APPLIANCE_KEYS
from disaggregate import MODEL_PATH, Disaggregator, train_default
from meter_sim import simulate_home

HISTORY_DAYS = 30

SCENARIOS = {
    "Typical summer home": [],
    "Laundry always in the evening peak": ["evening_laundry"],
    "Geyser left on by mistake": ["geyser_left_on"],
    "Fridge with a worn door seal": ["fridge_seal"],
    "Water pump running dry": ["pump_dry_run"],
    "All of the above": ["evening_laundry", "geyser_left_on", "fridge_seal", "pump_dry_run"],
}

# Names used in the setup screen -> appliance keys used by the detector.
_SETUP_NAMES = {
    "air conditioner": "ac", "ac": "ac",
    "refrigerator": "fridge", "fridge": "fridge",
    "washing machine": "washing_machine",
    "water heater": "geyser", "geyser": "geyser",
    "microwave": "microwave",
    "water pump": "water_pump", "pump": "water_pump", "motor": "water_pump",
}


def owned_appliances(home_details: Optional[dict]) -> Tuple[str, ...]:
    """
    Appliance keys this household has.  If setup listed none that the detector
    knows, the full demo set is used so the app is never empty.
    """
    found = []
    for item in (home_details or {}).get("appliances", []) or []:
        key = _SETUP_NAMES.get(str(item.get("name", "")).strip().lower())
        if key and key not in found:
            found.append(key)
    return tuple(k for k in APPLIANCE_KEYS if k in found) if found else tuple(APPLIANCE_KEYS)


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


@st.cache_data(show_spinner=False)
def detected_history(date: str, scenario: str, owned: Tuple[str, ...],
                     days: int = HISTORY_DAYS) -> pd.DataFrame:
    """`days` of 1-minute detection results, ending at the end of `date`."""
    start = pd.Timestamp(date) - pd.Timedelta(days=days - 1)
    df = simulate_home(days=days, seed=777, start=str(start.date()),
                       faults=SCENARIOS[scenario], include=owned)
    pred = get_model().predict(df[["datetime", "mains_kw"]])
    return restrict_to_owned(pred, owned)


def last_days(pred: pd.DataFrame, days: int) -> pd.DataFrame:
    return pred.tail(days * 1440).reset_index(drop=True)


def today_str() -> str:
    return str(pd.Timestamp.now().date())


SOURCE_NOTE = ("Simulated main-meter feed. No physical meter is connected, so appliance "
               "figures come from a simulated 1-minute meter signal. The detection model "
               "reads only the meter total.")
