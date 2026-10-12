"""
The household's clock
=====================
The server runs on UTC; a household lives in its own time zone.  Every date
and time the app shows (today's forecast, the calendar, activity times) comes
from local_now(), which uses the time zone the user confirmed during setup.
"""

from zoneinfo import ZoneInfo, available_timezones

import pandas as pd
import streamlit as st

DEFAULT_TZ = "Asia/Kolkata"
COMMON = ["Asia/Kolkata", "Asia/Dubai", "Asia/Singapore", "Asia/Tokyo", "Europe/London", "Europe/Berlin",
          "America/New_York", "America/Chicago", "America/Los_Angeles", "Australia/Sydney", "UTC"]
_ALIASES = {"Asia/Calcutta": "Asia/Kolkata"}


def valid(name) -> bool:
    try:
        return bool(name) and str(name) in available_timezones()
    except Exception:
        return False


def browser_tz() -> str:
    """The time zone the visitor's browser reports, or '' if it is not available."""
    try:
        name = st.context.timezone
    except Exception:
        return ""
    name = _ALIASES.get(name, name)
    return name if valid(name) else ""


def user_tz() -> str:
    try:
        name = st.session_state.get("tz")
    except Exception:
        name = None
    return name if valid(name) else DEFAULT_TZ


def set_tz(name: str) -> None:
    if valid(name):
        st.session_state["tz"] = name


def local_now() -> pd.Timestamp:
    """The current time on the household's wall clock (no tz attached, like the meter data)."""
    return pd.Timestamp.now(tz=ZoneInfo(user_tz())).tz_localize(None)


def to_local(stamp_utc) -> pd.Timestamp:
    """A UTC time from the database, on the household's wall clock."""
    return pd.Timestamp(stamp_utc, tz="UTC").tz_convert(ZoneInfo(user_tz())).tz_localize(None)


def options(current: str = "") -> list:
    seen = []
    for name in [current, browser_tz(), *COMMON]:
        if valid(name) and name not in seen:
            seen.append(name)
    return seen
