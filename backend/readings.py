"""
Checks on meter readings before they reach the forecast or the alert check,
and how fresh each stage of the live pipeline is.

Reading policy (one row per minute, `datetime` + `mains_kw` in kW):
  missing time or value          dropped
  not a number, NaN or infinite  dropped
  negative                       dropped (a household meter here does not export power)
  above MAX_KW                   dropped as an outlier (no home connection draws this much)
  same minute twice              the first reading is kept, later copies dropped
  out of order                   sorted by time
Every dropped reading is counted in the report, so nothing disappears silently.
An hour then needs 45 good readings to be used (sim_dataset.minute_to_hourly).

Freshness: the dashboard refreshes the live tile every second, but the
forecast is made once an hour and the alert check runs once an hour; only the
reading itself is new every minute.  `freshness` reports each stage's own
time, and calls a stage stale when it is older than its limit.
"""
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

MAX_KW = 25.0             # well above any single-phase home connection
COLUMNS = ("datetime", "mains_kw")

# How old each stage may be before the dashboard says it is out of date.
STALE_AFTER = {
    "measurement": pd.Timedelta(minutes=5),
    "prediction": pd.Timedelta(hours=2),
    "optimization": pd.Timedelta(hours=26),
    "alert_check": pd.Timedelta(hours=2),
}


def clean_readings(df: Optional[pd.DataFrame], column: str = "mains_kw",
                   max_kw: float = MAX_KW) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """(clean readings sorted by time, report of what was dropped and why)."""
    report = {"received": 0, "kept": 0, "missing": 0, "not_numeric": 0, "negative": 0,
              "outlier": 0, "duplicate": 0, "reordered": 0}
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=list(df.columns) if df is not None else list(COLUMNS)), report
    if "datetime" not in df.columns or column not in df.columns:
        raise ValueError(f"Readings need 'datetime' and '{column}' columns.")
    out = df.copy()
    report["received"] = len(out)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    raw = out[column]
    value = pd.to_numeric(raw, errors="coerce").astype(float)
    missing = out["datetime"].isna() | raw.isna()
    not_numeric = ~missing & (value.isna() | ~np.isfinite(value.fillna(0.0)))
    value = value.where(np.isfinite(value), np.nan)
    negative = ~missing & ~not_numeric & (value < 0)
    outlier = ~missing & ~not_numeric & (value > max_kw)
    report["missing"] = int(missing.sum())
    report["not_numeric"] = int(not_numeric.sum())
    report["negative"] = int(negative.sum())
    report["outlier"] = int(outlier.sum())
    out[column] = value
    out = out[~(missing | not_numeric | negative | outlier)]
    times = out["datetime"]
    report["reordered"] = int((times.diff() < pd.Timedelta(0)).sum())
    out = out.sort_values("datetime", kind="stable")
    minute = out["datetime"].dt.floor("min")
    dup = minute.duplicated(keep="first")
    report["duplicate"] = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    report["kept"] = len(out)
    return out, report


def freshness(stamps: Dict[str, Optional[pd.Timestamp]], now: pd.Timestamp,
              limits: Optional[Dict[str, pd.Timedelta]] = None) -> Dict[str, dict]:
    """
    For each stage (measurement, prediction, optimization, alert_check):
    {"at": time or None, "age_min": minutes or None, "stale": bool}.
    A stage that has never run is reported as stale with no time, never given a made-up one.
    """
    limits = limits or STALE_AFTER
    out = {}
    for stage, limit in limits.items():
        at = stamps.get(stage)
        if at is None or pd.isna(at):
            out[stage] = {"at": None, "age_min": None, "stale": True}
            continue
        at = pd.Timestamp(at)
        age = now - at
        out[stage] = {"at": at, "age_min": round(age.total_seconds() / 60.0, 1),
                      "stale": age > limit or age < -pd.Timedelta(minutes=1)}
    return out
