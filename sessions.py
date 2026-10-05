"""
Usage sessions
==============
Turns the minute-by-minute on/off output of disaggregate.py into things a
person understands:

  - sessions      "geyser ran 06:28-06:54, 0.87 kWh, Rs. 6.9"
  - daily summary energy, running time and cost per appliance per day
  - long runs     "geyser has been on for 3 h; a normal run is 10-40 min"
  - fridge duty   share of the day the compressor ran (a health signal)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from appliance_profiles import (
    ALWAYS_ON, APPLIANCE_KEYS, ApplianceProfile, DEFAULT_PROFILES,
)
from tariff import Tariff

SESSION_COLS = ["appliance", "start", "end", "span_min", "on_min", "kwh", "avg_kw", "cost_rs"]


def _runs(on: np.ndarray):
    """Start/end index pairs (end exclusive) of each True stretch."""
    padded = np.concatenate([[False], on, [False]])
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(change[::2], change[1::2]))


def extract_sessions(pred: pd.DataFrame, tariff: Optional[Tariff] = None,
                     profiles: Optional[Dict[str, ApplianceProfile]] = None) -> pd.DataFrame:
    """
    One row per use of an appliance.  Short thermostat pauses (an AC
    compressor resting for a few minutes) stay inside the same session.
    Always-on appliances have no sessions; see fridge_duty().
    """
    tariff = tariff or Tariff()
    profiles = profiles or DEFAULT_PROFILES
    dt = pd.to_datetime(pred["datetime"]).to_numpy()
    minute_rates = tariff.minute_rates()
    minute_of_day = (pd.to_datetime(pred["datetime"]).dt.hour * 60
                     + pd.to_datetime(pred["datetime"]).dt.minute).to_numpy()
    rows = []
    for key in APPLIANCE_KEYS:
        prof = profiles[key]
        if prof.category == ALWAYS_ON:
            continue
        on = pred[f"{key}_on"].to_numpy(dtype=bool)
        kw = pred[f"{key}_kw"].to_numpy(dtype=float)
        merged = []
        for a, b in _runs(on):
            if merged and a - merged[-1][1] <= prof.merge_gap_min:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        # A blip far shorter than any real run is a detection error.
        min_len = max(1, min(5, prof.typical_run_min[0] // 2))
        for a, b in merged:
            on_min = int(on[a:b].sum())
            if on_min < min_len:
                continue
            kwh = float(kw[a:b].sum() / 60.0)
            cost = float((kw[a:b] / 60.0 * minute_rates[minute_of_day[a:b]]).sum())
            rows.append({
                "appliance": key,
                "start": pd.Timestamp(dt[a]),
                "end": pd.Timestamp(dt[b - 1]) + pd.Timedelta(minutes=1),
                "span_min": int(b - a),
                "on_min": on_min,
                "kwh": round(kwh, 3),
                "avg_kw": round(kwh / (on_min / 60.0), 3) if on_min else 0.0,
                "cost_rs": round(cost, 2),
            })
    out = pd.DataFrame(rows, columns=SESSION_COLS)
    return out.sort_values("start").reset_index(drop=True)


def daily_summary(pred: pd.DataFrame, tariff: Optional[Tariff] = None) -> pd.DataFrame:
    """Per day and appliance: kwh, on_min, cost_rs.  Includes 'other'."""
    tariff = tariff or Tariff()
    dts = pd.to_datetime(pred["datetime"])
    day = dts.dt.date
    rate = tariff.minute_rates()[(dts.dt.hour * 60 + dts.dt.minute).to_numpy()]
    rows = []
    for key in APPLIANCE_KEYS + ["other"]:
        kw = pred[f"{key}_kw"].to_numpy(dtype=float)
        on = pred[f"{key}_on"].to_numpy(dtype=bool) if key != "other" else kw > 0
        part = pd.DataFrame({"date": day, "kwh": kw / 60.0, "on_min": on.astype(int),
                             "cost_rs": kw / 60.0 * rate})
        agg = part.groupby("date", as_index=False).sum()
        agg.insert(1, "appliance", key)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True).round({"kwh": 3, "cost_rs": 2})


def totals(pred: pd.DataFrame, tariff: Optional[Tariff] = None) -> pd.DataFrame:
    """Whole period per appliance, biggest consumer first, with % share."""
    daily = daily_summary(pred, tariff)
    agg = daily.groupby("appliance", as_index=False)[["kwh", "on_min", "cost_rs"]].sum()
    total = agg["kwh"].sum()
    agg["share_pct"] = (agg["kwh"] / total * 100).round(1) if total > 0 else 0.0
    agg["hours"] = (agg["on_min"] / 60.0).round(1)
    return agg.sort_values("kwh", ascending=False).reset_index(drop=True)


def fridge_duty(pred: pd.DataFrame, key: str = "fridge") -> pd.DataFrame:
    """Share of each day the compressor was running (0-1)."""
    dts = pd.to_datetime(pred["datetime"])
    part = pd.DataFrame({"date": dts.dt.date, "on": pred[f"{key}_on"].to_numpy(dtype=bool)})
    return part.groupby("date", as_index=False)["on"].mean().rename(columns={"on": "duty"})


def flag_long_runs(sessions: pd.DataFrame,
                   profiles: Optional[Dict[str, ApplianceProfile]] = None) -> List[dict]:
    """
    A run is flagged when it is longer than the appliance's sensible maximum,
    or, once the home has some history, much longer than this home's own
    usual run AND beyond the appliance's normal range.
    """
    profiles = profiles or DEFAULT_PROFILES
    flags = []
    for key, group in sessions.groupby("appliance"):
        prof = profiles[key]
        usual = float(group["span_min"].median()) if len(group) >= 5 else None
        for _, s in group.iterrows():
            too_long = s["span_min"] > prof.max_run_min
            unusual = (usual is not None and s["span_min"] > 2.0 * usual
                       and s["span_min"] > 1.25 * prof.typical_run_min[1])
            if too_long or unusual:
                flags.append({
                    "appliance": key,
                    "start": s["start"],
                    "minutes": int(s["span_min"]),
                    "normal_range_min": prof.typical_run_min,
                    "usual_min": round(usual) if usual else None,
                    "kwh": s["kwh"],
                    "cost_rs": s["cost_rs"],
                    "message": (f"{prof.name} ran for {_fmt_minutes(s['span_min'])} "
                                f"from {s['start']:%d %b %H:%M}. A normal run is "
                                f"{prof.typical_run_min[0]}-{prof.typical_run_min[1]} min. "
                                f"That run cost Rs. {s['cost_rs']:.0f}."),
                })
    return sorted(flags, key=lambda f: f["cost_rs"], reverse=True)


def _fmt_minutes(minutes: float) -> str:
    minutes = int(round(minutes))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"
