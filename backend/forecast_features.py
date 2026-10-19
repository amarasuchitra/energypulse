"""
Leak-free features for EnergyPulse forecasts.

Shared by training (model.py) and the live app (forecast.py), so the model is
always given exactly the inputs it was trained on.

Next hour
---------
Row t holds only what is known at the end of hour t: the power readings up to
t, the other meter readings up to t, and the calendar position of hour t+1.
The target is the power in hour t+1.  Nothing from hour t+1 is used.

Next 30 days
------------
For an origin day, features come from the last 90 days of daily energy only
(the replayed sample keeps 90 days), plus the weekday of each target day and,
for the household-specific model, its place in the year.
"""
from typing import Optional

import numpy as np
import pandas as pd

TARGET = "Global_active_power"
MEASUREMENTS = ["Global_reactive_power", "Voltage", "Global_intensity",
                "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
HOURLY_HISTORY = 170          # hours of history the next-hour features need
LSTM_WINDOW = 48              # hours the LSTM looks back
LSTM_INPUTS = [TARGET] + MEASUREMENTS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
DAILY_HISTORY = 90            # days of history the 30-day features use
DAILY_HORIZON = 62           # far enough to cover all of next calendar month


# ----------------------------------------------------------------- hourly
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per clock hour, missing hours as NaN, indexed by datetime."""
    cols = [c for c in [TARGET] + MEASUREMENTS if c in df.columns]
    d = df[["datetime"] + cols].copy()
    d["datetime"] = pd.to_datetime(d["datetime"]).dt.floor("h")
    for c in cols:              # text numbers become numbers; anything else, and infinities, become unknown
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    d = d.groupby("datetime", sort=True)[cols].mean()
    if d.empty:
        return d
    full = pd.date_range(d.index.min(), d.index.max(), freq="h")
    d = d.reindex(full)
    d.index.name = "datetime"
    for c in MEASUREMENTS:
        if c not in d.columns:
            d[c] = np.nan
    return d


def hourly_features(h: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Features for every row of an hourly frame (see module docstring)."""
    y = h[TARGET]
    cols = {}
    for k in range(24):
        cols[f"lag_{k}"] = y.shift(k)
    for k in (47, 71, 95, 119, 143, 167):
        cols[f"lag_{k}"] = y.shift(k)
    same = []
    for d in range(1, 8):
        cols[f"same_hour_d{d}"] = y.shift(24 * d - horizon)
        same.append(cols[f"same_hour_d{d}"])
    cols["same_hour_mean7"] = pd.concat(same, axis=1).mean(axis=1)
    for w in (3, 6, 12, 24, 72, 168):
        r = y.rolling(w, min_periods=max(1, w // 2))
        cols[f"mean_{w}"] = r.mean()
        cols[f"std_{w}"] = r.std()
        if w <= 24:
            cols[f"max_{w}"] = r.max()
            cols[f"min_{w}"] = r.min()
    cols["ewm_3"] = y.ewm(halflife=3, ignore_na=True).mean()
    cols["ewm_12"] = y.ewm(halflife=12, ignore_na=True).mean()
    cols["diff_1"] = y - y.shift(1)
    cols["diff_24"] = y - y.shift(24)
    for c in MEASUREMENTS:
        s = h[c] if c in h.columns else pd.Series(np.nan, index=h.index)
        cols[c] = s
        cols[c + "_lag1"] = s.shift(1)
        cols[c + "_mean24"] = s.rolling(24, min_periods=12).mean()
    when = h.index + pd.Timedelta(hours=horizon)
    cols["hour"] = pd.Series(when.hour, index=h.index)
    cols["dow"] = pd.Series(when.dayofweek, index=h.index)
    cols["is_weekend"] = pd.Series((when.dayofweek >= 5).astype(int), index=h.index)
    cols["hour_sin"] = pd.Series(np.sin(2 * np.pi * when.hour / 24), index=h.index)
    cols["hour_cos"] = pd.Series(np.cos(2 * np.pi * when.hour / 24), index=h.index)
    cols["dow_sin"] = pd.Series(np.sin(2 * np.pi * when.dayofweek / 7), index=h.index)
    cols["dow_cos"] = pd.Series(np.cos(2 * np.pi * when.dayofweek / 7), index=h.index)
    return pd.DataFrame(cols, index=h.index).astype("float64")


def hourly_training_set(df: pd.DataFrame, horizon: int = 1):
    """(X, y, index) with rows where both the current reading and target exist."""
    h = to_hourly(df)
    X = hourly_features(h, horizon)
    y = h[TARGET].shift(-horizon)
    keep = y.notna() & X["lag_0"].notna()
    return X[keep], y[keep], h


def lstm_frame(h: pd.DataFrame) -> pd.DataFrame:
    """Per-hour LSTM inputs (calendar of the hour itself), small gaps filled."""
    f = h.reindex(columns=[TARGET] + MEASUREMENTS).copy()
    f = f.ffill(limit=3)
    idx = f.index
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return f[LSTM_INPUTS]


# ----------------------------------------------------------------- daily
def daily_energy(h: pd.DataFrame) -> pd.Series:
    """kWh per day; days with fewer than 18 readings are scaled up, under 18 dropped."""
    g = h[TARGET].resample("D").agg(["sum", "count"])
    ok = g["count"] >= 18
    s = (g["sum"] * 24 / g["count"]).where(ok)
    return s.interpolate(limit=3)


def daily_features(d: pd.Series, origin: int, seasonal: bool = True,
                   horizons=range(1, DAILY_HORIZON + 1),
                   season_dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    """
    Features for forecasting days origin+1 .. origin+30 from the 90 days up to origin.
    `season_dates` lets the replayed sample use its real (unshifted) dates
    for the time-of-year features.
    """
    hist = d.iloc[max(0, origin - DAILY_HISTORY + 1):origin + 1]
    base = {}
    for w in (7, 14, 28, 56, 90):
        base[f"m{w}"] = hist.iloc[-w:].mean()
    base["sd28"] = hist.iloc[-28:].std()
    base["last"] = hist.iloc[-1]
    base["trend"] = base["m7"] - base["m28"]
    base["trend2"] = base["m28"] - base["m90"]
    origin_day = d.index[origin]
    season_origin = season_dates[origin] if season_dates is not None else origin_day
    rows = []
    for h in horizons:
        t = origin_day + pd.Timedelta(days=h)
        r = dict(base)
        r["h"] = h
        r["dow"] = t.dayofweek
        r["wkend"] = int(t.dayofweek >= 5)
        same = [hist.get(t - pd.Timedelta(days=7 * k)) for k in range(1, 13)]
        same = [v for v in same if v is not None and not pd.isna(v)]
        r["same_dow_mean"] = float(np.mean(same)) if same else np.nan
        r["dow_ratio"] = (r["same_dow_mean"] / base["m90"]) if same and base["m90"] else np.nan
        if seasonal:
            st = season_origin + pd.Timedelta(days=h)
            r["doy_sin"] = np.sin(2 * np.pi * st.dayofyear / 365.25)
            r["doy_cos"] = np.cos(2 * np.pi * st.dayofyear / 365.25)
        rows.append(r)
    return pd.DataFrame(rows).astype("float64")


def daily_training_set(d: pd.Series, origins, seasonal: bool = True) -> pd.DataFrame:
    parts = []
    for o in origins:
        F = daily_features(d, o, seasonal)
        F["y"] = [d.iloc[o + h] if o + h < len(d) else np.nan for h in range(1, DAILY_HORIZON + 1)]
        F["origin"] = o
        parts.append(F)
    return pd.concat(parts, ignore_index=True)


def hourly_shape(h: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    """Share of each day's energy in each hour, by weekday, from the last `days` days."""
    y = h[TARGET].iloc[-days * 24:]
    frame = pd.DataFrame({"kw": y.values, "dow": y.index.dayofweek, "hour": y.index.hour})
    prof = frame.groupby(["dow", "hour"])["kw"].mean().unstack().reindex(index=range(7), columns=range(24))
    overall = frame.groupby("hour")["kw"].mean().reindex(range(24))
    prof = prof.apply(lambda row: row.fillna(overall), axis=1).fillna(overall.mean())
    return prof.div(prof.sum(axis=1), axis=0)
