"""
PART 4 — Cost Engine
=====================
Three core functions for computing electricity costs from actual and
predicted data, plus peak/off-peak split analysis.

All functions accept a configurable tariff rate (Rs./kWh, default 8).

Functions:
  - daily_cost(df, date)           — cost for a specific day
  - weekly_cost(df, week_start)    — cost for a 7-day window with % change
  - next_month_cost(forecast_df)   — total projected cost for upcoming month

Peak/off-peak split:
  Configurable peak hours (default: 6-10 AM, 6-10 PM).
  Reports what fraction of each period's cost came from peak-hour usage.
"""

import pandas as pd
import numpy as np


# Default peak hours: 6-10 AM and 6-10 PM
DEFAULT_PEAK_MORNING = (6, 10)   # inclusive start, exclusive end
DEFAULT_PEAK_EVENING = (18, 22)  # inclusive start, exclusive end


def _is_peak_hour(hour: int, peak_morning=DEFAULT_PEAK_MORNING,
                  peak_evening=DEFAULT_PEAK_EVENING) -> bool:
    """Check if a given hour falls within peak hours."""
    m_start, m_end = peak_morning
    e_start, e_end = peak_evening
    return (m_start <= hour < m_end) or (e_start <= hour < e_end)


def _compute_cost_from_wh(total_wh: float, rate: float) -> float:
    """
    Convert energy in watt-hours to cost.
    cost = (Wh / 1000) * rate  = kWh * rate
    """
    return (total_wh / 1000.0) * rate


def daily_cost(df: pd.DataFrame, date: str, rate: float = 8.0,
               peak_morning=DEFAULT_PEAK_MORNING,
               peak_evening=DEFAULT_PEAK_EVENING) -> dict:
    """
    Compute cost for a specific day from actual readings.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with datetime, Global_active_power columns.
    date : str
        Date string (e.g. "2007-01-15").
    rate : float
        Tariff rate in Rs./kWh (default 8).
    peak_morning, peak_evening : tuple
        Peak hour ranges.

    Returns
    -------
    dict with keys:
      date, total_kwh, total_cost, peak_cost, offpeak_cost,
      peak_pct, offpeak_pct
    """
    day_mask = pd.to_datetime(df["datetime"]).dt.date == pd.to_datetime(date).date()
    day_data = df.loc[day_mask].copy()

    if day_data.empty:
        return {
            "date": date, "total_kwh": 0, "total_cost": 0,
            "peak_cost": 0, "offpeak_cost": 0,
            "peak_pct": 0, "offpeak_pct": 0,
        }

    # Global_active_power is in kW; each row = 1 hour, so kWh = kW * 1h
    day_data["kwh"] = day_data["Global_active_power"]
    day_data["hour"] = pd.to_datetime(day_data["datetime"]).dt.hour
    day_data["is_peak"] = day_data["hour"].apply(
        lambda h: _is_peak_hour(h, peak_morning, peak_evening)
    )

    total_kwh = day_data["kwh"].sum()
    total_cost = total_kwh * rate

    peak_kwh = day_data.loc[day_data["is_peak"], "kwh"].sum()
    offpeak_kwh = day_data.loc[~day_data["is_peak"], "kwh"].sum()

    peak_cost = peak_kwh * rate
    offpeak_cost = offpeak_kwh * rate

    return {
        "date": date,
        "total_kwh": round(total_kwh, 2),
        "total_cost": round(total_cost, 2),
        "peak_cost": round(peak_cost, 2),
        "offpeak_cost": round(offpeak_cost, 2),
        "peak_pct": round(peak_kwh / total_kwh * 100, 1) if total_kwh > 0 else 0,
        "offpeak_pct": round(offpeak_kwh / total_kwh * 100, 1) if total_kwh > 0 else 0,
    }


def weekly_cost(df: pd.DataFrame, week_start: str, rate: float = 8.0,
                peak_morning=DEFAULT_PEAK_MORNING,
                peak_evening=DEFAULT_PEAK_EVENING) -> dict:
    """
    Compute cost for a 7-day window starting on week_start.
    Also computes the previous 7-day window for percentage change comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    week_start : str
        Start date of the week (e.g. "2007-01-15").
    rate : float
        Tariff rate in Rs./kWh.

    Returns
    -------
    dict with keys:
      week_start, week_end, total_kwh, total_cost,
      prev_total_kwh, prev_total_cost, pct_change,
      peak_cost, offpeak_cost
    """
    ws = pd.to_datetime(week_start)
    we = ws + pd.Timedelta(days=6)

    # Current week
    dt_col = pd.to_datetime(df["datetime"])
    mask = (dt_col.dt.date >= ws.date()) & (dt_col.dt.date <= we.date())
    week_data = df.loc[mask].copy()

    total_kwh = week_data["Global_active_power"].sum() if not week_data.empty else 0
    total_cost = total_kwh * rate

    # Previous week
    pws = ws - pd.Timedelta(days=7)
    pwe = ws - pd.Timedelta(days=1)
    pmask = (dt_col.dt.date >= pws.date()) & (dt_col.dt.date <= pwe.date())
    prev_data = df.loc[pmask]

    prev_kwh = prev_data["Global_active_power"].sum() if not prev_data.empty else 0
    prev_cost = prev_kwh * rate

    # Percentage change vs previous week
    if prev_kwh > 0:
        pct_change = ((total_kwh - prev_kwh) / prev_kwh) * 100
    else:
        pct_change = 0.0

    # Peak/off-peak breakdown for current week
    if not week_data.empty:
        wd = week_data.copy()
        wd["hour"] = wd["datetime"].dt.hour
        wd["is_peak"] = wd["hour"].apply(
            lambda h: _is_peak_hour(h, peak_morning, peak_evening)
        )
        peak_cost = wd.loc[wd["is_peak"], "Global_active_power"].sum() * rate
        offpeak_cost = wd.loc[~wd["is_peak"], "Global_active_power"].sum() * rate
    else:
        peak_cost = offpeak_cost = 0

    return {
        "week_start": ws.strftime("%Y-%m-%d"),
        "week_end": we.strftime("%Y-%m-%d"),
        "total_kwh": round(total_kwh, 2),
        "total_cost": round(total_cost, 2),
        "prev_total_kwh": round(prev_kwh, 2),
        "prev_total_cost": round(prev_cost, 2),
        "pct_change": round(pct_change, 1),
        "peak_cost": round(peak_cost, 2),
        "offpeak_cost": round(offpeak_cost, 2),
    }


def next_month_cost(forecast_df: pd.DataFrame, rate: float = 8.0,
                    peak_morning=DEFAULT_PEAK_MORNING,
                    peak_evening=DEFAULT_PEAK_EVENING) -> dict:
    """
    Total projected cost for the upcoming calendar month, summed from
    the day-by-day forecast produced by model.next_month_forecast().

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of model.next_month_forecast(). Must have columns:
        datetime, predicted_kwh, day.
    rate : float
        Tariff rate in Rs./kWh.

    Returns
    -------
    dict with keys:
      month, total_kwh, total_cost, num_days, daily_avg_kwh,
      daily_avg_cost, peak_cost, offpeak_cost
    """
    if forecast_df.empty:
        return {
            "month": "N/A", "total_kwh": 0, "total_cost": 0,
            "num_days": 0, "daily_avg_kwh": 0, "daily_avg_cost": 0,
            "peak_cost": 0, "offpeak_cost": 0,
        }

    total_kwh = forecast_df["predicted_kwh"].sum()
    total_cost = total_kwh * rate
    num_days = forecast_df["day"].nunique()

    # Peak/off-peak split
    fc = forecast_df.copy()
    fc["hour"] = pd.to_datetime(fc["datetime"]).dt.hour
    fc["is_peak"] = fc["hour"].apply(
        lambda h: _is_peak_hour(h, peak_morning, peak_evening)
    )
    peak_cost = fc.loc[fc["is_peak"], "predicted_kwh"].sum() * rate
    offpeak_cost = fc.loc[~fc["is_peak"], "predicted_kwh"].sum() * rate

    month_label = pd.to_datetime(forecast_df["datetime"].iloc[0]).strftime("%B %Y")

    return {
        "month": month_label,
        "total_kwh": round(total_kwh, 2),
        "total_cost": round(total_cost, 2),
        "num_days": num_days,
        "daily_avg_kwh": round(total_kwh / num_days, 2) if num_days > 0 else 0,
        "daily_avg_cost": round(total_cost / num_days, 2) if num_days > 0 else 0,
        "peak_cost": round(peak_cost, 2),
        "offpeak_cost": round(offpeak_cost, 2),
    }
