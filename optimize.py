"""
PART 5 — Optimization Engine
==============================
Rule-based optimization module (no ML needed). Provides:

  1. Anomaly/spike detection:
     Flags any hour where usage > mean + 2*std of trailing 7-day rolling
     average for that hour, and identifies which appliance category is
     most likely responsible.

  2. Human-readable tips:
     Generates actionable tips from flagged anomalies.

  3. Calendar-aware tip generator:
     From the next-month forecast, identifies 3-5 highest-cost days and
     suggests planning heavy appliance use around lowest-cost days.

  4. What-if simulator:
     Given a percentage reduction target, recomputes daily/weekly/monthly
     cost as if usage were reduced by that percentage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Callable


def detect_anomalies(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """
    Flag any hour where usage exceeds mean + 2*std of the trailing 7-day
    rolling average for that specific hour-of-day.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with datetime, Global_active_power, and
        sub_metering columns.
    window_days : int
        Number of trailing days for rolling statistics (default 7).

    Returns
    -------
    pd.DataFrame with anomalous rows plus columns:
      - rolling_mean: rolling average for this hour-of-day
      - rolling_std: rolling std for this hour-of-day
      - threshold: rolling_mean + 2 * rolling_std
      - deviation: how far above threshold
      - top_appliance: which appliance deviates most from its average
    """
    df = df.copy()
    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
    df["date"] = pd.to_datetime(df["datetime"]).dt.date

    # Compute hourly totals in Wh for appliance comparison
    df["total_wh"] = df["Global_active_power"] * 1000.0
    df["kitchen_wh"] = df["Sub_metering_1"]
    df["laundry_wh"] = df["Sub_metering_2"]
    df["wh_ac_wh"] = df["Sub_metering_3"]
    df["other_wh"] = (df["total_wh"] - df["kitchen_wh"] - df["laundry_wh"] - df["wh_ac_wh"]).clip(lower=0)

    # For each hour-of-day, compute rolling statistics over trailing N days
    anomalies = []
    unique_dates = sorted(df["date"].unique())

    for i, current_date in enumerate(unique_dates):
        if i < window_days:
            continue  # need enough history

        # Trailing window dates
        trail_dates = unique_dates[max(0, i - window_days):i]
        trail_data = df[df["date"].isin(trail_dates)]

        # Current day data
        current_data = df[df["date"] == current_date]

        for _, row in current_data.iterrows():
            h = row["hour"]

            # Rolling stats for this hour-of-day across the trailing window
            hour_trail = trail_data[trail_data["hour"] == h]["Global_active_power"]
            if len(hour_trail) < 3:
                continue

            r_mean = hour_trail.mean()
            r_std = hour_trail.std()
            threshold = r_mean + 2 * r_std

            if row["Global_active_power"] > threshold:
                # Identify which appliance deviates most from its own recent avg
                app_cols = {
                    "Kitchen": ("kitchen_wh", "Sub_metering_1"),
                    "Laundry Room": ("laundry_wh", "Sub_metering_2"),
                    "Water Heater & AC": ("wh_ac_wh", "Sub_metering_3"),
                }
                max_dev = -1
                top_app = "Other"
                for app_name, (wh_col, raw_col) in app_cols.items():
                    app_trail = trail_data[trail_data["hour"] == h][raw_col].mean()
                    app_current = row[raw_col]
                    dev = abs(app_current - app_trail)
                    if dev > max_dev:
                        max_dev = dev
                        top_app = app_name

                anomalies.append({
                    "datetime": row["datetime"],
                    "hour": h,
                    "Global_active_power": row["Global_active_power"],
                    "rolling_mean": round(r_mean, 4),
                    "rolling_std": round(r_std, 4),
                    "threshold": round(threshold, 4),
                    "deviation": round(row["Global_active_power"] - threshold, 4),
                    "top_appliance": top_app,
                })

    return pd.DataFrame(anomalies)


def generate_anomaly_tips(anomalies: pd.DataFrame, rate: float = 8.0,
                         translate_fn: Callable = None) -> List[str]:
    """
    Generate human-readable tips from flagged anomalies.
    
    Parameters
    ----------
    anomalies : pd.DataFrame
        Dataframe of anomalies with columns: datetime, hour, Global_active_power, 
        rolling_mean, threshold, deviation, top_appliance
    rate : float
        Tariff rate in Rs./kWh
    translate_fn : Callable
        Translation function T(key, **kwargs) that returns translated strings.
        If None, uses English fallback.
    
    Returns
    -------
    List[str] of human-readable tips
    """
    if translate_fn is None:
        # Fallback: simple dict with English strings
        translate_fn = lambda key, **kwargs: {
            "tip_no_anomalies": "No anomalies detected in the recent data. Usage is within normal patterns.",
            "tip_anomaly_usage_high": "{appliance} usage was unusually high at {period} on {date} (actual: {usage:.2f} kW vs normal: {mean:.2f} kW) — shifting this to off-peak hours could save an estimated Rs. {saving:.0f} this month.",
            "time_12am": "12:00 AM",
            "time_12pm": "12:00 PM",
            "time_period_ampm": "{h}:00 AM",
            "time_period_pmpm": "{h}:00 PM",
            "time_period_range": "{start}-{end}",
        }.get(key, key).format(**kwargs)
    
    tips = []
    if anomalies.empty:
        return [translate_fn("tip_no_anomalies")]

    for _, row in anomalies.iterrows():
        dt = row["datetime"]
        hour = row["hour"]
        appliance = row["top_appliance"]
        usage = row["Global_active_power"]
        mean = row["rolling_mean"]

        # Estimate excess energy cost per month (anomaly happens ~4x/month)
        excess_kw = usage - mean
        excess_kwh_per_occurrence = excess_kw * 1  # 1 hour
        monthly_saving = excess_kwh_per_occurrence * rate * 4  # ~4 occurrences/month

        # Format hour range using translate_fn
        def _fmt_hour(h):
            if h == 0 or h == 24:
                return translate_fn("time_12am")
            elif h == 12:
                return translate_fn("time_12pm")
            elif h < 12:
                return translate_fn("time_period_ampm", h=h)
            else:
                return translate_fn("time_period_pmpm", h=h - 12)
        
        period = translate_fn("time_period_range", 
                             start=_fmt_hour(hour), 
                             end=_fmt_hour(hour + 1))

        tip = translate_fn("tip_anomaly_usage_high",
                          appliance=appliance,
                          period=period,
                          date=dt.strftime('%Y-%m-%d'),
                          usage=usage,
                          mean=mean,
                          saving=monthly_saving)
        tips.append(tip)

    return tips


def calendar_tips(forecast_df: pd.DataFrame, rate: float = 8.0,
                 translate_fn: Callable = None) -> List[str]:
    """
    From the next-month forecast, identify the 3-5 highest-projected-cost
    days and suggest the user plan heavy appliance use around the
    lowest-cost days instead.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast dataframe with day and predicted_kwh columns
    rate : float
        Tariff rate in Rs./kWh
    translate_fn : Callable
        Translation function T(key, **kwargs). If None, uses English fallback.
    
    Returns
    -------
    List[str] of calendar tips
    """
    if translate_fn is None:
        translate_fn = lambda key, **kwargs: {
            "tip_calendar_no_data": "No forecast data available for calendar tips.",
            "tip_calendar_insufficient": "Not enough forecast days for calendar tips.",
            "tip_calendar_expensive": "**Highest-cost days in the coming month:** {days}",
            "tip_calendar_cheap": "**Best days for heavy appliance use:** {days}",
            "tip_calendar_savings": "If you shift laundry and water heating from peak-cost days to low-cost days, you could save approximately Rs. {daily:.0f} per day, or Rs. {monthly:.0f} per month.",
        }.get(key, key).format(**kwargs)
    
    if forecast_df.empty:
        return [translate_fn("tip_calendar_no_data")]

    # Aggregate by day
    daily = forecast_df.groupby("day").agg(
        total_kwh=("predicted_kwh", "sum"),
    ).reset_index()
    daily["cost"] = daily["total_kwh"] * rate
    daily = daily.sort_values("day")

    if len(daily) < 5:
        return [translate_fn("tip_calendar_insufficient")]

    # Top 5 most expensive days
    expensive = daily.nlargest(5, "cost")
    # Top 5 cheapest days
    cheapest = daily.nsmallest(5, "cost")

    tips = []
    expensive_str = ", ".join([
        f"{row['day']} (Rs. {row['cost']:.0f})"
        for _, row in expensive.iterrows()
    ])
    tips.append(translate_fn("tip_calendar_expensive", days=expensive_str))
    
    cheap_str = ", ".join([
        f"{row['day']} (Rs. {row['cost']:.0f})"
        for _, row in cheapest.iterrows()
    ])
    tips.append(translate_fn("tip_calendar_cheap", days=cheap_str))

    # Savings estimate
    avg_expensive = expensive["cost"].mean()
    avg_cheap = cheapest["cost"].mean()
    saving_per_day = avg_expensive - avg_cheap

    tips.append(translate_fn("tip_calendar_savings", 
                            daily=saving_per_day, 
                            monthly=saving_per_day * 15))

    return tips


def whatif_simulator(df: pd.DataFrame, reduction_pct: float,
                     rate: float = 8.0) -> dict:
    """
    Recompute daily, weekly, and monthly costs as if usage were reduced
    by the given percentage. Powers the dashboard slider.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with datetime and Global_active_power.
    reduction_pct : float
        Percentage reduction target (0-100).
    rate : float
        Tariff rate in Rs./kWh.

    Returns
    -------
    dict with keys:
      original_daily_kwh, reduced_daily_kwh,
      original_weekly_cost, reduced_weekly_cost,
      original_monthly_cost, reduced_monthly_cost,
      monthly_saving, weekly_saving, daily_saving
    """
    # Daily average
    if df.empty or "datetime" not in df.columns or "Global_active_power" not in df.columns:
        return {
            "original_daily_kwh": 0, "reduced_daily_kwh": 0,
            "original_weekly_cost": 0, "reduced_weekly_cost": 0,
            "original_monthly_cost": 0, "reduced_monthly_cost": 0,
            "monthly_saving": 0, "weekly_saving": 0, "daily_saving": 0,
        }
    daily_totals = df.set_index("datetime")["Global_active_power"].resample("D").sum()
    orig_daily = daily_totals.mean()
    if pd.isna(orig_daily):
        orig_daily = 0.0
    red_daily = orig_daily * (1 - reduction_pct / 100.0)

    # Weekly = daily * 7
    orig_weekly = orig_daily * 7
    red_weekly = red_daily * 7

    # Monthly = daily * 30
    orig_monthly = orig_daily * 30
    red_monthly = red_daily * 30

    return {
        "original_daily_kwh": round(orig_daily, 2),
        "reduced_daily_kwh": round(red_daily, 2),
        "daily_saving": round((orig_daily - red_daily) * rate, 2),
        "original_weekly_cost": round(orig_weekly * rate, 2),
        "reduced_weekly_cost": round(red_weekly * rate, 2),
        "weekly_saving": round((orig_weekly - red_weekly) * rate, 2),
        "original_monthly_cost": round(orig_monthly * rate, 2),
        "reduced_monthly_cost": round(red_monthly * rate, 2),
        "monthly_saving": round((orig_monthly - red_monthly) * rate, 2),
    }
