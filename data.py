"""
PART 1 — Data Processing
=========================
Downloads the UCI "Individual Household Electric Power Consumption" dataset,
cleans it, engineers time-based features, remaps timestamps to current dates,
and saves cleaned_energy_data.csv.

Dataset columns:
  Date, Time, Global_active_power, Global_reactive_power, Voltage,
  Global_intensity, Sub_metering_1 (kitchen), Sub_metering_2 (laundry),
  Sub_metering_3 (water heater + AC)

Run FIRST before anything else:
    python data.py
"""

import json
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
RAW_TXT = os.path.join(DATA_DIR, "household_power_consumption.txt")
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_energy_data.csv")
SAMPLE_META_PATH = os.path.join(DATA_DIR, "sample_meta.json")

# UCI ML Repository download URL
UCI_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00235/household_power_consumption.zip"
)


def download_dataset():
    """Download the zipped dataset from UCI and extract the txt file."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_TXT):
        print("[OK] Raw data file already exists — skipping download.")
        return

    zip_path = os.path.join(DATA_DIR, "dataset.zip")
    print("[..] Downloading dataset from UCI (~20 MB) ...")
    urllib.request.urlretrieve(UCI_URL, zip_path)

    print("[..] Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    print("[OK] Downloaded and extracted.")


def load_and_clean() -> pd.DataFrame:
    """
    Load the semicolon-separated txt file.
    - Combines Date + Time into a single 'datetime' column.
    - Coerces numeric columns (the dataset marks NaN as '?').
    - Drops rows where the target (Global_active_power) is missing.
    - Forward-fills then back-fills any remaining NaN.
    """
    print("[..] Loading and cleaning ...")

    df = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values=["?"],  # dataset uses '?' for missing values
        low_memory=False,
    )

    # Combine Date and Time into a single datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df.drop(columns=["Date", "Time"], inplace=True)

    # Ensure every measurement column is truly numeric
    num_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows where the target is completely missing
    df.dropna(subset=["Global_active_power"], inplace=True)

    # Forward-fill then back-fill remaining NaN in other columns
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    print(f"[OK] Rows after cleaning: {len(df):,}")
    return df


MINUTES_PER_HOUR = 60

# UCI sub-metering columns are energy in Wh per *minute*. Once the series is
# resampled to hourly frequency the stored column must mean Wh per *hour* so
# that .sum() yields real energy. Without this conversion the sub-meters
# appear 60x too small and every appliance breakdown collapses to
# "99% other".
SUB_METER_COLS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level data to hourly frequency.

    Power columns (kW) become the hourly mean power, so kWh = kW x 1h.
    Sub-metering columns (Wh/min) are converted to Wh/hour by multiplying the
    hourly mean by 60. This reduces ~2M rows to ~34K rows - manageable for a
    Streamlit demo.
    """
    print("[..] Resampling to hourly frequency ...")
    df = df.set_index("datetime").sort_index()

    hourly = df.resample("h").agg(
        {
            "Global_active_power": "mean",
            "Global_reactive_power": "mean",
            "Voltage": "mean",
            "Global_intensity": "mean",
            "Sub_metering_1": "mean",
            "Sub_metering_2": "mean",
            "Sub_metering_3": "mean",
        }
    )
    hourly.dropna(inplace=True)
    hourly = hourly.reset_index()

    for col in SUB_METER_COLS:
        if col in hourly.columns:
            hourly[col] = hourly[col] * MINUTES_PER_HOUR

    # Sanity check: sub-meters must be a plausible share of the main meter.
    # Global_active_power is kW x 1h = kWh; sub-meters are Wh per hour.
    total_kwh = hourly["Global_active_power"].sum()
    sub_kwh = hourly[[c for c in SUB_METER_COLS if c in hourly.columns]].sum().sum() / 1000.0
    if total_kwh > 0:
        share = sub_kwh / total_kwh * 100.0
        print(f"[OK] Sub-meters account for {share:.1f}% of metered consumption")
        if share > 100:
            print("[!!] Sub-meters exceed the main meter - units look inconsistent")
        elif share < 5:
            print("[!!] Sub-meters look implausibly small - units look inconsistent")

    print(f"[OK] Hourly rows: {len(hourly):,}")
    return hourly


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features that the ML models and dashboard need:
      - hour          (0-23)
      - day_of_week   (0=Mon ... 6=Sun)
      - is_weekend    (1 if Sat/Sun, else 0)
      - day_of_month  (1-31)
      - week_of_year  (1-53)

    These are needed for:
      - Calendar view coloring (day_of_month, week_of_year)
      - Weekly/monthly aggregation (day_of_week, week_of_year)
      - ML model input (hour, day_of_week, is_weekend)
    """
    print("[..] Engineering features ...")
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    return df


def remap_to_current_dates(df: pd.DataFrame, last_n_days: int = 90,
                           anchor_end=None) -> pd.DataFrame:
    """
    Re-map the dataset's timestamps so the most recent data point falls on the
    current hour, while preserving ALL usage values and patterns.

    Keeps each row's hour-of-day, day-of-week and relative position in the
    dataset exactly as they are (preserving real usage patterns). Only the
    calendar dates are shifted.

    How it works:
      1. Takes the last `last_n_days` worth of hourly data from the dataset.
      2. Anchors the LAST row on the current hour (or on `anchor_end`).
      3. Every earlier row keeps the original spacing between readings.

    Anchoring on the end rather than on "today minus N days at midnight"
    matters: the window usually holds a fractional number of days, and
    starting from midnight used to leave the newest reading several days
    behind the clock.

    NOTE: this fabricates freshness and is only ever applied to the bundled
    sample data. The UI labels that data as a sample; user uploads keep their
    real timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a 'datetime' column (already cleaned and resampled).
    last_n_days : int
        Number of trailing days to include in the remapped window (default 90).
    anchor_end : pd.Timestamp, optional
        Timestamp to assign to the final row. Defaults to the current hour.

    Returns
    -------
    pd.DataFrame with remapped datetime column and updated time features.
    """
    print(f"[..] Remapping timestamps to current dates (last {last_n_days} days) ...")
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Keep the last N days of hourly data
    last_ts = df["datetime"].max()
    cutoff = last_ts - pd.Timedelta(days=last_n_days)
    df = df[df["datetime"] >= cutoff].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Compute the original hourly interval from the data
    if len(df) > 1:
        median_interval = df["datetime"].diff().dropna().median()
    else:
        median_interval = pd.Timedelta(hours=1)
    if not isinstance(median_interval, pd.Timedelta) or median_interval <= pd.Timedelta(0):
        median_interval = pd.Timedelta(hours=1)

    end = pd.Timestamp(anchor_end) if anchor_end is not None else pd.Timestamp.now()
    end = end.floor("h")
    start = end - (len(df) - 1) * median_interval

    df["datetime"] = pd.date_range(start=start, periods=len(df),
                                   freq=median_interval)

    # Re-extract time features from the new timestamps
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"] = df["datetime"].dt.day
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)

    lag_hours = (end - df["datetime"].max()).total_seconds() / 3600.0
    print(f"[OK] Remapped {len(df):,} rows: {df['datetime'].min()} -> {df['datetime'].max()}"
          f" (newest reading is {lag_hours:.0f}h from now)")
    return df


def shift_forecast_to_current_dates(forecast_df: pd.DataFrame, source_raw_df: pd.DataFrame,
                                    current_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift a forecast DataFrame's datetimes so "next month" lands after the
    current (remapped) measurements — exactly the rule the dashboard uses.

    The stored forecast covers the month after the raw dataset ends (Dec 2010
    vs Nov 2010). Without this shift, "next month" reads as "December 2010".
    The offset is `current_df.max - source_raw_df.max`, i.e. the same shift
    `remap_to_current_dates` applied to the real measurements.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast rows with a 'datetime' column.
    source_raw_df : pd.DataFrame
        The cleaned dataset with original (un-remapped) datetimes.
    current_df : pd.DataFrame
        The same data after remapping to current dates (or the dashboard's
        `full_data`).

    Returns
    -------
    pd.DataFrame (a copy) with shifted datetimes, or the input unchanged if
    required inputs/columns are missing.
    """
    need = ["datetime"]
    if (forecast_df is None or forecast_df.empty
            or source_raw_df is None or source_raw_df.empty
            or current_df is None or current_df.empty
            or not all(c in forecast_df.columns for c in need)
            or not all(c in source_raw_df.columns for c in need)
            or not all(c in current_df.columns for c in need)):
        return forecast_df
    out = forecast_df.copy()
    shift = pd.to_datetime(current_df["datetime"].max()) - pd.to_datetime(source_raw_df["datetime"].max())
    out["datetime"] = pd.to_datetime(out["datetime"]) + shift
    return out


def main():
    """
    Full pipeline: download -> clean -> resample -> feature engineer -> save.

    The dataset is saved with its REAL timestamps. The demo shift that makes
    these 2006-2010 readings look recent is applied at runtime by the app
    (see :func:`remap_to_current_dates`), which labels the result as sample
    data. Baking the shift into the stored file made historical records
    indistinguishable from live ones in any downstream copy of the CSV.
    """
    download_dataset()
    df = load_and_clean()
    df = resample_hourly(df)
    df = engineer_features(df)

    # Save cleaned dataset
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)

    # Record what this sample really is, so the dashboard can say so instead
    # of presenting 2006-2010 recordings as live readings.
    meta = {
        "kind": "sample",
        "source": "UCI Individual household electric power consumption",
        "source_url": "https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption",
        "original_start": str(df["datetime"].min()),
        "original_end": str(df["datetime"].max()),
        "rows": int(len(df)),
        "granularity": "hourly (aggregated from 1-minute readings)",
        "note": "Demo sample. The dashboard shifts these timestamps forward "
                "for display and labels the result as sample data.",
    }
    with open(SAMPLE_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n{'='*55}")
    print(f"  CLEANED DATASET SAVED -> {CLEAN_CSV}")
    print(f"{'='*55}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print(f"  Range   : {df['datetime'].min()} -> {df['datetime'].max()}")
    print("  Note    : real timestamps kept; the dashboard applies its")
    print("            clearly labelled demo shift at runtime.")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
