"""
Simulated weather
=================
The app has no weather feed, so the daily high temperature is simulated: a
seasonal curve for the household's city plus day-to-day variation that is the
same every time it is asked for (it depends only on the city and the date).

It matters because air-conditioner use follows temperature.  The simulator
uses it to decide how long the AC runs; the daily forecast uses it to predict
that.  The "forecast" for today is the simulated high with a small error
added, the way a real next-day forecast is close but not exact.
"""

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

# (yearly mean of the daily high in C, summer-to-winter swing in C)
CLIMATE = {
    "bengaluru": (29.0, 3.5), "bangalore": (29.0, 3.5), "mysuru": (30.0, 3.5), "mysore": (30.0, 3.5),
    "delhi": (31.5, 9.5), "new delhi": (31.5, 9.5), "mumbai": (31.5, 2.5), "pune": (31.5, 4.5),
    "chennai": (33.5, 4.0), "hyderabad": (32.5, 5.0), "kolkata": (31.5, 5.0), "ahmedabad": (34.0, 6.5),
    "jaipur": (32.5, 8.5), "lucknow": (32.0, 8.5), "kochi": (31.0, 1.5), "visakhapatnam": (31.0, 3.0),
}
DEFAULT_CLIMATE = (31.5, 5.0)
COMFORT_C = 24.0        # below this the AC is not needed
REFERENCE_C = 32.0      # the temperature the reference home was written for


def climate(city: str):
    return CLIMATE.get(str(city or "").strip().lower(), DEFAULT_CLIMATE)


def _noise(city: str, day: pd.Timestamp) -> float:
    seed = int(hashlib.md5(f"{str(city).strip().lower()}|{day.date()}".encode()).hexdigest()[:8], 16)
    return float(np.random.default_rng(seed).normal(0.0, 1.0))


def daily_high(dates: Iterable, city: str = "") -> pd.Series:
    """Simulated daily high in C for each date.  Warm and cool spells last a few days."""
    mean, swing = climate(city)
    out = {}
    for d in pd.to_datetime(list(dates)):
        d = d.normalize()
        season = mean + swing * np.cos(2 * np.pi * (d.dayofyear - 135) / 365.25)      # hottest in mid May
        spell = sum(w * _noise(city, d - pd.Timedelta(days=i)) for i, w in enumerate((0.55, 0.3, 0.15)))
        out[d] = round(float(season + 3.6 * spell), 1)
    return pd.Series(out)


def forecast_high(day, city: str = "") -> float:
    """What a forecast issued the evening before would say: the simulated high, about 1 C off."""
    day = pd.Timestamp(day).normalize()
    return round(float(daily_high([day], city).iloc[0]) + 0.9 * _noise(city + "|forecast", day), 1)


def cooling_need(temp_c) -> float:
    """1.0 at the reference temperature, 0 at or below the comfort temperature."""
    return float(np.clip((float(temp_c) - COMFORT_C) / (REFERENCE_C - COMFORT_C), 0.0, 1.7))
