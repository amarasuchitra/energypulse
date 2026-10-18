"""
Tariff
======
Price of one unit (kWh) at a given time of day.

Two modes:
  flat          - the same Rs./kWh all day.  Moving an appliance to another
                  hour saves NOTHING, and the scheduler says so.
  time-of-day   - cheaper in solar hours, costlier in the evening peak.

The default time-of-day pattern follows the shape of India's Electricity
(Rights of Consumers) Amendment Rules, 2023: a lower rate in daytime "solar
hours" and a higher rate in evening peak hours.  The exact hours and
percentages are set by each state's regulator, so they are settings here,
not constants.  Check the user's own bill before quoting savings.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd

# (start_hour, end_hour, multiplier, label); hours not listed use 1.0
DEFAULT_TOD_PERIODS = [
    (9, 17, 0.80, "Solar hours"),
    (18, 22, 1.20, "Evening peak"),
]


@dataclass
class Tariff:
    rate: float = 8.0                       # normal Rs./kWh
    tod_enabled: bool = True
    periods: List[Tuple[int, int, float, str]] = field(
        default_factory=lambda: list(DEFAULT_TOD_PERIODS))

    def multiplier_at(self, hour: float) -> float:
        if self.tod_enabled:
            for start, end, mult, _ in self.periods:
                if start <= hour % 24 < end:
                    return mult
        return 1.0

    def rate_at(self, hour: float) -> float:
        return self.rate * self.multiplier_at(hour)

    def label_at(self, hour: float) -> str:
        if self.tod_enabled:
            for start, end, _, label in self.periods:
                if start <= hour % 24 < end:
                    return label
        return "Normal"

    def minute_rates(self) -> np.ndarray:
        """Rs./kWh for each of the 1440 minutes of a day (computed once per tariff)."""
        return _minute_rates(float(self.rate), bool(self.tod_enabled),
                             tuple(tuple(p) for p in self.periods))

    def cost(self, datetimes, kw) -> float:
        """Cost in Rs. of a per-minute kW series."""
        dt = pd.to_datetime(pd.Series(datetimes))
        minute_of_day = (dt.dt.hour * 60 + dt.dt.minute).to_numpy()
        return float((np.asarray(kw, dtype=float) / 60.0
                      * self.minute_rates()[minute_of_day]).sum())

    def run_cost(self, start_minute: int, duration_min: int, avg_kw: float) -> float:
        """Cost of a steady run starting at `start_minute` of the day."""
        idx = (start_minute + np.arange(duration_min)) % 1440
        return float((avg_kw / 60.0 * self.minute_rates()[idx]).sum())


@lru_cache(maxsize=64)
def _minute_rates(rate: float, tod_enabled: bool, periods: tuple) -> np.ndarray:
    hours = np.arange(1440) / 60.0
    mult = np.ones(1440)
    if tod_enabled:
        # Earlier periods win where periods overlap, as in Tariff.multiplier_at.
        for start, end, m, _ in reversed(periods):
            mult[(hours >= start) & (hours < end)] = m
    out = rate * mult
    out.setflags(write=False)
    return out
