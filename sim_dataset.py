"""
Training and test data for the simulated meter's next-hour model.

The simulated meter (meter_sim.py) stands in for a real smart meter.  A model
trained on the UCI home in France does not transfer to these simulated Indian
homes (it scores worse than a same-hour average on them), so the meter gets
its own model, trained on many simulated homes and tested on DIFFERENT
simulated homes (different seeds, so different appliances, habits, sizes,
cities and seasons).

Each home: hourly average of the simulated mains (kW) over `days` days.
"""
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from appliance_profiles import APPLIANCE_KEYS
from forecast_features import MEASUREMENTS, TARGET, hourly_training_set
from meter_sim import random_ratings, simulate_home
from meter_source import SCENARIOS, scenario_ratings, split_scenario, temperatures

CITIES = ["bengaluru", "delhi", "mumbai", "chennai", "kolkata", "hyderabad"]
WARMUP_HOURS = 24 * 7          # the first week of each home only provides history

TRAIN_SEEDS = range(1000, 1080)
VALIDATION_SEEDS = range(3000, 3020)
TEST_SEEDS = range(5000, 5030)


def simulated_home(seed: int, days: int = 60) -> pd.DataFrame:
    """One random simulated home: datetime + Global_active_power (hourly mean kW)."""
    rng = np.random.default_rng(seed)
    scenario = list(SCENARIOS)[rng.integers(len(SCENARIOS))]
    owned = tuple(k for k in APPLIANCE_KEYS if k == "fridge" or rng.random() < 0.75)
    usage = {k: float(rng.choice([0.6, 1.0, 1.5])) for k in owned}
    people = float(rng.integers(1, 7))
    area = float(rng.choice([500, 800, 1200, 1800, 2500]))
    city = CITIES[rng.integers(len(CITIES))]
    end = pd.Timestamp("2026-01-15") + pd.Timedelta(days=int(rng.integers(0, 330)))
    start = end - pd.Timedelta(days=days - 1)
    full = f"{scenario}||;people={int(people)};city={city}"
    temps = temperatures(str(end.date()), full, days)
    name, _, _ = split_scenario(scenario)
    sim = simulate_home(days=days, seed=int(seed), start=str(start.date()),
                        ratings={**random_ratings(rng), **scenario_ratings(scenario)},
                        faults=SCENARIOS[name], include=owned, usage=usage, people=people,
                        temps=temps.tolist(), area_sqft=area)
    return minute_to_hourly(sim)


def minute_to_hourly(minute_df: pd.DataFrame, column: str = "mains_kw") -> pd.DataFrame:
    """1-minute meter readings -> hourly mean kW (hours with under 45 readings are dropped)."""
    s = minute_df.set_index(pd.to_datetime(minute_df["datetime"]))[column].astype(float)
    g = s.resample("h").agg(["mean", "count"])
    out = g["mean"].where(g["count"] >= 45).rename(TARGET).to_frame().reset_index()
    return out.rename(columns={out.columns[0]: "datetime"})


def power_features(columns) -> List[str]:
    return [c for c in columns if not any(c.startswith(m) for m in MEASUREMENTS)]


def build(seeds: Iterable[int], days: int = 60) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, home_id) for the given homes, power-only features, warm-up week removed."""
    Xs, ys, ids = [], [], []
    for seed in seeds:
        X, y, _ = hourly_training_set(simulated_home(seed, days))
        X, y = X.iloc[WARMUP_HOURS:], y.iloc[WARMUP_HOURS:]
        Xs.append(X[power_features(X.columns)])
        ys.append(y)
        ids.append(pd.Series(seed, index=X.index))
    return (pd.concat(Xs).reset_index(drop=True), pd.concat(ys).reset_index(drop=True),
            pd.concat(ids).reset_index(drop=True))
