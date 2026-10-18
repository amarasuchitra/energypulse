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


def home_spec(seed: int, days: int = 60) -> dict:
    """The random choices that define one simulated home (same draws, same order as before)."""
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
    return dict(rng=rng, scenario=scenario, owned=owned, usage=usage, people=people, area=area, city=city,
                start=start, end=end, full=full, days=days,
                temps=temperatures(str(end.date()), full, days))


def _simulate(spec: dict) -> pd.DataFrame:
    name, _, _ = split_scenario(spec["scenario"])
    rng = spec["rng"]
    return simulate_home(days=spec["days"], seed=int(spec["seed"]), start=str(spec["start"].date()),
                         ratings={**random_ratings(rng), **scenario_ratings(spec["scenario"])},
                         faults=SCENARIOS[name], include=spec["owned"], usage=spec["usage"],
                         people=spec["people"], temps=spec["temps"].tolist(), area_sqft=spec["area"])


def simulated_home(seed: int, days: int = 60) -> pd.DataFrame:
    """One random simulated home: datetime + Global_active_power (hourly mean kW)."""
    spec = home_spec(seed, days)
    spec["seed"] = seed
    return minute_to_hourly(_simulate(spec))


CONTEXT_FEATURES = ["temp_next", "temp_prev_day", "cooling_deg", "people", "area_sqft", "has_ac",
                    "has_geyser", "n_appliances"]


def context_features(index: pd.DatetimeIndex, daily_temps: pd.Series, people: float, area: float,
                     owned) -> pd.DataFrame:
    """
    What the app knows besides the meter, for the hour after each row: that day's
    forecast high temperature (simulated weather), the day before's, and the
    household's own setup (people, home size, which appliances it has).
    All of it is known before the hour starts.
    """
    temps = pd.Series(np.asarray(daily_temps, float), index=pd.to_datetime(daily_temps.index).normalize())
    target_day = (index + pd.Timedelta(hours=1)).normalize()
    t_next = temps.reindex(target_day).to_numpy()
    t_prev = temps.reindex(target_day - pd.Timedelta(days=1)).to_numpy()
    owned = set(owned or ())
    has_ac = float("ac" in owned)
    return pd.DataFrame({
        "temp_next": t_next,
        "temp_prev_day": t_prev,
        "cooling_deg": np.maximum(t_next - 24.0, 0.0) * has_ac,
        "people": float(people),
        "area_sqft": float(area),
        "has_ac": has_ac,
        "has_geyser": float("geyser" in owned),
        "n_appliances": float(len(owned)),
    }, index=index)


def simulated_home_with_context(seed: int, days: int = 60):
    """(hourly frame, spec) for one home, the spec carrying what context_features needs."""
    spec = home_spec(seed, days)
    spec["seed"] = seed
    return minute_to_hourly(_simulate(spec)), spec


def minute_to_hourly(minute_df: pd.DataFrame, column: str = "mains_kw") -> pd.DataFrame:
    """1-minute meter readings -> hourly mean kW (hours with under 45 readings are dropped)."""
    s = minute_df.set_index(pd.to_datetime(minute_df["datetime"]))[column].astype(float)
    g = s.resample("h").agg(["mean", "count"])
    out = g["mean"].where(g["count"] >= 45).rename(TARGET).to_frame().reset_index()
    return out.rename(columns={out.columns[0]: "datetime"})


def power_features(columns) -> List[str]:
    return [c for c in columns if not any(c.startswith(m) for m in MEASUREMENTS)]


def build(seeds: Iterable[int], days: int = 60, context: bool = False
          ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, y, home_id) for the given homes, power-only features (+ context), warm-up week removed."""
    Xs, ys, ids = [], [], []
    for seed in seeds:
        hourly, spec = simulated_home_with_context(seed, days)
        X, y, h = hourly_training_set(hourly)
        X = X[power_features(X.columns)]
        if context:
            X = pd.concat([X, context_features(X.index, spec["temps"], spec["people"], spec["area"],
                                               spec["owned"])], axis=1)
        X, y = X.iloc[WARMUP_HOURS:], y.iloc[WARMUP_HOURS:]
        Xs.append(X)
        ys.append(y)
        ids.append(pd.Series(seed, index=X.index))
    return (pd.concat(Xs).reset_index(drop=True), pd.concat(ys).reset_index(drop=True),
            pd.concat(ids).reset_index(drop=True))
