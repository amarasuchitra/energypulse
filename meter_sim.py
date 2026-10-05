"""
Main-meter signal simulator (1-minute resolution)
=================================================
Builds what a single main meter would record for a home, by adding together
physically-shaped power traces of individual appliances plus a background
load and meter noise.

Why it exists
-------------
1. Training / testing data for disaggregate.py when no labelled recording of
   a real home is available.  The same model can be retrained on a real
   dataset (iAWE, REDD, UK-DALE) through disaggregate.load_labelled_csv().
2. "Test mode" in the app: the user switches appliances on and off, this
   module produces the combined meter signal, and the detection model has to
   work out what is running from that total alone.

THIS IS SIMULATED DATA.  Accuracy measured on it is a best case; a real home
has more appliances and messier signals.  The app labels it as simulated.

Every home gets slightly different appliance ratings (a 1.2 kW AC in one,
1.8 kW in another) so the model cannot simply memorise one wattage.

Usage:
    from meter_sim import simulate_home
    df = simulate_home(days=7, seed=1)   # datetime, mains_kw, fridge, ac, ...
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from appliance_profiles import APPLIANCE_KEYS

MIN_PER_DAY = 1440


# --------------------------------------------------------------------------
# Single-appliance traces.  Each returns kW per minute for `n` minutes.
# --------------------------------------------------------------------------
def _fridge(n: int, rng, kw: float, duty: float = 0.40) -> np.ndarray:
    """Compressor cycling round the clock, with a short start-up surge."""
    out = np.zeros(n)
    i = int(rng.integers(0, 20))
    while i < n:
        on = max(4, int(rng.normal(16, 3)))
        off = max(3, int(on * (1 - duty) / max(duty, 0.05) * rng.normal(1.0, 0.12)))
        end = min(n, i + on)
        out[i:end] = kw * rng.normal(1.0, 0.03, end - i)
        out[i] = kw * 1.8                      # start-up surge
        i = end + off
    return out


def _resistive(n_run: int, rng, kw: float) -> np.ndarray:
    """Flat heating element (geyser)."""
    return kw * rng.normal(1.0, 0.01, n_run)


def _ac_run(n_run: int, rng, kw: float) -> np.ndarray:
    """Compressor cycles on a thermostat; the indoor fan keeps running."""
    out = np.full(n_run, 0.09 * rng.normal(1.0, 0.05))    # fan only
    i = 0
    first = True
    while i < n_run:
        on = int(rng.integers(18, 30)) if first else int(rng.integers(8, 16))
        off = int(rng.integers(3, 8))
        end = min(n_run, i + on)
        out[i:end] = kw * rng.normal(1.0, 0.04, end - i)
        out[i] = kw * 1.25
        i = end + off
        first = False
    return out


def _washer_run(n_run: int, rng, kw: float) -> np.ndarray:
    """Wash (motor pulses) -> rinse -> spin (higher draw)."""
    out = np.zeros(n_run)
    wash_end = int(n_run * 0.55)
    rinse_end = int(n_run * 0.85)
    for i in range(n_run):
        if i < wash_end:
            out[i] = kw * (0.9 if (i % 4) < 3 else 0.25)
        elif i < rinse_end:
            out[i] = kw * (0.5 if (i % 5) < 3 else 0.2)
        else:
            out[i] = kw * 1.3
    return out * rng.normal(1.0, 0.06, n_run)


def _place(day: np.ndarray, start: int, trace: np.ndarray) -> None:
    end = min(len(day), start + len(trace))
    if start < len(day):
        day[start:end] = np.maximum(day[start:end], trace[: end - start])


def _background(n: int, rng, start_minute: int = 0) -> np.ndarray:
    """Lights, fans, TV, chargers: slow daily shape plus random steps."""
    minute = (np.arange(n) + start_minute) % MIN_PER_DAY
    hour = minute / 60.0
    shape = (0.11
             + 0.10 * np.exp(-((hour - 7.5) ** 2) / 3.0)
             + 0.28 * np.exp(-((hour - 20.5) ** 2) / 5.0))
    # Things being switched on and off through the day: steps that drift back
    # to the usual level, so one day's lighting and TV use resembles the next.
    steps = np.zeros(n)
    level, i = 0.0, 0
    while i < n:
        hold = int(rng.integers(20, 180))
        level = float(np.clip(0.6 * level + rng.normal(0, 0.035), -0.06, 0.14))
        steps[i:i + hold] = level
        i += hold
    # Some days the household is simply home more: a few per cent either way.
    days = int(np.ceil((n + start_minute) / MIN_PER_DAY))
    busy = np.repeat(rng.normal(1.0, 0.05, days), MIN_PER_DAY)[start_minute:start_minute + n]
    return np.clip((shape + steps) * busy, 0.03, None)


# --------------------------------------------------------------------------
# Whole home
# --------------------------------------------------------------------------
def random_ratings(rng) -> Dict[str, float]:
    """Appliance sizes for one home."""
    return {
        "fridge": float(rng.uniform(0.10, 0.18)),
        "ac": float(rng.uniform(1.1, 1.9)),
        "geyser": float(rng.choice([1.5, 2.0, 2.0, 3.0]) * rng.uniform(0.95, 1.05)),
        "washing_machine": float(rng.uniform(0.35, 0.60)),
        "water_pump": float(rng.uniform(0.55, 0.80)),
        "microwave": float(rng.uniform(0.9, 1.3)),
    }


def simulate_home(days: int = 7, seed: int = 0, start: str = "2026-04-01",
                  ratings: Optional[Dict[str, float]] = None,
                  season: str = "summer",
                  faults: Optional[List[str]] = None,
                  noise_kw: float = 0.012,
                  include: Optional[tuple] = None,
                  usage: Optional[Dict[str, float]] = None,
                  people: float = 4.0,
                  temps: Optional[List[float]] = None,
                  area_sqft: float = 1000.0) -> pd.DataFrame:
    """
    Simulate `days` of one home at 1-minute resolution.

    usage:  how heavily the household uses each appliance, as a factor on how
            long and how often it runs (0.6 light, 1.0 typical, 1.5 heavy).
    people: number of people at home; baths, laundry, cooking and the
            background load scale with it (4 is the reference home).
    temps:  daily high in C, one per day.  The AC runs longer and more often on
            hotter days and not at all on cool ones; hot days need a little
            less water heating.  None = every day is the reference 32 C.
    area_sqft: floor area; a larger home has more lights and fans and takes
            longer to cool (1000 sq ft is the reference home).

    include: appliance keys this home owns; the rest draw nothing.  None = all.

    faults: optional list from
        "fridge_seal"    - fridge compressor runs ~75% of the time
        "geyser_left_on" - one geyser run of ~3 hours on the last day
        "pump_dry_run"   - one pump run of ~90 minutes on the last day
        "evening_laundry"- washing machine always run in the evening peak

    Returns columns: datetime, mains_kw, <one kW column per appliance>, other_kw
    """
    rng = np.random.default_rng(seed)
    faults = set(faults or [])
    ratings = ratings or random_ratings(rng)
    n = days * MIN_PER_DAY
    traces = {k: np.zeros(n) for k in APPLIANCE_KEYS}
    crowd = float(np.clip(people / 4.0, 0.4, 2.5))
    use = {k: float((usage or {}).get(k, 1.0)) for k in APPLIANCE_KEYS}
    for k in ("geyser", "washing_machine", "microwave", "water_pump"):
        use[k] *= crowd ** 0.7                               # people-driven appliances
    long = lambda minutes, k: max(1, int(round(minutes * min(use[k], 1.6))))
    often = lambda p, k: min(0.98, p * use[k])
    size = float(np.clip((area_sqft or 1000.0) / 1000.0, 0.4, 3.0))
    first_day = pd.Timestamp(start).normalize()
    from weather import cooling_need

    traces["fridge"] = _fridge(n, rng, ratings["fridge"],
                               duty=0.75 if "fridge_seal" in faults else 0.40)

    for d in range(days):
        base = d * MIN_PER_DAY
        last_day = d == days - 1
        heat = cooling_need(temps[d]) if temps is not None and d < len(temps) else 1.0
        use["geyser"] = use["geyser"] / use.get("_geyser_heat", 1.0) * (1.25 - 0.25 * heat)
        use["_geyser_heat"] = 1.25 - 0.25 * heat

        # Geyser: morning bath, sometimes an evening one
        dur = long(int(rng.integers(15, 35)), "geyser")
        if "geyser_left_on" in faults and last_day:
            dur = int(rng.integers(170, 200))
        _place(traces["geyser"], base + int(rng.normal(6.5 * 60, 25)),
               _resistive(dur, rng, ratings["geyser"]))
        if rng.random() < often(0.35, "geyser"):
            _place(traces["geyser"], base + int(rng.normal(19.5 * 60, 30)),
                   _resistive(int(rng.integers(10, 25)), rng, ratings["geyser"]))

        # Water pump: once or twice a day
        dur = long(int(rng.integers(12, 28)), "water_pump")
        if "pump_dry_run" in faults and last_day:
            dur = int(rng.integers(85, 100))
        _place(traces["water_pump"], base + int(rng.normal(7.5 * 60, 40)),
               _resistive(dur, rng, ratings["water_pump"]))
        if rng.random() < often(0.5, "water_pump"):
            _place(traces["water_pump"], base + int(rng.normal(17.5 * 60, 40)),
                   _resistive(int(rng.integers(10, 22)), rng, ratings["water_pump"]))

        # Washing machine: about every second day
        if rng.random() < often(0.55, "washing_machine") or ("evening_laundry" in faults):
            hour = 19.5 if "evening_laundry" in faults else rng.choice([9.5, 11.0, 19.0])
            _place(traces["washing_machine"], base + int(rng.normal(hour * 60, 20)),
                   _washer_run(int(rng.integers(45, 75)), rng, ratings["washing_machine"]))

        # Microwave: short bursts round meal times
        for meal_hour in (8.0, 13.2, 20.3):
            for _ in range(int(round(int(rng.integers(0, 3)) * use["microwave"]))):
                _place(traces["microwave"], base + int(rng.normal(meal_hour * 60, 25)),
                       _resistive(int(rng.integers(1, 6)), rng, ratings["microwave"]))

        # AC: afternoon (sometimes) and night, summer only
        if season == "summer":
            cool = heat * size ** 0.35                      # hotter day or bigger home: more cooling
            weathered = temps is not None
            # With real weather the day's heat decides most of it: a hot night
            # means the AC is on, and for about as long as it is hot.
            steady = lambda draw, mid: mid + 0.35 * (draw - mid) if weathered else draw
            weekend = (first_day + pd.Timedelta(days=d)).dayofweek >= 5
            p_day = (0.85 if weekend else 0.3) if weathered else 0.5
            p_night = float(np.clip(1.3 * cool - 0.15, 0.0, 0.97)) / 0.9 * 0.9 if weathered else 0.9 * min(1.1, cool * 1.15)
            if rng.random() < often(p_day, "ac") * min(1.6, cool):
                minutes = long(steady(int(rng.integers(60, 150)), 105), "ac") * (0.45 + 0.55 * cool)
                if minutes >= 20:
                    _place(traces["ac"], base + int(rng.normal(14 * 60, 30)),
                           _ac_run(int(minutes), rng, ratings["ac"]))
            if rng.random() < (min(0.98, p_night * use["ac"]) if weathered else often(0.9, "ac") * min(1.1, cool * 1.15)):
                minutes = long(steady(int(rng.integers(240, 440)), 340), "ac") * (0.45 + 0.55 * cool)
                if minutes >= 30:
                    _place(traces["ac"], base + int(rng.normal(22 * 60, 30)),
                           _ac_run(int(min(minutes, 700)), rng, ratings["ac"]))

    if include is not None:
        for key in APPLIANCE_KEYS:
            if key not in include:
                traces[key] = np.zeros(n)
    return _assemble(traces, _background(n, rng) * (0.6 + 0.4 * crowd) * size ** 0.3, rng, start, noise_kw)


def _assemble(traces: Dict[str, np.ndarray], other: np.ndarray, rng,
              start: str, noise_kw: float) -> pd.DataFrame:
    n = len(other)
    total = other + sum(traces[k] for k in APPLIANCE_KEYS)
    mains = np.clip(total + rng.normal(0, noise_kw, n), 0, None)
    df = pd.DataFrame({"datetime": pd.date_range(start, periods=n, freq="min"),
                       "mains_kw": np.round(mains, 4)})
    for k in APPLIANCE_KEYS:
        df[k] = np.round(traces[k], 4)
    df["other_kw"] = np.round(other, 4)
    return df


def simulate_from_switches(switches: Dict[str, List[Tuple[int, int]]],
                           minutes: int = MIN_PER_DAY, seed: int = 0,
                           start: str = "2026-04-01",
                           ratings: Optional[Dict[str, float]] = None,
                           noise_kw: float = 0.012) -> pd.DataFrame:
    """
    Test mode.  `switches` says when the USER turned each appliance on:
        {"geyser": [(390, 415)], "ac": [(1320, 1440)]}   # minute ranges
    The fridge always runs, whatever the user asks, because it is always-on.
    """
    rng = np.random.default_rng(seed)
    ratings = ratings or {k: v for k, v in zip(
        APPLIANCE_KEYS, (0.14, 1.5, 2.0, 0.5, 0.75, 1.1))}
    traces = {k: np.zeros(minutes) for k in APPLIANCE_KEYS}
    traces["fridge"] = _fridge(minutes, rng, ratings["fridge"])
    makers = {"ac": _ac_run, "washing_machine": _washer_run}
    for key, ranges in switches.items():
        if key not in traces or key == "fridge":
            continue
        for a, b in ranges:
            a, b = max(0, int(a)), min(minutes, int(b))
            if b > a:
                maker = makers.get(key, _resistive)
                _place(traces[key], a, maker(b - a, rng, ratings[key]))
    start_minute = pd.Timestamp(start).hour * 60 + pd.Timestamp(start).minute
    return _assemble(traces, _background(minutes, rng, start_minute), rng, start, noise_kw)


_RANDOM_RUN = {            # (max runs per day, shortest, longest, chance of an abnormally long run)
    "ac": (3, 30, 400, 0.0),
    "geyser": (3, 8, 60, 0.12),
    "washing_machine": (2, 40, 80, 0.0),
    "water_pump": (3, 8, 40, 0.12),
    "microwave": (5, 1, 8, 0.0),
}


def simulate_random_day(seed: int, start: str = "2026-04-01") -> pd.DataFrame:
    """
    One day where appliances run at arbitrary times and overlap freely, in a
    home with its own appliance sizes.  Used in training so the model does
    not lean on "geysers run at 6:30 am" and copes with several appliances
    starting together, which is what a user does in test mode.
    """
    rng = np.random.default_rng(seed)
    switches = {}
    for key, (max_runs, lo, hi, long_chance) in _RANDOM_RUN.items():
        runs = []
        for _ in range(int(rng.integers(0, max_runs + 1))):
            length = int(rng.integers(lo, hi + 1))
            if rng.random() < long_chance:
                length = int(rng.integers(90, 200))
            begin = int(rng.integers(0, MIN_PER_DAY - length))
            runs.append((begin, begin + length))
        if runs:
            switches[key] = runs
    return simulate_from_switches(switches, seed=seed, start=start, ratings=random_ratings(rng))


_LEFT_ON = {"ac": (400, 1300), "geyser": (60, 600), "washing_machine": (100, 300),
            "water_pump": (45, 400), "microwave": (15, 180)}


def simulate_left_on_day(seed: int, start: str = "2026-04-01") -> pd.DataFrame:
    """
    One day where one to three appliances are switched on and left on for far
    longer than any normal run.  In the app an appliance stays on until the
    user switches it off, so the model has to recognise this too.
    """
    rng = np.random.default_rng(seed)
    keys = list(rng.choice(list(_LEFT_ON), size=int(rng.integers(1, 4)), replace=False))
    switches = {}
    for key in keys:
        lo, hi = _LEFT_ON[key]
        length = int(rng.integers(lo, hi + 1))
        begin = int(rng.integers(0, max(1, MIN_PER_DAY - length)))
        switches[key] = [(begin, min(MIN_PER_DAY, begin + length))]
    if rng.random() < 0.5:                                   # some ordinary use alongside
        other = [k for k in _RANDOM_RUN if k not in switches]
        key = str(rng.choice(other))
        _, lo, hi, _ = _RANDOM_RUN[key]
        length = int(rng.integers(lo, hi + 1))
        begin = int(rng.integers(0, MIN_PER_DAY - length))
        switches[key] = [(begin, begin + length)]
    return simulate_from_switches(switches, seed=seed, start=start, ratings=random_ratings(rng))


def simulate_many(homes: int, days: int, seed: int = 0, **kwargs) -> pd.DataFrame:
    """Several different homes stacked, with a `home` column."""
    parts = []
    for h in range(homes):
        season = "summer" if h % 4 != 3 else "winter"
        part = simulate_home(days=days, seed=seed * 1000 + h, season=season, **kwargs)
        part.insert(0, "home", h)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    demo = simulate_home(days=2, seed=1)
    print(demo.describe().T[["mean", "max"]].round(3))
