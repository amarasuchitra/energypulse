"""
Appliance detection from the main meter (NILM / energy disaggregation)
=====================================================================
Input : ONE signal - total household power from the main meter, per minute.
Output: for every minute, which appliances are ON and how much power each
        one is drawing.

Two methods are implemented so they can be compared in the report:

1. EdgeBaseline  - classic rule-based approach (Hart, 1992).  Looks for a
   step up in total power that matches an appliance's rated wattage, and a
   matching step down later.  No training needed.
2. Disaggregator - learned model.  For each minute it looks at a window of
   the mains signal around that minute (level, jumps, rolling statistics,
   time of day) and uses gradient-boosted trees to decide, per appliance,
   on/off and power.

The model only ever sees the mains signal.  Per-appliance readings are used
for training labels and for scoring, never as an input.

Training data
-------------
By default the model is trained on simulated homes from meter_sim.py and
scored on DIFFERENT simulated homes it has never seen.  To train on a real
recording, export it to CSV (one mains column plus one column per appliance)
and call load_labelled_csv() - see the bottom of this file.

Run:
    python disaggregate.py          # train, evaluate, save to models/
"""

import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from appliance_profiles import APPLIANCE_KEYS, DEFAULT_PROFILES

MODEL_PATH = os.path.join("models", "nilm_model.pkl")
METRICS_PATH = os.path.join("models", "nilm_metrics.json")

# The model uses readings after the minute it is judging as well as before.
# The short windows that matter most reach 30 minutes ahead, so the freshest
# half hour of a live feed is provisional until those readings arrive.
LOOKAHEAD_MIN = 30


# --------------------------------------------------------------------------
# Features from the mains signal only
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df needs `datetime` and `mains_kw` (1-minute rows, one home)."""
    p = df["mains_kw"].astype(float).reset_index(drop=True)
    f = pd.DataFrame({"p": p})
    for lag in (1, 2, 5, 15):
        f[f"back_{lag}"] = p - p.shift(lag)
        f[f"fwd_{lag}"] = p.shift(-lag) - p
    for w in (5, 15, 61):
        roll = p.rolling(w, center=True, min_periods=1)
        f[f"mean_{w}"] = roll.mean()
        f[f"std_{w}"] = roll.std().fillna(0)
        f[f"min_{w}"] = roll.min()
        f[f"max_{w}"] = roll.max()
        f[f"above_min_{w}"] = p - f[f"min_{w}"]
    # Long windows, so a run lasting hours still stands out from the base load.
    for w in (181, 361, 721):
        f[f"above_min_{w}"] = p - p.rolling(w, center=True, min_periods=1).min()
    step = (p - p.shift(1)).abs().fillna(0)
    f["max_step_5"] = step.rolling(5, center=True, min_periods=1).max()
    f["max_step_15"] = step.rolling(15, center=True, min_periods=1).max()
    f["mean_before_15"] = p.shift(1).rolling(15, min_periods=1).mean()
    f["mean_after_15"] = p[::-1].shift(1).rolling(15, min_periods=1).mean()[::-1]
    hour = pd.to_datetime(df["datetime"]).dt.hour.to_numpy() \
        + pd.to_datetime(df["datetime"]).dt.minute.to_numpy() / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return f.fillna(0.0)


def _features_by_home(df: pd.DataFrame) -> pd.DataFrame:
    """Windows must not run across two different homes."""
    if "home" not in df.columns:
        return build_features(df)
    parts = [build_features(g) for _, g in df.groupby("home", sort=False)]
    return pd.concat(parts, ignore_index=True)


def true_states(df: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth on/off from the per-appliance columns."""
    return pd.DataFrame({k: (df[k].to_numpy() > DEFAULT_PROFILES[k].on_threshold_kw)
                         for k in APPLIANCE_KEYS})


# --------------------------------------------------------------------------
# Learned model
# --------------------------------------------------------------------------
class Disaggregator:
    def __init__(self, appliances: Optional[List[str]] = None, seed: int = 0):
        self.appliances = appliances or list(APPLIANCE_KEYS)
        self.seed = seed
        self.classifiers: Dict[str, HistGradientBoostingClassifier] = {}
        self.regressors: Dict[str, HistGradientBoostingRegressor] = {}
        self.trained_on = "not trained"

    def fit(self, df: pd.DataFrame, trained_on: str = "simulated homes") -> "Disaggregator":
        X = _features_by_home(df)
        states = true_states(df)
        for k in self.appliances:
            y_on = states[k].to_numpy()
            clf = HistGradientBoostingClassifier(
                max_iter=340, learning_rate=0.1, max_leaf_nodes=47,
                class_weight="balanced", random_state=self.seed)
            clf.fit(X, y_on)
            self.classifiers[k] = clf
            # Power is learned only from minutes when the appliance was on.
            reg = HistGradientBoostingRegressor(
                max_iter=340, learning_rate=0.1, max_leaf_nodes=47,
                random_state=self.seed)
            if y_on.sum() > 50:
                reg.fit(X[y_on], df[k].to_numpy()[y_on])
                self.regressors[k] = reg
        self.trained_on = trained_on
        return self

    def tune_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Pick, per appliance, how sure the classifier must be before it says
        "on".  The classifiers are trained to miss nothing, which makes them
        call some appliances too readily; the cut-off that gives the best F1
        on `df` (labelled homes NOT used for training or testing) corrects it.
        """
        X = _features_by_home(df)
        states = true_states(df)
        self.thresholds = {}
        for k in self.appliances:
            truth = states[k].to_numpy()
            proba = self.classifiers[k].predict_proba(X)[:, 1]
            best, best_f1 = 0.5, -1.0
            for cut in np.arange(0.30, 0.96, 0.05):
                on = proba >= cut
                tp = int((truth & on).sum())
                f1 = 2 * tp / max(int(truth.sum()) + int(on.sum()), 1)
                if f1 > best_f1 + 1e-9:
                    best, best_f1 = float(round(cut, 2)), f1
            self.thresholds[k] = best
        return self.thresholds

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df: `datetime`, `mains_kw`.  Returns datetime, mains_kw, and for each
        appliance `<key>_on` (bool) and `<key>_kw`, plus `other_kw`.
        """
        X = _features_by_home(df)
        mains = df["mains_kw"].to_numpy(dtype=float)
        out = pd.DataFrame({"datetime": pd.to_datetime(df["datetime"]).to_numpy(),
                            "mains_kw": mains})
        est = {}
        thresholds = getattr(self, "thresholds", None) or {}
        for k in self.appliances:
            on = self.classifiers[k].predict_proba(X)[:, 1] >= thresholds.get(k, 0.5)
            if k in self.regressors:
                kw = np.clip(self.regressors[k].predict(X), 0, None)
            else:
                kw = np.full(len(X), DEFAULT_PROFILES[k].rated_kw)
            est[k] = np.where(on, kw, 0.0)
            out[f"{k}_on"] = on
        # Appliances together can never draw more than the meter measured.
        total = sum(est.values())
        scale = np.where(total > mains, mains / np.maximum(total, 1e-9), 1.0)
        for k in self.appliances:
            out[f"{k}_kw"] = est[k] * scale
        out["other_kw"] = np.clip(mains - sum(out[f"{k}_kw"] for k in self.appliances), 0, None)
        return out

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "Disaggregator":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# --------------------------------------------------------------------------
# Rule-based baseline
# --------------------------------------------------------------------------
class EdgeBaseline:
    """
    Match step changes in the mains to nameplate ratings.  It is given each
    appliance's rated power (what a user could read off the label) and
    nothing else.
    """

    def __init__(self, ratings: Optional[Dict[str, float]] = None, tolerance: float = 0.25):
        self.ratings = ratings or {k: DEFAULT_PROFILES[k].rated_kw for k in APPLIANCE_KEYS}
        self.tolerance = tolerance
        self.appliances = list(APPLIANCE_KEYS)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        mains = df["mains_kw"].to_numpy(dtype=float)
        smooth = pd.Series(mains).rolling(3, center=True, min_periods=1).median().to_numpy()
        step = np.diff(smooth, prepend=smooth[0])
        n = len(mains)
        on = {k: np.zeros(n, dtype=bool) for k in self.appliances}
        running = {k: False for k in self.appliances}
        started = {k: 0 for k in self.appliances}
        for i in range(n):
            s = step[i]
            if abs(s) >= 0.04:
                best, best_err = None, self.tolerance
                for k, r in self.ratings.items():
                    err = abs(abs(s) - r) / r
                    wanted_state = s < 0          # falling edge needs it running
                    if err < best_err and running[k] == wanted_state:
                        best, best_err = k, err
                if best is not None:
                    if s > 0:
                        running[best], started[best] = True, i
                    else:
                        on[best][started[best]:i] = True
                        running[best] = False
            # Give up on a run that outlives the longest plausible one.
            for k in self.appliances:
                if running[k] and i - started[k] > DEFAULT_PROFILES[k].max_run_min:
                    running[k] = False
        out = pd.DataFrame({"datetime": pd.to_datetime(df["datetime"]).to_numpy(),
                            "mains_kw": mains})
        for k in self.appliances:
            out[f"{k}_on"] = on[k]
            out[f"{k}_kw"] = np.where(on[k], np.minimum(self.ratings[k], mains), 0.0)
        out["other_kw"] = np.clip(mains - sum(out[f"{k}_kw"] for k in self.appliances), 0, None)
        return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def evaluate(pred: pd.DataFrame, truth: pd.DataFrame) -> Dict[str, dict]:
    """
    Per appliance:
      precision / recall / f1 - on/off detection, minute by minute
      mae_w                   - mean absolute power error in watts
      energy_error_pct        - error in total kWh assigned to the appliance
    """
    states = true_states(truth)
    result = {}
    for k in APPLIANCE_KEYS:
        t = states[k].to_numpy()
        p = pred[f"{k}_on"].to_numpy()
        tp = int((t & p).sum())
        fp = int((~t & p).sum())
        fn = int((t & ~p).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        true_kw = truth[k].to_numpy()
        est_kw = pred[f"{k}_kw"].to_numpy()
        true_kwh = true_kw.sum() / 60.0
        est_kwh = est_kw.sum() / 60.0
        result[k] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "mae_w": round(float(np.abs(true_kw - est_kw).mean() * 1000), 1),
            "true_kwh": round(float(true_kwh), 2),
            "estimated_kwh": round(float(est_kwh), 2),
            "energy_error_pct": round(abs(est_kwh - true_kwh) / true_kwh * 100, 1)
            if true_kwh > 0 else None,
        }
    return result


# --------------------------------------------------------------------------
# Real datasets
# --------------------------------------------------------------------------
def load_labelled_csv(path: str, datetime_col: str, mains_col: str,
                      appliance_cols: Dict[str, str], unit: str = "W") -> pd.DataFrame:
    """
    Load a real recording for training or testing.

    appliance_cols maps this project's keys to the CSV's column names, e.g.
        {"fridge": "fridge", "ac": "air conditioner 1", "geyser": "geyser",
         "washing_machine": "washing machine", "water_pump": "motor"}
    Appliances missing from the recording are filled with zero.
    Readings are averaged to 1-minute rows.
    """
    raw = pd.read_csv(path)
    raw[datetime_col] = pd.to_datetime(raw[datetime_col])
    factor = 0.001 if unit.upper() == "W" else 1.0
    df = pd.DataFrame({"datetime": raw[datetime_col],
                       "mains_kw": raw[mains_col].astype(float) * factor})
    for k in APPLIANCE_KEYS:
        col = appliance_cols.get(k)
        df[k] = raw[col].astype(float) * factor if col else 0.0
    df = (df.set_index("datetime").resample("1min").mean()
            .interpolate(limit=5).dropna().reset_index())
    df["other_kw"] = np.clip(df["mains_kw"] - df[APPLIANCE_KEYS].sum(axis=1), 0, None)
    return df


# --------------------------------------------------------------------------
# Train + evaluate on simulated homes
# --------------------------------------------------------------------------
def train_default(train_homes: int = 16, test_homes: int = 6, days: int = 10,
                  save: bool = True, verbose: bool = True):
    from meter_sim import (simulate_home, simulate_left_on_day, simulate_many, simulate_random_day,
                           simulate_stacked_day)

    faults = ["geyser_left_on", "pump_dry_run", "fridge_seal"]

    def with_fault_homes(normal, n_fault, seed):
        # Short extra homes where something is wrong, so abnormal runs
        # (a geyser left on for hours) are seen in training and in testing.
        faulty = simulate_many(n_fault, 2, seed=seed, faults=faults)
        faulty["home"] += normal["home"].max() + 1
        return pd.concat([normal, faulty], ignore_index=True)

    def with_random_days(base, n, seed):
        # Days with appliances at arbitrary, overlapping times (see meter_sim).
        parts = [base]
        for i in range(n):
            day = simulate_random_day(seed + i)
            day.insert(0, "home", base["home"].max() + 1 + i)
            parts.append(day)
        return pd.concat(parts, ignore_index=True)

    def with_left_on_days(base, n, seed, maker=simulate_left_on_day):
        # Days where something is switched on and left on for hours.
        parts = [base]
        for i in range(n):
            day = maker(seed + i)
            day.insert(0, "home", base["home"].max() + 1 + i)
            parts.append(day)
        return pd.concat(parts, ignore_index=True)

    def with_varied_homes(base, n, seed):
        # Heavy and light users, big and small households, hot and mild weeks.
        rng = np.random.default_rng(seed)
        parts = [base]
        for i in range(n):
            home = simulate_home(
                days=5, seed=seed + i,
                usage={k: float(rng.choice([0.6, 1.0, 1.5])) for k in APPLIANCE_KEYS},
                people=float(rng.integers(1, 8)), area_sqft=float(rng.integers(500, 2400)),
                temps=list(rng.normal(31, 4, 5)))
            home.insert(0, "home", base["home"].max() + 1 + i)
            parts.append(home)
        return pd.concat(parts, ignore_index=True)

    train = with_fault_homes(simulate_many(train_homes, days, seed=1), 16, seed=5)
    train = with_random_days(train, 120, seed=10_000)
    train = with_left_on_days(train, 110, seed=30_000)
    train = with_varied_homes(train, 14, seed=50_000)
    train = with_left_on_days(train, 120, seed=70_000, maker=simulate_stacked_day)
    test = with_fault_homes(simulate_many(test_homes, days, seed=99), 6, seed=98)  # unseen homes
    test = with_random_days(test, 30, seed=20_000)
    left_on_test = with_left_on_days(simulate_left_on_day(40_000).assign(home=0), 39, seed=40_001)
    stacked_test = with_left_on_days(simulate_stacked_day(80_000).assign(home=0), 39, seed=80_001,
                                     maker=simulate_stacked_day)
    varied_test = with_varied_homes(simulate_home(days=5, seed=60_000).assign(home=0), 7, seed=60_001)
    model = Disaggregator().fit(train, trained_on=f"{train_homes} simulated homes x {days} days")
    # A third set of homes, used only to set each appliance's decision cut-off.
    tune = with_fault_homes(simulate_many(4, 5, seed=501), 3, seed=502)
    tune = with_random_days(tune, 20, seed=90_000)
    tune = with_left_on_days(tune, 15, seed=91_000)
    tune = with_left_on_days(tune, 15, seed=92_000, maker=simulate_stacked_day)
    tune = with_varied_homes(tune, 4, seed=93_000)
    cutoffs = model.tune_thresholds(tune)

    learned = evaluate(model.predict(test), test)
    baseline_parts = [EdgeBaseline().predict(g) for _, g in test.groupby("home", sort=False)]
    baseline = evaluate(pd.concat(baseline_parts, ignore_index=True), test)

    report = {
        "data": "SIMULATED homes (meter_sim.py). Test homes are different from "
                "training homes. Real homes will score lower.",
        "train": f"{train_homes} homes x {days} days, 120 days of random overlapping use, 110 days with "
                 f"appliances left on, 120 days with several switched on together, and 14 homes "
                 f"with varied habits, 1-minute",
        "test": f"{test_homes} unseen homes x {days} days plus 30 unseen random days, 1-minute",
        "model": learned,
        "decision_cutoffs": cutoffs,
        "edge_baseline": baseline,
        "left_on_days": evaluate(model.predict(left_on_test), left_on_test),
        "varied_homes": evaluate(model.predict(varied_test), varied_test),
        "stacked_days": evaluate(model.predict(stacked_test), stacked_test),
        "extra_tests": "left_on_days: 40 unseen days with appliances left on for hours. "
                       "varied_homes: 8 unseen homes with different habits, sizes and weather. "
                       "stacked_days: 40 unseen days with several appliances switched on together.",
    }
    if save:
        model.save()
        with open(METRICS_PATH, "w") as fh:
            json.dump(report, fh, indent=2)
    if verbose:
        print(f"{'appliance':<17}{'F1 model':>9}{'F1 base':>9}{'MAE W':>8}{'kWh err %':>11}")
        for k in APPLIANCE_KEYS:
            print(f"{k:<17}{learned[k]['f1']:>9}{baseline[k]['f1']:>9}"
                  f"{learned[k]['mae_w']:>8}{str(learned[k]['energy_error_pct']):>11}")
    return model, report


if __name__ == "__main__":
    train_default()
