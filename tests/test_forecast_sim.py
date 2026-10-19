"""The simulated meter's next-hour forecasts: no look-ahead, sensible output, and graceful absence."""
import numpy as np
import pandas as pd
import pytest

from forecast import Forecaster, meter_hour_forecasts, meter_upcoming
from forecast_features import TARGET, hourly_features, to_hourly
from sim_dataset import power_features


class SumModel:
    """Stand-in for the trained model: any change to an input changes the output."""
    def __init__(self, scale=1.0):
        self.scale = scale

    def predict(self, X):
        return np.abs(np.nan_to_num(X.to_numpy(dtype=float)).sum(axis=1)) * 1e-3 * self.scale


def _series(hours=24 * 10, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-03-01", periods=hours, freq="h")
    kw = 0.6 + 0.4 * np.sin(np.arange(hours) * 2 * np.pi / 24) ** 2 + rng.gamma(2, 0.1, hours)
    return pd.DataFrame({"datetime": t, TARGET: kw})


def _forecaster(series, with_band=True):
    fc = object.__new__(Forecaster)
    feats = power_features(hourly_features(to_hourly(series)).columns)
    fc.xgb_sim, fc.sim_features, fc.sim_offset = SumModel(), feats, 0.05
    fc.q05_sim, fc.q95_sim = (SumModel(0.5), SumModel(1.5)) if with_band else (None, None)
    return fc


def test_forecasts_only_use_earlier_hours():
    s = _series()
    fc = _forecaster(s)
    before = meter_hour_forecasts(fc, s)
    changed = s.copy()
    changed.loc[changed.index[-1], TARGET] += 5.0          # the last completed hour changes
    after = meter_hour_forecasts(fc, changed)
    last_hour = s["datetime"].iloc[-1]
    keep = before["target_hour"] <= last_hour              # forecasts made before that hour
    pd.testing.assert_series_equal(before.loc[keep, "predicted_kw"], after.loc[keep, "predicted_kw"])
    assert before["predicted_kw"].iloc[-1] != after["predicted_kw"].iloc[-1]


def test_each_forecast_is_for_the_following_hour_and_band_contains_it():
    s = _series()
    out = meter_hour_forecasts(_forecaster(s), s)
    assert (out["target_hour"].iloc[-1] - s["datetime"].iloc[-1]) == pd.Timedelta(hours=1)
    assert (out["predicted_kw"] >= 0).all()
    assert (out["lower_kw"] <= out["predicted_kw"]).all() and (out["upper_kw"] >= out["predicted_kw"]).all()


def test_upcoming_gives_consecutive_future_hours():
    s = _series()
    up = meter_upcoming(_forecaster(s), s, steps=2)
    last = s["datetime"].iloc[-1]
    assert list(up["target_hour"]) == [last + pd.Timedelta(hours=1), last + pd.Timedelta(hours=2)]


def test_no_model_means_no_forecast_not_a_made_up_one():
    s = _series()
    fc = _forecaster(s)
    fc.xgb_sim = None
    assert meter_hour_forecasts(fc, s).empty and meter_upcoming(fc, s).empty


def test_missing_band_models_leave_band_empty():
    s = _series()
    out = meter_hour_forecasts(_forecaster(s, with_band=False), s)
    assert out["lower_kw"].isna().all() and not out["predicted_kw"].isna().any()


def test_weather_and_household_context_reach_the_model():
    from sim_dataset import CONTEXT_FEATURES
    s = _series()
    fc = _forecaster(s)
    fc.sim_features = fc.sim_features + CONTEXT_FEATURES
    seen = {}

    class Spy(SumModel):
        def predict(self, X):
            seen["X"] = X
            return super().predict(X)

    fc.xgb_sim = Spy()
    ctx = {"city": "bengaluru", "people": 3, "area": 900, "owned": ("fridge", "ac", "geyser")}
    out = meter_hour_forecasts(fc, s, ctx)
    X = seen["X"]
    assert not X[CONTEXT_FEATURES].isna().any().any()
    assert (X["people"] == 3).all() and (X["has_ac"] == 1).all() and (X["n_appliances"] == 3).all()
    assert len(out) == len(s)
    # without context the model still answers (columns left empty), never a made-up value
    assert len(meter_hour_forecasts(fc, s)) == len(s)


def test_saved_live_meter_model_is_the_one_the_report_names():
    """The committed model loads as the type forecast_meta.json says was chosen (no silent fallback)."""
    import json
    import os
    from forecast import _LightGBMModel
    from paths import MODEL_DIR

    meta_file = os.path.join(MODEL_DIR, "forecast_meta.json")
    if not os.path.exists(os.path.join(MODEL_DIR, "forecast_xgb_sim.json")):
        pytest.skip("live-meter model not trained")
    chosen = json.load(open(meta_file)).get("simulated_meter", {}).get("model")
    pytest.importorskip("xgboost")
    if chosen == "lightgbm":
        pytest.importorskip("lightgbm")
    fc = Forecaster.load(use_lstm=False)
    assert fc.xgb_sim is not None
    assert isinstance(fc.xgb_sim, _LightGBMModel) == (chosen == "lightgbm")
    n = fc.xgb_sim.booster.num_feature() if chosen == "lightgbm" else fc.xgb_sim.n_features_in_
    pred = fc.xgb_sim.predict(np.zeros((2, n)))
    assert len(pred) == 2 and np.all(np.isfinite(pred))
