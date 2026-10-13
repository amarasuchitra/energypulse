"""
Forecast pipeline: leak-free features, trained artifacts, live forecasts and
the honesty of the reported accuracy.

Tests that need the trained XGBoost model are skipped when xgboost is not
installed; CI installs it.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from forecast_features import (
    MEASUREMENTS, TARGET, daily_energy, daily_features, hourly_features, hourly_training_set, to_hourly,
)

MODEL_DIR = "models"
META = os.path.join(MODEL_DIR, "forecast_meta.json")


def _sample(hours=24 * 30, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01", periods=hours, freq="h")
    base = 1.0 + 0.6 * np.sin(2 * np.pi * t.hour / 24) + rng.gamma(2, 0.15, hours)
    df = pd.DataFrame({"datetime": t, TARGET: base})
    for c in MEASUREMENTS:
        df[c] = rng.random(hours)
    return df


def _meta():
    with open(META, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------- leak-free features
def test_features_use_nothing_after_the_current_hour():
    df = _sample()
    h = to_hourly(df)
    X = hourly_features(h)
    t = 400
    changed = h.copy()
    changed.iloc[t + 1:] = changed.iloc[t + 1:] * 5 + 3        # rewrite the future, including the target hour
    X2 = hourly_features(changed)
    pd.testing.assert_series_equal(X.iloc[t], X2.iloc[t])


def test_training_target_is_the_next_hour():
    df = _sample()
    X, y, h = hourly_training_set(df)
    for t in X.index[200:205]:
        assert y.loc[t] == pytest.approx(h.loc[t + pd.Timedelta(hours=1), TARGET])
        assert X.loc[t, "lag_0"] == pytest.approx(h.loc[t, TARGET])


def test_missing_hours_stay_missing_not_shifted():
    df = _sample().drop(index=range(300, 305))
    h = to_hourly(df)
    assert len(h) == 24 * 30 and h[TARGET].isna().sum() == 5


def test_daily_features_use_only_the_last_90_days():
    d = daily_energy(to_hourly(_sample(24 * 200)))
    o = 150
    F = daily_features(d, o, seasonal=False)
    d2 = d.copy()
    d2.iloc[:o - 89] = 999.0                                    # older than 90 days
    d2.iloc[o + 1:] = 999.0                                     # the future
    pd.testing.assert_frame_equal(F, daily_features(d2, o, seasonal=False))


# ------------------------------------------------------------- honest metrics
def test_reported_metrics_beat_the_baselines():
    m = _meta()
    mae = m["hourly"]["mae_kw"]
    assert mae["hybrid"] < mae["persistence"] < mae["same_hour_yesterday"]
    assert m["hourly"]["improvement_over_persistence_pct"] > 15
    assert abs(m["hourly"]["interval"]["test_coverage"] - 0.90) < 0.03
    for key in ("daily_seasonal", "daily_generic"):
        t = m["daily"][key]["test"]["next_30_days"]
        assert t["model_total_error_pct"] < t["baseline_last7_total_error_pct"]


def test_choices_were_made_on_validation_not_test():
    m = _meta()
    assert m["blend"]["chosen_on"] == "validation MAE"
    split = m["split"]
    assert split["train_until"] < split["validation_until"] < split["test_until"]


# ------------------------------------------------------------- live forecasts

def _forecaster():
    pytest.importorskip("xgboost")
    from forecast import Forecaster
    fc = Forecaster.load(MODEL_DIR, use_lstm=False)
    assert fc.ready
    return fc


def test_live_forecast_matches_training_features():
    fc = _forecaster()
    from model import load_data
    df = load_data()
    X, y, h = hourly_training_set(df)
    t = X.index[-300]
    live = fc.next_hour(df[df["datetime"] <= t], w_lstm=0.0)
    batch = float(fc.xgb.predict(X.loc[[t]].reindex(columns=fc.features))[0])
    assert live["pred_kw"] == pytest.approx(max(batch, 0.0), rel=1e-6)
    assert live["for_hour"] == t + pd.Timedelta(hours=1)
    assert live["lower_kw"] <= live["pred_kw"] <= live["upper_kw"]


def test_power_only_data_uses_the_power_only_model():
    fc = _forecaster()
    df = _sample()[["datetime", TARGET]]
    out = fc.next_hour(df)
    assert out["power_only"] is True and out["lstm_used"] is False
    assert out["pred_kw"] >= 0 and out["lower_kw"] <= out["upper_kw"]


def test_bill_forecast_covers_next_month():
    fc = _forecaster()
    from forecast import month_slice
    df = _sample(24 * 60)
    out = fc.next_days(df, seasonal=False)
    assert out["day"].nunique() == 62 and (out["predicted_kwh"] >= 0).all()
    nxt = pd.Timestamp(df["datetime"].max()).normalize().replace(day=1) + pd.DateOffset(months=1)
    assert len(month_slice(out, nxt.year, nxt.month)) == 24 * nxt.days_in_month


def test_sample_forecast_file_matches_columns_the_app_uses():
    fc = pd.read_csv(os.path.join("data", "next_month_forecast.csv"), parse_dates=["datetime"])
    assert {"datetime", "predicted_kw", "predicted_kwh", "day"} <= set(fc.columns)
    assert fc["datetime"].min() > pd.Timestamp("2010-11-26")
    assert len(fc) == 62 * 24
