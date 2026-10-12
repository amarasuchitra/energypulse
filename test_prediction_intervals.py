"""The shipped 5%-95% band: loads, is ordered, and keeps its stated coverage."""
import json
import os
import pickle

import numpy as np
import pandas as pd

MODELS = "models"


def _load():
    models = {}
    for q in (5, 50, 95):
        with open(os.path.join(MODELS, f"quantile_model_{q}.pkl"), "rb") as fh:
            models[q] = pickle.load(fh)
    with open(os.path.join(MODELS, "quantile_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return models, meta


def _holdout(meta):
    df = pd.read_csv(os.path.join("data", "cleaned_energy_data.csv"), parse_dates=["datetime"])
    df = df.sort_values("datetime").dropna(subset=meta["feature_cols"] + [meta["target"]])
    return df.iloc[int(len(df) * 0.85):]


def test_meta_matches_the_v1_model_features():
    from model import FEATURE_COLS, TARGET
    _, meta = _load()
    assert meta["feature_cols"] == FEATURE_COLS and meta["target"] == TARGET
    assert meta["quantiles"] == [0.05, 0.5, 0.95]


def test_band_is_ordered_and_non_negative():
    models, meta = _load()
    X = _holdout(meta)[meta["feature_cols"]].head(500)
    off = meta["conformal_offset_kw"]
    lo = np.maximum(0, np.minimum(models[5].predict(X), models[95].predict(X)) - off)
    hi = np.maximum(models[5].predict(X), models[95].predict(X)) + off
    mid = models[50].predict(X)
    assert (lo <= hi).all() and (lo >= 0).all()
    assert np.mean((mid >= lo) & (mid <= hi)) > 0.99


def test_calibrated_band_keeps_its_coverage_on_unseen_hours():
    models, meta = _load()
    test = _holdout(meta)
    X, y = test[meta["feature_cols"]], test[meta["target"]].values
    off = meta["conformal_offset_kw"]
    lo = np.maximum(0, np.minimum(models[5].predict(X), models[95].predict(X)) - off)
    hi = np.maximum(models[5].predict(X), models[95].predict(X)) + off
    coverage = np.mean((y >= lo) & (y <= hi))
    assert coverage >= meta["nominal_coverage"] - 0.02
    assert abs(coverage - meta["pi_coverage"]) < 0.01
