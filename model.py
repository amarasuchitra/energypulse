"""
PART 3 — Forecasting models (XGBoost + LSTM hybrid, bill forecast)
==================================================================

Trains every forecasting model the dashboard uses, on the cleaned UCI
household data, with a strict time-ordered split:

    train       first 64% of the hours      (fit)
    validation  next 16%                    (early stopping, model choice, blend weights, interval calibration)
    test        last 20%                    (reported once; never used to choose anything)

1. Next-hour power (kW)
   - XGBoost on leak-free features: past power, past sub-meter and other
     readings, rolling statistics and the clock of the target hour
     (forecast_features.hourly_features).  Trained on absolute error.
   - LSTM on the last 48 hours of readings.
   - Hybrid: the blend weight that gives the lowest validation error.
   - 90% band: 5% and 95% quantile models, conformally calibrated on validation.
   All are compared with the persistence baseline ("next hour = this hour")
   and seasonal baselines on the test period.

2. Daily energy for the next 62 days (bill forecast)
   - Gradient-boosted model on the last 90 days of daily energy.  A
     household-specific variant also uses the time of year; a generic variant
     (for uploaded data from other homes) does not.
   - Compared with "average of the last 7 / 28 days" on the test period.

Artifacts (models/):
    forecast_xgb.json            XGBoost booster (version-portable JSON)
    forecast_lstm.keras          LSTM
    forecast_lstm_scaling.json   LSTM input scaling
    forecast_q05.pkl, forecast_q95.pkl   interval models
    daily_seasonal.pkl, daily_generic.pkl  bill-forecast models
    forecast_meta.json           features, weights, calibration and every test metric
and data/next_month_forecast.csv for the bundled sample.

Run after data.py:
    python model.py
"""

import json
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd

from forecast_features import (
    DAILY_HISTORY, DAILY_HORIZON, LSTM_INPUTS, LSTM_WINDOW, MEASUREMENTS, TARGET,  # noqa: F401 (TARGET re-exported)
    daily_energy, daily_features, daily_training_set,
    hourly_shape, hourly_training_set, lstm_frame,
)

DATA_DIR = "data"
MODEL_DIR = "models"
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_energy_data.csv")
SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.64, 0.16
NOMINAL_COVERAGE = 0.90

PATHS = {
    "xgb": os.path.join(MODEL_DIR, "forecast_xgb.json"),
    "xgb_power": os.path.join(MODEL_DIR, "forecast_xgb_power.json"),
    "q05_power": os.path.join(MODEL_DIR, "forecast_q05_power.pkl"),
    "q95_power": os.path.join(MODEL_DIR, "forecast_q95_power.pkl"),
    "lstm": os.path.join(MODEL_DIR, "forecast_lstm.keras"),
    "lstm_scaling": os.path.join(MODEL_DIR, "forecast_lstm_scaling.json"),
    "q05": os.path.join(MODEL_DIR, "forecast_q05.pkl"),
    "q95": os.path.join(MODEL_DIR, "forecast_q95.pkl"),
    "daily_seasonal": os.path.join(MODEL_DIR, "daily_seasonal.pkl"),
    "daily_generic": os.path.join(MODEL_DIR, "daily_generic.pkl"),
    "meta": os.path.join(MODEL_DIR, "forecast_meta.json"),
}


def load_data() -> pd.DataFrame:
    """Load the cleaned CSV that data.py produced."""
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"{CLEAN_CSV} not found.\nRun:  python data.py")
    return pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])


def _mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m])))


def _rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def _log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------- next hour
def train_xgboost(Xtr, ytr, Xva, yva, only=None):
    import xgboost as xgb
    grid = [only] if only else [
        dict(max_depth=6, learning_rate=0.03, min_child_weight=10, subsample=0.8, colsample_bytree=0.8),
        dict(max_depth=8, learning_rate=0.03, min_child_weight=20, subsample=0.8, colsample_bytree=0.7),
        dict(max_depth=4, learning_rate=0.05, min_child_weight=10, subsample=0.9, colsample_bytree=0.9),
        dict(max_depth=7, learning_rate=0.02, min_child_weight=30, subsample=0.7, colsample_bytree=0.6,
             reg_lambda=5.0),
    ]
    best = None
    for params in grid:
        t = time.time()
        model = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=4000,
                                 early_stopping_rounds=150, tree_method="hist",
                                 random_state=SEED, n_jobs=-1, **params)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        mae = _mae(yva, model.predict(Xva))
        _log(f"  XGBoost {params} -> val MAE {mae:.4f} ({model.best_iteration} trees, {time.time() - t:.0f}s)")
        if best is None or mae < best[0]:
            best = (mae, model, params)
    _log(f"  XGBoost chosen: {best[2]}  val MAE {best[0]:.4f}")
    params = dict(best[2], n_estimators=int(best[1].best_iteration) + 1)
    return best[1], params


def refit_xgboost(X, y, params, scale):
    """Refit on more data with the chosen settings; tree count scaled with the data."""
    import xgboost as xgb
    p = dict(params)
    p["n_estimators"] = max(50, int(round(p["n_estimators"] * scale)))
    model = xgb.XGBRegressor(objective="reg:absoluteerror", tree_method="hist",
                             random_state=SEED, n_jobs=-1, **p)
    return model.fit(X, y, verbose=False)


def lstm_samples(h, origins, mean, std):
    """
    (X, valid) for LSTM windows ending at each origin hour.
    X[i] holds the scaled inputs for the 48 hours up to and including origins[i].
    """
    F = lstm_frame(h)
    arr = ((F.values - mean) / std).astype("float32")
    pos = F.index.get_indexer(origins)
    X = np.zeros((len(origins), LSTM_WINDOW, arr.shape[1]), dtype="float32")
    valid = np.zeros(len(origins), dtype=bool)
    for i, p in enumerate(pos):
        if p < LSTM_WINDOW - 1:
            continue
        w = arr[p - LSTM_WINDOW + 1:p + 1]
        if np.isfinite(w).all():
            X[i] = w
            valid[i] = True
    return X, valid


def _lstm_model():
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(LSTM_WINDOW, len(LSTM_INPUTS))),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mae")
    return model


def train_lstm(h, idx_tr, ytr, idx_va, yva):
    from tensorflow import keras
    keras.utils.set_random_seed(SEED)
    F = lstm_frame(h)
    train_rows = F.loc[F.index <= idx_tr[-1]]
    mean = np.nanmean(train_rows.values, axis=0)
    std = np.nanstd(train_rows.values, axis=0)
    std[std == 0] = 1.0
    y_mean, y_std = float(np.mean(ytr)), float(np.std(ytr))

    Xtr, vtr = lstm_samples(h, idx_tr, mean, std)
    Xva, vva = lstm_samples(h, idx_va, mean, std)
    ttr = ((np.asarray(ytr) - y_mean) / y_std).astype("float32")
    tva = ((np.asarray(yva) - y_mean) / y_std).astype("float32")
    _log(f"  LSTM windows: train {vtr.sum()}, validation {vva.sum()}")

    model = _lstm_model()
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    t = time.time()
    hist = model.fit(Xtr[vtr], ttr[vtr], validation_data=(Xva[vva], tva[vva]),
                     epochs=60, batch_size=128, callbacks=[stop], verbose=2)
    best_epochs = int(np.argmin(hist.history["val_loss"])) + 1
    _log(f"  LSTM trained in {time.time() - t:.0f}s; best after {best_epochs} epochs")
    scaling = {"inputs": LSTM_INPUTS, "window": LSTM_WINDOW,
               "mean": [float(v) for v in mean], "std": [float(v) for v in std],
               "y_mean": y_mean, "y_std": y_std, "epochs": best_epochs}
    return model, scaling


def refit_lstm(h, idx, y, scaling, scale):
    """Refit on train + validation for the chosen number of epochs (scaled with the data)."""
    from tensorflow import keras
    keras.utils.set_random_seed(SEED)
    X, valid = lstm_samples(h, idx, np.array(scaling["mean"]), np.array(scaling["std"]))
    t = ((np.asarray(y) - scaling["y_mean"]) / scaling["y_std"]).astype("float32")
    model = _lstm_model()
    epochs = max(3, int(round(scaling["epochs"] * scale)))
    model.fit(X[valid], t[valid], epochs=epochs, batch_size=128, verbose=2)
    return model


def lstm_predict(model, scaling, h, origins):
    mean, std = np.array(scaling["mean"]), np.array(scaling["std"])
    X, valid = lstm_samples(h, origins, mean, std)
    out = np.full(len(origins), np.nan)
    if valid.any():
        raw = model.predict(X[valid], verbose=0, batch_size=512).ravel()
        out[valid] = raw * scaling["y_std"] + scaling["y_mean"]
    return out


def best_blend(y, p_xgb, p_lstm):
    """Weight on the LSTM (0..1) that minimises MAE where both predictions exist."""
    m = np.isfinite(p_lstm)
    best_w, best_mae = 0.0, _mae(y[m], p_xgb[m])
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        mae = _mae(y[m], (1 - w) * p_xgb[m] + w * p_lstm[m])
        if mae < best_mae - 1e-9:
            best_w, best_mae = float(w), mae
    return best_w


def blend(p_xgb, p_lstm, w_lstm):
    out = np.asarray(p_xgb, float).copy()
    m = np.isfinite(p_lstm)
    out[m] = (1 - w_lstm) * out[m] + w_lstm * np.asarray(p_lstm)[m]
    return np.maximum(out, 0.0)


def train_interval(Xtr, ytr, Xva, yva):
    from sklearn.ensemble import HistGradientBoostingRegressor
    q = {}
    for alpha in (0.05, 0.95):
        q[alpha] = HistGradientBoostingRegressor(
            loss="quantile", quantile=alpha, learning_rate=0.05, max_iter=600,
            max_leaf_nodes=31, min_samples_leaf=40, random_state=SEED).fit(Xtr, ytr)
    lo, hi = q[0.05].predict(Xva), q[0.95].predict(Xva)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    yv = np.asarray(yva)
    scores = np.maximum(lo - yv, yv - hi)
    n = len(scores)
    offset = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * NOMINAL_COVERAGE) / n)))
    return q, offset


def band(q, offset, X):
    lo, hi = q[0.05].predict(X), q[0.95].predict(X)
    return np.maximum(0.0, np.minimum(lo, hi) - offset), np.maximum(lo, hi) + offset


# ----------------------------------------------------------------- daily
def train_daily(d, a_day, b_day, seasonal):
    from sklearn.ensemble import HistGradientBoostingRegressor
    train = daily_training_set(d, range(DAILY_HISTORY - 1, b_day - DAILY_HORIZON), seasonal)
    train = train.dropna(subset=["y"])
    feats = [c for c in train.columns if c not in ("y", "origin")]
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.03, max_iter=800, max_leaf_nodes=15,
        min_samples_leaf=60, l2_regularization=1.0, random_state=SEED).fit(train[feats], train["y"])
    return model, feats


def daily_metrics(d, model, feats, origins, seasonal):
    test = daily_training_set(d, origins, seasonal)
    test["p"] = model.predict(test[feats])
    for name in ("m7", "m28"):
        test["b_" + name] = test[name]
    out = {}
    for h_max, label in ((30, "next_30_days"), (DAILY_HORIZON, "next_62_days")):
        part = test[test["h"] <= h_max].dropna(subset=["y"])
        g = part.groupby("origin").agg(y=("y", "sum"), p=("p", "sum"), m7=("b_m7", "sum"),
                                       m28=("b_m28", "sum"), k=("y", "count"))
        g = g[g["k"] == h_max]
        out[label] = {
            "model_total_error_pct": round(float((np.abs(g.y - g.p) / g.y).mean() * 100), 2),
            "baseline_last7_total_error_pct": round(float((np.abs(g.y - g.m7) / g.y).mean() * 100), 2),
            "baseline_last28_total_error_pct": round(float((np.abs(g.y - g.m28) / g.y).mean() * 100), 2),
            "origins": int(len(g)),
        }
    part = test[test["h"] <= 30].dropna(subset=["y"])
    out["daily_mae_kwh"] = round(_mae(part.y, part.p), 3)
    out["daily_mae_kwh_baseline_last7"] = round(_mae(part.y, part.b_m7), 3)
    return out


def sample_month_forecast(h, d, model, feats, seasonal=True):
    """Hourly forecast for the 62 days after the sample data ends (real dates)."""
    origin = len(d) - 1
    F = daily_features(d, origin, seasonal)
    daily_kwh = np.maximum(model.predict(F[feats]), 0.0)
    shape = hourly_shape(h)
    start = d.index[origin] + pd.Timedelta(days=1)
    rows = []
    for i, kwh in enumerate(daily_kwh):
        day = start + pd.Timedelta(days=i)
        frac = shape.loc[day.dayofweek].values
        for hour in range(24):
            rows.append((day + pd.Timedelta(hours=hour), kwh * frac[hour]))
    out = pd.DataFrame(rows, columns=["datetime", "predicted_kw"])
    out["predicted_kwh"] = out["predicted_kw"]
    out["day"] = out["datetime"].dt.date
    out["hour"] = out["datetime"].dt.hour
    return out


# ----------------------------------------------------------------- main
def main(skip_lstm: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    started = time.time()
    df = load_data()
    X, y, h = hourly_training_set(df)
    n = len(X)
    a, b = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    Xtr, ytr, Xva, yva, Xte, yte = X.iloc[:a], y.iloc[:a], X.iloc[a:b], y.iloc[a:b], X.iloc[b:], y.iloc[b:]
    _log(f"Hourly rows {n}: train to {X.index[a - 1]}, validation to {X.index[b - 1]}, test to {X.index[-1]}")
    yv, yt = yva.values, yte.values

    _log("\n== Baselines (test) ==")
    baselines = {
        "persistence": _mae(yt, Xte["lag_0"]),
        "same_hour_yesterday": _mae(yt, Xte["same_hour_d1"]),
        "same_hour_last_week": _mae(yt, Xte["same_hour_d7"]),
        "same_hour_mean_7_days": _mae(yt, Xte["same_hour_mean7"]),
    }
    for k, v in baselines.items():
        _log(f"  {k:24s} MAE {v:.4f} kW")

    _log("\n== XGBoost ==")
    xgb_model, xgb_params = train_xgboost(Xtr, ytr, Xva, yva)
    px_va, px_te_trainonly = xgb_model.predict(Xva), xgb_model.predict(Xte)

    w_lstm, lstm_model, scaling = 0.0, None, None
    pl_va = np.full(len(yv), np.nan)
    pl_te = np.full(len(yt), np.nan)
    pl_te_trainonly = pl_te
    if not skip_lstm:
        _log("\n== LSTM ==")
        try:
            lstm_model, scaling = train_lstm(h, Xtr.index, ytr.values, Xva.index, yv)
            pl_va = lstm_predict(lstm_model, scaling, h, Xva.index)
            pl_te_trainonly = lstm_predict(lstm_model, scaling, h, Xte.index)
            w_lstm = best_blend(yv, px_va, pl_va)
        except Exception as exc:                       # keep XGBoost if TensorFlow fails
            _log(f"  LSTM skipped: {type(exc).__name__}: {exc}")
            lstm_model = None
    ph_va = blend(px_va, pl_va, w_lstm)
    trainonly_mae = _mae(yt, blend(px_te_trainonly, pl_te_trainonly if lstm_model is not None else pl_te, w_lstm))

    # Final models: every choice above is kept fixed and the models are refit on
    # train + validation, so they also learn from the most recent history.  The
    # test period is still unseen.
    _log("\n== Refit on train + validation ==")
    scale = b / a
    Xtv, ytv = X.iloc[:b], y.iloc[:b]
    xgb_model = refit_xgboost(Xtv, ytv, xgb_params, scale)
    xgb_params = dict(xgb_params, n_estimators=xgb_model.get_params()["n_estimators"])
    px_te = xgb_model.predict(Xte)
    if lstm_model is not None and w_lstm > 0:
        lstm_model = refit_lstm(h, Xtv.index, ytv.values, scaling, scale)
        pl_te = lstm_predict(lstm_model, scaling, h, Xte.index)
    elif lstm_model is not None:
        pl_te = pl_te_trainonly
    ph_te = blend(px_te, pl_te, w_lstm)
    _log(f"  test MAE: train-only models {trainonly_mae:.4f} -> refit models {_mae(yt, ph_te):.4f}")

    _log("\n== 90% band ==")
    q, offset = train_interval(Xtr, ytr, Xva, yva)
    lo, hi = band(q, offset, Xte)
    coverage = float(np.mean((yt >= lo) & (yt <= hi)))
    _log(f"  conformal offset {offset:.4f} kW; test coverage {coverage:.3f}; mean width {np.mean(hi - lo):.3f} kW")

    _log("\n== Power-only variant (for data without sub-meter readings) ==")
    power_feats = [c for c in X.columns if not any(c.startswith(m) for m in MEASUREMENTS)]
    _, power_params = train_xgboost(Xtr[power_feats], ytr, Xva[power_feats], yva,
                                    only={k: v for k, v in xgb_params.items() if k != "n_estimators"})
    xgb_power = refit_xgboost(Xtv[power_feats], ytv, power_params, scale)
    pp_te = np.maximum(xgb_power.predict(Xte[power_feats]), 0.0)
    q_p, offset_p = train_interval(Xtr[power_feats], ytr, Xva[power_feats], yva)
    lo_p, hi_p = band(q_p, offset_p, Xte[power_feats])
    power_only = {
        "features": power_feats,
        "interval_offset_kw": offset_p,
        "mae_kw": round(_mae(yt, pp_te), 4),
        "improvement_over_persistence_pct": round((1 - _mae(yt, pp_te) / baselines["persistence"]) * 100, 1),
        "interval": {"test_coverage": round(float(np.mean((yt >= lo_p) & (yt <= hi_p))), 4),
                     "mean_width_kw": round(float(np.mean(hi_p - lo_p)), 4)},
    }
    _log(f"  power-only XGBoost test MAE {power_only['mae_kw']} "
         f"({power_only['improvement_over_persistence_pct']}% better than persistence); "
         f"band coverage {power_only['interval']['test_coverage']}")

    lm = np.isfinite(pl_te)
    hourly_metrics = {
        "test_rows": int(len(yt)),
        "test_mean_kw": round(float(np.mean(yt)), 4),
        "mae_kw": {
            "hybrid": round(_mae(yt, ph_te), 4),
            "xgboost": round(_mae(yt, px_te), 4),
            "lstm": round(_mae(yt[lm], pl_te[lm]), 4) if lm.any() else None,
            **{k: round(v, 4) for k, v in baselines.items()},
        },
        "rmse_kw": {"hybrid": round(_rmse(yt, ph_te), 4), "persistence": round(_rmse(yt, Xte["lag_0"]), 4)},
        "improvement_over_persistence_pct": round((1 - _mae(yt, ph_te) / baselines["persistence"]) * 100, 1),
        "hybrid_mae_train_only_models_kw": round(trainonly_mae, 4),
        "validation_mae_kw": {"hybrid": round(_mae(yv, ph_va), 4), "xgboost": round(_mae(yv, px_va), 4),
                              "lstm": round(_mae(yv[np.isfinite(pl_va)], pl_va[np.isfinite(pl_va)]), 4)
                              if np.isfinite(pl_va).any() else None},
        "interval": {"nominal": NOMINAL_COVERAGE, "test_coverage": round(coverage, 4),
                     "mean_width_kw": round(float(np.mean(hi - lo)), 4), "conformal_offset_kw": round(offset, 5)},
    }
    _log("\n== Next hour, test period ==")
    for k, v in hourly_metrics["mae_kw"].items():
        _log(f"  {k:24s} MAE {v}")
    _log(f"  hybrid beats persistence by {hourly_metrics['improvement_over_persistence_pct']}%")

    _log("\n== Bill forecast (daily energy) ==")
    d = daily_energy(h)
    nd = len(d)
    a_day, b_day = int(nd * TRAIN_FRAC), int(nd * (TRAIN_FRAC + VAL_FRAC))
    test_origins = range(b_day, nd - 30)
    daily = {}
    for seasonal, key in ((True, "daily_seasonal"), (False, "daily_generic")):
        model, feats = train_daily(d, a_day, b_day, seasonal)
        daily[key] = {"features": feats, "test": daily_metrics(d, model, feats, test_origins, seasonal)}
        with open(PATHS[key], "wb") as fh:
            pickle.dump(model, fh)
        _log(f"  {key}: {json.dumps(daily[key]['test']['next_30_days'])}")
        if seasonal:
            forecast = sample_month_forecast(h, d, model, feats, seasonal=True)
            forecast.to_csv(os.path.join(DATA_DIR, "next_month_forecast.csv"), index=False)

    # ---- save
    xgb_model.save_model(PATHS["xgb"])
    xgb_power.save_model(PATHS["xgb_power"])
    for alpha, key in ((0.05, "q05_power"), (0.95, "q95_power")):
        with open(PATHS[key], "wb") as fh:
            pickle.dump(q_p[alpha], fh)
    if lstm_model is not None:
        lstm_model.save(PATHS["lstm"])
        with open(PATHS["lstm_scaling"], "w") as fh:
            json.dump(scaling, fh)
    for alpha, key in ((0.05, "q05"), (0.95, "q95")):
        with open(PATHS[key], "wb") as fh:
            pickle.dump(q[alpha], fh)

    import sklearn
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__}
    try:
        import xgboost
        versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    if lstm_model is not None:
        import tensorflow
        versions["tensorflow"] = tensorflow.__version__
    meta = {
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "split": {"train_until": str(X.index[a - 1]), "validation_until": str(X.index[b - 1]),
                  "test_until": str(X.index[-1]), "fractions": [TRAIN_FRAC, VAL_FRAC, 1 - TRAIN_FRAC - VAL_FRAC]},
        "hourly_features": list(X.columns),
        "xgb_params": xgb_params,
        "blend": {"w_xgb": round(1 - w_lstm, 2), "w_lstm": round(w_lstm, 2),
                  "chosen_on": "validation MAE", "lstm_available": lstm_model is not None},
        "interval_offset_kw": offset,
        "hourly": hourly_metrics,
        "power_only": power_only,
        "daily": daily,
        "versions": versions,
        "training_seconds": round(time.time() - started),
    }
    with open(PATHS["meta"], "w") as fh:
        json.dump(meta, fh, indent=2)
    _log(f"\nSaved models to {MODEL_DIR}/ in {meta['training_seconds']}s")
    return meta


if __name__ == "__main__":
    import sys
    main(skip_lstm="--no-lstm" in sys.argv)
