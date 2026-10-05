"""
PART 3 — Hybrid Prediction Model (XGBoost + LSTM)
===================================================
Trains two models on Global_active_power using time-based features:
  1. XGBoost regressor — trained on time + measurement features
  2. LSTM neural network — trained on a sliding window of the last 10 readings

The hybrid prediction is the average of both models.
Also provides next_month_forecast() for calendar view and cost projection.

Models saved to models/:
  - xgboost_model.pkl
  - lstm_model.keras
  - lstm_scaler.pkl
  - model_meta.pkl

Run AFTER data.py:
    python model.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_energy_data.csv")

# ── Model settings ─────────────────────────────────────────────────────
TARGET = "Global_active_power"  # the column we predict (in kW)

# Features used by both XGBoost and LSTM (time + raw measurements)
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

# Extended features for the next-month forecast (uses only time-based features
# since we won't have future measurements — model extrapolates from time alone)
TIME_FEATURES = ["hour", "day_of_week", "is_weekend"]

WINDOW_SIZE = 10      # LSTM sliding-window length (past 10 hours)
TEST_FRACTION = 0.2   # hold out last 20% for evaluation
SEED = 42


# ──────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load the cleaned CSV that data.py produced."""
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"{CLEAN_CSV} not found.\nRun:  python data.py"
        )
    return pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])


def split_time(df, test_frac=TEST_FRACTION):
    """
    Time-ordered train/test split (no shuffling!).
    The first (1-test_frac) rows train, the rest evaluate.
    """
    idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


# ──────────────────────────────────────────────────────────────────────
# XGBoost training
# ──────────────────────────────────────────────────────────────────────

def train_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train an XGBoost regressor on time + measurement features.
    Returns (trained_model, predictions_on_test, MAE).
    """
    print("\n" + "=" * 55)
    print("  Training XGBoost Regressor")
    print("=" * 55)

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"  XGBoost MAE : {mae:.4f} kW")
    return model, preds, mae


# ──────────────────────────────────────────────────────────────────────
# LSTM training
# ──────────────────────────────────────────────────────────────────────

def make_sequences(features_2d: np.ndarray, target_1d: np.ndarray, window: int):
    """
    Build sliding-window (X, y) pairs for the LSTM.
    For window=10: X = features[i-10:i], y = target[i].
    """
    X, y = [], []
    for i in range(window, len(features_2d)):
        X.append(features_2d[i - window : i])
        y.append(target_1d[i])
    return np.array(X), np.array(y)


def train_lstm(train: pd.DataFrame, test: pd.DataFrame):
    """
    Train a 2-layer LSTM neural network using a sliding window.
    Returns (trained_model, scaler, predictions_on_test, MAE).
    """
    print("\n" + "=" * 55)
    print("  Training LSTM Neural Network")
    print("=" * 55)

    # Scale features to [0, 1] — LSTMs are sensitive to input scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[FEATURE_COLS + [TARGET]])
    test_scaled = scaler.transform(test[FEATURE_COLS + [TARGET]])

    y_train = train[TARGET].values
    y_test = test[TARGET].values

    # Create sliding window sequences
    X_train_seq, y_train_seq = make_sequences(train_scaled, y_train, WINDOW_SIZE)
    X_test_seq, y_test_seq = make_sequences(test_scaled, y_test, WINDOW_SIZE)

    n_features = X_train_seq.shape[2]

    # LSTM architecture:
    #   Layer 1: LSTM 64 units, return_sequences for stacking
    #   Dropout 20% to prevent overfitting
    #   Layer 2: LSTM 32 units
    #   Dropout 20%
    #   Dense 16 (ReLU)
    #   Output: single neuron (predicted kW)
    model = keras.Sequential([
        keras.layers.LSTM(
            64, return_sequences=True,
            input_shape=(WINDOW_SIZE, n_features),
        ),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("  Training (this may take a few minutes) ...")
    model.fit(
        X_train_seq, y_train_seq,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    preds = model.predict(X_test_seq, verbose=0).flatten()
    mae = mean_absolute_error(y_test_seq, preds)
    print(f"  LSTM MAE    : {mae:.4f} kW")
    return model, scaler, preds, mae


# ──────────────────────────────────────────────────────────────────────
# Hybrid model
# ──────────────────────────────────────────────────────────────────────

def hybrid_weights(xgb_mae, lstm_mae):
    """
    Blend the two models in inverse proportion to their validation MAE.

    A plain 50/50 average is only sensible when the two models are equally
    accurate. Here XGBoost is roughly an order of magnitude more accurate
    than the LSTM, so an equal average is dragged well below XGBoost alone.
    Weighting by 1/MAE keeps both models in the blend while letting the more
    accurate one dominate. Returns (w_xgb, w_lstm) summing to 1.
    """
    xgb_mae = abs(float(xgb_mae))
    lstm_mae = abs(float(lstm_mae))
    if not (np.isfinite(xgb_mae) and np.isfinite(lstm_mae)):
        return 0.5, 0.5
    if xgb_mae <= 0 and lstm_mae <= 0:
        return 0.5, 0.5
    if xgb_mae <= 0:
        return 0.0, 1.0
    if lstm_mae <= 0:
        return 1.0, 0.0
    inv_x, inv_l = 1.0 / xgb_mae, 1.0 / lstm_mae
    total = inv_x + inv_l
    return inv_x / total, inv_l / total


def compute_hybrid(xgb_preds, lstm_preds, y_true, weights=None):
    """
    Blend XGBoost and LSTM predictions.
    LSTM preds start at index WINDOW_SIZE (sequences discard first rows).
    XGBoost preds cover every test row. Align them.
    Returns (hybrid_predictions, hybrid_MAE).
    """
    n = min(len(xgb_preds), len(lstm_preds))
    if weights is None:
        weights = hybrid_weights(
            mean_absolute_error(y_true[WINDOW_SIZE: WINDOW_SIZE + n], xgb_preds[:n]),
            mean_absolute_error(y_true[WINDOW_SIZE: WINDOW_SIZE + n], lstm_preds[:n]),
        )
    w_xgb, w_lstm = weights
    hybrid = (w_xgb * xgb_preds[WINDOW_SIZE: WINDOW_SIZE + n]
              + w_lstm * lstm_preds[:n])
    aligned_y = y_true[WINDOW_SIZE: WINDOW_SIZE + n]
    mae = mean_absolute_error(aligned_y, hybrid)
    return hybrid, mae


# ──────────────────────────────────────────────────────────────────────
# Next-month forecast
# ──────────────────────────────────────────────────────────────────────

def next_month_forecast(xgb_model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrapolate the hybrid model's hour/day-of-week pattern across every day
    of the coming calendar month.

    Uses XGBoost's ability to predict from time-features alone by filling
    measurement columns with the historical average for that hour-of-day.

    Returns a DataFrame with columns:
      datetime, predicted_kwh (total for that hour), day
    """
    # Determine the "next month" from the last date in the data
    last_date = df["datetime"].max()
    next_month = last_date.month + 1 if last_date.month < 12 else 1
    next_year = last_date.year if last_date.month < 12 else last_date.year + 1

    # Generate every hour in the next calendar month
    start = pd.Timestamp(f"{next_year}-{next_month:02d}-01")
    # End is last hour of the month
    if next_month == 12:
        end_month_start = pd.Timestamp(f"{next_year + 1}-01-01")
    else:
        end_month_start = pd.Timestamp(f"{next_year}-{next_month + 1:02d}-01")
    end = end_month_start - pd.Timedelta(hours=1)

    hours = pd.date_range(start, end, freq="h")
    forecast_df = pd.DataFrame({"datetime": hours})

    # Add time features
    forecast_df["hour"] = forecast_df["datetime"].dt.hour
    forecast_df["day_of_week"] = forecast_df["datetime"].dt.dayofweek
    forecast_df["is_weekend"] = forecast_df["datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    # For measurements, use the historical hourly average for each hour-of-day
    # This gives the model realistic input values to predict from
    hourly_avg = df.groupby("hour")[FEATURE_COLS].mean()
    for col in FEATURE_COLS:
        if col not in TIME_FEATURES:
            forecast_df[col] = forecast_df["hour"].map(hourly_avg[col])

    # XGBoost prediction for each hour
    X_pred = forecast_df[FEATURE_COLS]
    forecast_df["predicted_kw"] = xgb_model.predict(X_pred)

    # Each row is 1 hour; energy (kWh) = power (kW) * 1 hour
    forecast_df["predicted_kwh"] = forecast_df["predicted_kw"]

    # Add day column for daily aggregation
    forecast_df["day"] = forecast_df["datetime"].dt.date

    return forecast_df


def _as_float(value, default=0.0):
    """Coerce anything (None/NaN/str) into a finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _feature_frame(row_dict):
    """Build a single-row feature frame with numeric, finite values."""
    data = {col: _as_float(row_dict.get(col, 0.0)) for col in FEATURE_COLS}
    return pd.DataFrame([data], columns=FEATURE_COLS)


def _lstm_window(recent_df):
    """
    Return the last WINDOW_SIZE rows as a float frame, or (None, reason) if
    the window cannot be used.

    The scaler was fitted on FEATURE_COLS + [TARGET] in that exact order, so
    the window must be built the same way and must be exactly WINDOW_SIZE long.
    """
    if recent_df is None or not isinstance(recent_df, pd.DataFrame):
        return None, "no_history"
    needed = FEATURE_COLS + [TARGET]
    missing = [c for c in needed if c not in recent_df.columns]
    if missing:
        return None, "missing_columns:" + ",".join(missing[:3])
    if len(recent_df) < WINDOW_SIZE:
        return None, "insufficient_history"
    window = recent_df[needed].tail(WINDOW_SIZE)
    if window.isna().any().any():
        return None, "non_finite_history"
    return window.astype("float32"), "ok"


def predict_next_period(row_dict, xgb_model, lstm_model, scaler, recent_df,
                        return_info=False, weights=None):
    """
    Make a hybrid prediction for the next period given the current row
    and a window of recent readings.

    Returns (hybrid_pred, xgb_pred, lstm_pred) all in kW.
    With return_info=True a fourth element is appended:
    {"lstm_used": bool, "lstm_reason": str, "w_xgb": float, "w_lstm": float}
    """
    if weights is None:
        weights = (0.5, 0.5)
    w_xgb, w_lstm = float(weights[0]), float(weights[1])
    if not (np.isfinite(w_xgb) and np.isfinite(w_lstm)) or (w_xgb + w_lstm) <= 0:
        w_xgb, w_lstm = 0.5, 0.5
    total_w = w_xgb + w_lstm
    w_xgb, w_lstm = w_xgb / total_w, w_lstm / total_w

    feat_df = _feature_frame(row_dict)
    xgb_pred = _as_float(xgb_model.predict(feat_df)[0])

    lstm_pred = xgb_pred
    lstm_used = False

    if lstm_model is None or scaler is None:
        reason = "lstm_unavailable"
        w_lstm, w_xgb = 0.0, 1.0
    else:
        window, reason = _lstm_window(recent_df)
        if window is not None:
            try:
                scaled = np.asarray(scaler.transform(window), dtype="float32")
                if scaled.shape != (WINDOW_SIZE, len(FEATURE_COLS) + 1):
                    reason = "scaler_shape_mismatch"
                else:
                    lstm_input = scaled.reshape(1, WINDOW_SIZE, -1)
                    raw = lstm_model.predict(lstm_input, verbose=0)
                    candidate = _as_float(np.asarray(raw).ravel()[0], default=float("nan"))
                    if np.isfinite(candidate):
                        lstm_pred = candidate
                        lstm_used = True
                        reason = "ok"
                    else:
                        reason = "non_finite_output"
            except Exception as exc:  # noqa: BLE001 - never let the LSTM kill the UI
                reason = f"lstm_error:{type(exc).__name__}"

    hybrid = _as_float(w_xgb * xgb_pred + w_lstm * lstm_pred)
    if return_info:
        return hybrid, xgb_pred, lstm_pred, {
            "lstm_used": lstm_used,
            "lstm_reason": reason,
            "w_xgb": round(w_xgb, 4),
            "w_lstm": round(w_lstm, 4),
        }
    return hybrid, xgb_pred, lstm_pred


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    train, test = split_time(df)
    y_test = test[TARGET].values

    # Train both models
    xgb_model, xgb_preds, xgb_mae = train_xgboost(train, test)
    lstm_model, scaler, lstm_preds, lstm_mae = train_lstm(train, test)

    # Combine into hybrid
    w_xgb, w_lstm = hybrid_weights(xgb_mae, lstm_mae)
    _, h_mae = compute_hybrid(xgb_preds, lstm_preds, y_test, weights=(w_xgb, w_lstm))
    _, equal_mae = compute_hybrid(xgb_preds, lstm_preds, y_test, weights=(0.5, 0.5))

    # Print comparison
    print(f"\n{'=' * 55}")
    print("  MODEL COMPARISON (lower MAE = better)")
    print("-" * 55)
    print(f"  {'Model':<18} {'MAE (kW)':>12}")
    print("-" * 55)
    print(f"  {'XGBoost':<18} {xgb_mae:>12.4f}")
    print(f"  {'LSTM':<18} {lstm_mae:>12.4f}")
    print(f"  {'Hybrid 50/50':<18} {equal_mae:>12.4f}")
    print(f"  {'Hybrid weighted':<18} {h_mae:>12.4f}   "
          f"(XGBoost {w_xgb:.3f} / LSTM {w_lstm:.3f})")
    print("=" * 55)

    # Save XGBoost model
    xgb_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_model, f)
    print(f"\n[OK] XGBoost model  -> {xgb_path}")

    # Save LSTM model
    lstm_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    lstm_model.save(lstm_path)
    print(f"[OK] LSTM model     -> {lstm_path}")

    # Save scaler (needed at inference time for LSTM inputs)
    scaler_path = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[OK] LSTM scaler    -> {scaler_path}")

    # Save model metadata for the dashboard
    meta_path = os.path.join(MODEL_DIR, "model_meta.pkl")
    meta = {
        "xgb_mae": xgb_mae,
        "lstm_mae": lstm_mae,
        "hybrid_mae": h_mae,
        "hybrid_mae_equal": equal_mae,
        "hybrid_weights": (w_xgb, w_lstm),
        "feature_cols": FEATURE_COLS,
        "target": TARGET,
        "window_size": WINDOW_SIZE,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"[OK] Model metadata -> {meta_path}")

    # Generate and save next-month forecast
    forecast = next_month_forecast(xgb_model, df)
    forecast_path = os.path.join(DATA_DIR, "next_month_forecast.csv")
    forecast.to_csv(forecast_path, index=False)
    print(f"[OK] Next-month forecast -> {forecast_path}")


if __name__ == "__main__":
    main()
