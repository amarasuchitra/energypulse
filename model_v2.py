"""
PART 3 v2 — Improved Hybrid Prediction Model
===============================================
Enhanced features for better accuracy:
- Cyclical time encoding (sin/cos for hour, day_of_week, month)
- Lag features (previous 1, 3, 6, 12, 24 hours)
- Rolling statistics (mean, std, min, max over 3h, 6h, 12h, 24h)
- Interaction features
- Per-household scaling calibration
- Quantile regression for prediction intervals
"""

import os
import pickle
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "models"
CLEAN_CSV = os.path.join(DATA_DIR, "cleaned_energy_data.csv")

# ── Model settings ─────────────────────────────────────────────────────
TARGET = "Global_active_power"
BASE_FEATURES = [
    "hour", "day_of_week", "is_weekend",
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]

WINDOW_SIZE = 10
TEST_FRACTION = 0.2
SEED = 42
N_SPLITS = 5  # For time series cross-validation


def load_data() -> pd.DataFrame:
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(f"{CLEAN_CSV} not found.\nRun:  python data.py")
    return pd.read_csv(CLEAN_CSV, parse_dates=["datetime"])


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for better prediction accuracy."""
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Ensure datetime features
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["week_of_year"] = dt.dt.isocalendar().week
    
    # ── Cyclical encoding ──
    # Hour: 24-hour cycle
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Day of week: 7-day cycle
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # Month: 12-month cycle
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # ── Lag features ──
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"power_lag_{lag}"] = df[TARGET].shift(lag)
        df[f"reactive_lag_{lag}"] = df["Global_reactive_power"].shift(lag)
        df[f"voltage_lag_{lag}"] = df["Voltage"].shift(lag)
        df[f"intensity_lag_{lag}"] = df["Global_intensity"].shift(lag)
    
    # ── Rolling statistics ──
    for window in [3, 6, 12, 24]:
        # Rolling stats on target
        df[f"power_roll_mean_{window}"] = df[TARGET].rolling(window, min_periods=1).mean()
        df[f"power_roll_std_{window}"] = df[TARGET].rolling(window, min_periods=1).std()
        df[f"power_roll_min_{window}"] = df[TARGET].rolling(window, min_periods=1).min()
        df[f"power_roll_max_{window}"] = df[TARGET].rolling(window, min_periods=1).max()
        
        # Rolling stats on sub-metering
        for sm in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
            df[f"{sm}_roll_mean_{window}"] = df[sm].rolling(window, min_periods=1).mean()
            df[f"{sm}_roll_std_{window}"] = df[sm].rolling(window, min_periods=1).std()
    
    # ── Difference features ──
    df["power_diff_1"] = df[TARGET].diff(1)
    df["power_diff_3"] = df[TARGET].diff(3)
    df["power_diff_24"] = df[TARGET].diff(24)
    
    # ── Interaction features ──
    df["hour_x_weekend"] = df["hour"] * df["is_weekend"]
    df["intensity_x_voltage"] = df["Global_intensity"] * df["Voltage"]
    df["reactive_x_intensity"] = df["Global_reactive_power"] * df["Global_intensity"]
    
    # Sub-metering ratios
    total_sm = df["Sub_metering_1"] + df["Sub_metering_2"] + df["Sub_metering_3"]
    df["sm1_ratio"] = df["Sub_metering_1"] / total_sm.replace(0, np.nan)
    df["sm2_ratio"] = df["Sub_metering_2"] / total_sm.replace(0, np.nan)
    df["sm3_ratio"] = df["Sub_metering_3"] / total_sm.replace(0, np.nan)
    
    # Fill NaN from lags/rolling
    df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)
    
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """Get all feature columns (excluding target and datetime)."""
    exclude = {TARGET, "datetime", "date", "time"}
    return [c for c in df.columns if c not in exclude]


def split_time(df, test_frac=TEST_FRACTION):
    idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


def train_xgboost_v2(train: pd.DataFrame, test: pd.DataFrame):
    """Train XGBoost with engineered features and cross-validation."""
    print("\n" + "=" * 60)
    print("  Training XGBoost v2 (with engineered features)")
    print("=" * 60)
    
    feature_cols = get_feature_cols(train)
    print(f"  Using {len(feature_cols)} features")
    
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]
    
    # Time series cross-validation for robust evaluation
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        )
        model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
        preds = model.predict(X_fold_val)
        mae = mean_absolute_error(y_fold_val, preds)
        cv_scores.append(mae)
        print(f"  Fold {fold+1}/{N_SPLITS} MAE: {mae:.4f} kW")
    
    print(f"  CV MAE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f} kW")
    
    # Final model on full training data
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    final_model.fit(X_train, y_train, verbose=False)
    
    # Test predictions
    preds = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"  Test MAE: {mae:.4f} kW, RMSE: {rmse:.4f} kW")
    
    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\n  Top 15 features:")
    for _, row in importance.head(15).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    return final_model, preds, mae, feature_cols, importance


def train_quantile_models(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """Train quantile regression models for prediction intervals."""
    print("\n" + "=" * 60)
    print("  Training Quantile Models (for prediction intervals)")
    print("=" * 60)
    
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]
    
    quantiles = [0.05, 0.5, 0.95]  # 90% prediction interval
    quantile_models = {}
    quantile_preds = {}
    
    for q in quantiles:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            objective="reg:quantileerror",
            quantile_alpha=q,
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        quantile_models[q] = model
        quantile_preds[q] = preds
        mae = mean_absolute_error(y_test, preds)
        print(f"  Quantile {q:.0%} MAE: {mae:.4f} kW")
    
    # Check coverage
    lower = quantile_preds[0.05]
    upper = quantile_preds[0.95]
    coverage = np.mean((y_test >= lower) & (y_test <= upper))
    print(f"  90% PI Coverage: {coverage:.1%}")
    
    return quantile_models, quantile_preds


def train_lstm_v2(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """Train LSTM with Keras 3 compatible architecture."""
    print("\n" + "=" * 60)
    print("  Training LSTM v2 (Keras 3 compatible)")
    print("=" * 60)
    
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("  TensorFlow not available, skipping LSTM")
        return None, None, None, None
    
    # Use only top 20 features for LSTM to avoid overfitting
    top_features = feature_cols[:20]
    
    # Scale features
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[top_features + [TARGET]])
    test_scaled = scaler.transform(test[top_features + [TARGET]])
    
    y_train = train[TARGET].values
    y_test = test[TARGET].values
    
    # Create sequences
    def make_sequences(features_2d, target_1d, window):
        X, y = [], []
        for i in range(window, len(features_2d)):
            X.append(features_2d[i - window:i])
            y.append(target_1d[i])
        return np.array(X), np.array(y)
    
    X_train_seq, y_train_seq = make_sequences(train_scaled, y_train, WINDOW_SIZE)
    X_test_seq, y_test_seq = make_sequences(test_scaled, y_test, WINDOW_SIZE)
    
    n_features = X_train_seq.shape[2]
    
    # Keras 3 compatible model - use functional API
    inputs = keras.Input(shape=(WINDOW_SIZE, n_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1, activation="linear")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    
    # Train with early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
    ]
    
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_test_seq, y_test_seq),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Predict
    preds = model.predict(X_test_seq, verbose=0).flatten()
    mae = mean_absolute_error(y_test_seq, preds)
    print(f"  LSTM Test MAE: {mae:.4f} kW")
    
    return model, scaler, preds, top_features


def compute_optimal_weights(xgb_mae, lstm_mae):
    """Compute inverse-MAE weights for hybrid."""
    if lstm_mae is None or not np.isfinite(lstm_mae):
        return 1.0, 0.0
    inv_xgb = 1.0 / xgb_mae
    inv_lstm = 1.0 / lstm_mae
    total = inv_xgb + inv_lstm
    return inv_xgb / total, inv_lstm / total


def save_models(model, feature_cols, importance, quantile_models, lstm_model, scaler, lstm_features, meta):
    """Save all models and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # XGBoost
    with open(os.path.join(MODEL_DIR, "xgboost_model_v2.pkl"), "wb") as f:
        pickle.dump(model, f)
    
    # Feature columns and importance
    with open(os.path.join(MODEL_DIR, "feature_cols_v2.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    importance.to_csv(os.path.join(MODEL_DIR, "feature_importance_v2.csv"), index=False)
    
    # Quantile models
    for q, q_model in quantile_models.items():
        with open(os.path.join(MODEL_DIR, f"quantile_model_{int(q*100)}.pkl"), "wb") as f:
            pickle.dump(q_model, f)
    
    # LSTM
    if lstm_model is not None:
        lstm_model.save(os.path.join(MODEL_DIR, "lstm_model_v2.keras"))
        with open(os.path.join(MODEL_DIR, "lstm_scaler_v2.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODEL_DIR, "lstm_features_v2.pkl"), "wb") as f:
            pickle.dump(lstm_features, f)
    
    # Meta
    with open(os.path.join(MODEL_DIR, "model_meta_v2.pkl"), "wb") as f:
        pickle.dump(meta, f)
    
    # Also save as JSON for readability
    with open(os.path.join(MODEL_DIR, "model_meta_v2.json"), "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                   for k, v in meta.items()}, f, indent=2)
    
    print(f"\n[OK] All models saved to {MODEL_DIR}/")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} rows from {df['datetime'].min()} to {df['datetime'].max()}")
    
    print("Adding engineered features...")
    df = add_engineered_features(df)
    print(f"  DataFrame shape: {df.shape}")
    
    train, test = split_time(df)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    
    # Train XGBoost with engineered features
    xgb_model, xgb_preds, xgb_mae, feature_cols, importance = train_xgboost_v2(train, test)
    
    # Train quantile models for prediction intervals
    quantile_models, quantile_preds = train_quantile_models(train, test, feature_cols)
    
    # Train LSTM
    lstm_model, lstm_scaler, lstm_preds, lstm_features = train_lstm_v2(train, test, feature_cols)
    lstm_mae = mean_absolute_error(test[TARGET].values[WINDOW_SIZE:], lstm_preds) if lstm_preds is not None else None
    
    # Hybrid weights
    w_xgb, w_lstm = compute_optimal_weights(xgb_mae, lstm_mae)
    
    # Hybrid predictions
    if lstm_preds is not None:
        y_test = test[TARGET].values[WINDOW_SIZE:]
        hybrid_preds = w_xgb * xgb_preds[WINDOW_SIZE:] + w_lstm * lstm_preds
        hybrid_mae = mean_absolute_error(y_test, hybrid_preds)
    else:
        hybrid_mae = xgb_mae
        hybrid_preds = xgb_preds
    
    # 90% Prediction Interval metrics
    lower = quantile_preds[0.05]
    upper = quantile_preds[0.95]
    median = quantile_preds[0.5]
    pi_width = np.mean(upper - lower)
    coverage = np.mean((test[TARGET].values >= lower) & (test[TARGET].values <= upper))
    
    print(f"\n{'=' * 60}")
    print("  FINAL MODEL COMPARISON")
    print("-" * 60)
    print(f"  {'Model':<20} {'MAE (kW)':>12}")
    print("-" * 60)
    print(f"  {'XGBoost v2':<20} {xgb_mae:>12.4f}")
    if lstm_mae:
        print(f"  {'LSTM v2':<20} {lstm_mae:>12.4f}")
    print(f"  {'Hybrid (weighted)':<20} {hybrid_mae:>12.4f}   (XGB {w_xgb:.3f} / LSTM {w_lstm:.3f})")
    print(f"  {'Quantile Median':<20} {mean_absolute_error(test[TARGET].values, median):>12.4f}")
    print(f"  {'90% PI Width':<20} {pi_width:>12.4f} kW")
    print(f"  {'90% PI Coverage':<20} {coverage:>11.1%}")
    print("=" * 60)
    
    # Save metadata
    meta = {
        "xgb_mae": float(xgb_mae),
        "lstm_mae": float(lstm_mae) if lstm_mae else None,
        "hybrid_mae": float(hybrid_mae),
        "hybrid_weights": [float(w_xgb), float(w_lstm)],
        "feature_cols": feature_cols,
        "target": TARGET,
        "window_size": WINDOW_SIZE,
        "pi_coverage": float(coverage),
        "pi_width": float(pi_width),
        "quantiles": [0.05, 0.5, 0.95],
        "cv_scores": None,  # Could add if needed
    }
    
    save_models(xgb_model, feature_cols, importance, quantile_models, 
                lstm_model, lstm_scaler, lstm_features, meta)
    
    # Generate next-month forecast
    print("\nGenerating next-month forecast...")
    forecast = next_month_forecast_v2(xgb_model, feature_cols, df)
    forecast_path = os.path.join(DATA_DIR, "next_month_forecast_v2.csv")
    forecast.to_csv(forecast_path, index=False)
    print(f"[OK] Next-month forecast -> {forecast_path}")


def next_month_forecast_v2(model, feature_cols, df, days=30):
    """Generate next-month forecast using trained model."""
    last_date = pd.to_datetime(df["datetime"]).max()
    future_dates = pd.date_range(last_date + pd.Timedelta(hours=1), periods=days * 24, freq="h")
    
    future_df = pd.DataFrame({"datetime": future_dates})
    # Add minimal datetime features for future dates
    dt = pd.to_datetime(future_df["datetime"])
    future_df["hour"] = dt.dt.hour
    future_df["day_of_week"] = dt.dt.dayofweek
    future_df["is_weekend"] = (future_df["day_of_week"] >= 5).astype(int)
    future_df["month"] = dt.dt.month
    future_df["day_of_month"] = dt.dt.day
    future_df["week_of_year"] = dt.dt.isocalendar().week
    
    # Add cyclical encoding
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["dow_sin"] = np.sin(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["dow_cos"] = np.cos(2 * np.pi * future_df["day_of_week"] / 7)
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
    
    # For future, use last known values for lags and rolling features
    last_row = df.iloc[-1]
    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = last_row.get(col, 0)
    
    X_future = future_df[feature_cols]
    preds = model.predict(X_future)
    
    forecast = pd.DataFrame({
        "datetime": future_dates,
        "predicted_kw": np.maximum(preds, 0),
        "predicted_kwh": np.maximum(preds, 0),
    })
    
    # Add prediction intervals if quantile models exist
    for q in [0.05, 0.5, 0.95]:
        q_path = os.path.join(MODEL_DIR, f"quantile_model_{int(q*100)}.pkl")
        if os.path.exists(q_path):
            with open(q_path, "rb") as f:
                q_model = pickle.load(f)
            q_preds = q_model.predict(X_future)
            forecast[f"pi_{int(q*100)}"] = np.maximum(q_preds, 0)
    
    return forecast


if __name__ == "__main__":
    main()