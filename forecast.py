"""
Live forecasts for the dashboard, from the models model.py trains.

    fc = Forecaster.load()
    fc.next_hour(history)        # power in the hour after the newest reading, with a 90% band
    fc.next_days(history, ...)   # hourly forecast for the next 62 days (bill, calendar)

`history` is the household's readings with real or replayed timestamps and at
least Global_active_power; the other meter columns are used when present.
Every number reported to the user about accuracy comes from
models/forecast_meta.json, which model.py writes from the test period.
"""
import json
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from forecast_features import (
    DAILY_HISTORY, LSTM_WINDOW, MEASUREMENTS, TARGET, daily_energy, daily_features,
    hourly_features, hourly_shape, lstm_frame, to_hourly,
)

MODEL_DIR = "models"


class Forecaster:
    def __init__(self, meta, xgb=None, lstm=None, scaling=None, q05=None, q95=None,
                 daily_seasonal=None, daily_generic=None, model_dir: Optional[str] = None,
                 xgb_power=None, q05_power=None, q95_power=None):
        self.meta = meta or {}
        self.xgb, self.lstm, self.scaling = xgb, lstm, scaling
        self.q05, self.q95 = q05, q95
        self.daily_seasonal, self.daily_generic = daily_seasonal, daily_generic
        self.features = self.meta.get("hourly_features", [])
        self.w_lstm = float(self.meta.get("blend", {}).get("w_lstm", 0.0))
        self.offset = float(self.meta.get("interval_offset_kw", 0.0))
        self._model_dir = model_dir
        # Variant for data without sub-meter / voltage / current readings.
        self.xgb_power, self.q05_power, self.q95_power = xgb_power, q05_power, q95_power
        power = self.meta.get("power_only", {})
        self.power_features = power.get("features", [])
        self.power_offset = float(power.get("interval_offset_kw", 0.0))
        self._lstm_tried = lstm is not None

    @property
    def lstm_available(self) -> bool:
        """True when an LSTM was trained and may be used (it is loaded on first use)."""
        if self.lstm is not None:
            return True
        if self._lstm_tried:
            return False
        return bool(self._model_dir) and os.path.exists(os.path.join(self._model_dir, "forecast_lstm.keras"))

    def _ensure_lstm(self) -> bool:
        """Load TensorFlow and the LSTM on first use; False if that is not possible."""
        if self.lstm is not None:
            return True
        if self._lstm_tried or not self.lstm_available:
            return False
        self._lstm_tried = True
        try:
            from tensorflow import keras
            self.lstm = keras.models.load_model(os.path.join(self._model_dir, "forecast_lstm.keras"))
            with open(os.path.join(self._model_dir, "forecast_lstm_scaling.json"), encoding="utf-8") as fh:
                self.scaling = json.load(fh)
            return True
        except Exception as exc:          # TensorFlow missing or too large for the host
            print(f"[forecast] LSTM not loaded, XGBoost alone: {exc}")
            self.lstm = self.scaling = None
            return False

    # ---------------------------------------------------------- loading
    @classmethod
    def load(cls, model_dir: str = MODEL_DIR, use_lstm: bool = True) -> "Forecaster":
        def path(name):
            return os.path.join(model_dir, name)

        def unpickle(name):
            try:
                with open(path(name), "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:
                print(f"[forecast] {name} not loaded: {exc}")
                return None

        try:
            with open(path("forecast_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            return cls({})
        def booster(name):
            try:
                import xgboost
                model = xgboost.XGBRegressor()
                model.load_model(path(name))
                return model
            except Exception as exc:
                print(f"[forecast] {name} not loaded: {exc}")
                return None

        xgb = booster("forecast_xgb.json")
        xgb_power = booster("forecast_xgb_power.json") if os.path.exists(path("forecast_xgb_power.json")) else None
        fc = cls(meta, xgb, None, None, unpickle("forecast_q05.pkl"), unpickle("forecast_q95.pkl"),
                 unpickle("daily_seasonal.pkl"), unpickle("daily_generic.pkl"), model_dir=model_dir,
                 xgb_power=xgb_power,
                 q05_power=unpickle("forecast_q05_power.pkl") if xgb_power is not None else None,
                 q95_power=unpickle("forecast_q95_power.pkl") if xgb_power is not None else None)
        if not use_lstm:
            fc._lstm_tried = True
        return fc

    @property
    def ready(self) -> bool:
        return self.xgb is not None and bool(self.features)

    # ---------------------------------------------------------- next hour
    def next_hour(self, history: pd.DataFrame, w_lstm: Optional[float] = None) -> Optional[dict]:
        """
        Forecast for the hour after the newest reading in `history`.
        `w_lstm` overrides the blend chosen in training (Settings > Forecast model).
        Returns None when there is too little recent history.
        """
        if not self.ready or history is None or history.empty or TARGET not in history.columns:
            return None
        recent = history[pd.to_datetime(history["datetime"]) >=
                         pd.to_datetime(history["datetime"]).max() - pd.Timedelta(days=10)]
        h = to_hourly(recent)
        if h.empty or pd.isna(h[TARGET].iloc[-1]):
            return None
        # Without sub-meter and other meter readings in the last day, use the
        # model trained on power alone rather than feed the full model blanks.
        last_day = h.iloc[-24:]
        has_meters = any(c in last_day.columns and last_day[c].notna().any() for c in MEASUREMENTS)
        power_only = not has_meters and self.xgb_power is not None and bool(self.power_features)
        feats = self.power_features if power_only else self.features
        X = hourly_features(h).iloc[[-1]].reindex(columns=feats)
        model = self.xgb_power if power_only else self.xgb
        p_xgb = float(model.predict(X)[0])
        w = self.w_lstm if w_lstm is None else min(1.0, max(0.0, float(w_lstm)))
        if power_only:
            w = 0.0                       # the LSTM needs the meter readings too
        p_lstm, lstm_used = None, False
        if w > 0 and self._ensure_lstm():
            p_lstm = self._lstm_next(h)
            lstm_used = p_lstm is not None
        pred = (1 - w) * p_xgb + w * p_lstm if lstm_used else p_xgb
        pred = max(0.0, pred)
        lo = hi = None
        q05, q95, offset = ((self.q05_power, self.q95_power, self.power_offset) if power_only
                            else (self.q05, self.q95, self.offset))
        if q05 is not None and q95 is not None:
            a, b = float(q05.predict(X)[0]), float(q95.predict(X)[0])
            lo, hi = max(0.0, min(a, b) - offset), max(a, b) + offset
            lo, hi = min(lo, pred), max(hi, pred)
        hourly = self.meta.get("hourly", {})
        if power_only:
            pw = self.meta.get("power_only", {})
            hourly = {"interval": pw.get("interval", {}), "mae_kw": {
                "hybrid": pw.get("mae_kw"), "persistence": hourly.get("mae_kw", {}).get("persistence")},
                "improvement_over_persistence_pct": pw.get("improvement_over_persistence_pct")}
        return {
            "pred_kw": pred, "xgb_kw": p_xgb, "lstm_kw": p_lstm, "lstm_used": lstm_used,
            "w_xgb": 1 - w if lstm_used else 1.0, "w_lstm": w if lstm_used else 0.0,
            "lower_kw": lo, "upper_kw": hi,
            "coverage": hourly.get("interval", {}).get("test_coverage"),
            "persistence_kw": float(h[TARGET].iloc[-1]),
            "for_hour": h.index[-1] + pd.Timedelta(hours=1),
            "test_mae_kw": hourly.get("mae_kw", {}).get("hybrid"),
            "persistence_mae_kw": hourly.get("mae_kw", {}).get("persistence"),
            "improvement_pct": hourly.get("improvement_over_persistence_pct"),
            "power_only": power_only,
        }

    def _lstm_next(self, h: pd.DataFrame) -> Optional[float]:
        s = self.scaling
        F = lstm_frame(h).iloc[-LSTM_WINDOW:]
        if len(F) < LSTM_WINDOW:
            return None
        arr = (F.values - np.array(s["mean"])) / np.array(s["std"])
        if not np.isfinite(arr).all():
            return None
        out = self.lstm.predict(arr.reshape(1, LSTM_WINDOW, -1).astype("float32"), verbose=0)
        return float(np.asarray(out).ravel()[0] * s["y_std"] + s["y_mean"])

    # ---------------------------------------------------------- next days
    def next_days(self, history: pd.DataFrame, seasonal: bool = False,
                  season_shift: Optional[pd.Timedelta] = None) -> pd.DataFrame:
        """
        Hourly forecast (datetime, predicted_kw, predicted_kwh, day, hour) for the
        62 days after the last full day in `history`.

        seasonal=True uses the model that knows this sample household's time of
        year; `season_shift` maps replayed dates back to their real dates for it.
        Other households get the generic model, which relies on recent use only.
        """
        model = self.daily_seasonal if seasonal else self.daily_generic
        key = "daily_seasonal" if seasonal else "daily_generic"
        feats = self.meta.get("daily", {}).get(key, {}).get("features")
        if model is None or not feats or history is None or history.empty:
            return pd.DataFrame()
        h = to_hourly(history)
        d = daily_energy(h).dropna()
        if len(d) < 14:
            return pd.DataFrame()
        d = d.iloc[-DAILY_HISTORY:]
        season_dates = (d.index - season_shift) if (seasonal and season_shift is not None) else None
        F = daily_features(d, len(d) - 1, seasonal, season_dates=season_dates)
        daily_kwh = np.maximum(model.predict(F.reindex(columns=feats)), 0.0)
        shape = hourly_shape(h)
        start = d.index[-1] + pd.Timedelta(days=1)
        rows = []
        for i, kwh in enumerate(daily_kwh):
            day = start + pd.Timedelta(days=i)
            frac = shape.loc[day.dayofweek].values
            rows.extend((day + pd.Timedelta(hours=hr), kwh * frac[hr]) for hr in range(24))
        out = pd.DataFrame(rows, columns=["datetime", "predicted_kw"])
        out["predicted_kwh"] = out["predicted_kw"]
        out["day"] = out["datetime"].dt.date
        out["hour"] = out["datetime"].dt.hour
        return out


def month_slice(forecast: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Rows of a forecast that fall in one calendar month."""
    if forecast is None or forecast.empty:
        return pd.DataFrame()
    dt = pd.to_datetime(forecast["datetime"])
    return forecast[(dt.dt.year == year) & (dt.dt.month == month)]
