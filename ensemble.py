"""
How the XGBoost and LSTM forecasts are blended.

"auto" uses the weights saved with the trained models (each model weighted by
1 / its test error, see model.hybrid_weights).  "custom" lets the household set
the XGBoost share itself; the LSTM gets the rest.
"""
from typing import Optional, Tuple

DEFAULT_SETTING = {"mode": "auto", "xgb_share": 0.5}


def clean_setting(setting: Optional[dict]) -> dict:
    """A valid setting whatever was stored."""
    out = dict(DEFAULT_SETTING)
    if isinstance(setting, dict):
        if setting.get("mode") in ("auto", "custom"):
            out["mode"] = setting["mode"]
        try:
            share = float(setting.get("xgb_share", out["xgb_share"]))
            if share == share:                                   # not NaN
                out["xgb_share"] = min(1.0, max(0.0, share))
        except (TypeError, ValueError):
            pass
    return out


def resolve_weights(setting: Optional[dict], auto_weights=None) -> Tuple[float, float]:
    """(w_xgb, w_lstm), summing to 1."""
    s = clean_setting(setting)
    if s["mode"] == "custom":
        return s["xgb_share"], 1.0 - s["xgb_share"]
    try:
        w_xgb, w_lstm = float(auto_weights[0]), float(auto_weights[1])
        total = w_xgb + w_lstm
        if total > 0 and w_xgb >= 0 and w_lstm >= 0:
            return w_xgb / total, w_lstm / total
    except (TypeError, ValueError, IndexError, KeyError):
        pass
    return 0.5, 0.5
