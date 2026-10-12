"""XGBoost/LSTM blend: automatic by accuracy, or set by the household."""
from ensemble import DEFAULT_SETTING, clean_setting, resolve_weights


def test_auto_uses_saved_weights_normalised():
    assert resolve_weights({"mode": "auto"}, (0.966, 0.034)) == (0.966 / 1.0, 0.034 / 1.0)
    w = resolve_weights(None, (3, 1))
    assert w == (0.75, 0.25)


def test_custom_share():
    assert resolve_weights({"mode": "custom", "xgb_share": 0.7}, (0.9, 0.1)) == (0.7, 1.0 - 0.7)
    assert resolve_weights({"mode": "custom", "xgb_share": 1.4}, None) == (1.0, 0.0)
    assert resolve_weights({"mode": "custom", "xgb_share": -1}, None) == (0.0, 1.0)


def test_bad_values_fall_back_safely():
    assert clean_setting("nonsense") == DEFAULT_SETTING
    assert clean_setting({"mode": "x", "xgb_share": "abc"}) == DEFAULT_SETTING
    assert clean_setting({"mode": "custom", "xgb_share": float("nan")})["xgb_share"] == 0.5
    assert resolve_weights({"mode": "auto"}, None) == (0.5, 0.5)
    assert resolve_weights({"mode": "auto"}, (0, 0)) == (0.5, 0.5)
    assert resolve_weights({"mode": "auto"}, (-1, 2)) == (0.5, 0.5)
