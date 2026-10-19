"""Prediction (brief section 6) and optimisation (section 7).

Tests marked "real model" load the committed model files and need xgboost/lightgbm
(installed in CI); they are skipped, never passed, where those are missing.
"""
import gzip
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pytest

import model as M
import sim_dataset as sd
from appliance_profiles import APPLIANCE_KEYS, DEFAULT_PROFILES, HouseholdPrefs
from forecast import (Forecaster, _lightgbm, meter_hour_forecasts, meter_upcoming, model_feature_count,
                      sim_schema_problem)
from forecast_features import TARGET, hourly_features, to_hourly
from meter_sim import simulate_home
from paths import MODEL_DIR
from scheduler import estimated_savings, feasible_starts, recommend, validate_plan
from sessions import extract_sessions
from sim_dataset import power_features
from tariff import Tariff


# ------------------------------------------------------------------ helpers
class Recorder:
    """Stand-in model that remembers what it was given and returns the last-hour value."""
    def __init__(self, out=None):
        self.out, self.seen = out, []

    def predict(self, X):
        self.seen.append(X.copy())
        if self.out is not None:
            return np.full(len(X), self.out, dtype=float)
        return X["lag_0"].fillna(0.0).to_numpy(dtype=float)


def _series(hours=24 * 10, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-03-01", periods=hours, freq="h")
    kw = 0.6 + 0.4 * np.sin(np.arange(hours) * 2 * np.pi / 24) ** 2 + rng.gamma(2, 0.1, hours)
    return pd.DataFrame({"datetime": t, TARGET: kw})


def _fc(model=None, features=None):
    fc = object.__new__(Forecaster)
    fc.xgb_sim = model or Recorder()
    fc.sim_features = features or power_features(hourly_features(to_hourly(_series())).columns)
    fc.sim_offset, fc.q05_sim, fc.q95_sim, fc.sim_error = 0.0, None, None, None
    return fc


def _real():
    pytest.importorskip("xgboost")
    meta = json.load(open(os.path.join(MODEL_DIR, "forecast_meta.json")))
    if meta.get("simulated_meter", {}).get("model") == "lightgbm":
        pytest.importorskip("lightgbm")
    fc = Forecaster.load(use_lstm=False)
    assert fc.xgb_sim is not None, getattr(fc, "sim_error", None)
    return fc, meta["simulated_meter"]


def _detected(days=21, seed=4):
    sim = simulate_home(days=days, seed=seed)
    pred = pd.DataFrame({"datetime": sim["datetime"], "mains_kw": sim["mains_kw"]})
    for key in APPLIANCE_KEYS:
        kw = sim[key].astype(float) if key in sim else pd.Series(0.0, index=sim.index)
        pred[f"{key}_kw"] = kw
        pred[f"{key}_on"] = kw > 0.02
    pred["other_kw"] = sim["other_kw"]
    return pred


# ------------------------------------------------------------------ prediction
@pytest.mark.brief("ML-001", "ML-012")
def test_forecast_uses_the_latest_hour_in_the_right_feature_order():
    model = Recorder()
    fc = _fc(model)
    s = _series()
    out = meter_hour_forecasts(fc, s)
    X = model.seen[-1]
    assert list(X.columns) == fc.sim_features
    assert out["target_hour"].iloc[-1] == s["datetime"].iloc[-1] + pd.Timedelta(hours=1)
    assert out["predicted_kw"].iloc[-1] == pytest.approx(s[TARGET].iloc[-1])     # lag_0 = newest reading
    s2 = s.copy()
    s2.loc[s2.index[-1], TARGET] += 1.0                                          # a new reading arrives
    assert meter_hour_forecasts(fc, s2)["predicted_kw"].iloc[-1] == pytest.approx(s2[TARGET].iloc[-1])


@pytest.mark.brief("ML-002", "UI-003", "UI-008")
def test_no_history_or_no_model_gives_no_forecast_not_a_made_up_one():
    assert meter_hour_forecasts(_fc(), pd.DataFrame(columns=["datetime", TARGET])).empty
    fc = _fc()
    fc.xgb_sim = None
    assert meter_hour_forecasts(fc, _series()).empty and meter_upcoming(fc, _series()).empty


@pytest.mark.brief("ML-003", "ML-008")
def test_features_are_matched_by_name_and_missing_ones_become_unknown():
    model = Recorder()
    feats = list(reversed(power_features(hourly_features(to_hourly(_series())).columns))) + ["not_a_feature"]
    meter_hour_forecasts(_fc(model, feats), _series())
    X = model.seen[-1]
    assert list(X.columns) == feats and X["not_a_feature"].isna().all()


@pytest.mark.brief("ML-004", "ML-005")
def test_wrong_types_and_infinities_reach_the_model_as_unknowns():
    model = Recorder(out=1.0)
    s = _series()
    s[TARGET] = s[TARGET].astype(object)
    s.loc[s.index[-5], TARGET] = np.inf
    s.loc[s.index[-6], TARGET] = "1.5"
    meter_hour_forecasts(_fc(model), s)
    X = model.seen[-1]
    assert np.isfinite(X.to_numpy(dtype=float)[~np.isnan(X.to_numpy(dtype=float))]).all()


@pytest.mark.brief("ML-009", "ML-010")
def test_non_finite_outputs_are_dropped_and_negative_ones_clipped():
    class Odd:
        def predict(self, X):
            out = np.full(len(X), -0.3)
            out[-1], out[-2] = np.nan, np.inf
            return out
    out = meter_hour_forecasts(_fc(Odd()), _series())
    assert len(out) > 0 and np.isfinite(out["predicted_kw"]).all() and (out["predicted_kw"] == 0.0).all()


@pytest.mark.brief("ML-006")
def test_missing_model_files_mean_no_live_forecast(tmp_path):
    fc = Forecaster.load(model_dir=str(tmp_path), use_lstm=False)
    assert getattr(fc, "xgb_sim", None) is None
    assert meter_hour_forecasts(fc, _series()).empty if hasattr(fc, "sim_features") else True


@pytest.mark.brief("ML-007")
def test_corrupted_lightgbm_file_is_refused(tmp_path):
    bad = tmp_path / "forecast_lgb_sim.txt.gz"
    with gzip.open(bad, "wt") as fh:
        fh.write("this is not a model")
    assert _lightgbm(str(bad)) is None
    (tmp_path / "plain.txt").write_text("garbage")
    assert _lightgbm(str(tmp_path / "plain.txt")) is None


@pytest.mark.brief("ML-007", "ML-020")
def test_corrupted_model_files_fall_back_without_fabricating(tmp_path):
    pytest.importorskip("xgboost")
    for f in os.listdir(MODEL_DIR):
        if f.endswith((".json", ".pkl", ".gz")):
            shutil.copy(os.path.join(MODEL_DIR, f), tmp_path / f)
    (tmp_path / "forecast_xgb_sim.json").write_text("{ broken")
    fc = Forecaster.load(model_dir=str(tmp_path), use_lstm=False)
    meta = json.load(open(tmp_path / "forecast_meta.json"))["simulated_meter"]
    if meta.get("model") != "lightgbm":
        assert fc.xgb_sim is None and meter_hour_forecasts(fc, _series()).empty


@pytest.mark.brief("ML-008", "ML-020")
def test_schema_check_catches_a_model_with_the_wrong_number_of_inputs():
    class Sized:
        n_features_in_ = 5

        def predict(self, X):
            return np.zeros(len(X))
    fc = _fc(Sized())
    assert "takes 5 inputs" in sim_schema_problem(fc)
    fc.sim_features = ["a", "b", "c", "d", "e"]
    assert sim_schema_problem(fc) is None
    fc.sim_features = ["a", "a", "c", "d", "e"]
    assert "duplicates" in sim_schema_problem(fc)
    assert model_feature_count(object()) is None


@pytest.mark.brief("ML-014")
def test_same_input_gives_the_same_forecast():
    fc = _fc(Recorder())
    a, b = meter_hour_forecasts(fc, _series()), meter_hour_forecasts(fc, _series())
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.brief("ML-015")
def test_metrics_match_their_definitions():
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rng = np.random.default_rng(1)
    y, p = rng.random(200), rng.random(200)
    s = M._scores(y, p)
    assert s["mae_kw"] == pytest.approx(mean_absolute_error(y, p), abs=1e-4)
    assert s["rmse_kw"] == pytest.approx(np.sqrt(mean_squared_error(y, p)), abs=1e-4)
    assert s["r2"] == pytest.approx(r2_score(y, p), abs=1e-4)
    y2 = np.append(y, np.nan)
    assert M._mae(y2, np.append(p, 1.0)) == pytest.approx(mean_absolute_error(y, p))   # NaN rows ignored


@pytest.mark.brief("ML-016")
def test_train_validation_and_test_homes_never_overlap_and_features_look_back_only():
    tr, va, te = set(sd.TRAIN_SEEDS), set(sd.VALIDATION_SEEDS), set(sd.TEST_SEEDS)
    assert not (tr & va) and not (tr & te) and not (va & te)
    s = _series()
    X = hourly_features(to_hourly(s))
    future = to_hourly(s).copy()
    future.iloc[-1, future.columns.get_loc(TARGET)] += 100.0          # change only the newest hour
    X2 = hourly_features(future)
    # Rows before the change must not move: no feature looks ahead in time.
    pd.testing.assert_frame_equal(X.iloc[:-1], X2.iloc[:-1])


@pytest.mark.brief("ML-018")
def test_forecast_horizon_is_the_next_hours_in_order():
    s = _series()
    up = meter_upcoming(_fc(Recorder()), s, steps=3)
    last = s["datetime"].iloc[-1]
    assert list(up["target_hour"]) == [last + pd.Timedelta(hours=k) for k in (1, 2, 3)]


@pytest.mark.brief("ML-001", "ML-015", "ML-020", "SMOKE-003")
def test_real_model_loads_and_beats_simple_baselines_on_unseen_homes(metric):
    """Real model: fresh homes (seeds never used in training), next-hour error vs two baselines."""
    fc, meta = _real()
    assert sim_schema_problem(fc) is None
    X, y, _ = sd.build(range(7000, 7004), days=30, context=bool(meta.get("context_features")))
    X = X.reindex(columns=fc.sim_features)
    pred = np.maximum(fc.xgb_sim.predict(X), 0.0)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    mae = M._mae(y, pred)
    persistence = M._mae(y, X["lag_0"])
    same_hour = M._mae(y, X["same_hour_mean7"])
    rmse = float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))
    metric("unseen_homes", {"homes": "seeds 7000-7003, 30 days each", "hours": int(len(y)),
                            "model": meta.get("model"), "mae_kw": round(mae, 4), "rmse_kw": round(rmse, 4),
                            "r2": round(M._r2(y, pred), 4), "persistence_mae_kw": round(persistence, 4),
                            "same_hour_mean7_mae_kw": round(same_hour, 4)})
    assert mae < min(persistence, same_hour)
    assert mae < 0.2                                    # report: 0.108 kWh/h on 30 test homes


@pytest.mark.brief("ML-020", "SMOKE-003")
def test_real_model_files_agree_with_their_metadata():
    fc, meta = _real()
    from forecast import _LightGBMModel
    assert isinstance(fc.xgb_sim, _LightGBMModel) == (meta.get("model") == "lightgbm")
    assert model_feature_count(fc.xgb_sim) == len(meta["features"])
    assert meta["comparison"]["chosen"]["mae_kw"] <= min(meta["comparison"]["baseline_persistence"]["mae_kw"],
                                                         meta["comparison"]["baseline_same_hour_mean_7_days"]["mae_kw"])


# ------------------------------------------------------------------ optimisation
@pytest.fixture(scope="module")
def home():
    pred = _detected(days=28, seed=7)
    return pred, extract_sessions(pred)


def _prefs(**kw):
    return HouseholdPrefs(min_monthly_saving_rs=0, max_suggestions=10, **kw)


@pytest.mark.brief("OPT-001", "OPT-009", "OPT-018", "OPT-019")
def test_plan_completes_every_move_is_feasible_and_figures_add_up(home, metric):
    pred, sessions = home
    result = recommend(pred, sessions, Tariff(), _prefs())
    v = validate_plan(pred, sessions, result["suggestions"], Tariff(), _prefs())
    assert v["all_constraints_ok"] and all(c["ok"] for c in v["checks"])
    assert v["savings_label"] == "estimate"
    metric("plan_seed7_28d", {k: v[k] for k in ("baseline", "optimised", "energy_saving_kwh_month",
                                                "energy_saving_pct", "cost_saving_month", "cost_saving_pct",
                                                "peak_change_kw", "all_constraints_ok", "load_cap_violations")}
           | {"moves_checked": len(v["checks"]), "suggestions": len(result["suggestions"])})
    assert v["energy_saving_kwh_month"] == pytest.approx(v["baseline"]["kwh_month"] - v["optimised"]["kwh_month"], abs=0.02)
    assert v["cost_saving_month"] == pytest.approx(v["baseline"]["cost_month"] - v["optimised"]["cost_month"], abs=0.02)
    for s in result["suggestions"]:
        for mv in s.get("moves", []):
            assert 0 <= mv["to"] < 1440 and mv["dur"] > 0                 # an actual time on a real day


@pytest.mark.brief("OPT-002", "OPT-014")
def test_no_flexible_appliances_means_no_shift_and_no_invented_saving(home):
    pred, sessions = home
    fixed = pred.copy()
    for key in ("washing_machine", "geyser", "water_pump", "dishwasher"):
        if f"{key}_kw" in fixed:
            fixed[f"{key}_kw"], fixed[f"{key}_on"] = 0.0, False
    result = recommend(fixed, extract_sessions(fixed), Tariff(), _prefs())
    assert not [s for s in result["suggestions"] if s["kind"] == "shift"]
    v = validate_plan(fixed, extract_sessions(fixed), [], Tariff(), _prefs())
    assert v["cost_saving_month"] == 0 and v["energy_saving_kwh_month"] == 0


@pytest.mark.brief("OPT-003", "OPT-007")
def test_user_limits_are_respected(home):
    pred, sessions = home
    # No shifting allowed at all for the washing machine: it must stay where it is.
    prefs = _prefs(overrides={"washing_machine": {"max_advance_min": 0, "max_delay_min": 0}})
    result = recommend(pred, sessions, Tariff(), prefs)
    for s in result["suggestions"]:
        for mv in s.get("moves", []):
            if s.get("appliance") == "washing_machine" or mv.get("appliance") == "washing_machine":
                assert mv["to"] == mv["from"]
    v = validate_plan(pred, sessions, result["suggestions"], Tariff(), prefs)
    assert v["all_constraints_ok"]


@pytest.mark.brief("OPT-004", "OPT-010")
def test_a_run_longer_than_the_allowed_window_has_no_start_time():
    prof = DEFAULT_PROFILES["washing_machine"]
    window = (prof.allowed_hours[1] - prof.allowed_hours[0]) * 60
    assert feasible_starts(prof, window + 15, 600, HouseholdPrefs()) == []
    assert feasible_starts(prof, 60, 600, HouseholdPrefs())


@pytest.mark.brief("OPT-006")
def test_higher_power_raises_the_cost_of_the_same_run():
    t = Tariff()
    assert t.run_cost(600, 60, 2.0) == pytest.approx(2 * t.run_cost(600, 60, 1.0))


@pytest.mark.brief("OPT-008", "OPT-012", "OPT-013", "OPT-019")
def test_savings_formula_horizons_zero_baseline_and_worse_plans():
    assert estimated_savings(100.0, 90.0, 720, 720) == {"saving_kwh": 10.0, "saving_pct": 10.0, "uses_more": False,
                                                       "label": "estimate"}
    with pytest.raises(ValueError):
        estimated_savings(100.0, 90.0, 720, 24)                   # different horizons
    assert estimated_savings(0.0, 0.0, 24, 24)["saving_pct"] is None
    worse = estimated_savings(10.0, 12.0, 24, 24)
    assert worse["saving_kwh"] == -2.0 and worse["uses_more"] and worse["saving_pct"] == -20.0
    for bad in ((-1, 0, 1, 1), (float("nan"), 1, 1, 1), (1, float("inf"), 1, 1)):
        with pytest.raises(ValueError):
            estimated_savings(*bad)


@pytest.mark.brief("OPT-009")
def test_a_plan_that_breaks_a_hard_limit_is_flagged(home):
    pred, sessions = home
    day = pd.Timestamp(pred["datetime"].iloc[0]).normalize()
    bad = [{"kind": "shift", "appliance": "geyser", "monthly_saving_rs": 50,
            "moves": [{"day": day, "from": 6 * 60, "to": 14 * 60, "dur": 30, "kw": 2.0, "kwh": 1.0}]}]
    v = validate_plan(pred, sessions, bad, Tariff(), HouseholdPrefs())
    assert not v["all_constraints_ok"] and "moved later than this appliance allows" in v["checks"][0]["problems"]


@pytest.mark.brief("OPT-016")
def test_same_inputs_give_the_same_plan(home):
    pred, sessions = home
    a = recommend(pred, sessions, Tariff(), _prefs())
    b = recommend(pred, sessions, Tariff(), _prefs())
    assert json.dumps(a["suggestions"], default=str, sort_keys=True) == json.dumps(b["suggestions"], default=str,
                                                                                   sort_keys=True)


@pytest.mark.brief("OPT-017")
def test_planning_a_month_stays_within_a_time_limit(home, metric):
    pred, sessions = home
    t0 = time.perf_counter()
    recommend(pred, sessions, Tariff(), _prefs())
    took = time.perf_counter() - t0
    metric("recommend_28_days_s", round(took, 3))
    assert took < 20.0


@pytest.mark.brief("OPT-020")
def test_the_words_match_the_moves(home):
    pred, sessions = home
    result = recommend(pred, sessions, Tariff(), _prefs())
    shifts = [s for s in result["suggestions"] if s["kind"] == "shift" and s.get("moves")]
    for s in shifts:
        assert f"Rs. {s['monthly_saving_rs']:.0f}" in s["detail"] or f"Rs {s['monthly_saving_rs']:.0f}" in s["detail"] \
            or str(int(s["monthly_saving_rs"])) in s["detail"] + s.get("title", "")
