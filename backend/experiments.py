"""
Can the next-hour forecast be made more accurate?  Candidate changes are tried
here and kept only if they help.

Rules (the same as model.py): every choice is made on the validation data; the
test data is used only to report.  All variants are trained on the same rows,
so the comparison is fair.

    python backend/experiments.py   -> docs/experiments.json + printed table

Part A: the real home (UCI).  Calendar features (French public holidays, time
of year) and LightGBM, alone and blended with XGBoost.
Part B: the simulated meter.  Context the app already knows before the hour
(that day's forecast temperature, people, home size, which appliances) and
LightGBM.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import model as M
from forecast_features import hourly_training_set

SEED = 42
XGB_PARAMS = dict(max_depth=7, learning_rate=0.02, min_child_weight=30, subsample=0.7,
                  colsample_bytree=0.6, reg_lambda=5.0)
LGB_GRID = [
    dict(num_leaves=63, learning_rate=0.02, min_child_samples=40, subsample=0.7, subsample_freq=1,
         colsample_bytree=0.6, reg_lambda=5.0),
    dict(num_leaves=31, learning_rate=0.03, min_child_samples=20, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=1.0),
    dict(num_leaves=127, learning_rate=0.02, min_child_samples=80, subsample=0.7, subsample_freq=1,
         colsample_bytree=0.5, reg_lambda=10.0),
]


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------- calendar
def _easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def french_holidays(years) -> set:
    days = set()
    for y in years:
        for m, d in ((1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)):
            days.add(pd.Timestamp(year=y, month=m, day=d))
        e = _easter(y)
        for off in (1, 39, 50):                       # Easter Monday, Ascension, Whit Monday
            days.add(e + pd.Timedelta(days=off))
    return days


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Known in advance for the hour being forecast: holiday flags and time of year."""
    when = index + pd.Timedelta(hours=1)
    day = when.normalize()
    hol = french_holidays(range(day.year.min() - 1, day.year.max() + 2))
    is_hol = np.array([d in hol for d in day], float)
    nxt = np.array([(d + pd.Timedelta(days=1)) in hol for d in day], float)
    prv = np.array([(d - pd.Timedelta(days=1)) in hol for d in day], float)
    doy = when.dayofyear.to_numpy()
    return pd.DataFrame({
        "is_holiday": is_hol,
        "holiday_tomorrow": nxt,
        "holiday_yesterday": prv,
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)


# ----------------------------------------------------------------- models
def fit_xgb(Xtr, ytr, Xva, yva, params=XGB_PARAMS, lr=None):
    import xgboost as xgb
    p = dict(params)
    if lr:
        p["learning_rate"] = lr
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=6000, early_stopping_rounds=150,
                         tree_method="hist", random_state=SEED, n_jobs=-1, **p)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def fit_lgb(Xtr, ytr, Xva, yva, grid=LGB_GRID):
    import lightgbm as lgb
    best = None
    for params in grid:
        t = time.time()
        m = lgb.LGBMRegressor(objective="l1", n_estimators=8000, random_state=SEED, n_jobs=-1,
                              verbose=-1, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        mae = M._mae(yva, m.predict(Xva))
        log(f"    LightGBM {params} -> val MAE {mae:.4f} ({m.best_iteration_} trees, {time.time() - t:.0f}s)")
        if best is None or mae < best[0]:
            best = (mae, m, params)
    return best[1], best[2]


def best_weight(y, a, b):
    """Weight on `a` (0..1, steps of 0.05) that minimises validation MAE of w*a + (1-w)*b."""
    ws = np.round(np.arange(0, 1.0001, 0.05), 2)
    errs = [M._mae(y, w * a + (1 - w) * b) for w in ws]
    return float(ws[int(np.argmin(errs))])


def run_variants(name, Xtr, ytr, Xva, yva, Xte, yte, variants, xgb_lr=None):
    """variants: {label: column list}.  Returns rows of results (val + test) and test predictions."""
    rows, preds = [], {}
    for label, cols in variants.items():
        t = time.time()
        m = fit_xgb(Xtr[cols], ytr, Xva[cols], yva, lr=xgb_lr)
        pv, pt = np.maximum(m.predict(Xva[cols]), 0), np.maximum(m.predict(Xte[cols]), 0)
        rows.append({"part": name, "model": f"XGBoost, {label}", "val_mae": round(M._mae(yva, pv), 4),
                     **{f"test_{k}": v for k, v in M._scores(yte, pt).items()}})
        preds[("xgb", label)] = (pv, pt)
        log(f"  XGBoost, {label}: val {rows[-1]['val_mae']:.4f}  test {rows[-1]['test_mae_kw']:.4f} "
            f"({m.best_iteration} trees, {time.time() - t:.0f}s)")
        lg, lp = fit_lgb(Xtr[cols], ytr, Xva[cols], yva)
        pv2, pt2 = np.maximum(lg.predict(Xva[cols]), 0), np.maximum(lg.predict(Xte[cols]), 0)
        rows.append({"part": name, "model": f"LightGBM, {label}", "val_mae": round(M._mae(yva, pv2), 4),
                     "params": lp, **{f"test_{k}": v for k, v in M._scores(yte, pt2).items()}})
        log(f"  LightGBM, {label}: val {rows[-1]['val_mae']:.4f}  test {rows[-1]['test_mae_kw']:.4f}")
        w = best_weight(yva, pv, pv2)
        pvb, ptb = w * pv + (1 - w) * pv2, w * pt + (1 - w) * pt2
        rows.append({"part": name, "model": f"Blend {w:.2f} XGBoost + {1 - w:.2f} LightGBM, {label}",
                     "weight_xgb": w, "val_mae": round(M._mae(yva, pvb), 4),
                     **{f"test_{k}": v for k, v in M._scores(yte, ptb).items()}})
        log(f"  Blend (w_xgb={w:.2f}), {label}: val {rows[-1]['val_mae']:.4f}  test {rows[-1]['test_mae_kw']:.4f}")
    return rows


def part_uci():
    log("\n== Part A: real home (UCI), next hour ==")
    df = M.load_data()
    X, y, _ = hourly_training_set(df)
    X = pd.concat([X, calendar_features(X.index)], axis=1)
    n = len(X)
    a, b = int(n * M.TRAIN_FRAC), int(n * (M.TRAIN_FRAC + M.VAL_FRAC))
    base = [c for c in X.columns if c not in calendar_features(X.index[:1]).columns]
    cal = list(X.columns)
    log(f"  {n} rows; holidays in test period: {int(X['is_holiday'].iloc[b:].sum() / 24)} days")
    return run_variants("uci", X.iloc[:a], y.iloc[:a], X.iloc[a:b], y.iloc[a:b], X.iloc[b:], y.iloc[b:],
                        {"current features": base, "+ holidays and time of year": cal})


def part_sim():
    log("\n== Part B: simulated meter (unseen homes), next hour ==")
    import sim_dataset as sd
    t = time.time()
    Xtr, ytr, _ = sd.build(sd.TRAIN_SEEDS, context=True)
    Xva, yva, _ = sd.build(sd.VALIDATION_SEEDS, context=True)
    Xte, yte, _ = sd.build(sd.TEST_SEEDS, context=True)
    log(f"  built {len(ytr):,} / {len(yva):,} / {len(yte):,} hours in {time.time() - t:.0f}s")
    base = [c for c in Xtr.columns if c not in sd.CONTEXT_FEATURES]
    ctx = list(Xtr.columns)
    rows = run_variants("sim", Xtr, ytr, Xva, yva, Xte, yte,
                        {"meter only (current)": base, "+ weather and home context": ctx}, xgb_lr=0.05)
    for k, col in (("persistence", "lag_0"), ("same_hour_mean_7_days", "same_hour_mean7")):
        rows.append({"part": "sim", "model": f"Baseline: {k}", "val_mae": round(M._mae(yva, Xva[col]), 4),
                     **{f"test_{kk}": v for kk, v in M._scores(yte, Xte[col]).items()}})
    return rows


def main():
    started = time.time()
    results = part_uci() + part_sim()
    from paths import DOCS_DIR
    with open(os.path.join(DOCS_DIR, "experiments.json"), "w", encoding="utf-8") as fh:
        json.dump({"created": pd.Timestamp.now().isoformat(timespec="seconds"), "results": results}, fh, indent=1)
    log("\n== Summary (choose by validation; test shown for the record) ==")
    for r in results:
        log(f"  {r['part']:4s} {r['model'][:62]:62s} val {r['val_mae']:.4f}  test MAE {r['test_mae_kw']:.4f}  "
            f"RMSE {r['test_rmse_kw']:.4f}  R2 {r['test_r2']:.3f}")
    log(f"\nDone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
