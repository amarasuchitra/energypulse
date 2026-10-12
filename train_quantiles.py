"""Quantile models for the 5%-95% prediction band, calibrated by conformalized
quantile regression so that the stated coverage holds on unseen data."""
import json, pickle
import numpy as np, pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
FEATURES = ["hour","day_of_week","is_weekend","Global_reactive_power","Voltage","Global_intensity",
            "Sub_metering_1","Sub_metering_2","Sub_metering_3"]
TARGET = "Global_active_power"
NOMINAL = 0.90
df = pd.read_csv("data/cleaned_energy_data.csv", parse_dates=["datetime"]).sort_values("datetime")
df = df.dropna(subset=FEATURES+[TARGET]).reset_index(drop=True)
a, b = int(len(df)*0.70), int(len(df)*0.85)
fit, cal, test = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
def model(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=40, random_state=42)
models = {q: model(q).fit(fit[FEATURES], fit[TARGET]) for q in (0.05, 0.5, 0.95)}
def band(X):
    lo, hi = models[0.05].predict(X), models[0.95].predict(X)
    return np.minimum(lo, hi), np.maximum(lo, hi)
lo, hi = band(cal[FEATURES]); y = cal[TARGET].values
scores = np.maximum(lo - y, y - hi)
n = len(scores)
offset = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * NOMINAL) / n)))
lo, hi = band(test[FEATURES]); y = test[TARGET].values
raw_cov = float(np.mean((y >= lo) & (y <= hi)))
lo, hi = np.maximum(0, lo - offset), hi + offset
cov = float(np.mean((y >= lo) & (y <= hi)))
width = float(np.mean(hi - lo))
mae50 = float(np.mean(np.abs(y - models[0.5].predict(test[FEATURES]))))
print(f"offset={offset:.4f} raw_cov={raw_cov:.3f} calibrated_cov={cov:.3f} width={width:.4f} median_mae={mae50:.4f}")
for q, m in models.items():
    with open(f"models/quantile_model_{int(round(q*100))}.pkl", "wb") as f:
        pickle.dump(m, f)
json.dump({"quantiles": [0.05, 0.5, 0.95], "feature_cols": FEATURES, "target": TARGET,
           "nominal_coverage": NOMINAL, "conformal_offset_kw": round(offset, 5),
           "pi_coverage": round(cov, 4), "pi_coverage_uncalibrated": round(raw_cov, 4),
           "pi_mean_width_kw": round(width, 4), "median_mae_kw": round(mae50, 4),
           "split": "time-ordered: fit 70%, calibrate 15%, test last 15% of the sample data",
           "sklearn_version": sklearn.__version__,
           "model": "HistGradientBoostingRegressor(loss='quantile') + conformal calibration"},
          open("models/quantile_meta.json", "w"), indent=2)
