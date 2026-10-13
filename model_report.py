"""
Write MODEL_REPORT.md from models/forecast_meta.json (produced by model.py).

    python model_report.py
"""
import json
import os


def _pct(a, b):
    return (1 - a / b) * 100 if b else 0.0


def render(meta: dict) -> str:
    h = meta["hourly"]
    mae = h["mae_kw"]
    sp = meta["split"]
    blend = meta["blend"]
    pw = meta.get("power_only", {})
    ds, dg = meta["daily"]["daily_seasonal"]["test"], meta["daily"]["daily_generic"]["test"]
    rows = [
        ("Hybrid (XGBoost + LSTM)", mae["hybrid"]),
        ("XGBoost", mae["xgboost"]),
        ("LSTM", mae["lstm"]),
        ("XGBoost, power readings only", pw.get("mae_kw")),
        ("Baseline: next hour = this hour", mae["persistence"]),
        ("Baseline: mean of same hour, last 7 days", mae["same_hour_mean_7_days"]),
        ("Baseline: same hour yesterday", mae["same_hour_yesterday"]),
        ("Baseline: same hour last week", mae["same_hour_last_week"]),
    ]
    table = "\n".join(
        f"| {name} | {v:.3f} | {_pct(v, mae['persistence']):+.1f}% |" if v is not None else f"| {name} | n/a | |"
        for name, v in rows)
    names = {"hybrid_xgboost_lstm": "Hybrid (XGBoost + LSTM)", "xgboost": "XGBoost", "lstm": "LSTM",
             "random_forest": "Random forest", "linear_ridge": "Linear (ridge) regression",
             "baseline_persistence": "Baseline: next hour = this hour",
             "baseline_same_hour_mean_7_days": "Baseline: mean of same hour, last 7 days"}

    def score_table(comp):
        return "\n".join(f"| {names.get(k, k)} | {v['mae_kw']:.3f} | {v['rmse_kw']:.3f} | {v['r2']:.3f} |"
                         for k, v in comp.items() if v)
    comparison = score_table(h.get("comparison", {})) or "| (not recorded in this training run) | | | |"
    sim = meta.get("simulated_meter")
    if sim:
        sc = sim["comparison"]
        sim_section = f"""## Simulated meter: next-hour model

The live dashboard reads a **simulated** smart meter (an Indian home built from the user's own appliance
list), not a physical meter. The UCI model above does not transfer to it, so it has its own model:
XGBoost on power-only features (the same lags, same-hour history and rolling statistics; no sub-meter
readings), trained on {sim['homes']['train']} simulated homes, early-stopped on {sim['homes']['validation']}
other homes and tested on {sim['homes']['test']} homes it never saw (different appliances, habits, sizes,
cities and seasons; {sim['test_hours']:,} test hours, mean {sim['test_mean_kw']:.2f} kW).

| Model (unseen simulated homes) | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
{score_table(sc)}

- **{sim['improvement_over_best_baseline_pct']:.0f}% lower error** than the better of the two baselines.
- 90% range coverage on the test homes: **{sim['interval']['test_coverage']:.1%}** (target 90%; calibrated on the
  validation homes, so new homes come out slightly below target), average width {sim['interval']['mean_width_kw']:.2f} kW.
- These are results on simulated data. They show the model learns the simulator's patterns; they are not
  a measurement of accuracy on a real home's meter.
- The dashboard logs each forecast before its hour happens and fills in the actual afterwards
  (`forecast_log` table), so live accuracy and overconsumption alerts use the forecast that was really made.
"""
    else:
        sim_section = ""
    versions = ", ".join(f"{k} {v}" for k, v in meta.get("versions", {}).items())
    return f"""# EnergyPulse forecasting: model report

Generated from `models/forecast_meta.json` (training run {meta['created']}).
Regenerate both with `python model.py && python model_report.py`.

## Data and evaluation

- **Data:** UCI *Individual household electric power consumption* (one home in Sceaux, France), 1-minute
  readings averaged to hours; missing hours are kept as gaps, never filled with neighbours.
- **Split, in time order:** train until {sp['train_until']}, validation until {sp['validation_until']},
  test until {sp['test_until']} ({h['test_rows']:,} test hours, mean {h['test_mean_kw']:.2f} kW).
- **Rules:** every choice (model settings, early stopping, blend weight, interval calibration) is made on
  the validation period. The test period is used once, to report the numbers below. The final models are
  refit on train + validation with those choices fixed.
- **No look-ahead:** the features for hour *t + 1* use only readings up to hour *t*
  (checked by `test_prediction_pipeline.py::test_features_use_nothing_after_the_current_hour`).

## Next-hour forecast

Inputs: the last 24 hourly readings and readings 2-7 days back, the same clock hour on each of the
last 7 days, rolling means/spread/peaks over 3 h to 7 days, the latest sub-meter, voltage, current and
reactive-power readings with their daily averages, and the hour and weekday being forecast.
XGBoost is trained on absolute error; the LSTM reads the last 48 hours.

| Model (test period) | Mean absolute error (kW) | vs "next hour = this hour" |
|---|---|---|
{table}

- **Blend:** XGBoost {blend['w_xgb']:.0%} + LSTM {blend['w_lstm']:.0%}, chosen on validation error.
- **90% range:** 5% and 95% quantile models, conformally calibrated on validation.
  Test coverage **{h['interval']['test_coverage']:.1%}** (target 90%), average width {h['interval']['mean_width_kw']:.2f} kW.
  Power-only variant coverage {pw.get('interval', {}).get('test_coverage', 0):.1%}.
- Models trained on the training period only scored {h['hybrid_mae_train_only_models_kw']:.3f} kW on the same test
  hours; refitting on train + validation gave {mae['hybrid']:.3f} kW.

All models, same test hours (R² is the share of hour-to-hour variation explained):

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
{comparison}

Random forest and ridge regression use the same features and training data; they are recorded for
comparison and are not used by the app.

Hourly household use is noisy (an appliance switching on changes the hour by kilowatts), so a
next-hour error of about {mae['hybrid']:.2f} kW is close to what the readings allow. The fair comparison is the
baseline: the forecast is **{h['improvement_over_persistence_pct']:.0f}% more accurate** than assuming the next hour
repeats this one.

{sim_section}
## Bill forecast (energy over the next 30 days)

A gradient-boosted model forecasts each of the next 62 days from the last 90 days of daily energy
(levels, trends, weekday pattern); the daily totals are spread over hours with the home's own
weekday/hour profile. Error is the mean absolute percentage error of the 30-day total, over
{ds['next_30_days']['origins']} forecast start days in the test period.

| Method | 30-day total error | Daily error (kWh) |
|---|---|---|
| Household model (with time of year) | **{ds['next_30_days']['model_total_error_pct']:.1f}%** | {ds['daily_mae_kwh']:.2f} |
| Generic model (any home, no time of year) | **{dg['next_30_days']['model_total_error_pct']:.1f}%** | {dg['daily_mae_kwh']:.2f} |
| Baseline: average of the last 7 days | {ds['next_30_days']['baseline_last7_total_error_pct']:.1f}% | {ds['daily_mae_kwh_baseline_last7']:.2f} |
| Baseline: average of the last 28 days | {ds['next_30_days']['baseline_last28_total_error_pct']:.1f}% | |

The household model is used for the bundled sample; uploaded data from other homes gets the generic
model, because this home's seasons (French winter heating) say nothing about another home's.

## Appliance scheduling (optimisation)

- For each shiftable appliance run, every allowed start time (15-minute steps) is costed against the
  time-of-day tariff, within the household's limits (quiet hours, allowed hours, how far ahead a geyser
  may heat).
- Appliances are planned together, biggest saving first, against each past day's actual load: a run is
  moved only to a time where the home would have stayed under 85% of its sanctioned load on that day,
  and each accepted move is added to that day's load before the next is planned.
- Every plan is re-checked (`scheduler.validate_plan`): baseline vs optimised energy, cost and peak, and
  every moved run against allowed hours, quiet hours, how far ahead it may run and the load cap. Moving a
  run saves money, not energy; energy savings from the efficiency tips are estimates from BEE guidance
  and are labelled as such.
- Moves that are each too small to mention are offered together when they add up.
- On a flat tariff no timing advice is given, because moving use saves nothing.

## Limits

- One household, recorded 2006-2010, in France. Accuracy on other homes will differ; the app says so.
- The live meter is simulated; its next-hour and daily figures are evaluated on simulated homes only
  (daily brief backtest: about 11% daily error) and the app labels them as simulated.

Software: {versions}. Training took {meta['training_seconds']} s.
"""


def main():
    with open(os.path.join("models", "forecast_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open("MODEL_REPORT.md", "w", encoding="utf-8") as fh:
        fh.write(render(meta))
    print("Wrote MODEL_REPORT.md")


if __name__ == "__main__":
    main()
