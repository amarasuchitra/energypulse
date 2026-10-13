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

Hourly household use is noisy (an appliance switching on changes the hour by kilowatts), so a
next-hour error of about {mae['hybrid']:.2f} kW is close to what the readings allow. The fair comparison is the
baseline: the forecast is **{h['improvement_over_persistence_pct']:.0f}% more accurate** than assuming the next hour
repeats this one.

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
- Appliances are planned together, biggest saving first: each accepted move is added to the day's
  load (90th percentile of everything else), and a time is offered only if the home stays under
  85% of its sanctioned load.
- Moves that are each too small to mention are offered together when they add up.
- On a flat tariff no timing advice is given, because moving use saves nothing.

## Limits

- One household, recorded 2006-2010, in France. Accuracy on other homes will differ; the app says so.
- The simulated meter used for appliance-level views and the daily brief is a simulation; its own
  forecast was backtested on simulated homes (about 11% daily error) and is labelled as simulated.

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
