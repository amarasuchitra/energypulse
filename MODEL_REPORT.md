# EnergyPulse forecasting: model report

Generated from `models/forecast_meta.json` (training run 2026-10-14T00:33:54).
Regenerate both with `python model.py && python model_report.py`.

## Data and evaluation

- **Data:** UCI *Individual household electric power consumption* (one home in Sceaux, France), 1-minute
  readings averaged to hours; missing hours are kept as gaps, never filled with neighbours.
- **Split, in time order:** train until 2009-06-19 12:00:00, validation until 2010-02-05 02:00:00,
  test until 2010-11-26 20:00:00 (6,832 test hours, mean 1.02 kW).
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
| Hybrid (XGBoost + LSTM) | 0.285 | +28.0% |
| XGBoost | 0.288 | +27.5% |
| LSTM | 0.309 | +22.2% |
| XGBoost, power readings only | 0.305 | +23.1% |
| Baseline: next hour = this hour | 0.397 | +0.0% |
| Baseline: mean of same hour, last 7 days | 0.445 | -12.2% |
| Baseline: same hour yesterday | 0.534 | -34.7% |
| Baseline: same hour last week | 0.566 | -42.7% |

- **Blend:** XGBoost 80% + LSTM 20%, chosen on validation error.
- **90% range:** 5% and 95% quantile models, conformally calibrated on validation.
  Test coverage **90.0%** (target 90%), average width 1.38 kW.
  Power-only variant coverage 90.8%.
- Models trained on the training period only scored 0.289 kW on the same test
  hours; refitting on train + validation gave 0.285 kW.

Hourly household use is noisy (an appliance switching on changes the hour by kilowatts), so a
next-hour error of about 0.29 kW is close to what the readings allow. The fair comparison is the
baseline: the forecast is **28% more accurate** than assuming the next hour
repeats this one.

## Bill forecast (energy over the next 30 days)

A gradient-boosted model forecasts each of the next 62 days from the last 90 days of daily energy
(levels, trends, weekday pattern); the daily totals are spread over hours with the home's own
weekday/hour profile. Error is the mean absolute percentage error of the 30-day total, over
198 forecast start days in the test period.

| Method | 30-day total error | Daily error (kWh) |
|---|---|---|
| Household model (with time of year) | **5.0%** | 4.13 |
| Generic model (any home, no time of year) | **9.6%** | 4.53 |
| Baseline: average of the last 7 days | 15.2% | 5.55 |
| Baseline: average of the last 28 days | 17.5% | |

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

Software: python 3.11.8, sklearn 1.9.1, xgboost 3.2.0, tensorflow 2.21.0. Training took 756 s.
