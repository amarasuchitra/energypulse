# EnergyPulse forecasting: model report

Generated from `models/forecast_meta.json` (training run 2026-10-14T01:33:35).
Regenerate both with `python backend/model.py && python backend/model_report.py`.

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

All models, same test hours (R² is the share of hour-to-hour variation explained):

| Model | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| Hybrid (XGBoost + LSTM) | 0.285 | 0.445 | 0.642 |
| XGBoost | 0.288 | 0.448 | 0.638 |
| LSTM | 0.309 | 0.477 | 0.591 |
| Random forest | 0.315 | 0.459 | 0.619 |
| Linear (ridge) regression | 0.353 | 0.495 | 0.556 |
| Baseline: next hour = this hour | 0.397 | 0.603 | 0.344 |
| Baseline: mean of same hour, last 7 days | 0.445 | 0.624 | 0.296 |

Random forest and ridge regression use the same features and training data; they are recorded for
comparison and are not used by the app.

Hourly household use is noisy (an appliance switching on changes the hour by kilowatts), so a
next-hour error of about 0.29 kW is close to what the readings allow. The fair comparison is the
baseline: the forecast is **28% more accurate** than assuming the next hour
repeats this one.

## Simulated meter: next-hour model

The live dashboard reads a **simulated** smart meter (an Indian home built from the user's own appliance
list), not a physical meter. The UCI model above does not transfer to it, so it has its own model:
LightGBM on power-only features (the same lags, same-hour history and rolling statistics; no sub-meter
readings) plus what the app knows before the hour: the day's forecast temperature, the number of people, home size and which appliances the home has. XGBoost and LightGBM were both trained; the one with the lower error on the validation
homes is used. Trained on 80 simulated homes, early-stopped on 20
other homes and tested on 30 homes it never saw (different appliances, habits, sizes,
cities and seasons; 38,130 test hours, mean 0.49 kW).

| Model (unseen simulated homes) | MAE (kW) | RMSE (kW) | R² |
|---|---|---|---|
| XGBoost | 0.108 | 0.208 | 0.749 |
| LightGBM (used by the app) | 0.108 | 0.207 | 0.752 |
| Baseline: next hour = this hour | 0.182 | 0.330 | 0.368 |
| Baseline: mean of same hour, last 7 days | 0.166 | 0.275 | 0.562 |

- **35% lower error** than the better of the two baselines.
- 90% range coverage on the test homes: **88.1%** (target 90%; calibrated on the
  validation homes, so new homes come out slightly below target), average width 0.47 kW.
- These are results on simulated data. They show the model learns the simulator's patterns; they are not
  a measurement of accuracy on a real home's meter.
- The dashboard logs each forecast before its hour happens and fills in the actual afterwards
  (`forecast_log` table), so live accuracy and overconsumption alerts use the forecast that was really made.

## Experiments: what else was tried

`experiments.py` (2026-10-18) tried LightGBM, XGBoost + LightGBM blends, calendar features
(French public holidays, time of year) on the real home, and weather and household context on the
simulated meter. Choices were made on validation error only; test error is shown for the record.

| Data | Model | Validation MAE (kW) | Test MAE (kW) |
|---|---|---|---|
| Real home | XGBoost, current features | 0.2985 | 0.2917 |
| Real home | LightGBM, current features | 0.2974 | 0.2922 |
| Real home | Blend 0.40 XGBoost + 0.60 LightGBM, current features | 0.2970 | 0.2911 |
| Real home | XGBoost, + holidays and time of year | 0.2996 | 0.2911 |
| Real home | LightGBM, + holidays and time of year | 0.2971 | 0.2919 |
| Real home | Blend 0.10 XGBoost + 0.90 LightGBM, + holidays and time of year | 0.2971 | 0.2916 |
| Simulated meter | XGBoost, meter only (current) | 0.0977 | 0.1100 |
| Simulated meter | LightGBM, meter only (current) | 0.0962 | 0.1081 |
| Simulated meter | Blend 0.15 XGBoost + 0.85 LightGBM, meter only (current) | 0.0962 | 0.1081 |
| Simulated meter | XGBoost, + weather and home context | 0.0969 | 0.1088 |
| Simulated meter | LightGBM, + weather and home context | 0.0954 | 0.1077 |
| Simulated meter | Blend 0.15 XGBoost + 0.85 LightGBM, + weather and home context | 0.0953 | 0.1075 |
| Simulated meter | Baseline: persistence | 0.1704 | 0.1825 |
| Simulated meter | Baseline: same_hour_mean_7_days | 0.1445 | 0.1655 |

Kept: LightGBM with weather and household context for the simulated meter (validation 0.0954 vs 0.0977
for the previous XGBoost). Not kept: changes to the real-home model, whose best variant gained 0.5% on
validation and 0.2% on test, too little to justify a second model there. (For speed, the
simulated-meter XGBoost runs in this experiment used a learning rate of 0.05; the final training above
compares XGBoost at its usual settings with LightGBM.)

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

- The scheduler plans the way a sensible person would: one fixed time per daily habit (runs grouped by time
  of day), the smallest change that keeps 90% of the best saving, and only directions that make sense
  (a geyser or water pump only earlier, a geyser at most 2 hours earlier with its heat loss counted, a
  washing machine at most 4 hours either way in waking hours).
- Every allowed start time (15-minute steps) is costed against the time-of-day tariff. Appliances are
  planned together against each past day's actual load; a run moves only where the home stays under 85%
  of its sanctioned load that day.
- Every plan is re-checked (`scheduler.validate_plan`): baseline vs optimised energy, cost and peak, and
  every moved run against all limits. Moving a run saves money, not energy; efficiency-tip savings are
  estimates from BEE guidance and are labelled as such.
- Small changes are offered together when they add up; on a flat tariff no timing advice is given.
- See AUDIT.md section 6 for the measured comparison with the previous planner.

## Limits

- One household, recorded 2006-2010, in France. Accuracy on other homes will differ; the app says so.
- The live meter is simulated; its next-hour and daily figures are evaluated on simulated homes only
  (daily brief backtest: about 11% daily error) and the app labels them as simulated.

Software: python 3.11.8, sklearn 1.9.1, xgboost 3.2.0, tensorflow 2.21.0. Training took 1065 s.
