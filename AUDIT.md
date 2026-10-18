# EnergyPulse — project audit (V4.5)

This report covers the audit requested in *"Complete Project Audit, AI Prediction, Optimization, and
Dashboard Improvement"*: what was found, what was changed, how it was verified, and what is still open.
Numbers come from `models/forecast_meta.json` (written by `python model.py`) and from the test suite;
nothing here is estimated by hand.

## 1. Audit findings (before this round)

| Area | Finding | Severity |
|---|---|---|
| Pipeline | Two separate data paths existed. The **UCI replay** used the trained XGBoost + LSTM forecaster; the **simulated meter** that the dashboard shows by default had no ML forecast at all, only a 14-day "usual for this hour" average. | High |
| Prediction | The UCI-trained model, applied to simulated homes, was *worse* than a same-hour average. It cannot be used for the live meter. | High |
| Optimisation | The scheduler built its "typical day" from history that included today's not-yet-happened minutes from the simulator (look-ahead). | High |
| Optimisation | Planning against a 90th-percentile profile allowed real days to exceed the load cap (worst case 3.86 kW against a 2.55 kW cap). | Medium |
| Optimisation | No check that a plan respected constraints; savings were not split into "money from moving use" vs "energy from tips". | Medium |
| Alerts | No comparison of actual vs predicted use; no overconsumption alert of any kind. | High |
| Persistence | Appliances were saved per household, but quantity and hours-per-day were not stored, and nothing validated what was written. | Medium |
| Dashboard | The whole page re-ran every 60 s; nothing updated faster. A failure showed the raw exception text in an expander. | Medium |
| Logging | Errors went to `print`; one path printed an email address. | Low |

## 2. Pipeline after the changes

```
simulated minute readings (meter_sim, user's own appliances/habits)
  -> hourly means; hours with < 45 readings are dropped (stale/partial data)
  -> next-hour forecast (simulated-meter XGBoost + 90% band)      forecast.meter_hour_forecasts
  -> forecast_log: forecast stored BEFORE its hour; actual filled once the hour ends   db.log_forecasts / record_actuals
  -> overconsumption check (actual vs forecast)                    monitoring.check_overconsumption
  -> alert: in-app + email if configured, honest status            monitoring.deliver
  -> scheduler plan on full past days only, validated              scheduler.recommend / validate_plan
  -> dashboard
```

Refresh intervals:

| What | Interval | Where |
|---|---|---|
| Live reading tile | 1 s (`st.fragment(run_every=1)`) | app.py `live_tile` |
| Today chart with forecast markers | 60 s | app.py `today_chart` |
| Next-hour forecast | once per hour; the page reruns when the clock hour changes | `_meter_forecast_cached` |
| Overconsumption check | once per hour per household in the session, plus a background job at :05 every hour | `run_meter_monitoring`, `scheduler_jobs.check_meter_usage` |
| Schedule plan | cached per day / appliance set | appliance_ui |

The forecast card shows the hour it is for, the 90% range and the time it was made ("made at HH:00").
Every stage logs through the `energypulse` logger (`ep_log.py`; level from `ENERGYPULSE_LOG_LEVEL`).

## 3. Appliance configuration persistence

- Stored in SQLite (`household_homes`), one row per household, written only through
  `DatabaseManager.save_home_details`, which runs `home_config.validate_home_details` first:
  ranges for people, home size, tariff, quantity (1–10), hours per day (0–24) and rated power; names
  required and trimmed; unknown fields dropped; duplicate catalogue items merged into one with a quantity.
- **Edit appliances** (Settings): quantity and hours-per-day inputs, usage level, remove, add, and
  **Save changes** / **Cancel**. Edits are a draft until saved; Cancel discards them. Save refuses
  invalid input with a readable message and clears the cached forecasts and plans, so the prediction
  and schedule immediately use the new setup.
- Quantity and hours change the simulated meter (`home_config.usage_factor`) and the yearly estimate.
- Each user's setup is keyed by their household id; tests confirm one household's edit does not touch
  another's and that the setup survives a new database connection (a new login).

## 4. Overconsumption alerts and notification channels

Rules (defaults; each household can change them under **Settings → Usage alerts**):

| Setting | Default |
|---|---|
| Above forecast by at least | 15% |
| and by at least (absolute) | 0.15 kWh in the hour |
| for consecutive hours | 2 |
| ignore readings older than | 3 h |

- A forecast below 0.05 kWh is treated as "near zero": the percentage is not computed (no divide-by-zero
  or absurd percentages); the absolute rule still applies.
- One alert per event: while use stays high no repeat is sent; once an hour is back within the limit the
  alert re-arms.
- The alert shows actual vs forecast, the excess in units and percent, the appliance that rose most
  compared with its usual for those hours, and a recommendation for that appliance. It appears as a
  banner on the dashboard and in the notification history.
- **Channels and their real status:** every alert is stored *in-app*. Email is attempted only when SMTP
  or SendGrid is configured (see `.env.example`); the stored status is `sent` only when the mail server
  accepted it, otherwise `failed` with the error, or `in-app` when email is not configured.
  **In this build no email was sent during testing; email needs your SMTP secrets.** SMS is not
  implemented and is not claimed.

## 5. Prediction evaluation

See `MODEL_REPORT.md` for the full tables. Summary:

- **UCI home (historical replay):** time-ordered train/validation/test split, features use only hours up to
  *t* for target *t + 1* (tested). Hybrid XGBoost + LSTM test MAE 0.285 kW, RMSE 0.445, R² 0.642;
  28% lower error than "next hour = this hour". Random forest (MAE 0.315, R² 0.619) and ridge regression
  (0.354, 0.556) were trained on the same data for comparison and lose to XGBoost, so they are not used.
- **Simulated meter (live dashboard):** separate XGBoost model trained on 80 simulated homes, early-stopped
  on 20 others and tested on 30 unseen homes (38,130 hours): **MAE 0.109 kW, RMSE 0.210, R² 0.744**, against
  0.166 kW / R² 0.562 for the best baseline (same-hour mean of the last 7 days), i.e. **34% lower error**.
  The 90% range covered 88.2% of test hours (slightly under target on new homes).
  These are results on simulated data and are labelled as such in the app.
- **90% range:** conformally calibrated; UCI test coverage 90.0%.
- **Bill forecast:** 30-day total error 5.0% (household model) vs 15.2% for the last-7-days average.

Limitations: one real home (France, 2006–2010) for the real-data model; the live meter is a simulation, so
its accuracy says how well the model learns the simulator, not a real home.

## 6. Optimisation evaluation

The scheduler plans the way a sensible person would (V4.6):

- **One routine time per habit.** Runs are grouped by time of day (a morning and an evening geyser are
  two habits), and each habit gets one fixed time to remember, not a different time every day.
- **The smallest change that keeps most of the saving.** Among the times that keep at least 90% of the
  best saving, the one closest to the current habit is chosen. Changes under 15 minutes or under
  Rs. 0.25 a run are not suggested.
- **Directions that make sense.** A geyser and a water pump only move *earlier* (hot water and a full
  tank must be ready when needed); a geyser at most 2 hours earlier, with its standing heat loss
  (about 0.025 kWh an hour, an estimate) charged against the saving. A washing machine moves at most
  4 hours either way, inside waking hours and outside quiet hours.
- **Advice written like an energy advisor:** what the habit is now, what it costs, the new time, why it is
  safe, the saving per month and per year, and a practical tip (a geyser timer, delay start).

`scheduler.validate_plan` re-checks every plan and is shown on **Optimize → Save** as "Plan check":
baseline vs optimised energy, cost and average daily peak; every moved run against allowed hours, quiet
hours, how far earlier or later it may move, its category, and the household load cap (85% of sanctioned
load) on every day of history. Moving a run keeps its units the same, apart from an early geyser's small
heat loss, which is counted. Energy savings come only from the efficiency tips and are labelled
"estimate from BEE guidance, not simulated".

**Measured on 5 simulated homes, 28 days each, timing advice only:**

| | Previous planner | Human-like planner |
|---|---|---|
| Saving from timing | Rs. 26 a home a month (0.76% of the bill) | Rs. 8 a home a month (0.23%) |
| Hot water or pumping moved to *after* it is needed | 137 of 243 moves | 0 |
| Average move | 4.9 hours | 1.4 hours |
| Largest move | 10.4 hours | 3.4 hours |
| Different times to remember | 146 | 12 (about 2 a home) |
| Constraint violations | 0 | 0 |

The previous planner's larger figure came mostly from moves nobody would follow. Timing alone saves
little on this tariff (evening peak +20%, solar hours −20%); the larger savings are the efficiency tips
(the AC set-point and sleep-timer tips: an estimated Rs. 181–283 a month on three of these homes). An earlier version
of this report quoted "4.3–4.8% from timing alone"; that figure could not be reproduced and is withdrawn.
On a flat tariff no timing advice is given because moving use saves nothing.

## 7. Error handling

- The dashboard never shows exception text: failures are logged with traceback (`log.exception`) and the
  user sees a short message with *Retry* / *Go home*. `client.showErrorDetails = "none"` is set in
  `.streamlit/config.toml` so Streamlit itself does not print tracebacks either.
- Non-critical sections (forecast card, bill forecast, alerts, plan check) fail on their own with a
  friendly note instead of taking the page down.
- A missing model file means "forecast not available", never a made-up number.
- Hours with too few readings are skipped; readings older than the freshness limit are not used for alerts.
- Logs carry household ids, not email addresses or secrets.

## 8. Tests

New: `test_monitoring.py` (12), `test_home_config.py` (12), `test_forecast_sim.py` (5);
updated: `test_optimization.py` (9). Mapping to the spec's tests:

| Spec test | Covered by |
|---|---|
| A. Persistence (save, re-login, isolation) | `test_home_config.py::test_saved_setup_survives_a_new_connection_and_is_isolated` |
| B. Normal use raises nothing | `test_monitoring.py::test_normal_use_raises_nothing`, small-excess test |
| C. Sustained overconsumption alerts once, re-arms | `test_monitoring.py` sustained, single-spike, de-duplication and recovery tests |
| D. Notification failure is reported, not hidden | `test_monitoring.py::test_delivery_failure_is_recorded_not_raised` |
| E. Prediction has no look-ahead | `test_forecast_sim.py::test_forecasts_only_use_earlier_hours`, `test_prediction_pipeline.py` |

The full run is recorded in the delivery notes (see `README.md`, *Running the tests*).

## 9. Files changed in this round

New: `ep_log.py`, `home_config.py`, `monitoring.py`, `sim_dataset.py`, `test_monitoring.py`,
`test_home_config.py`, `test_forecast_sim.py`, `AUDIT.md`, `models/forecast_xgb_sim.json`,
`models/forecast_q05_sim.pkl`, `models/forecast_q95_sim.pkl`.

Changed: `app.py`, `appliance_ui.py`, `catalog.py`, `db.py`, `forecast.py`, `meter_source.py`, `model.py`,
`model_report.py`, `MODEL_REPORT.md`, `scheduler.py`, `scheduler_jobs.py`, `test_optimization.py`,
`.streamlit/config.toml`, `.github/workflows/ci.yml`, `.gitignore`, `README.md`, `models/forecast_meta.json`
and the retrained model files.

## 10. Still open

- Email delivery needs your SMTP (or SendGrid) secrets; until then alerts are in-app only.
- The live meter is simulated. Connecting a real meter (the meter API already exists) would let the same
  forecast log and alerts run on real readings; the simulated-meter model would then need retraining on
  real data.
- Changes are committed locally; they are not on GitHub until the repository is pushed from a linked account.
