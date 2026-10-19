"""
Predicted versus actual use, and overconsumption alerts, for the simulated meter.

Pipeline for one household (run on the dashboard and by the hourly job):

    simulated 1-minute meter (meter_sim)  ->  validation (readings.clean_readings: missing, negative,
    outlier, duplicate and out-of-order readings; then hours with < 45 good readings are dropped)
    ->  hourly kWh  ->  next-hour model (forecast.meter_hour_forecasts, one call for all hours)
    ->  forecast_log table (one forecast per hour, made before the hour, never overwritten)
    ->  actual kWh recorded when the hour has finished
    ->  check_overconsumption (threshold, minimum excess, consecutive hours, freshness)
    ->  one alert per event (stored in notifications_log; emailed only if email is set up)

Units: one row per clock hour; average kW over an hour equals kWh in that hour.
Everything here is simulated - the app never presents it as a physical meter.
"""
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from appliance_profiles import DEFAULT_PROFILES
from ep_log import get_logger
from forecast import meter_hour_forecasts
from forecast_features import TARGET
from readings import clean_readings
from sim_dataset import minute_to_hourly

log = get_logger("monitoring")

SOURCE = "simulated_meter"
ALERT_TYPE = "overconsumption"
STATE_KEY = "overuse_state"
SETTINGS_KEY = "alert_settings"
NEAR_ZERO_KWH = 0.05          # below this a percentage would be misleading


@dataclass
class AlertSettings:
    enabled: bool = True
    threshold_pct: float = 15.0        # actual must exceed the forecast by more than this ...
    min_excess_kwh: float = 0.15       # ... and by at least this many units in the hour
    consecutive_hours: int = 2         # for this many hours in a row
    max_age_hours: float = 3.0         # the newest of them finished within this many hours

    @classmethod
    def clean(cls, data: Optional[dict]) -> "AlertSettings":
        s = cls()
        if not isinstance(data, dict):
            return s

        def num(key, lo, hi, cast=float):
            try:
                v = cast(data.get(key, getattr(s, key)))
                if v != v:
                    raise ValueError
                return min(hi, max(lo, v))
            except (TypeError, ValueError):
                return getattr(s, key)
        return cls(enabled=bool(data.get("enabled", True)),
                   threshold_pct=num("threshold_pct", 5.0, 200.0),
                   min_excess_kwh=num("min_excess_kwh", 0.05, 5.0),
                   consecutive_hours=num("consecutive_hours", 1, 6, int),
                   max_age_hours=num("max_age_hours", 1.0, 24.0))


# ----------------------------------------------------------------- pure logic
def compare(row_pred: float, row_actual: float, s: AlertSettings) -> dict:
    """Difference between one hour's actual and predicted units, with a safe percentage."""
    pred, actual = float(row_pred), float(row_actual)
    diff = actual - pred
    pct = None if pred < NEAR_ZERO_KWH else diff / pred * 100.0
    over = diff >= s.min_excess_kwh and (pct is None or pct > s.threshold_pct)
    return {"predicted_kwh": pred, "actual_kwh": actual, "excess_kwh": diff, "excess_pct": pct, "over": over}


def check_overconsumption(log_df: pd.DataFrame, settings: AlertSettings, now: pd.Timestamp) -> dict:
    """
    Decide from the forecast log whether use is currently above forecast.

    Returns {"status": "over" | "ok" | "insufficient", "hours": [...compare rows...]}:
      over          the last `consecutive_hours` finished hours, back to back, were all over,
                    and the newest finished no more than `max_age_hours` ago
      ok            the newest finished hour is not over (an open event has recovered)
      insufficient  not enough recent, finished hours to judge
    """
    done = log_df.dropna(subset=["actual_kwh"]).sort_values("target_hour") if log_df is not None else pd.DataFrame()
    if done.empty:
        return {"status": "insufficient", "hours": []}
    newest_end = pd.Timestamp(done["target_hour"].iloc[-1]) + pd.Timedelta(hours=1)
    if now - newest_end > pd.Timedelta(hours=settings.max_age_hours):
        return {"status": "insufficient", "hours": []}          # stale readings never raise an alert
    tail = done.tail(settings.consecutive_hours)
    rows = [dict(compare(r.predicted_kwh, r.actual_kwh, settings), target_hour=pd.Timestamp(r.target_hour))
            for r in tail.itertuples()]
    if not rows[-1]["over"]:
        return {"status": "ok", "hours": rows}
    if len(rows) < settings.consecutive_hours:
        return {"status": "insufficient", "hours": rows}
    hours = [r["target_hour"] for r in rows]
    back_to_back = all(b - a == pd.Timedelta(hours=1) for a, b in zip(hours, hours[1:]))
    if back_to_back and all(r["over"] for r in rows):
        return {"status": "over", "hours": rows}
    return {"status": "insufficient", "hours": rows}


def next_state(state: Optional[dict], result: dict) -> tuple:
    """
    (new_state, should_alert).  One alert per event: a new alert only after the
    previous event has recovered (a finished hour back within the threshold).
    """
    state = dict(state or {"open": False})
    if result["status"] == "over":
        if state.get("open"):
            return state, False                                   # same event: suppressed
        start = result["hours"][0]["target_hour"]
        return {"open": True, "since": str(start), "last": str(result["hours"][-1]["target_hour"])}, True
    if result["status"] == "ok" and state.get("open"):
        return {"open": False, "recovered_at": str(result["hours"][-1]["target_hour"])}, False
    return state, False


RECOMMENDATIONS = {
    "ac": "Raise the AC set temperature by a degree or two, and switch it off in rooms nobody is using.",
    "geyser": "Check the water heater has not been left on; switch it off once the water is hot.",
    "water_pump": "Check the water pump is not running dry or the overhead tank overflowing.",
    "washing_machine": "Run the washing machine with full loads, at a cooler setting where possible.",
    "fridge": "Check the fridge door is shut and its seal is not worn; avoid putting hot food in.",
    "microwave": "Use the microwave in shorter runs; it draws a lot of power while on.",
}
GENERIC_ADVICE = "Switch off lights, fans and devices that are not in use."


def attribute(minute_df: pd.DataFrame, hours: List[pd.Timestamp], owned) -> Optional[dict]:
    """
    Which appliance used most above its own usual amount in the event hours
    (usual = the same clock hours on the previous 14 days).
    """
    if minute_df is None or minute_df.empty or not hours:
        return None
    dts = pd.to_datetime(minute_df["datetime"])
    best = None
    for key in list(owned) + ["other"]:
        col = f"{key}_kw"
        if col not in minute_df.columns:
            continue
        hourly = minute_df[col].astype(float).groupby(dts.dt.floor("h")).mean()
        now_kwh, usual_kwh = 0.0, 0.0
        for hr in hours:
            now_kwh += float(hourly.get(hr, 0.0))
            past = [hourly.get(hr - pd.Timedelta(days=d)) for d in range(1, 15)]
            past = [p for p in past if p is not None and not pd.isna(p)]
            usual_kwh += float(np.mean(past)) if past else 0.0
        excess = now_kwh - usual_kwh
        if best is None or excess > best["excess_kwh"]:
            best = {"key": key, "excess_kwh": excess, "kwh": now_kwh, "usual_kwh": usual_kwh}
    if best is None or best["excess_kwh"] < 0.05:
        return None
    best["name"] = DEFAULT_PROFILES[best["key"]].name if best["key"] in DEFAULT_PROFILES else "Lights, fans and other devices"
    best["advice"] = RECOMMENDATIONS.get(best["key"], GENERIC_ADVICE)
    return best


def alert_text(result: dict, culprit: Optional[dict]) -> tuple:
    rows = result["hours"]
    actual = sum(r["actual_kwh"] for r in rows)
    pred = sum(r["predicted_kwh"] for r in rows)
    pct = (actual - pred) / pred * 100.0 if pred >= NEAR_ZERO_KWH else None
    start, end = rows[0]["target_hour"], rows[-1]["target_hour"] + pd.Timedelta(hours=1)
    span = f"{start:%H:%M}-{end:%H:%M}"
    subject = "High energy use detected"
    body = (f"Between {span} your home used {actual:.2f} units (kWh), against a forecast of "
            f"{pred:.2f} units for the same hours"
            + (f": about {pct:.0f}% more than expected." if pct is not None else ".")
            + " (Simulated meter.)")
    if culprit:
        body += (f" Most of the extra came from the {culprit['name'].lower()}: {culprit['kwh']:.2f} units "
                 f"against its usual {culprit['usual_kwh']:.2f}. {culprit['advice']}")
    else:
        body += f" {GENERIC_ADVICE}"
    return subject, body, {"start": str(start), "end": str(end), "actual_kwh": round(actual, 3),
                           "predicted_kwh": round(pred, 3), "excess_kwh": round(actual - pred, 3),
                           "excess_pct": None if pct is None else round(pct, 1),
                           "appliance": culprit["key"] if culprit else None}


# ----------------------------------------------------------------- the pipeline
def run_check(db, fc, household_id: str, minute_df: pd.DataFrame, now: pd.Timestamp, owned,
              people: Optional[List[dict]] = None, service=None, language: str = "en",
              context: Optional[dict] = None) -> dict:
    """
    Bring the forecast log up to date for one household and raise an alert if
    use is above forecast.  Safe to call on every dashboard refresh: forecasts
    already logged are not recomputed into the log, finished hours are filled in
    once, and an ongoing event alerts only once.
    Returns {"result", "alert", "settings", "logged", "filled"}.
    """
    settings = AlertSettings.clean(db.get_store(household_id, SETTINGS_KEY))
    out = {"result": {"status": "insufficient", "hours": []}, "alert": None,
           "settings": asdict(settings), "logged": 0, "filled": 0}
    if minute_df is None or minute_df.empty:
        return out
    if "mains_kw" in minute_df.columns:
        minute_df, out["readings"] = clean_readings(minute_df)
        if minute_df.empty:
            return out
    upto = minute_df[pd.to_datetime(minute_df["datetime"]) < now.floor("min")]
    hourly = minute_to_hourly(upto)                          # completed hours only (>= 45 readings)
    hourly = hourly[pd.to_datetime(hourly["datetime"]) < now.floor("h")].dropna(subset=[TARGET])
    if hourly.empty:
        return out
    preds = meter_hour_forecasts(fc, hourly, context)
    recent = preds[preds["target_hour"] >= now.floor("h") - pd.Timedelta(hours=48)]
    out["logged"] = db.log_forecasts(household_id, SOURCE, recent[["target_hour", "predicted_kw", "lower_kw",
                                                                    "upper_kw"]].itertuples(index=False),
                                     model="xgb_sim")
    done = hourly[pd.to_datetime(hourly["datetime"]) >= now.floor("h") - pd.Timedelta(hours=48)]
    out["filled"] = db.record_actuals(household_id, SOURCE, done[["datetime", TARGET]].itertuples(index=False))
    log_df = db.forecast_log(household_id, SOURCE, since=now.floor("h") - pd.Timedelta(hours=48))
    if not settings.enabled:
        return out
    result = check_overconsumption(log_df, settings, now)
    out["result"] = result
    out["checked_at"] = now
    state, should_alert = next_state(db.get_store(household_id, STATE_KEY), result)
    db.set_store(household_id, STATE_KEY, state)
    if should_alert:
        culprit = attribute(minute_df[pd.to_datetime(minute_df["datetime"]) < now.floor("min")],
                            [r["target_hour"] for r in result["hours"]], owned)
        subject, body, data = alert_text(result, culprit)
        out["alert"] = {"subject": subject, "body": body, "data": data,
                        "delivery": deliver(db, household_id, people or [], subject, body, data, service, language)}
        log.info("Overconsumption alert for %s: %s", household_id, data)
    elif result["status"] == "over":
        log.debug("Overconsumption continues for %s; alert already sent", household_id)
    return out


def deliver(db, household_id: str, people: List[dict], subject: str, body: str, data: dict,
            service=None, language: str = "en") -> List[dict]:
    """
    Record the alert for every household member (shown in the app) and email it
    where email is configured.  The status says what really happened:
    "in-app" (not emailed), "sent", or "failed" with the error.
    """
    report = []
    for person in people or []:
        email = (person.get("email") or "").strip().lower()
        if not email:
            continue
        status, error = "in-app", None
        if service is not None and not email.endswith(".local"):
            try:
                if service.email_configured():
                    ok, error = service._deliver(email, subject,
                                                 service._compose(person.get("name") or "there", body, language),
                                                 None)
                    status = "sent" if ok else "failed"
            except Exception as exc:                       # delivery must never break the dashboard
                status, error = "failed", type(exc).__name__
                log.warning("Alert email for household %s failed: %s", household_id, type(exc).__name__)
        try:
            db.log_notification(household_id, email, person.get("name") or email, ALERT_TYPE,
                                "actual above forecast", data, subject, body, language,
                                status=status, error_message=error)
        except Exception:
            log.exception("Could not store the alert for %s", household_id)
        report.append({"email": email, "status": status, "error": error})
    return report


def latest_alert(db, household_id: str) -> Optional[dict]:
    for entry in db.get_notification_log(household_id, limit=50):
        if entry.get("notification_type") == ALERT_TYPE:
            return entry
    return None
