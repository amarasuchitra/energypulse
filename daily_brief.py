"""
Daily prediction
================
Once a day the household is told, before the day is used up:

  1. how many units each appliance is expected to use today, and the cost;
  2. how far off this forecast has been on recent days (measured, not claimed);
  3. the exact changes worth making, each with the rupee arithmetic behind it.

The forecast uses only days BEFORE today.  For each appliance:

    forecast = average of ( mean of the last 7 days,
                            mean of the same weekday in the last 4 weeks )

The same rule is replayed on each of the last 14 days, using only the days
before it, to measure the error that is printed next to the forecast.

    brief = build_brief(pred, owned, tariff)
    subject, body = brief_text(brief)
    send_daily_brief(db, household_id, people, brief)     # once per day
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from appliance_profiles import ALWAYS_ON, DEFAULT_PROFILES, HouseholdPrefs
from scheduler import DAYS_PER_MONTH, recommend
from sessions import extract_sessions
from tariff import Tariff
from weather import COMFORT_C

NOTIFICATION_TYPE = "daily_forecast"
BACKTEST_DAYS = 14
MIN_HISTORY_DAYS = 7


def _hhmm(minute: float) -> str:
    minute = int(minute) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _daily(pred: pd.DataFrame, owned, tariff: Tariff) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """kWh and cost per calendar day: one column per appliance, plus other and total."""
    dts = pd.to_datetime(pred["datetime"])
    rates = tariff.minute_rates()[(dts.dt.hour * 60 + dts.dt.minute).to_numpy()]
    day = dts.dt.normalize()
    kwh, cost = {}, {}
    for key in list(owned) + ["other"]:
        units = pred[f"{key}_kw"].to_numpy(dtype=float) / 60.0
        kwh[key] = pd.Series(units).groupby(day.to_numpy()).sum()
        cost[key] = pd.Series(units * rates).groupby(day.to_numpy()).sum()
    kwh, cost = pd.DataFrame(kwh), pd.DataFrame(cost)
    kwh["total"], cost["total"] = kwh.sum(axis=1), cost.sum(axis=1)
    return kwh, cost


def _forecast_row(history: pd.DataFrame, target_day: pd.Timestamp,
                  temps: Optional[pd.Series] = None, target_temp: Optional[float] = None) -> pd.Series:
    """
    Expected units per column for `target_day`, from the days in `history`.
    Every appliance: the average of the last 7 days and of the same weekday.
    The air conditioner, when temperatures are known: a straight line fitted to
    its daily units against the daily high, read off at the forecast high.
    """
    recent = history.tail(7).mean()
    same = history[history.index.dayofweek == target_day.dayofweek].tail(4)
    out = (recent if same.empty else (recent + same.mean()) / 2.0).copy()
    if temps is not None and target_temp is not None and "ac" in history.columns and len(history) >= 10:
        # Cooling is only needed above a comfortable temperature, so the line is
        # fitted to degrees above it, not to the temperature itself.
        above = lambda t: np.clip(np.asarray(t, dtype=float) - COMFORT_C, 0.0, None)
        # A night run starts on one calendar day and finishes on the next, so a
        # day's AC units depend on that day's heat and on the day before's.
        all_days = temps.sort_index()
        today_x = above(all_days.reindex(history.index).to_numpy(dtype=float))
        prev_x = above(all_days.shift(1).reindex(history.index).to_numpy(dtype=float))
        y = history["ac"].to_numpy(dtype=float)
        ok = ~np.isnan(today_x) & ~np.isnan(prev_x)
        before = all_days.get(target_day - pd.Timedelta(days=1))
        if ok.sum() >= 10 and np.ptp(today_x[ok]) >= 2.0 and before is not None and not np.isnan(before):
            design = np.column_stack([today_x[ok], prev_x[ok], np.ones(int(ok.sum()))])
            (w_today, w_prev, base), *_ = np.linalg.lstsq(design, y[ok], rcond=None)
            if w_today + w_prev > 0:                         # hotter must mean more, or the fit is noise
                fitted = w_today * float(above(target_temp)) + w_prev * float(above(before)) + base
                fitted = float(np.clip(fitted, 0.0, y.max() * 1.5 + 0.5))
                out["ac"] = 0.7 * fitted + 0.3 * float(out["ac"])
    if "total" in out.index:
        out["total"] = float(out.drop("total").sum())
    return out


def _usual_run(sessions: pd.DataFrame, key: str) -> Optional[str]:
    group = sessions[sessions["appliance"] == key]
    if len(group) < 3 or DEFAULT_PROFILES[key].category == ALWAYS_ON:
        return None
    start = (group["start"].dt.hour * 60 + group["start"].dt.minute).median()
    return f"usually starts about {_hhmm(start)}, runs about {int(group['span_min'].median())} min"


def build_brief(pred: pd.DataFrame, owned, tariff: Optional[Tariff] = None,
                min_saving_rs: float = 30.0, max_actions: int = 3,
                temps: Optional[pd.Series] = None, forecast_temp: Optional[float] = None,
                temp_forecasts: Optional[pd.Series] = None) -> dict:
    """
    temps: daily high for each day of the history; forecast_temp: the forecast
    high for today; temp_forecasts: what the forecast said on each past day
    (used when replaying the rule to measure its error, so the replay is as
    blind as the real thing).  Without temperatures the plain averages are used.
    """
    """`pred` is the 1-minute detection history; its LAST calendar day is 'today'."""
    tariff = tariff or Tariff()
    owned = [k for k in owned if f"{k}_kw" in pred.columns]
    kwh, cost = _daily(pred, owned, tariff)
    today = pd.Timestamp(kwh.index[-1])
    past_kwh, past_cost = kwh.iloc[:-1], cost.iloc[:-1]
    if len(past_kwh) < MIN_HISTORY_DAYS:
        raise ValueError("At least a week of meter history is needed for a daily forecast.")

    if temps is not None:
        temps = temps.copy()
        temps.index = pd.to_datetime(temps.index).normalize()
    use_temp = temps is not None and forecast_temp is not None
    f_kwh = _forecast_row(past_kwh, today, temps if use_temp else None, forecast_temp)
    # Cost follows units at each appliance's own average price per unit (timing of use).
    price = (past_cost.sum() / past_kwh.sum().replace(0, np.nan)).fillna(float(tariff.rate))
    f_cost = f_kwh * price
    f_cost["total"] = float(f_cost.drop("total").sum())

    # Replay the rule on recent days to measure how wrong it has been.
    errors, actuals = [], []
    for i in range(max(MIN_HISTORY_DAYS, len(past_kwh) - BACKTEST_DAYS), len(past_kwh)):
        day_i = pd.Timestamp(past_kwh.index[i])
        seen = None
        if use_temp:
            seen = temp_forecasts.get(day_i) if temp_forecasts is not None else None
            seen = float(seen) if seen is not None else float(temps.get(day_i, np.nan))
        guess = _forecast_row(past_kwh.iloc[:i], day_i, temps if use_temp else None,
                              None if seen is None or np.isnan(seen) else seen)["total"]
        errors.append(abs(guess - past_kwh["total"].iloc[i]))
        actuals.append(past_kwh["total"].iloc[i])
    mae = float(np.mean(errors)) if errors else 0.0
    error_pct = 100.0 * float(np.sum(errors)) / max(float(np.sum(actuals)), 1e-9) if errors else 0.0
    rate = float(f_cost["total"]) / max(float(f_kwh["total"]), 1e-9)

    history = pred.iloc[: len(pred) - int((pd.to_datetime(pred["datetime"]).dt.normalize() == today).sum())]
    sessions = extract_sessions(history, tariff)
    appliances = []
    for key in sorted(owned, key=lambda k: -float(f_cost[k])):
        appliances.append({
            "key": key, "name": DEFAULT_PROFILES[key].name,
            "kwh": round(float(f_kwh[key]), 2), "cost": round(float(f_cost[key]), 1),
            "share": round(100.0 * float(f_kwh[key]) / max(float(f_kwh["total"]), 1e-9), 0),
            "usual": _usual_run(sessions, key),
        })

    result = recommend(history, sessions, tariff,
                       HouseholdPrefs(min_monthly_saving_rs=min_saving_rs, max_suggestions=max_actions))
    actions = [{
        "appliance": s["appliance"], "title": s["title"], "detail": s["detail"],
        "calculation": s["calculation"].rstrip("."),
        "saving_month": float(s["monthly_saving_rs"]),
        "saving_day": round(float(s["monthly_saving_rs"]) / DAYS_PER_MONTH, 1),
    } for s in result["suggestions"]]
    yesterday = pd.Timestamp(past_kwh.index[-1])
    y_seen = None
    if use_temp:
        y_seen = temp_forecasts.get(yesterday) if temp_forecasts is not None else temps.get(yesterday)
    y_guess = (float(_forecast_row(past_kwh.iloc[:-1], yesterday, temps if use_temp else None,
                                   None if y_seen is None else float(y_seen))["total"])
               if len(past_kwh) > MIN_HISTORY_DAYS else None)

    return {
        "date": str(today.date()), "date_label": f"{today.strftime('%a')} {today.day} {today.strftime('%b')}",
        "total_kwh": round(float(f_kwh["total"]), 1), "total_cost": round(float(f_cost["total"]), 0),
        "low_kwh": round(max(0.0, float(f_kwh["total"]) - mae), 1),
        "high_kwh": round(float(f_kwh["total"]) + mae, 1),
        "low_cost": round(max(0.0, float(f_kwh["total"]) - mae) * rate, 0),
        "high_cost": round((float(f_kwh["total"]) + mae) * rate, 0),
        "error_pct": round(error_pct, 1), "mae_kwh": round(mae, 2), "backtest_days": len(errors),
        "history_days": len(past_kwh),
        "temp_today": None if not use_temp else float(forecast_temp),
        "temp_yesterday": None if not use_temp or yesterday not in temps.index else float(temps[yesterday]),
        "uses_weather": bool(use_temp and "ac" in past_kwh.columns),
        "appliances": appliances,
        "other_kwh": round(float(f_kwh["other"]), 2), "other_cost": round(float(f_cost["other"]), 1),
        "month_cost": round(float(past_cost["total"].mean()) * DAYS_PER_MONTH, 0),
        "actions": actions,
        "saving_month": round(sum(a["saving_month"] for a in actions), 0),
        "notes": result["notes"],
        "alerts": [f"{DEFAULT_PROFILES[a['appliance']].name}: {a['message']}" if "message" in a
                   else DEFAULT_PROFILES[a["appliance"]].name for a in result["alerts"][:2]],
        "yesterday": None if y_guess is None else {
            "forecast": round(y_guess, 1), "actual": round(float(past_kwh["total"].iloc[-1]), 1)},
    }


def right_now(pred: pd.DataFrame, owned, tariff: Tariff, minute: int) -> dict:
    """
    The meter at `minute` of today (the last calendar day of `pred`): load now,
    units and cost so far, what the next hour usually looks like at this time
    of day, and the last seven days.
    """
    owned = [k for k in owned if f"{k}_kw" in pred.columns]
    dts = pd.to_datetime(pred["datetime"])
    day = dts.dt.normalize()
    today = pred[day == day.iloc[-1]].reset_index(drop=True)
    minute = int(np.clip(minute, 0, len(today) - 1))
    so_far = today.iloc[: minute + 1]
    rates = tariff.minute_rates()
    kwh = float(so_far["mains_kw"].sum() / 60.0)
    cost = float((so_far["mains_kw"].to_numpy() / 60.0 * rates[: minute + 1]).sum())
    running = [DEFAULT_PROFILES[k].name for k in owned if bool(today[f"{k}_on"].iloc[minute])]

    # The coming hour, from the same clock hour on each earlier day.
    past = pred[day < day.iloc[-1]]
    clock = (pd.to_datetime(past["datetime"]).dt.hour * 60 + pd.to_datetime(past["datetime"]).dt.minute)
    lo, hi = minute + 1, minute + 60
    window = (clock >= lo) & (clock <= hi) if hi < 1440 else (clock >= lo) | (clock <= hi - 1440)
    hourly = past.loc[window].groupby(pd.to_datetime(past.loc[window, "datetime"]).dt.normalize())["mains_kw"].mean().tail(14)
    week = past.tail(7 * 1440)
    week_cost = float((week["mains_kw"].to_numpy() / 60.0
                       * rates[(pd.to_datetime(week["datetime"]).dt.hour * 60
                                + pd.to_datetime(week["datetime"]).dt.minute).to_numpy()]).sum())
    before = past.iloc[-14 * 1440:-7 * 1440]
    before_kwh = float(before["mains_kw"].sum() / 60.0)
    week_kwh = float(week["mains_kw"].sum() / 60.0)
    return {
        "minute": minute, "kw": float(today["mains_kw"].iloc[minute]), "running": running,
        "kwh": kwh, "cost": cost, "rate": float(rates[minute]), "period": tariff.label_at(minute / 60.0),
        "next_kw": float(hourly.mean()) if len(hourly) else 0.0,
        "next_lo": float(hourly.quantile(0.1)) if len(hourly) else 0.0,
        "next_hi": float(hourly.quantile(0.9)) if len(hourly) else 0.0,
        "next_days": int(len(hourly)),
        "week_kwh": week_kwh, "week_cost": week_cost,
        "week_change": 100.0 * (week_kwh - before_kwh) / before_kwh if before_kwh > 0 else None,
        "trace": today["mains_kw"].iloc[: minute + 1].round(3).tolist(),
        "typical": pred[day < day.iloc[-1]].assign(m=clock.to_numpy()).groupby("m")["mains_kw"].mean().round(3).tolist(),
    }


def brief_text(brief: dict) -> Tuple[str, str]:
    """Subject and plain-text body, used for the in-app notification and for email."""
    subject = (f"Today's forecast, {brief['date_label']}: {brief['total_kwh']:.1f} units, "
               f"about Rs. {brief['total_cost']:.0f}")
    lines = [
        f"Expected today: {brief['total_kwh']:.1f} units (likely {brief['low_kwh']:.1f} to "
        f"{brief['high_kwh']:.1f}), about Rs. {brief['total_cost']:.0f} "
        f"(Rs. {brief['low_cost']:.0f} to {brief['high_cost']:.0f}).",
        f"On the last {brief['backtest_days']} days this forecast was off by {brief['error_pct']:.1f}% "
        f"on average ({brief['mae_kwh']:.2f} units a day).",
    ]
    if brief.get("temp_today") is not None:
        lines.append(f"Forecast high today: {brief['temp_today']:.0f} C"
                     + (f" (yesterday {brief['temp_yesterday']:.0f} C)" if brief.get("temp_yesterday") is not None else "")
                     + (". The air-conditioner figure follows it." if brief.get("uses_weather") else "."))
    if brief["yesterday"]:
        lines.append(f"Yesterday: forecast {brief['yesterday']['forecast']:.1f} units, "
                     f"actual {brief['yesterday']['actual']:.1f}.")
    lines += ["", "By appliance:"]
    for a in brief["appliances"]:
        usual = f" ({a['usual']})" if a["usual"] else ""
        lines.append(f"  {a['name']}: {a['kwh']:.2f} units, Rs. {a['cost']:.1f}, {a['share']:.0f}% of the day{usual}")
    lines.append(f"  Everything else (lights, fans, TV, standby): {brief['other_kwh']:.2f} units, "
                 f"Rs. {brief['other_cost']:.1f}")
    lines.append("")
    if brief["actions"]:
        lines.append("What to change, biggest saving first:")
        for i, a in enumerate(brief["actions"], 1):
            lines.append(f"  {i}. {a['title']}. Saves about Rs. {a['saving_month']:.0f} a month "
                         f"(Rs. {a['saving_day']:.1f} a day).")
            lines.append(f"     How it is worked out: {a['calculation']}.")
        lines.append(f"All together: about Rs. {brief['saving_month']:.0f} a month, out of a bill of "
                     f"about Rs. {brief['month_cost']:.0f}.")
    else:
        lines.append("No change is worth making today. Nothing would save more than the small-saving limit.")
    if any(a["key"] == "fridge" for a in brief["appliances"]):
        lines.append("The refrigerator is never told to switch off or run at another time.")
    lines.append("Figures come from the simulated main-meter feed and simulated weather, not a physical meter.")
    return subject, "\n".join(lines)


def already_sent(db, household_id: str, date: str) -> bool:
    import json
    for entry in db.get_notification_log(household_id, limit=300):
        if entry.get("notification_type") != NOTIFICATION_TYPE:
            continue
        try:
            if json.loads(entry.get("trigger_data_json") or "{}").get("date") == date:
                return True
        except ValueError:
            continue
    return False


def send_daily_brief(db, household_id: str, people: List[dict], brief: dict,
                     language: str = "en", service=None) -> Optional[dict]:
    """
    Record today's forecast for every household member, once per day.  It is
    always shown in the app; it is also emailed when an email service is set up.
    Returns {"subject", "emailed"} when it was newly sent, None if today's was already sent.
    """
    if already_sent(db, household_id, brief["date"]):
        return None
    subject, body = brief_text(brief)
    data = {"date": brief["date"], "total_kwh": brief["total_kwh"], "total_cost": brief["total_cost"],
            "error_pct": brief["error_pct"], "saving_month": brief["saving_month"]}
    emailed = 0
    for person in people:
        email = (person.get("email") or "").strip().lower()
        if not email:
            continue
        status, error = "in-app", None
        if service is not None and not email.endswith(".local"):
            try:
                if service.email_configured():
                    ok, error = service._deliver(email, subject,
                                                 service._compose(person.get("name") or "there", body, language), None)
                    status = "sent" if ok else "in-app"
                    emailed += int(bool(ok))
            except Exception as exc:                  # email must never stop the in-app notice
                error = type(exc).__name__
        if service is not None and person.get("phone"):
            try:
                if service.sms_configured():
                    service.send_sms(person["phone"], subject[:150])
            except Exception:
                pass
        db.log_notification(household_id, email, person.get("name") or email, NOTIFICATION_TYPE,
                            "daily schedule", data, subject, body, language,
                            status=status, error_message=error)
    return {"subject": subject, "emailed": emailed}
