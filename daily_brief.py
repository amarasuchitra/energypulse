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


def _forecast_row(history: pd.DataFrame, target_day: pd.Timestamp) -> pd.Series:
    recent = history.tail(7).mean()
    same = history[history.index.dayofweek == target_day.dayofweek].tail(4)
    return recent if same.empty else (recent + same.mean()) / 2.0


def _usual_run(sessions: pd.DataFrame, key: str) -> Optional[str]:
    group = sessions[sessions["appliance"] == key]
    if len(group) < 3 or DEFAULT_PROFILES[key].category == ALWAYS_ON:
        return None
    start = (group["start"].dt.hour * 60 + group["start"].dt.minute).median()
    return f"usually starts about {_hhmm(start)}, runs about {int(group['span_min'].median())} min"


def build_brief(pred: pd.DataFrame, owned, tariff: Optional[Tariff] = None,
                min_saving_rs: float = 30.0, max_actions: int = 3) -> dict:
    """`pred` is the 1-minute detection history; its LAST calendar day is 'today'."""
    tariff = tariff or Tariff()
    owned = [k for k in owned if f"{k}_kw" in pred.columns]
    kwh, cost = _daily(pred, owned, tariff)
    today = pd.Timestamp(kwh.index[-1])
    past_kwh, past_cost = kwh.iloc[:-1], cost.iloc[:-1]
    if len(past_kwh) < MIN_HISTORY_DAYS:
        raise ValueError("At least a week of meter history is needed for a daily forecast.")

    f_kwh = _forecast_row(past_kwh, today)
    f_cost = _forecast_row(past_cost, today)

    # Replay the rule on recent days to measure how wrong it has been.
    errors, actuals = [], []
    for i in range(max(MIN_HISTORY_DAYS, len(past_kwh) - BACKTEST_DAYS), len(past_kwh)):
        guess = _forecast_row(past_kwh.iloc[:i], pd.Timestamp(past_kwh.index[i]))["total"]
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
    y_guess = float(_forecast_row(past_kwh.iloc[:-1], yesterday)["total"]) if len(past_kwh) > MIN_HISTORY_DAYS else None

    return {
        "date": str(today.date()), "date_label": f"{today.strftime('%a')} {today.day} {today.strftime('%b')}",
        "total_kwh": round(float(f_kwh["total"]), 1), "total_cost": round(float(f_cost["total"]), 0),
        "low_kwh": round(max(0.0, float(f_kwh["total"]) - mae), 1),
        "high_kwh": round(float(f_kwh["total"]) + mae, 1),
        "low_cost": round(max(0.0, float(f_kwh["total"]) - mae) * rate, 0),
        "high_cost": round((float(f_kwh["total"]) + mae) * rate, 0),
        "error_pct": round(error_pct, 1), "mae_kwh": round(mae, 2), "backtest_days": len(errors),
        "history_days": len(past_kwh),
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
    lines.append("Figures come from the simulated main-meter feed, not a physical meter.")
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
        db.log_notification(household_id, email, person.get("name") or email, NOTIFICATION_TYPE,
                            "daily schedule", data, subject, body, language,
                            status=status, error_message=error)
    return {"subject": subject, "emailed": emailed}
