"""
Realistic recommendations
=========================
Turns detected appliance usage into a short list of suggestions a household
could actually follow.

Rules the output always obeys
-----------------------------
1. Category first.  An always-on appliance (fridge) is never told to switch
   off or move.  A comfort appliance (AC) is never told to run at another
   time.  An on-demand appliance (microwave) gets no advice.
2. User limits.  Noisy appliances stay out of quiet hours; every shiftable
   appliance stays inside its allowed hours; a geyser may only heat EARLIER
   than the bath, and only by as long as the tank holds heat.
3. No saving, no tip.  On a flat tariff a change of time saves nothing, so
   no timing tip is produced.  Tips below the household's minimum monthly
   saving are dropped.
4. Few and ranked.  Biggest monthly saving first, capped.
5. Every number is shown with its working.
6. A tip the user dismissed does not come back.

Usage:
    from scheduler import recommend
    result = recommend(pred, sessions, tariff, prefs)
    result["suggestions"], result["alerts"], result["notes"]
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from appliance_profiles import (
    ALWAYS_ON, COMFORT, SHIFTABLE, ApplianceProfile, HouseholdPrefs,
    allowed_advice, can_switch_off, get_profiles,
)
from sessions import flag_long_runs, fridge_duty
from tariff import Tariff

DAYS_PER_MONTH = 30
# Bureau of Energy Efficiency (India) guidance: each 1 deg C increase in the
# AC set temperature saves about 6% of its electricity.
AC_SAVING_PER_DEGREE = 0.06


def _hhmm(minute: int) -> str:
    minute = int(minute) % 1440
    h, m = divmod(minute, 60)
    suffix = "am" if h < 12 else "pm"
    return f"{(h % 12) or 12}:{m:02d} {suffix}"


def _in_quiet(minute: int, quiet) -> bool:
    start, end = quiet[0] * 60, quiet[1] * 60
    minute %= 1440
    return (start <= minute < end) if start <= end else (minute >= start or minute < end)


def feasible_starts(profile: ApplianceProfile, duration: int, current_start: int,
                    prefs: HouseholdPrefs, step: int = 15) -> List[int]:
    """Start minutes (same day) at which the whole run respects every limit."""
    lo, hi = profile.allowed_hours[0] * 60, profile.allowed_hours[1] * 60
    starts = []
    for s in range(lo, hi - duration + 1, step):
        if profile.noisy and any(_in_quiet(m, prefs.quiet_hours)
                                 for m in (s, s + duration - 1)):
            continue
        if profile.max_advance_min is not None:
            if not (current_start - profile.max_advance_min <= s <= current_start):
                continue
        starts.append(s)
    return starts


def _shift_suggestion(key: str, group: pd.DataFrame, profile: ApplianceProfile,
                      tariff: Tariff, prefs: HouseholdPrefs, days: float) -> Optional[dict]:
    """Best alternative time for the runs that currently fall in costly hours."""
    moves = []
    for _, s in group.iterrows():
        start = s["start"].hour * 60 + s["start"].minute
        dur = int(s["span_min"])
        avg_kw = s["kwh"] / (dur / 60.0)
        now_cost = tariff.run_cost(start, dur, avg_kw)
        options = feasible_starts(profile, dur, start, prefs)
        if not options:
            continue
        costs = [tariff.run_cost(o, dur, avg_kw) for o in options]
        # Among the cheapest options pick the one closest to the current habit.
        cheapest = min(costs)
        near = [o for o, c in zip(options, costs) if c <= cheapest + 1e-9]
        best = min(near, key=lambda o: abs(o - start))
        if now_cost - cheapest > 0.01:
            moves.append({"from": start, "to": best, "saving": now_cost - cheapest,
                          "dur": dur, "kwh": s["kwh"], "now": now_cost, "new": cheapest})
    if not moves:
        return None
    m = pd.DataFrame(moves)
    runs_per_month = len(m) / days * DAYS_PER_MONTH
    saving_per_run = float(m["saving"].mean())
    monthly = saving_per_run * runs_per_month
    typical_from = int(m["from"].median())
    typical_to = int(m["to"].median() // 15 * 15)
    dur = int(m["dur"].median())
    return {
        "id": f"shift:{key}",
        "appliance": key,
        "kind": "shift",
        "title": f"Run the {profile.name.lower()} around {_hhmm(typical_to)} "
                 f"instead of {_hhmm(typical_from)}",
        "detail": (f"{len(m)} of your {len(group)} runs in the last {days:.0f} days were in "
                   f"{tariff.label_at(typical_from / 60).lower()} "
                   f"(Rs. {tariff.rate_at(typical_from / 60):.2f}/unit). "
                   f"{_hhmm(typical_to)} is {tariff.label_at(typical_to / 60).lower()} "
                   f"(Rs. {tariff.rate_at(typical_to / 60):.2f}/unit) and inside your allowed hours."),
        "monthly_saving_rs": round(monthly, 0),
        "calculation": (f"Average run {dur} min, {m['kwh'].mean():.2f} kWh: "
                        f"Rs. {m['now'].mean():.2f} now vs Rs. {m['new'].mean():.2f} moved "
                        f"= Rs. {saving_per_run:.2f} per run x {runs_per_month:.0f} runs/month"),
    }


def _ac_suggestions(group: pd.DataFrame, profile: ApplianceProfile,
                    tariff: Tariff, days: float) -> List[dict]:
    out = []
    monthly_cost = group["cost_rs"].sum() / days * DAYS_PER_MONTH
    hours_per_day = group["span_min"].sum() / 60.0 / days
    if hours_per_day >= 2.0:
        saving = monthly_cost * AC_SAVING_PER_DEGREE * 2
        out.append({
            "id": "duration:ac:setpoint",
            "appliance": "ac",
            "kind": "duration",
            "title": "If your AC is set below 24°C, raise it by 2°C",
            "detail": (f"Your AC runs about {hours_per_day:.1f} h a day and costs about "
                       f"Rs. {monthly_cost:.0f} a month. A higher set temperature makes the "
                       f"compressor rest more often. Skip this if it is already at 24°C or above."),
            "monthly_saving_rs": round(saving, 0),
            "calculation": (f"About 6% less AC energy per degree (BEE guidance) x 2 degrees "
                            f"= 12% of Rs. {monthly_cost:.0f}. An estimate, not a measurement."),
        })
    long_nights = group[(group["span_min"] >= 360)]
    if len(long_nights) >= max(2, 0.3 * days):
        # Compressor energy in the final hour of those nights.
        kw_last_hour = float((long_nights["kwh"] / (long_nights["span_min"] / 60.0)).mean())
        nights_per_month = len(long_nights) / days * DAYS_PER_MONTH
        end_hour = float((long_nights["end"].dt.hour + long_nights["end"].dt.minute / 60).median())
        saving = kw_last_hour * tariff.rate_at(end_hour - 1) * nights_per_month
        out.append({
            "id": "duration:ac:timer",
            "appliance": "ac",
            "kind": "duration",
            "title": "Set the AC sleep timer to switch off 1 hour earlier",
            "detail": (f"On {len(long_nights)} of the last {days:.0f} nights the AC ran for 6 hours "
                       f"or more, until about {_hhmm(int(end_hour * 60))}. Rooms stay cool for a "
                       f"while after the compressor stops, and early morning is the coolest part of the night."),
            "monthly_saving_rs": round(saving, 0),
            "calculation": (f"{kw_last_hour:.2f} kW average x 1 h x Rs. "
                            f"{tariff.rate_at(end_hour - 1):.2f} x {nights_per_month:.0f} nights/month"),
        })
    return out


def _fridge_health(pred: pd.DataFrame, profile: ApplianceProfile,
                   tariff: Tariff, days: float) -> Optional[dict]:
    duty = fridge_duty(pred, profile.key)
    if duty.empty or profile.normal_duty is None:
        return None
    observed = float(duty["duty"].median())
    if observed < profile.normal_duty * 1.5:
        return None
    kwh_month = pred[f"{profile.key}_kw"].sum() / 60.0 / days * DAYS_PER_MONTH
    excess = kwh_month * (1 - profile.normal_duty / observed)
    return {
        "id": f"health:{profile.key}",
        "appliance": profile.key,
        "kind": "health",
        "title": f"Have the {profile.name.lower()} checked - its compressor is running too much",
        "detail": (f"The compressor is running {observed * 100:.0f}% of the time; about "
                   f"{profile.normal_duty * 100:.0f}% is normal. Common causes: a worn door seal, "
                   f"the thermostat set too cold, dusty coils, or no gap behind the unit. "
                   f"Do not switch it off to save power."),
        "monthly_saving_rs": round(excess * tariff.rate, 0),
        "calculation": (f"{kwh_month:.1f} kWh/month now; at normal running it would be "
                        f"{kwh_month - excess:.1f} kWh. Difference {excess:.1f} kWh x Rs. {tariff.rate:.2f}"),
    }


def recommend(pred: pd.DataFrame, sessions: pd.DataFrame,
              tariff: Optional[Tariff] = None,
              prefs: Optional[HouseholdPrefs] = None) -> Dict[str, list]:
    """
    Returns
      suggestions - ranked, filtered, capped list of dicts
      alerts      - abnormal long runs (safety/waste warnings, not ranked by saving)
      notes       - plain statements explaining why something was NOT suggested
    """
    tariff = tariff or Tariff()
    prefs = prefs or HouseholdPrefs()
    profiles = get_profiles(prefs)
    dts = pd.to_datetime(pred["datetime"])
    days = max(1.0, (dts.iloc[-1] - dts.iloc[0]).total_seconds() / 86400.0)
    candidates, notes = [], []

    if not tariff.tod_enabled:
        notes.append("Your tariff is the same all day, so running an appliance at a "
                     "different time would not change your bill. No timing tips are shown.")

    for key, profile in profiles.items():
        group = sessions[sessions["appliance"] == key]
        if profile.category == SHIFTABLE and tariff.tod_enabled and not group.empty:
            s = _shift_suggestion(key, group, profile, tariff, prefs, days)
            if s:
                candidates.append(s)
        elif profile.category == COMFORT and not group.empty:
            candidates.extend(_ac_suggestions(group, profile, tariff, days))
        elif profile.category == ALWAYS_ON:
            s = _fridge_health(pred, profile, tariff, days)
            if s:
                candidates.append(s)

    suggestions = []
    for s in candidates:
        profile = profiles[s["appliance"]]
        # Hard guard: the category decides what may be said at all.
        if s["kind"] not in allowed_advice(profile):
            continue
        if s["id"] in prefs.dismissed:
            continue
        if s["monthly_saving_rs"] < prefs.min_monthly_saving_rs:
            notes.append(f"{profile.name}: a possible change would save only about "
                         f"Rs. {s['monthly_saving_rs']:.0f} a month, so it is not suggested.")
            continue
        suggestions.append(s)
    suggestions.sort(key=lambda s: s["monthly_saving_rs"], reverse=True)

    return {
        "suggestions": suggestions[: prefs.max_suggestions],
        "alerts": flag_long_runs(sessions, profiles),
        "notes": notes,
    }


def explain_refusal(appliance_key: str, action: str,
                    prefs: Optional[HouseholdPrefs] = None) -> Optional[str]:
    """
    For the chatbot.  `action` is "switch_off" or "shift".  Returns a reason
    when the request is something the system must not recommend, else None.
    """
    profile = get_profiles(prefs).get(appliance_key)
    if profile is None:
        return None
    if action == "switch_off" and not can_switch_off(profile):
        return (f"The {profile.name.lower()} should stay on all the time. {profile.note} "
                f"Switching it off for a few hours saves very little, because it then runs "
                f"longer to cool down again, and food can spoil.")
    if action == "shift" and "shift" not in allowed_advice(profile):
        return (f"The {profile.name.lower()} is used when it is needed, so moving it to "
                f"another time is not a practical suggestion. {profile.note}")
    return None
