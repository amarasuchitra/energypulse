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
        if not within_shift_limits(profile, current_start, s):
            continue
        starts.append(s)
    return starts


def within_shift_limits(profile: ApplianceProfile, current_start: int, new_start: int) -> bool:
    """Earlier by at most max_advance_min, later by at most max_delay_min (None = no limit)."""
    if profile.max_advance_min is not None and new_start < current_start - profile.max_advance_min:
        return False
    if profile.max_delay_min is not None and new_start > current_start + profile.max_delay_min:
        return False
    return True


def _run_fits(profile: ApplianceProfile, start: int, dur: int, prefs: HouseholdPrefs) -> bool:
    """The whole run stays in the appliance's allowed hours and out of quiet hours if noisy."""
    lo, hi = profile.allowed_hours[0] * 60, profile.allowed_hours[1] * 60
    if not (lo <= start and start + dur <= hi):
        return False
    return not (profile.noisy and any(_in_quiet(m, prefs.quiet_hours) for m in (start, start + dur - 1)))


SLOT_MIN = 15
SLOTS = 1440 // SLOT_MIN


def _slots(start: int, dur: int) -> np.ndarray:
    """15-minute slots of the day that a run starting at `start` touches."""
    return np.unique(((start + np.arange(max(dur, 1))) % 1440) // SLOT_MIN)


def background_load(pred: pd.DataFrame, movable) -> np.ndarray:
    """
    Load per 15-minute slot of the day from everything that is NOT being moved
    (90th percentile across days), in kW.  Moved runs are added on top of it.
    """
    dts = pd.to_datetime(pred["datetime"])
    other = pred["mains_kw"].astype(float).copy() if "mains_kw" in pred else pd.Series(0.0, index=pred.index)
    for key in movable:
        if f"{key}_kw" in pred:
            other = other - pred[f"{key}_kw"].astype(float)
    slot = ((dts.dt.hour * 60 + dts.dt.minute) // SLOT_MIN).to_numpy()
    frame = pd.DataFrame({"slot": slot, "day": dts.dt.date.to_numpy(), "kw": other.clip(lower=0).to_numpy()})
    per_day = frame.groupby(["day", "slot"])["kw"].mean().unstack()
    return np.array(per_day.quantile(0.9).reindex(range(SLOTS)).fillna(0.0), dtype=float)


def day_loads(pred: pd.DataFrame) -> Dict[pd.Timestamp, np.ndarray]:
    """The home's measured load per 15-minute slot, for each day (kW)."""
    dts = pd.to_datetime(pred["datetime"])
    slot = ((dts.dt.hour * 60 + dts.dt.minute) // SLOT_MIN).to_numpy()
    frame = pd.DataFrame({"day": dts.dt.normalize().to_numpy(), "slot": slot,
                          "kw": pred["mains_kw"].astype(float).to_numpy()})
    table = frame.groupby(["day", "slot"])["kw"].mean().unstack().reindex(columns=range(SLOTS)).fillna(0.0)
    return {pd.Timestamp(d): np.array(row, dtype=float) for d, row in table.iterrows()}


# How a person would put it, per appliance.  Used by the advice text only.
HUMAN = {
    "geyser": dict(
        verb="Switch the geyser on", thing="geyser",
        why=("An insulated tank keeps water hot for a couple of hours, so it will still be ready "
             "when you need it. The little heat it loses while waiting is already taken off the saving."),
        tip="If your geyser has a timer, set it once and you won't have to remember."),
    "water_pump": dict(
        verb="Run the water pump", thing="water pump",
        why=("Filling the tank earlier is safe: the water simply waits in the tank. Filling it later "
             "could leave you short, so that is never suggested."),
        tip="Doing it at the same time every day turns it into an easy habit."),
    "washing_machine": dict(
        verb="Start the washing machine", thing="washing machine",
        why=("It's the same day and still in waking hours, so the laundry gets done and the noise "
             "won't disturb anyone."),
        tip="If your machine has a delay-start button, load it as usual and let it start itself."),
}

ROUTINE_GAP_MIN = 180          # runs further apart than this in the day are different habits
NEAR_BEST = 0.90               # take the smallest change that keeps 90% of the best saving
MIN_MOVE_MIN = 15              # a change of a few minutes is not a change worth asking for
MIN_RS_PER_RUN = 0.25          # nor is one that saves less than this each time


def _part_of_day(minute: int) -> str:
    h = (int(minute) % 1440) // 60
    return "mornings" if 4 <= h < 12 else "afternoons" if h < 17 else "evenings" if h < 22 else "nights"


def _routines(runs: List[dict]) -> List[List[dict]]:
    """Split runs into daily habits (e.g. a morning and an evening geyser) by time of day."""
    runs = sorted(runs, key=lambda r: r["start"])
    groups, current = [], [runs[0]]
    for r in runs[1:]:
        if r["start"] - current[-1]["start"] > ROUTINE_GAP_MIN:
            groups.append(current)
            current = []
        current.append(r)
    groups.append(current)
    return groups


def _run_gain(profile, run, to, tariff, prefs, base, cap_kw):
    """(saving in Rs., extra kWh) of moving one run to `to`, or None when that is not allowed."""
    start, dur, kw = run["start"], run["dur"], run["kw"]
    if to == start or not within_shift_limits(profile, start, to) or not _run_fits(profile, to, dur, prefs):
        return None
    if base is not None and cap_kw and float(base[_slots(to, dur)].max()) + kw > cap_kw:
        return None
    hours_early = max(0, start - to) / 60.0
    loss_kwh = profile.standby_loss_kwh_per_h * hours_early
    new_cost = tariff.run_cost(to, dur, kw) + loss_kwh * tariff.rate_at(to / 60.0)
    return tariff.run_cost(start, dur, kw) - new_cost, loss_kwh


def _base_without(load, run):
    """That day's load with this run taken out (None when the day is unknown)."""
    if load is None or run["day"] not in load:
        return None
    base = load[run["day"]].copy()
    old = _slots(run["start"], run["dur"])
    base[old] = np.maximum(base[old] - run["kw"], 0.0)
    return base


def _shift_suggestion(key: str, group: pd.DataFrame, profile: ApplianceProfile,
                      tariff: Tariff, prefs: HouseholdPrefs, days: float,
                      load: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
                      cap_kw: Optional[float] = None) -> Optional[dict]:
    """
    A new routine time for a shiftable appliance, chosen the way a sensible
    person would:

    - one fixed time per daily habit (runs are grouped by time of day), so it
      is easy to remember, rather than a different time every day;
    - the smallest change that keeps most (90%) of the best possible saving;
    - only directions that make sense (a geyser or a water tank is never
      filled after it is needed) and within each appliance's limits, quiet
      hours and allowed hours;
    - heat lost by a geyser that heats early is charged against the saving;
    - with `load` and `cap_kw`, a day is only moved if the home stays under the
      cap; accepted moves are added to `load` so later appliances plan on top.
    """
    runs = []
    for _, r in group.iterrows():
        dur = max(int(r["span_min"]), 1)
        runs.append({"start": r["start"].hour * 60 + r["start"].minute, "dur": dur,
                     "kwh": float(r["kwh"]), "kw": float(r["kwh"]) / (dur / 60.0),
                     "day": pd.Timestamp(r["start"]).normalize()})
    if not runs:
        return None
    moves, habits = [], []
    for habit in _routines(runs):
        typical = int(np.median([r["start"] for r in habit]))
        dur = int(np.median([r["dur"] for r in habit]))
        candidates = [t for t in feasible_starts(profile, dur, typical, prefs) if abs(t - typical) >= MIN_MOVE_MIN]
        if not candidates:
            continue
        bases = [_base_without(load, r) for r in habit]
        totals = {}
        for to in candidates:
            gains = [_run_gain(profile, r, to, tariff, prefs, bs, cap_kw) for r, bs in zip(habit, bases)]
            totals[to] = sum(g[0] for g in gains if g and g[0] > 0.01)
        best = max(totals.values())
        if best <= 0.01:
            continue
        good = [t for t, v in totals.items() if v >= NEAR_BEST * best]
        chosen = min(good, key=lambda t: (abs(t - typical), -totals[t]))
        habit_moves = []
        for r in habit:                                   # commit, day by day, on the updated load
            base = _base_without(load, r)
            g = _run_gain(profile, r, chosen, tariff, prefs, base, cap_kw)
            if not g or g[0] <= 0.01:
                continue
            saving, loss = g
            if base is not None:
                base[_slots(chosen, r["dur"])] += r["kw"]
                load[r["day"]] = base
            habit_moves.append({"day": r["day"], "from": r["start"], "to": chosen, "dur": r["dur"],
                                "kw": r["kw"], "kwh": r["kwh"], "loss_kwh": loss, "saving": saving,
                                "now": tariff.run_cost(r["start"], r["dur"], r["kw"])})
        habit_saving = sum(m["saving"] for m in habit_moves)
        if habit_moves and habit_saving / len(habit_moves) < MIN_RS_PER_RUN:
            for m in habit_moves:                         # not worth asking for: put the load back
                if load is not None and m["day"] in load:
                    day_load = load[m["day"]].copy()
                    day_load[_slots(m["to"], m["dur"])] -= m["kw"]
                    day_load[_slots(m["from"], m["dur"])] += m["kw"]
                    load[m["day"]] = np.maximum(day_load, 0.0)
            habit_moves = []
        if habit_moves:
            moves.extend(habit_moves)
            habits.append({"typical": typical, "to": chosen, "dur": dur, "runs": len(habit),
                           "moved": len(habit_moves), "saving": sum(m["saving"] for m in habit_moves),
                           "best": best, "best_to": max(totals, key=totals.get)})
    if not moves:
        return None
    habits.sort(key=lambda h: -h["saving"])
    monthly = sum(h["saving"] for h in habits) / days * DAYS_PER_MONTH
    title, detail, calculation = _shift_words(key, profile, habits, moves, tariff, days, monthly, len(runs))
    return {
        "id": f"shift:{key}",
        "appliance": key,
        "kind": "shift",
        "title": title,
        "detail": detail,
        "monthly_saving_rs": round(monthly, 0),
        "monthly_saving_exact": monthly,
        "calculation": calculation,
        "routines": [{"from": _hhmm(h["typical"]), "to": _hhmm(h["to"]), "runs_moved": h["moved"],
                      "monthly_saving_rs": round(h["saving"] / days * DAYS_PER_MONTH, 1)} for h in habits],
        "moves": [{k: m[k] for k in ("day", "from", "to", "dur", "kw", "kwh", "loss_kwh")} for m in moves],
        "history_days": days,
    }


def _shift_words(key, profile, habits, moves, tariff, days, monthly, n_runs):
    """Title, explanation and arithmetic, written the way an energy advisor would say it."""
    words = HUMAN.get(key, dict(verb=f"Run the {profile.name.lower()}", thing=profile.name.lower(),
                                why="It stays within the hours you allow.", tip=""))
    h = habits[0]
    frm, to = h["typical"], h["to"]
    title = f"{words['verb']} at {_hhmm(to)} instead of about {_hhmm(frm)}"
    rate_now, rate_new = tariff.rate_at(frm / 60.0), tariff.rate_at(to / 60.0)
    parts = [f"Most {_part_of_day(frm)} your {words['thing']} starts around {_hhmm(frm)}, "
             f"in {tariff.label_at(frm / 60.0).lower()} hours when a unit costs Rs. {rate_now:.2f}. "
             f"At {_hhmm(to)} it costs Rs. {rate_new:.2f} ({tariff.label_at(to / 60.0).lower()})."]
    parts.append(words["why"])
    for other in habits[1:]:
        parts.append(f"The same works for your {_part_of_day(other['typical'])[:-1]} run: "
                     f"{_hhmm(other['to'])} instead of about {_hhmm(other['typical'])}.")
    extra = (h["best"] - h["saving"]) / days * DAYS_PER_MONTH
    if h["best_to"] != to and extra >= 1.0:
        parts.append(f"This is the smallest change that gets most of the saving; moving it all the way to "
                     f"{_hhmm(h['best_to'])} would add only about Rs. {extra:.0f} a month.")
    moved = len(moves)
    if moved < n_runs:
        parts.append(f"It applies to {moved} of your {n_runs} runs in the last {days:.0f} days; the rest are "
                     f"already at a good time, or moving them would take the home too close to its load limit.")
    parts.append(f"That's about Rs. {monthly:.0f} a month, or Rs. {monthly * 12:.0f} a year.")
    if words.get("tip"):
        parts.append(words["tip"])
    m = pd.DataFrame(moves)
    per_month = len(m) / days * DAYS_PER_MONTH
    loss_rs = float((m["loss_kwh"] * tariff.rate).mean()) if "loss_kwh" in m else 0.0
    calc = (f"Average run {int(m['dur'].median())} min, {m['kwh'].mean():.2f} kWh: "
            f"Rs. {m['now'].mean():.2f} now vs Rs. {(m['now'] - m['saving']).mean() - loss_rs:.2f} moved")
    if loss_rs > 0.005:
        calc += f", plus Rs. {loss_rs:.2f} for heat lost while waiting (estimate)"
    calc += f" = Rs. {m['saving'].mean():.2f} per run x {per_month:.0f} runs/month"
    return title, " ".join(parts), calc


def _combined_routine(small: List[dict], profiles) -> Optional[dict]:
    """One tip for several moves that are each too small to show but add up."""
    small = sorted(small, key=lambda s: -s["monthly_saving_exact"])
    total = sum(s["monthly_saving_exact"] for s in small)
    names = [profiles[s["appliance"]].name.lower() for s in small]
    listed = ", ".join(names[:-1]) + " and " + names[-1]
    count = {2: "Two", 3: "Three", 4: "Four"}.get(len(small), str(len(small)))
    return {
        "id": "shift:routine:" + "+".join(sorted(s["appliance"] for s in small)),
        "appliance": small[0]["appliance"],
        "kind": "shift",
        "title": f"{count} small timing changes for the {listed}",
        "detail": ("None of these is worth much on its own, but together they add up to about "
                   f"Rs. {total:.0f} a month (Rs. {total * 12:.0f} a year). "
                   + " ".join(f"{s['title']} (about Rs. {s['monthly_saving_exact']:.0f}/month)." for s in small)),
        "monthly_saving_rs": round(total, 0),
        "monthly_saving_exact": total,
        "calculation": " + ".join(f"Rs. {s['monthly_saving_exact']:.1f}" for s in small)
                       + f" = Rs. {total:.1f} a month. " + " | ".join(s["calculation"] for s in small),
        "members": [s["appliance"] for s in small],
        "moves": [dict(mv, appliance=s["appliance"]) for s in small for mv in s.get("moves", [])],
        "history_days": small[0].get("history_days"),
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
            "title": "If your AC is set below 24°C, nudge it up by 2°C",
            "detail": (f"Your AC runs about {hours_per_day:.1f} hours a day, which costs roughly "
                       f"Rs. {monthly_cost:.0f} a month. Each degree warmer cuts its electricity by about 6%, "
                       f"so 2°C saves around Rs. {saving:.0f} a month. Most people stop noticing the "
                       f"difference within a few minutes, especially with a ceiling fan on. "
                       f"If it's already at 24°C or higher, you can skip this one."),
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
            "title": "Let the AC switch itself off an hour earlier at night",
            "detail": (f"On {len(long_nights)} of the last {days:.0f} nights the AC ran for 6 hours "
                       f"or more, until about {_hhmm(int(end_hour * 60))}. Early morning is the coolest "
                       f"part of the night and a closed room stays cool for a while after the compressor "
                       f"stops, so a sleep timer set an hour earlier is unlikely to wake you. "
                       f"That's about Rs. {saving:.0f} a month."),
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
        "title": f"Your {profile.name.lower()} is working harder than it should",
        "detail": (f"Its compressor is running {observed * 100:.0f}% of the time, where about "
                   f"{profile.normal_duty * 100:.0f}% is normal. The usual culprits are easy to check: "
                   f"a door seal that no longer grips (try closing the door on a sheet of paper), the "
                   f"thermostat set colder than needed, dusty coils at the back, or no gap behind the unit. "
                   f"Please don't switch it off to save power; fixing the cause is what saves money."),
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

    # Shiftable appliances are planned together: each accepted move is added
    # to the day's load, so later moves cannot stack on top of it and push the
    # home past its sanctioned load.  Biggest possible saving is planned first.
    shiftable = [k for k, p in profiles.items()
                 if p.category == SHIFTABLE and tariff.tod_enabled and not sessions[sessions["appliance"] == k].empty]
    cap = (prefs.sanctioned_kw * prefs.load_margin) if prefs.sanctioned_kw else None
    load = day_loads(pred) if shiftable else None
    if shiftable:
        unconstrained = {k: _shift_suggestion(k, sessions[sessions["appliance"] == k], profiles[k],
                                              tariff, prefs, days) for k in shiftable}
        order = sorted(shiftable, key=lambda k: -(unconstrained[k] or {}).get("monthly_saving_exact", 0.0))
        for key in order:
            s = _shift_suggestion(key, sessions[sessions["appliance"] == key], profiles[key],
                                  tariff, prefs, days, load, cap)
            if s is None:
                if unconstrained[key] is not None and cap:
                    notes.append(f"{profiles[key].name}: a cheaper time exists, but running it then "
                                 f"together with your other appliances would go above "
                                 f"{prefs.load_margin * 100:.0f}% of your {prefs.sanctioned_kw:g} kW "
                                 f"sanctioned load, so it is not suggested.")
                continue
            candidates.append(s)

    for key, profile in profiles.items():
        group = sessions[sessions["appliance"] == key]
        if profile.category == SHIFTABLE:
            continue
        elif profile.category == COMFORT and not group.empty:
            candidates.extend(_ac_suggestions(group, profile, tariff, days))
        elif profile.category == ALWAYS_ON:
            s = _fridge_health(pred, profile, tariff, days)
            if s:
                candidates.append(s)

    suggestions, small_shifts = [], []
    for s in candidates:
        profile = profiles[s["appliance"]]
        # Hard guard: the category decides what may be said at all.
        if s["kind"] not in allowed_advice(profile):
            continue
        if s["id"] in prefs.dismissed:
            continue
        if s["monthly_saving_rs"] < prefs.min_monthly_saving_rs:
            if s["kind"] == "shift":
                small_shifts.append(s)
            notes.append(f"{profile.name}: a possible change would save only about "
                         f"Rs. {s['monthly_saving_rs']:.0f} a month, so it is not suggested on its own.")
            continue
        suggestions.append(s)
    # Several small moves can be worth making together.
    if len(small_shifts) >= 2:
        combo = _combined_routine(small_shifts, profiles)
        if combo["monthly_saving_rs"] >= prefs.min_monthly_saving_rs and combo["id"] not in prefs.dismissed:
            suggestions.append(combo)
            covered = {profiles[k].name for k in combo["members"]}
            notes = [n for n in notes if not (n.split(":")[0] in covered and "on its own" in n)]
    for s in suggestions:
        s.pop("_reserve", None)
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


# ----------------------------------------------------------------- plan check
def estimated_savings(baseline_kwh: float, optimised_kwh: float,
                      baseline_hours: float, optimised_hours: float) -> dict:
    """
    Estimated savings = baseline - optimised; percentage = savings / baseline x 100 when the
    baseline is positive (otherwise None, never a made-up 0% or infinity).  Both figures must
    cover the same span of time, or the comparison means nothing and is refused.  A negative
    saving (the plan uses more) is reported as it is.
    """
    for v in (baseline_kwh, optimised_kwh, baseline_hours, optimised_hours):
        if v is None or not np.isfinite(float(v)):
            raise ValueError("Savings need finite numbers.")
    if baseline_kwh < 0 or optimised_kwh < 0:
        raise ValueError("Energy cannot be negative.")
    if abs(float(baseline_hours) - float(optimised_hours)) > 1e-6:
        raise ValueError(f"Baseline covers {baseline_hours:g} h but the plan covers {optimised_hours:g} h; "
                         "they cannot be compared.")
    saving = float(baseline_kwh) - float(optimised_kwh)
    return {"saving_kwh": round(saving, 4),
            "saving_pct": round(saving / baseline_kwh * 100.0, 2) if baseline_kwh > 0 else None,
            "uses_more": saving < 0, "label": "estimate"}


def validate_plan(pred: pd.DataFrame, sessions: pd.DataFrame, suggestions: List[dict],
                  tariff: Optional[Tariff] = None, prefs: Optional[HouseholdPrefs] = None) -> dict:
    """
    Check the suggested plan against the household's own recent days and report
    baseline versus optimised use, cost and peak load, per month.

    Shifting a run moves it in time but does not change its units, so its
    energy saving is zero by construction; the saving is in cost (cheaper
    hours) and, where runs move out of busy hours, in peak load.  Setpoint and
    timer tips save energy; those figures are estimates from BEE guidance and
    are labelled as such.  Every moved run is re-checked against every limit.
    """
    tariff = tariff or Tariff()
    prefs = prefs or HouseholdPrefs()
    profiles = get_profiles(prefs)
    dts = pd.to_datetime(pred["datetime"])
    days = max(1.0, (dts.iloc[-1] - dts.iloc[0]).total_seconds() / 86400.0)
    per_month = DAYS_PER_MONTH / days
    rates = tariff.minute_rates()
    minute_of_day = (dts.dt.hour * 60 + dts.dt.minute).to_numpy()
    mains = pred["mains_kw"].to_numpy(dtype=float)
    base_kwh = float(mains.sum() / 60.0)
    base_cost = float((mains / 60.0 * rates[minute_of_day]).sum())

    # Per-day 15-minute load, before and after moving the runs.
    day_index = dts.dt.normalize()
    slot = minute_of_day // SLOT_MIN
    load = pd.DataFrame({"day": day_index, "slot": slot, "kw": mains}).groupby(["day", "slot"])["kw"].mean().unstack(fill_value=0.0)
    after = load.copy()
    moves = [dict(mv, appliance=mv.get("appliance", s["appliance"]))
             for s in suggestions if s.get("kind") == "shift" for mv in s.get("moves", [])]
    checks, shift_cost_saving, shift_loss_kwh = [], 0.0, 0.0
    cap = (prefs.sanctioned_kw * prefs.load_margin) if prefs.sanctioned_kw else None
    for mv in moves:
        profile = profiles.get(mv["appliance"])
        start, to, dur, kw = int(mv["from"]), int(mv["to"]), int(mv["dur"]), float(mv["kw"])
        problems = []
        lo, hi = profile.allowed_hours[0] * 60, profile.allowed_hours[1] * 60
        if not (lo <= to and to + dur <= hi):
            problems.append("outside allowed hours")
        if profile.noisy and any(_in_quiet(m, prefs.quiet_hours) for m in (to, to + dur - 1)):
            problems.append("in quiet hours")
        if profile.max_advance_min is not None and to < start - profile.max_advance_min:
            problems.append("moved earlier than this appliance allows")
        if profile.max_delay_min is not None and to > start + profile.max_delay_min:
            problems.append("moved later than this appliance allows")
        if profile.category != SHIFTABLE:
            problems.append("not a shiftable appliance")
        day = mv["day"]
        if day in after.index:
            for sl in _slots(start, dur):
                after.loc[day, sl] = max(0.0, after.loc[day, sl] - kw)
            for sl in _slots(to, dur):
                after.loc[day, sl] = after.loc[day, sl] + kw
        loss = float(mv.get("loss_kwh", 0.0) or 0.0)            # heat lost waiting (geyser)
        shift_loss_kwh += loss
        shift_cost_saving += (tariff.run_cost(start, dur, kw) - tariff.run_cost(to, dur, kw)
                              - loss * tariff.rate_at(to / 60.0))
        checks.append({"appliance": profile.name, "from": _hhmm(start), "to": _hhmm(to), "minutes": dur,
                       "ok": not problems, "problems": problems})
    peak_before = float(load.max(axis=1).mean()) if not load.empty else 0.0
    peak_after = float(after.max(axis=1).mean()) if not after.empty else 0.0
    max_after = float(after.max().max()) if not after.empty else 0.0
    # The plan may only add load where the home stays under the cap (or under
    # what was already there before the plan).
    raised = after > load + 1e-9
    over = raised & (after > np.maximum(load, cap if cap is not None else np.inf) + 1e-6)
    cap_ok = not bool(over.to_numpy().any())
    cap_violations = int(over.to_numpy().sum())

    # Energy-saving tips (estimates, not simulated).
    est_kwh = 0.0
    for s in suggestions:
        if s.get("kind") == "duration" and s.get("appliance") == "ac":
            rate = float(tariff.rate) or 1.0
            est_kwh += float(s.get("monthly_saving_rs", 0)) / rate / per_month    # back to the history period
    month = lambda x: round(x * per_month, 2)          # noqa: E731
    shift_saving_m = month(shift_cost_saving)
    est_saving_m = round(sum(float(s.get("monthly_saving_rs", 0)) for s in suggestions
                             if s.get("kind") != "shift"), 2)
    base_month_kwh, base_month_cost = month(base_kwh), month(base_cost)
    opt_month_kwh = round(base_month_kwh - month(est_kwh) + month(shift_loss_kwh), 2)
    opt_month_cost = round(base_month_cost - shift_saving_m - est_saving_m, 2)
    energy = estimated_savings(base_month_kwh, max(0.0, opt_month_kwh), DAYS_PER_MONTH * 24, DAYS_PER_MONTH * 24)
    return {
        "history_days": round(days, 1),
        "baseline": {"kwh_month": base_month_kwh, "cost_month": base_month_cost, "avg_daily_peak_kw": round(peak_before, 2)},
        "optimised": {"kwh_month": opt_month_kwh, "cost_month": opt_month_cost, "avg_daily_peak_kw": round(peak_after, 2)},
        "shift": {"energy_saving_kwh_month": -month(shift_loss_kwh), "cost_saving_month": shift_saving_m,
                  "runs_moved": len(moves), "note": "moving a run does not change its units; an early "
                  "geyser loses a little heat while it waits (estimate), counted here"},
        "estimated_tips": {"energy_saving_kwh_month": month(est_kwh), "cost_saving_month": est_saving_m,
                           "label": "estimate from BEE guidance, not simulated"},
        "energy_saving_kwh_month": round(energy["saving_kwh"], 2),
        "energy_saving_pct": energy["saving_pct"] if energy["saving_pct"] is not None else 0.0,
        "uses_more_energy": energy["uses_more"],
        "savings_label": "estimate",
        "cost_saving_month": round(base_month_cost - opt_month_cost, 2),
        "cost_saving_pct": round((base_month_cost - opt_month_cost) / base_month_cost * 100, 2) if base_month_cost > 0 else 0.0,
        "peak_change_kw": round(peak_after - peak_before, 2),
        "checks": checks,
        "all_constraints_ok": all(c["ok"] for c in checks) and cap_ok,
        "load_cap_kw": None if cap is None else round(cap, 2),
        "load_cap_violations": cap_violations,
        "max_load_after_kw": round(max_after, 2),
    }
