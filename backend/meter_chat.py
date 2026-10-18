"""
Answers from the household's own meter
======================================
The Energy Chat page answers questions about use, cost, forecast, savings and
faults from the same simulated-meter figures every other page shows, so the
chat can never quote a different home.  Each answer is assembled from numbers
already worked out elsewhere (toolkit.py, daily_brief.py); nothing is invented.

    text = answer(question, base, brief, owned, switched)     # None = not a meter question
"""

import re
from typing import Optional

import pandas as pd

import toolkit as tk
from appliance_profiles import ALWAYS_ON, DEFAULT_PROFILES
from scheduler import explain_refusal

SOURCE = "\n\n_From your simulated meter._"
_NAMES = {
    "ac": r"\bac\b|a/c|air ?con", "fridge": r"fridge|refrigerator|freezer", "geyser": r"geyser|water heater",
    "washing_machine": r"washing|washer|laundry", "water_pump": r"pump|motor", "microwave": r"microwave|oven",
}


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _appliance(text: str, owned) -> Optional[str]:
    return next((k for k, pattern in _NAMES.items() if k in owned and _has(text, pattern)), None)


def _breakdown(base: dict, owned, days: int = 7) -> str:
    kwh, cost = base["kwh"].iloc[:-1].tail(days), base["cost"].iloc[:-1].tail(days)
    total = float(kwh["total"].sum())
    rows = sorted(((("Everything else (lights, fans, TV, standby)" if c == "other" else DEFAULT_PROFILES[c].name),
                    float(kwh[c].sum()), float(cost[c].sum())) for c in kwh.columns if c != "total"),
                  key=lambda r: -r[1])
    lines = [f"- **{n}**: {u:.1f} units, Rs. {c:.0f} ({100 * u / max(total, 1e-9):.0f}%)" for n, u, c in rows]
    named = [r for r in rows if not r[0].startswith("Everything else")]
    head = (f"Over the last {len(kwh)} days the biggest single appliance was the **{named[0][0].lower()}** "
            f"at {named[0][1]:.1f} units." if named else "No meter-detected appliance is listed for this home.")
    return f"{head} The home used {total:.1f} units, Rs. {float(cost['total'].sum()):.0f}, in all:\n\n" + "\n".join(lines)


def _one_appliance(key: str, base: dict) -> str:
    prof = DEFAULT_PROFILES[key]
    kwh, cost = base["kwh"].iloc[:-1], base["cost"].iloc[:-1]
    days = len(kwh)
    text = (f"**{prof.name}** ({prof.product}). Over the last {days} days it used {kwh[key].sum():.1f} units, "
            f"Rs. {cost[key].sum():.0f}, which is {100 * kwh[key].sum() / max(kwh['total'].sum(), 1e-9):.0f}% of the home. "
            f"That is about {kwh[key].mean():.2f} units a day.")
    runs = base["sessions"][base["sessions"]["appliance"] == key]
    if prof.category != ALWAYS_ON and len(runs):
        start = (runs["start"].dt.hour * 60 + runs["start"].dt.minute).median()
        text += (f" It ran {len(runs)} times, usually starting about {int(start) // 60:02d}:{int(start) % 60:02d} "
                 f"for about {int(runs['span_min'].median())} minutes.")
    return text


def answer(question: str, base: dict, brief: Optional[dict], owned, switched) -> Optional[str]:
    q = " " + (question or "").lower().strip() + " "
    brief = brief or {}
    key = _appliance(q, owned)

    if key and _has(q, r"switch(ed)? off|turn(ed)? off|shut"):
        refusal = explain_refusal(key, "switch_off")
        if refusal:
            return refusal + SOURCE
    if _has(q, r"weather|temperature|how hot|degrees"):
        if brief.get("temp_today") is None:
            return "No temperature is available for today." + SOURCE
        return (f"A high of {brief['temp_today']:.0f} C is expected today"
                + (f" in {brief['city']}" if brief.get("city") else "")
                + (f"; yesterday reached {brief['temp_yesterday']:.0f} C" if brief.get("temp_yesterday") is not None else "")
                + ". The weather is simulated, and the air-conditioner forecast follows it." + SOURCE)
    if _has(q, r"fault|wrong with|broken|left on|safe|overload|trip"):
        events = tk.health_events(base["history"], base["sessions"], owned)
        if not events:
            return "No faults or abnormally long runs were found in the last 30 days." + SOURCE
        lines = [f"- {e['when']:%a %d %b}: **{e['name']}**, {e['kind'].lower()}. {e['text']}" for e in events[:5]]
        return f"{len(events)} fault(s) or long runs in the last 30 days. Most recent:\n\n" + "\n".join(lines) + SOURCE
    if _has(q, r"\bwhy\b|went up|increase|higher than|more than (last|usual)"):
        why = tk.explain_change(base["kwh"], base["cost"])
        if not why["span"]:
            return "There is not enough meter history yet to compare two periods." + SOURCE
        lines = [f"- **{r['name']}**: {r['delta_kwh']:+.1f} units, {'+' if r['delta_cost'] >= 0 else '-'}Rs. {abs(r['delta_cost']):.0f}"
                 for r in why["rows"] if abs(r["delta_cost"]) >= 1][:5]
        return (f"The last {why['span']} days cost Rs. {abs(why['delta_cost']):.0f} "
                f"{'more' if why['delta_cost'] > 0 else 'less'} than the {why['span']} days before. By cause:\n\n"
                + "\n".join(lines) + SOURCE)
    if _has(q, r"save|saving|reduce|cut |lower|tip|suggest|advice|optimi"):
        actions = brief.get("actions") or []
        if not actions:
            return ("Nothing would save more than Rs. 30 a month right now, so no change is suggested." + SOURCE)
        lines = [f"{i}. **{a['title']}**: about Rs. {a['saving_month']:.0f} a month. _{a['calculation']}._"
                 for i, a in enumerate(actions, 1)]
        return (f"Changes worth making, biggest saving first (about Rs. {brief.get('saving_month', 0):.0f} a month together):\n\n"
                + "\n".join(lines) + SOURCE)
    if key:
        return _one_appliance(key, base) + SOURCE
    if _has(q, r"forecast|predict|expect|tomorrow|today|tonight"):
        if not brief:
            return None
        lines = [f"- {a['name']}: {a['kwh']:.2f} units, Rs. {a['cost']:.0f}" for a in brief["appliances"]]
        return (f"Today is expected to use **{brief['total_kwh']:.1f} units**, about **Rs. {brief['total_cost']:.0f}** "
                f"(likely {brief['low_kwh']:.1f} to {brief['high_kwh']:.1f} units). On recent days this forecast was off by "
                f"{brief['error_pct']:.1f}%.\n\n" + "\n".join(lines)
                + f"\n- Everything else: {brief['other_kwh']:.2f} units, Rs. {brief['other_cost']:.0f}" + SOURCE)
    if _has(q, r"bill|month|cost|spend|spent|rupee|rs\b"):
        pos = tk.month_position(base["kwh"]["total"])
        rate = base["avg_rate"]
        return (f"This month so far: {pos['used']:.0f} units over {pos['days_done']} day(s). At the pace of the last "
                f"two weeks ({pos['pace']:.1f} units a day) the month ends near **{pos['projected']:.0f} units**, about "
                f"**Rs. {pos['projected'] * rate:.0f}** in energy charges at your average Rs. {rate:.2f} a unit. "
                f"The My Bills page adds fixed charges and duty for your tariff." + SOURCE)
    if _has(q, r"most|highest|biggest|top|which appliance|where .* go|break ?down|consum|usage|using|units"):
        return _breakdown(base, owned) + SOURCE
    if _has(q, r"appliance|device|what do i have|list"):
        names = [DEFAULT_PROFILES[k].name for k in owned] + [d["name"] for d in switched]
        return ("Your home has: " + ", ".join(names) + ". The first "
                f"{len(owned)} are found from the meter; the rest are shown from their switch." + SOURCE)
    return None
