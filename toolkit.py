"""
Household toolkit
=================
The sums behind the pages that answer the other questions a household has
about electricity, beyond "which appliance is running":

  bills     is my bill right, will I cross a slab, why did it change
  safety    faults over time, overload against the sanctioned load, left-on
  upgrades  replace an appliance, rooftop solar, inverter and battery size
  control   monthly goal and streak, schedules, carbon
  family    who switched what, weekly report
  help      complaint letter, outage log

Everything here is plain arithmetic on the detected meter history, with every
assumption a named, editable number.  No function here draws anything.
"""

import calendar
import io
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from appliance_profiles import ALWAYS_ON, DEFAULT_PROFILES
from sessions import flag_long_runs, fridge_duty

# ------------------------------------------------------------------ tariffs
# Starting points only.  Tariff orders change every year and differ by
# supplier, so the Bills page lets the user edit every number.
TARIFF_PRESETS = {
    "Karnataka (BESCOM, domestic)": {
        "slabs": [(None, 5.80)],                       # one rate for all units
        "fixed": [(None, 145.0)],                      # Rs per kW of sanctioned load
        "duty_pct": 9.0, "surcharge_pct": 0.0,
        "note": "One rate for every unit, so there is no slab jump. Fuel adjustment varies by month.",
    },
    "Delhi (domestic)": {
        "slabs": [(200, 3.00), (400, 4.50), (800, 6.50), (1200, 7.00), (None, 8.00)],
        "fixed": [(2, 20.0), (5, 50.0), (15, 100.0), (25, 200.0), (None, 250.0)],
        "duty_pct": 8.0, "surcharge_pct": 0.0,
        "note": "Add your supplier's current PPAC percentage as the surcharge; it changes through the year.",
    },
    "Custom": {
        "slabs": [(100, 4.00), (200, 5.50), (None, 7.00)],
        "fixed": [(None, 100.0)],
        "duty_pct": 5.0, "surcharge_pct": 0.0,
        "note": "Enter the slabs printed on your own bill.",
    },
}
GRID_KG_CO2_PER_KWH = 0.71          # India grid average, CEA baseline database (editable in the app)


def energy_charge(units: float, slabs) -> Tuple[float, List[dict]]:
    """Telescopic slabs: each block of units is charged at its own rate."""
    left, lower, total, lines = float(units), 0.0, 0.0, []
    for upto, rate in slabs:
        if left <= 0:
            break
        width = left if upto is None else max(0.0, min(left, float(upto) - lower))
        if width > 0:
            amount = width * float(rate)
            lines.append({"from": lower, "to": lower + width, "units": width, "rate": float(rate), "amount": amount})
            total += amount
            left -= width
        lower = lower if upto is None else float(upto)
    return total, lines


def fixed_charge(sanctioned_kw: float, tiers) -> float:
    for upto, per_kw in tiers:
        if upto is None or sanctioned_kw <= float(upto):
            return float(per_kw) * float(sanctioned_kw)
    return 0.0


def bill_check(units: float, tariff: dict, sanctioned_kw: float, billed: Optional[float] = None,
               other_charges: float = 0.0) -> dict:
    """Recalculate a bill from its units and compare it with the amount charged."""
    energy, lines = energy_charge(units, tariff["slabs"])
    fixed = fixed_charge(sanctioned_kw, tariff["fixed"])
    surcharge = (energy + fixed) * float(tariff.get("surcharge_pct", 0.0)) / 100.0
    duty = energy * float(tariff.get("duty_pct", 0.0)) / 100.0
    total = energy + fixed + surcharge + duty + float(other_charges)
    out = {"units": float(units), "lines": lines, "energy": energy, "fixed": fixed, "surcharge": surcharge,
           "duty": duty, "other": float(other_charges), "total": total, "billed": billed,
           "difference": None, "verdict": None}
    if billed:
        diff = float(billed) - total
        pct = 100.0 * diff / max(total, 1e-9)
        out["difference"], out["difference_pct"] = diff, pct
        out["verdict"] = ("matches" if abs(pct) <= 3 else "higher" if diff > 0 else "lower")
    return out


def month_position(daily_kwh: pd.Series) -> dict:
    """Units used this calendar month so far and the projection to month end.
    The last entry of `daily_kwh` is today, which is still in progress, so it is projected too."""
    today = pd.Timestamp(daily_kwh.index[-1])
    past = daily_kwh.iloc[:-1]
    this_month = past[(past.index.month == today.month) & (past.index.year == today.year)]
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    pace = float(past.tail(14).mean()) if len(past) else 0.0
    days_left = days_in_month - today.day + 1                # today included
    used = float(this_month.sum())
    return {"today": today, "used": used, "days_done": int(len(this_month)), "days_left": int(days_left),
            "days_in_month": days_in_month, "pace": pace, "projected": used + pace * days_left,
            "last30": float(past.tail(30).sum())}


def slab_warning(position: dict, slabs) -> dict:
    """Where the month is heading relative to the slab boundaries."""
    projected, used = position["projected"], position["used"]
    bounds = [float(u) for u, _ in slabs if u is not None]
    rate_at = lambda x: next(r for u, r in slabs if u is None or x <= float(u))
    out = {"flat": not bounds, "projected": projected, "rate_now": rate_at(max(used, 0.01)),
           "rate_end": rate_at(max(projected, 0.01)), "boundary": None}
    ahead = [b for b in bounds if b > used]
    if not ahead:
        return out
    boundary = ahead[0]
    out["boundary"], out["units_left"] = boundary, boundary - used
    out["will_cross"] = projected > boundary
    out["next_rate"] = rate_at(boundary + 0.01)
    out["allowance"] = (boundary - used) / max(position["days_left"], 1)
    if out["will_cross"] and position["pace"] > 0:
        days_to_cross = (boundary - used) / position["pace"]
        out["cross_day"] = min(position["days_in_month"], int(position["today"].day + days_to_cross))
        over = projected - boundary
        out["extra_cost"] = over * (out["next_rate"] - rate_at(boundary))
    return out


def explain_change(kwh: pd.DataFrame, cost: pd.DataFrame, span: int = 15) -> dict:
    """Why the latest `span` days cost more or less than the `span` before them, by appliance."""
    kwh, cost = kwh.iloc[:-1], cost.iloc[:-1]                # leave out today, still in progress
    span = min(span, len(kwh) // 2)
    if span < 3:
        return {"span": 0, "rows": [], "delta_cost": 0.0, "delta_kwh": 0.0}
    rows = []
    for col in [c for c in kwh.columns if c != "total"]:
        d_kwh = float(kwh[col].tail(span).sum() - kwh[col].iloc[-2 * span:-span].sum())
        d_cost = float(cost[col].tail(span).sum() - cost[col].iloc[-2 * span:-span].sum())
        name = "Everything else" if col == "other" else DEFAULT_PROFILES[col].name
        rows.append({"key": col, "name": name, "delta_kwh": d_kwh, "delta_cost": d_cost,
                     "now_kwh": float(kwh[col].tail(span).sum())})
    rows.sort(key=lambda r: -abs(r["delta_cost"]))
    return {"span": span, "rows": rows,
            "delta_cost": float(cost["total"].tail(span).sum() - cost["total"].iloc[-2 * span:-span].sum()),
            "delta_kwh": float(kwh["total"].tail(span).sum() - kwh["total"].iloc[-2 * span:-span].sum()),
            "now_cost": float(cost["total"].tail(span).sum())}


# ------------------------------------------------------------------- safety
def health_events(pred: pd.DataFrame, sessions: pd.DataFrame, owned: Iterable[str]) -> List[dict]:
    """Every abnormal run and every day the fridge compressor ran too much, newest first."""
    events = []
    for f in flag_long_runs(sessions):
        events.append({"when": pd.Timestamp(f["start"]), "appliance": f["appliance"],
                       "name": DEFAULT_PROFILES[f["appliance"]].name, "kind": "Long run",
                       "text": f"Ran {f['minutes']} min; normal is {f['normal_range_min'][0]}-"
                               f"{f['normal_range_min'][1]} min.", "cost": float(f["cost_rs"])})
    for key in owned:
        prof = DEFAULT_PROFILES[key]
        if prof.category != ALWAYS_ON or prof.normal_duty is None:
            continue
        duty = fridge_duty(pred, key)
        for _, row in duty[duty["duty"] > prof.normal_duty * 1.5].iterrows():
            events.append({"when": pd.Timestamp(row["date"]), "appliance": key, "name": prof.name,
                           "kind": "Compressor overworking",
                           "text": f"Running {row['duty'] * 100:.0f}% of the day; about "
                                   f"{prof.normal_duty * 100:.0f}% is normal.", "cost": 0.0})
    return sorted(events, key=lambda e: e["when"], reverse=True)


def peak_demand(pred: pd.DataFrame, sanctioned_kw: float, owned: Iterable[str], window: int = 15) -> dict:
    """Highest sustained load against the home's sanctioned load (breakers trip on sustained load)."""
    owned = list(owned)
    load = pred["mains_kw"].rolling(window, min_periods=1).mean()
    dts = pd.to_datetime(pred["datetime"])
    idx = load.groupby(dts.dt.normalize()).idxmax()
    days = []
    for day, i in idx.items():
        running = [DEFAULT_PROFILES[k].name for k in owned if bool(pred[f"{k}_on"].iloc[i])]
        days.append({"day": pd.Timestamp(day), "when": dts.iloc[i], "kw": float(load.iloc[i]),
                     "pct": 100.0 * float(load.iloc[i]) / max(sanctioned_kw, 1e-9), "running": running})
    worst = max(days, key=lambda d: d["kw"]) if days else None
    connected = sum(DEFAULT_PROFILES[k].rated_kw for k in owned)
    return {"days": days, "worst": worst, "sanctioned": float(sanctioned_kw),
            "near": [d for d in days if d["pct"] >= 80], "over": [d for d in days if d["pct"] >= 100],
            "connected_kw": float(connected)}


def send_left_on(db, household_id: str, people: List[dict], flags: List[dict], language: str = "en") -> List[str]:
    """One in-app notification per abnormal run, to every household member.  Returns the new messages."""
    import json
    seen = set()
    for entry in db.get_notification_log(household_id, limit=300):
        if entry.get("notification_type") == "left_on":
            try:
                seen.add(json.loads(entry.get("trigger_data_json") or "{}").get("id"))
            except ValueError:
                pass
    sent = []
    for f in flags:
        run_id = f"{f['appliance']}|{pd.Timestamp(f['start']).isoformat()}"
        if run_id in seen:
            continue
        name = DEFAULT_PROFILES[f["appliance"]].name
        subject = f"{name} has been on for {f['minutes']} min"
        for person in people:
            if person.get("email"):
                db.log_notification(household_id, person["email"], person.get("name") or "", "left_on",
                                    "meter", {"id": run_id, "appliance": f["appliance"]},
                                    subject, f["message"], language, status="in-app")
        sent.append(subject)
    return sent


# ----------------------------------------------------------------- upgrades
# Share of energy a current 5-star model saves against a typical older unit,
# and a typical price.  Both are shown to the user and can be changed there.
REPLACEMENTS = {
    "fridge": {"saving": 0.35, "price": 30000, "what": "5-star inverter refrigerator"},
    "ac": {"saving": 0.30, "price": 42000, "what": "5-star inverter split AC"},
    "geyser": {"saving": 0.15, "price": 9000, "what": "5-star insulated storage geyser"},
    "washing_machine": {"saving": 0.25, "price": 28000, "what": "5-star front-load washing machine"},
    "water_pump": {"saving": 0.20, "price": 9000, "what": "high-efficiency pump with a tank level cut-off"},
}


def replacement_table(month_kwh: Dict[str, float], rate: float, overrides: Optional[dict] = None) -> List[dict]:
    rows = []
    for key, base in REPLACEMENTS.items():
        if key not in month_kwh:
            continue
        spec = {**base, **(overrides or {}).get(key, {})}
        saved_kwh = month_kwh[key] * spec["saving"]
        saved_rs = saved_kwh * rate
        rows.append({"key": key, "name": DEFAULT_PROFILES[key].name, "what": spec["what"],
                     "now_kwh": month_kwh[key], "saved_kwh": saved_kwh, "saved_rs": saved_rs,
                     "price": spec["price"], "saving_pct": spec["saving"] * 100,
                     "payback_months": spec["price"] / saved_rs if saved_rs > 0.5 else None})
    return sorted(rows, key=lambda r: r["payback_months"] if r["payback_months"] else 1e9)


def solar_subsidy(kwp: float) -> float:
    """Central subsidy for home rooftop solar: Rs 30,000 per kW for the first 2 kW,
    Rs 18,000 for the third, capped at Rs 78,000."""
    return min(78000.0, 30000.0 * min(kwp, 2.0) + 18000.0 * max(0.0, min(kwp, 3.0) - 2.0))


def solar_estimate(pred: pd.DataFrame, rate: float, kwp: Optional[float] = None, yield_per_kwp: float = 4.0,
                   cost_per_kwp: float = 60000.0, net_metering: bool = True) -> dict:
    dts = pd.to_datetime(pred["datetime"])
    days = max(1.0, len(pred) / 1440.0)
    month_kwh = float(pred["mains_kw"].sum() / 60.0 / days * 30.0)
    sun = (dts.dt.hour >= 9) & (dts.dt.hour < 17)
    day_kwh = float(pred.loc[sun, "mains_kw"].sum() / 60.0 / days * 30.0)
    suggested = max(1.0, round(month_kwh / (yield_per_kwp * 30.0) * 2) / 2)
    kwp = float(kwp) if kwp else suggested
    generated = kwp * yield_per_kwp * 30.0
    used = min(generated, month_kwh if net_metering else day_kwh)
    cost = kwp * cost_per_kwp
    subsidy = solar_subsidy(kwp)
    saving = used * rate
    return {"month_kwh": month_kwh, "day_kwh": day_kwh, "suggested_kwp": suggested, "kwp": kwp,
            "generated": generated, "used": used, "covered_pct": 100.0 * used / max(month_kwh, 1e-9),
            "cost": cost, "subsidy": subsidy, "net_cost": cost - subsidy, "saving_month": saving,
            "payback_years": (cost - subsidy) / (saving * 12.0) if saving > 0 else None,
            "roof_sqft": kwp * 100.0}


def backup_size(devices: List[dict], hours: float, battery_v: float = 12.0, battery_ah: float = 150.0) -> dict:
    """Inverter VA and battery Ah for the chosen devices.  0.8 power factor, 80% usable battery."""
    watts = sum(float(d["kw"]) * 1000.0 for d in devices)
    va = watts / 0.8 * 1.25                                  # power factor, then 25% headroom
    ah = watts * float(hours) / (battery_v * 0.8)
    standard = next((s for s in [600, 900, 1100, 1500, 2000, 2500, 3500, 5000] if s >= va), None)
    return {"watts": watts, "va": va, "standard_va": standard, "ah": ah,
            "batteries": int(np.ceil(ah / battery_ah)) if ah > 0 else 0, "battery_ah": battery_ah,
            "heavy": [d["name"] for d in devices if float(d["kw"]) >= 1.0]}


# ------------------------------------------------------------------ control
def goal_status(daily_cost: pd.Series, target: float) -> dict:
    """Progress against a monthly rupee target, and the run of days at or under the daily share."""
    today = pd.Timestamp(daily_cost.index[-1])
    past = daily_cost.iloc[:-1]
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    this_month = past[(past.index.month == today.month) & (past.index.year == today.year)]
    spent = float(this_month.sum())
    pace = float(past.tail(14).mean()) if len(past) else 0.0
    days_left = days_in_month - today.day + 1
    per_day = float(target) / days_in_month
    streak = 0
    for value in reversed(past.tolist()):
        if value <= per_day:
            streak += 1
        else:
            break
    projected = spent + pace * days_left
    return {"spent": spent, "projected": projected, "target": float(target), "per_day": per_day,
            "allowance": max(0.0, (float(target) - spent)) / max(days_left, 1), "pace": pace,
            "days_left": days_left, "streak": streak, "on_track": projected <= float(target),
            "gap": projected - float(target),
            "best_day": float(past.tail(30).min()) if len(past) else 0.0}


def schedule_report(sessions: pd.DataFrame, windows: Dict[str, Tuple[int, int]], days: int = 7) -> List[dict]:
    """For each appliance with a planned window (minutes from midnight), which recent runs fell outside it."""
    out = []
    if sessions.empty:
        return out
    cutoff = sessions["start"].max().normalize() - pd.Timedelta(days=days - 1)
    recent = sessions[sessions["start"] >= cutoff]
    for key, (lo, hi) in windows.items():
        group = recent[recent["appliance"] == key]
        start = group["start"].dt.hour * 60 + group["start"].dt.minute
        inside = (start >= lo) & (start < hi) if lo <= hi else (start >= lo) | (start < hi)
        outside = group[~inside]
        out.append({"key": key, "name": DEFAULT_PROFILES[key].name, "runs": int(len(group)),
                    "inside": int(inside.sum()), "window": (lo, hi),
                    "kept_pct": 100.0 * float(inside.sum()) / len(group) if len(group) else None,
                    "outside": [{"start": s["start"], "minutes": int(s["span_min"]), "cost": float(s["cost_rs"])}
                                for _, s in outside.iterrows()],
                    "outside_cost": float(outside["cost_rs"].sum())})
    return out


def carbon_kg(kwh: float, factor: float = GRID_KG_CO2_PER_KWH) -> float:
    return float(kwh) * float(factor)


# ------------------------------------------------------------------- family
def member_activity(log: List[dict], device_kw: Dict[str, float], device_names: Dict[str, str],
                    rate: float) -> List[dict]:
    """Who switched what, from the household activity log.  On-time and cost are
    worked out for switched devices by pairing each 'on' with the next 'off'."""
    import json
    seen, events = set(), []
    for e in log:
        if e.get("notification_type") != "appliance_switch":
            continue
        mark = (e.get("subject"), e.get("sent_at"))           # one row per recipient is logged
        if mark in seen:
            continue
        seen.add(mark)
        try:
            data = json.loads(e.get("trigger_data_json") or "{}")
        except ValueError:
            continue
        events.append({"who": e.get("triggered_by") or "Someone", "device": data.get("device"),
                       "on": bool(data.get("on")), "at": pd.Timestamp(e.get("sent_at"))})
    events.sort(key=lambda x: x["at"])
    people: Dict[str, dict] = {}
    open_since: Dict[str, Tuple[pd.Timestamp, str]] = {}
    for ev in events:
        p = people.setdefault(ev["who"], {"name": ev["who"], "switches": 0, "minutes": 0.0, "kwh": 0.0,
                                          "devices": {}, "last": None})
        p["switches"] += 1
        p["last"] = ev["at"]
        label = device_names.get(ev["device"], ev["device"] or "device")
        p["devices"][label] = p["devices"].get(label, 0) + 1
        if ev["on"]:
            open_since[ev["device"]] = (ev["at"], ev["who"])
        elif ev["device"] in open_since:
            since, who = open_since.pop(ev["device"])
            minutes = max(0.0, (ev["at"] - since).total_seconds() / 60.0)
            owner = people[who]
            owner["minutes"] += minutes
            owner["kwh"] += device_kw.get(ev["device"], 0.0) * minutes / 60.0
    rows = []
    for p in people.values():
        top = max(p["devices"].items(), key=lambda kv: kv[1])[0] if p["devices"] else ""
        rows.append({**p, "cost": p["kwh"] * rate, "most_used": top})
    return sorted(rows, key=lambda r: -r["switches"])


def weekly_report_pdf(report: dict) -> bytes:
    """A one-page PDF summary of the last seven days."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=14 * mm, title="EnergyPulse weekly report",
                            author=report["household"])
    base = getSampleStyleSheet()
    ink, mist, amber = colors.HexColor("#141824"), colors.HexColor("#5b6478"), colors.HexColor("#b96d05")
    h1 = ParagraphStyle("h1", parent=base["Title"], fontSize=20, leading=24, alignment=0, textColor=ink, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, leading=14, textColor=amber, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13, textColor=ink)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=mist)

    def table(rows, widths, head=True):
        t = Table(rows, colWidths=widths)
        style = [("FONTSIZE", (0, 0), (-1, -1), 9), ("TEXTCOLOR", (0, 0), (-1, -1), ink),
                 ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#d8dce7")),
                 ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
        if head:
            style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TEXTCOLOR", (0, 0), (-1, 0), mist)]
        t.setStyle(TableStyle(style))
        return t

    story = [Paragraph("EnergyPulse weekly report", h1),
             Paragraph(f"{report['household']} &nbsp;|&nbsp; {report['period']} &nbsp;|&nbsp; "
                       f"prepared {report['prepared']}", small), Spacer(1, 6),
             table([["Units used", "Cost", "Daily average", "Change on the week before"],
                    [f"{report['kwh']:.1f}", f"Rs. {report['cost']:.0f}", f"{report['kwh'] / 7:.1f} units",
                     f"{report['change_pct']:+.0f}%"]], [42 * mm] * 4),
             Paragraph("Where it went", h2),
             table([["Appliance", "Units", "Cost", "Share"]] + [
                 [r["name"], f"{r['kwh']:.2f}", f"Rs. {r['cost']:.0f}", f"{r['share']:.0f}%"]
                 for r in report["appliances"]], [78 * mm, 30 * mm, 30 * mm, 30 * mm]),
             Paragraph("Changes worth making", h2)]
    if report["actions"]:
        for i, a in enumerate(report["actions"], 1):
            story.append(Paragraph(f"{i}. <b>{a['title']}</b>. About Rs. {a['saving_month']:.0f} a month. "
                                   f"<font color='#5b6478'>{a['calculation']}.</font>", body))
    else:
        story.append(Paragraph("Nothing this week would save more than the small-saving limit.", body))
    story.append(Paragraph("Faults and unusual runs", h2))
    if report["events"]:
        for e in report["events"][:5]:
            story.append(Paragraph(f"{e['when']:%a %d %b}: <b>{e['name']}</b>, {e['kind'].lower()}. {e['text']}", body))
    else:
        story.append(Paragraph("None this week.", body))
    if report.get("members"):
        story.append(Paragraph("Household activity", h2))
        story.append(table([["Member", "Switches", "Most used"]] + [
            [m["name"], str(m["switches"]), m["most_used"]] for m in report["members"][:6]],
            [70 * mm, 30 * mm, 68 * mm]))
    story += [Spacer(1, 10),
              Paragraph(f"Carbon: about {report['co2']:.0f} kg of CO2 for the week. "
                        f"Figures come from the simulated main-meter feed, not a physical meter. "
                        f"This file is fingerprinted when it is created; upload it on the Family page, "
                        f"under 'Check a report file', to confirm nobody has changed it.", small)]
    doc.build(story)
    return buffer.getvalue()


# --------------------------------------------------------------------- help
COMPLAINT_KINDS = {
    "Bill amount looks wrong": "I believe the amount charged on my bill is not correct and request a re-check.",
    "Meter may be faulty": "I believe my meter is recording incorrectly and request that it be tested.",
    "Frequent power cuts": "My supply has been interrupted repeatedly and I request that the cause be attended to.",
    "Low or fluctuating voltage": "The supply voltage at my premises is low or unstable and I request an inspection.",
}


def complaint_letter(kind: str, name: str, consumer_no: str, address: str, supplier: str,
                     facts: List[str], extra: str = "", today: Optional[datetime] = None) -> str:
    today = today or datetime.now()
    lines = [f"Date: {today:%d %B %Y}", "", "To", "The Assistant Executive Engineer / Customer Care",
             supplier or "[Name of your electricity supplier]", "",
             f"Subject: {kind}" + (f" - Consumer No. {consumer_no}" if consumer_no else ""), "", "Sir/Madam,", "",
             COMPLAINT_KINDS.get(kind, kind)]
    if facts:
        lines += ["", "The details are:"] + [f"  {i}. {f}" for i, f in enumerate(facts, 1)]
    if extra.strip():
        lines += ["", extra.strip()]
    lines += ["", "I request you to look into this and inform me of the action taken. Please treat this "
                  "letter as a formal complaint and give me a complaint reference number.", "",
              "Yours faithfully,", name or "[Your name]"]
    if consumer_no:
        lines.append(f"Consumer No.: {consumer_no}")
    if address.strip():
        lines.append(address.strip())
    return "\n".join(lines)


def outage_summary(outages: List[dict], today: Optional[pd.Timestamp] = None) -> dict:
    today = today or pd.Timestamp.now()
    frame = pd.DataFrame(outages) if outages else pd.DataFrame(columns=["date", "start", "minutes", "note"])
    if frame.empty:
        return {"count": 0, "hours_month": 0.0, "hours_30": 0.0, "longest": 0, "by_month": {}}
    frame["date"] = pd.to_datetime(frame["date"])
    this_month = frame[(frame["date"].dt.month == today.month) & (frame["date"].dt.year == today.year)]
    last30 = frame[frame["date"] >= today.normalize() - pd.Timedelta(days=29)]
    by_month = (frame.groupby(frame["date"].dt.strftime("%b %Y"), sort=False)["minutes"].sum() / 60.0).round(1)
    return {"count": int(len(frame)), "hours_month": float(this_month["minutes"].sum() / 60.0),
            "count_month": int(len(this_month)), "hours_30": float(last30["minutes"].sum() / 60.0),
            "count_30": int(len(last30)), "longest": int(frame["minutes"].max()), "by_month": by_month.to_dict()}
