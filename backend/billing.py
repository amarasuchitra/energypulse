"""
Real meter readings and the monthly bill, from the household's own meter.

Readings
  The user types (or photographs) the number on their electricity meter.  Units used
  between two readings = the difference.  A reading lower than the last one, or a jump
  larger than the home's connection could draw in that time, is refused.

This month
  Units so far = this month's share of the readings (straight-line between the readings
  either side of the 1st).  Projected month = units so far + the recent daily average for
  the days left.  At least two readings, a day apart, are needed for a daily average.

Bill
  Domestic tariffs in India charge by slab: the first N units at one rate, the next at a
  higher one, plus a fixed charge per kW.  Rates differ by supply company and change every
  year, so the slab table is the household's own and can be edited; the default is an
  example to check against the latest bill.

Gruha Jyothi (Karnataka)
  A home is billed nothing up to its eligible limit = its average monthly units
  (last financial year) plus 10%, at most 200; above the limit it pays for the extra units;
  above 200 units it gets no benefit that month.  The rules are kept editable for the same
  reason.  All bill figures are estimates.
"""
import calendar
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

READINGS_KEY = "meter_readings"
PLAN_KEY = "bill_plan"
MAX_READINGS = 3000
MAX_KW = 25.0                   # a home connection cannot draw more than this
GJ_CAP = 200.0                  # Gruha Jyothi: no benefit above this many units
GJ_MARGIN = 0.10                # eligible = average + 10%

# Example slabs (domestic, per unit).  Check your own latest bill and edit.
DEFAULT_SLABS = [{"upto": 100.0, "rate": 4.75}, {"upto": None, "rate": 7.00}]
DEFAULT_FIXED_PER_KW = 120.0
DEFAULT_SANCTIONED_KW = 2.0


class BadReading(ValueError):
    pass


# ----------------------------------------------------------------- readings
def clean_readings(raw) -> List[dict]:
    """Stored readings, sorted by time, each {"at": Timestamp, "value": float, "source": str}."""
    out = []
    for r in raw or []:
        try:
            at, value = pd.Timestamp(r["at"]), float(r["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            out.append({"at": at, "value": value, "source": str(r.get("source") or "typed")})
    return sorted(out, key=lambda r: r["at"])


def add_reading(raw, value, at, source: str = "typed") -> List[dict]:
    """The readings list with one more, after checks.  Raises BadReading with a plain reason."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise BadReading("Type the number shown on the meter, for example 4521.7.")
    if not math.isfinite(value) or value < 0 or value >= 10_000_000:
        raise BadReading("That is not a meter reading. Type the number shown on the meter.")
    at = pd.Timestamp(at)
    readings = clean_readings(raw)
    before = [r for r in readings if r["at"] < at]
    after = [r for r in readings if r["at"] > at]
    if any(r["at"] == at for r in readings):
        raise BadReading("There is already a reading at that time.")
    if before:
        prev = before[-1]
        if value < prev["value"]:
            raise BadReading(f"The meter only counts up: the last reading was {prev['value']:g} on "
                             f"{prev['at']:%d %b %H:%M}. Check the number.")
        hours = (at - prev["at"]).total_seconds() / 3600.0
        if value - prev["value"] > MAX_KW * max(hours, 1 / 60):
            raise BadReading(f"That is {value - prev['value']:.0f} units in {hours:.1f} hours, more than a home "
                             "connection can use. Check the number (and the decimal point).")
    if after and value > after[0]["value"]:
        raise BadReading(f"A later reading ({after[0]['value']:g} on {after[0]['at']:%d %b}) is lower than this one.")
    readings.append({"at": at, "value": value, "source": source})
    readings.sort(key=lambda r: r["at"])
    return [{"at": str(r["at"]), "value": r["value"], "source": r["source"]} for r in readings[-MAX_READINGS:]]


def _value_at(readings: List[dict], t: pd.Timestamp) -> Optional[float]:
    """Meter value at time t, straight-line between the readings either side (None outside them)."""
    for a, b in zip(readings, readings[1:]):
        if a["at"] <= t <= b["at"]:
            span = (b["at"] - a["at"]).total_seconds()
            f = (t - a["at"]).total_seconds() / span if span else 0.0
            return a["value"] + f * (b["value"] - a["value"])
    if readings and t == readings[-1]["at"]:
        return readings[-1]["value"]
    return None


def daily_average(readings: List[dict], days: int = 14) -> Optional[float]:
    """Units a day over (up to) the last `days` days of readings; None with under a day of readings."""
    if len(readings) < 2:
        return None
    last = readings[-1]
    start_t = max(readings[0]["at"], last["at"] - pd.Timedelta(days=days))
    start_v = _value_at(readings, start_t)
    span = (last["at"] - start_t).total_seconds() / 86400.0
    if start_v is None or span < 1.0:
        return None
    return (last["value"] - start_v) / span


def month_status(readings_raw, now: pd.Timestamp) -> dict:
    """
    {"units_so_far", "daily_avg", "projected", "days_left", "days_in_month", "since", "readings"}
    for the calendar month of `now`.  Figures are None when the readings cannot support them.
    """
    readings = clean_readings(readings_raw)
    now = pd.Timestamp(now)
    month_start = now.normalize().replace(day=1)
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    month_end = month_start + pd.Timedelta(days=days_in_month)
    out = {"units_so_far": None, "daily_avg": daily_average(readings), "projected": None,
           "days_left": None, "days_in_month": days_in_month, "since": None, "readings": len(readings)}
    if len(readings) < 2:
        return out
    last = readings[-1]
    in_month = [r for r in readings if r["at"] >= month_start]
    start_v = _value_at(readings, month_start)
    if start_v is not None:
        since = month_start
    elif in_month:                          # no reading before the 1st: count from the first one this month
        since, start_v = in_month[0]["at"], in_month[0]["value"]
    else:
        return out
    if last["at"] < month_start:
        return out
    out["units_so_far"] = round(last["value"] - start_v, 1)
    out["since"] = since
    left = max(0.0, (month_end - last["at"]).total_seconds() / 86400.0)
    out["days_left"] = round(left, 1)
    if out["daily_avg"] is not None:
        counted = (last["at"] - since).total_seconds() / 86400.0
        missing_start = (since - month_start).total_seconds() / 86400.0      # days before the first reading
        out["projected"] = round(out["units_so_far"] + out["daily_avg"] * (left + missing_start), 1)
        out["counted_days"] = round(counted, 1)
    return out


# ----------------------------------------------------------------- the bill
@dataclass
class BillPlan:
    slabs: List[dict] = field(default_factory=lambda: [dict(s) for s in DEFAULT_SLABS])
    fixed_per_kw: float = DEFAULT_FIXED_PER_KW
    sanctioned_kw: float = DEFAULT_SANCTIONED_KW
    gruha_jyothi: bool = False
    gj_average: Optional[float] = None      # last year's average monthly units

    @classmethod
    def clean(cls, data) -> "BillPlan":
        p = cls()
        if not isinstance(data, dict):
            return p

        def num(v, lo, hi, default):
            try:
                v = float(v)
                return min(hi, max(lo, v)) if math.isfinite(v) else default
            except (TypeError, ValueError):
                return default
        slabs = []
        for s in data.get("slabs") or []:
            if not isinstance(s, dict):
                continue
            upto = s.get("upto")
            upto = None if upto in (None, "", 0) or (isinstance(upto, float) and not math.isfinite(upto)) \
                else num(upto, 1, 100000, None)
            rate = num(s.get("rate"), 0, 100, None)
            if rate is not None:
                slabs.append({"upto": upto, "rate": rate})
        if slabs:
            capped = sorted([s for s in slabs if s["upto"] is not None], key=lambda s: s["upto"])
            open_ = [s for s in slabs if s["upto"] is None][:1]
            p.slabs = capped + (open_ or [{"upto": None, "rate": capped[-1]["rate"]}])
        p.fixed_per_kw = num(data.get("fixed_per_kw"), 0, 1000, p.fixed_per_kw)
        p.sanctioned_kw = num(data.get("sanctioned_kw"), 0.5, 50, p.sanctioned_kw)
        p.gruha_jyothi = bool(data.get("gruha_jyothi", False))
        avg = data.get("gj_average")
        p.gj_average = None if avg in (None, "") else num(avg, 0, 10000, None)
        return p

    def to_dict(self) -> dict:
        return {"slabs": self.slabs, "fixed_per_kw": self.fixed_per_kw, "sanctioned_kw": self.sanctioned_kw,
                "gruha_jyothi": self.gruha_jyothi, "gj_average": self.gj_average}

    @property
    def free_limit(self) -> Optional[float]:
        if not self.gruha_jyothi or self.gj_average is None:
            return None
        return round(min(GJ_CAP, self.gj_average * (1 + GJ_MARGIN)), 1)


def energy_charge(units: float, slabs: List[dict]) -> Tuple[float, List[dict]]:
    """(charge, per-slab breakdown) for `units` on telescopic slabs."""
    units = max(0.0, float(units))
    total, lower, rows = 0.0, 0.0, []
    for s in slabs:
        top = s["upto"] if s["upto"] is not None else math.inf
        in_slab = max(0.0, min(units, top) - lower)
        if in_slab > 0:
            cost = in_slab * s["rate"]
            total += cost
            rows.append({"from": lower, "to": None if top == math.inf else top, "units": round(in_slab, 2),
                         "rate": s["rate"], "amount": round(cost, 2)})
        lower = top
        if units <= lower:
            break
    return round(total, 2), rows


def bill(units: float, plan: BillPlan) -> dict:
    """Estimated bill for a month's units: energy (by slab) + fixed charge, after Gruha Jyothi if it applies."""
    units = max(0.0, float(units))
    fixed = round(plan.fixed_per_kw * plan.sanctioned_kw, 2)
    energy, rows = energy_charge(units, plan.slabs)
    full = round(energy + fixed, 2)
    out = {"units": round(units, 1), "energy": energy, "fixed": fixed, "full": full, "payable": full,
           "slabs": rows, "free_limit": plan.free_limit, "scheme": None}
    limit = plan.free_limit
    if limit is None:
        return out
    if units <= limit:
        out.update(payable=0.0, scheme="free")
    elif units <= GJ_CAP:
        # Pays for the units above the eligible limit (charged at the slab rates they fall in).
        extra = round(energy - energy_charge(limit, plan.slabs)[0], 2)
        out.update(payable=extra, scheme="partly free")
    else:
        out.update(scheme="above 200: no benefit")
    return out


def advice(status: dict, plan: BillPlan) -> Optional[dict]:
    """What the month is heading for, and what it takes to stay in the free limit or under the next slab."""
    proj = status.get("projected")
    if proj is None:
        return None
    b = bill(proj, plan)
    days_left = status.get("days_left") or 0
    so_far = status.get("units_so_far") or 0
    out = {"projected_bill": b, "target": None, "per_day_now": status.get("daily_avg"), "per_day_needed": None,
           "message": ""}
    limit = plan.free_limit
    if limit is not None:
        target, what = limit, "your free limit"
    else:
        edges = [s["upto"] for s in plan.slabs if s["upto"] is not None and s["upto"] > so_far]
        target, what = (edges[0], "the next, dearer slab") if edges else (None, "")
    if target is None or days_left <= 0:
        out["message"] = f"Heading for about {proj:.0f} units, a bill of about Rs. {b['payable']:,.0f}."
        return out
    out["target"] = target
    if proj <= target:
        room = (target - so_far) / days_left if days_left else 0
        out["per_day_needed"] = round(room, 2)
        out["message"] = (f"On track: about {proj:.0f} units this month, within {what} of {target:.0f}. "
                          f"You can use up to {room:.1f} units a day and stay within it.")
        return out
    remaining = target - so_far
    if remaining <= 0:
        out["message"] = (f"This month has already used {so_far:.0f} units, past {what} of {target:.0f}. "
                          f"Heading for about {proj:.0f} units, a bill of about Rs. {b['payable']:,.0f}.")
        return out
    need = remaining / days_left
    cut = (status.get("daily_avg") or 0) - need
    stay = bill(target, plan)
    out["per_day_needed"] = round(need, 2)
    out["message"] = (f"Heading for about {proj:.0f} units, past {what} of {target:.0f}: a bill of about "
                      f"Rs. {b['payable']:,.0f}. Use at most {need:.1f} units a day (about {cut:.1f} less than now) "
                      f"for the last {days_left:.0f} days to stay within it; the bill would then be about "
                      f"Rs. {stay['payable']:,.0f}.")
    return out


# ----------------------------------------------------------------- reading a meter photo
def read_meter_photo(data: bytes) -> Tuple[List[float], Optional[str]]:
    """Candidate readings (largest digit groups first) from a meter photo; ([], reason) when none."""
    try:
        import io
        import re
        import pytesseract
        from PIL import Image, ImageOps
        pytesseract.get_tesseract_version()
    except Exception:
        return [], "Reading photos needs Tesseract OCR, which is not installed here. Type the number instead."
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("L")
        if max(img.size) < 1200:
            img = img.resize((img.width * 2, img.height * 2))
        found = []
        for im in (img, ImageOps.invert(img), ImageOps.autocontrast(img)):
            for psm in (7, 6, 11):
                text = pytesseract.image_to_string(im, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.")
                found += re.findall(r"\d{3,7}(?:\.\d)?", text)
    except Exception:
        return [], "The photo could not be read. Type the number instead."
    seen, out = set(), []
    for f in sorted(found, key=lambda s: (-len(s.replace(".", "")), s)):
        v = float(f)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out[:5], (None if out else "No meter number was found in the photo. Type the number instead.")
