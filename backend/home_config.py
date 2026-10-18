"""
Validation of a household's saved setup (home and appliances).

The database is the authoritative copy of each household's setup.  Everything
written to it goes through `validate_home_details`, so a bad value from the
UI (or a tampered session) cannot be stored: numbers are checked against
sensible ranges, text is trimmed, unknown fields are dropped, and duplicate
entries of the same catalogue appliance are merged into one with a quantity.

`usage_factor` turns what the user said about an appliance (hours a day and
how many they have, or a Low/Medium/High level) into the factor the simulated
meter uses, so editing an appliance changes the prediction and the optimisation.
"""
from typing import Any, Dict, List, Tuple

import catalog

HOME_TYPES = ["Apartment", "Independent House", "Villa", "Studio", "Other"]
USAGE_LEVELS = {"Low": 0.6, "Medium": 1.0, "High": 1.5}
MAX_APPLIANCES = 60
LIMITS = {
    "occupants": (1, 30),
    "home_size": (0, 20000),          # sq.ft.; 0 = not given
    "tariff_rate": (0.5, 50.0),       # Rs./kWh
    "quantity": (1, 10),
    "hours_per_day": (0.0, 24.0),
    "rated_w": (0.0, 20000.0),
}
ITEM_FIELDS = {"name", "type", "usage", "icon", "type_key", "model_id", "spec_mode", "size", "age_years",
               "own", "photos", "quantity", "hours_per_day"}


class InvalidSetup(ValueError):
    pass


def _number(value, lo, hi, field, errors, integer=False, default=None):
    try:
        v = float(value)
        if v != v:                                     # NaN
            raise ValueError
    except (TypeError, ValueError):
        if default is not None:
            return default
        errors.append(f"{field} must be a number.")
        return None
    if v < lo or v > hi:
        errors.append(f"{field} must be between {lo:g} and {hi:g}.")
        v = min(hi, max(lo, v))
    return int(round(v)) if integer else v


def clean_item(item: Any, errors: List[str]) -> Dict[str, Any]:
    if not isinstance(item, dict):
        item = {"name": str(item)}
    out = {k: v for k, v in item.items() if k in ITEM_FIELDS}
    name = str(out.get("name") or "").strip()[:40]
    if not name:
        errors.append("Every appliance needs a name.")
        name = "Appliance"
    out["name"] = name
    if out.get("type_key") and catalog.get_type(out["type_key"]) is None:
        out.pop("type_key")
    out["usage"] = out.get("usage") if out.get("usage") in USAGE_LEVELS else "Medium"
    out["quantity"] = _number(out.get("quantity", 1), *LIMITS["quantity"], f"{name}: quantity", errors,
                              integer=True, default=1)
    if out.get("hours_per_day") not in (None, ""):
        out["hours_per_day"] = _number(out["hours_per_day"], *LIMITS["hours_per_day"],
                                       f"{name}: hours per day", errors)
    else:
        out.pop("hours_per_day", None)
    if isinstance(out.get("own"), dict) and out["own"].get("rated_w") not in (None, ""):
        out["own"] = dict(out["own"])
        out["own"]["rated_w"] = _number(out["own"]["rated_w"], *LIMITS["rated_w"],
                                        f"{name}: rated power", errors)
    return out


def validate_home_details(details: Any) -> Tuple[Dict[str, Any], List[str]]:
    """(cleaned copy, list of problems).  Problems are fixed in the copy where possible."""
    if not isinstance(details, dict):
        raise InvalidSetup("The home setup must be a dictionary.")
    errors: List[str] = []
    out = dict(details)
    if out.get("home_type") not in HOME_TYPES:
        if out.get("home_type") is not None:
            errors.append("Unknown home type; set to Other.")
        out["home_type"] = "Other"
    out["occupants"] = _number(out.get("occupants", 4), *LIMITS["occupants"], "People", errors,
                               integer=True, default=4)
    out["home_size"] = _number(out.get("home_size", 0) or 0, *LIMITS["home_size"], "Home size", errors,
                               integer=True, default=0)
    out["tariff_rate"] = _number(out.get("tariff_rate", 8.0), *LIMITS["tariff_rate"], "Electricity rate",
                                 errors, default=8.0)
    out["city"] = str(out.get("city") or "").strip()[:60]
    items = out.get("appliances") or []
    if not isinstance(items, list):
        raise InvalidSetup("The appliance list must be a list.")
    if len(items) > MAX_APPLIANCES:
        errors.append(f"At most {MAX_APPLIANCES} appliances can be saved.")
        items = items[:MAX_APPLIANCES]
    merged: List[Dict[str, Any]] = []
    seen = {}
    for item in items:
        clean = clean_item(item, errors)
        # The same catalogue model added twice is one entry with a quantity.
        key = (clean.get("type_key"), clean.get("model_id"), clean.get("spec_mode", "catalog"), clean["name"])
        if clean.get("type_key") and clean.get("spec_mode", "catalog") == "catalog" and key in seen:
            prev = merged[seen[key]]
            prev["quantity"] = min(LIMITS["quantity"][1], prev["quantity"] + clean["quantity"])
            continue
        seen[key] = len(merged)
        merged.append(clean)
    out["appliances"] = merged
    return out, errors


def typical_hours(item: Dict[str, Any]) -> float:
    t = catalog.get_type(item.get("type_key", "")) if isinstance(item, dict) else None
    return float(t.hours_per_day) if t and t.hours_per_day else 0.0


def usage_factor(item: Dict[str, Any]) -> float:
    """
    How heavily this appliance is used compared with a typical one, for the
    simulated meter.  Hours a day (when given) against the catalogue's typical
    hours, otherwise the Low/Medium/High level; more units run more often.
    """
    if not isinstance(item, dict):
        return 1.0
    typical = typical_hours(item)
    hours = item.get("hours_per_day")
    if hours not in (None, "") and typical > 0:
        factor = float(hours) / typical
    else:
        factor = USAGE_LEVELS.get(item.get("usage", "Medium"), 1.0)
    quantity = int(item.get("quantity", 1) or 1)
    return round(min(4.0, max(0.2, factor * quantity)), 2)


def expected_daily_kwh(item: Dict[str, Any]) -> float:
    """Units a day from the user's own figures: running power x hours x quantity (x duty)."""
    spec = catalog.effective_specs(item)
    if not spec:
        return 0.0
    hours = item.get("hours_per_day")
    hours = float(hours) if hours not in (None, "") else float(spec.get("hours_per_day") or 0)
    t = catalog.get_type(item.get("type_key", ""))
    duty = float(getattr(t, "duty", 1.0) or 1.0)
    return round(float(spec.get("running_w", 0)) * duty * hours * int(item.get("quantity", 1) or 1) / 1000.0, 3)
