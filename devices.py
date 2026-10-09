"""
Devices in the home
===================
Turns the appliance list the user typed in during setup into what the 3D
home shows.  Every item becomes one of two things:

  detected   one of the six appliances the detection model can pick out of
             the main-meter signal (fridge, AC, geyser, washing machine,
             water pump, microwave).  Its on/off state comes from the meter.
  switched   anything else (TV, fans, lights, iron, laptop...).  The model
             does not know these, so the home shows them from their switch.

    owned, switched, notes = household_devices(home_details)

The home shows exactly what was entered: every listed item is placed, and
nothing that was not listed is added.  An empty list gives an empty home.
"""

import re
from typing import Dict, List, Optional, Tuple

from appliance_profiles import APPLIANCE_KEYS, DEFAULT_PROFILES

NOT_DETECTED = "Not one of the appliances the detection model knows, so it is shown from its switch."

# Words in a typed name -> detector key.  Checked in order.
_DETECTED_WORDS = [
    (r"air ?con|\bac\b|a/c|split ac|window ac", "ac"),
    (r"refrigerator|fridge|freezer", "fridge"),
    (r"washing|washer", "washing_machine"),
    (r"water heater|geyser", "geyser"),
    (r"microwave", "microwave"),
    (r"pump|borewell|\bmotor\b", "water_pump"),
]

# Words in a typed name -> (kind, kW, product, room preference).
_SWITCHED_WORDS = [
    (r"television|\btv\b", ("tv", 0.10, "LED television, 43 inch class", "living")),
    (r"\bfan\b|fans", ("fan", 0.07, "Ceiling fan, 1200 mm sweep", "any")),
    (r"light|lamp|bulb|\bled\b|tube", ("light", 0.04, "LED ceiling lights", "any")),
    (r"cooler", ("generic", 0.18, "Desert air cooler", "living")),
    (r"room heater|heater", ("generic", 1.50, "Room heater", "bedroom")),
    (r"iron", ("generic", 1.00, "Steam iron", "utility")),
    (r"induction", ("generic", 1.80, "Induction cooktop", "kitchen")),
    (r"kettle", ("generic", 1.50, "Electric kettle", "kitchen")),
    (r"toaster", ("generic", 0.80, "Pop-up toaster", "kitchen")),
    (r"mixer|grinder|blender|juicer", ("generic", 0.50, "Mixer grinder", "kitchen")),
    (r"oven|otg", ("generic", 1.20, "Oven toaster grill", "kitchen")),
    (r"dishwasher", ("generic", 1.20, "Dishwasher", "kitchen")),
    (r"chimney|exhaust", ("generic", 0.20, "Kitchen chimney", "kitchen")),
    (r"purifier|\bro\b", ("generic", 0.04, "Water purifier", "kitchen")),
    (r"laptop", ("generic", 0.065, "Laptop charger", "bedroom")),
    (r"computer|desktop|\bpc\b", ("generic", 0.15, "Desktop computer", "bedroom")),
    (r"router|wi-?fi|modem", ("generic", 0.01, "Wi-Fi router", "living")),
    (r"charger|phone", ("generic", 0.01, "Phone charger", "bedroom")),
    (r"sewing", ("generic", 0.10, "Sewing machine motor", "bedroom")),
    (r"vacuum", ("generic", 1.00, "Vacuum cleaner", "utility")),
    (r"dryer", ("generic", 1.20, "Hair or clothes dryer", "utility")),
    (r"speaker|music|console|playstation|xbox|set.?top", ("generic", 0.08, "Home entertainment unit", "living")),
]

# A typed name that matches nothing falls back on the category chosen in setup.
_BY_TYPE = {
    "Cooling": (0.15, "Cooling appliance", "living"),
    "Heating": (1.00, "Heating appliance", "utility"),
    "Kitchen": (0.50, "Kitchen appliance", "kitchen"),
    "Laundry": (0.40, "Laundry appliance", "utility"),
    "Electronics": (0.08, "Electronic device", "living"),
    "Lighting": (0.04, "Light fitting", "any"),
    "Other": (0.10, "Household device", "any"),
}

# Where things can stand in the model (x, z in house units; see home.js ROOMS).
_ROOM_ORDER = ["living", "bedroom", "kitchen", "utility", "bathroom"]
_SLOTS = {
    "fan": {"living": (3.3, 2.75), "bedroom": (2.5, 7.4), "kitchen": (9.4, 2.1), "utility": (10.2, 6.6)},
    "light": {"living": (5.3, 3.6), "bedroom": (4.1, 5.7), "kitchen": (9.2, 3.3),
              "bathroom": (7.0, 7.1), "utility": (9.4, 7.9)},
    "tv": {"living": (2.2, 0.21)},
    "generic": {"living": [(0.5, 0.75), (4.4, 4.35), (6.35, 3.3)], "bedroom": [(0.6, 8.3), (4.0, 8.5)],
                "kitchen": [(9.0, 3.4), (10.6, 3.5)], "utility": [(9.2, 7.0), (11.2, 6.3)],
                "bathroom": [(7.9, 6.2)]},
}
_ROOM_NAMES = {"living": "Living room", "bedroom": "Bedroom", "kitchen": "Kitchen",
               "utility": "Utility", "bathroom": "Bathroom"}
_FIXED_KEYS = {("fan", "living"): "fan_living", ("fan", "bedroom"): "fan_bedroom"}

_EXTRA_ROOM = {"ac": "living", "fridge": "kitchen", "microwave": "kitchen", "geyser": "bathroom",
               "washing_machine": "utility", "water_pump": "utility"}


def _item_name(item) -> Tuple[str, str]:
    if isinstance(item, dict):
        return str(item.get("name", "")).strip(), str(item.get("type", "Other") or "Other")
    return str(item).strip(), "Other"


def _item_spec(item) -> Optional[dict]:
    """The catalogue specification of a picked appliance (None for typed-in items)."""
    from catalog import specs
    return specs(item.get("model_id", "")) if isinstance(item, dict) else None


_CATEGORY_ROOM = {"Cooling": "living", "Kitchen": "kitchen", "Laundry": "utility", "Water and heating": "bathroom",
                  "Entertainment and computing": "living", "Lighting": "any"}


def appliance_specs(home_details: Optional[dict]) -> Dict[str, dict]:
    """Specification of the first unit of each meter-detected appliance, by detector key."""
    out = {}
    for item in (home_details or {}).get("appliances", []) or []:
        spec = _item_spec(item)
        key = (spec or {}).get("detector") or (None if spec else detected_key(_item_name(item)[0]))
        if spec and key and key not in out:
            out[key] = spec
    return out


def household_ratings(home_details: Optional[dict]) -> Dict[str, float]:
    """Running power in kW of each picked meter-detected appliance, for the simulated home."""
    return {k: round(v["running_w"] / 1000.0, 3) for k, v in appliance_specs(home_details).items()}


def detected_key(name: str) -> Optional[str]:
    low = name.lower()
    for pattern, key in _DETECTED_WORDS:
        if re.search(pattern, low):
            return key
    return None


def _expand(name: str, kind_type: str) -> List[Tuple[str, str]]:
    """The 'Lights & Fans' preset stands for four separate things."""
    low = name.lower()
    if re.search(r"light", low) and re.search(r"fan", low):
        return [("Living room fan", kind_type), ("Bedroom fan", kind_type),
                ("Living room lights", kind_type), ("Bedroom lights", kind_type)]
    return [(name, kind_type)]


class _Placer:
    def __init__(self):
        self.used = set()

    def take(self, kind: str, prefer: str, name: str):
        low = name.lower()
        named = next((r for r in _ROOM_ORDER if r in low), None)
        order = [r for r in [named, prefer if prefer != "any" else None, *_ROOM_ORDER] if r]
        table = _SLOTS[kind]
        for room in dict.fromkeys(order):
            spots = table.get(room)
            if spots is None:
                continue
            for i, spot in enumerate(spots if isinstance(spots, list) else [spots]):
                if (kind, room, i) not in self.used:
                    self.used.add((kind, room, i))
                    return room, spot
        return None, None


def household_devices(home_details: Optional[dict]):
    """
    Returns (owned, switched, notes).
      owned     tuple of detector keys, in the standard order
      switched  list of dicts: key, name, product, kw, kind, pos, room, description
      notes     plain sentences about items that were merged or could not be placed
    Only listed items are returned; nothing is added on the household's behalf.
    """
    raw = (home_details or {}).get("appliances", []) or []
    items = []
    for item in raw:
        name, kind_type = _item_name(item)
        spec = _item_spec(item)
        if spec:
            items.append((name, kind_type, spec))
        elif name:
            items.extend((n, k, None) for n, k in _expand(name, kind_type))

    found, switched, notes = [], [], []
    placer, counts = _Placer(), {}
    for name, kind_type, spec in items:
        key = spec["detector"] if spec else detected_key(name)
        if key and key not in found:
            found.append(key)
            continue
        low = name.lower()
        match = next((spec for pattern, spec in _SWITCHED_WORDS if re.search(pattern, low)), None)
        if key:
            # A second unit of a detectable type: the meter model follows one per
            # home, so the extra one is still shown, from its switch.
            prof = DEFAULT_PROFILES[key]
            kind, kw, product, prefer = "generic", prof.rated_kw, prof.product, _EXTRA_ROOM.get(key, "any")
            if spec:
                kw, product = spec["running_w"] / 1000.0, f"{spec['type_name']}, {spec['label']}"
            name = f"{name} {sum(1 for d in switched if d.get('extra_of') == key) + 2}"
            low = name.lower()
            notes.append(f"{name}: the meter model follows one {prof.name.lower()} per home, "
                         f"so this extra one is shown from its switch.")
        elif spec:
            kind, kw = spec["kind"], spec["running_w"] / 1000.0
            product = f"{spec['type_name']}, {spec['label']}"
            prefer = "any" if spec["kind"] in ("fan", "light") else _CATEGORY_ROOM.get(spec["category"], "any")
        elif match:
            kind, kw, product, prefer = match
        else:
            kw, product, prefer = _BY_TYPE.get(kind_type, _BY_TYPE["Other"])
            kind = "light" if kind_type == "Lighting" else "generic"
        room, pos = placer.take(kind, prefer, name)
        if pos is None and kind != "generic":
            kind = "generic"
            room, pos = placer.take(kind, prefer, name)
        if pos is None:
            notes.append(f"{name}: there is no free place left in the model, so it is listed but not drawn.")
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", low).strip("_")[:24] or "device"
        counts[slug] = counts.get(slug, 0) + 1
        dev_key = _FIXED_KEYS.get((kind, room)) or f"dev_{slug}_{counts[slug]}"
        label = name if room is None or _ROOM_NAMES[room].lower() in low or kind == "tv" and room == "living" \
            else f"{name} ({_ROOM_NAMES[room].lower()})"
        switched.append({
            "specs": spec,
            "extra_of": key,
            "key": dev_key, "name": label[:48], "product": product, "kw": round(float(kw), 3),
            "kind": kind, "pos": [float(pos[0]), float(pos[1])], "room": _ROOM_NAMES[room],
            "description": f"{product}, about {_watts(kw)} while running. In the {_ROOM_NAMES[room].lower()}. {NOT_DETECTED}",
        })
    owned = tuple(k for k in APPLIANCE_KEYS if k in found)
    if switched and not owned:
        notes.append("None of the items you listed is one the meter model can detect, so they are all "
                     "shown from their switches and the meter shows only the background load.")
    return owned, switched, notes


def _watts(kw: float) -> str:
    return f"{kw * 1000:.0f} W" if kw < 1 else f"{kw:.1f} kW"
