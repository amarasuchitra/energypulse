"""
Appliance catalogue
===================
The list the user picks appliances from, with a specification for each model
class: rated power, typical running power, standby power, supply voltage,
current, power factor, star rating and how much energy it uses in a year.

The figures are TYPICAL for that class of model sold in India (single-phase
230 V, 50 Hz).  A real unit's rating plate gives the exact values, and the
app says so wherever these are shown.

    for cat in CATEGORIES: ...                 # category -> appliance types
    spec = specs("ac_15t_inv_5s")              # full specification of one model class
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

SUPPLY_V = 230.0
SUPPLY_HZ = 50


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    rated_w: float          # the most it draws (rating plate)
    running_w: float        # what it draws on average while running
    standby_w: float = 0.0  # plugged in and switched off at the appliance
    pf: float = 0.95        # power factor
    star: Optional[int] = None
    capacity: str = ""
    note: str = ""


@dataclass(frozen=True)
class ApplianceType:
    key: str
    name: str
    category: str
    hours_per_day: float            # typical use, for the yearly estimate
    models: List[Model] = field(default_factory=list)
    detector: Optional[str] = None  # appliance key the meter model can detect
    kind: str = "generic"           # how the 3D home draws it: fan, light, tv, generic
    duty: float = 1.0               # share of "on" time it actually draws power (fridge compressor)
    idle_reason: str = ""           # why it can be on but drawing little


M = Model
CATALOG: List[ApplianceType] = [
    # ---------------------------------------------------------------- cooling
    ApplianceType("ac", "Air conditioner", "Cooling", 7, [
        M("ac_1t_window_3s", "1 ton window, 3-star", 1150, 1000, 2, 0.92, 3, "1 ton (3.5 kW cooling)"),
        M("ac_15t_split_3s", "1.5 ton split, 3-star fixed speed", 1700, 1500, 3, 0.93, 3, "1.5 ton (5.2 kW cooling)"),
        M("ac_15t_inv_5s", "1.5 ton split, 5-star inverter", 1650, 1150, 3, 0.97, 5, "1.5 ton (5.2 kW cooling)",
          "An inverter compressor slows down instead of stopping, so it draws well below its rating most of the time."),
        M("ac_2t_inv_3s", "2 ton split, 3-star inverter", 2250, 1800, 3, 0.96, 3, "2 ton (7.0 kW cooling)"),
    ], detector="ac", idle_reason="the room has reached the set temperature, so the compressor is resting"),
    ApplianceType("fan", "Ceiling fan", "Cooling", 12, [
        M("fan_std", "1200 mm, standard motor", 75, 70, 0, 0.90, None, "1200 mm sweep"),
        M("fan_bldc", "1200 mm, BLDC 5-star", 32, 28, 0.5, 0.95, 5, "1200 mm sweep",
          "A BLDC motor uses less than half the power of a standard fan at the same speed."),
    ], kind="fan"),
    ApplianceType("cooler", "Air cooler", "Cooling", 8, [
        M("cooler_desert", "Desert cooler, 50 L", 190, 170, 0, 0.90, None, "50 L tank"),
        M("cooler_personal", "Personal cooler, 20 L", 120, 100, 0, 0.90, None, "20 L tank"),
    ]),
    # ---------------------------------------------------------------- kitchen
    ApplianceType("fridge", "Refrigerator", "Kitchen", 24, [
        M("fridge_sd_190", "Single door 190 L, 2-star", 140, 100, 0, 0.85, 2, "190 L"),
        M("fridge_dd_250", "Double door 250 L, 3-star inverter", 150, 110, 0, 0.90, 3, "250 L"),
        M("fridge_ff_340", "Frost-free 340 L, 3-star", 190, 150, 0, 0.85, 3, "340 L"),
        M("fridge_sbs_600", "Side-by-side 600 L, 3-star", 260, 200, 0, 0.85, 3, "600 L"),
    ], detector="fridge", duty=0.4, idle_reason="the cabinet is cold enough, so the compressor is resting"),
    ApplianceType("microwave", "Microwave oven", "Kitchen", 0.3, [
        M("mw_solo_20", "Solo 20 L", 1200, 1150, 2, 0.95, None, "20 L, 800 W output"),
        M("mw_conv_28", "Convection 28 L", 1450, 1400, 3, 0.95, None, "28 L, 900 W output"),
    ], detector="microwave"),
    ApplianceType("induction", "Induction cooktop", "Kitchen", 1, [
        M("induction_1", "Single zone", 2000, 1600, 1, 0.98, None, "1 zone"),
    ]),
    ApplianceType("kettle", "Electric kettle", "Kitchen", 0.2, [M("kettle_15", "1.5 L", 1500, 1500, 0, 1.0, None, "1.5 L")]),
    ApplianceType("mixer", "Mixer grinder", "Kitchen", 0.3, [M("mixer_750", "750 W, 3 jars", 750, 500, 0, 0.85, None, "3 jars")]),
    ApplianceType("otg", "Oven toaster grill", "Kitchen", 0.3, [M("otg_28", "28 L", 1500, 1200, 0, 1.0, None, "28 L")]),
    ApplianceType("chimney", "Kitchen chimney", "Kitchen", 1.5, [M("chimney_60", "60 cm, auto-clean", 200, 160, 1, 0.90, None, "1200 m3/h")]),
    ApplianceType("purifier", "Water purifier (RO)", "Kitchen", 2, [M("ro_7", "RO + UV, 7 L", 40, 25, 2, 0.90, None, "7 L tank")]),
    ApplianceType("dishwasher", "Dishwasher", "Kitchen", 1.5, [M("dw_12", "12 place settings", 1800, 900, 1, 0.98, 4, "12 place settings")]),
    # ---------------------------------------------------------------- laundry and cleaning
    ApplianceType("washing_machine", "Washing machine", "Laundry", 1, [
        M("wm_semi_7", "Semi-automatic 7 kg", 380, 300, 1, 0.85, 5, "7 kg"),
        M("wm_top_7", "Fully automatic top-load 7 kg", 450, 380, 1, 0.88, 5, "7 kg"),
        M("wm_front_7", "Front-load 7 kg with heater", 2000, 500, 1, 0.95, 5, "7 kg",
          "The heater draws about 2 kW for a few minutes on a hot wash; the motor draws far less."),
    ], detector="washing_machine", idle_reason="it is between wash stages (soaking or draining)"),
    ApplianceType("iron", "Electric iron", "Laundry", 0.4, [M("iron_steam", "Steam iron", 1250, 750, 0, 1.0, None, "",
                                                             "The thermostat switches the heater on and off.")]),
    ApplianceType("vacuum", "Vacuum cleaner", "Laundry", 0.2, [M("vac_1200", "Canister, 1200 W", 1200, 1000, 0, 0.95, None, "")]),
    # ---------------------------------------------------------------- water and heating
    ApplianceType("geyser", "Water heater (geyser)", "Water and heating", 0.7, [
        M("geyser_15", "Storage 15 L, 5-star", 2000, 2000, 0, 1.0, 5, "15 L"),
        M("geyser_25", "Storage 25 L, 5-star", 2000, 2000, 0, 1.0, 5, "25 L"),
        M("geyser_instant_3", "Instant 3 L", 3000, 3000, 0, 1.0, None, "3 L"),
    ], detector="geyser", idle_reason="the water is already hot, so the thermostat has cut the heater"),
    ApplianceType("water_pump", "Water pump", "Water and heating", 0.7, [
        M("pump_05", "0.5 HP", 550, 500, 0, 0.80, None, "0.5 HP (370 W output)"),
        M("pump_1", "1 HP", 900, 800, 0, 0.82, None, "1 HP (750 W output)"),
    ], detector="water_pump"),
    ApplianceType("heater", "Room heater", "Water and heating", 2, [M("heater_fan", "Fan heater, 2 kW", 2000, 1500, 0, 1.0, None, "",
                                                                      "Runs at full power only until the room warms up.")]),
    # ---------------------------------------------------------------- entertainment and computing
    ApplianceType("tv", "Television", "Entertainment and computing", 5, [
        M("tv_32", "32 inch LED", 50, 45, 0.5, 0.95, 4, "32 inch"),
        M("tv_43", "43 inch LED", 85, 75, 0.5, 0.95, 4, "43 inch"),
        M("tv_55", "55 inch 4K LED", 130, 110, 0.5, 0.95, 3, "55 inch"),
    ], kind="tv"),
    ApplianceType("settop", "Set-top box", "Entertainment and computing", 6, [M("stb_hd", "HD set-top box", 15, 12, 8, 0.6, None, "",
                                                                                "Draws about 8 W even in standby.")]),
    ApplianceType("speaker", "Sound bar", "Entertainment and computing", 3, [M("soundbar", "2.1 sound bar", 40, 20, 1, 0.9, None, "")]),
    ApplianceType("router", "Wi-Fi router", "Entertainment and computing", 24, [M("router_ac", "Dual-band router", 12, 9, 0, 0.6, None, "")]),
    ApplianceType("laptop", "Laptop", "Entertainment and computing", 6, [M("laptop_65", "65 W charger", 65, 40, 1, 0.9, None, "")]),
    ApplianceType("desktop", "Desktop computer", "Entertainment and computing", 5, [M("desktop", "Desktop with monitor", 250, 150, 2, 0.9, None, "")]),
    ApplianceType("charger", "Phone charger", "Entertainment and computing", 3, [M("charger_20", "20 W fast charger", 20, 10, 0.1, 0.6, None, "")]),
    # ---------------------------------------------------------------- lighting
    ApplianceType("lights", "Room lights", "Lighting", 6, [
        M("lights_led_4", "4 LED bulbs, 9 W each", 36, 36, 0, 0.9, None, "4 x 9 W"),
        M("lights_tube_2", "2 LED tube lights, 20 W each", 40, 40, 0, 0.95, None, "2 x 20 W"),
        M("lights_cfl_4", "4 CFL bulbs, 15 W each", 60, 60, 0, 0.6, None, "4 x 15 W"),
    ], kind="light"),
]
_TYPES: Dict[str, ApplianceType] = {t.key: t for t in CATALOG}
_MODELS: Dict[str, tuple] = {m.id: (t, m) for t in CATALOG for m in t.models}
CATEGORIES: List[str] = list(dict.fromkeys(t.category for t in CATALOG))


def types_in(category: str) -> List[ApplianceType]:
    return [t for t in CATALOG if t.category == category]


def get_type(key: str) -> Optional[ApplianceType]:
    return _TYPES.get(key)


def default_model(type_key: str) -> Model:
    t = _TYPES[type_key]
    return t.models[min(1, len(t.models) - 1)] if t.key in ("ac", "fridge", "tv", "water_pump", "washing_machine") \
        else t.models[0]


def specs(model_id: str) -> Optional[dict]:
    """Every figure the app shows for one model class."""
    if model_id not in _MODELS:
        return None
    t, m = _MODELS[model_id]
    yearly = m.running_w * t.duty * t.hours_per_day * 365 / 1000.0
    return {
        "model_id": m.id, "type": t.key, "type_name": t.name, "category": t.category, "label": m.label,
        "rated_w": m.rated_w, "running_w": m.running_w, "standby_w": m.standby_w,
        "voltage_v": SUPPLY_V, "frequency_hz": SUPPLY_HZ, "power_factor": m.pf,
        "current_a": round(m.rated_w / (SUPPLY_V * m.pf), 2),
        "running_current_a": round(m.running_w / (SUPPLY_V * m.pf), 2),
        "star": m.star, "capacity": m.capacity, "note": m.note,
        "hours_per_day": t.hours_per_day, "yearly_kwh": round(yearly, 0),
        "detector": t.detector, "kind": t.kind, "idle_reason": t.idle_reason,
    }


def spec_lines(spec: dict) -> List[tuple]:
    """(label, value) rows for display."""
    rows = [("Model class", spec["label"])]
    if spec.get("capacity"):
        rows.append(("Capacity", spec["capacity"]))
    rows += [
        ("Rated power", f"{spec['rated_w']:.0f} W"),
        ("Typical running power", f"{spec['running_w']:.0f} W"),
        ("Supply", f"{spec['voltage_v']:.0f} V AC, {spec['frequency_hz']} Hz, single phase"),
        ("Rated current", f"{spec['current_a']:.2f} A"),
        ("Power factor", f"{spec['power_factor']:.2f}"),
        ("Standby power", f"{spec['standby_w']:g} W" if spec["standby_w"] else "none"),
        ("Star rating (BEE)", f"{spec['star']}-star" if spec.get("star") else "not rated"),
        ("Energy per year", f"about {spec['yearly_kwh']:.0f} units at {spec['hours_per_day']:g} h a day"),
    ]
    return rows


def efficient_saving(spec: dict) -> float:
    """
    Share of energy a 5-star model of the same class would save.  Each star
    step is roughly 8% less energy on the BEE scale; an estimate, not a test result.
    """
    star = spec.get("star")
    if not star or star >= 5:
        return 0.0
    return round(1 - 0.92 ** (5 - star), 3)


def item_for(type_key: str, model_id: Optional[str] = None, usage: str = "Medium") -> dict:
    """The setup-list entry for one chosen appliance."""
    t = _TYPES[type_key]
    model = model_id if model_id in _MODELS else default_model(type_key).id
    return {"name": t.name, "type": t.category, "usage": usage, "icon": "", "type_key": t.key, "model_id": model}
