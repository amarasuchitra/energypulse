"""
Appliance profiles
==================
One place that says what each appliance IS and what advice the system is
ALLOWED to give about it.  Every recommendation in scheduler.py reads the
category here first, so the app can never tell a user to switch off a fridge.

Categories
----------
always_on  Must stay powered (fridge, router).  Never "switch off", never
           "shift".  Only health checks (e.g. compressor running too much).
shiftable  The job can be done at another time (washing machine, geyser,
           water pump).  Timing advice, inside the hours the user allows.
comfort    Used because someone needs it right now (AC).  No timing advice.
           Only duration and setting advice.
on_demand  Short, deliberate use (microwave).  No advice at all, unless it
           is left running abnormally long.

Usage:
    from appliance_profiles import get_profiles, allowed_advice
    profiles = get_profiles()
    allowed_advice(profiles["fridge"])     # -> {"health"}
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Optional, Tuple

ALWAYS_ON = "always_on"
SHIFTABLE = "shiftable"
COMFORT = "comfort"
ON_DEMAND = "on_demand"

CATEGORY_LABELS = {
    ALWAYS_ON: "Always on",
    SHIFTABLE: "Shiftable",
    COMFORT: "Comfort",
    ON_DEMAND: "On demand",
}

# What each category may be advised to do.
#   shift    - run at a cheaper time
#   duration - run for less time / change a setting
#   long_run - warn when one run is abnormally long
#   health   - warn when the appliance itself looks faulty
_ALLOWED = {
    ALWAYS_ON: {"health"},
    SHIFTABLE: {"shift", "long_run"},
    COMFORT: {"duration", "long_run"},
    ON_DEMAND: {"long_run"},
}


@dataclass(frozen=True)
class ApplianceProfile:
    key: str
    name: str
    icon: str
    category: str
    rated_kw: float                     # typical running power
    on_threshold_kw: float              # above this the appliance counts as ON
    typical_run_min: Tuple[int, int]    # normal length of one run (min, max)
    max_run_min: int                    # a run longer than this is flagged
    merge_gap_min: int = 0              # thermostat gaps shorter than this stay one run
    # Scheduling limits (shiftable only).  Hours are 0-24, end exclusive.
    allowed_hours: Tuple[int, int] = (0, 24)
    noisy: bool = False                 # respects the household's quiet hours
    # If set, the run may only move EARLIER, and by at most this many minutes
    # (a geyser can heat ahead of a bath because the tank holds heat, but
    # heating after the bath is useless).
    max_advance_min: Optional[int] = None
    # Always-on only: normal share of time the compressor runs.
    normal_duty: Optional[float] = None
    note: str = ""
    product: str = ""                   # what kind of unit this is, shown on hover


DEFAULT_PROFILES: Dict[str, ApplianceProfile] = {
    "fridge": ApplianceProfile(
        key="fridge", name="Refrigerator", icon="", category=ALWAYS_ON,
        rated_kw=0.14, on_threshold_kw=0.05, typical_run_min=(10, 25),
        max_run_min=24 * 60, normal_duty=0.40,
        product="Double-door refrigerator, 250 L class",
        note="Keeps food safe. Must never be switched off or scheduled."),
    "ac": ApplianceProfile(
        key="ac", name="Air Conditioner", icon="", category=COMFORT,
        rated_kw=1.5, on_threshold_kw=0.6, typical_run_min=(60, 480),
        max_run_min=600, merge_gap_min=20,
        product="1.5-ton split air conditioner",
        note="Used for comfort. Advice is about temperature and hours, not timing."),
    "geyser": ApplianceProfile(
        key="geyser", name="Water Heater (Geyser)", icon="", category=SHIFTABLE,
        rated_kw=2.0, on_threshold_kw=1.0, typical_run_min=(10, 40),
        max_run_min=60, allowed_hours=(4, 23), max_advance_min=180,
        product="Storage water heater, 15 L",
        note="Water stays hot for a few hours in an insulated tank, so heating can move earlier."),
    "washing_machine": ApplianceProfile(
        key="washing_machine", name="Washing Machine", icon="", category=SHIFTABLE,
        rated_kw=0.5, on_threshold_kw=0.08, typical_run_min=(40, 80),
        max_run_min=150, merge_gap_min=6, allowed_hours=(7, 21), noisy=True,
        product="Front-load washing machine, 7 kg",
        note="Noisy, so it is only scheduled in waking hours."),
    "water_pump": ApplianceProfile(
        key="water_pump", name="Water Pump", icon="", category=SHIFTABLE,
        rated_kw=0.75, on_threshold_kw=0.4, typical_run_min=(10, 30),
        max_run_min=45, allowed_hours=(6, 22), noisy=True,
        product="Water pump, 1 HP",
        note="Fills the overhead tank. A long run usually means a dry run or a leak."),
    "microwave": ApplianceProfile(
        key="microwave", name="Microwave", icon="", category=ON_DEMAND,
        rated_kw=1.1, on_threshold_kw=0.5, typical_run_min=(1, 8),
        max_run_min=20,
        product="Solo microwave oven, 20 L",
        note="Used when food is needed. No advice is given."),
}

# Order used everywhere (model outputs, charts, tables).
APPLIANCE_KEYS = list(DEFAULT_PROFILES.keys())


@dataclass
class HouseholdPrefs:
    """Limits the user sets. The scheduler never suggests anything outside them."""
    quiet_hours: Tuple[int, int] = (22, 6)     # no noisy appliance in this window
    min_monthly_saving_rs: float = 30.0        # smaller tips are not shown
    max_suggestions: int = 3                   # shown per day, biggest saving first
    dismissed: set = field(default_factory=set)  # suggestion ids the user rejected
    # Per-appliance overrides, e.g. {"geyser": {"max_advance_min": 120}}
    overrides: Dict[str, dict] = field(default_factory=dict)


def get_profiles(prefs: Optional[HouseholdPrefs] = None) -> Dict[str, ApplianceProfile]:
    """Default profiles with the user's overrides applied."""
    profiles = dict(DEFAULT_PROFILES)
    if prefs:
        for key, changes in prefs.overrides.items():
            if key in profiles:
                valid = {k: v for k, v in changes.items()
                         if k in ApplianceProfile.__dataclass_fields__ and k != "category"}
                profiles[key] = replace(profiles[key], **valid)
    return profiles


def allowed_advice(profile: ApplianceProfile) -> set:
    return set(_ALLOWED[profile.category])


def can_switch_off(profile: ApplianceProfile) -> bool:
    """False for anything that has to stay powered."""
    return profile.category != ALWAYS_ON


def profile_as_dict(profile: ApplianceProfile) -> dict:
    d = asdict(profile)
    d["category_label"] = CATEGORY_LABELS[profile.category]
    d["allowed_advice"] = sorted(allowed_advice(profile))
    return d


# Small switched loads.  They draw too little to be picked out of the meter
# total, so the app shows them from their switch, not from detection.
FANS = [
    {"key": "fan_living", "name": "Living room fan", "product": "Ceiling fan, 1200 mm sweep", "kw": 0.07},
    {"key": "fan_bedroom", "name": "Bedroom fan", "product": "Ceiling fan, 1200 mm sweep", "kw": 0.07},
]
