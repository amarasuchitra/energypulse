"""Seed a throwaway database with one demo household for report screenshots (CI only, localhost)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "backend")]

import catalog  # noqa: E402
import live_watch  # noqa: E402
from auth import Accounts  # noqa: E402
from db import DatabaseManager  # noqa: E402

EMAIL, PW = os.environ["SHOT_EMAIL"], os.environ["SHOT_PW"]
ok, msg = Accounts().sign_up(EMAIL, PW, "Asha Rao")
print("sign up:", ok, msg)
db = DatabaseManager()
hid = db.ensure_user_login(EMAIL, "Asha Rao")["household_id"]
items = [catalog.item_for(k) for k in ("ac", "fridge", "geyser", "washing_machine", "water_pump", "microwave",
                                        "fan", "lights", "tv")]
db.save_home_details(hid, {"home_type": "Apartment", "occupants": 4, "tariff_rate": 8.0, "city": "Bengaluru",
                           "timezone": "Asia/Kolkata", "home_size": 1100, "appliances": items})
db.mark_onboarding_completed(hid)
# Sensitive live-alert limits so the screenshots show real alerts raised from the simulated readings.
db.set_store(hid, live_watch.SETTINGS_KEY, {"enabled": True, "ratio": 1.2, "extra_kw": 0.3, "min_minutes": 3})
print("household", hid)
