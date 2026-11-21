"""Seed a throwaway database for the demo recording (CI only, localhost)."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "backend")]

import billing  # noqa: E402
import catalog  # noqa: E402
import live_watch  # noqa: E402
from auth import Accounts  # noqa: E402
from db import DatabaseManager  # noqa: E402

EMAIL, PW = os.environ["SHOT_EMAIL"], os.environ["SHOT_PW"]
print("sign up:", *Accounts().sign_up(EMAIL, PW, "Asha Rao"))
db = DatabaseManager()
hid = db.ensure_user_login(EMAIL, "Asha Rao")["household_id"]
# A few appliances only, so more can be added on camera.
items = [catalog.item_for(k) for k in ("ac", "fridge", "geyser", "lights", "tv")]
db.save_home_details(hid, {"home_type": "Apartment", "occupants": 4, "tariff_rate": 8.0, "city": "Bengaluru",
                           "timezone": "Asia/Kolkata", "home_size": 1100, "appliances": items})
db.mark_onboarding_completed(hid)
db.set_store(hid, live_watch.SETTINGS_KEY, {"enabled": True, "ratio": 1.2, "extra_kw": 0.3, "min_minutes": 3})
# Two earlier meter readings, so the reading typed on camera gives units used this month.
now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("min")
start = now.replace(day=1, hour=7, minute=30)
mid = now - pd.Timedelta(days=6, hours=2)
raw = billing.add_reading([], 4300.0, start)
raw = billing.add_reading(raw, round(4300.0 + 6.4 * (mid - start).total_seconds() / 86400, 1), mid)
nxt = round(raw[-1]["value"] + 6.9 * (now - mid).total_seconds() / 86400, 1)
db.set_store(hid, billing.READINGS_KEY, raw)
open("/tmp/next_reading.txt", "w").write(f"{nxt:.1f}")
print("household", hid, "readings", raw, "next", nxt)
