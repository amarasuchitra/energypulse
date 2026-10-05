"""
Real main-meter feed
====================
Lets a physical meter (for example an ESP32 with a PZEM-004T sensor and a
clip-on CT clamp on the main line) send readings to the app.

    meter  --HTTP POST-->  receiver (this file)  -->  data/live_meter.csv
                                                         |
                                     Home tab reads it and runs detection

Start the receiver on the computer that runs the app:
    python live_meter.py serve                 # listens on port 8600

The meter posts JSON to http://<computer-ip>:8600/reading
    {"power_w": 1840.5, "device": "esp32-pzem"}
    ("kw" is accepted in place of "power_w"; "timestamp" is optional)

No hardware yet?  Exercise the same path with simulated values:
    python live_meter.py demo-feed             # needs the receiver running
The Home tab then labels the source "demo-feed (simulated values)".

    python live_meter.py clear                 # forget stored readings

docs/esp32_pzem_meter.ino is a matching sketch for the ESP32.
"""

import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

import pandas as pd

FEED_CSV = os.path.join("data", "live_meter.csv")
MIN_READINGS = 30          # fewer than this today -> the app keeps the simulated feed
STALE_MINUTES = 15         # feed counts as live only if the last reading is this recent


def append_reading(kw: float, timestamp: Optional[str] = None, device: str = "meter",
                   path: str = FEED_CSV) -> None:
    kw = float(kw)
    if not (0 <= kw < 100):
        raise ValueError("power out of range for a home (0-100 kW)")
    ts = pd.Timestamp(timestamp) if timestamp else pd.Timestamp.now()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["timestamp", "kw", "device"])
        writer.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), round(kw, 4), device])


def load_today(now: Optional[pd.Timestamp] = None, path: str = FEED_CSV
               ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Today's readings as 1-minute rows (datetime, mains_kw) from midnight to
    the latest reading, plus the device name.  (None, None) when there is no
    usable feed.  Gaps of up to 5 minutes are bridged; longer gaps are zero.
    """
    if not os.path.exists(path):
        return None, None
    now = now or pd.Timestamp.now()
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    raw = raw[raw["timestamp"].dt.date == now.date()]
    if len(raw) < MIN_READINGS:
        return None, None
    last = raw["timestamp"].max()
    if (now - last).total_seconds() > STALE_MINUTES * 60:
        return None, None
    series = raw.set_index("timestamp")["kw"].resample("1min").mean()
    full = pd.date_range(now.normalize(), last.floor("min"), freq="1min")
    series = series.reindex(full).interpolate(limit=5).fillna(0.0)
    df = pd.DataFrame({"datetime": full, "mains_kw": series.to_numpy()})
    return df, str(raw["device"].iloc[-1])


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/reading":
            return self._reply(404, {"error": "POST readings to /reading"})
        # If ENERGYPULSE_METER_TOKEN is set, only a meter that sends it is accepted.
        token = os.environ.get("ENERGYPULSE_METER_TOKEN")
        if token and self.headers.get("X-Meter-Token") != token:
            return self._reply(401, {"error": "missing or wrong X-Meter-Token"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 2048:
                return self._reply(413, {"error": "reading too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            kw = body["kw"] if "kw" in body else float(body["power_w"]) / 1000.0
            append_reading(kw, body.get("timestamp"), str(body.get("device", "meter"))[:60])
            self._reply(200, {"ok": True})
        except (KeyError, ValueError, TypeError) as exc:
            self._reply(400, {"error": f"need 'power_w' or 'kw': {exc}"})

    def do_GET(self):
        df, device = load_today()
        self._reply(200, {"receiver": "EnergyPulse", "readings_today": 0 if df is None else len(df),
                          "device": device})

    def _reply(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def serve(port: int = 8600):
    print(f"Meter receiver listening on port {port}. POST readings to /reading")
    ThreadingHTTPServer(("0.0.0.0", port), _Handler).serve_forever()


def demo_feed(url: str = "http://127.0.0.1:8600/reading", keep_running: bool = True):
    """Send simulated readings through the real HTTP path: backfill today, then one a minute."""
    from meter_sim import simulate_home
    now = pd.Timestamp.now()
    day = simulate_home(days=1, seed=now.dayofyear, start=str(now.date()))

    def send(row):
        body = json.dumps({"kw": float(row.mains_kw), "timestamp": str(row.datetime),
                           "device": "demo-feed (simulated values)"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()

    past = day[day["datetime"] <= now]
    for row in past.itertuples():
        send(row)
    print(f"Backfilled {len(past)} readings for today.")
    while keep_running:
        time.sleep(60)
        now = pd.Timestamp.now()
        row = day[day["datetime"] <= now].tail(1)
        if row.empty or now.date() != day["datetime"].iloc[0].date():
            break
        send(next(row.itertuples()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if cmd == "serve":
        serve(int(sys.argv[2]) if len(sys.argv) > 2 else 8600)
    elif cmd == "demo-feed":
        demo_feed()
    elif cmd == "clear":
        if os.path.exists(FEED_CSV):
            os.remove(FEED_CSV)
        print("Stored readings cleared.")
    else:
        print(__doc__)
