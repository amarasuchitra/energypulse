"""
Live alerts: checked every minute on the meter reading, so a pop-up appears while it is happening.

The hourly check in monitoring.py compares finished hours with the model's forecast; it needs two
whole hours above forecast before it says anything.  This module watches the minute readings:

  high use      the load has been above BOTH  ratio x the usual load for this minute  AND  usual + extra_kw
                for `min_minutes` minutes in a row.  "Usual" is the average of the same minute on the
                earlier days of the history.  The event ends after `recover_minutes` minutes back below.
  left on       an appliance has been running longer than its sensible maximum (appliance_profiles
                max_run_min), e.g. a water heater left on.

Each event has a stable id (date, kind, start minute) so it is announced once, however often the page
refreshes.  Pure functions; the app shows the pop-up and stores the alert.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from appliance_profiles import DEFAULT_PROFILES

LIVE_TYPE = "live_alert"
SENT_KEY = "live_alerts_sent"
SETTINGS_KEY = "live_alert_settings"


@dataclass
class LiveSettings:
    enabled: bool = True
    ratio: float = 1.5            # load above 1.5 x usual ...
    extra_kw: float = 1.0         # ... and at least 1 kW above usual
    min_minutes: int = 10         # for 10 minutes in a row
    recover_minutes: int = 5

    @classmethod
    def clean(cls, data) -> "LiveSettings":
        s = cls()
        if not isinstance(data, dict):
            return s

        def num(key, lo, hi, cast=float):
            try:
                v = cast(data.get(key, getattr(s, key)))
                if v != v:
                    raise ValueError
                return min(hi, max(lo, v))
            except (TypeError, ValueError):
                return getattr(s, key)
        return cls(enabled=bool(data.get("enabled", True)), ratio=num("ratio", 1.1, 5.0),
                   extra_kw=num("extra_kw", 0.2, 5.0), min_minutes=num("min_minutes", 2, 60, int))


def _hhmm(minute: int) -> str:
    minute = int(minute) % 1440
    return f"{minute // 60:02d}:{minute % 60:02d}"


def high_use_events(trace, typical, minute: int, s: LiveSettings = LiveSettings()) -> List[dict]:
    """Runs of high use up to `minute` (inclusive).  An event is reported from its alert minute on."""
    n = min(len(trace), int(minute) + 1)
    if n <= 0 or not len(typical):
        return []
    kw = np.asarray(trace[:n], dtype=float)
    usual = np.asarray(list(typical) + [typical[-1]] * max(0, n - len(typical)), dtype=float)[:n]
    usual = np.nan_to_num(usual, nan=0.0)
    over = np.isfinite(kw) & (kw > np.maximum(usual * s.ratio, usual + s.extra_kw))
    events, start, below = [], None, 0
    for i in range(n):
        if over[i]:
            if start is None:
                start = i
            below = 0
        elif start is not None:
            below += 1
            if below >= s.recover_minutes:
                end = i - below + 1
                if end - start >= s.min_minutes:
                    events.append((start, end))
                start, below = None, 0
    if start is not None:
        end = n - below if below else n
        if end - start >= s.min_minutes:
            events.append((start, None if below < s.recover_minutes else end))
    out = []
    for a, b in events:
        stop = b if b is not None else n
        seg_kw, seg_usual = kw[a:stop], usual[a:stop]
        out.append({"kind": "high_use", "start": a, "end": b, "ongoing": b is None,
                    "alert_at": a + s.min_minutes - 1,
                    "avg_kw": float(seg_kw.mean()), "peak_kw": float(seg_kw.max()),
                    "usual_kw": float(seg_usual.mean()),
                    "extra_kwh": float(np.clip(seg_kw - seg_usual, 0, None).sum() / 60.0)})
    return out


def culprit(today: pd.DataFrame, start: int, stop: int, owned) -> Optional[str]:
    """The appliance drawing the most power during minutes [start, stop)."""
    best, best_kw = None, 0.3
    for key in owned or ():
        col = f"{key}_kw"
        if col in today.columns:
            v = float(today[col].iloc[start:stop].astype(float).mean())
            if v > best_kw:
                best, best_kw = key, v
    return best


def left_on_events(today: pd.DataFrame, minute: int, owned) -> List[dict]:
    """Appliances that have run longer than their sensible maximum, up to `minute`."""
    out = []
    n = min(len(today), int(minute) + 1)
    for key in owned or ():
        col, prof = f"{key}_on", DEFAULT_PROFILES.get(key)
        if prof is None or col not in today.columns or prof.max_run_min >= 24 * 60:
            continue                                         # always-on appliances (fridge) are not "left on"
        on = today[col].iloc[:n].astype(bool).to_numpy()
        gap, i = max(int(getattr(prof, "merge_gap_min", 10) or 10), 5), 0
        while i < n:
            if not on[i]:
                i += 1
                continue
            start, last, j = i, i, i
            while j < n and j - last <= gap:                  # short pauses stay in the same run
                if on[j]:
                    last = j
                j += 1
            length = last - start + 1
            if length > prof.max_run_min:
                ongoing = n - 1 - last <= gap
                out.append({"kind": "left_on", "appliance": key, "name": prof.name, "start": start,
                            "end": None if ongoing else last + 1, "ongoing": ongoing,
                            "alert_at": start + prof.max_run_min, "minutes": length,
                            "max_min": prof.max_run_min})
            i = j
    return out


ADVICE = {
    "ac": "Raise the AC temperature a degree or two, or switch it off in empty rooms.",
    "geyser": "Check the water heater has not been left on; switch it off once the water is hot.",
    "water_pump": "Check the water pump is not running dry or the overhead tank overflowing.",
    "washing_machine": "Check the washing machine programme has not stalled.",
    "microwave": "Check the microwave is not still running.",
    "iron": "Check the iron has not been left on.",
}


def describe(ev: dict, date: str) -> dict:
    """Subject, body and a stable id for one event."""
    when = _hhmm(ev["start"])
    if ev["kind"] == "left_on":
        name = ev["name"]
        subject = f"{name} left on?"
        body = (f"The {name.lower()} has been running since {when}, "
                f"{ev['minutes']} minutes so far" if ev["ongoing"] else
                f"The {name.lower()} ran from {when} to {_hhmm(ev['end'])}, {ev['minutes']} minutes")
        body += f"; it normally runs at most {ev['max_min']} minutes. " + ADVICE.get(ev["appliance"], "Check it is switched off.")
    else:
        subject = "High energy use right now" if ev["ongoing"] else "High energy use earlier today"
        body = (f"Since {when} the home has drawn {ev['avg_kw']:.1f} kW on average (peak {ev['peak_kw']:.1f} kW), "
                f"against a usual {ev['usual_kw']:.1f} kW at this time" if ev["ongoing"] else
                f"From {when} to {_hhmm(ev['end'])} the home drew {ev['avg_kw']:.1f} kW on average "
                f"(peak {ev['peak_kw']:.1f} kW), against a usual {ev['usual_kw']:.1f} kW")
        body += f": about {ev['extra_kwh']:.1f} extra units."
        if ev.get("culprit"):
            name = DEFAULT_PROFILES[ev["culprit"]].name if ev["culprit"] in DEFAULT_PROFILES else ev["culprit"]
            body += f" Most of it is the {name.lower()}. " + ADVICE.get(ev["culprit"], "")
        else:
            body += " Switch off what is not in use."
    body = body.strip() + " (Simulated meter.)"
    key = ev.get("appliance") or "home"
    return {"id": f"{date}|{ev['kind']}|{key}|{ev['start']}", "subject": subject, "body": body,
            "kind": ev["kind"], "ongoing": ev["ongoing"], "start": ev["start"], "alert_at": ev["alert_at"]}


def check(today: pd.DataFrame, typical, minute: int, owned, date: str,
          s: LiveSettings = LiveSettings()) -> List[dict]:
    """Every event of today announced by `minute`, newest first, described and with an id."""
    if not s.enabled or today is None or today.empty:
        return []
    events = high_use_events(today["mains_kw"].tolist(), typical, minute, s)
    for ev in events:
        ev["culprit"] = culprit(today, ev["start"], ev["end"] or minute + 1, owned)
    events += left_on_events(today, minute, owned)
    shown = [describe(ev, date) for ev in events if ev["alert_at"] <= minute]
    return sorted(shown, key=lambda e: e["alert_at"], reverse=True)


def to_announce(events: List[dict], seen, minute: int, fresh_minutes: int = 30) -> List[dict]:
    """
    The events to pop up now: not announced before, and still going on or announced in the last
    `fresh_minutes` (opening the app in the evening does not replay the morning's alerts as pop-ups;
    they are still listed).
    """
    seen = set(seen or ())
    return [e for e in events if e["id"] not in seen and (e["ongoing"] or minute - e["alert_at"] <= fresh_minutes)]
