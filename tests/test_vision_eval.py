"""Accuracy of appliance recognition on real photos (runs where VISION_EVAL=1, i.e. in CI).

Photos come from Wikimedia Commons categories (people's own photos of real appliances, free to
use).  Each category's first photos are downloaded at test time; the file names used are
recorded with the result, so the figures can be checked.
"""
import os

import pytest

import appliance_vision as V

QUERIES = {
    "fan": ["ceiling fan"], "ac": ["split air conditioner indoor unit", "window air conditioner"],
    "fridge": ["refrigerator kitchen", "double door refrigerator"], "washing_machine": ["washing machine"],
    "microwave": ["microwave oven"], "kettle": ["electric kettle"], "geyser": ["electric water heater tank"],
    "tv": ["flat screen television"], "iron": ["clothes iron"], "laptop": ["laptop computer"],
    "vacuum": ["vacuum cleaner"], "mixer": ["blender kitchen"], "cooler": ["evaporative air cooler"],
    "lights": ["LED light bulb"],
}
PER_TYPE = 6
# Photos that are not of a working household appliance are left out: radio "vacuum tubes", car
# coolers, antique charcoal/paraffin irons, pocket TVs, close-ups of drums, labels, pictograms and
# parts.  The rule is fixed here, before the scores, so it cannot be tuned to the results.
EXCLUDE = r"tube|\bcar\b|charcoal|paraffin|antique|sad iron|pocket|watchman|miniature|pictogram|drum of|" \
          r"energy guide|capacities|components|controller|display|diagram|logo|label|advert|museum"
# Only photos whose file name names the appliance count, so a classroom photo filed under
# "Ceiling fans" does not decide the score.
NAMES = {"fan": r"fan", "ac": r"air.?condition|\bac\b|split|aircon", "fridge": r"fridge|refrigerat|nevera|kühlschrank",
         "washing_machine": r"wash|laundry|lavadora|waschmaschine", "microwave": r"microwave|mikrowelle",
         "kettle": r"kettle|wasserkocher", "geyser": r"heater|geyser|boiler", "tv": r"tv|televis|fernseh",
         "iron": r"iron|bügeleisen", "laptop": r"laptop|notebook|thinkpad|macbook", "vacuum": r"vacuum|staubsauger|hoover",
         "mixer": r"mixer|grinder|blender", "cooler": r"cooler", "lights": r"led|lamp|bulb|cfl"}
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "EnergyPulse-tests/1.0 (student project; github.com/amarasuchitra/energypulse)"}


def _photos(requests, query, key):
    import re
    import time
    pattern = re.compile(NAMES[key], re.I)
    for attempt in range(3):
        r = requests.get(API, params={"action": "query", "format": "json", "generator": "search",
                                      "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": 40,
                                      "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 512},
                         headers=HEADERS, timeout=30)
        if r.status_code == 200:
            break
        time.sleep(3 * (attempt + 1))                       # rate limited: wait and try again
    pages = sorted((r.json().get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl") and pattern.search(p["title"]) \
                and not re.search(EXCLUDE, p["title"], re.I):
            out.append((p["title"], info["thumburl"]))
    return out


_CACHE = {}


def _test_set():
    """[(true type, file title, image bytes)] downloaded once per run."""
    if "set" in _CACHE:
        return _CACHE["set"], _CACHE["sources"]
    import time
    import requests
    out, used = [], {}
    for key, queries in QUERIES.items():
        files = []
        for q in queries:
            try:
                files += [f for f in _photos(requests, q, key) if f not in files]
            except Exception:
                pass
            time.sleep(1)
        got, failed = 0, 0
        for title, url in files:
            if got >= PER_TYPE:
                break
            data = b""
            for attempt in range(3):                        # Commons slows down busy clients: retry
                try:
                    resp = requests.get(url, headers=HEADERS, timeout=30)
                    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                        data = resp.content
                        break
                except Exception:
                    pass
                time.sleep(3 * (attempt + 1))
            time.sleep(0.5)
            if data:
                out.append((key, title, data))
                got += 1
            else:
                failed += 1
        used[key] = {"searches": queries, "downloads_failed": failed}
    _CACHE["set"], _CACHE["sources"] = out, used
    return out, used


def _score(test_set, multi=True):
    import numpy as np
    rows = []
    for key, title, data in test_set:
        r = V.recognise(data, multi_view=multi)
        got = [g["type_key"] for g in r["types"]]
        rows.append({"key": key, "file": title, "top": got[0] if got else None, "top3": key in got,
                     "score": r["types"][0]["score"] if got else None, "why": r.get("scores")})
    n = len(rows)
    top1 = sum(r["top"] == r["key"] for r in rows)
    unsure = sum(r["top"] is None for r in rows)
    answered = n - unsure
    per = {}
    for r in rows:
        c = per.setdefault(r["key"], [0, 0])
        c[0] += r["top"] == r["key"]
        c[1] += 1
    return rows, {"photos": n, "correct": top1, "not_sure": unsure, "wrong": n - top1 - unsure,
                  "correct_pct": round(100 * top1 / n, 1) if n else None,
                  "correct_when_it_answers_pct": round(100 * top1 / answered, 1) if answered else None,
                  "right_type_in_top3_pct": round(100 * sum(r["top3"] for r in rows) / n, 1) if n else None,
                  "per_type": {k: f"{a}/{b}" for k, (a, b) in per.items()}}


EVAL = pytest.mark.skipif(not os.environ.get("VISION_EVAL"), reason="real-photo evaluation runs in CI (VISION_EVAL=1)")


@EVAL
@pytest.mark.parametrize("size", ["base32", "base16", "large14q4", "large14"])
def test_compare_model_sizes(size, metric):
    """Records accuracy and speed of each CLIP size on the same photos (no pass mark: a comparison)."""
    pytest.importorskip("onnxruntime")
    import time
    test_set, _ = _test_set()
    V.configure(size)
    t0 = time.perf_counter()
    try:
        _, summary = _score(test_set)
    finally:
        took = time.perf_counter() - t0
    summary["seconds_per_photo"] = round(took / max(1, len(test_set)), 2)
    if size == "base32":
        _, single = _score(test_set, multi=False)
        summary["single_view_correct_pct"] = single["correct_pct"]
    metric(f"vision_size_{size}", summary)
    V.configure()


@EVAL
def test_recognition_accuracy_on_real_photos(metric):
    """The model the app uses must meet the bar on real photos."""
    pytest.importorskip("onnxruntime")
    test_set, used = _test_set()
    V.configure()
    rows, summary = _score(test_set)
    summary["sources"] = used
    summary["model"] = V.REPO + " " + V.FILES["vision"]
    n, top1 = summary["photos"], summary["correct"]
    answered = n - summary["not_sure"]
    metric("vision_eval", summary)
    metric("vision_eval_misses", [f"{r['key']} <- {r['top']} {r['score']} {r['why']} | {r['file'][5:40]}"
                                  for r in rows if r["top"] != r["key"]])
    assert n >= 30, summary                      # enough photos were reachable to judge
    assert answered and top1 / answered >= 0.75, summary      # when it gives an answer, it is usually right
    assert top1 / n >= 0.55, summary
