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
        if info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl") and pattern.search(p["title"]):
            out.append((p["title"], info["thumburl"]))
    return out


@pytest.mark.skipif(not os.environ.get("VISION_EVAL"), reason="real-photo evaluation runs in CI (VISION_EVAL=1)")
def test_recognition_accuracy_on_real_photos(metric):
    requests = pytest.importorskip("requests")
    pytest.importorskip("onnxruntime")
    results, used = {}, {}
    import time
    for key, queries in QUERIES.items():
        files = []
        for q in queries:
            try:
                files += [f for f in _photos(requests, q, key) if f not in files]
            except Exception:
                pass
            time.sleep(1)
        used[key] = queries
        rows = []
        skipped = 0
        for title, url in files:
            if len(rows) >= PER_TYPE:
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
            if not data:
                skipped += 1
                continue
            r = V.recognise(data)
            got = [g["type_key"] for g in r["types"]]
            rows.append({"file": title, "top": got[0] if got else None, "top3": key in got, "why": r.get("scores"),
                         "score": r["types"][0]["score"] if got else None, "class": (r.get("model_class") or {}).get("label")})
        results[key] = rows
        used[key] = {"searches": queries, "downloads_failed": skipped}
    n = sum(len(v) for v in results.values())
    top1 = sum(r["top"] == k for k, v in results.items() for r in v)
    top3 = sum(r["top3"] for v in results.values() for r in v)
    unsure = sum(r["top"] is None for v in results.values() for r in v)
    wrong = n - top1 - unsure
    answered = n - unsure
    summary = {"photos": n, "correct": top1, "not_sure": unsure, "wrong": wrong,
               "correct_pct": round(100 * top1 / n, 1) if n else None,
               "correct_when_it_answers_pct": round(100 * top1 / answered, 1) if answered else None,
               "right_type_in_top3_pct": round(100 * top3 / n, 1) if n else None,
               "per_type": {k: f"{sum(r['top'] == k for r in v)}/{len(v)}" for k, v in results.items()},
               "sources": used}
    metric("vision_eval", summary)
    metric("vision_tokens", V._SESS.get("tokens_seen"))
    metric("vision_eval_misses", [f"{k} <- {r['top']} {r['score']} {r['why']} | {r['file'][5:40]}"
                                  for k, v in results.items() for r in v if r["top"] != k])
    assert n >= 30, summary                      # enough photos were reachable to judge
    assert answered and top1 / answered >= 0.75, summary      # when it gives an answer, it is usually right
    assert top1 / n >= 0.55, summary
