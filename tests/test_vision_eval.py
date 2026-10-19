"""Accuracy of appliance recognition on real photos (runs where VISION_EVAL=1, i.e. in CI).

Photos come from Wikimedia Commons categories (people's own photos of real appliances, free to
use).  Each category's first photos are downloaded at test time; the file names used are
recorded with the result, so the figures can be checked.
"""
import os

import pytest

import appliance_vision as V

CATEGORIES = {
    "fan": ["Ceiling fans"],
    "ac": ["Split-system air conditioners", "Split air conditioners", "Wall-mounted air conditioners",
           "Window air conditioners"],
    "fridge": ["Refrigerators in India", "Double-door refrigerators", "Refrigerators"],
    "washing_machine": ["Front-loading washing machines", "Top-loading washing machines", "Washing machines"],
    "microwave": ["Microwave ovens"],
    "kettle": ["Electric kettles"],
    "geyser": ["Electric water heaters", "Tank water heaters", "Water heaters"],
    "tv": ["Flat panel televisions", "LCD televisions", "Television sets"],
    "iron": ["Clothes irons", "Steam irons"],
    "laptop": ["Laptops"],
    "vacuum": ["Vacuum cleaners"],
    "mixer": ["Mixer grinders", "Blenders (kitchen)"],
    "cooler": ["Evaporative coolers", "Air coolers"],
    "lights": ["LED lamps", "Compact fluorescent lamps"],
}
PER_TYPE = 5
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "EnergyPulse-tests/1.0 (student project; github.com/amarasuchitra/energypulse)"}


def _photos(requests, category):
    r = requests.get(API, params={"action": "query", "format": "json", "generator": "categorymembers",
                                  "gcmtitle": f"Category:{category}", "gcmtype": "file", "gcmlimit": 30,
                                  "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 512},
                     headers=HEADERS, timeout=30)
    pages = sorted((r.json().get("query") or {}).get("pages", {}).values(), key=lambda p: p["title"])
    out = []
    for p in pages:
        info = (p.get("imageinfo") or [{}])[0]
        if info.get("mime") in ("image/jpeg", "image/png") and info.get("thumburl"):
            out.append((p["title"], info["thumburl"]))
    return out


@pytest.mark.skipif(not os.environ.get("VISION_EVAL"), reason="real-photo evaluation runs in CI (VISION_EVAL=1)")
def test_recognition_accuracy_on_real_photos(metric):
    requests = pytest.importorskip("requests")
    pytest.importorskip("onnxruntime")
    results, used = {}, {}
    for key, cats in CATEGORIES.items():
        files = []
        for cat in cats:
            try:
                files = _photos(requests, cat)
            except Exception:
                files = []
            if len(files) >= 3:
                used[key] = cat
                break
        rows = []
        for title, url in files[:PER_TYPE]:
            try:
                data = requests.get(url, headers=HEADERS, timeout=30).content
            except Exception:
                continue
            r = V.recognise(data)
            got = [g["type_key"] for g in r["types"]]
            rows.append({"file": title, "top": got[0] if got else None, "top3": key in got,
                         "score": r["types"][0]["score"] if got else None, "class": (r.get("model_class") or {}).get("label")})
        results[key] = rows
    n = sum(len(v) for v in results.values())
    top1 = sum(r["top"] == k for k, v in results.items() for r in v)
    top3 = sum(r["top3"] for v in results.values() for r in v)
    summary = {"photos": n, "top1_correct": top1, "top3_correct": top3,
               "top1_pct": round(100 * top1 / n, 1) if n else None, "top3_pct": round(100 * top3 / n, 1) if n else None,
               "per_type": {k: f"{sum(r['top'] == k for r in v)}/{len(v)}" for k, v in results.items()},
               "categories": used}
    metric("vision_eval", summary)
    metric("vision_eval_detail", {k: [(r["file"][5:60], r["top"], r["score"], r["class"]) for r in v]
                                  for k, v in results.items()})
    assert n >= 30, summary                      # enough photos were reachable to judge
    assert top1 / n >= 0.70 and top3 / n >= 0.85, summary
