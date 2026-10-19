"""
Finding the exact product: the model number, read from what the household has, confirmed on the web.

A photo of the outside of an appliance cannot tell the exact model - many models share one
body and differ only in tonnage or star rating, which is printed on the label.  So the exact
product is READ, never guessed:

  1. Model number, from any of
       - a QR code or barcode (sticker, warranty card, box), decoded with OpenCV
       - a photo of the rating label or the box (text recognition)
       - the purchase invoice (a PDF's own text, or a photo of it)
  2. The model is searched on the web (spec_search.look_up), preferring the maker's own site.
  3. It is called EXACT only when the model number itself appears on at least two of the pages
     read and those pages agree on the power or the star rating/capacity.  Otherwise the result
     is "not confirmed", with the reason, and nothing is presented as exact.

Where the label is (WHERE_LABEL) is shown when no model number could be found.
"""
import io
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import appliance_vision as V

WHERE_LABEL = {
    "ac": "on the right side of the indoor unit (lift the front flap), or on the outdoor unit",
    "fan": "on top of the motor, near the down-rod (switch the fan off first)",
    "cooler": "on the back panel, near the power cord",
    "fridge": "on the inside wall near the vegetable tray, or at the back",
    "microwave": "on the back, or inside the door frame",
    "induction": "underneath the cooktop",
    "kettle": "under the base or under the kettle",
    "mixer": "under the motor base",
    "otg": "on the back panel",
    "chimney": "inside, behind the filter",
    "purifier": "on the side or the back of the unit",
    "dishwasher": "on the edge of the door",
    "washing_machine": "on the back panel, or inside the door/lid frame",
    "iron": "on the heel of the iron, near the cord",
    "vacuum": "under the body",
    "geyser": "on the side sticker of the tank",
    "water_pump": "on the motor body plate",
    "heater": "on the back or underneath",
    "tv": "on the back panel",
    "settop": "underneath the box",
    "speaker": "on the back or underneath",
    "router": "underneath the router",
    "laptop": "underneath the laptop",
    "desktop": "on the back or side of the cabinet",
    "charger": "printed on the charger body",
    "lights": "printed on the bulb base or the box",
}
ALSO = "It is also printed on the box and on the purchase invoice."

_LABELLED = re.compile(r"(?:model|mod\.?|model\s*no\.?|model\s*number|indoor\s*unit|i/?d\s*unit|product\s*code|"
                       r"cat(?:alogue)?\.?\s*no\.?|type)\s*[:#.\-]?\s*((?=[A-Z0-9\-/.]*\d)[A-Z0-9][A-Z0-9\-/.]{2,24})", re.I)


# ----------------------------------------------------------------- codes
def read_codes(data: bytes) -> List[str]:
    """Text inside QR codes and barcodes in a photo (empty when none or OpenCV is missing)."""
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return []
    found = []
    try:
        ok, texts, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(arr)
        if ok:
            found += [t for t in texts if t]
    except Exception:
        pass
    try:
        bd = cv2.barcode.BarcodeDetector()
        res = bd.detectAndDecode(arr)
        texts = res[0] if len(res) == 3 else res[1]       # (texts, types, points) or (ok, texts, types, points)
        found += [t for t in (texts if isinstance(texts, (list, tuple)) else [texts]) if t]
    except Exception:
        pass
    return list(dict.fromkeys(found))


# ----------------------------------------------------------------- text
def invoice_text(data: bytes, filename: str = "") -> str:
    """Text of an invoice: a PDF's own text, else text recognition on the image."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        try:
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages[:5])
        except Exception:
            return ""
    from nameplate import read_plate
    text, _ = read_plate(data)
    return text or ""


def photo_text(data: bytes) -> str:
    from nameplate import read_plate
    text, _ = read_plate(data)
    return (text or "") + "\n" + V.logo_text(data)


def model_candidates(texts: List[str], codes: List[str] = ()) -> List[Tuple[str, str]]:
    """Possible model numbers, best first, each with where it came from."""
    out: List[Tuple[str, str]] = []

    def add(m, why):
        m = m.strip(" .-/").upper()
        if len(m) >= 4 and re.search(r"\d", m) and re.search(r"[A-Z]", m) and m not in [x for x, _ in out] \
                and not re.fullmatch(r"\d+(V|W|HZ|A|KW|KG|L|MM|CM)", m):
            out.append((m, why))
    brands = {re.sub(r"[^a-z0-9]", "", b.lower()) for b in V.ALL_BRANDS}
    for c in codes:                                         # a QR code may hold a link or "MODEL:xyz"
        for m in _LABELLED.findall(c):
            add(m, "QR / barcode")
        if re.match(r"https?://", c, re.I):                 # a product link: the model is a part of the path
            from urllib.parse import urlparse
            for seg in reversed([x for x in urlparse(c).path.split("/") if x]):
                parts = seg.split("-")
                if len(parts) > 1 and parts[0].lower() in brands:
                    seg = "-".join(parts[1:])                 # "lg-rs-q19ynze" -> "rs-q19ynze"
                add(re.sub(r"\.(html?|php|aspx?)$", "", seg, flags=re.I), "QR / barcode (product link)")
            continue
        for m in V.model_numbers(c):
            add(m, "QR / barcode")
    for t in texts:
        for m in _LABELLED.findall(t or ""):
            add(m, "label text 'Model'")
    for t in texts:
        for m in V.model_numbers(t or ""):
            add(m, "text")
    return out[:6]


# ----------------------------------------------------------------- confirming on the web
def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def confirm(brand: str, model_no: str, type_name: str = "", look_up: Optional[Callable] = None) -> dict:
    """
    Search the model on the web.  EXACT when the model number appears on >= 2 of the pages read and
    they agree on power (W) or on star rating / capacity.  Returns status, reason, figures, sources.
    """
    import spec_search
    look_up = look_up or spec_search.look_up
    query = " ".join(x for x in (brand, model_no, type_name) if x)
    found = look_up(query, model=model_no)
    matching = [s for s in found.get("sources", []) if s.get("mentions_model")]
    votes: Dict[str, Counter] = {}
    for s in matching:
        for k, v in s.get("found", {}).items():
            if k in ("rated_w", "star", "capacity", "voltage_v", "current_a"):
                votes.setdefault(k, Counter())[round(v) if k == "rated_w" else v] += 1
    figures = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    agree = {k: c.most_common(1)[0][1] for k, c in votes.items()}
    exact = len(matching) >= 2 and any(agree.get(k, 0) >= 2 for k in ("rated_w", "star", "capacity"))
    if not found.get("ok"):
        reason = "The web search could not be reached just now. Try again in a minute."
    elif not matching:
        reason = (f"No page found that names the model {model_no}. Check the number against the label "
                  "(0 and O, 1 and I, 8 and B are easily mixed up).")
    elif len(matching) < 2:
        reason = f"Only one page names {model_no}, so it cannot be confirmed yet."
    elif not exact:
        reason = f"Pages naming {model_no} do not agree on its power or rating yet."
    else:
        reason = f"{len(matching)} pages name {model_no} and agree on its figures."
    return {"status": "exact" if exact else "not confirmed", "reason": reason, "query": query,
            "model_no": model_no, "brand": brand, "figures": figures if exact else {}, "agree": agree,
            "sources": [{"title": s.get("title", ""), "url": s.get("url", "")} for s in matching][:5]}


def find_exact(photos: List[bytes], invoice: Optional[Tuple[bytes, str]] = None, type_key: Optional[str] = None,
               brand: str = "", confirm_fn: Callable = confirm, max_tries: int = 3) -> dict:
    """
    The whole process for the photos (label, box, QR, front) and invoice a household gives.
    Returns {"status": "exact" | "not confirmed" | "no model number", "model_no", "brand", "figures",
             "sources", "reason", "tried", "where"}.
    """
    texts, codes = [], []
    for p in photos:
        codes += read_codes(p)
        texts.append(photo_text(p))
    if invoice:
        texts.append(invoice_text(*invoice))
    joined = "\n".join(texts + codes)
    if not brand:
        names = V.brands_in_text(joined, type_key)
        brand = names[0] if names else ""
    import catalog
    t = catalog.get_type(type_key or "")
    type_name = t.name if t else ""
    cands = model_candidates(texts, codes)
    where = (f"The model number is usually {WHERE_LABEL[type_key]}. {ALSO}" if type_key in WHERE_LABEL else ALSO)
    if not cands:
        return {"status": "no model number", "model_no": None, "brand": brand, "figures": {}, "sources": [],
                "reason": "No model number could be read from what was given.", "tried": [], "where": where}
    tried, best = [], None
    for m, why in cands[:max_tries]:
        r = confirm_fn(brand, m, type_name)
        r["from"] = why
        tried.append({"model_no": m, "from": why, "status": r["status"]})
        if r["status"] == "exact":
            best = r
            break
        best = best or r
    best.update({"tried": tried, "where": where})
    return best
