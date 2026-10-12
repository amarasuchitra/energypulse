"""
Rating plates and appliance photos
==================================
Most people do not know the model of an old fridge or pump, but every
appliance carries a rating plate (sticker or metal plate) with its power,
voltage and current.  The user can photograph it; Tesseract (free, runs on
the server, no paid service) reads the text and the figures are picked out
for the user to check before they are saved.

Photos are stored encrypted per household, like saved bills.

    text, ok = read_plate(image_bytes)
    figures  = parse_plate(text)      # {"rated_w": 1500, "voltage_v": 230, ...}
    pid      = store_photo(household_id, image_bytes, "jpg")
"""

import hashlib
import io
import os
import re
from typing import Optional, Tuple

PHOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "photos")
PHOTO_TYPES = ["jpg", "jpeg", "png", "webp"]


# ------------------------------------------------------------------ reading
def _prepare(data: bytes):
    from PIL import Image, ImageFilter, ImageOps
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1600:                                    # small text reads better enlarged
        scale = 1600 / max(img.width, 1)
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    return ImageOps.autocontrast(img).filter(ImageFilter.SHARPEN)


def read_plate(data: bytes) -> Tuple[str, Optional[bool]]:
    """(text, status): True read, False no OCR engine on this machine, None the image could not be read."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return "", False
    try:
        img = _prepare(data)
        # Two page layouts: a block of text, and scattered labels (typical of plates).
        text = pytesseract.image_to_string(img, config="--psm 6") + "\n" + \
            pytesseract.image_to_string(img, config="--psm 11")
        return text, True
    except Exception:
        return "", None


_NUM = r"(\d+(?:[.,]\d+)?)"


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def parse_plate(text: str) -> dict:
    """
    The figures found on a rating plate.  Lines about cooling or heating
    CAPACITY or motor OUTPUT are skipped for power, because the electricity
    drawn is the input power, not what the appliance delivers.
    """
    out, found = {}, []
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    preferred, other = [], []
    for line in lines:
        low = line.lower()
        for m in re.finditer(_NUM + r"\s*(kw|w)\b", low):
            value = _num(m.group(1)) * (1000 if m.group(2) == "kw" else 1)
            if not 1 <= value <= 10000:
                continue
            before = low[max(0, m.start() - 28):m.start()]       # the words just before this number
            if re.search(r"cool|capacity|output|heating cap|lumen", before):
                continue
            (preferred if re.search(r"input|consum|rated|power|watt", before) else other).append((value, line))
    pool = preferred or other
    if pool:
        out["rated_w"], line = max(pool)
        found.append(line)
    m = re.search(r"(\d{3})\s*(?:[-~/]\s*(\d{3}))?\s*v(?:olts?|ac)?\b", (text or "").lower())
    if m:
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if 100 <= lo <= 260 and 100 <= hi <= 260:
            out["voltage_v"] = 230.0 if lo <= 230 <= hi else float(hi)
            out["voltage_text"] = f"{lo}-{hi} V" if hi != lo else f"{lo} V"
    for line in lines:
        low = line.lower()
        m = re.search(_NUM + r"\s*a(?:mps?)?\b", low)
        if m and "ma" not in low[max(0, m.start(1)):m.end()] and "fuse" not in low:
            amps = _num(m.group(1))
            if 0.01 <= amps <= 40:
                out["current_a"] = amps
                found.append(line)
                break
    m = re.search(r"\b(50|60)\s*hz", (text or "").lower())
    if m:
        out["frequency_hz"] = int(m.group(1))
    m = re.search(r"\b([1-5])\s*[- ]?\s*star", (text or "").lower())
    if m:
        out["star"] = int(m.group(1))
    m = re.search(_NUM + r"\s*(litres?|liters?|ltrs?|l|kg|tons?|tr)\b", (text or "").lower())
    if m:
        unit = {"l": "L", "ltr": "L", "ltrs": "L", "litre": "L", "litres": "L", "liter": "L", "liters": "L",
                "kg": "kg", "ton": "ton", "tons": "ton", "tr": "ton"}[m.group(2)]
        out["capacity"] = f"{m.group(1)} {unit}"
    m = re.search(_NUM + r"\s*hp\b", (text or "").lower())
    hp = _num(m.group(1)) if m else None
    if m:
        out["capacity"] = f"{m.group(1)} HP"
    m = re.search(r"model\s*(?:no\.?|number|name)?\s*[:.\-]?\s*([A-Z0-9][A-Z0-9\-/.]{2,})", text or "", re.I)
    if m:
        out["model_no"] = m.group(1).strip(".-/")[:30]
    if "rated_w" not in out and "current_a" in out:          # only the current printed: P = V x I x pf
        out["rated_w"] = round(out.get("voltage_v", 230.0) * out["current_a"] * 0.85, 0)
        out["from_current"] = True
    elif "rated_w" not in out and hp:                        # only the motor size: output / typical efficiency
        out["rated_w"] = round(hp * 746 / 0.8, 0)
        out["from_hp"] = True
    out["lines"] = found
    return out


# ------------------------------------------------------------------- photos
def _photo_path(household_id: str, photo_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "", household_id)[:64]
    return os.path.join(PHOTO_DIR, safe, f"{re.sub(r'[^a-f0-9]', '', photo_id)[:64]}.bin")


def store_photo(household_id: str, data: bytes) -> str:
    """Encrypt and keep a photo for this household; returns its id."""
    from auth import protect
    photo_id = hashlib.sha256(data).hexdigest()[:32]
    path = _photo_path(household_id, photo_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(protect(data))
    return photo_id


def load_photo(household_id: str, photo_id: str) -> Optional[bytes]:
    from auth import unprotect
    try:
        with open(_photo_path(household_id, photo_id), "rb") as fh:
            return unprotect(fh.read())
    except OSError:
        return None
