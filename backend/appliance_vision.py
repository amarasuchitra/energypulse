"""
Recognise the appliance in a photo: what it is, which company made it and, when it shows,
its model number.  Runs inside the app; no outside service and no key.

How
  1. Type  CLIP (OpenAI's image-text model, ViT-B/32, the 8-bit version published as
           Xenova/clip-vit-base-patch32, about 154 MB, downloaded once on first use and run
           with onnxruntime) compares the photo with written descriptions of every appliance
           in the catalogue - "a split air conditioner indoor unit mounted high on a wall",
           "an electric water heater geyser tank on a bathroom wall" - several per type,
           plus descriptions of things that are not appliances, so a photo of a person or a
           page is not forced into a type.
  2. Brand Text printed anywhere in the photo (logo, label) is read with Tesseract and
           matched against the companies that make that appliance in India.  A brand read
           from the photo's own text is the strongest evidence.  If no name is printed,
           CLIP compares "an LG air conditioner", "a Voltas air conditioner", ... and a brand
           is offered only when it clearly wins.
  3. Model A model number printed in the photo (letters and digits, e.g. RSZ-123A) is
           picked out for the web lookup of exact figures.  The exact version is often only
           on the label, so when none shows the user picks the size or star class.

Nothing is filled in without the user confirming it.
"""
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = "Xenova/clip-vit-base-patch32"
FILES = {"vision": "onnx/vision_model_quantized.onnx", "text": "onnx/text_model_quantized.onnx",
         "tokenizer": "tokenizer.json"}
CACHE_DIR = os.environ.get("ENERGYPULSE_MODEL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache",
                                                                       "energypulse", "clip")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype="float32")
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype="float32")
SCALE = 100.0                 # CLIP's learned logit scale
MIN_TYPE = 0.20               # below this a type is not offered
MIN_BRAND, BRAND_MARGIN = 0.35, 0.15

# Several descriptions per type; their embeddings are averaged (prompt ensembling).
DESCRIPTIONS: Dict[str, List[str]] = {
    "ac": ["a split air conditioner indoor unit mounted high on a wall",
           "a white wall-mounted AC unit", "a window air conditioner", "an air conditioner outdoor unit"],
    "fan": ["a ceiling fan with three blades", "a ceiling fan seen from below", "a ceiling fan hanging from the ceiling",
            "a pedestal fan", "a table fan", "a wall mounted fan"],
    "cooler": ["a desert air cooler", "a plastic room air cooler with honeycomb pads", "an evaporative air cooler"],
    "fridge": ["a refrigerator", "a double door fridge", "a single door refrigerator in a kitchen"],
    "microwave": ["a microwave oven", "a microwave oven on a kitchen counter"],
    "induction": ["an induction cooktop", "an electric induction stove with a glass top"],
    "kettle": ["an electric kettle", "a stainless steel electric kettle"],
    "mixer": ["an Indian mixer grinder with jars", "a mixer grinder", "a kitchen blender"],
    "otg": ["an oven toaster grill", "an OTG oven", "a small countertop electric oven"],
    "chimney": ["a kitchen chimney hood above a stove", "a kitchen exhaust hood"],
    "purifier": ["an RO water purifier mounted on a kitchen wall", "a water purifier with a tap"],
    "dishwasher": ["a dishwasher", "an open dishwasher with racks"],
    "washing_machine": ["a front load washing machine", "a top load washing machine", "a semi automatic washing machine"],
    "iron": ["an electric clothes iron", "a steam iron"],
    "vacuum": ["a vacuum cleaner", "a canister vacuum cleaner"],
    "geyser": ["an electric water heater geyser tank on a bathroom wall", "a geyser", "a storage water heater"],
    "water_pump": ["an electric water pump motor", "a monoblock water pump", "a submersible pump motor"],
    "heater": ["a room heater", "an electric fan heater", "an oil filled radiator heater"],
    "tv": ["a television", "a flat screen TV on a wall", "an LED TV"],
    "settop": ["a DTH set-top box", "a TV set top box with a remote"],
    "speaker": ["a sound bar", "a speaker", "a home theatre speaker system"],
    "router": ["a Wi-Fi router with antennas", "a wireless router", "a broadband modem"],
    "laptop": ["a laptop computer", "an open laptop on a desk"],
    "desktop": ["a desktop computer with a monitor", "a computer CPU cabinet", "a computer monitor"],
    "charger": ["a mobile phone charger", "a phone charging on a cable"],
    "lights": ["an LED tube light", "an LED bulb", "a ceiling light", "a lamp"],
}
NOT_APPLIANCE = ["a photo of a person", "a selfie", "a photo of a document or a page of text", "a screenshot",
                 "a photo of food", "a photo of a plant", "a photo of a car or a bike", "a photo of an animal",
                 "a photo of a building outside", "a photo of furniture"]

BRANDS: Dict[str, List[str]] = {
    "ac": ["LG", "Samsung", "Voltas", "Daikin", "Blue Star", "Lloyd", "Hitachi", "Panasonic", "Carrier", "Godrej",
           "Haier", "Whirlpool", "Mitsubishi", "O General", "Croma"],
    "fan": ["Havells", "Crompton", "Orient", "Bajaj", "Usha", "Atomberg", "Polar", "Luminous", "V-Guard", "Anchor",
            "Khaitan"],
    "cooler": ["Symphony", "Bajaj", "Crompton", "Kenstar", "Voltas", "Havells", "Orient"],
    "fridge": ["LG", "Samsung", "Whirlpool", "Godrej", "Haier", "Bosch", "Panasonic", "Voltas Beko", "Hitachi"],
    "microwave": ["LG", "Samsung", "IFB", "Panasonic", "Godrej", "Morphy Richards", "Bajaj"],
    "induction": ["Prestige", "Philips", "Pigeon", "Bajaj", "Havells", "Usha"],
    "kettle": ["Philips", "Prestige", "Pigeon", "Kent", "Bajaj", "Havells", "Morphy Richards"],
    "mixer": ["Preethi", "Prestige", "Philips", "Bajaj", "Butterfly", "Sujata", "Usha", "Havells", "Panasonic"],
    "otg": ["Morphy Richards", "Bajaj", "Philips", "Prestige", "Agaro", "Usha"],
    "chimney": ["Elica", "Faber", "Hindware", "Glen", "Kaff", "Sunflame"],
    "purifier": ["Kent", "Aquaguard", "Pureit", "Livpure", "AO Smith"],
    "dishwasher": ["Bosch", "IFB", "LG", "Voltas Beko", "Faber"],
    "washing_machine": ["LG", "Samsung", "IFB", "Whirlpool", "Bosch", "Godrej", "Haier", "Panasonic"],
    "iron": ["Philips", "Bajaj", "Usha", "Havells", "Morphy Richards"],
    "vacuum": ["Eureka Forbes", "Philips", "Dyson", "Karcher", "Agaro", "Inalsa"],
    "geyser": ["AO Smith", "Racold", "Bajaj", "Havells", "V-Guard", "Crompton", "Venus", "Haier"],
    "water_pump": ["Kirloskar", "Crompton", "CRI", "Havells", "Lubi", "V-Guard", "Texmo"],
    "heater": ["Bajaj", "Havells", "Orpat", "Usha", "Morphy Richards", "Orient"],
    "tv": ["Samsung", "LG", "Sony", "Xiaomi", "TCL", "OnePlus", "Panasonic", "Vu", "Hisense", "Acer"],
    "settop": ["Tata Play", "Airtel", "Dish TV", "Sun Direct"],
    "speaker": ["Sony", "JBL", "boAt", "Samsung", "LG", "Philips"],
    "router": ["TP-Link", "D-Link", "Netgear", "Tenda", "JioFiber", "Airtel"],
    "laptop": ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple", "MSI"],
    "desktop": ["HP", "Dell", "Lenovo", "Asus", "Acer", "Apple"],
    "charger": ["Samsung", "Apple", "Xiaomi", "OnePlus", "Realme", "Ambrane"],
    "lights": ["Philips", "Syska", "Wipro", "Havells", "Bajaj", "Crompton"],
}
ALL_BRANDS = sorted({b for v in BRANDS.values() for b in v}, key=len, reverse=True)

_SESS: dict = {}
_TEXT_CACHE: Dict[str, np.ndarray] = {}


# ----------------------------------------------------------------- the model
def _download(rel: str) -> str:
    """The model file in the local cache, downloaded from Hugging Face the first time."""
    path = os.path.join(CACHE_DIR, rel.replace("/", "_"))
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    import requests
    os.makedirs(CACHE_DIR, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/main/{rel}"
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    os.replace(tmp, path)
    return path


def _load():
    if not _SESS:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _SESS["vision"] = ort.InferenceSession(_download(FILES["vision"]), opts, providers=["CPUExecutionProvider"])
        _SESS["text"] = ort.InferenceSession(_download(FILES["text"]), opts, providers=["CPUExecutionProvider"])
        _SESS["tok"] = Tokenizer.from_file(_download(FILES["tokenizer"]))
    return _SESS


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)


def _pick_output(outputs, sess, want: str) -> np.ndarray:
    names = [o.name for o in sess.get_outputs()]
    return np.asarray(outputs[names.index(want)] if want in names else outputs[0])


def embed_texts(texts: List[str]) -> np.ndarray:
    """Unit-length CLIP embeddings for descriptions (cached per text)."""
    missing = [t for t in texts if t not in _TEXT_CACHE]
    if missing:
        s = _load()
        enc = s["tok"].encode_batch(missing)
        ids = [e.ids[:77] for e in enc]
        width = max(len(i) for i in ids)
        input_ids = np.full((len(ids), width), 49407, dtype="int64")      # CLIP pads with its end token
        mask = np.zeros((len(ids), width), dtype="int64")
        for row, i in enumerate(ids):
            input_ids[row, :len(i)], mask[row, :len(i)] = i, 1
        feed = {}
        for inp in s["text"].get_inputs():
            feed[inp.name] = mask if "mask" in inp.name else input_ids
        out = _unit(_pick_output(s["text"].run(None, feed), s["text"], "text_embeds"))
        for t, v in zip(missing, out):
            _TEXT_CACHE[t] = v
    return np.stack([_TEXT_CACHE[t] for t in texts])


def preprocess(data: bytes) -> np.ndarray:
    """CLIP's own preparation: shortest side to 224, centre crop, scale, normalise, channels first."""
    import io
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    w, h = img.size
    k = 224 / min(w, h)
    img = img.resize((max(224, round(w * k)), max(224, round(h * k))), Image.BICUBIC)
    w, h = img.size
    left, top_ = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top_, left + 224, top_ + 224))
    x = (np.asarray(img, dtype="float32") / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype("float32")


def embed_image(data: bytes) -> np.ndarray:
    s = _load()
    x = preprocess(data)
    name = s["vision"].get_inputs()[0].name
    return _unit(_pick_output(s["vision"].run(None, {name: x}), s["vision"], "image_embeds"))[0]


# ----------------------------------------------------------------- deciding
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def classify(image_vec: np.ndarray, groups: Dict[str, List[str]], embed=embed_texts) -> Dict[str, float]:
    """Probability per group: each group's descriptions averaged into one direction, then softmax."""
    keys = list(groups)
    centres = np.stack([_unit(embed(groups[k]).mean(axis=0)) for k in keys])
    p = _softmax(SCALE * centres @ image_vec)
    return dict(zip(keys, p.tolist()))


def brands_in_text(text: str, type_key: Optional[str] = None) -> List[str]:
    """Company names printed in the photo (longest names first, so 'Voltas Beko' beats 'Voltas')."""
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()) + " "
    pool = BRANDS.get(type_key, []) + [b for b in ALL_BRANDS if b not in BRANDS.get(type_key, [])]
    found = []
    for b in sorted(pool, key=len, reverse=True):
        key = " " + re.sub(r"[^a-z0-9 ]+", " ", b.lower()) + " "
        if key in low and not any(b.lower() in f.lower() for f in found):
            found.append(b)
    return found


_MODEL_NO = re.compile(r"\b(?=[A-Z0-9\-/]{4,20}\b)(?=[A-Z0-9\-/]*\d)(?=[A-Z0-9\-/]*[A-Z])[A-Z0-9][A-Z0-9\-/]+\b")
_NOT_MODEL = re.compile(r"^\d+(V|W|HZ|A|KW|KG|L|MM|CM)$|^(ISI|BEE|IS\d+|CM/L.*)$", re.I)


def model_numbers(text: str) -> List[str]:
    """Things that look like a model number (letters and digits) in the photo's text."""
    seen = []
    for m in _MODEL_NO.finditer((text or "").upper()):
        s = m.group(0).strip("-/")
        if len(s) >= 4 and not _NOT_MODEL.match(s) and s not in seen:
            seen.append(s)
    return seen[:3]


def recognise(data: bytes, text: str = "", image_vec: Optional[np.ndarray] = None, embed=None) -> dict:
    """
    {"types": [{"type_key","name","score","typical_w"}], "brand": {"name","source","score"} | None,
     "model_no": str | None, "why": str | None}.  `image_vec`/`embed` let tests pass stand-ins.
    """
    out = {"types": [], "brand": None, "model_no": None, "why": None}
    embed = embed or embed_texts
    try:
        if image_vec is None:
            image_vec = embed_image(data)
    except Exception as exc:                                    # no onnxruntime, no download, bad file
        if isinstance(exc, (OSError, ValueError)) and "cannot identify image" in str(exc).lower():
            out["why"] = "This file could not be opened as a photo."
        else:
            out["why"] = f"Recognising appliances from photos is not available right now ({type(exc).__name__})."
        return out
    groups = dict(DESCRIPTIONS)
    groups["_none"] = NOT_APPLIANCE
    probs = classify(image_vec, groups, embed)
    import catalog
    ranked = sorted(((k, v) for k, v in probs.items()), key=lambda kv: -kv[1])
    if ranked[0][0] == "_none":
        out["why"] = "This does not look like a photo of an appliance. Take a photo of the appliance itself."
        return out
    if ranked[0][1] < MIN_TYPE:
        out["why"] = ("The app could not tell which appliance this is. Take a clearer photo of the whole "
                      "appliance in good light, or pick it from the list.")
        return out
    for k, v in ranked:
        if k == "_none" or v < MIN_TYPE and out["types"]:
            continue
        t = catalog.get_type(k)
        if t is None:
            continue
        spec = catalog.specs(catalog.default_model(k).id) or {}
        out["types"].append({"type_key": k, "name": t.name, "score": round(v, 3), "typical_w": spec.get("rated_w")})
        if len(out["types"]) == 3:
            break
    top = out["types"][0]["type_key"]
    printed = brands_in_text(text, top)
    if printed:
        out["brand"] = {"name": printed[0], "source": "printed on it", "score": 1.0}
    elif BRANDS.get(top):
        name = catalog.get_type(top).name.lower()
        bp = classify(image_vec, {b: [f"a {b} {name}", f"the {b} logo on a {name}"] for b in BRANDS[top]}, embed)
        best = sorted(bp.items(), key=lambda kv: -kv[1])
        if best[0][1] >= MIN_BRAND and best[0][1] - (best[1][1] if len(best) > 1 else 0) >= BRAND_MARGIN:
            out["brand"] = {"name": best[0][0], "source": "logo or look", "score": round(best[0][1], 3)}
    nums = model_numbers(text)
    out["model_no"] = nums[0] if nums else None
    return out
