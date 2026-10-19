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

# Which CLIP to use, chosen by measured accuracy on real photos (tests/test_vision_eval.py).
# ENERGYPULSE_CLIP="<repo>|<vision file>" overrides it (used by the test that compares sizes).
MODELS = {
    "base32": ("Xenova/clip-vit-base-patch32", "onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx"),
    "base16": ("Xenova/clip-vit-base-patch16", "onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx"),
    "large14": ("Xenova/clip-vit-large-patch14", "onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx"),
    "large14q4": ("Xenova/clip-vit-large-patch14", "onnx/vision_model_q4.onnx", "onnx/text_model_quantized.onnx"),
}
# Measured on 84 real photos (CI, 2026-10-19): base32 75.0% correct, base16 84.5%, large14 86.9%,
# large14q4 89.3% (and 95.2% with the right type in the top three).  large14q4 is used; if the
# host cannot load it (memory), FALLBACK is used instead.
DEFAULT_MODEL = "large14q4"
FALLBACK_MODEL = "base16"
REPO, FILES = "", {}
CACHE_DIR = os.environ.get("ENERGYPULSE_MODEL_CACHE") or os.path.join(os.path.expanduser("~"), ".cache",
                                                                       "energypulse", "clip")


def configure(name: Optional[str] = None) -> str:
    """Pick the CLIP size (forgets anything loaded or worked out with the previous one)."""
    global REPO, FILES
    name = name or os.environ.get("ENERGYPULSE_CLIP") or DEFAULT_MODEL
    repo, vision, text = MODELS.get(name, MODELS[DEFAULT_MODEL])
    REPO, FILES = repo, {"vision": vision, "text": text, "tokenizer": "tokenizer.json"}
    _SESS.clear()
    _TEXT_CACHE.clear()
    return name


MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype="float32")
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype="float32")
SCALE = 100.0                 # CLIP's learned logit scale
BOS, EOS = 49406, 49407       # CLIP's start- and end-of-text tokens
MIN_TYPE = 0.20               # the best type must reach this, or the app says it is not sure
MIN_SHOW_OTHER = 0.10         # other likely types listed only above this
NONE_VETO = 0.50              # "not an appliance" must beat all appliances together by this much
MIN_BRAND, BRAND_MARGIN = 0.35, 0.15

# Several descriptions per type; their embeddings are averaged (prompt ensembling).
DESCRIPTIONS: Dict[str, List[str]] = {
    "ac": ["a split air conditioner indoor unit mounted high on a wall",
           "a white wall-mounted AC unit", "a window air conditioner", "an air conditioner outdoor unit"],
    "fan": ["a ceiling fan with three blades", "a ceiling fan seen from below", "a ceiling fan hanging from the ceiling",
            "a pedestal fan", "a table fan", "a wall mounted fan"],
    "cooler": ["a swamp cooler", "a portable evaporative cooler with a fan and water tank", "a desert air cooler", "a plastic room air cooler with honeycomb pads", "an evaporative air cooler"],
    "fridge": ["a refrigerator", "a double door fridge", "a single door refrigerator in a kitchen"],
    "microwave": ["a microwave oven", "a microwave oven on a kitchen counter"],
    "induction": ["an induction cooktop", "an electric induction stove with a glass top"],
    "kettle": ["a cordless electric kettle on its base", "a glass electric kettle boiling water", "an electric kettle", "a stainless steel electric kettle"],
    "mixer": ["a blender with a glass jar", "an Indian mixer grinder with jars", "a mixer grinder", "a kitchen blender"],
    "otg": ["an oven toaster grill", "an OTG oven", "a small countertop electric oven"],
    "chimney": ["a kitchen chimney hood above a stove", "a kitchen exhaust hood"],
    "purifier": ["an RO water purifier mounted on a kitchen wall", "a water purifier with a tap"],
    "dishwasher": ["a dishwasher", "an open dishwasher with racks"],
    "washing_machine": ["a washing machine in a laundry room", "a row of washing machines in a laundromat", "a front load washing machine", "a top load washing machine", "a semi automatic washing machine"],
    "iron": ["a modern steam iron with a pointed soleplate", "an iron for pressing clothes with a handle and a cord", "a person ironing clothes on an ironing board", "an old vintage flat iron", "an electric clothes iron", "a steam iron"],
    "vacuum": ["an upright vacuum cleaner", "a robot vacuum cleaner on the floor", "a handheld cordless vacuum cleaner", "a person vacuuming a carpet", "a vacuum cleaner", "a canister vacuum cleaner"],
    "geyser": ["a white electric water heater with a temperature dial", "a tall cylindrical water heater tank standing on the floor", "a hot water cylinder boiler", "an electric water heater geyser tank on a bathroom wall", "a geyser", "a storage water heater"],
    "water_pump": ["an electric water pump motor", "a monoblock water pump", "a submersible pump motor"],
    "heater": ["a portable space heater", "a room heater", "an electric fan heater", "an oil filled radiator heater"],
    "tv": ["a television in a living room", "an old CRT television set", "a TV on a stand", "a television", "a flat screen TV on a wall", "an LED TV"],
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

# Model classes in the catalogue that LOOK different, so a photo can tell them apart.  Models
# that look alike (1.5 ton 3-star vs 5-star, TV sizes) are not guessed: the user picks.
VISUAL_CLASSES: Dict[str, Dict[str, List[str]]] = {
    "ac": {"ac_1t_window_3s": ["a window air conditioner fitted in a window", "a box shaped window AC"],
           "ac_15t_split_3s": ["a split air conditioner indoor unit mounted on a wall", "a long white wall AC unit"]},
    "fridge": {"fridge_sd_190": ["a single door refrigerator", "a small fridge with one door"],
               "fridge_dd_250": ["a double door refrigerator with the freezer door on top", "a fridge with two doors one above the other"],
               "fridge_sbs_600": ["a side by side refrigerator with two tall doors", "a large two door fridge opening from the middle"]},
    "washing_machine": {"wm_semi_7": ["a semi automatic twin tub washing machine with two lids"],
                        "wm_top_7": ["a top loading washing machine with a lid on top"],
                        "wm_front_7": ["a front loading washing machine with a round glass door"]},
    "geyser": {"geyser_25": ["a large storage water heater tank"], "geyser_instant_3": ["a small instant water heater"]},
    "cooler": {"cooler_desert": ["a large desert air cooler"], "cooler_personal": ["a small personal air cooler"]},
}
MIN_CLASS, CLASS_MARGIN = 0.55, 0.20

_SESS: dict = {}
_TEXT_CACHE: Dict[str, np.ndarray] = {}


# ----------------------------------------------------------------- the model
def _download(rel: str) -> str:
    """The model file in the local cache, downloaded from Hugging Face the first time."""
    path = os.path.join(CACHE_DIR, REPO.replace("/", "_") + "__" + rel.replace("/", "_"))
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


def _load(part: str = "vision"):
    """The vision or the text half of CLIP (and the tokenizer), each loaded when first needed."""
    if not FILES:
        configure()
    if part not in _SESS:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        try:
            if part == "text":
                from tokenizers import Tokenizer
                _SESS["tok"] = Tokenizer.from_file(_download(FILES["tokenizer"]))
            _SESS[part] = ort.InferenceSession(_download(FILES[part]), opts, providers=["CPUExecutionProvider"])
        except (MemoryError, RuntimeError, OSError) as exc:
            current = next((k for k, v in MODELS.items() if v[0] == REPO and v[1] == FILES["vision"]), "")
            if current == FALLBACK_MODEL:
                raise
            import logging
            logging.getLogger("energypulse.vision").warning(
                "Could not load %s (%s); using the smaller %s", current, type(exc).__name__, FALLBACK_MODEL)
            configure(FALLBACK_MODEL)                       # smaller model: the descriptions are worked out again
            return _load(part)
    return _SESS


def release_text_model() -> None:
    """Free the text half once every fixed description has been worked out (saves memory)."""
    _SESS.pop("text", None)
    _SESS.pop("tok", None)


def all_fixed_texts() -> List[str]:
    """Every description the app uses, so they can be worked out once at start."""
    texts = [t for v in DESCRIPTIONS.values() for t in v] + NOT_APPLIANCE
    texts += [t for v in VISUAL_CLASSES.values() for d in v.values() for t in d]
    for key, brands in BRANDS.items():
        t = key.replace("_", " ")
        try:
            import catalog
            t = catalog.get_type(key).name.lower()
        except Exception:
            pass
        texts += [x for b in brands for x in (f"a {b} {t}", f"the {b} logo on a {t}")]
    return _phrasings(texts)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)


def _pick_output(outputs, sess, want: str) -> np.ndarray:
    names = [o.name for o in sess.get_outputs()]
    return np.asarray(outputs[names.index(want)] if want in names else outputs[0])


def embed_texts(texts: List[str]) -> np.ndarray:
    """Unit-length CLIP embeddings for descriptions (cached per text)."""
    missing = [t for t in texts if t not in _TEXT_CACHE]
    if missing:
        s = _load("text")
        enc = s["tok"].encode_batch(missing)
        ids = []
        for e in enc:
            i = list(e.ids)
            if not i or i[0] != BOS:                        # CLIP reads the sentence between these two marks
                i = [BOS] + i
            if i[-1] != EOS:
                i = i + [EOS]
            ids.append(i[:76] + [EOS] if len(i) > 77 else i)
        _SESS["tokens_seen"] = ids[0]
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


def _open(data: bytes):
    import io
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if img.mode in ("RGBA", "LA", "P"):                     # a transparent background becomes white, not black
        img = img.convert("RGBA")
        back = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(back, img)
    return img.convert("RGB")


def _to_input(img) -> np.ndarray:
    x = (np.asarray(img, dtype="float32") / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1).astype("float32")


def views(img) -> list:
    """Several looks at one photo: the centre, the whole photo (padded, nothing cut off), a closer
    centre, and the centre mirrored.  Averaging them is steadier than one crop."""
    from PIL import Image, ImageOps
    out = []
    w, h = img.size
    k = 224 / min(w, h)
    big = img.resize((max(224, round(w * k)), max(224, round(h * k))), Image.BICUBIC)
    bw, bh = big.size
    centre = big.crop(((bw - 224) // 2, (bh - 224) // 2, (bw - 224) // 2 + 224, (bh - 224) // 2 + 224))
    out.append(centre)
    side = max(w, h)
    pad = Image.new("RGB", (side, side), (255, 255, 255))
    pad.paste(img, ((side - w) // 2, (side - h) // 2))
    out.append(pad.resize((224, 224), Image.BICUBIC))
    cw, ch = int(w * 0.8), int(h * 0.8)
    close = img.crop(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    k2 = 224 / min(close.size)
    close = close.resize((max(224, round(close.width * k2)), max(224, round(close.height * k2))), Image.BICUBIC)
    out.append(close.crop(((close.width - 224) // 2, (close.height - 224) // 2,
                           (close.width - 224) // 2 + 224, (close.height - 224) // 2 + 224)))
    out.append(ImageOps.mirror(centre))
    return out


def preprocess(data: bytes, multi: bool = True) -> np.ndarray:
    """CLIP's own preparation (resize, crop, normalise, channels first), for one or several views."""
    img = _open(data)
    vs = views(img) if multi else views(img)[:1]
    return np.stack([_to_input(v) for v in vs])


def embed_image(data: bytes, multi: bool = True) -> np.ndarray:
    s = _load("vision")
    x = preprocess(data, multi)
    name = s["vision"].get_inputs()[0].name
    vecs = []
    for i in range(len(x)):                                 # one view at a time: lower peak memory
        vecs.append(_unit(_pick_output(s["vision"].run(None, {name: x[i:i + 1]}), s["vision"], "image_embeds"))[0])
    return _unit(np.mean(vecs, axis=0))


def logo_text(data: bytes) -> str:
    """Text read for logos: enlarged, normal and inverted (white logo on a dark panel), sparse layout."""
    try:
        import io
        import pytesseract
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("L")
        if max(img.size) < 1600:
            k = 1600 / max(img.size)
            img = img.resize((round(img.width * k), round(img.height * k)))
        img = ImageOps.autocontrast(img)
        return "\n".join(pytesseract.image_to_string(im, config="--psm 11") for im in (img, ImageOps.invert(img)))
    except Exception:
        return ""


_WARM = {"started": False}


def warm_up() -> None:
    """Fetch and load the model in the background, so the first photo does not wait for it."""
    if _WARM["started"] or os.environ.get("ENERGYPULSE_NO_WARMUP"):
        return
    _WARM["started"] = True
    import threading

    def run():
        try:
            _load("vision")
            embed_texts(all_fixed_texts())
            release_text_model()                            # descriptions are kept; the text half is not needed
        except Exception:
            pass                                            # the first photo will try again and say why
    threading.Thread(target=run, name="clip-warm-up", daemon=True).start()


# ----------------------------------------------------------------- deciding
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _phrasings(descs: List[str]) -> List[str]:
    """Each description as written and as 'a photo of ...' (the wording CLIP was trained on)."""
    out = []
    for d in descs:
        out.append(d)
        if not d.startswith("a photo of"):
            out.append(f"a photo of {d}")
    return out


def classify(image_vec: np.ndarray, groups: Dict[str, List[str]], embed=embed_texts) -> Dict[str, float]:
    """Probability per group: each group's descriptions averaged into one direction, then softmax."""
    keys = list(groups)
    centres = np.stack([_unit(embed(_phrasings(groups[k])).mean(axis=0)) for k in keys])
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


def recognise(data: bytes, text: str = "", image_vec: Optional[np.ndarray] = None, embed=None,
              multi_view: bool = True) -> dict:
    """
    {"types": [{"type_key","name","score","typical_w"}], "brand": {"name","source","score"} | None,
     "model_no": str | None, "why": str | None}.  `image_vec`/`embed` let tests pass stand-ins.
    """
    out = {"types": [], "brand": None, "model_no": None, "model_class": None, "why": None}
    embed = embed or embed_texts
    try:
        if image_vec is None:
            image_vec = embed_image(data, multi_view)
    except Exception as exc:                                    # no onnxruntime, no download, bad file
        if isinstance(exc, (OSError, ValueError)) and "cannot identify image" in str(exc).lower():
            out["why"] = "This file could not be opened as a photo."
        else:
            out["why"] = f"Recognising appliances from photos is not available right now ({type(exc).__name__})."
        return out
    # "Not an appliance" is a veto, judged against the best appliance; the type itself is chosen
    # among appliance descriptions only, so a plain product photo is not lost to "a document".
    probs = classify(image_vec, dict(DESCRIPTIONS), embed)
    import catalog
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    veto = classify(image_vec, {"_none": NOT_APPLIANCE, ranked[0][0]: DESCRIPTIONS[ranked[0][0]]}, embed)
    out["scores"] = {"top": [(k, round(v, 3)) for k, v in ranked[:3]], "not_appliance": round(veto["_none"], 3)}
    if veto["_none"] > NONE_VETO:
        out["why"] = "This does not look like a photo of an appliance. Take a photo of the appliance itself."
        return out
    if ranked[0][1] < MIN_TYPE:
        out["why"] = ("The app could not tell which appliance this is. Take a clearer photo of the whole "
                      "appliance in good light, or pick it from the list.")
        return out
    for k, v in ranked[:3]:
        if out["types"] and v < MIN_SHOW_OTHER:
            break
        t = catalog.get_type(k)
        if t is None:
            continue
        spec = catalog.specs(catalog.default_model(k).id) or {}
        out["types"].append({"type_key": k, "name": t.name, "score": round(v, 3), "typical_w": spec.get("rated_w")})
    top = out["types"][0]["type_key"]
    if VISUAL_CLASSES.get(top):
        cp = sorted(classify(image_vec, VISUAL_CLASSES[top], embed).items(), key=lambda kv: -kv[1])
        if cp[0][1] >= MIN_CLASS and cp[0][1] - (cp[1][1] if len(cp) > 1 else 0) >= CLASS_MARGIN:
            m = catalog.specs(cp[0][0]) or {}
            out["model_class"] = {"model_id": cp[0][0], "label": m.get("label", cp[0][0]), "score": round(cp[0][1], 3)}
    printed = brands_in_text(text, top)
    if not printed and data:
        extra = logo_text(data)                             # logos are often small or light-on-dark
        printed = brands_in_text(extra, top)
        text = (text or "") + "\n" + extra
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


configure()
