"""
Recognise the appliance in a photo, for people who have no rating plate to photograph.

An image model (ConvNeXt-Tiny, trained on the 1,000 everyday objects of ImageNet,
about 110 MB downloaded once on first use, run by the TensorFlow the app already uses)
scores the photo.  It was chosen by testing on a real user's ceiling-fan photo: ConvNeXt-Tiny
said "electric fan" at 53%, EfficientNetV2-S at 29%, while MobileNetV2 and EfficientNet-B0
did not see a fan at all.  The
scores of the ImageNet objects that are household appliances are added up per
appliance type in the app's catalogue, e.g. "electric fan" -> ceiling fan,
"washer" -> washing machine.  The top few types are offered for the user to confirm;
the typical figures for that type then come from the catalogue.

Limits, said to the user: ImageNet has no air conditioner, geyser, water pump,
chimney or RO purifier, so those cannot be recognised and are picked from the list.
A guess below MIN_SHOW is not offered.  Nothing is ever filled in without the user
choosing it.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

MIN_SHOW = 0.08          # a type must get at least this share of the score to be offered

# ImageNet class name -> catalogue type key.  Several ImageNet names can point to one type.
IMAGENET_TO_TYPE: Dict[str, str] = {
    "electric_fan": "fan",
    "refrigerator": "fridge",
    "microwave": "microwave",
    "washer": "washing_machine",
    "dishwasher": "dishwasher",
    "iron": "iron",
    "vacuum": "vacuum",
    "television": "tv", "screen": "tv", "home_theater": "tv", "entertainment_center": "tv",
    "monitor": "desktop", "desktop_computer": "desktop", "computer_keyboard": "desktop", "mouse": "desktop",
    "laptop": "laptop", "notebook": "laptop",
    "space_heater": "heater", "radiator": "heater",
    "toaster": "otg", "waffle_iron": "otg", "rotisserie": "otg",
    "stove": "induction", "Dutch_oven": "induction", "frying_pan": "induction", "wok": "induction",
    "teapot": "kettle", "coffeepot": "kettle", "water_jug": "kettle",
    "loudspeaker": "speaker", "radio": "speaker",
    "modem": "router",
    "table_lamp": "lights", "lampshade": "lights", "spotlight": "lights", "candle": "lights",
    "hand_blower": "heater",
    "espresso_maker": "kettle",
    "Crock_Pot": "induction",
    "cellular_telephone": "charger", "iPod": "charger",
    "remote_control": "settop", "cassette_player": "settop", "tape_player": "settop",
}
CANNOT_RECOGNISE = ("ac", "geyser", "water_pump", "chimney", "purifier", "cooler", "mixer")

_MODEL = None
_LABELS: Optional[List[str]] = None


def _load():
    """The image model and its 1,000 labels, loaded once (downloads ~110 MB the first time)."""
    global _MODEL, _LABELS
    if _MODEL is None:
        import keras
        _MODEL = keras.applications.ConvNeXtTiny(weights="imagenet", include_top=True)
        from keras.applications.convnext import decode_predictions
        dummy = np.zeros((1, 1000), dtype="float32")
        dummy[0] = np.arange(1000)                      # descending order gives every label once
        decoded = decode_predictions(dummy, top=1000)[0]
        labels = [""] * 1000
        for _, name, score in decoded:
            labels[int(score)] = name
        _LABELS = labels
    return _MODEL, _LABELS


def _prepare(data: bytes) -> np.ndarray:
    import io
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img = ImageOps.fit(img, (224, 224))                 # centre crop to the model's input size
    return np.asarray(img, dtype="float32")[None]       # 0-255: ConvNeXt normalises inside the model


def scores_to_types(probs: np.ndarray, labels: List[str], top: int = 3) -> List[dict]:
    """ImageNet probabilities -> appliance types, best first: [{"type_key", "score", "seen_as"}]."""
    totals: Dict[str, float] = {}
    best_label: Dict[str, Tuple[float, str]] = {}
    for p, name in zip(np.asarray(probs, dtype=float).ravel(), labels):
        key = IMAGENET_TO_TYPE.get(name)
        if key is None:
            continue
        totals[key] = totals.get(key, 0.0) + float(p)
        if p > best_label.get(key, (-1.0, ""))[0]:
            best_label[key] = (float(p), name)
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    return [{"type_key": k, "score": round(v, 3), "seen_as": best_label[k][1].replace("_", " ")}
            for k, v in ranked[:top] if v >= MIN_SHOW]


def recognise(data: bytes, model=None, labels=None) -> Tuple[List[dict], Optional[str]]:
    """
    ([guesses], None) or ([], reason).  Each guess: {"type_key", "name", "score", "typical_w", "seen_as"}.
    `model`/`labels` let tests pass a stand-in.
    """
    try:
        x = _prepare(data)
    except Exception:
        return [], "This file could not be opened as a photo."
    try:
        if model is None:
            model, labels = _load()
        probs = np.asarray(model.predict(x, verbose=0) if hasattr(model, "predict") else model(x))[0]
    except Exception as exc:                            # no TensorFlow, or the weights could not be fetched
        return [], f"Recognising appliances from photos is not available right now ({type(exc).__name__})."
    import catalog
    out = []
    for g in scores_to_types(probs, labels):
        t = catalog.get_type(g["type_key"])
        if t is None:
            continue
        spec = catalog.specs(catalog.default_model(t.key).id) or {}
        out.append({**g, "name": t.name, "typical_w": spec.get("rated_w")})
    if not out:
        return [], ("The photo does not look like an appliance the app can recognise. Pick it from the list. "
                    "(Air conditioners, geysers, water pumps, chimneys and RO purifiers cannot be recognised "
                    "from a photo.)")
    return out, None
