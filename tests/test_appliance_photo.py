"""Photos of appliances: rating plates are read only when the text really is a plate; photos of the
appliance itself are recognised - type, company, model number (appliance_vision.py)."""
import os

import numpy as np
import pytest

import appliance_vision as V
from nameplate import looks_like_plate, parse_plate

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.join(HERE, "fixtures", "ceiling_fan.jpg")      # a user's photo of their ceiling fan (no label)
FAN_NOISE = "te\n> ae gt to a\n— eats a aE ie\neg = SS\n: Sy Ly Baa\" i :\n5 A\n"
PLATE = "CEILING FAN\nModel No. ACE-1200\n230V~ 50Hz\nRated Input 75W\nMade in India\n"


# ------------------------------------------------------------------ rating plates
def test_noise_from_an_appliance_photo_is_not_taken_as_a_plate():
    found = parse_plate(FAN_NOISE)
    assert found.get("current_a") == 5.0 and not looks_like_plate(found)


def test_a_real_plate_is_trusted():
    found = parse_plate(PLATE)
    assert looks_like_plate(found) and found["rated_w"] == 75 and found["voltage_v"] == 230


def test_one_sign_alone_is_not_enough():
    assert not looks_like_plate(parse_plate("1200 W"))
    assert looks_like_plate(parse_plate("1200 W 230 V"))


# ------------------------------------------------------------------ deciding (stand-in embeddings)
def _fake_world(tie=()):
    """Every description gets its own random direction; an image 'is' whichever texts we point it at.
    Descriptions naming a brand in `tie` all share one direction, so none of those brands can win."""
    rng = np.random.default_rng(0)
    cache = {}
    shared = rng.normal(size=64)

    def embed(texts):
        for t in texts:
            if t in cache:
                continue
            if any(f" {b} " in f" {t} " for b in tie):
                cache[t] = shared / np.linalg.norm(shared)
            else:
                v = rng.normal(size=64)
                cache[t] = v / np.linalg.norm(v)
        return np.stack([cache[t] for t in texts])

    def image_like(*texts):
        v = embed(list(texts)).mean(axis=0)
        return v / np.linalg.norm(v)
    return embed, image_like


def test_type_comes_from_the_descriptions():
    embed, like = _fake_world()
    r = V.recognise(b"", image_vec=like(*V.DESCRIPTIONS["geyser"]), embed=embed)
    assert r["types"][0]["type_key"] == "geyser" and r["types"][0]["name"] == "Water heater (geyser)"
    assert r["types"][0]["score"] > 0.9 and r["types"][0]["typical_w"]


def test_every_catalogue_type_has_descriptions_and_brands():
    import catalog
    keys = {t.key for t in catalog.CATALOG}
    assert keys <= set(V.DESCRIPTIONS) and keys <= set(V.BRANDS)


def test_not_an_appliance_is_said_plainly():
    embed, like = _fake_world()
    r = V.recognise(b"", image_vec=like(*V.NOT_APPLIANCE), embed=embed)
    assert r["types"] == [] and "does not look like a photo of an appliance" in r["why"]


def test_brand_printed_on_it_beats_a_guess_and_model_number_is_picked_out():
    embed, like = _fake_world()
    text = "LG\nDUAL Inverter\nModel: RS-Q19YNZE  230V 50Hz\nISI"
    r = V.recognise(b"", text=text, image_vec=like(*V.DESCRIPTIONS["ac"]), embed=embed)
    assert r["brand"] == {"name": "LG", "source": "printed on it", "score": 1.0}
    assert r["model_no"] == "RS-Q19YNZE"


def test_brand_from_the_look_only_when_it_clearly_wins():
    embed, like = _fake_world()
    name = "air conditioner"
    img = like(*V.DESCRIPTIONS["ac"], f"a Daikin {name}", f"the Daikin logo on a {name}")
    r = V.recognise(b"", image_vec=img, embed=embed)
    assert r["types"][0]["type_key"] == "ac" and r["brand"]["name"] == "Daikin" and r["brand"]["source"] == "logo or look"
    embed2, like2 = _fake_world(tie=V.BRANDS["ac"])
    plain = V.recognise(b"", image_vec=like2(*V.DESCRIPTIONS["ac"]), embed=embed2)
    assert plain["types"][0]["type_key"] == "ac" and plain["brand"] is None    # no clear winner: none claimed


def test_longest_brand_name_wins_and_numbers_are_not_models():
    assert V.brands_in_text("VOLTAS BEKO refrigerator", "fridge") == ["Voltas Beko"]
    assert V.brands_in_text("made by A.O. Smith", "geyser") == []        # punctuation inside the name
    assert V.brands_in_text("AO SMITH geyser", "geyser") == ["AO Smith"]
    assert V.model_numbers("230V 50HZ 1200W ISI") == []
    assert V.model_numbers("Model GL-B201 230V") == ["GL-B201"]


def test_unavailable_model_and_bad_files_are_reported():
    r = V.recognise(b"not an image")
    assert r["types"] == [] and r["why"]


# ------------------------------------------------------------------ the real model
def test_real_clip_recognises_the_users_fan_photo(metric):
    """The real CLIP model (downloads ~154 MB) on the ceiling-fan photo a user uploaded."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    r = V.recognise(open(FAN, "rb").read())
    if r["why"] and "not available" in r["why"]:
        pytest.skip(r["why"])
    metric("clip_fan_photo", r)
    assert r["types"] and r["types"][0]["type_key"] == "fan", r


def test_real_clip_refuses_a_page_of_text(metric):
    pytest.importorskip("onnxruntime")
    from PIL import Image, ImageDraw
    import io
    img = Image.new("RGB", (400, 500), "white")
    d = ImageDraw.Draw(img)
    for i in range(20):
        d.text((20, 20 + i * 22), "Invoice line item quantity price total amount due " * 2, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r = V.recognise(buf.getvalue())
    if r["why"] and "not available" in r["why"]:
        pytest.skip(r["why"])
    metric("clip_text_page", r)
    assert r["types"] == [] and r["why"], r
