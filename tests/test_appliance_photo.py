"""Photos of appliances: rating plates are read only when the text really is a plate; photos of the
appliance itself are recognised (appliance_vision.py)."""
import io
import os

import numpy as np
import pytest

from appliance_vision import IMAGENET_TO_TYPE, MIN_SHOW, recognise, scores_to_types
from nameplate import looks_like_plate, parse_plate

HERE = os.path.dirname(os.path.abspath(__file__))
FAN = os.path.join(HERE, "fixtures", "ceiling_fan.jpg")      # a user's photo of their ceiling fan (no label)

# What text recognition made of that fan photo in the app (random letters, one stray "5 A").
FAN_NOISE = "te\n> ae gt to a\n— eats a aE ie\neg = SS\n: Sy Ly Baa\" i :\n5 A\n"
PLATE = "CEILING FAN\nModel No. ACE-1200\n230V~ 50Hz\nRated Input 75W\nMade in India\n"


def test_noise_from_an_appliance_photo_is_not_taken_as_a_plate():
    found = parse_plate(FAN_NOISE)
    assert found.get("current_a") == 5.0 and found.get("from_current")     # the stray "5 A" is still parsed ...
    assert not looks_like_plate(found)                                     # ... but never trusted


def test_a_real_plate_is_trusted():
    found = parse_plate(PLATE)
    assert looks_like_plate(found)
    assert found["rated_w"] == 75 and found["voltage_v"] == 230 and found["frequency_hz"] == 50


def test_one_sign_alone_is_not_enough():
    assert not looks_like_plate(parse_plate("1200 W"))
    assert looks_like_plate(parse_plate("1200 W 230 V"))


def _labels():
    names = list(IMAGENET_TO_TYPE) + [f"other_{i}" for i in range(1000 - len(IMAGENET_TO_TYPE))]
    return names[:1000]


def test_scores_add_up_per_appliance_type():
    labels = _labels()
    p = np.zeros(1000)
    p[labels.index("television")], p[labels.index("screen")], p[labels.index("electric_fan")] = 0.3, 0.25, 0.4
    out = scores_to_types(p, labels)
    assert [g["type_key"] for g in out] == ["tv", "fan"] and out[0]["score"] == pytest.approx(0.55)
    assert out[0]["seen_as"] == "television"
    p2 = np.zeros(1000)
    p2[labels.index("electric_fan")] = MIN_SHOW / 2
    assert scores_to_types(p2, labels) == []                 # too unsure: nothing offered


class _Stub:
    def __init__(self, label, labels):
        self.p = np.zeros((1, 1000))
        self.p[0, labels.index(label)] = 0.9

    def predict(self, x, verbose=0):
        assert x.shape == (1, 224, 224, 3) and 0.0 <= x.min() and x.max() <= 255.0
        return self.p


def test_recognise_returns_the_type_name_and_typical_power():
    labels = _labels()
    out, why = recognise(open(FAN, "rb").read(), model=_Stub("electric_fan", labels), labels=labels)
    assert why is None and out[0]["type_key"] == "fan" and out[0]["name"] == "Ceiling fan"
    assert 30 <= out[0]["typical_w"] <= 120


def test_recognise_says_why_when_it_cannot_help():
    labels = _labels()
    out, why = recognise(open(FAN, "rb").read(), model=_Stub("other_5", labels), labels=labels)
    assert out == [] and "Air conditioners" in why
    out, why = recognise(b"not an image")
    assert out == [] and "could not be opened" in why

    class Broken:
        def predict(self, *a, **k):
            raise RuntimeError("no weights")
    out, why = recognise(open(FAN, "rb").read(), model=Broken(), labels=labels)
    assert out == [] and "not available right now" in why


def test_real_model_recognises_the_users_fan_photo(metric):
    """The real ConvNeXt-Tiny (downloads ImageNet weights) on the fan photo from the app."""
    pytest.importorskip("keras")
    out, why = recognise(open(FAN, "rb").read())
    if why and "not available" in why:
        pytest.skip(f"model could not be loaded here: {why}")
    metric("fan_photo_guesses", out or why)
    assert out and out[0]["type_key"] == "fan", (out, why)
