"""TEMPORARY probe: what several free image models see in the fan photo (records only)."""
import os
import numpy as np
import pytest

FAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ceiling_fan.jpg")


def _img(size):
    from PIL import Image, ImageOps
    return np.asarray(ImageOps.fit(Image.open(FAN).convert("RGB"), (size, size)), dtype="float32")[None]


@pytest.mark.parametrize("name", ["MobileNetV2", "EfficientNetB0", "EfficientNetV2S", "ConvNeXtTiny"])
def test_probe(name, metric):
    keras = pytest.importorskip("keras")
    app = keras.applications
    cfg = {"MobileNetV2": (app.MobileNetV2, app.mobilenet_v2, 224),
           "EfficientNetB0": (app.EfficientNetB0, app.efficientnet, 224),
           "EfficientNetV2S": (app.EfficientNetV2S, app.efficientnet_v2, 384),
           "ConvNeXtTiny": (app.ConvNeXtTiny, app.convnext, 224)}[name]
    model = cfg[0](weights="imagenet")
    x = cfg[1].preprocess_input(_img(cfg[2]))
    p = model.predict(x, verbose=0)
    top = cfg[1].decode_predictions(p, top=8)[0]
    metric(f"probe_{name}", [(n, round(float(s), 3)) for _, n, s in top])
