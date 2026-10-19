"""Exact product: model number read from label / QR / invoice, confirmed on the web (product_finder.py)."""
import pytest

import product_finder as pf
import spec_search

LABEL = """LG
SPLIT TYPE ROOM AIR CONDITIONER
Indoor Unit Model: RS-Q19YNZE
Rated Voltage 230V~ 50Hz
Cooling Capacity 5000 W  Power Input 1540 W
Made in India"""
INVOICE = """TAX INVOICE  Croma, Mysuru
1  LG 1.5 Ton 5 Star Dual Inverter Split AC  Model No: RS-Q19YNZE  Qty 1  Rs 41,990
GSTIN 29ABCDE1234F1Z5"""


def test_model_number_from_a_label_an_invoice_and_a_qr_code():
    assert pf.model_candidates([LABEL])[0] == ("RS-Q19YNZE", "label text 'Model'")
    assert pf.model_candidates([INVOICE])[0][0] == "RS-Q19YNZE"
    assert pf.model_candidates([], ["https://www.lg.com/in/air-conditioners/lg-rs-q19ynze"])[0] == \
        ("RS-Q19YNZE", "QR / barcode (product link)")
    assert pf.model_candidates(["230V 50Hz 1540W"]) == []                     # ratings are not models


def _look_up(pages):
    def fake(query, model=""):
        return {"ok": True, "query": query, "sources": pages}
    return fake


def _page(mentions, w=1540, star=5):
    return {"title": "t", "url": "https://example.com", "mentions_model": mentions, "found": {"rated_w": w, "star": star}}


def test_exact_only_when_two_pages_name_the_model_and_agree():
    r = pf.confirm("LG", "RS-Q19YNZE", "Air conditioner", look_up=_look_up([_page(True), _page(True), _page(False, 999)]))
    assert r["status"] == "exact" and r["figures"]["rated_w"] == 1540 and r["figures"]["star"] == 5


def test_one_page_or_disagreeing_pages_are_not_exact():
    one = pf.confirm("LG", "RS-Q19YNZE", look_up=_look_up([_page(True), _page(False)]))
    assert one["status"] == "not confirmed" and "Only one page" in one["reason"] and one["figures"] == {}
    differ = pf.confirm("LG", "X1234", look_up=_look_up([_page(True, 1000, 3), _page(True, 1500, 5)]))
    assert differ["status"] == "not confirmed" and "do not agree" in differ["reason"]
    none = pf.confirm("LG", "X1234", look_up=_look_up([_page(False)]))
    assert "No page found that names the model X1234" in none["reason"]


def test_search_offline_is_said_plainly():
    r = pf.confirm("LG", "X1234", look_up=lambda q, model="": {"ok": False, "sources": []})
    assert r["status"] == "not confirmed" and "could not be reached" in r["reason"]


def test_whole_process_tries_the_best_candidates_in_order(monkeypatch):
    monkeypatch.setattr(pf, "read_codes", lambda data: [])
    monkeypatch.setattr(pf, "photo_text", lambda data: LABEL)
    calls = []

    def confirm_fn(brand, model, type_name):
        calls.append((brand, model))
        return {"status": "exact", "reason": "ok", "query": f"{brand} {model}", "model_no": model, "brand": brand,
                "figures": {"rated_w": 1540}, "sources": []}
    r = pf.find_exact([b"img"], type_key="ac", confirm_fn=confirm_fn)
    assert r["status"] == "exact" and calls == [("LG", "RS-Q19YNZE")]
    assert "right side of the indoor unit" in r["where"]


def test_no_model_number_says_where_to_find_it(monkeypatch):
    monkeypatch.setattr(pf, "read_codes", lambda data: [])
    monkeypatch.setattr(pf, "photo_text", lambda data: "just a fan")
    r = pf.find_exact([b"img"], type_key="fan", confirm_fn=lambda *a: pytest.fail("should not search"))
    assert r["status"] == "no model number" and "top of the motor" in r["where"]


def test_web_lookup_marks_pages_that_name_the_model():
    pages = {"https://a.example/x": "<html><table><tr><td>Model</td><td>RS-Q19YNZE</td></tr>"
                                    "<tr><td>Power Input</td><td>1540 W</td></tr></table></html>",
             "https://b.example/y": "<html><p>Some other AC. Power consumption 1200 W</p></html>"}

    def post(url, data):
        return ('<a class="result__a" href="https://a.example/x">LG AC</a>'
                '<a class="result__a" href="https://b.example/y">Other</a>')

    def get(url, **kw):
        return pages.get(url)
    found = spec_search.look_up("LG RS-Q19YNZE", fetch_get=get, fetch_post=post, model="RS-Q19YNZE")
    by_url = {s["url"]: s for s in found["sources"]}
    assert by_url["https://a.example/x"]["mentions_model"] is True
    assert by_url["https://b.example/y"]["mentions_model"] is False


def test_real_qr_code_is_read():
    cv2 = pytest.importorskip("cv2")
    import io
    import numpy as np
    from PIL import Image
    qr = cv2.QRCodeEncoder.create().encode("MODEL:RS-Q19YNZE;SN:123456")
    img = Image.fromarray(np.kron(qr, np.ones((8, 8), dtype=qr.dtype))).convert("RGB")
    pad = Image.new("RGB", (img.width + 80, img.height + 80), "white")
    pad.paste(img, (40, 40))
    buf = io.BytesIO()
    pad.save(buf, "PNG")
    codes = pf.read_codes(buf.getvalue())
    assert codes == ["MODEL:RS-Q19YNZE;SN:123456"]
    assert pf.model_candidates([], codes)[0] == ("RS-Q19YNZE", "QR / barcode")


def test_pdf_invoice_text_is_read():
    """A minimal PDF written by hand (no extra library) with the invoice line in it."""
    pytest.importorskip("pypdf")
    text = "LG Split AC Model No: RS-Q19YNZE Qty 1"
    stream = f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for n, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    got = pf.invoice_text(out, "invoice.pdf")
    assert "RS-Q19YNZE" in got
    assert pf.model_candidates([got])[0][0] == "RS-Q19YNZE"
