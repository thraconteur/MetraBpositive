"""
Regression tests for bugs found by independent review (30 Sept 2026).
Each one produced a false finding or refused a valid photo.
"""

import io
import re

import cv2
import numpy as np
import pytest


def test_decimal_price_is_not_read_as_an_iso_date():
    from src.extraction.fields import clean

    assert clean("MRP Rs. 1999.10 (Incl. of all taxes)") == "MRP Rs. 1999.10 (Incl. of all taxes)"
    assert clean("Energy 2012.5 kJ") == "Energy 2012.5 kJ"
    assert clean("Date of Mfg. 2026-01-15") == "Date of Mfg. 15/01/2026"


def test_tax_negation_needs_the_word_tax():
    from src.core.rules_engine import _TAX_NEGATED

    for ok in ("Batteries not included", "Marketed exclusively by ABC", "Excluding gift box"):
        assert not _TAX_NEGATED.search(ok), ok
    for bad in ("(not inclusive of all taxes)", "(excl. of all taxes)", "+ taxes", "plus applicable taxes"):
        assert _TAX_NEGATED.search(bad), bad


def test_banned_word_needs_its_own_word():
    from src.core.rules_engine import _find_banned_phrase

    phrases = ["min", "min.", "around", "approx"]
    assert _find_banned_phrase("Net 50 g 5min noodles", phrases) is None
    assert _find_banned_phrase("Jeera (Cumin)", phrases) is None
    assert _find_banned_phrase("Wraparound", phrases) is None
    assert _find_banned_phrase("Net Wt 100 g (Min.)", phrases) in ("min", "min.")


def test_second_mrp_takes_only_the_amount_after_the_label():
    from src.core.rules_engine import _AMOUNT_AFTER_TAG, _MRP_TAG

    def amounts(t):
        out = []
        for m in _MRP_TAG.finditer(t):
            a = _AMOUNT_AFTER_TAG.match(t[m.end():])
            if a:
                out.append(float(a.group(1).replace(",", "")))
        return out

    assert amounts("MRP Rs. 10.00 (Incl. of all taxes) PKD 05/26 B.No. 123") == [10.0]
    assert amounts("MRP: Rs. 120 USP Rs. 2.40 per g") == [120.0]
    assert amounts("MRP Rs. 50.00 / MRP Rs. 45.00") == [50.0, 45.0]


def test_when_packed_is_a_phrase_not_a_word_ending():
    pat = re.compile(r"(?<![a-z])(?:when|hen|wen)\s*pa?c?ked\b", re.I)
    assert pat.search("Net Wt. 200 g (when packed)")
    assert not pat.search("Chicken packed in brine")
    assert not pat.search("Golden packed")


def test_import_pack_detection_is_anchored():
    pat = re.compile(r"imported\.?\s*(?:(?:,|&|and)\s*\w+\.?\s*){0,2}by\b|\bimporter\s*[:\-]|country of origin")
    assert not pat.search("made with imported cocoa. manufactured by abc foods")
    assert pat.search("imported & marketed by: xyz traders")
    assert pat.search("importer: xyz traders")


fastapi = pytest.importorskip("fastapi")


def test_phone_photo_keeps_its_exif_rotation():
    from PIL import Image

    from src.api.main import _decode_photo

    img = Image.new("RGB", (400, 200), (255, 255, 255))
    exif = img.getexif()
    exif[0x0112] = 6                      # "rotate 90": a portrait phone photo
    b = io.BytesIO()
    img.save(b, "JPEG", exif=exif.tobytes())
    assert _decode_photo(b.getvalue()).shape[:2] == (400, 200)


def test_motion_photo_with_trailing_data_is_accepted_but_a_cut_jpeg_is_not():
    from fastapi import HTTPException

    from src.api.main import _decode_photo

    jp = cv2.imencode(".jpg", np.full((300, 400, 3), 200, np.uint8))[1].tobytes()
    assert _decode_photo(jp + b"MotionPhoto_Data" + b"\x01" * 500).shape[:2] == (300, 400)
    with pytest.raises(HTTPException) as e:
        _decode_photo(jp[: len(jp) // 2])
    assert e.value.status_code == 400
