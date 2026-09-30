"""
Tests for the PaddleOCR path and for everything downstream of OCR that
had to change because Paddle returns LINES rather than words.

No Paddle model is ever loaded here. The backend is exercised through a
fake `paddleocr` module with the same call contract, which is enough to
pin down the things that went wrong on real photos: coordinates, the
constructor settings, caching, refinement, the bilingual re-read, and
the dump / replay round trip.
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.rules_engine import RuleConfig, RulesEngine, _has_tax_qualifier  # noqa: E402
from src.core.schema import BBox, Declaration, Outcome, ScanResult, TextSpan  # noqa: E402
from src.extraction.fields import RegexExtractor  # noqa: E402
from src.vision.ink import ink_extent  # noqa: E402
from src.vision.ocr import paddle as P  # noqa: E402
from src.vision.ocr.backends import OCRUnavailable, build_default_ocr  # noqa: E402
from src.vision.overlay import find_top_token  # noqa: E402


# =====================================================================
# A fake `paddleocr` module
# =====================================================================

class _FakeResult(dict):
    pass


def _poly(x, y, w, h):
    return np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])


class FakeRecogniser:
    """TextRecognition stand-in: returns queued (text, score) per crop."""

    def __init__(self, reads):
        self.reads = list(reads)
        self.calls = 0

    def predict(self, crops):
        self.calls += 1
        out = []
        for i, _ in enumerate(crops):
            text, score = self.reads[i % len(self.reads)]
            out.append({"rec_text": text, "rec_score": score})
        return out


def fake_paddle_module(regions_fn, created: list, recogniser=None):
    """
    `regions_fn(img)` -> list of (text, score, poly) in the coordinates
    of the image Paddle was handed.
    """
    mod = types.ModuleType("paddleocr")
    mod.__version__ = "fake-3.7"

    class PaddleOCR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.predict_calls = 0
            created.append(self)

        def predict(self, img, **kw):
            self.predict_calls += 1
            regs = regions_fn(img)
            return [_FakeResult(
                rec_texts=[r[0] for r in regs],
                rec_scores=[r[1] for r in regs],
                rec_polys=[r[2] for r in regs],
            )]

    class TextRecognition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._r = recogniser or FakeRecogniser([("", 0.0)])

        def predict(self, crops):
            return self._r.predict(crops)

    mod.PaddleOCR = PaddleOCR
    mod.TextRecognition = TextRecognition
    return mod


@pytest.fixture
def fresh_caches():
    P._PIPELINE_CACHE.clear()
    P._RECOGNISER_CACHE.clear()
    yield
    P._PIPELINE_CACHE.clear()
    P._RECOGNISER_CACHE.clear()


def _canvas(h=800, w=1400):
    img = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(img, "MRP Rs. 120.00", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


# =====================================================================
# Backend construction and coordinates
# =====================================================================

def test_document_preprocessing_is_disabled(monkeypatch, tmp_path, fresh_caches):
    """UVDoc / orientation put every box in a warped frame - must be off."""
    created = []
    monkeypatch.setitem(sys.modules, "paddleocr",
                        fake_paddle_module(lambda img: [], created))
    ocr = P.PaddleOCRBackend(cache_dir=None, dump_dir=None)
    ocr.recognise(_canvas())
    kw = created[0].kwargs
    assert kw["use_doc_unwarping"] is False
    assert kw["use_doc_orientation_classify"] is False
    assert kw["text_det_limit_type"] == "min"
    assert kw["text_det_limit_side_len"] >= 960
    assert kw["text_rec_score_thresh"] == 0.0
    assert kw["enable_mkldnn"] is P.mkldnn_safe()   # on for 3.2.x, off for 3.3.x


def test_small_images_upscaled_and_boxes_mapped_back(monkeypatch, fresh_caches):
    created = []
    seen_shapes = []

    def regions(img):
        seen_shapes.append(img.shape[:2])
        # Box drawn in the UPSCALED frame.
        return [("MRP Rs. 120.00", 0.95, _poly(200, 300, 600, 60))]

    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddle_module(regions, created))
    ocr = P.PaddleOCRBackend(cache_dir=None, dump_dir=None)
    img = np.full((500, 700, 3), 255, np.uint8)       # short side < 1200
    spans = ocr.recognise(img)
    assert seen_shapes[0] == (1000, 1400)             # 2x upscale for OCR only
    b = spans[0].bbox
    assert (b.x, b.y, b.w, b.h) == pytest.approx((100, 150, 300, 30))
    assert ocr.last_stats["ocr_scale"] == pytest.approx(2.0)


def test_cache_skips_the_model_on_a_rerun(monkeypatch, tmp_path, fresh_caches):
    created = []
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddle_module(
        lambda img: [("NET QUANTITY 1 PIECE", 0.97, _poly(10, 10, 300, 30))], created))
    ocr = P.PaddleOCRBackend(cache_dir=tmp_path / "cache", dump_dir=None)
    img = _canvas()
    a = ocr.recognise(img)
    calls = created[0].predict_calls        # a sparse read also re-reads with contrast
    b = ocr.recognise(img)
    assert created[0].predict_calls == calls
    assert ocr.last_stats["cache_hit"] is True
    assert [s.text for s in a] == [s.text for s in b]


def test_weak_line_is_re_read_and_better_read_wins(monkeypatch, fresh_caches):
    created = []
    rec = FakeRecogniser([("Net wt. 6 g", 0.93)])
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddle_module(
        lambda img: [("Net wt. 69", 0.55, _poly(100, 170, 300, 40))], created, rec))
    ocr = P.PaddleOCRBackend(config=P.PaddleConfig(contrast_reread_below=0, zoom_below_area=0),
                             cache_dir=None, dump_dir=None)
    spans = ocr.recognise(_canvas())
    assert spans[0].text == "Net wt. 6 g"
    assert spans[0].source_engine == "paddleocr(refined)"
    assert ocr.last_stats["refined"] == 1


def test_re_read_needs_a_real_margin(monkeypatch, fresh_caches):
    created = []
    rec = FakeRecogniser([("Net wt. 6 q", 0.57)])
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddle_module(
        lambda img: [("Net wt. 69", 0.55, _poly(100, 170, 300, 40))], created, rec))
    ocr = P.PaddleOCRBackend(cache_dir=None, dump_dir=None)
    assert ocr.recognise(_canvas())[0].text == "Net wt. 69"


def test_devanagari_re_read_replaces_only_devanagari_lines(monkeypatch, fresh_caches):
    created = []
    rec = FakeRecogniser([("अधिकतम खुदरा मूल्य", 0.90)])
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddle_module(
        lambda img: [("aDhkt Kudra", 0.95, _poly(100, 170, 300, 40))], created, rec))
    cfg = P.PaddleConfig(secondary_lang="hi", refine_below_score=0.0)
    ocr = P.PaddleOCRBackend(config=cfg, cache_dir=None, dump_dir=None)
    sp = ocr.recognise(_canvas())[0]
    assert sp.text == "अधिकतम खुदरा मूल्य" and sp.language == "hi"


def test_prefer_devanagari_rules():
    assert P.prefer_devanagari("MRP", 0.95, "एमआरपी", 0.90)
    assert not P.prefer_devanagari("MRP Rs. 10", 0.95, "MRP Rs. 10", 0.99)   # no script
    assert not P.prefer_devanagari("MRP", 0.95, "एम", 0.20)                  # too weak


def test_vertical_flag():
    assert P.is_vertical(_poly(0, 0, 20, 160), "MAR/26-NOV/26")
    assert not P.is_vertical(_poly(0, 0, 160, 20), "MAR/26-NOV/26")
    assert not P.is_vertical(_poly(0, 0, 10, 40), "|")


def test_dump_and_replay_round_trip(monkeypatch, tmp_path, fresh_caches):
    created = []
    monkeypatch.setitem(sys.modules, "paddleocr", fake_paddle_module(
        lambda img: [("MRP Rs. 120.00", 0.95, _poly(200, 300, 600, 60))], created))
    ocr = P.PaddleOCRBackend(cache_dir=None, dump_dir=tmp_path)
    img = _canvas()
    live = ocr.recognise(img, source=str(tmp_path / "back_9.jpeg"))
    dump = tmp_path / "back_9.paddle.json"
    assert dump.exists()
    replay = P.ReplayOCR(dump).recognise(img)
    assert [(s.text, s.bbox.to_dict()) for s in replay] == \
           [(s.text, s.bbox.to_dict()) for s in live]


def test_no_silent_fallback_when_paddle_missing(monkeypatch):
    monkeypatch.setattr(P.PaddleOCRBackend, "available", lambda self: False)
    with pytest.raises(OCRUnavailable):
        build_default_ocr(backend="paddle")


def test_tesseract_is_gone():
    from src.vision.ocr import backends

    assert not hasattr(backends, "TesseractOCR")
    src = "".join(p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py"))
    assert "pytesseract" not in src
    assert "pytesseract" not in (ROOT / "requirements.txt").read_text(encoding="utf-8")


# =====================================================================
# Extraction on real Paddle line layouts
# =====================================================================

def _spans(rows, engine="paddleocr"):
    return [TextSpan(text=t, bbox=BBox(x, y0, len(t) * 11, y1 - y0),
                     confidence=0.95, source_engine=engine)
            for t, x, y0, y1 in rows]


CASIO = [
    ("IMPORTED BY:-CASIO INDIA CO., PVT. LTD.", 148, 118, 142),
    ("A-41,1ST FLOORMCIE,MATHURA ROAD, NEWDELHI-110044", 148, 143, 166),
    ("IN CASE OF CONSUMER COMPLAINTS CONTACT :", 150, 169, 192),
    ("CONSUMER CARE OFFICER AT ABOVE ADDRESS.", 150, 194, 217),
    ("TEL. NO: +918447114400", 150, 221, 245),
    ("EMAIL:casiocare@casio.co.in", 147, 245, 276),
    ("D375", 465, 236, 270),
    ("COUNTRY OF ORIGIN: CHINA", 146, 275, 302),
    ("COMMODITY: WRIST WATCH", 142, 306, 330),
    ("MODEL:W-218HM-5BVDF", 143, 331, 354),
    ("MANUFACTURED ON: 03/2026", 141, 352, 379),
    ("NET QUANTITY", 144, 376, 401),
    ("1 PIECE", 483, 377, 403),
    ("MRP: Rs. 1495.00", 145, 401, 425),
    ("INCL. OF ALL TAXES", 327, 401, 428),
]


def _by_field(decls):
    return {d.field_id: d for d in decls}


def test_real_casio_layout_label_and_value_rejoined():
    """'NET QUANTITY' and '1 PIECE' come back as separate Paddle lines,
    far apart on the same row. They are one declaration."""
    d = _by_field(RegexExtractor().extract(_spans(CASIO)))
    assert d["net_quantity"].present and d["net_quantity"].value == 1.0
    assert d["retail_sale_price"].value == 1495.0
    assert "incl" in d["retail_sale_price"].raw_text.lower()
    assert d["manufacture_date"].value == "03/2026"
    assert d["country_of_origin"].value.upper().startswith("CHINA")
    cc = d["consumer_care"].raw_text
    assert "918447114400" in cc and "casiocare@casio.co.in" in cc


def _maggi():
    def S(t, x, y, h=22, w=None, v=False):
        return TextSpan(text=t, bbox=BBox(x, y, w or len(t) * 10, h),
                        confidence=0.9, source_engine="paddleocr", vertical=v)
    return [
        S("Net wt. 69", 40, 80),
        S("MRP 10.00 (Incl.of all taxes)", 40, 110),
        S("(0.83 per g)", 40, 138),
        S("Mfd. by: Nestle India Limited, 100/101 World Trade Centre,", 40, 170),
        S("Barakhamba Lane, New Delhi 110001", 40, 196),
        S("For queries/feedback/complaints contact our Consumer Services", 40, 228),
        S("Manager at Nestle India Ltd., at the above address", 40, 254),
        S("Toll free: 1800-103-1947  E-mail: wecare@in.nestle.com", 40, 280),
        S("For Mfd/Pkd and Best before date see below", 40, 310),
        S("MAR/26-NOV/26-D", 700, 60, h=160, w=24, v=True),
    ]


def test_maggi_sachet_fields():
    d = _by_field(RegexExtractor().extract(_maggi()))
    nq = d["net_quantity"]
    assert nq.value == 6.0 and nq.unit == "g"
    assert nq.notes, "a unit inferred from a '9' glyph must say so on the report"
    assert d["unit_sale_price"].value == pytest.approx(0.83)      # no currency printed
    assert d["manufacture_date"].value == "03/2026"               # coded pair, earlier date
    assert d["manufacture_date"].notes
    cc = d["consumer_care"].raw_text
    assert "1800-103-1947" in cc and "wecare@in.nestle.com" in cc


def test_vertical_date_code_not_spliced_into_a_row():
    d = _by_field(RegexExtractor().extract(_maggi()))
    assert "MAR/26" not in d["net_quantity"].raw_text
    assert "MAR/26" not in d["retail_sale_price"].raw_text


def test_hindi_qualifier_and_digits():
    spans = [TextSpan(text="अधिकतम खुदरा मूल्य ₹ १०.०० (सभी करों सहित)",
                      bbox=BBox(10, 10, 400, 30), confidence=0.9)]
    d = _by_field(RegexExtractor().extract(spans))
    rsp = d["retail_sale_price"]
    assert rsp.present and rsp.value == 10.0
    assert _has_tax_qualifier(rsp.raw_text, [])


# =====================================================================
# Rules: things OCR loses must not become violations
# =====================================================================

def _scan_with_price(raw: str) -> ScanResult:
    scan = ScanResult()
    scan.declarations.append(Declaration(
        field_id="retail_sale_price", raw_text=raw, value=10.0,
        bbox=BBox(0, 0, 100, 20), present=True, extraction_confidence=0.9))
    return scan


def _findings(scan, suffix):
    RulesEngine(RuleConfig()).evaluate(scan)
    return [f for f in scan.findings if f.rule_id.endswith(suffix)]


def test_dropped_rupee_sign_is_not_a_violation():
    f = _findings(_scan_with_price("MRP 10.00 (Incl. of all taxes)"), "currency_present")[0]
    assert f.outcome == Outcome.INDETERMINATE


def test_rupee_sign_read_is_compliant():
    f = _findings(_scan_with_price("MRP ₹10.00 (Incl. of all taxes)"), "currency_present")[0]
    assert f.outcome == Outcome.COMPLIANT


@pytest.mark.parametrize("raw", [
    "MRP ₹10.00 (Incl.of all taxes)",
    "MRP ₹10.00 INCL OFALLTAXES",
    "MRP ₹10.00 (inclusive of all taxes)",
    "MRP ₹10.00 (सभी करों सहित)",
])
def test_tax_qualifier_survives_ocr_spacing(raw):
    f = _findings(_scan_with_price(raw), "required_phrasing")[0]
    assert f.outcome == Outcome.COMPLIANT


def test_missing_qualifier_is_still_a_violation():
    f = _findings(_scan_with_price("MRP ₹10.00"), "required_phrasing")[0]
    assert f.outcome == Outcome.VIOLATION


def test_presence_row_does_not_contradict_content_checks():
    f = _findings(_scan_with_price("MRP ₹10.00"), "retail_sale_price.presence")[0]
    assert f.outcome == Outcome.COMPLIANT
    assert "inclusive of all taxes" not in f.message.lower()
    assert "located" in f.message


# =====================================================================
# Geometry that Paddle's padded line boxes broke
# =====================================================================

def test_ink_extent_shrinks_a_padded_box_and_ignores_neighbour_bleed():
    img = np.full((200, 600), 255, np.uint8)
    cv2.putText(img, "NET QTY 500 g", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    cv2.putText(img, "BEST VALUE", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    ys, xs = np.where(img[:100] < 128)
    ink_bottom = ys.max()
    padded = BBox(10, 40, 400, 72)            # reaches into the next line
    tight = ink_extent(img, padded)
    assert tight is not None
    assert tight.y2 <= ink_bottom + 2
    assert tight.y >= 40 and tight.y2 <= 112


def test_gm_mark_found_inside_a_line():
    spans = [TextSpan(text="GM  NET WT 500 g", bbox=BBox(0, 5, 200, 20)),
             TextSpan(text="GMP certified", bbox=BBox(0, 400, 200, 20))]
    present, at_top = find_top_token(spans, "GM", BBox(0, 0, 300, 500))
    assert present and at_top
    present, _ = find_top_token(spans[1:], "GM", BBox(0, 0, 300, 500))
    assert not present


def test_brackets_are_not_the_numeral_height(tmp_path):
    from src.synth.generator import LabelRenderer, LabelSpec, apply_violation
    from src.vision.glyph import measure_glyphs

    r = LabelRenderer(dpi=300, seed=3)
    p = tmp_path / "bp.png"
    gt = r.render(apply_violation(LabelSpec(), "banned_phrase"), out_path=str(p),
                  violations=["banned_phrase"])
    f = next(f for f in gt.fields if f.field_id == "net_quantity")
    img = cv2.imread(str(p))
    pad = 0.15 * f.h                                   # Paddle-style padding
    g = measure_glyphs(img, BBox(f.x - pad, f.y - pad, f.w + 2 * pad, f.h + 2 * pad),
                       text_hint=f.text)
    assert g.cap_height_px == pytest.approx(f.cap_height_px, abs=2.0)
    assert g.width_over_height > 0.5


# =====================================================================
# scan_photo.py --replay, end to end, no model
# =====================================================================

def test_scan_photo_replay_end_to_end(tmp_path):
    from src.synth.generator import LabelRenderer, LabelSpec

    photo = tmp_path / "label.png"
    LabelRenderer(dpi=300, seed=5).render(LabelSpec(), out_path=str(photo))
    lines = json.loads(Path(f"{photo}.lines.json").read_text(encoding="utf-8"))["lines"]
    img = cv2.imread(str(photo))
    dump = tmp_path / "label.paddle.json"
    dump.write_text(json.dumps({
        "source": str(photo),
        "ocr_input_shape": list(img.shape),
        "regions": [{"text": ln["text"], "score": 0.97, "poly": ln["poly"],
                     "engine": "paddleocr"} for ln in lines],
    }))
    out = tmp_path / "label.html"
    env = dict(os.environ, SIH_OCR_BACKEND="paddle")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_photo.py"), str(photo),
         "--replay", str(dump), "--out", str(out)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "What was read" in html and "Raw OCR lines" in html
    assert "net_quantity" in proc.stdout


def test_single_piece_pack_needs_no_separate_unit_price():
    """Rule 6(11) second proviso: per-unit price of a 1-piece pack IS the MRP."""
    scan = ScanResult()
    scan.declarations = RegexExtractor().extract(_spans(CASIO))
    RulesEngine(RuleConfig()).evaluate(scan)
    usp = [f for f in scan.findings if f.field_id == "unit_sale_price"]
    assert usp and all(f.outcome == Outcome.NOT_APPLICABLE for f in usp)


def test_address_block_stops_at_the_next_declaration():
    spans = _spans([
        ("Manufactured by: Sparkle Consumer Products Pvt Ltd, Plot 14,", 40, 100, 122),
        ("MIDC Industrial Area, Pune 411018", 40, 124, 146),
        ("Consumer Care: care@sparkle.example 1800-200-3000", 40, 150, 172),
    ])
    d = _by_field(RegexExtractor().extract(spans))
    assert "Consumer Care" not in d["manufacturer_details"].raw_text
    assert "411018" in d["manufacturer_details"].raw_text
    assert "1800-200-3000" in d["consumer_care"].raw_text


# =====================================================================
# API: what the inspector app consumes
# =====================================================================

def test_api_scan_returns_app_view_with_evidence_urls(tmp_path, monkeypatch):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api
    from src.synth.generator import LabelRenderer, LabelSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(api, "_pipeline", None)
    client = TestClient(api.app)
    photo = tmp_path / "a.png"
    LabelRenderer(dpi=300, seed=11).render(LabelSpec(), out_path=str(photo))
    r = client.post("/scan?product_name=Sparkle", headers={"Authorization": "Bearer demo-inspector"},
                    files={"file": ("a.png", photo.read_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    app_view = r.json()["app"]
    assert app_view["verdict"] in ("compliant", "non_compliant", "undecided")
    assert set(app_view["counts"]["violations"]) == {"CRITICAL", "MAJOR", "MINOR"}
    assert app_view["product"]["name"] == "Sparkle"
    urls = [f["evidence_url"] for f in app_view["findings"] if f["evidence_url"]]
    assert urls, "the app needs at least one evidence crop to show"
    img = client.get(urls[0], headers={"Authorization": "Bearer demo-inspector"})
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    # No path tricks through the evidence endpoint.
    bad = client.get(f"/scans/{app_view['scan_id']}/evidence/..%2F..%2Ft.db",
                     headers={"Authorization": "Bearer demo-inspector"})
    assert bad.status_code == 404
    again = client.get(f"/scans/{app_view['scan_id']}/summary",
                       headers={"Authorization": "Bearer demo-inspector"})
    assert again.status_code == 200 and again.json()["scan_id"] == app_view["scan_id"]


def test_api_allows_browser_clients():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from src.api import main as api

    # Same-origin by default (audit, 30 Sept: "*" let any web page call the
    # API). A web build on another origin is allowed with SIH_CORS_ORIGINS.
    r = TestClient(api.app).options(
        "/health", headers={"Origin": "https://evil.example",
                            "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") is None
    import inspect
    assert "SIH_CORS_ORIGINS" in inspect.getsource(api)
