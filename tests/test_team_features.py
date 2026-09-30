"""
Features built from what the team was trying to do:
  - fuzzy label anchors from rules/lexicon.yaml
  - a millimetre scale from a ruler-measured panel (instead of a pasted marker)
  - Gemini as a second reader, with provenance and an OCR cross-check
  - rotated re-read of vertical inkjet codes, CJK noise removal
plus the rule changes that stop OCR losses turning into violations.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.rules_engine import RuleConfig, RulesEngine  # noqa: E402
from src.core.schema import BBox, Declaration, Outcome, ScanResult, TextSpan  # noqa: E402
from src.extraction.fields import RegexExtractor  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "real"


def _by(decls):
    return {d.field_id: d for d in decls}


def _span(t, x, y, w=None, h=22, conf=0.95, v=False):
    return TextSpan(text=t, bbox=BBox(x, y, w or len(t) * 10, h), confidence=conf,
                    source_engine="paddleocr", vertical=v)


# ---------------------------------------------------------------- lexicon

def test_damaged_net_quantity_label_found_by_lexicon():
    d = _by(RegexExtractor().extract([_span("TINET QUNIIITY:", 40, 100), _span("45 ml", 260, 101)]))
    nq = d["net_quantity"]
    assert nq.present and nq.value == 45.0 and nq.unit == "ml"


def test_product_title_is_not_a_manufacturer_label():
    """'Imported Olive Oil' scored 90 against 'imported by' on characters alone."""
    d = _by(RegexExtractor().extract([_span("Imported Olive Oil 1 l", 40, 100)]))
    assert not d["manufacturer_details"].present


def test_toll_free_misread_still_anchors_consumer_care():
    d = _by(RegexExtractor().extract([_span("Tol Free: 1800-103-1644", 40, 100)]))
    assert d["consumer_care"].present and "1800-103-1644" in d["consumer_care"].raw_text


def test_block_heading_with_value_on_next_line():
    spans = [_span("Regd. Office & Consumer Cell", 40, 100),
             _span("8/3 Asaf Ali Road, New Delhi-110002", 40, 124),
             _span("E-mail: daburcares@dabur.com", 40, 148)]
    cc = _by(RegexExtractor().extract(spans))["consumer_care"]
    assert cc.present and "daburcares@dabur.com" in cc.raw_text


def test_columns_are_not_spliced_into_one_line():
    """Three tight columns on a real Maggi back panel read as separate lines."""
    spans = [_span("Green cardamom powder, Nutmeg), Red chilli", 500, 241, w=270),
             _span("Mfg. by: Nestle India Ltd., 100/101 World Trade", 772, 234, w=152),
             _span("Centre, New Delhi-110001", 772, 256, w=152),
             _span("NESTLE CONSUMER CARE", 980, 206, w=163),
             _span("WECARE@IN.NESTLE.COM", 980, 230, w=164)]
    d = _by(RegexExtractor().extract(spans))
    assert "cardamom" not in d["manufacturer_details"].raw_text
    assert "WECARE@IN.NESTLE.COM" in d["consumer_care"].raw_text


# ------------------------------------------------------------------ dates

def test_stacked_unlabelled_dates_earlier_is_manufacture():
    spans = [_span("03/2026 (B4)", 40, 100), _span("03/2029", 40, 122),
             _span("For MRP, Batch No., Mfd. & Use before, See above.", 40, 170)]
    md = _by(RegexExtractor().extract(spans))["manufacture_date"]
    assert md.present and md.value == "03/2026" and md.notes


def test_decimals_are_never_dates():
    spans = [_span("of Total Sugars (of which 12.6g is Added Sugars), 9.1g", 40, 100),
             _span("Lot No.-MFD.-USE BY-MFG. BY see coding area", 40, 130)]
    assert not _by(RegexExtractor().extract(spans))["manufacture_date"].present


def test_day_month_year_date():
    md = _by(RegexExtractor().extract([_span("MFG. DATE: 19/06/26", 40, 100)]))["manufacture_date"]
    assert md.value == "06/2026"


def test_future_check_uses_the_manufacture_date_not_the_use_by():
    scan = ScanResult()
    scan.declarations.append(Declaration(
        field_id="manufacture_date", raw_text="03/2026 (B4) / 03/2029", value="03/2026",
        bbox=BBox(0, 0, 50, 20), present=True, extraction_confidence=0.6))
    RulesEngine(RuleConfig()).evaluate(scan)
    f = [f for f in scan.findings if f.rule_id == "manufacture_date.date_plausible"][0]
    assert f.outcome == Outcome.COMPLIANT


# -------------------------------------------------------- rules and OCR

def _price_scan(raw, conf, extra_lines=()):
    scan = ScanResult()
    sp = TextSpan(text=raw, bbox=BBox(0, 0, 100, 20), confidence=conf)
    scan.spans = [sp] + [TextSpan(text=t, bbox=BBox(0, 30 + 20 * i, 100, 18), confidence=conf)
                         for i, t in enumerate(extra_lines)]
    scan.declarations.append(Declaration(
        field_id="retail_sale_price", raw_text=raw, value=18.0, bbox=sp.bbox,
        spans=[sp], present=True, extraction_confidence=0.92 * conf))
    RulesEngine(RuleConfig()).evaluate(scan)
    return [f for f in scan.findings if f.rule_id.endswith("required_phrasing")][0]


def test_garbled_qualifier_on_mrp_line_is_indeterminate():
    f = _price_scan("Rs.18.00", 0.99, ["For MRP Rs. (nd of al aes), Batch No,"])
    assert f.outcome == Outcome.INDETERMINATE and "nd of al aes" in f.message


def test_clearly_read_price_without_qualifier_is_still_a_violation():
    f = _price_scan("MRP Rs. 18.00", 0.99)
    assert f.outcome == Outcome.VIOLATION


def test_poorly_read_price_without_qualifier_is_not_asserted():
    f = _price_scan("MRP Rs. 18.00", 0.6)
    assert f.outcome == Outcome.INDETERMINATE


def test_three_ls_in_all():
    f = _price_scan("MRP5 (incl. of alll taxes)", 0.95)
    assert f.outcome == Outcome.COMPLIANT


def test_tiny_text_is_not_measured_into_a_violation():
    from src.core.rules_engine import _glyph_measurement_plausible
    from src.core.schema import GlyphMetrics

    g = GlyphMetrics(cap_height_px=11, mean_width_px=6, width_over_height=0.2,
                     n_glyphs_measured=6)
    assert not _glyph_measurement_plausible(g)


# ------------------------------------------------- scale from ruler size

def test_panel_size_gives_a_millimetre_scale_without_a_marker(tmp_path):
    os.environ["SIH_OCR_BACKEND"] = "synthetic"
    from src.core.pipeline import CompliancePipeline
    from src.synth.generator import LabelRenderer, LabelSpec
    from src.vision.calibration import detect_markers

    spec = LabelSpec()
    p = tmp_path / "l.png"
    gt = LabelRenderer(dpi=300, seed=4).render(spec, out_path=str(p))
    img = cv2.imread(str(p))
    x, y, w, h = cv2.boundingRect(np.int32(detect_markers(img)[0].corners))
    img[y - 20:y + h + 20, x - 20:x + w + 20] = 255          # no marker in frame
    p2 = tmp_path / "nomarker.png"
    cv2.imwrite(str(p2), img)
    man = json.loads(Path(f"{p}.lines.json").read_text(encoding="utf-8"))
    man["marker_corners"] = None
    Path(f"{p2}.lines.json").write_text(json.dumps(man))

    pl = CompliancePipeline()
    assert not pl.scan(str(p2)).calibration.available
    res = pl.scan(str(p2), panel_size_mm=(spec.panel_width_mm, spec.panel_height_mm))
    assert res.calibration.method.value == "user_supplied_dimension"
    nq = res.declaration("net_quantity")
    truth = next(f for f in gt.fields if f.field_id == "net_quantity").cap_height_mm
    assert nq.glyph.cap_height_mm == pytest.approx(truth, abs=0.15)
    assert res.context.pdp_area_cm2 == pytest.approx(
        spec.panel_width_mm * spec.panel_height_mm / 100)


def test_panel_size_that_does_not_fit_the_photo_is_refused():
    from src.core.pipeline import _user_scale
    from src.core.schema import PackageContext

    ctx = PackageContext()
    ctx.pdp_bbox = BBox(0, 0, 1000, 1000)            # square panel found
    ctx.pdp_confidence, ctx.pdp_detection_method = 1.0, "contour"
    cal, _ = _user_scale(ctx, (120.0, 40.0), None)     # 3:1 given
    assert not cal.available and "proportions" in cal.notes


# ------------------------------------------------------------- Gemini

VLM_FIX = Path(__file__).resolve().parent / "fixtures" / "vlm"


class _FakeGemini:
    def __init__(self, items):
        self.items = items

    def generate(self, prompt, image):
        return "```json\n" + json.dumps(self.items) + "\n```"


def _box(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    return [int(y0 / H * 1000), int(x0 / W * 1000), int(y1 / H * 1000), int(x1 / W * 1000)]


def test_gemini_only_read_never_decides_a_verdict():
    from src.core.pipeline import CompliancePipeline
    from src.vision.ocr.paddle import ReplayOCR
    from src.vision.ocr.vlm_gemini import GeminiReader

    img = cv2.imread(str(FIX / "photo5.jpg"))
    fake = _FakeGemini([{"field": "manufacture_date", "text": "MAR/26-NOV/26-D",
                         "box_2d": _box(img, 1160, 250, 1245, 640)}])
    # A real read of this sachet in which Paddle missed the inkjet date
    # (later reads get it, which is exactly why this one is kept).
    res = CompliancePipeline(ocr=ReplayOCR(VLM_FIX / "photo5_date_missed.paddle.json"),
                             vlm=GeminiReader(client=fake)).scan(str(FIX / "photo5.jpg"))
    md = res.declaration("manufacture_date")
    assert md.present and md.value == "03/2026" and md.source == "vlm"
    outs = {f.outcome for f in res.findings if f.field_id == "manufacture_date"}
    assert outs <= {Outcome.INDETERMINATE, Outcome.NOT_APPLICABLE, Outcome.UNVERIFIED_RULE}


def test_gemini_read_confirmed_by_ocr_counts():
    from src.core.pipeline import CompliancePipeline
    from src.vision.ocr.paddle import ReplayOCR
    from src.vision.ocr.vlm_gemini import GeminiReader

    img = cv2.imread(str(FIX / "photo5.jpg"))
    fake = _FakeGemini([{"field": "unit_sale_price", "text": "(₹0.83 per g)",
                         "box_2d": _box(img, 775, 435, 965, 475)}])
    res = CompliancePipeline(ocr=ReplayOCR(VLM_FIX / "photo5_date_missed.paddle.json"),
                             vlm=GeminiReader(client=fake)).scan(str(FIX / "photo5.jpg"))
    usp = res.declaration("unit_sale_price")
    assert usp.source == "vlm+ocr" and usp.value == pytest.approx(0.83)


def test_missing_gemini_key_is_a_clear_error(monkeypatch):
    from src.vision.ocr.vlm_gemini import GeminiReader, VLMUnavailable

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(VLMUnavailable, match="GEMINI_API_KEY"):
        GeminiReader().check()


# ------------------------------------------------------- Paddle additions

def _fake_mod(pipe_predict):
    import types

    mod = types.ModuleType("paddleocr")
    mod.__version__ = "fake"

    class PaddleOCR:
        def __init__(self, **kw):
            pass

        def predict(self, img, **kw):
            return pipe_predict(img)

    class TextRecognition:
        def __init__(self, **kw):
            pass

        def predict(self, crops):
            return [{"rec_text": "", "rec_score": 0.0} for _ in crops]

    mod.PaddleOCR, mod.TextRecognition = PaddleOCR, TextRecognition
    return mod


def _res(texts_scores_polys):
    return [{"rec_texts": [t for t, _, _ in texts_scores_polys],
             "rec_scores": [s for _, s, _ in texts_scores_polys],
             "rec_polys": [np.float32(p) for _, _, p in texts_scores_polys]}]


def _poly(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_vertical_inkjet_band_is_re_read_rotated(monkeypatch):
    from src.vision.ocr import paddle as P

    P._PIPELINE_CACHE.clear()
    img = np.full((1300, 1400, 3), 255, np.uint8)       # >= 1200: no upscale

    def predict(im):
        if im.shape[0] == 1300:                           # the full photo: fragments
            return _res([("5", 0.9, _poly(1200, 300, 40, 60)),
                         ("寸D", 0.9, _poly(1200, 380, 40, 60)),
                         ("ON", 0.6, _poly(1200, 460, 40, 60)),
                         ("心", 0.6, _poly(1200, 540, 40, 60)),
                         ("MRP Rs. 10.00", 0.99, _poly(100, 100, 400, 40))])
        # the rotated band: one good line across the middle
        h, w = im.shape[:2]
        return _res([("MAR/26-NOV/26-D", 0.95, _poly(10, h / 2 - 15, w - 20, 30))])

    monkeypatch.setitem(sys.modules, "paddleocr", _fake_mod(predict))
    b = P.PaddleOCRBackend(cache_dir=None, dump_dir=None)
    spans = b.recognise(img)
    texts = [s.text for s in spans]
    assert "MAR/26-NOV/26-D" in texts and "心" not in texts
    band = next(s for s in spans if s.text == "MAR/26-NOV/26-D")
    assert band.vertical and 1180 <= band.bbox.x and band.bbox.x2 <= 1260
    assert band.bbox.y < 400 and band.bbox.y2 > 500      # spans the band, top to bottom
    assert b.last_stats["vertical_bands"] == 1
    P._PIPELINE_CACHE.clear()


def test_chinese_noise_is_dropped():
    from src.vision.ocr.paddle import PaddleOCRBackend

    r = {"text": "印", "score": 0.9, "poly": _poly(0, 0, 20, 20)}
    assert PaddleOCRBackend._to_span(r).text == ""
    # In a price line, a glyph right before the number is the rupee sign
    # misread (real Paper Boat inkjet: "29/08/26,天10:USP车 0.071/ml").
    r2 = {"text": "MRP 寸5", "score": 0.9, "poly": _poly(0, 0, 80, 20)}
    assert PaddleOCRBackend._to_span(r2).text == "MRP ₹5"
    # Anywhere else it is still noise.
    r3 = {"text": "BATCH 印123", "score": 0.9, "poly": _poly(0, 0, 80, 20)}
    assert PaddleOCRBackend._to_span(r3).text == "BATCH 123"


def test_re_read_that_loses_the_line_is_rejected(monkeypatch):
    from src.vision.ocr import paddle as P
    import types

    P._PIPELINE_CACHE.clear()
    P._RECOGNISER_CACHE.clear()

    def predict(im):
        return _res([("Aniseed, Black pepper, Fenugreek, Ginger, Clove,", 0.75,
                      _poly(100, 170, 500, 30))])

    mod = _fake_mod(predict)

    class Rec:
        def __init__(self, **kw):
            pass

        def predict(self, crops):
            return [{"rec_text": "A", "rec_score": 0.95} for _ in crops]

    mod.TextRecognition = Rec
    monkeypatch.setitem(sys.modules, "paddleocr", mod)
    b = P.PaddleOCRBackend(cache_dir=None, dump_dir=None)
    out = b.recognise(np.full((1300, 1400, 3), 255, np.uint8))
    assert out[0].text.startswith("Aniseed")
    P._PIPELINE_CACHE.clear()
    P._RECOGNISER_CACHE.clear()


# ---------------------------------------------------------- scripts

def test_scan_real_replay_runs_and_writes_a_report(tmp_path):
    out = tmp_path / "r.html"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_real.py"), str(FIX / "photo2.jpg"),
         "--replay", str(FIX / "photo2.paddle.json"), "--no-vlm", "--out", str(out)],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
        env=dict(os.environ, SIH_OCR_BACKEND="paddle"))
    assert proc.returncode == 0, proc.stderr
    assert "VERDICT" in proc.stdout and out.exists()
    assert "PASS (Compliant)" not in proc.stdout
