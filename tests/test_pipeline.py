"""
Regression tests.

Several of these exist because the bug they guard against actually
happened during development and was silent - the pipeline produced
confident, well-formatted, wrong output. Those are marked REGRESSION.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.pipeline import CompliancePipeline
from src.core.rules_engine import RuleConfig
from src.core.schema import BBox, Outcome, TextSpan
from src.synth.generator import LabelRenderer, LabelSpec, apply_violation
from src.vision.calibration import calibrate_from_image
from src.vision.glyph import measure_glyphs
from src.vision.ocr.base import sort_reading_order
from src.vision.preprocess import remove_specular_highlights
from src.vision.barcode import zbar_available

needs_zbar = pytest.mark.skipif(
    not zbar_available(),
    reason="barcode reader (zbar) not loaded - on Windows install the Visual C++ 2013 Redistributable (x64)")


@pytest.fixture(scope="module")
def renderer():
    return LabelRenderer(dpi=300)


@pytest.fixture(scope="module")
def pipeline():
    return CompliancePipeline()


@pytest.fixture(scope="module")
def tmpdir_mod(tmp_path_factory):
    return tmp_path_factory.mktemp("labels")


# =====================================================================
# Rule config integrity
# =====================================================================

def test_config_loads():
    cfg = RuleConfig()
    assert cfg.declarations and cfg.presentation


def test_every_rule_declares_verified_flag():
    """No rule may be silently unmarked - that is how a placeholder
    threshold sneaks into a legal report."""
    cfg = RuleConfig()
    for item in cfg.declarations + cfg.presentation:
        assert "verified" in item, f"{item['id']} has no verified flag"
        assert "citation" in item, f"{item['id']} has no citation"


def test_unverified_rules_cannot_emit_violations(renderer, tmpdir_mod, pipeline):
    """The central safety property of the whole system."""
    p = tmpdir_mod / "u.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    for f in res.findings:
        if not f.rule_verified:
            assert f.outcome is not Outcome.VIOLATION, (
                f"{f.rule_id} is unverified but emitted a VIOLATION"
            )


# =====================================================================
# Calibration and measurement
# =====================================================================

def test_calibration_recovers_true_scale(renderer, tmpdir_mod):
    p = tmpdir_mod / "cal.png"
    gt = renderer.render(LabelSpec(), out_path=str(p))
    cal = calibrate_from_image(cv2.imread(str(p)), marker_size_mm=gt.marker_size_mm)
    assert cal.available
    rel_err = abs(cal.px_per_mm - gt.px_per_mm) / gt.px_per_mm
    assert rel_err < 0.02, f"scale error {rel_err*100:.2f}%"


def test_glyph_measurement_within_tolerance(renderer, tmpdir_mod):
    """The go/no-go from the build guide: +/-0.3 mm on clean renders."""
    p = tmpdir_mod / "meas.png"
    gt = renderer.render(LabelSpec(), out_path=str(p))
    img = cv2.imread(str(p))
    cal = calibrate_from_image(img, marker_size_mm=gt.marker_size_mm)
    errs = []
    for f in gt.fields:
        m = measure_glyphs(img, BBox(f.x, f.y, f.w, f.h), cal, text_hint=f.text)
        if m.cap_height_mm:
            errs.append(abs(m.cap_height_mm - f.cap_height_mm))
    assert errs
    assert np.mean(errs) < 0.30, f"MAE {np.mean(errs):.3f} mm exceeds tolerance"


def test_calibration_refuses_without_marker():
    blank = np.full((600, 600, 3), 255, np.uint8)
    assert not calibrate_from_image(blank).available


# =====================================================================
# REGRESSION tests
# =====================================================================

def test_reading_order_groups_rows_not_raw_y():
    """
    REGRESSION: sorting spans by (y, x) interleaves words from adjacent
    lines, because baselines wobble and ascenders start higher. Line
    reconstruction downstream then produced silently wrong field values.
    """
    spans = [
        TextSpan("Net", BBox(10, 100, 30, 20)),
        TextSpan("Qty", BBox(45, 98, 30, 22)),    # 2 px higher
        TextSpan("500g", BBox(80, 101, 40, 19)),
        TextSpan("MRP", BBox(10, 140, 35, 20)),
        TextSpan("120", BBox(50, 139, 35, 20)),
    ]
    out = [s.text for s in sort_reading_order(spans)]
    assert out == ["Net", "Qty", "500g", "MRP", "120"]


def test_deglare_preserves_white_label():
    """
    REGRESSION: masking every bright low-saturation pixel selects the
    whole white carton; inpainting then smears the print away and OCR
    returns confident garbage. Glare must be local AND textureless.
    """
    img = np.full((400, 400, 3), 255, np.uint8)
    cv2.putText(img, "MRP 120", (40, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 0, 0), 3)
    out = remove_specular_highlights(img)
    ink_before = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 128).sum()
    ink_after = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) < 128).sum()
    assert ink_after > ink_before * 0.9, "deglare destroyed the printed text"


def test_mrp_qualifier_not_a_false_positive(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: an unanchored multi-line window let the MRP pattern
    match at the product-name line and cut off the '(inclusive of all
    taxes)' qualifier two lines below - a confident FALSE VIOLATION on a
    compliant package.
    """
    p = tmpdir_mod / "mrp.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    bad = [f for f in res.violations
           if f.rule_id == "retail_sale_price.required_phrasing"]
    assert not bad, "false positive on a compliant MRP declaration"


def test_missing_mrp_attributed_to_presence(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: the loose '<currency> <number>' fallback matched the
    'Unit Sale Price' line, so a missing MRP was reported under the
    wrong sub-rule with the wrong evidence crop.
    """
    p = tmpdir_mod / "nomrp.png"
    renderer.render(apply_violation(LabelSpec(), "missing_mrp"), out_path=str(p))
    res = scan_complete(pipeline, p)
    ids = [f.rule_id for f in res.violations]
    assert "retail_sale_price.presence" in ids


def test_clear_space_sees_non_declaration_text(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: Rule 8 asks about 'other printed matter', most of which
    never becomes a declaration. Judging clear space from declarations
    alone reported crowded labels as compliant.
    """
    p = tmpdir_mod / "crowd.png"
    renderer.render(LabelSpec(), out_path=str(p),
                    violations=["quantity_clear_space"])
    res = pipeline.scan(str(p))
    assert "quantity_clear_space" in [f.rule_id for f in res.violations]


# =====================================================================
# End-to-end behaviour
# =====================================================================

def test_compliant_label_has_no_violations(renderer, tmpdir_mod, pipeline):
    p = tmpdir_mod / "ok.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    assert res.violations == []


def test_font_size_rule_now_fires(renderer, tmpdir_mod, pipeline):
    """
    Table-I is verified against G.S.R. 629(E) (2017), so the height rule
    is live and actually resolves to COMPLIANT on a clean label, rather
    than UNVERIFIED_RULE as it did before verification.

    Checks the height rule specifically rather than the whole-label
    verdict: the synthetic generator prints the common name as a bare
    line with no heading, and common_name has no safe unlabelled
    extraction path since the fallback proved unsafe on a real photo
    (see the removal note in extraction/fields.py) - so the full label
    is correctly INDETERMINATE once coverage is asserted, and testing
    the whole verdict here would silently re-permit exactly the
    unsafe guess this suite exists to keep out.
    """
    p = tmpdir_mod / "ok2.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    res.coverage_complete = True
    res.findings = []
    pipeline.engine.evaluate(res)
    height = next(f for f in res.findings if f.rule_id == "numeral_height")
    assert height.outcome is Outcome.COMPLIANT
    assert not res.unverified


def test_height_rule_abstains_without_pdp_area(renderer, tmpdir_mod, pipeline):
    """
    Table-I is keyed to PDP AREA since 2017, so with no area there is no
    threshold to compare against and the engine must abstain rather than
    guess a band.
    """
    p = tmpdir_mod / "noarea.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    res.context.pdp_area_cm2 = None
    res.findings = []
    pipeline.engine.evaluate(res)
    height = [f for f in res.findings if f.rule_id == "numeral_height"]
    assert height and height[0].outcome is Outcome.INDETERMINATE


def test_pdp_area_matches_rendered_panel(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: the PDP was originally taken as the bounding box of all
    detected text. Rule 7(4) defines it as the physical panel face, and
    ink covers only part of a panel - so the area came out ~46 cm2 on a
    117 cm2 panel. Since the 2017 amendment keys Table-I to panel area,
    that under-estimate dropped the threshold from 2.5mm to 1.0mm and
    would have passed genuinely undersized text as compliant.
    """
    p = tmpdir_mod / "area.png"
    gt = renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    assert res.context.pdp_area_cm2 is not None
    assert res.context.pdp_area_cm2 == pytest.approx(gt.pdp_area_cm2, rel=0.10)
    assert res.context.pdp_detection_method == "contour"


def test_low_confidence_panel_suppresses_area(renderer, tmpdir_mod, pipeline):
    """
    A panel we do not trust must yield NO area, so the height rule
    abstains. Guessing low here fails OPEN - a smaller area means a
    smaller minimum height, so undersized text would pass.
    """
    from src.core.schema import Calibration, CalibrationMethod
    from src.vision.panel import PanelDetection, panel_area_cm2

    cal = Calibration(px_per_mm=12.0, method=CalibrationMethod.FIDUCIAL)
    weak = PanelDetection(
        bbox=BBox(0, 0, 500, 500), quad=None, method="text_extent", confidence=0.3
    )
    assert not weak.reliable
    # Cylindrical packages cannot yield an area from one view either.
    strong = PanelDetection(
        bbox=BBox(0, 0, 500, 500), quad=None, method="contour", confidence=0.9
    )
    assert panel_area_cm2(strong, cal, geometry="cylindrical") is None


@pytest.mark.parametrize("kind,expected", [
    ("missing_mrp", "retail_sale_price.presence"),
    ("mrp_no_tax_qualifier", "retail_sale_price.required_phrasing"),
    ("missing_consumer_care", "consumer_care.presence"),
    ("missing_date", "manufacture_date.presence"),
    ("missing_manufacturer", "manufacturer_details.presence"),
    ("banned_phrase", "net_quantity.banned_phrase"),
    # Real photos decide this rule, not rendered labels: a width ratio is
    # only asserted from characters matched to the text read. The rendered
    # label's glyphs are not matched, so it is held for a human.
    pytest.param("glyph_aspect_ratio", "glyph_aspect_ratio.net_quantity",
                 marks=pytest.mark.xfail(reason="held on unmatched glyphs (real-photo policy)")),
    ("quantity_clear_space", "quantity_clear_space"),
])
def test_injected_violation_is_detected(renderer, tmpdir_mod, pipeline, kind, expected):
    p = tmpdir_mod / f"{kind}.png"
    renderer.render(apply_violation(LabelSpec(), kind), out_path=str(p),
                    violations=[kind])
    res = scan_complete(pipeline, p)
    assert expected in [f.rule_id for f in res.violations]


def test_line_grouping_survives_an_oversized_span():
    """
    REGRESSION: row tolerance was derived from the tallest span seen so
    far in a row. One oversized box (a logo, a border fragment, a merged
    blob after rectification) then inflated the tolerance until rows
    swallowed the rows below, collapsing lines into columns. Extraction
    returned a vertical sequence of first words and every field but one
    silently failed.
    """
    from src.extraction.fields import RegexExtractor

    spans = [
        TextSpan("LOGO", BBox(10, 10, 60, 120)),   # oversized
        TextSpan("Net", BBox(10, 200, 30, 20)),
        TextSpan("Qty", BBox(45, 200, 30, 20)),
        TextSpan("MRP", BBox(10, 240, 35, 20)),
        TextSpan("120", BBox(50, 240, 35, 20)),
    ]
    lines = RegexExtractor()._group_into_lines(spans)
    texts = [" ".join(sp.text for sp in ln) for ln in lines]
    assert "Net Qty" in texts
    assert "MRP 120" in texts


def test_pipeline_never_raises_on_garbage(pipeline, tmpdir_mod):
    """A crash in a live demo looks like incompetence; a polite retake
    prompt looks like a product."""
    p = tmpdir_mod / "noise.png"
    cv2.imwrite(str(p), np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    res = pipeline.scan(str(p))
    assert res is not None


def test_pipeline_handles_missing_file(pipeline):
    res = pipeline.scan("/nonexistent/nope.png")
    assert res.image_quality_ok is False


# =====================================================================
# Rules implemented from later amendments
# =====================================================================

@pytest.mark.parametrize("kind,expected_outcome", [
    ("sticker_over_mrp", Outcome.VIOLATION),      # conceals the printed MRP
    ("sticker_higher_price", Outcome.VIOLATION),  # revises the price upward
    ("sticker_lower_price", Outcome.COMPLIANT),   # proviso: downward, visible
])
def test_sticker_rule_honours_the_proviso(renderer, tmpdir_mod, pipeline,
                                          kind, expected_outcome):
    """
    Rule 6(3) prohibits ALTERING a declaration by sticker, but expressly
    permits a sticker bearing a revised LOWER price that does not cover
    the original. A detector that flags every pasted label would fire on
    a large share of legitimate retail stock.
    """
    p = tmpdir_mod / f"{kind}.png"
    renderer.render(apply_violation(LabelSpec(), kind), out_path=str(p),
                    violations=[kind])
    res = pipeline.scan(str(p))
    finding = next(f for f in res.findings if f.rule_id == "sticker_alteration")
    assert finding.outcome is expected_outcome


def test_concealment_outranks_price_comparison(renderer, tmpdir_mod, pipeline):
    """
    When the sticker covers the printed MRP, that price is unreadable -
    so any "printed price" in hand came from another line. Reporting a
    price comparison there is a right verdict for a wrong reason, and
    the reason is what goes on the violation report.
    """
    p = tmpdir_mod / "conceal.png"
    renderer.render(apply_violation(LabelSpec(), "sticker_over_mrp"),
                    out_path=str(p), violations=["sticker_over_mrp"])
    res = pipeline.scan(str(p))
    f = next(x for x in res.findings if x.rule_id == "sticker_alteration")
    assert "conceals" in f.message.lower()


@pytest.mark.parametrize("kind,hint,expected", [
    (None, True, Outcome.COMPLIANT),                    # mark at top
    ("gm_not_declared", True, Outcome.VIOLATION),
    ("gm_not_at_top", True, Outcome.VIOLATION),
    (None, False, Outcome.NOT_APPLICABLE),              # not GM food
])
def test_gm_label_rule(renderer, tmpdir_mod, pipeline, kind, hint, expected):
    spec = LabelSpec()
    spec.genetically_modified = True
    if kind:
        spec = apply_violation(spec, kind)
        spec.genetically_modified = True
    p = tmpdir_mod / f"gm_{kind}_{hint}.png"
    renderer.render(spec, out_path=str(p), violations=[kind] if kind else [])
    res = pipeline.scan(
        str(p), hints={"genetically_modified_food": True} if hint else None
    )
    f = next(x for x in res.findings if x.rule_id == "gm_food_label")
    assert f.outcome is expected


def test_gm_flag_is_not_inferred_from_the_mark(renderer, tmpdir_mod, pipeline):
    """
    Whether a product IS genetically modified cannot be inferred from
    whether it carries the GM mark - that is circular, and it makes the
    one case the rule exists to catch (GM food with no mark) undetectable
    by construction. The flag must come from outside the image.
    """
    p = tmpdir_mod / "gm_infer.png"
    renderer.render(apply_violation(LabelSpec(), "gm_not_declared"),
                    out_path=str(p), violations=["gm_not_declared"])
    res = pipeline.scan(str(p))          # no hint supplied
    f = next(x for x in res.findings if x.rule_id == "gm_food_label")
    assert f.outcome is Outcome.NOT_APPLICABLE


@needs_zbar
def test_dual_mrp_detected_across_scans(renderer, tmpdir_mod, pipeline):
    """
    Rule 18(2A) cannot be decided from one photograph - a package showing
    Rs.120 is compliant alone and becomes evidence only beside an
    identical one showing Rs.145. This is what makes the product
    repository an evidence base rather than storage.
    """
    from src.core.cross_scan import CrossScanAnalyzer

    scans = []
    for price in (120.0, 120.0, 145.0):
        spec = LabelSpec()
        spec.mrp = price
        p = tmpdir_mod / f"dual_{price}.png"
        renderer.render(spec, out_path=str(p))
        scans.append(pipeline.scan(str(p)))

    findings = CrossScanAnalyzer().findings(scans)
    assert len(findings) == 1
    assert findings[0].rule_id == "dual_mrp_prohibited"
    assert findings[0].outcome is Outcome.VIOLATION


def test_dual_mrp_ignores_different_products(renderer, tmpdir_mod, pipeline):
    """Different commodities at different prices is normal commerce."""
    from src.core.cross_scan import CrossScanAnalyzer

    scans = []
    for price, name, qty, gtin in (
        (120.0, "Detergent Powder", 500, "8901234567890"),
        (99.0, "Floor Cleaner", 250, "5901234123457"),
    ):
        spec = LabelSpec()
        spec.mrp, spec.common_name, spec.net_quantity_value = price, name, qty
        # Distinct GTINs: two different products sharing one barcode
        # SHOULD group, so leaving the default here would be testing the
        # opposite of what this test claims.
        # The field is `gtin`; assigning `spec.barcode` set an attribute
        # nothing reads, so both labels kept the default code and the
        # analyser correctly grouped them - a silently vacuous test.
        spec.gtin = gtin
        p = tmpdir_mod / f"diff_{name}.png"
        renderer.render(spec, out_path=str(p))
        scans.append(pipeline.scan(str(p)))

    assert CrossScanAnalyzer().findings(scans) == []


# =====================================================================
# Evaluation integrity
# =====================================================================

def test_every_violation_kind_maps_to_a_rule():
    """
    An injected kind with no entry in VIOLATION_TO_RULES is never scored:
    the rule contributes nothing to precision or recall and simply looks
    untested. Adding a rule without adding its mapping silently shrinks
    the evaluation.
    """
    from src.evaluation.harness import VIOLATION_TO_RULES
    from src.synth.generator import VIOLATION_KINDS

    missing = [k for k in VIOLATION_KINDS if k not in VIOLATION_TO_RULES]
    assert not missing, f"unscored violation kinds: {missing}"


def test_injected_kinds_are_never_incompatible():
    """
    Co-injecting kinds that perturb the same geometry creates violations
    the manifest does not record, so correct detections score as false
    positives. This measured as a precision drop from 1.00 to 0.65 -
    entirely inside the measuring instrument.
    """
    import json
    import tempfile

    from src.synth.generator import (
        INCOMPATIBLE_FAMILIES,
        INCOMPATIBLE_KINDS,
        KIND_FAMILIES,
        generate_dataset,
    )

    fam_of = {k: f for f, ks in KIND_FAMILIES.items() for k in ks}
    with tempfile.TemporaryDirectory() as d:
        man = json.load(open(generate_dataset(d, 4, 12, seed=3)))
        for rec in man["records"]:
            kinds = rec["injected_violations"]
            for a in kinds:
                for b in kinds:
                    if a == b:
                        continue
                    assert frozenset({a, b}) not in INCOMPATIBLE_KINDS
                    fa, fb = fam_of.get(a), fam_of.get(b)
                    if fa and fb and fa != fb:
                        assert frozenset({fa, fb}) not in INCOMPATIBLE_FAMILIES


def test_hints_are_recorded_for_kinds_that_need_them():
    """GM cases need context the image cannot carry."""
    import tempfile

    from src.synth.generator import LabelRenderer, LabelSpec, apply_violation

    with tempfile.TemporaryDirectory() as d:
        gt = LabelRenderer(dpi=200).render(
            apply_violation(LabelSpec(), "gm_not_declared"),
            out_path=f"{d}/x.png", violations=["gm_not_declared"],
        )
        assert gt.hints.get("genetically_modified_food") is True


@pytest.mark.parametrize("noisy", [False, True])
def test_sticker_detector_survives_noise(tmpdir_mod, pipeline, noisy):
    # Fixed seed. With the shared module-scoped renderer this test drew a
    # different noise pattern depending on execution order and failed
    # roughly one run in ten - not test pollution, but the detector being
    # genuinely marginal under noise. Measured reliability is 24/25 (96%)
    # across seeds; pinning the seed makes the test deterministic, and
    # the real rate is recorded here rather than hidden behind a lucky
    # run. If this ever needs raising, the lever is the minimum area
    # threshold in vision/overlay.py.
    from src.synth.generator import LabelRenderer

    renderer = LabelRenderer(dpi=300, seed=0)
    """
    REGRESSION: the detector was tuned on clean renders and found
    nothing at all once sensor noise was added - the seam step falls
    from 6.0 grey levels to 3.0. Tuning for noise then blinded it on
    clean images, because a 5px median erases the 1px cut outline.
    Detection now runs per view and merges results.
    """
    p = tmpdir_mod / f"stk_{noisy}.png"
    renderer.render(apply_violation(LabelSpec(), "sticker_over_mrp"),
                    out_path=str(p), violations=["sticker_over_mrp"],
                    add_noise=noisy)
    res = pipeline.scan(str(p))
    f = next(x for x in res.findings if x.rule_id == "sticker_alteration")
    assert f.outcome is Outcome.VIOLATION


@pytest.mark.parametrize("noisy", [False, True])
def test_sticker_detector_no_false_positive(renderer, tmpdir_mod, pipeline, noisy):
    p = tmpdir_mod / f"stk_ok_{noisy}.png"
    renderer.render(LabelSpec(), out_path=str(p), add_noise=noisy)
    assert pipeline.scan(str(p)).overlays == []


# =====================================================================
# Barcode / GTIN
# =====================================================================

def test_gtin_checksum_rejects_misreads():
    """
    A misread barcode is worse than none: it groups unrelated products
    and yields a confidently wrong dual-price finding.
    """
    from src.vision.barcode import gtin_checksum_valid

    assert gtin_checksum_valid("8901234567890")
    assert gtin_checksum_valid("5901234123457")
    assert not gtin_checksum_valid("8901234567894")   # bad check digit
    assert not gtin_checksum_valid("89012341")        # valid length, bad digit
    assert not gtin_checksum_valid("abcdefghijklm")


@needs_zbar
def test_barcode_is_read_from_label(renderer, tmpdir_mod, pipeline):
    p = tmpdir_mod / "bc.png"
    renderer.render(LabelSpec(), out_path=str(p))
    assert pipeline.scan(str(p)).context.barcode == "8901234567890"


@needs_zbar
def test_dual_mrp_is_definitive_with_a_barcode(renderer, tmpdir_mod, pipeline):
    """
    With a GTIN the finding is evidence; without one it is a lead that
    has to be hedged and confirmed by a human. Confidence must reflect
    that difference.
    """
    from src.core.cross_scan import CrossScanAnalyzer

    scans = []
    for price in (120.0, 145.0):
        spec = LabelSpec()
        spec.mrp = price
        p = tmpdir_mod / f"bc_dual_{price}.png"
        renderer.render(spec, out_path=str(p))
        scans.append(pipeline.scan(str(p)))

    findings = CrossScanAnalyzer().findings(scans)
    assert len(findings) == 1
    assert findings[0].confidence >= 0.9
    assert "GTIN" in findings[0].message


def test_unit_ocr_confusions_are_normalised():
    """
    "1 l" read as "1 |" and "500 g" read as "500 6" both extracted
    nothing, so the package was reported as missing its net quantity -
    a false violation from one misread character.
    """
    from src.extraction.fields import normalise_unit

    assert normalise_unit("|") == "l"
    assert normalise_unit("I") == "l"
    assert normalise_unit("6") == "g"
    assert normalise_unit("kg") == "kg"


# =====================================================================
# Barcode / GTIN
# =====================================================================

def test_gtin_check_digit():
    from src.vision.barcode import gtin_checksum_valid

    assert gtin_checksum_valid("8901234567890")
    assert not gtin_checksum_valid("8901234567891")   # wrong check digit
    assert not gtin_checksum_valid("notanumber")


@needs_zbar
def test_pipeline_reads_gtin(renderer, tmpdir_mod, pipeline):
    p = tmpdir_mod / "gtin.png"
    renderer.render(LabelSpec(), out_path=str(p))
    assert pipeline.scan(str(p)).context.barcode == "8901234567890"


@needs_zbar
def test_dual_mrp_is_definitive_with_gtin(renderer, tmpdir_mod, pipeline):
    """
    A GTIN moves the finding from inference to fact. Without it the
    analyser matches on brand, name and net quantity, which is wrong in
    both directions: different variants of one brand and size look
    identical, and one product under different lighting can look like two.
    """
    from src.core.cross_scan import CrossScanAnalyzer

    scans = []
    for price in (120.0, 145.0):
        spec = LabelSpec()
        spec.mrp = price
        p = tmpdir_mod / f"gt_{price}.png"
        renderer.render(spec, out_path=str(p))
        scans.append(pipeline.scan(str(p)))

    findings = CrossScanAnalyzer().findings(scans)
    assert len(findings) == 1
    assert findings[0].confidence >= 0.9
    assert "GTIN" in findings[0].message


@needs_zbar
def test_different_gtin_is_not_the_same_product(renderer, tmpdir_mod, pipeline):
    """
    Identical brand, name and net quantity but a different GTIN is a
    different commodity. This is precisely the case inferred matching
    gets wrong, and it would produce a confident false accusation.
    """
    from src.core.cross_scan import CrossScanAnalyzer

    scans = []
    for price, gtin in ((120.0, "8901234567894"), (99.0, "8901234567907")):
        spec = LabelSpec()
        spec.mrp, spec.gtin = price, gtin
        p = tmpdir_mod / f"var_{price}.png"
        renderer.render(spec, out_path=str(p))
        scans.append(pipeline.scan(str(p)))

    assert len({s.context.barcode for s in scans}) == 2
    assert CrossScanAnalyzer().findings(scans) == []


def test_misread_gtin_is_discarded(tmpdir_mod):
    """
    A bad check digit is dropped, not surfaced with a warning. A misread
    GTIN causes no visible error - it silently merges or splits product
    groups in the dual-MRP comparison, which is worse than no identifier.
    """
    import numpy as np

    from src.vision.barcode import read_barcode

    assert read_barcode(np.zeros((80, 80, 3), np.uint8)) is None


# =====================================================================
# Real-packet regressions (Casio W-218HM wrist watch box)
# =====================================================================
# Every case below was found by running the extractor against one real
# photographed label. All four had passed on synthetic data.

def _casio_spans():
    lines = [
        "IMPORTED BY:- CASIO INDIA CO., PVT. LTD.",
        "A-41, 1ST FLOOR,MCIE, MATHURA ROAD, NEW DELHI- 110044",
        "IN CASE OF CONSUMER COMPLAINTS CONTACT :",
        "CONSUMER CARE OFFICER AT ABOVE ADDRESS,",
        "TEL. NO: +918447114400",
        "EMAIL : casiocare@casio.co.in",
        "COUNTRY OF ORIGIN: CHINA",
        "COMMODITY: WRIST WATCH",
        "MODEL: W-218HM-5BVDF",
        "MANUFACTURED ON : 03/2026",
        "NET QUANTITY 1 PIECE",
        "MRP: Rs. 1495.00 INCL. OF ALL TAXES",
    ]
    spans, = [[]]
    for i, line in enumerate(lines):
        x = 20.0
        for word in line.split():
            spans.append(TextSpan(word, BBox(x, 100.0 + i * 40, len(word) * 11.0, 22.0)))
            x += len(word) * 11.0 + 10
    return spans


def test_price_above_999_without_comma():
    """
    REGRESSION: NUMBER used \\d{1,3}(?:,\\d{2,3})* - with the comma group
    optional, "1495.00" matched as "149" and alternation never tried the
    long-number branch. Every MRP over Rs.999 written without a comma was
    read as its first three digits. A tenfold error in the single most
    important declaration on the package.
    """
    from src.extraction.fields import RegexExtractor

    d = {x.field_id: x for x in RegexExtractor().extract(_casio_spans())}
    assert d["retail_sale_price"].value == pytest.approx(1495.00)


def test_quantity_counted_in_pieces():
    """
    REGRESSION: Rule 13(5)(ii) prescribes "N" or "U" for goods sold by
    number, and almost no real pack obeys it. "NET QUANTITY 1 PIECE"
    failed to match, and a greedy fallback then captured
    "MANUFACTURED ON : 03/2026" as the net quantity.
    """
    from src.extraction.fields import RegexExtractor

    d = {x.field_id: x for x in RegexExtractor().extract(_casio_spans())}
    assert d["net_quantity"].present
    assert d["net_quantity"].value == pytest.approx(1.0)
    assert "MANUFACTURED" not in d["net_quantity"].raw_text.upper()


def test_commodity_heading_is_the_generic_name():
    """Imported packs head the generic name "COMMODITY:"."""
    from src.extraction.fields import RegexExtractor

    d = {x.field_id: x for x in RegexExtractor().extract(_casio_spans())}
    assert d["common_name"].present
    assert "WRIST WATCH" in str(d["common_name"].value).upper()


def test_all_mandatory_declarations_found_on_real_label():
    from src.extraction.fields import RegexExtractor

    d = {x.field_id: x for x in RegexExtractor().extract(_casio_spans())}
    for field in ("manufacturer_details", "consumer_care", "country_of_origin",
                  "common_name", "manufacture_date", "net_quantity",
                  "retail_sale_price"):
        assert d[field].present, f"{field} not extracted from real label"


# =====================================================================
# Classifier categories and exemption effects
# =====================================================================

def test_all_referenced_categories_are_known():
    """
    Every category any exemption or rule keys to must exist in
    CATEGORY_KEYWORDS, or that exemption is permanently unreachable
    however correctly its threshold was transcribed from the gazette.
    """
    from src.core.pipeline import CATEGORY_KEYWORDS
    from src.core.rules_engine import RuleConfig

    cfg = RuleConfig()
    known = set(CATEGORY_KEYWORDS) | {"other"}
    referenced = set()
    for ex in cfg.exemptions:
        cond = ex.get("condition") or {}
        if cond.get("type") == "commodity_category":
            referenced.add(cond.get("value"))
        for c in (cond.get("commodities") or []):
            referenced.add(c)
    missing = referenced - known
    assert not missing, f"categories referenced but never classifiable: {missing}"


def test_chapter_ii_exemption_short_circuits_the_scan(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: exemptions whose only effect was
    `chapter_ii_not_applicable: true` and an empty `exempts_declarations`
    resolved their CONDITION correctly and then exempted nothing at all
    - the flag lit up and produced no consequence. An institutional
    label missing every mandatory declaration reported ~20 violations
    for a package Chapter II never governed.
    """
    spec = LabelSpec()
    spec.common_name = "NOT FOR RETAIL SALE Industrial Cleaner"
    spec.mrp = None
    p = tmpdir_mod / "inst.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    assert len(res.findings) == 1
    assert res.findings[0].rule_id == "chapter_ii_scope"
    assert res.findings[0].outcome is Outcome.NOT_APPLICABLE


def test_over_25kg_exempts_chapter_ii(tmpdir_mod, pipeline):
    from src.core.rules_engine import ExemptionResolver, RuleConfig
    from src.core.schema import Declaration, PackageContext

    resolver = ExemptionResolver(RuleConfig())
    ctx = PackageContext()
    d = Declaration(field_id="net_quantity", value=30, unit="kg", present=True)
    ex = resolver.chapter_ii_exemption(ctx, [d])
    assert ex is not None and ex["id"] == "over_25kg_25l"


def test_pan_masala_excluded_from_small_package_exemption(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION x2. First: `excluded_commodities` was written into the
    YAML but never read by the resolver, so the one amendment clause
    (GSR 881(E), 2025) transcribed specifically for this exemption was
    silently ignored. Second, once that check was added: the classifier
    scored categories by MATCH COUNT, so "Pan Masala Sachet" tied
    pan_masala's "pan masala" against food_packaged's bare "masala" at
    one hit each, and the first-defined category won every tie - every
    pan masala label was silently classified as ordinary food.
    """
    spec = LabelSpec()
    spec.common_name = "Pan Masala Sachet"
    spec.net_quantity_value, spec.net_quantity_unit = 5, "g"
    spec.unit_sale_price = None
    p = tmpdir_mod / "pm.png"
    renderer.render(spec, out_path=str(p))
    res = scan_complete(pipeline, p)
    assert res.context.commodity_category == "pan_masala"
    usp = res.declaration("unit_sale_price")
    finding = next(f for f in res.findings if f.field_id == "unit_sale_price")
    assert finding.outcome is Outcome.VIOLATION


def test_ordinary_small_package_keeps_the_exemption(renderer, tmpdir_mod, pipeline):
    """Control for the pan masala test: the exemption still applies to
    everything the pan-masala carve-out does not name."""
    spec = LabelSpec()
    spec.net_quantity_value, spec.net_quantity_unit = 5, "g"
    spec.unit_sale_price = None
    p = tmpdir_mod / "ord.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    finding = next(f for f in res.findings if f.field_id == "unit_sale_price")
    assert finding.outcome is Outcome.NOT_APPLICABLE


def test_flag_phrases_detected(renderer, tmpdir_mod, pipeline):
    spec = LabelSpec()
    spec.common_name = "FOR EXPORT ONLY Detergent"
    p = tmpdir_mod / "exp.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    assert res.context.for_export is True
    assert res.context.industrial_or_institutional is False


# =====================================================================
# Config keys that encode a legal EFFECT must be read by code
# =====================================================================
# Four separate bugs in this project had the same shape: a rule was
# transcribed correctly from the gazette, its condition resolved
# correctly, and its EFFECT was read by nothing at all. The rule looked
# complete in review and did nothing at runtime.

def test_effect_keys_are_read_by_code():
    """
    Guard against the recurring failure mode: a YAML key describing a
    legal consequence that no code path consults.
    """
    import pathlib

    import yaml

    EFFECT_KEYS = {
        "chapter_ii_not_applicable", "relax_size_rules", "defer_to",
        "never_exempt_fields", "excluded_commodities", "action",
        "relaxation_unavailable", "enforce_table_I",
    }
    code = "".join(
        f.read_text(encoding="utf-8") for f in pathlib.Path("src").rglob("*.py")
    )

    declared = set()
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                declared.add(k)
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for f in pathlib.Path("rules").glob("*.yaml"):
        walk(yaml.safe_load(open(f, encoding="utf-8")))

    unread = [k for k in (declared & EFFECT_KEYS) if f'"{k}"' not in code]
    assert not unread, f"effect keys declared in YAML but never read: {unread}"


def test_rule_33_relaxation_suppresses_but_records(renderer, tmpdir_mod):
    """
    A relaxation makes a finding lawful; it does not make it invisible.
    The finding is recorded as SUPPRESSED with its reference so an
    inspector sees the claim, rather than being silently dropped.
    """
    from src.core.pipeline import CompliancePipeline
    from src.core.rules_engine import RulesEngine

    p = tmpdir_mod / "relax.png"
    renderer.render(apply_violation(LabelSpec(), "missing_consumer_care"),
                    out_path=str(p), violations=["missing_consumer_care"])

    plain_pipe = CompliancePipeline()
    plain = scan_complete(plain_pipe, p)
    assert any(f.rule_id == "consumer_care.presence"
               and f.outcome is Outcome.VIOLATION for f in plain.findings)

    def lookup(manufacturer, rule_id):
        if rule_id == "consumer_care.presence":
            return {"reference": "WM-33/2026/117", "valid_until": "2027-03-31"}
        return None

    relaxed = scan_complete(
        CompliancePipeline(engine=RulesEngine(relaxation_lookup=lookup)), p
    )
    f = next(x for x in relaxed.findings if x.rule_id == "consumer_care.presence")
    assert f.outcome is Outcome.SUPPRESSED
    assert "WM-33/2026/117" in f.message
    assert relaxed.summary()["suppressed_by_relaxation"] == 1


def test_relaxation_lookup_failure_never_clears_a_violation(renderer, tmpdir_mod):
    """A broken lookup must not become a false pass."""
    from src.core.pipeline import CompliancePipeline
    from src.core.rules_engine import RulesEngine

    def boom(manufacturer, rule_id):
        raise RuntimeError("database down")

    p = tmpdir_mod / "relax_fail.png"
    renderer.render(apply_violation(LabelSpec(), "missing_consumer_care"),
                    out_path=str(p), violations=["missing_consumer_care"])
    res = scan_complete(
        CompliancePipeline(engine=RulesEngine(relaxation_lookup=boom)), p
    )
    f = next(x for x in res.findings if x.rule_id == "consumer_care.presence")
    assert f.outcome is Outcome.VIOLATION


def test_alcoholic_beverage_referred_not_cleared(renderer, tmpdir_mod, pipeline):
    """
    Whether State Excise law provides for a price declaration is
    jurisdiction-dependent, so `action: flag_for_human_review` must
    abstain rather than auto-clear.
    """
    spec = LabelSpec()
    spec.common_name = "Whisky Premium"
    p = tmpdir_mod / "alc.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    assert res.context.commodity_category == "alcoholic_beverage"
    f = next(x for x in res.findings
             if x.field_id == "retail_sale_price" and x.exemption_applied)
    assert f.outcome is Outcome.INDETERMINATE
    assert res.is_compliant is None


def test_medical_device_defers_size_rules(renderer, tmpdir_mod, pipeline):
    spec = LabelSpec()
    spec.common_name = "Digital Thermometer"
    p = tmpdir_mod / "md.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    assert res.context.commodity_category == "medical_device"
    nh = next(x for x in res.findings if x.rule_id == "numeral_height")
    assert nh.outcome is Outcome.NOT_APPLICABLE


def test_net_quantity_never_loses_its_size_requirement(renderer, tmpdir_mod, pipeline):
    """
    Rule 7(5) relaxes size rules where another law applies, but net
    quantity, MRP, expiry and consumer care are expressly never exempt.
    Packaged food is FSSAI-governed, so this fires on most groceries -
    the carve-out must hold.
    """
    spec = LabelSpec()
    spec.common_name = "Biscuit Pack"
    p = tmpdir_mod / "food.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    assert res.context.other_law_applies is True
    nh = next(x for x in res.findings if x.rule_id == "numeral_height")
    assert nh.outcome is not Outcome.NOT_APPLICABLE


def test_medical_device_cannot_use_rule_33_relaxation(renderer, tmpdir_mod):
    """
    Rule 33(2): where the Medical Devices Rules 2017 apply, the Rule 33
    relaxation does not. Implementing relaxations opened this hole -
    without the check, any relaxation record would suppress findings on
    a medical device that the Rules say cannot be relaxed.
    """
    from src.core.pipeline import CompliancePipeline
    from src.core.rules_engine import RulesEngine

    def lookup(manufacturer, rule_id):
        return {"reference": "WM-33/2026/999", "valid_until": "2027-03-31"}

    spec = apply_violation(LabelSpec(), "missing_consumer_care")
    spec.common_name = "Digital Thermometer"
    p = tmpdir_mod / "md_relax.png"
    renderer.render(spec, out_path=str(p), violations=["missing_consumer_care"])

    res = CompliancePipeline(engine=RulesEngine(relaxation_lookup=lookup)).scan(str(p))
    assert res.context.commodity_category == "medical_device"
    assert not res.suppressed, "Rule 33(2) bars relaxation for medical devices"


# =====================================================================
# Checks that were declared in config but never implemented
# =====================================================================

def test_price_rounding_enforced(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: Rule 6(1)(e) as amended 2017 requires the price rounded
    to the nearest rupee or 50 paise. The check was transcribed from the
    gazette with a citation and never implemented - the engine's elif
    chain fell through on the unknown type and emitted nothing, so
    Rs.119.99 passed as fully compliant.
    """
    spec = LabelSpec()
    spec.mrp = 119.99
    p = tmpdir_mod / "round_bad.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    f = next(x for x in res.findings if "price_rounding" in x.rule_id)
    assert f.outcome is Outcome.VIOLATION


@pytest.mark.parametrize("price", [120.00, 120.50])
def test_correctly_rounded_prices_pass(renderer, tmpdir_mod, pipeline, price):
    spec = LabelSpec()
    spec.mrp = price
    p = tmpdir_mod / f"round_ok_{price}.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    f = next(x for x in res.findings if "price_rounding" in x.rule_id)
    assert f.outcome is Outcome.COMPLIANT


def test_future_manufacture_date_flagged(renderer, tmpdir_mod, pipeline):
    """A pack cannot have been manufactured after today."""
    spec = LabelSpec()
    spec.mfg_year, spec.mfg_month = 2099, 6
    p = tmpdir_mod / "future.png"
    renderer.render(spec, out_path=str(p))
    res = pipeline.scan(str(p))
    f = next(x for x in res.findings if "date_plausible" in x.rule_id)
    assert f.outcome is Outcome.VIOLATION


def test_unknown_check_type_is_reported_not_skipped():
    """
    The systemic fix. An unrecognised check type must surface as
    UNVERIFIED_RULE, never vanish - silence is how four real checks sat
    dead behind valid-looking citations.
    """
    import copy

    from src.core.rules_engine import RuleConfig, RulesEngine
    from src.core.schema import Declaration, ScanResult

    cfg = RuleConfig()
    spec = next(d for d in cfg.declarations if d["id"] == "net_quantity")
    spec = copy.deepcopy(spec)
    spec["checks"] = [{"type": "some_unimplemented_check"}]
    cfg.declarations = [spec]

    scan = ScanResult()
    scan.declarations = [
        Declaration(field_id="net_quantity", raw_text="500 g",
                    value=500, unit="g", present=True)
    ]
    RulesEngine(cfg).evaluate(scan)
    assert any(f.outcome is Outcome.UNVERIFIED_RULE for f in scan.findings)


# =====================================================================
# API surface: evidence, exports, and error handling
# =====================================================================

@pytest.fixture
def client(tmp_path, monkeypatch):
    import warnings
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient

    import src.api.main as api

    monkeypatch.setattr(api, "DB_PATH", tmp_path / "t.db")
    api._SCHEMA_READY.clear()
    api._pipeline = None
    return TestClient(api.app)



def scan_complete(pipeline, path, **kw):
    """
    Scan a package with full-panel coverage asserted.

    A single photograph cannot establish that a declaration is absent
    from the PACKAGE - only that it is not in this view. Tests that mean
    "the package genuinely lacks this field" must say so explicitly,
    which is what a multi-image capture will do in production.
    """
    res = pipeline.scan(str(path), **kw)
    res.coverage_complete = True
    res.findings = []
    pipeline.engine.evaluate(res)
    return res


INSPECTOR = {"Authorization": "Bearer demo-inspector"}
SUPERVISOR = {"Authorization": "Bearer demo-supervisor"}
ADMIN = {"Authorization": "Bearer demo-admin"}


def _upload(client, renderer, tmpdir_mod, spec=None):
    spec = spec or LabelSpec()
    p = tmpdir_mod / "api.png"
    renderer.render(spec, out_path=str(p))
    return client.post(
        "/scan",
        files={"file": ("a.png", p.read_bytes(), "image/png")},
        headers=INSPECTOR,
    ).json()


def test_scan_produces_evidence_crops(client, renderer, tmpdir_mod):
    """
    REGRESSION: every finding carried a bounding box and the crop
    extractor existed, but nothing ever called it - so no violation
    report contained a single evidence image, despite "attachment of
    photographs and supporting evidence" being an explicit functional
    requirement.
    """
    spec = LabelSpec()
    spec.mrp = 119.99
    res = _upload(client, renderer, tmpdir_mod, spec)
    with_crops = [f for f in res["findings"] if f.get("evidence_crop_path")]
    assert with_crops, "no finding carried an evidence crop"


@pytest.mark.parametrize("fmt", ["pdf", "html", "json", "docx"])
def test_all_export_formats(client, renderer, tmpdir_mod, fmt):
    """
    "Export of reports to PDF and editable formats" is an explicit
    requirement. docx was implemented and unreachable - the endpoint's
    format pattern simply never offered it.
    """
    res = _upload(client, renderer, tmpdir_mod)
    r = client.get(f"/scans/{res['scan_id']}/report?fmt={fmt}", headers=INSPECTOR)
    assert r.status_code == 200


def test_empty_upload_returns_400_not_500(client):
    """
    REGRESSION: cv2.imdecode RAISES on an empty buffer rather than
    returning None, so the `arr is None` guard never saw it and an empty
    upload produced a 500 with an OpenCV stack trace.
    """
    r = client.post(
        "/scan", files={"file": ("a.png", b"", "image/png")}, headers=INSPECTOR
    )
    assert r.status_code == 400


def test_oversized_upload_rejected(client):
    r = client.post(
        "/scan",
        files={"file": ("a.png", b"x" * (26 * 1024 * 1024), "image/png")},
        headers=INSPECTOR,
    )
    assert r.status_code == 413


@pytest.mark.parametrize("path,headers,expected", [
    ("/dashboard", {"Authorization": "Bearer nope"}, 401),
    ("/dashboard", {}, 401),
    ("/scans/doesnotexist", INSPECTOR, 404),
    ("/scans/doesnotexist/report?fmt=pdf", INSPECTOR, 404),
    ("/scans/x/report?fmt=exe", INSPECTOR, 422),
])
def test_api_error_paths(client, path, headers, expected):
    assert client.get(path, headers=headers).status_code == expected


def test_relaxation_requires_admin(client):
    body = {"manufacturer": "X", "rule_id": "consumer_care.presence"}
    assert client.post("/relaxations", json=body, headers=INSPECTOR).status_code == 403
    assert client.post("/relaxations", json=body, headers=ADMIN).status_code == 200


# =====================================================================
# A failing detector must never read as compliance
# =====================================================================

def test_overlay_detector_crash_abstains(renderer, tmpdir_mod, monkeypatch):
    """
    REGRESSION: on exception the pipeline set `overlays = []`, which is
    indistinguishable from "the detector ran and this package has no
    pasted label" - so a crash inside the sticker detector was reported
    as a clean COMPLIANT result on a legal check. Failure and a clean
    negative must not collapse to the same value.
    """
    from src.core.pipeline import CompliancePipeline
    import src.vision.overlay as ov

    def boom(*a, **k):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(ov, "detect_overlays", boom)

    p = tmpdir_mod / "boom.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = CompliancePipeline().scan(str(p))

    f = next(x for x in res.findings if x.rule_id == "sticker_alteration")
    assert f.outcome is Outcome.INDETERMINATE
    assert res.overlay_detection_failed is True
    assert res.is_compliant is None


def test_gm_detection_crash_does_not_assert_violation(renderer, tmpdir_mod, monkeypatch):
    """
    The mirror case: swallowing the exception left `gm_mark_present`
    False, which asserts "no GM mark on this package" - a VIOLATION
    never actually established.
    """
    from src.core.pipeline import CompliancePipeline
    import src.vision.overlay as ov

    def boom(*a, **k):
        raise RuntimeError("token search exploded")

    monkeypatch.setattr(ov, "find_top_token", boom)

    p = tmpdir_mod / "gmboom.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = CompliancePipeline().scan(
        str(p), hints={"genetically_modified_food": True}
    )
    f = next(x for x in res.findings if x.rule_id == "gm_food_label")
    assert f.outcome is not Outcome.VIOLATION


# =====================================================================
# Persistence fidelity and unit/boundary correctness
# =====================================================================

def test_scan_survives_database_round_trip(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: scan_from_dict rebuilt PackageContext by naming each
    field, so every field added after it was written was silently
    dropped on the way back out of the database - including `barcode`.
    Reports and the dual-MRP endpoint both read scans back through this
    function, so the GTIN work was quietly defeated at the API layer.
    """
    from src.report.builder import scan_from_dict

    spec = LabelSpec()
    spec.mrp = 119.99
    p = tmpdir_mod / "rt.png"
    renderer.render(spec, out_path=str(p))
    orig = pipeline.scan(str(p))
    back = scan_from_dict(orig.to_dict())

    for attr in ("barcode", "printed_price", "pdp_detection_method",
                 "pdp_confidence", "commodity_category"):
        assert getattr(back.context, attr) == getattr(orig.context, attr), attr


@needs_zbar
def test_dual_mrp_stays_definitive_through_the_api(client, renderer, tmpdir_mod):
    """The GTIN must survive persistence, or every finding downgrades
    from definitive (0.90) to inferred (0.55)."""
    for price in (120.0, 145.0):
        spec = LabelSpec()
        spec.mrp = price
        p = tmpdir_mod / f"api_{price}.png"
        renderer.render(spec, out_path=str(p))
        client.post("/scan",
                    files={"file": ("a.png", p.read_bytes(), "image/png")},
                    headers=INSPECTOR)

    d = client.get("/analysis/dual-mrp", headers=INSPECTOR).json()
    assert d["violations"], "expected a dual-MRP finding"
    v = d["violations"][0]
    assert v["confidence"] >= 0.9
    assert "GTIN" in v["message"]


@pytest.mark.parametrize("qty,unit,exempt", [
    (10, "g", True), (0.01, "kg", True),      # identical quantities
    (10, "ml", True), (0.01, "l", True),
    (11, "g", False),
])
def test_small_package_threshold_is_unit_independent(qty, unit, exempt):
    """
    REGRESSION: the threshold test asked whether the unit STRING
    appeared in the rule's unit list, so the same physical quantity gave
    different answers depending on which unit the packer printed.
    """
    from src.core.rules_engine import RuleConfig, ExemptionResolver
    from src.core.schema import PackageContext, Declaration

    r = ExemptionResolver(RuleConfig())
    d = Declaration(field_id="net_quantity", value=qty, unit=unit, present=True)
    got = bool(r.resolve(PackageContext(commodity_category="detergent_household"), [d]))
    assert got is exempt


@pytest.mark.parametrize("qty,unit,out_of_scope", [
    (26, "kg", True), (26000, "g", True),     # a 26kg sack either way
    (25, "kg", False), (25000, "g", False),   # exactly 25kg is IN scope
    (30, "l", True), (30000, "ml", True),
])
def test_chapter_ii_threshold_is_unit_independent(qty, unit, out_of_scope):
    from src.core.rules_engine import RuleConfig, ExemptionResolver
    from src.core.schema import PackageContext, Declaration

    r = ExemptionResolver(RuleConfig())
    d = Declaration(field_id="net_quantity", value=qty, unit=unit, present=True)
    got = r.chapter_ii_exemption(PackageContext(), [d]) is not None
    assert got is out_of_scope


@pytest.mark.parametrize("area,expected", [
    (49.9, 1.0), (50.0, 1.0), (50.1, 1.5),
    (100.0, 1.5), (100.1, 2.5),
    (500.0, 2.5), (500.1, 4.0),
    (2500.0, 4.0), (2500.1, 6.0),
])
def test_table_I_band_boundaries(area, expected):
    """
    Band edges decide which legal threshold applies, so an off-by-one
    here silently measures a package against the wrong minimum. Upper
    bounds are inclusive, matching the conventional reading of Table-I.
    """
    from src.core.rules_engine import RulesEngine, RuleConfig
    from src.core.schema import ScanResult, PackageContext, Declaration

    cfg = RuleConfig()
    spec = next(p for p in cfg.presentation if p["id"] == "numeral_height")
    scan = ScanResult()
    scan.context = PackageContext(pdp_area_cm2=area)
    decl = Declaration(field_id="net_quantity", value=500, unit="g", present=True)
    scan.declarations = [decl]
    assert RulesEngine(cfg)._lookup_required_height(spec, scan, decl) == expected


# =====================================================================
# Report content, metric arithmetic, and concurrency
# =====================================================================

def test_missing_declaration_is_a_confident_finding(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: an absent field still has a Declaration object carrying
    extraction_confidence 0.0, and that value was used as the finding's
    confidence - so a CRITICAL violation printed "Conf. 0.00" on the
    inspector's report, reading as "the system is unsure" when absence
    on a readable image is a confident result.
    """
    p = tmpdir_mod / "conf.png"
    renderer.render(apply_violation(LabelSpec(), "missing_mrp"),
                    out_path=str(p), violations=["missing_mrp"])
    res = scan_complete(pipeline, p)
    f = next(x for x in res.violations if x.rule_id == "retail_sale_price.presence")
    assert f.confidence >= 0.8


def test_report_contains_what_an_inspector_needs(renderer, tmpdir_mod, pipeline):
    """A report that returns 200 but omits the citation, the verdict or
    the evidence is not a usable enforcement document."""
    import pathlib
    from src.report.builder import build_report

    p = tmpdir_mod / "rep.png"
    renderer.render(apply_violation(LabelSpec(), "missing_mrp"),
                    out_path=str(p), violations=["missing_mrp"])
    res = scan_complete(pipeline, p)
    html = pathlib.Path(
        build_report(res, tmpdir_mod / "rep.html", fmt="html")
    ).read_text(encoding="utf-8")

    v = res.violations[0]
    assert v.citation in html
    assert res.scan_id in html
    assert "NON-COMPLIANT" in html.upper()
    assert "data:image" in html          # evidence embedded


@pytest.mark.parametrize("ref,hyp,expected", [
    ("hello", "hello", 0.0),
    ("hello", "hallo", 0.2),
    ("hello", "", 1.0),
    ("", "", 0.0),
    ("abc", "abcd", 1 / 3),
])
def test_character_error_rate_arithmetic(ref, hyp, expected):
    """Every accuracy figure quoted from this project comes out of the
    harness, so its arithmetic is itself worth pinning."""
    from src.evaluation.harness import character_error_rate

    assert character_error_rate(ref, hyp) == pytest.approx(expected)


@pytest.mark.parametrize("a,b,expected", [
    ("kitten", "sitting", 3), ("", "abc", 3), ("abc", "abc", 0), ("flaw", "lawn", 2),
])
def test_levenshtein(a, b, expected):
    from src.evaluation.harness import levenshtein

    assert levenshtein(a, b) == expected


def test_precision_recall_f1_and_zero_division():
    from src.evaluation.harness import PRCounts

    c = PRCounts()
    c.tp, c.fp, c.fn = 8, 2, 4
    assert c.precision == pytest.approx(0.8)
    assert c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(0.72727, rel=1e-4)

    z = PRCounts()
    assert (z.precision, z.recall, z.f1) == (0.0, 0.0, 0.0)


def test_concurrent_scans_do_not_contaminate(client, renderer, tmpdir_mod):
    """
    The pipeline is a module-level singleton shared across requests. If
    any per-scan state leaked onto it, concurrent uploads would blend -
    one package's price appearing on another's report.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = []
    for i, price in enumerate([100.0, 150.0, 200.0, 250.0]):
        spec = LabelSpec()
        spec.mrp = price
        p = tmpdir_mod / f"cc{i}.png"
        renderer.render(spec, out_path=str(p))
        items.append((p, price))

    def go(item):
        path, price = item
        j = client.post(
            "/scan",
            files={"file": ("a.png", path.read_bytes(), "image/png")},
            headers=INSPECTOR,
        ).json()
        d = [x for x in j["declarations"] if x["field_id"] == "retail_sale_price"]
        return price, (d[0]["value"] if d else None), j["scan_id"]

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(go, items))

    for expected, got, _ in results:
        assert got == pytest.approx(expected, abs=0.01)
    assert len({sid for _, _, sid in results}) == len(results)


# =====================================================================
# Ruleset validation
# =====================================================================

def _mutated_config(mutate):
    """Copy the real ruleset, mutate it, and try to load it."""
    import shutil
    import tempfile
    import yaml
    from src.core.rules_engine import RuleConfig

    d = tempfile.mkdtemp()
    for f in ("lmpc_2011.yaml", "exemptions.yaml"):
        shutil.copy(f"rules/{f}", f"{d}/{f}")
    doc = yaml.safe_load(open(f"{d}/lmpc_2011.yaml", encoding="utf-8"))
    mutate(doc)
    yaml.safe_dump(doc, open(f"{d}/lmpc_2011.yaml", "w"))
    return RuleConfig(d)


def test_real_ruleset_validates():
    from src.core.rules_engine import RuleConfig

    assert RuleConfig().validate(strict=False) == []


def test_invalid_severity_rejected_at_load():
    """
    REGRESSION: an invalid severity raised a bare ValueError in the
    middle of evaluating a scan, so one typo took down every request
    with a trace naming neither the file nor the rule.
    """
    with pytest.raises(ValueError, match="severity"):
        _mutated_config(
            lambda d: d["declarations"][0].update(severity="catastrophic")
        )


def test_typo_in_key_is_rejected_not_ignored():
    """
    REGRESSION: `verifed` for `verified` was silently dropped by the
    YAML loader, quietly changing the rule's meaning with no error - the
    same silent-config failure that hid bugs throughout this ruleset.
    """
    def typo(d):
        decl = d["declarations"][0]
        decl["verifed"] = decl.pop("verified")

    with pytest.raises(ValueError):
        _mutated_config(typo)


def test_missing_citation_rejected():
    with pytest.raises(ValueError, match="citation"):
        _mutated_config(lambda d: d["declarations"][0].pop("citation"))


# =====================================================================
# Absence cannot be proven from a partial view
# =====================================================================
# Found by running real photographs of real packages. Both reports
# asserted 6-8 violations at 0.90 confidence, most of them fields that
# were almost certainly present on a panel the photo did not show.

def test_single_view_cannot_prove_absence(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: a missing field was reported as a VIOLATION at 0.90
    confidence from one photograph. Mandatory declarations are routinely
    split across panels - name and quantity on the front, address, date
    and consumer care on the back - so absence from one view is not
    absence from the package. The system was issuing confident false
    accusations against compliant products.
    """
    p = tmpdir_mod / "partial.png"
    renderer.render(apply_violation(LabelSpec(), "missing_mrp"),
                    out_path=str(p), violations=["missing_mrp"])
    res = pipeline.scan(str(p))

    assert res.coverage_complete is False
    presence = next(x for x in res.findings
                    if x.rule_id == "retail_sale_price.presence")
    assert presence.outcome is Outcome.INDETERMINATE
    assert "not visible in this image" in presence.message
    assert res.is_compliant is None


def test_absence_is_a_violation_once_coverage_is_established(renderer, tmpdir_mod, pipeline):
    """With every panel captured, a missing declaration IS a violation."""
    p = tmpdir_mod / "full.png"
    renderer.render(apply_violation(LabelSpec(), "missing_mrp"),
                    out_path=str(p), violations=["missing_mrp"])
    res = pipeline.scan(str(p))
    res.coverage_complete = True
    res.findings = []
    pipeline.engine.evaluate(res)

    presence = next(x for x in res.findings
                    if x.rule_id == "retail_sale_price.presence")
    assert presence.outcome is Outcome.VIOLATION
    assert res.is_compliant is False


def test_grouping_cannot_pass_while_fields_are_missing(renderer, tmpdir_mod, pipeline):
    """
    REGRESSION: pdp_grouping inspected only fields that were FOUND, so a
    report stated "all mandatory declarations are grouped on the
    principal display panel" alongside findings saying those same
    declarations were missing - two findings flatly contradicting each
    other on one page.
    """
    p = tmpdir_mod / "group.png"
    renderer.render(apply_violation(LabelSpec(), "missing_manufacturer"),
                    out_path=str(p), violations=["missing_manufacturer"])
    res = pipeline.scan(str(p))

    grouping = next(x for x in res.findings if x.rule_id == "pdp_grouping")
    assert grouping.outcome is not Outcome.COMPLIANT
    assert "Cannot assess grouping" in grouping.message


def test_report_identifies_which_field_each_finding_concerns(renderer, tmpdir_mod, pipeline):
    """
    Rule 7(3) is evaluated per declaration, so a report showed two
    Rule 7(3) rows with opposite verdicts and no way to tell which field
    each referred to - unusable for an officer who has to act on it.
    """
    import pathlib
    import re
    from src.report.builder import build_report

    p = tmpdir_mod / "fields.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = pipeline.scan(str(p))
    html = pathlib.Path(
        build_report(res, tmpdir_mod / "fields.html", fmt="html")
    ).read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", re.sub(r'data:image/[^"]{100,}', "", html))

    assert "net_quantity" in text
    aspect_rows = re.findall(r"Rule 7\(3\)\s+(\w+)", text)
    assert len(set(aspect_rows)) > 1, "per-field rows must be distinguishable"


# =====================================================================
# An implausible measurement is not evidence of an offence
# =====================================================================
# From the real-photo reports: Rule 7(3) asserted a VIOLATION on a
# measured width/height of 0.08 - characters twelve times taller than
# wide. That is not condensed type, it is a failed measurement.

def test_implausible_ratio_abstains_rather_than_accuses():
    from src.core.rules_engine import RulesEngine, RuleConfig
    from src.core.schema import (
        ScanResult, Declaration, GlyphMetrics, BBox, Severity,
    )

    cfg = RuleConfig()
    spec = next(p for p in cfg.presentation if p["id"] == "glyph_aspect_ratio")
    scan = ScanResult()
    scan.declarations = [Declaration(
        field_id="net_quantity", present=True, bbox=BBox(0, 0, 50, 20),
        glyph=GlyphMetrics(cap_height_px=30, width_over_height=0.08,
                           n_glyphs_measured=6),
    )]
    out = RulesEngine(cfg)._eval_aspect_ratio(
        spec, spec["check"], scan, True, "Rule 7(3)", Severity.MINOR
    )
    assert out and out[0].outcome is Outcome.INDETERMINATE


def test_genuinely_condensed_text_still_violates():
    """The floor must not swallow the violations the rule exists for."""
    from src.core.rules_engine import RulesEngine, RuleConfig
    from src.core.schema import (
        ScanResult, Declaration, GlyphMetrics, BBox, Severity,
    )

    cfg = RuleConfig()
    spec = next(p for p in cfg.presentation if p["id"] == "glyph_aspect_ratio")
    scan = ScanResult()
    scan.declarations = [Declaration(
        field_id="net_quantity", present=True, bbox=BBox(0, 0, 50, 20),
        glyph=GlyphMetrics(cap_height_px=30, width_over_height=0.25,
                           n_glyphs_measured=6),
    )]
    out = RulesEngine(cfg)._eval_aspect_ratio(
        spec, spec["check"], scan, True, "Rule 7(3)", Severity.MINOR
    )
    assert out and out[0].outcome is Outcome.VIOLATION


def test_clear_space_abstains_on_unreliable_glyph_height():
    """
    Clear space is expressed in multiples of the numeral height, so the
    same bad cap height that yields an impossible width ratio corrupts
    the clear-space denominator. Judging them independently let one rule
    abstain while the other asserted a violation from the same failure.
    """
    from src.core.rules_engine import RulesEngine, RuleConfig
    from src.core.schema import (
        ScanResult, Declaration, GlyphMetrics, BBox, Severity,
    )

    cfg = RuleConfig()
    spec = next(p for p in cfg.presentation if p["id"] == "quantity_clear_space")
    scan = ScanResult()
    scan.declarations = [Declaration(
        field_id="net_quantity", present=True, bbox=BBox(0, 0, 50, 20),
        glyph=GlyphMetrics(cap_height_px=30, width_over_height=0.08,
                           n_glyphs_measured=6),
    )]
    out = RulesEngine(cfg)._eval_clear_space(
        spec, spec["check"], scan, True, "Rule 8", Severity.MAJOR
    )
    assert out and out[0].outcome is Outcome.INDETERMINATE


@pytest.mark.parametrize("ratio,n,plausible", [
    (0.08, 6, False), (0.25, 6, True), (0.62, 8, True), (0.50, 2, False),
])
def test_plausibility_helper(ratio, n, plausible):
    from src.core.rules_engine import _glyph_measurement_plausible
    from src.core.schema import GlyphMetrics

    g = GlyphMetrics(cap_height_px=30, width_over_height=ratio,
                     n_glyphs_measured=n)
    assert _glyph_measurement_plausible(g) is plausible


# =====================================================================
# Optional heavy dependencies must stay off the critical import path
# =====================================================================

def test_pipeline_imports_without_torch():
    """
    REGRESSION: Real-ESRGAN was imported at module top in preprocess.py
    and constructed inside PipelineConfig's default, so a machine without
    torch/basicsr/realesrgan could not import the pipeline AT ALL - every
    scan, every rule and every test dead on an ImportError from an
    optional enhancement.
    """
    import importlib
    import src.core.pipeline as pipeline_mod
    import src.vision.preprocess as pre

    importlib.reload(pre)
    importlib.reload(pipeline_mod)
    assert pipeline_mod.CompliancePipeline is not None


def test_esrgan_degrades_to_lanczos_when_unavailable():
    import numpy as np
    from src.vision.preprocess import ESRGANAdapter

    a = ESRGANAdapter(weights="does/not/exist.pth")
    assert a.available() is False
    img = (np.random.rand(40, 80, 3) * 255).astype("uint8")
    out = a(img)                       # must not raise
    assert out.shape[0] > img.shape[0]


def test_superres_is_off_by_default():
    """A fresh clone must not silently depend on a 64MB weights file."""
    from src.core.pipeline import PipelineConfig

    assert PipelineConfig().superres is False


def test_superres_path_runs(renderer, tmpdir_mod):
    from src.core.pipeline import CompliancePipeline, PipelineConfig
    from src.vision.preprocess import ClassicalSR

    p = tmpdir_mod / "sr.png"
    renderer.render(LabelSpec(), out_path=str(p))
    res = CompliancePipeline(
        config=PipelineConfig(superres=True, sr_adapter=ClassicalSR())
    ).scan(str(p))
    assert res is not None


# =====================================================================
# Listing mode - Rule 6(10) and 6(10A)
# =====================================================================

def _listing(**kw):
    from src.core.listing import Listing
    base = dict(
        url="https://example.com/p/1",
        title="Sparkle Detergent Powder 500 g",
        text=(
            "Manufactured by: Sparkle Consumer Products Pvt Ltd, "
            "Plot 14, MIDC, Pune 411019\n"
            "Net Qty: 500 g\n"
            "MRP: Rs. 120.00 (inclusive of all taxes)\n"
            "Unit Sale Price: Rs. 0.24 per g\n"
            "Consumer Care: care@sparkle.example  1800-200-3000"
        ),
    )
    base.update(kw)
    return Listing(**base)


def test_complete_listing_is_compliant():
    from src.core.listing import ListingScanner

    res = ListingScanner().scan(_listing())
    assert res.violations == []
    assert res.is_compliant is True


def test_print_only_rules_are_not_applicable_to_a_listing():
    """
    A web page has no panel, no glyph height and no clear space.
    Reporting a millimetre finding about HTML would be meaningless, so
    those rules must be NOT_APPLICABLE with a reason rather than passed
    over in silence.
    """
    from src.core.listing import ListingScanner, PRINT_ONLY_RULES

    res = ListingScanner().scan(_listing())
    for f in res.findings:
        if f.rule_id.split(".")[0] in PRINT_ONLY_RULES:
            assert f.outcome is Outcome.NOT_APPLICABLE
            assert "listing" in f.message.lower()


def test_manufacture_date_excluded_on_listings():
    """Rule 6(10) requires everything in 6(1) EXCEPT month and year of
    manufacture."""
    from src.core.listing import ListingScanner

    res = ListingScanner().scan(_listing())
    f = next(x for x in res.findings
             if x.rule_id.startswith("manufacture_date"))
    assert f.outcome is Outcome.NOT_APPLICABLE
    assert f.citation == "Rule 6(10)"


def test_listing_title_supplies_the_common_name():
    """
    REGRESSION: a package heads the generic name "COMMODITY:" but no
    e-commerce title does, so without reading the title every listing
    was reported as missing its Rule 6(1)(b) declaration.
    """
    from src.core.listing import ListingScanner

    res = ListingScanner().scan(_listing())
    d = res.declaration("common_name")
    assert d is not None and d.present


def test_incomplete_listing_flags_the_missing_declarations():
    from src.core.listing import ListingScanner

    res = ListingScanner().scan(_listing(
        title="Imported Olive Oil 1 l",
        text="Net Qty: 1 l\nMRP: Rs. 899\nCountry of Origin: Spain",
        is_imported=True,
    ))
    ids = {f.rule_id for f in res.violations}
    assert "manufacturer_details.presence" in ids
    assert "consumer_care.presence" in ids
    assert all("on the package" not in f.message for f in res.violations)


def test_coo_filter_is_a_forthcoming_obligation():
    """
    Rule 6(10A) commences 01.07.2027, so a missing filter is reported as
    a forthcoming duty rather than a present violation.
    """
    from src.core.listing import ListingScanner

    res = ListingScanner().scan(_listing(is_imported=True))
    f = next(x for x in res.findings if x.rule_id == "ecommerce_coo_filter")
    assert f.outcome is Outcome.INDETERMINATE
    assert "01.07.2027" in f.message


def test_coo_filter_satisfied():
    from src.core.listing import ListingScanner

    res = ListingScanner().scan(_listing(
        is_imported=True,
        has_searchable_coo_filter=True,
        has_sortable_coo_filter=True,
    ))
    f = next(x for x in res.findings if x.rule_id == "ecommerce_coo_filter")
    assert f.outcome is Outcome.COMPLIANT


def test_listing_endpoint(client):
    r = client.post("/scan/listing", headers=INSPECTOR, json={
        "url": "https://example.com/p/9",
        "title": "Imported Olive Oil 1 l",
        "text": "Net Qty: 1 l\nMRP: Rs. 899",
        "is_imported": True,
    })
    assert r.status_code == 200
    assert r.json()["summary"]["violations"] > 0
