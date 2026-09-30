"""
Synthetic packaged-commodity label generator.

WHY THIS IS THE MOST VALUABLE FILE IN THE REPO
----------------------------------------------
The hardest claim your system makes is "this numeral is 2.1 mm tall".
To validate that claim you need images where you KNOW the true height in
millimetres. Measuring hundreds of real packages with callipers is not
feasible in a hackathon. So generate them.

This module renders labels at a known DPI, sizing each font so that the
digit cap height is EXACTLY the requested millimetre value, and stamps a
real ArUco calibration marker of known physical size into the frame.
That gives you:

  * unlimited perfectly-labelled data for the measurement module;
  * a ground-truth mm value per field, so mean absolute error is a real
    number and not a vibe;
  * deliberately non-compliant variants, so precision and recall on
    violation detection are measurable too;
  * an end-to-end pipeline test that runs offline with no ML dependency.

Fine-tune on real photographs afterwards - synthetic data alone will not
survive foil glare and curved bottles. But it lets you build, measure and
debug the whole measurement chain on day zero, and it is the slide most
teams will not have.

FONT SIZING, PRECISELY
----------------------
PIL font sizes are nominal and do not map linearly to cap height across
typefaces. So we binary-search the font size until the rendered cap
height of "0123456789" matches the target pixel height. The result is
exact to the pixel, and we record the residual error in the manifest so
you never overstate your ground truth.

DEVANAGARI
----------
Real Indian labels are bilingual and the current container has no
Devanagari font installed, so this generator emits Latin script only.
Install Noto Sans Devanagari and add a bilingual variant before you use
this to benchmark a multilingual OCR claim - otherwise you will report a
CER that does not reflect the packs you will actually be handed.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..vision.calibration import ARUCO_DICTS

MM_PER_INCH = 25.4

FONT_CANDIDATES = [
    # Windows / Matplotlib - the team develops on Windows, and without
    # this the generator raises "No usable TrueType font found" there.
    str(Path.home() / "AppData/Roaming/Python/Python313/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"),
    "C:/Windows/Fonts/arial.ttf",

    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


# =====================================================================
# Ground truth records
# =====================================================================

@dataclass
class FieldGroundTruth:
    field_id: str
    text: str
    # Pixel box in the FINAL rendered image.
    x: float
    y: float
    w: float
    h: float
    # The number that matters: true cap height of the digits/caps in mm.
    cap_height_mm: float
    cap_height_px: float
    cap_height_residual_px: float   # sizing error from the binary search
    font_path: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LabelGroundTruth:
    image_path: str
    px_per_mm: float
    dpi: int
    panel_width_mm: float
    panel_height_mm: float
    pdp_area_cm2: float
    marker_size_mm: float
    marker_id: int
    fields: list[FieldGroundTruth] = field(default_factory=list)
    # Violations deliberately injected, by rule id. Your evaluation
    # harness compares detected violations against exactly this set.
    injected_violations: list[str] = field(default_factory=list)
    # Context the image cannot carry (e.g. whether the food is actually
    # GM). Supplied to the pipeline at scan time, as an inspector would.
    hints: dict = field(default_factory=dict)
    commodity_category: str = "food_packaged"
    package_class: str = "retail"
    is_embossed: bool = False
    net_quantity_value: Optional[float] = None
    net_quantity_unit: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fields"] = [f.to_dict() for f in self.fields]
        return d


# =====================================================================
# Font sizing
# =====================================================================

def _cap_height_px(font: ImageFont.FreeTypeFont, sample: str = "0123456789") -> float:
    """
    True inked height of upright digits, measured from a render.

    Not font.size, not getbbox on a word with descenders - an actual
    measurement of the glyph mask, which is the same definition the
    measurement module uses. Using the same definition on both sides is
    the whole point: otherwise you are measuring your own inconsistency.
    """
    mask = font.getmask(sample, mode="L")
    arr = np.array(mask, dtype=np.uint8).reshape(mask.size[1], mask.size[0])
    rows = np.where(arr.max(axis=1) > 0)[0]
    return float(rows[-1] - rows[0] + 1) if len(rows) else 0.0


def font_for_cap_height(
    font_path: str,
    target_cap_px: float,
    lo: int = 4,
    hi: int = 400,
) -> tuple[ImageFont.FreeTypeFont, float]:
    """
    Binary-search the nominal font size whose digit cap height matches
    `target_cap_px`. Returns the font and the residual error in pixels.
    """
    best_font, best_err = None, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(font_path, mid)
        cap = _cap_height_px(f)
        err = cap - target_cap_px
        if abs(err) < best_err:
            best_font, best_err = f, abs(err)
        if cap < target_cap_px:
            lo = mid + 1
        elif cap > target_cap_px:
            hi = mid - 1
        else:
            return f, 0.0
    return best_font, best_err


# =====================================================================
# Label specification
# =====================================================================

@dataclass
class LabelSpec:
    """One package's declared content and layout intent."""
    brand: str = "SPARKLE"
    common_name: str = "Detergent Powder"
    manufacturer: str = "Sparkle Consumer Products Pvt Ltd"
    manufacturer_address: str = "Plot 14, MIDC Industrial Area, Pune 411018"
    net_quantity_value: float = 500.0
    net_quantity_unit: str = "g"
    mrp: float = 120.0
    mrp_qualifier: str = "(inclusive of all taxes)"
    mfg_month: int = 3
    mfg_year: int = 2026
    consumer_care: str = "Consumer Care: care@sparkle.example  1800-200-3000"
    # Default to a COMPLIANT baseline label. If this is None the
    # baseline scan always reports a unit-sale-price violation, which
    # pollutes every evaluation run with a constant false positive and
    # makes injected-violation precision impossible to read.
    unit_sale_price: Optional[float] = 0.24   # 120.00 / 500 g (was 24.00 "per g": wrong arithmetic)
    # A real EAN-13 (valid check digit) so identity can be definitive.
    barcode: Optional[str] = "8901234567890"
    # Rule 6(7): GM food must bear "GM" at the TOP of the panel.
    # EAN-13 GTIN. Rule 6(4A) expressly permits a barcode/GTIN/QR in
    # addition to the mandatory declarations, and it is the only
    # definitive way to decide that two packages are the same commodity
    # for the Rule 18(2A) dual-MRP comparison.
    gtin: Optional[str] = "8901234567894"
    genetically_modified: bool = False
    gm_mark: bool = True           # whether the mark is drawn at all
    gm_at_top: bool = True         # whether it is drawn at the top
    country_of_origin: Optional[str] = None
    commodity_category: str = "detergent_household"

    # Presentation
    panel_width_mm: float = 90.0
    panel_height_mm: float = 130.0
    quantity_cap_mm: float = 3.0
    price_cap_mm: float = 3.0
    body_cap_mm: float = 1.8
    background: tuple[int, int, int] = (245, 242, 235)
    text_colour: tuple[int, int, int] = (20, 20, 24)


# =====================================================================
# Violation injection
# =====================================================================

VIOLATION_KINDS = [
    "numeral_height",          # quantity numerals below the minimum height
    "missing_mrp",             # no price declaration at all
    "missing_consumer_care",
    "missing_manufacturer",
    "missing_date",
    "mrp_no_tax_qualifier",    # price present, prescribed wording absent
    "banned_phrase",           # "approx." attached to the quantity
    "quantity_clear_space",    # other print crowded against the quantity
    "glyph_aspect_ratio",      # horizontally condensed numerals
    "sticker_over_mrp",        # pasted label concealing the printed MRP
    "sticker_higher_price",    # sticker revises the price UPWARD
    "gm_not_declared",         # GM food with no 'GM' mark
    "gm_not_at_top",           # 'GM' present but not at the top of the PDP
]

# Kinds grouped by WHAT THEY PERTURB. At most one per family, and
# sticker kinds are never combined with layout kinds.
#
# Why: injecting two kinds that touch the same geometry produces
# violations the manifest does not record. A pasted sticker adds text
# and shifts the panel, which genuinely crowds the quantity declaration
# and skews glyph ratios - so the system correctly reports
# quantity_clear_space, the manifest says only "sticker_over_mrp" was
# injected, and a correct detection is scored as a false positive.
# Precision fell from 1.00 to 0.65 this way, entirely inside the
# measuring instrument.
KIND_FAMILIES = {
    "content": [
        "missing_mrp", "missing_consumer_care", "missing_manufacturer",
        "missing_date", "mrp_no_tax_qualifier", "banned_phrase",
    ],
    "layout": ["numeral_height", "quantity_clear_space", "glyph_aspect_ratio"],
    "sticker": ["sticker_over_mrp", "sticker_higher_price"],
    "gm": ["gm_not_declared", "gm_not_at_top"],
}

# Families that must not appear together in one label.
INCOMPATIBLE_FAMILIES = {frozenset({"sticker", "layout"})}

# Pairs of individual kinds that cannot coexist coherently. A sticker
# over the MRP needs an MRP to cover: combine it with missing_mrp and
# the renderer draws no sticker at all, while the manifest still claims
# one was injected - scoring a guaranteed false negative against a label
# that never showed the violation.
INCOMPATIBLE_KINDS = {
    frozenset({"sticker_over_mrp", "missing_mrp"}),
    frozenset({"sticker_higher_price", "missing_mrp"}),
}

# Kinds that require the inspector to supply context the image cannot
# carry. The harness must pass these through or the finding comes back
# NOT_APPLICABLE and scores as a miss.
KIND_HINTS = {
    "gm_not_declared": {"genetically_modified_food": True},
    "gm_not_at_top": {"genetically_modified_food": True},
}

# Renderer-level effects that are legitimate and must NOT be flagged.
# A detector that fires on these is useless in the field.
COMPLIANT_VARIANTS = [
    "sticker_lower_price",     # permitted downward revision, original visible
]


def apply_violation(spec: LabelSpec, kind: str) -> LabelSpec:
    """Return a copy of the spec mutated to be non-compliant in one way."""
    import copy
    s = copy.deepcopy(spec)

    if kind == "numeral_height":
        # 1.8mm: comfortably under the 2.5mm threshold that applies to a
        # typical 100-500 cm2 panel, but still legible.
        #
        # This was 0.9mm, which is not how real violations look. At that
        # size OCR cannot read the line at all, so the system reported
        # "net quantity declaration missing" instead of "numerals below
        # the minimum height" - the wrong rule, the wrong evidence, and
        # a false positive on a declaration that is plainly present. Real
        # undersized print is marginal, not invisible; an inspector can
        # still read it, which is exactly why the measurement matters.
        s.quantity_cap_mm = 1.8
    elif kind == "missing_mrp":
        s.mrp = None
    elif kind == "missing_consumer_care":
        s.consumer_care = None
    elif kind == "missing_manufacturer":
        s.manufacturer = None
        s.manufacturer_address = None
    elif kind == "missing_date":
        s.mfg_month = None
        s.mfg_year = None
    elif kind == "mrp_no_tax_qualifier":
        s.mrp_qualifier = ""
    elif kind == "banned_phrase":
        s.net_quantity_unit = f"{s.net_quantity_unit} (approx.)"
    elif kind == "gm_not_declared":
        s.genetically_modified = True
        s.gm_mark = False
    elif kind == "gm_not_at_top":
        s.genetically_modified = True
        s.gm_mark = True
        s.gm_at_top = False
    # quantity_clear_space and glyph_aspect_ratio are layout effects,
    # handled inside the renderer.
    return s


# =====================================================================
# Renderer
# =====================================================================

class LabelRenderer:

    def __init__(self, dpi: int = 300, font_path: Optional[str] = None,
                 seed: Optional[int] = None):
        self.dpi = dpi
        self.px_per_mm = dpi / MM_PER_INCH
        self.font_path = font_path or self._pick_font()
        self.rng = random.Random(seed)
        # Separate NumPy generator, seeded from the same value so image
        # noise is reproducible alongside every other random choice.
        self._np_rng = np.random.default_rng(seed)

    @staticmethod
    def _pick_font() -> str:
        for p in FONT_CANDIDATES:
            if Path(p).exists():
                return p
        raise RuntimeError(
            "No usable TrueType font found. Install fonts-dejavu-core."
        )

    def mm(self, v: float) -> float:
        return v * self.px_per_mm

    # -----------------------------------------------------------------
    def render(
        self,
        spec: LabelSpec,
        out_path: str,
        violations: Optional[list[str]] = None,
        marker_size_mm: float = 25.0,
        marker_id: int = 0,
        margin_mm: float = 12.0,
        add_noise: bool = False,
        add_perspective: bool = False,
    ) -> LabelGroundTruth:
        violations = violations or []
        # Every line of text drawn, in draw order: {"text", "box"}. This
        # is what SyntheticOCR returns - the test oracle for everything
        # downstream of OCR, at the same LINE granularity as PaddleOCR.
        self._drawn: list[dict] = []

        panel_w = int(round(self.mm(spec.panel_width_mm)))
        panel_h = int(round(self.mm(spec.panel_height_mm)))
        margin = int(round(self.mm(margin_mm)))
        marker_px = int(round(self.mm(marker_size_mm)))

        # Canvas: panel plus a margin band that hosts the marker.
        canvas_w = panel_w + 2 * margin
        canvas_h = panel_h + 2 * margin + marker_px + margin

        img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [margin, margin, margin + panel_w, margin + panel_h],
            fill=spec.background, outline=(180, 178, 170), width=2,
        )

        gt_fields: list[FieldGroundTruth] = []
        pad = int(round(self.mm(5)))
        x = margin + pad
        y = margin + pad
        content_w = panel_w - 2 * pad

        # ---- GM mark (Rule 6(7): at the TOP of the panel) -----------
        if spec.genetically_modified and spec.gm_mark and spec.gm_at_top:
            y = self._draw_line(
                draw, "GM", x, y, cap_mm=3.0, colour=(20, 90, 20),
                bold=True, gt=None,
            )
            y += int(self.mm(2))

        # ---- brand (decorative, not a legal declaration) ------------
        y = self._draw_line(
            draw, spec.brand, x, y, cap_mm=6.0, colour=(60, 70, 120),
            bold=True, gt=None,
        )
        y += int(self.mm(3))

        # ---- common name --------------------------------------------
        if spec.common_name:
            y = self._draw_line(
                draw, spec.common_name, x, y, cap_mm=2.6,
                colour=spec.text_colour, gt=(gt_fields, "common_name"),
            )
            y += int(self.mm(4))

        # ---- net quantity -------------------------------------------
        qty_cap = spec.quantity_cap_mm
        qty_text = f"Net Qty {_fmt_num(spec.net_quantity_value)} {spec.net_quantity_unit}"
        condense = 0.35 if "glyph_aspect_ratio" in violations else None

        qty_y = y
        y = self._draw_line(
            draw, qty_text, x, y, cap_mm=qty_cap, colour=spec.text_colour,
            bold=True, gt=(gt_fields, "net_quantity"), condense=condense,
        )

        # Rule 8 violation: crowd other print into the clear space.
        if "quantity_clear_space" in violations:
            crowd_cap = qty_cap * 0.7
            self._draw_line(
                draw, "BEST VALUE", x + int(self.mm(qty_cap * 0.3)), y,
                cap_mm=crowd_cap, colour=(140, 40, 40), gt=None,
            )
            y += int(self.mm(crowd_cap * 2.0))
        else:
            y += int(self.mm(qty_cap * 1.6))   # generous clear space

        # ---- MRP -----------------------------------------------------
        # Remember where the printed price landed so a sticker can be
        # placed either over it (unlawful) or beside it (lawful).
        sticker_strip_h = int(self.mm(9))
        mrp_box = None
        if spec.mrp is not None:
            mrp_y0 = y
            mrp_text = f"MRP Rs. {spec.mrp:.2f} {spec.mrp_qualifier}".strip()
            y = self._draw_wrapped(
                draw, mrp_text, x, y, content_w, cap_mm=spec.price_cap_mm,
                colour=spec.text_colour, bold=True,
                gt=(gt_fields, "retail_sale_price"),
            )
            mrp_box = (x, mrp_y0, content_w, max(1, y - mrp_y0))
            y += int(self.mm(2.5))
            # Reserve room for a revised-price sticker so it can be drawn
            # later without landing on another declaration.
            if any(k in violations for k in
                   ("sticker_higher_price", "sticker_lower_price")):
                y += sticker_strip_h
            # Reserve a clear band for a revised-price sticker so it can
            # sit below the MRP without covering anything. Placing it
            # over the MRP's own row instead just moved the problem: it
            # then obscured the "(inclusive of all taxes)" qualifier and
            # produced false required_phrasing violations. A sticker has
            # to go somewhere, so the layout must make room for it.
            if any(k in (violations or []) for k in
                   ("sticker_lower_price", "sticker_higher_price")):
                y += int(self.mm(spec.price_cap_mm * 2.2 + 6))

        # ---- unit sale price ----------------------------------------
        if spec.unit_sale_price is not None:
            y = self._draw_line(
                draw,
                f"Unit Sale Price: Rs. {spec.unit_sale_price:.2f} per {spec.net_quantity_unit}",
                x, y, cap_mm=spec.body_cap_mm, colour=spec.text_colour,
                gt=(gt_fields, "unit_sale_price"),
            )
            y += int(self.mm(2))

        # ---- manufacture date ---------------------------------------
        if spec.mfg_month and spec.mfg_year:
            y = self._draw_line(
                draw, f"Mfg: {spec.mfg_month:02d}/{spec.mfg_year}",
                x, y, cap_mm=spec.body_cap_mm, colour=spec.text_colour,
                gt=(gt_fields, "manufacture_date"),
            )
            y += int(self.mm(2))

        # ---- manufacturer -------------------------------------------
        if spec.manufacturer:
            y = self._draw_wrapped(
                draw,
                f"Manufactured by: {spec.manufacturer}, {spec.manufacturer_address}",
                x, y, content_w, cap_mm=spec.body_cap_mm,
                colour=spec.text_colour, gt=(gt_fields, "manufacturer_details"),
            )
            y += int(self.mm(2))

        # ---- consumer care ------------------------------------------
        if spec.consumer_care:
            y = self._draw_wrapped(
                draw, spec.consumer_care, x, y, content_w,
                cap_mm=spec.body_cap_mm, colour=spec.text_colour,
                gt=(gt_fields, "consumer_care"),
            )
            y += int(self.mm(2))

        # ---- country of origin --------------------------------------
        if spec.country_of_origin:
            y = self._draw_line(
                draw, f"Country of Origin: {spec.country_of_origin}",
                x, y, cap_mm=spec.body_cap_mm, colour=spec.text_colour,
                gt=(gt_fields, "country_of_origin"),
            )

        # ---- GM mark in the wrong place -----------------------------
        if spec.genetically_modified and spec.gm_mark and not spec.gm_at_top:
            self._draw_line(
                draw, "GM", x, margin + panel_h - int(self.mm(10)),
                cap_mm=3.0, colour=(20, 90, 20), bold=True, gt=None,
            )

        # ---- barcode (Rule 6(4A) permits it) ------------------------
        if spec.gtin:
            try:
                import barcode as _bc
                from barcode.writer import ImageWriter

                ean = _bc.get("ean13", spec.gtin[:12], writer=ImageWriter())
                bimg = ean.render(writer_options={
                    "module_height": 8.0, "font_size": 6,
                    "text_distance": 2.0, "quiet_zone": 2.0, "dpi": self.dpi,
                }).convert("RGB")
                bw_target = int(content_w * 0.55)
                bh_target = int(bimg.height * bw_target / bimg.width)
                bimg = bimg.resize((bw_target, bh_target), Image.LANCZOS)
                by_pos = margin + panel_h - bh_target - int(self.mm(4))
                img.paste(bimg, (x, by_pos))
            except Exception:
                pass

        # ---- pasted stickers (Rule 6(3)) ----------------------------
        # Sized to the MRP line and no larger. The first version padded
        # the sticker by 2mm on every side and, for the lawful-revision
        # cases, dropped it BELOW the MRP - straight on top of the
        # manufacture date, manufacturer address and consumer care
        # block. Those declarations then genuinely vanished from the
        # image while the manifest still listed them as present, so the
        # system correctly reported them missing and scored three false
        # positives per label. The injected violation must perturb the
        # thing it names and nothing else.
        sticker_kind = next(
            (k for k in violations
             if k in ("sticker_over_mrp", "sticker_higher_price",
                      "sticker_lower_price")),
            None,
        )
        if sticker_kind and mrp_box is not None:
            bx, by, bw, bh = mrp_box
            new_price = (
                spec.mrp if sticker_kind == "sticker_over_mrp"
                else (spec.mrp or 0) + 25.0 if sticker_kind == "sticker_higher_price"
                else max(1.0, (spec.mrp or 0) - 25.0)
            )
            if sticker_kind == "sticker_over_mrp":
                # Exactly the MRP box: covers the printed price without
                # touching the lines above or below it.
                sx, sy, sw, sh = bx, by, bw, bh
            else:
                # Lawful revisions sit in the reserved strip that was
                # left free after the MRP line, so nothing is occluded.
                sx, sy = bx, by + bh
                sw, sh = bw, sticker_strip_h
            self._drop_occluded((sx, sy, sw + 3, sh + 3))
            draw.rectangle([sx + 3, sy + 3, sx + sw + 3, sy + sh + 3],
                           fill=(205, 203, 198))
            draw.rectangle([sx, sy, sx + sw, sy + sh],
                           fill=(238, 236, 228), outline=(190, 188, 182))
            self._draw_line(
                draw, f"MRP Rs. {new_price:.2f} {spec.mrp_qualifier}".strip(),
                sx + int(self.mm(2)), sy + int(self.mm(1)),
                cap_mm=2.6, colour=spec.text_colour, gt=None,
            )

        # ---- ArUco calibration marker -------------------------------
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS["ARUCO_4X4_50"])
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        my = margin + panel_h + margin
        mx = margin
        arr[my:my + marker_px, mx:mx + marker_px] = marker_bgr

        M = None
        if add_perspective:
            arr, gt_fields, M = self._apply_perspective(arr, gt_fields, return_matrix=True)
        if add_noise:
            arr = self._apply_noise(arr)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, arr)
        self._write_line_manifest(out_path, M)

        pdp_area_cm2 = (spec.panel_width_mm / 10.0) * (spec.panel_height_mm / 10.0)

        return LabelGroundTruth(
            image_path=str(out_path),
            px_per_mm=self.px_per_mm,
            dpi=self.dpi,
            panel_width_mm=spec.panel_width_mm,
            panel_height_mm=spec.panel_height_mm,
            pdp_area_cm2=pdp_area_cm2,
            marker_size_mm=marker_size_mm,
            marker_id=marker_id,
            fields=gt_fields,
            injected_violations=list(violations),
            hints={k: v for kind in violations
                   for k, v in KIND_HINTS.get(kind, {}).items()},
            commodity_category=spec.commodity_category,
            net_quantity_value=spec.net_quantity_value,
            net_quantity_unit=(spec.net_quantity_unit or "").split()[0] or None,
        )

    # -- drawing primitives -------------------------------------------

    def _draw_line(
        self, draw, text, x, y, cap_mm, colour, bold=False, gt=None,
        condense: Optional[float] = None,
    ) -> int:
        if not text:
            return y
        path = self.font_path
        if bold:
            b = path.replace("-Regular", "-Bold").replace("Sans.ttf", "Sans-Bold.ttf")
            if Path(b).exists():
                path = b

        target_px = self.mm(cap_mm)
        font, residual = font_for_cap_height(path, target_px)
        actual_cap = _cap_height_px(font)

        if condense is not None:
            # Render then squeeze horizontally, to produce a real
            # width/height ratio violation rather than a fake one.
            #
            # `condense` is a DIRECT horizontal scale factor: 0.35 means
            # the glyphs end up 35% of their natural width, at unchanged
            # height. It is deliberately NOT the target width/height
            # ratio - a font's natural ratio varies per face and per
            # character, so a "target ratio" parameter cannot be honoured
            # exactly and silently under-condenses. The previous form
            # (`condense / 0.5`) meant an injected 0.28 produced a
            # measured ratio of 0.41, comfortably ABOVE the one-third
            # threshold, so the evaluation harness scored a miss on a
            # label that was never actually non-compliant.
            #
            # Typical DejaVu digits sit near 0.75 width/height, so a
            # scale of 0.35 lands around 0.26 - unambiguously below the
            # one-third minimum.
            tmp_w = int(draw.textlength(text, font=font)) + 4
            tmp_h = int(actual_cap * 2.2) + 4
            tmp = Image.new("RGBA", (max(tmp_w, 1), max(tmp_h, 1)), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((0, 0), text, font=font, fill=colour + (255,))
            squeezed = tmp.resize(
                (max(1, int(tmp_w * condense)), tmp_h), Image.LANCZOS
            )
            draw._image.paste(squeezed, (int(x), int(y)), squeezed)
            width = squeezed.width
            ink = squeezed.getbbox()
            if ink:
                self._record(text, x + ink[0], y + ink[1], ink[2] - ink[0], ink[3] - ink[1])
        else:
            draw.text((x, y), text, font=font, fill=colour)
            width = int(draw.textlength(text, font=font))
            self._record_text(draw, text, x, y, font)

        ascent, descent = font.getmetrics()
        line_h = ascent + descent

        if gt is not None:
            gt_list, field_id = gt
            gt_list.append(
                FieldGroundTruth(
                    field_id=field_id,
                    text=text,
                    x=float(x), y=float(y),
                    w=float(width), h=float(line_h),
                    cap_height_mm=float(actual_cap / self.px_per_mm),
                    cap_height_px=float(actual_cap),
                    cap_height_residual_px=float(residual),
                    font_path=path,
                )
            )
        return int(y + line_h)

    def _draw_wrapped(
        self, draw, text, x, y, max_w, cap_mm, colour, bold=False, gt=None
    ) -> int:
        if not text:
            return y
        path = self.font_path
        if bold:
            b = path.replace("-Regular", "-Bold").replace("Sans.ttf", "Sans-Bold.ttf")
            if Path(b).exists():
                path = b
        font, residual = font_for_cap_height(path, self.mm(cap_mm))
        actual_cap = _cap_height_px(font)

        words, lines, cur = text.split(), [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if draw.textlength(trial, font=font) <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        y0, max_line_w = y, 0
        for ln in lines:
            draw.text((x, y), ln, font=font, fill=colour)
            self._record_text(draw, ln, x, y, font)
            max_line_w = max(max_line_w, int(draw.textlength(ln, font=font)))
            y += line_h

        if gt is not None:
            gt_list, field_id = gt
            gt_list.append(
                FieldGroundTruth(
                    field_id=field_id, text=text,
                    x=float(x), y=float(y0),
                    w=float(max_line_w), h=float(y - y0),
                    cap_height_mm=float(actual_cap / self.px_per_mm),
                    cap_height_px=float(actual_cap),
                    cap_height_residual_px=float(residual),
                    font_path=path,
                )
            )
        return int(y)

    def _record_text(self, draw, text, x, y, font) -> None:
        l, t, r, b = draw.textbbox((x, y), text, font=font)
        self._record(text, l, t, r - l, b - t)

    def _record(self, text, x, y, w, h) -> None:
        if not hasattr(self, "_drawn"):
            self._drawn = []
        # A little padding, like a text detector's box.
        pad = max(1.0, 0.12 * h)
        self._drawn.append({
            "text": text,
            "box": [float(x - pad), float(y - pad), float(w + 2 * pad), float(h + 2 * pad)],
        })

    def _drop_occluded(self, rect) -> None:
        """Forget lines a pasted sticker now covers - OCR cannot read them."""
        sx, sy, sw, sh = rect
        keep = []
        for ln in self._drawn:
            x, y, w, h = ln["box"]
            ix = max(0.0, min(x + w, sx + sw) - max(x, sx))
            iy = max(0.0, min(y + h, sy + sh) - max(y, sy))
            if (ix * iy) / max(1.0, w * h) < 0.3:
                keep.append(ln)
        self._drawn = keep

    def _write_line_manifest(self, out_path: str, M=None) -> None:
        """
        `<image>.lines.json` beside the image, plus a copy in the
        registry keyed by the saved image's pixels (so a rendered label
        uploaded through the API - which arrives under a temp name - is
        still recognised). Marker corners are detected on the SAVED image
        with the pipeline's own detector, so SyntheticOCR can map these
        boxes into whatever frame the pipeline later hands it.
        """
        from ..vision.ocr.backends import SYNTH_REGISTRY, pixel_key
        from ..vision.calibration import detect_markers

        lines = []
        for ln in self._drawn:
            x, y, w, h = ln["box"]
            poly = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            if M is not None:
                poly = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), M).reshape(-1, 2)
            lines.append({"text": ln["text"], "poly": poly.round(2).tolist()})

        saved = cv2.imread(str(out_path))
        corners = None
        try:
            dets = detect_markers(saved)
            if dets:
                corners = np.float32(max(dets, key=lambda d: d.area_px).corners).tolist()
        except Exception:
            corners = None
        manifest = {"lines": lines, "marker_corners": corners,
                    "shape": list(saved.shape[:2])}
        Path(f"{out_path}.lines.json").write_text(json.dumps(manifest), encoding="utf-8")
        try:
            SYNTH_REGISTRY.mkdir(parents=True, exist_ok=True)
            (SYNTH_REGISTRY / f"{pixel_key(saved)}.json").write_text(json.dumps(manifest), encoding="utf-8")
        except OSError:
            pass

    # -- degradations --------------------------------------------------

    def _apply_noise(self, arr: np.ndarray) -> np.ndarray:
        # Draw from the renderer's OWN generator, not global np.random.
        # LabelRenderer takes a seed and every other random choice
        # honoured it, but noise did not - so `seed=` never actually made
        # a noisy render reproducible. Every noise-based measurement in
        # this project was therefore unrepeatable, and a test pinned to a
        # seed still failed intermittently.
        out = arr.astype(np.float32)
        out += self._np_rng.normal(0, 4.0, out.shape)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return cv2.GaussianBlur(out, (3, 3), 0.6)

    def _apply_perspective(self, arr, gt_fields, return_matrix: bool = False):
        """
        Mild perspective warp, with the ground-truth boxes transformed by
        the same homography so they stay correct.

        Note the honest limitation: after a warp the true cap height in
        millimetres varies across the image, so mm ground truth is only
        exact on the fronto-parallel renders. Use warped images to test
        detection robustness, not measurement accuracy - or rectify to
        the marker plane first, which is what the pipeline does anyway.
        """
        h, w = arr.shape[:2]
        d = 0.04
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [w * self.rng.uniform(0, d), h * self.rng.uniform(0, d)],
            [w * (1 - self.rng.uniform(0, d)), h * self.rng.uniform(0, d)],
            [w * (1 - self.rng.uniform(0, d)), h * (1 - self.rng.uniform(0, d))],
            [w * self.rng.uniform(0, d), h * (1 - self.rng.uniform(0, d))],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(arr, M, (w, h), borderValue=(255, 255, 255))

        new_fields = []
        for f in gt_fields:
            pts = np.float32([
                [f.x, f.y], [f.x + f.w, f.y],
                [f.x + f.w, f.y + f.h], [f.x, f.y + f.h],
            ]).reshape(-1, 1, 2)
            tp = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            nx, ny = float(tp[:, 0].min()), float(tp[:, 1].min())
            nw = float(tp[:, 0].max() - nx)
            nh = float(tp[:, 1].max() - ny)
            new_fields.append(
                FieldGroundTruth(
                    field_id=f.field_id, text=f.text, x=nx, y=ny, w=nw, h=nh,
                    cap_height_mm=f.cap_height_mm, cap_height_px=f.cap_height_px,
                    cap_height_residual_px=f.cap_height_residual_px,
                    font_path=f.font_path,
                )
            )
        if return_matrix:
            return warped, new_fields, M
        return warped, new_fields


def _fmt_num(v: float) -> str:
    return f"{v:.0f}" if float(v).is_integer() else f"{v:g}"


# =====================================================================
# Dataset generation
# =====================================================================

BRANDS = [
    ("SPARKLE", "Detergent Powder", "detergent_household"),
    ("SUNRAY", "Refined Sunflower Oil", "food_packaged"),
    ("VEDA", "Ayurvedic Hair Oil", "cosmetic"),
    ("GRAINWELL", "Basmati Rice", "food_packaged"),
    ("MILKO", "Toned Milk", "food_packaged"),
    ("FRESHA", "Toilet Soap", "soap"),
    ("CRUNCHO", "Potato Chips", "food_packaged"),
    ("AQUAPURE", "Packaged Drinking Water", "beverage_non_alcoholic"),
]

QUANTITIES = [
    (100.0, "g"), (200.0, "g"), (500.0, "g"), (1.0, "kg"),
    (250.0, "ml"), (500.0, "ml"), (1.0, "l"), (5.0, "g"),
]


def generate_dataset(
    out_dir: str,
    n_compliant: int = 20,
    n_violating: int = 20,
    dpi: int = 300,
    seed: int = 42,
    add_noise: bool = True,
) -> str:
    """
    Build a labelled synthetic dataset with a manifest.

    The manifest is the contract your evaluation harness reads. Keep the
    schema stable - when you swap synthetic images for real annotated
    photographs, emit the same manifest format and every metric keeps
    working unchanged.
    """
    rng = random.Random(seed)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    renderer = LabelRenderer(dpi=dpi, seed=seed)

    manifest: list[dict] = []

    def random_spec(i: int) -> LabelSpec:
        brand, common, cat = rng.choice(BRANDS)
        qv, qu = rng.choice(QUANTITIES)
        mrp = round(rng.uniform(20, 900), 2)
        return LabelSpec(
            brand=brand, common_name=common, commodity_category=cat,
            net_quantity_value=qv, net_quantity_unit=qu, mrp=mrp,
            mfg_month=rng.randint(1, 12), mfg_year=rng.choice([2025, 2026]),
            quantity_cap_mm=round(rng.uniform(2.5, 5.0), 2),
            price_cap_mm=round(rng.uniform(2.5, 4.5), 2),
            body_cap_mm=round(rng.uniform(1.4, 2.2), 2),
            panel_width_mm=round(rng.uniform(70, 120), 1),
            panel_height_mm=round(rng.uniform(100, 170), 1),
            unit_sale_price=round(mrp / max(qv, 1) * 100, 2) if rng.random() < 0.5 else None,
        )

    idx = 0
    for _ in range(n_compliant):
        spec = random_spec(idx)
        path = out / "images" / f"label_{idx:04d}_ok.png"
        gt = renderer.render(spec, str(path), violations=[],
                             marker_id=idx % 50, add_noise=add_noise)
        manifest.append(gt.to_dict())
        idx += 1

    for _ in range(n_violating):
        spec = random_spec(idx)
        fams = list(KIND_FAMILIES)
        rng.shuffle(fams)
        chosen: list[str] = []
        used: set[str] = set()
        for fam in fams:
            if len(used) >= rng.randint(1, 2):
                break
            if any(frozenset({fam, u}) in INCOMPATIBLE_FAMILIES for u in used):
                continue
            candidate = rng.choice(KIND_FAMILIES[fam])
            if any(frozenset({candidate, c}) in INCOMPATIBLE_KINDS for c in chosen):
                continue
            chosen.append(candidate)
            used.add(fam)
        # A sticker needs a price to sit on. Injecting "missing_mrp"
        # alongside a sticker kind renders no sticker at all, while the
        # manifest still claims one - a guaranteed false negative
        # manufactured by the generator.
        if "missing_mrp" in chosen:
            chosen = [k for k in chosen if not k.startswith("sticker_")]
        kinds = chosen or [rng.choice(KIND_FAMILIES["content"])]
        mutated = spec
        for k in kinds:
            mutated = apply_violation(mutated, k)
        path = out / "images" / f"label_{idx:04d}_bad.png"
        gt = renderer.render(mutated, str(path), violations=kinds,
                             marker_id=idx % 50, add_noise=add_noise)
        manifest.append(gt.to_dict())
        idx += 1

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generator": "synthetic-labels-v1",
                "dpi": dpi,
                "seed": seed,
                "n_images": len(manifest),
                "caveat": (
                    "Synthetic Latin-script labels. Ground-truth mm values are "
                    "exact only for fronto-parallel renders. Do not report "
                    "OCR accuracy on this set as if it were real packaging."
                ),
                "records": manifest,
            },
            fh, indent=2,
        )
    return str(manifest_path)
