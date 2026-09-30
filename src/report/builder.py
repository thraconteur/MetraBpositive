"""
Compliance report generation.

The PS asks for reports in "PDF and editable formats" with photographs
and supporting evidence attached. This module produces both, plus an
evidence crop per finding.

WHAT MAKES A REPORT USABLE RATHER THAN IMPRESSIVE
-------------------------------------------------
An inspector may have to defend this document. So every finding carries:

    rule citation | measured vs required | evidence crop | confidence
    | the measurement method used | space for a human decision

And the report NEVER states a bare verdict when the system abstained.
"Indeterminate - retake required" is a real outcome and it is printed as
prominently as a violation. A tool that always produces a verdict is a
tool that sometimes produces a confident wrong one, and the first time
that happens in front of an officer, the tool is finished.

The human-decision column is not decoration either. The system proposes;
a person disposes. That is both better law and a much better answer when
a judge asks who is accountable for a wrong flag.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Optional

import cv2

from ..core.schema import Outcome, ScanResult, Severity

OUTCOME_STYLE = {
    Outcome.VIOLATION:       ("#96271F", "#FBF4F3", "VIOLATION"),
    Outcome.COMPLIANT:       ("#2C6349", "#F2F7F4", "COMPLIANT"),
    Outcome.INDETERMINATE:   ("#8A6D1F", "#FCF8EC", "INDETERMINATE"),
    Outcome.UNVERIFIED_RULE: ("#5B5B6B", "#F4F4F7", "RULE NOT VERIFIED"),
    Outcome.NOT_APPLICABLE:  ("#69757E", "#F4F6F6", "NOT APPLICABLE"),
    Outcome.SUPPRESSED:      ("#69757E", "#F4F6F6", "SUPPRESSED"),
}


# ---------------------------------------------------------------------
# Evidence crops
# ---------------------------------------------------------------------

def extract_evidence_crops(
    scan: ScanResult,
    out_dir: str | Path,
    pad: int = 14,
    annotate: bool = True,
) -> dict[str, str]:
    """
    Save one annotated crop per finding that has a bounding box.

    Annotated with the measurement overlay where a measurement was made,
    so the crop shows not just WHERE the finding is but WHAT was measured
    to produce it.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    image = getattr(scan, "analysed_image", None)
    if image is None:
        if not scan.image_path or not Path(scan.image_path).exists():
            return paths
        image = cv2.imread(scan.image_path)
    if image is None:
        return paths

    h, w = image.shape[:2]
    for i, finding in enumerate(scan.findings):
        if finding.evidence_bbox is None:
            continue
        b = finding.evidence_bbox
        x0, y0 = max(0, int(b.x) - pad), max(0, int(b.y) - pad)
        x1, y1 = min(w, int(b.x2) + pad), min(h, int(b.y2) + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        crop = image[y0:y1, x0:x1].copy()
        if annotate:
            colour = (60, 60, 220) if finding.outcome is Outcome.VIOLATION else (70, 150, 70)
            cv2.rectangle(crop, (pad, pad), (crop.shape[1] - pad, crop.shape[0] - pad),
                          colour, 2)

        path = out_dir / f"{scan.scan_id}_{i:02d}_{finding.rule_id.replace('.', '_')}.png"
        cv2.imwrite(str(path), crop)
        finding.evidence_crop_path = str(path)
        paths[finding.rule_id] = str(path)

    return paths


def _img_b64(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).exists():
        return None
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


def _view_b64(path: Optional[str]) -> Optional[str]:
    """The photo for SHOWING in the report: JPEG, bounded size. Embedding the
    full-resolution PNG made one pack report 51 MB."""
    if not path or not Path(path).exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return _img_b64(path)
    h, w = img.shape[:2]
    s_ = min(1.0, 1600.0 / max(h, w)) if h <= 2.5 * w else min(1.0, 900.0 / w)
    if s_ < 1.0:
        img = cv2.resize(img, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


# ---------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------

def build_html_report(
    scan: ScanResult,
    rule_audit: Optional[dict] = None,
    embed_images: bool = True,
) -> str:
    verdict, vcolour, vnote = _verdict(scan)

    rows = []
    order = [Outcome.VIOLATION, Outcome.INDETERMINATE, Outcome.UNVERIFIED_RULE,
             Outcome.COMPLIANT, Outcome.NOT_APPLICABLE, Outcome.SUPPRESSED]
    findings = sorted(scan.findings, key=lambda f: order.index(f.outcome))

    for f in findings:
        colour, bg, label = OUTCOME_STYLE[f.outcome]

        measured = "—"
        if f.measured_value is not None:
            measured = f"{f.measured_value:g}"
            if f.unit:
                measured += f" {f.unit}"
            if f.uncertainty:
                measured += f" ±{f.uncertainty:g}"
        required = "—"
        if f.required_value is not None:
            required = f"{f.required_value:g}" + (f" {f.unit}" if f.unit else "")

        crop_html = ""
        b64 = _view_b64(f.evidence_crop_path) if embed_images else None
        if b64:
            crop_html = f'<img class="crop" src="data:image/jpeg;base64,{b64}" alt="evidence">'

        rows.append(f"""
        <tr>
          <td><span class="badge" style="color:{colour};background:{bg};border-color:{colour}">{label}</span></td>
          <td class="mono">{html.escape(f.citation or '—')}{
            f'<br><span class="muted">{html.escape(f.field_id)}</span>'
            if f.field_id else ''
          }</td>
          <td>{html.escape(f.message)}</td>
          <td class="num">{html.escape(measured)}</td>
          <td class="num">{html.escape(required)}</td>
          <td class="num">{f.confidence:.2f}</td>
          <td>{crop_html}</td>
          <td class="decision"></td>
        </tr>""")

    audit_html = ""
    if rule_audit:
        unverified = rule_audit.get("unverified", [])
        cov = rule_audit.get("coverage", 0) * 100
        if unverified:
            audit_html = f"""
      <div class="audit">
        <b>Rule verification status: {cov:.0f}% of rules verified against the gazette text.</b>
        The following rules are still placeholders and CANNOT produce a violation:
        <span class="mono">{html.escape(', '.join(unverified))}</span>.
        Findings against them are reported as "RULE NOT VERIFIED" and must not be
        treated as compliance decisions.
      </div>"""

    full_b64 = _view_b64(scan.image_path) if embed_images else None
    photo_html = (
        f'<img class="full" src="data:image/jpeg;base64,{full_b64}" alt="scanned package">'
        if full_b64 else "<p class='muted'>Source image not embedded.</p>"
    )

    # -- what OCR / extraction produced --------------------------------
    engines = sorted({sp.source_engine for sp in scan.spans}) if scan.spans else []
    st = scan.ocr_stats or {}
    ocr_line = ", ".join(engines) if engines else "no text read"
    if scan.spans:
        ocr_line += f" · {len(scan.spans)} lines"
    if st.get("cache_hit"):
        ocr_line += " · cached result"
    elif st.get("total_s") is not None:
        ocr_line += f" · {st['total_s']}s"
    if st.get("refined"):
        ocr_line += f" · {st['refined']} weak line(s) re-read"
    if st.get("secondary_replaced"):
        ocr_line += f" · {st['secondary_replaced']} Devanagari line(s)"
    if st.get("vertical_bands"):
        ocr_line += f" · {st['vertical_bands']} rotated band(s) re-read"
    v = st.get("vlm") or {}
    if v:
        ocr_line += (f" · second reader {v.get('model')}: found "
                     f"{len(v.get('found') or [])} of {len(v.get('asked') or [])} missing "
                     f"({len(v.get('confirmed_by_ocr') or [])} confirmed by OCR)")
        if v.get("error"):
            ocr_line += f" - FAILED: {v['error'][:80]}"

    decl_rows = []
    for d in scan.declarations:
        if not d.present:
            continue
        val = "—" if d.value is None else str(d.value)
        if d.unit:
            val += f" {d.unit}"
        notes = "<br>".join(html.escape(n) for n in (d.notes or []))
        src = getattr(d, "source", "ocr")
        src_tag = ("" if src == "ocr" else
                   f"<br><span class='badge' style='color:#8A6D1F;border-color:#8A6D1F'>"
                   f"{'VISION MODEL + OCR' if src == 'vlm+ocr' else 'VISION MODEL ONLY'}</span>")
        decl_rows.append(
            f"<tr><td class='mono'>{html.escape(d.field_id)}{src_tag}</td>"
            f"<td>{html.escape((d.raw_text or '')[:220])}</td>"
            f"<td class='mono'>{html.escape(val[:80])}</td>"
            f"<td class='num'>{d.extraction_confidence:.2f}</td>"
            f"<td class='muted'>{notes}</td></tr>"
        )
    decl_rows = "".join(decl_rows)

    ocr_html = ""
    if scan.spans:
        lines = sorted(scan.spans, key=lambda sp: (round(sp.bbox.y / 12), sp.bbox.x))
        vert_tag = ' <span class="muted">(vertical)</span>'
        items = "".join(
            f"<tr><td class='num'>{sp.confidence:.2f}</td>"
            f"<td>{html.escape(sp.text)}{vert_tag if sp.vertical else ''}</td>"
            f"<td class='mono muted'>{html.escape(sp.source_engine)}</td></tr>"
            for sp in lines
        )
        ocr_html = f"""
<details>
  <summary><b>Raw OCR lines ({len(lines)})</b> <span class="muted">- exactly what the
  engine read. If a finding looks wrong, check here first: a misread line is an OCR
  problem, a correct line with a wrong finding is an extraction or rules problem.</span></summary>
  <table><thead><tr><th>Conf.</th><th>Text</th><th>Engine</th></tr></thead>
  <tbody>{items}</tbody></table>
</details>"""

    cal = scan.calibration
    cal_line = (
        f"{cal.method.value} · {cal.px_per_mm:.3f} px/mm"
        if cal.available else f"not calibrated ({html.escape(cal.notes or 'no marker')})"
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Compliance Report {scan.scan_id}</title>
<style>
  @page {{ size: A4; margin: 14mm; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color:#2A343C; font-size:12px; line-height:1.5; margin:0; }}
  h1 {{ font-size:19px; color:#101820; margin:0 0 2px; }}
  h2 {{ font-size:13px; color:#101820; margin:18px 0 6px;
        border-bottom:1.5px solid #2B3A67; padding-bottom:3px; }}
  .muted {{ color:#69757E; font-size:11px; }}
  .head {{ background:#E7E9F2; border-left:5px solid #2B3A67; padding:10px 14px; }}
  .verdict {{ display:inline-block; padding:6px 14px; border-radius:2px;
              font-weight:700; font-size:14px; color:#fff; }}
  table {{ width:100%; border-collapse:collapse; margin-top:6px; }}
  th {{ text-align:left; font-size:10px; text-transform:uppercase; letter-spacing:.04em;
        color:#69757E; border-bottom:1.5px solid #101820; padding:5px 6px; }}
  td {{ border-bottom:1px solid #E3E7E6; padding:6px; vertical-align:top; }}
  td.num {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  td.decision {{ width:70px; border-left:1px dashed #C9CFCE; }}
  .badge {{ display:inline-block; padding:2px 6px; border:1px solid; border-radius:2px;
            font-size:9.5px; font-weight:700; white-space:nowrap; }}
  .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size:10.5px; }}
  .crop {{ max-width:150px; max-height:56px; border:1px solid #D3D9D7; }}
  .full {{ max-width:250px; border:1px solid #D3D9D7; }}
  .grid {{ display:flex; gap:18px; }}
  .audit {{ background:#F4F4F7; border-left:4px solid #5B5B6B; padding:8px 12px;
            margin:12px 0; font-size:11px; }}
  .kv {{ font-size:11px; }} .kv b {{ color:#101820; }}
  .sign {{ margin-top:22px; border-top:1px solid #C9CFCE; padding-top:8px; font-size:11px; }}
</style></head><body>

<div class="head">
  <h1>Legal Metrology Compliance Report</h1>
  <div class="muted">Legal Metrology (Packaged Commodities) Rules, 2011 ·
    Scan {scan.scan_id} · {scan.created_at[:19].replace('T', ' ')} UTC ·
    Inspector: {html.escape(scan.inspector_id or 'unassigned')}</div>
</div>

<p><span class="verdict" style="background:{vcolour}">{verdict}</span>
   <span class="muted">{html.escape(vnote)}</span></p>
{_advice_html(scan)}

<div class="grid">
  <div style="flex:1">
    <h2>Package</h2>
    <div class="kv">
      <b>Category:</b> {html.escape(scan.context.commodity_category)}<br>
      <b>Class:</b> {html.escape(scan.context.package_class)}<br>
      <b>Geometry:</b> {html.escape(scan.context.geometry.value)}<br>
      <b>PDP area:</b> {f'{scan.context.pdp_area_cm2:.1f} cm²' if scan.context.pdp_area_cm2 else '—'}<br>
      <b>Calibration:</b> {html.escape(cal_line)}<br>
      <b>Image quality:</b> {'acceptable' if scan.image_quality_ok else html.escape(scan.quality_notes)}<br>
      <b>OCR:</b> {html.escape(ocr_line)}
    </div>
  </div>
  <div>{photo_html}</div>
</div>

{audit_html}

<h2>Findings</h2>
<table>
  <thead><tr>
    <th>Outcome</th><th>Rule</th><th>Finding</th><th>Measured</th>
    <th>Required</th><th>Conf.</th><th>Evidence</th><th>Officer</th>
  </tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>

<h2>What was read</h2>
<table>
  <thead><tr><th>Field</th><th>Read as</th><th>Value</th><th>Conf.</th><th>Notes</th></tr></thead>
  <tbody>{decl_rows or '<tr><td colspan="5" class="muted">No declarations extracted.</td></tr>'}</tbody>
</table>

{ocr_html}

<div class="sign">
  <b>Inspecting officer:</b> ________________________ &nbsp;&nbsp;
  <b>Signature:</b> ________________ &nbsp;&nbsp; <b>Date:</b> ____________<br>
  <span class="muted">This report is machine-generated decision support. Every finding
  requires confirmation by an authorised officer before any enforcement action.
  Measured values carry the stated uncertainty. INDETERMINATE means this photo
  could not decide the point (not visible, not read clearly, or within
  measurement tolerance); no violation is asserted for it.</span>
</div>

</body></html>"""


def _advice_html(scan: ScanResult) -> str:
    try:
        from ..core.package import capture_advice

        adv = capture_advice(scan)
    except Exception:
        adv = []
    extra = ""
    if scan.package_photos:
        extra = ("<p class='muted'><b>Package scan:</b> "
                 + html.escape(", ".join(f"photo {i + 1} = {n}" for i, n in enumerate(scan.package_photos)))
                 + ". Each declaration notes the photo it was read from.</p>")
    if not adv:
        return extra
    return extra + "".join(
        f"<p style='background:#FFF6DA;border-left:4px solid #8A6D1F;padding:6px 10px'>"
        f"<b>Next photo:</b> {html.escape(a)}</p>" for a in adv)


def _verdict(scan: ScanResult) -> tuple[str, str, str]:
    if not scan.image_quality_ok:
        return "RETAKE REQUIRED", "#8A6D1F", scan.quality_notes
    state = scan.is_compliant
    if state is False:
        n = len(scan.violations)
        crit = sum(1 for f in scan.violations if f.severity is Severity.CRITICAL)
        return ("NON-COMPLIANT", "#96271F",
                f"{n} violation(s), {crit} critical.")
    if state is True:
        return "COMPLIANT", "#2C6349", "All applicable checks passed."
    mixed = next((f for f in scan.findings if f.rule_id == "package.same_product"), None)
    if mixed is not None:
        return "INDETERMINATE", "#8A6D1F", mixed.message
    return ("INDETERMINATE", "#8A6D1F", _undecided_note(scan))


_MEASURE = ("glyph_aspect_ratio", "numeral_height", "quantity_clear_space", "pdp_")


def _undecided_note(scan: ScanResult) -> str:
    """Say WHY there is no verdict, in the inspector's terms: what to
    photograph, what needs the calibration card, what to check by eye."""
    missing, measure, by_eye = [], 0, 0
    for f in scan.indeterminate:
        if f.rule_id.endswith(".presence"):
            if f.field_id != "common_name":
                missing.append((f.field_id or "").replace("_", " "))
        elif f.rule_id.startswith(_MEASURE):
            measure += 1
        else:
            by_eye += 1
    parts = ["No violation found."]
    if missing:
        where = ("on any photo" if scan.package_photos or scan.coverage_complete
                 else "on this photo - photograph the other sides")
        parts.append(f"Not read {where}: {', '.join(missing)}.")
    if measure:
        parts.append(f"{measure} size/spacing check(s) need the calibration card in the photo.")
    if by_eye:
        parts.append(f"{by_eye} item(s) to confirm by eye (see the findings).")
    if scan.unverified:
        parts.append(f"{len(scan.unverified)} rule(s) unverified.")
    return " ".join(parts)


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------

def save_html(scan: ScanResult, out_path: str | Path, rule_audit=None) -> str:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html_report(scan, rule_audit), encoding="utf-8")
    return str(out)


_FONT_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def _pdf_fonts() -> tuple[str, str]:
    """Regular and bold font names for the PDF. Noto Sans Devanagari (in
    assets/fonts, OFL licence) covers Hindi, Latin, digits and the rupee
    sign; with uharfbuzz installed ReportLab also joins Hindi letters
    correctly (matras, conjuncts). Falls back to Helvetica - which has no
    Hindi and no rupee sign - only if the font files are missing."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping

        if "NotoDeva" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("NotoDeva", str(_FONT_DIR / "NotoSansDevanagari-Regular.ttf")))
            pdfmetrics.registerFont(TTFont("NotoDeva-Bold", str(_FONT_DIR / "NotoSansDevanagari-Bold.ttf")))
            addMapping("NotoDeva", 0, 0, "NotoDeva")
            addMapping("NotoDeva", 1, 0, "NotoDeva-Bold")
            addMapping("NotoDeva", 0, 1, "NotoDeva")
            addMapping("NotoDeva", 1, 1, "NotoDeva-Bold")
        return "NotoDeva", "NotoDeva-Bold"
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "Hindi font not found in %s: the PDF will show no Hindi text.", _FONT_DIR)
        return "Helvetica", "Helvetica-Bold"


def _save_pdf_reportlab(scan: ScanResult, out_path: str | Path, rule_audit=None) -> str:
    # Every text goes through html.escape here. With Helvetica (no Hindi
    # font available) "₹" would print as a black box, so it is swapped for
    # "Rs." in that case only.
    import html as _html_mod

    F, FB = _pdf_fonts()

    class html:  # noqa: N801 - shadows the module inside this function only
        @staticmethod
        def escape(t, quote=True):
            t = str(t)
            if F == "Helvetica":
                t = t.replace("₹", "Rs.")
            return _html_mod.escape(t, quote)

    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle as _PS

    def ParagraphStyle(*a, **k):  # noqa: N802
        # shaping=1: ReportLab joins Devanagari letters (matras, conjuncts)
        # only when asked, even with uharfbuzz installed - without it
        # "अधिकतम" printed as "अधकितम".
        k.setdefault("shaping", 1)
        return _PS(*a, **k)
    from reportlab.lib.units import mm

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName=FB,
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#101820')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName=F,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#69757E')
    )
    section_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName=FB,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#101820'),
        spaceBefore=8,
        spaceAfter=3,
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName=FB,
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#495057')
    )
    table_body = ParagraphStyle(
        'TableBody',
        parent=styles['Normal'],
        fontName=F,
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#2A343C')
    )
    table_body_mono = ParagraphStyle(
        'TableBodyMono',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor('#2A343C')
    )
    badge_style = ParagraphStyle(
        'BadgeStyle',
        parent=styles['Normal'],
        fontName=FB,
        fontSize=6.5,
        leading=8,
        alignment=1,
    )

    verdict, vcolour, vnote = _verdict(scan)
    verdict_hex = vcolour if vcolour.startswith('#') else '#2C6349'

    story = []

    # 1. Header banner
    created_str = (scan.created_at or '')[:19].replace('T', ' ')
    header_table = Table([
        [
            Paragraph("<b>Legal Metrology Compliance Report</b>", title_style),
            Paragraph(f"<b>Scan ID:</b> {html.escape(scan.scan_id or '')}", ParagraphStyle('RightH', parent=subtitle_style, alignment=2, fontName=FB))
        ],
        [
            Paragraph(f"Legal Metrology (Packaged Commodities) Rules, 2011 · Inspector: {html.escape(scan.inspector_id or 'Unassigned')}", subtitle_style),
            Paragraph(f"Date: {created_str} UTC", ParagraphStyle('RightDate', parent=subtitle_style, alignment=2))
        ]
    ], colWidths=[340, 185])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))

    # 2. Verdict Banner
    verdict_badge = Paragraph(f"<font color='white'><b>{verdict}</b></font>", ParagraphStyle('VB', fontName=FB, fontSize=10, leading=12, alignment=1))
    verdict_desc = Paragraph(f"<b>Assessment:</b> {html.escape(vnote or '')}", ParagraphStyle('VD', fontName=F, fontSize=8, leading=10, textColor=colors.HexColor('#2A343C')))
    verdict_table = Table([[verdict_badge, verdict_desc]], colWidths=[120, 405])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor(verdict_hex)),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#F4F6F8')),
        ('ALIGN', (0,0), (0,0), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#DFE3E8')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (1,0), (1,0), 8),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 8))

    # 3. Package Context Box
    pdp_area = f"{scan.context.pdp_area_cm2:.1f} cm²" if scan.context and scan.context.pdp_area_cm2 else "—"
    cal = scan.calibration
    cal_line = f"{cal.method.value} ({cal.px_per_mm:.2f} px/mm)" if cal and cal.available else f"Uncalibrated ({(cal.notes if cal else '') or 'no marker'})"
    quality = "Acceptable" if scan.image_quality_ok else (scan.quality_notes or "Quality issues")

    pkg_data = [
        [
            Paragraph(f"<b>Category:</b> {html.escape((scan.context.commodity_category if scan.context else '') or 'General')}", table_body),
            Paragraph(f"<b>Class:</b> {html.escape((scan.context.package_class if scan.context else '') or 'Standard')}", table_body),
            Paragraph(f"<b>Geometry:</b> {html.escape((scan.context.geometry.value if scan.context and hasattr(scan.context.geometry, 'value') else str(scan.context.geometry if scan.context else '')) or 'Unknown')}", table_body),
        ],
        [
            Paragraph(f"<b>PDP Area:</b> {pdp_area}", table_body),
            Paragraph(f"<b>Calibration:</b> {html.escape(cal_line)}", table_body),
            Paragraph(f"<b>Image Quality:</b> {html.escape(quality)}", table_body),
        ]
    ]
    pkg_table = Table(pkg_data, colWidths=[175, 175, 175])
    pkg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F9FAFB')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#E5E8EC')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(pkg_table)
    story.append(Spacer(1, 8))

    # Audit notice if present
    if rule_audit and rule_audit.get("unverified"):
        unverified = rule_audit.get("unverified", [])
        cov = rule_audit.get("coverage", 0) * 100
        audit_text = (
            f"<b>Rule Verification Notice:</b> {cov:.0f}% of rules verified against gazette text. "
            f"Unverified placeholder rules: {html.escape(', '.join(unverified))}."
        )
        audit_table = Table([[Paragraph(audit_text, table_body)]], colWidths=[525])
        audit_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F4F7')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#5B5B6B')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(audit_table)
        story.append(Spacer(1, 6))

    # 4. Findings Section
    story.append(Paragraph("<b>Compliance Findings &amp; Rule Verifications</b>", section_h2))

    findings_header = [
        Paragraph("Outcome", table_header),
        Paragraph("Rule / Citation", table_header),
        Paragraph("Finding Description", table_header),
        Paragraph("Measured", table_header),
        Paragraph("Required", table_header),
        Paragraph("Conf.", table_header),
        Paragraph("Action", table_header),
    ]
    findings_rows = [findings_header]

    order = [Outcome.VIOLATION, Outcome.INDETERMINATE, Outcome.UNVERIFIED_RULE,
             Outcome.COMPLIANT, Outcome.NOT_APPLICABLE, Outcome.SUPPRESSED]
    sorted_findings = sorted(scan.findings, key=lambda f: order.index(f.outcome) if f.outcome in order else 99)

    for f in sorted_findings:
        colour_hex, bg_hex, label = OUTCOME_STYLE.get(f.outcome, ("#69757E", "#F4F6F6", "UNKNOWN"))
        badge_p = Paragraph(
            f"<font color='{colour_hex}'><b>{label}</b></font>",
            badge_style
        )
        rule_text = html.escape(f.citation or f.rule_id or '—')
        if f.field_id:
            rule_text += f"<br/><font color='#69757E'>{html.escape(f.field_id)}</font>"
        rule_p = Paragraph(rule_text, table_body_mono)

        desc_p = Paragraph(html.escape(f.message or ''), table_body)

        meas = "—"
        if f.measured_value is not None:
            meas = f"{f.measured_value:g}" + (f" {f.unit}" if f.unit else "")
            if f.uncertainty:
                meas += f" ±{f.uncertainty:g}"
        meas_p = Paragraph(html.escape(meas), table_body)

        req = "—"
        if f.required_value is not None:
            req = f"{f.required_value:g}" + (f" {f.unit}" if f.unit else "")
        req_p = Paragraph(html.escape(req), table_body)

        conf_p = Paragraph(f"{f.confidence:.2f}", table_body)
        officer_p = Paragraph("[  ] Pass<br/>[  ] Flag", table_body)

        findings_rows.append([badge_p, rule_p, desc_p, meas_p, req_p, conf_p, officer_p])

    if len(findings_rows) == 1:
        findings_rows.append([
            Paragraph("—", table_body),
            Paragraph("—", table_body),
            Paragraph("No compliance findings recorded.", table_body),
            Paragraph("—", table_body),
            Paragraph("—", table_body),
            Paragraph("—", table_body),
            Paragraph("—", table_body),
        ])

    findings_table = Table(findings_rows, colWidths=[70, 75, 175, 55, 55, 35, 60], repeatRows=1)
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EAEDF1')),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('TOPPADDING', (0,0), (-1,0), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E6EA')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,1), (-1,-1), 3),
        ('BOTTOMPADDING', (0,1), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(findings_table)
    story.append(Spacer(1, 8))

    # 4b. The photograph and the evidence crops - an inspection record that
    # cites a rule and shows nothing is not evidence an officer can use.
    from reportlab.platypus import Image as RLImage

    import io as _io

    def _img(path, max_w, max_h):
        """Embedded as a JPEG at ~150 dpi for its printed size (a raw PNG
        crop was 3 MB of the PDF on its own)."""
        try:
            im = cv2.imread(str(path))
            if im is None:
                return None
            h_, w_ = im.shape[:2]
            s_ = min(max_w / w_, max_h / h_)
            px_w = max(1, int(w_ * s_ / 72.0 * 150))       # points -> px at 150 dpi
            if px_w < w_:
                im = cv2.resize(im, (px_w, max(1, int(h_ * px_w / w_))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return None
            return RLImage(_io.BytesIO(buf.tobytes()), width=w_ * s_, height=h_ * s_)
        except Exception:
            return None

    photo = _img(scan.image_path, 180 * mm, 110 * mm) if scan.image_path and Path(scan.image_path).exists() else None
    ev = []
    for f in sorted_findings:
        if f.outcome in (Outcome.VIOLATION, Outcome.INDETERMINATE) and f.evidence_crop_path \
                and Path(f.evidence_crop_path).exists():
            im = _img(f.evidence_crop_path, 85 * mm, 30 * mm)
            if im is not None:
                ev.append((f, im))
        if len(ev) >= 12:
            break
    if photo is not None or ev:
        story.append(Paragraph("<b>Photograph and Evidence</b>", section_h2))
        if photo is not None:
            story.append(photo)
            story.append(Spacer(1, 6))
        cells = [[im, Paragraph(f"<b>{html.escape(OUTCOME_STYLE.get(f.outcome, ('', '', ''))[2])}</b> "
                                f"{html.escape(f.rule_id)}<br/>{html.escape((f.message or '')[:140])}",
                                table_body)] for f, im in ev]
        if cells:
            et = Table(cells, colWidths=[90 * mm, 90 * mm])
            et.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E6EA')),
                ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            story.append(et)
        story.append(Spacer(1, 8))

    # 5. Declarations extracted section
    if scan.declarations:
        decl_header = [
            Paragraph("Field", table_header),
            Paragraph("Extracted Text", table_header),
            Paragraph("Parsed Value", table_header),
            Paragraph("Conf.", table_header),
            Paragraph("Source / Notes", table_header),
        ]
        decl_rows = [decl_header]
        for d in scan.declarations:
            if not d.present:
                continue
            val = "—" if d.value is None else str(d.value)
            if d.unit:
                val += f" {d.unit}"
            notes = "; ".join(d.notes or [])
            src = getattr(d, 'source', 'ocr')
            if src != 'ocr':
                notes = f"[{src.upper()}] " + notes
            decl_rows.append([
                Paragraph(html.escape(d.field_id), table_body_mono),
                Paragraph(html.escape((d.raw_text or '')[:160]), table_body),
                Paragraph(html.escape(val[:60]), table_body),
                Paragraph(f"{d.extraction_confidence:.2f}", table_body),
                Paragraph(html.escape(notes), table_body),
            ])
        if len(decl_rows) > 1:
            story.append(Paragraph("<b>Mandatory Declarations Extracted (OCR &amp; Vision)</b>", section_h2))
            decl_table = Table(decl_rows, colWidths=[85, 180, 85, 35, 140], repeatRows=1)
            decl_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EAEDF1')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E6EA')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('TOPPADDING', (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('LEFTPADDING', (0,0), (-1,-1), 3),
                ('RIGHTPADDING', (0,0), (-1,-1), 3),
            ]))
            story.append(decl_table)
            story.append(Spacer(1, 10))

    # 6. Every line read, in reading order - Hindi included, as printed
    # (nothing is translated). Declarations above are only the fields the
    # rules need; this is what the officer checks them against.
    if scan.spans:
        lines = sorted(scan.spans, key=lambda sp: (round(sp.bbox.y / 12), sp.bbox.x))
        text_rows = [[Paragraph("Text read from the label", table_header),
                      Paragraph("Script", table_header), Paragraph("Conf.", table_header)]]
        for sp in lines[:250]:
            script = "Hindi" if getattr(sp, "language", "en") == "hi" else "Latin"
            text_rows.append([
                Paragraph(html.escape(sp.text[:200]), table_body),
                Paragraph(script, table_body),
                Paragraph(f"{sp.confidence:.2f}", table_body),
            ])
        story.append(Paragraph("<b>Text Read from the Label (as printed, not translated)</b>", section_h2))
        text_table = Table(text_rows, colWidths=[420, 60, 45], repeatRows=1)
        text_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EAEDF1')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E6EA')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(text_table)
        story.append(Spacer(1, 10))

    # 7. Signature block
    sign_block = [
        Paragraph("<b>Statutory Inspection Certificate:</b>", ParagraphStyle('SC', fontName=FB, fontSize=8, leading=10)),
        Spacer(1, 3),
        Paragraph(
            "This report is machine-generated decision support under the Legal Metrology (Packaged Commodities) Rules, 2011. "
            "Findings marked INDETERMINATE fall within measurement tolerance and do not assert a violation. "
            "Official enforcement requires verification and signature by an authorised Inspector.",
            ParagraphStyle('SM', fontName=F, fontSize=7, leading=9, textColor=colors.HexColor('#69757E'))
        ),
        Spacer(1, 6),
        Table([
            [
                Paragraph("<b>Inspecting Officer:</b> ___________________________", table_body),
                Paragraph("<b>Signature:</b> ___________________________", table_body),
                Paragraph("<b>Date:</b> ______________", table_body)
            ]
        ], colWidths=[200, 200, 125], style=[
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
        ])
    ]
    story.append(KeepTogether(sign_block))

    doc.build(story)
    return str(out)


def save_pdf(scan: ScanResult, out_path: str | Path, rule_audit=None) -> Optional[str]:
    """
    Generate a publication-grade compliance PDF report.
    Uses ReportLab as the primary engine (pure Python, selectable text,
    A4 layout with auto-pagination and signature certificate).
    Falls back to WeasyPrint if available.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Primary: ReportLab (standard in requirements.txt)
    try:
        return _save_pdf_reportlab(scan, out, rule_audit)
    except ImportError:
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("ReportLab PDF generation error: %s", e)

    # 2. Fallback: WeasyPrint
    try:
        from weasyprint import HTML  # type: ignore
        HTML(string=build_html_report(scan, rule_audit)).write_pdf(str(out))
        return str(out)
    except (ImportError, Exception):
        pass

    return None


def save_docx(scan: ScanResult, out_path: str | Path) -> Optional[str]:
    """
    Editable format, per the PS requirement. Officers routinely need to
    add context before filing, so this deliberately carries the findings
    as editable text rather than as a flattened image.
    """
    try:
        from docx import Document  # type: ignore
        from docx.shared import Pt
    except ImportError:
        return None

    doc = Document()
    doc.add_heading("Legal Metrology Compliance Report", level=1)
    doc.add_paragraph(
        f"Scan {scan.scan_id} · {scan.created_at[:19].replace('T', ' ')} UTC · "
        f"Inspector: {scan.inspector_id or 'unassigned'}"
    )
    verdict, _, note = _verdict(scan)
    p = doc.add_paragraph()
    p.add_run(f"Verdict: {verdict}. ").bold = True
    p.add_run(note)

    doc.add_heading("Findings", level=2)
    table = doc.add_table(rows=1, cols=6)
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(["Outcome", "Rule", "Finding", "Measured", "Required", "Conf."]):
        cell = table.rows[0].cells[i]
        cell.text = h
        for r in cell.paragraphs[0].runs:
            r.bold = True
            r.font.size = Pt(9)

    for f in scan.findings:
        if f.outcome is Outcome.NOT_APPLICABLE:
            continue
        cells = table.add_row().cells
        cells[0].text = OUTCOME_STYLE[f.outcome][2]
        # Include the field. Rule 7(3) is evaluated per declaration, so a
        # report could show two Rule 7(3) rows with opposite verdicts and
        # no way to tell which field each referred to - unusable for an
        # officer who has to act on it.
        cells[1].text = (
            f"{f.citation or '—'}\n{f.field_id}" if f.field_id
            else (f.citation or "—")
        )
        cells[2].text = f.message
        cells[3].text = (
            f"{f.measured_value:g} {f.unit}".strip() if f.measured_value is not None else "—"
        )
        cells[4].text = (
            f"{f.required_value:g} {f.unit}".strip() if f.required_value is not None else "—"
        )
        cells[5].text = f"{f.confidence:.2f}"

    def _hindi_font(cell):
        # Word draws Hindi with the complex-script font; name one every
        # Windows has (Nirmala UI) instead of leaving it to fallback.
        from docx.oxml.ns import qn
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)
                rpr = r._element.get_or_add_rPr()
                fonts = rpr.find(qn("w:rFonts"))
                if fonts is None:
                    fonts = rpr.makeelement(qn("w:rFonts"), {})
                    rpr.append(fonts)
                fonts.set(qn("w:cs"), "Nirmala UI")

    decls = [d for d in scan.declarations if d.present]
    if decls:
        doc.add_heading("Declarations extracted", level=2)
        t = doc.add_table(rows=1, cols=3)
        t.style = "Light Grid Accent 1"
        for cell, h in zip(t.rows[0].cells, ("Field", "Text on the label", "Value")):
            cell.text = h
        for d in decls:
            cells = t.add_row().cells
            cells[0].text = d.field_id
            cells[1].text = (d.raw_text or "")[:200]
            cells[2].text = ("—" if d.value is None else str(d.value)) + (f" {d.unit}" if d.unit else "")
            _hindi_font(cells[1])

    if scan.spans:
        doc.add_heading("Text read from the label (as printed, not translated)", level=2)
        t = doc.add_table(rows=1, cols=3)
        t.style = "Light Grid Accent 1"
        for cell, h in zip(t.rows[0].cells, ("Text", "Script", "Conf.")):
            cell.text = h
        for sp in sorted(scan.spans, key=lambda sp: (round(sp.bbox.y / 12), sp.bbox.x))[:250]:
            cells = t.add_row().cells
            cells[0].text = sp.text[:200]
            cells[1].text = "Hindi" if getattr(sp, "language", "en") == "hi" else "Latin"
            cells[2].text = f"{sp.confidence:.2f}"
            _hindi_font(cells[0])

    doc.add_paragraph()
    doc.add_paragraph(
        "Machine-generated decision support. Every finding requires confirmation "
        "by an authorised officer before enforcement action."
    ).italic = True

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return str(out)


# =====================================================================
# Rehydration + unified entry point
# =====================================================================

def scan_from_dict(payload: dict) -> ScanResult:
    """
    Rebuild a ScanResult from its stored JSON.

    Reports are generated from the database long after the scan ran, so
    the builder must work from persisted JSON, not only from a live
    object. Keeping this in one place means the report always reflects
    exactly what was stored - no second, subtly different code path that
    formats a live scan differently from a recalled one.
    """
    from ..core.schema import (
        BBox,
        Calibration,
        CalibrationMethod,
        Declaration,
        Finding,
        GlyphMetrics,
        Outcome,
        PackageContext,
        PackageGeometry,
        Severity,
        TextSpan,
    )

    def box(d):
        return BBox(**d) if d else None

    scan = ScanResult(
        scan_id=payload.get("scan_id", ""),
        created_at=payload.get("created_at", ""),
        image_path=payload.get("image_path"),
        inspector_id=payload.get("inspector_id"),
        image_quality_ok=payload.get("image_quality_ok", True),
        quality_notes=payload.get("quality_notes", ""),
        pipeline_version=payload.get("pipeline_version", ""),
    )
    scan.ocr_stats = dict(payload.get("ocr_stats") or {})
    for sp in payload.get("spans") or []:
        try:
            scan.spans.append(TextSpan(
                text=sp.get("text", ""),
                bbox=BBox(**sp["bbox"]),
                confidence=sp.get("confidence", 1.0),
                source_engine=sp.get("source_engine", "unknown"),
                language=sp.get("language", "en"),
                vertical=sp.get("vertical", False),
                ink_bbox=box(sp.get("ink_bbox")),
            ))
        except (KeyError, TypeError):
            continue

    c = payload.get("calibration") or {}
    scan.calibration = Calibration(
        px_per_mm=c.get("px_per_mm"),
        method=CalibrationMethod(c.get("method", "none")),
        uncertainty_px_per_mm=c.get("uncertainty_px_per_mm", 0.0),
        marker_id=c.get("marker_id"),
        marker_size_mm=c.get("marker_size_mm"),
        notes=c.get("notes", ""),
    )

    # Rebuild PackageContext GENERICALLY from its dataclass fields
    # rather than by naming each one. The hand-written version listed
    # fields explicitly and silently dropped every field added after it
    # was written - barcode, printed_price, pdp_detection_method,
    # pdp_confidence and the detection-failure flags all vanished on the
    # way out of the database. That quietly defeated the GTIN work
    # entirely: the dual-MRP endpoint reads scans back through this
    # function, found no barcode, and downgraded every finding from
    # definitive (0.90) to inferred (0.55).
    import dataclasses

    ctx = payload.get("context") or {}
    ctx_kwargs = {}
    for f in dataclasses.fields(PackageContext):
        if f.name not in ctx:
            continue
        val = ctx[f.name]
        if f.name == "geometry":
            val = PackageGeometry(val)
        elif f.name == "pdp_bbox":
            val = box(val)
        ctx_kwargs[f.name] = val
    scan.context = PackageContext(**ctx_kwargs)

    for d in payload.get("declarations", []):
        g = d.get("glyph")
        scan.declarations.append(Declaration(
            field_id=d["field_id"],
            raw_text=d.get("raw_text", ""),
            value=d.get("value"),
            unit=d.get("unit"),
            bbox=box(d.get("bbox")),
            glyph=GlyphMetrics(**g) if g else None,
            extraction_confidence=d.get("extraction_confidence", 0.0),
            present=d.get("present", False),
            notes=list(d.get("notes") or []),
            source=d.get("source", "ocr"),
        ))

    for f in payload.get("findings", []):
        scan.findings.append(Finding(
            rule_id=f["rule_id"],
            citation=f.get("citation", ""),
            outcome=Outcome(f["outcome"]),
            severity=Severity(f.get("severity", "minor")),
            message=f.get("message", ""),
            field_id=f.get("field_id"),
            measured_value=f.get("measured_value"),
            required_value=f.get("required_value"),
            unit=f.get("unit", ""),
            uncertainty=f.get("uncertainty", 0.0),
            evidence_bbox=box(f.get("evidence_bbox")),
            evidence_crop_path=f.get("evidence_crop_path"),
            confidence=f.get("confidence", 0.0),
            rule_verified=f.get("rule_verified", True),
            exemption_applied=f.get("exemption_applied"),
            human_override=f.get("human_override"),
            human_note=f.get("human_note", ""),
        ))

    return scan


def build_report(payload, out_path, fmt: str = "pdf", rule_audit=None):
    """
    Single entry point used by the API. Accepts a live ScanResult or the
    stored JSON dict. PDF is made with ReportLab (WeasyPrint as a fallback);
    if neither can make one, the report degrades to HTML rather than
    failing the request.
    """
    scan = payload if isinstance(payload, ScanResult) else scan_from_dict(payload)
    out_path = Path(out_path)

    if fmt == "html":
        return save_html(scan, out_path, rule_audit)

    if fmt == "docx":
        docx = save_docx(scan, out_path)
        if docx and Path(docx).exists():
            return docx
        raise RuntimeError("DOCX generation failed: python-docx not installed.")

    pdf = save_pdf(scan, out_path, rule_audit)
    if pdf and Path(pdf).exists():
        return pdf
    return save_html(scan, out_path.with_suffix(".html"), rule_audit)
