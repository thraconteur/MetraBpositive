#!/usr/bin/env python3
"""
Scan one real photo: PaddleOCR first, Gemini for whatever OCR missed,
then the rules - printed as a console summary AND saved as the HTML
inspector report.

    python scripts/scan_real.py data/photo5.jpeg
    python scripts/scan_real.py data/photo5.jpeg --panel-mm 95x70
    python scripts/scan_real.py data/photo5.jpeg --no-vlm

Gemini runs automatically when GEMINI_API_KEY is set (pip install
google-genai). Without the key the scan still runs, OCR-only, and says so.

What changed from the first version, and why:

  * Gemini's answers are no longer written straight into the declarations.
    A vision model can "read" words that are not printed - it knows MRP
    lines usually say "inclusive of all taxes" - so each answer now comes
    with the box it was read from, is checked against OCR at that spot,
    and is marked on the report ("vlm" / "vlm+ocr"). A finding that rests
    on the model alone is INDETERMINATE with a crop to confirm, never a
    verdict.

  * No forced calibration. Setting px_per_mm = 14.5 by hand (and pasting
    an ArUco marker into the photo) produced millimetre findings from a
    scale nobody measured. For the Table-I height checks without a
    printed marker, measure the panel with a ruler and pass --panel-mm:
    the scale then comes from the real panel edges in the photo, with an
    honest (wider) uncertainty, and the report says where it came from.
    (The old line `res.calibration.available = True` also could not run:
    `available` is computed from px_per_mm, not a field you can set.)

  * The verdict comes from the result itself. "PASS" used to be printed
    whenever no violation was found, even if half the declarations were
    unreadable. Now: COMPLIANT / NON-COMPLIANT / UNDECIDED (the system
    abstained - retake or photograph the other panels).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--panel-mm", default=None, metavar="WxH",
                    help="Real panel size in mm (ruler), for mm checks without a marker")
    ap.add_argument("--no-vlm", action="store_true", help="OCR only, even if a key is set")
    ap.add_argument("--coverage-complete", action="store_true",
                    help="This photo shows every panel of the pack")
    ap.add_argument("--out", default=None, help="HTML report path (default data/reports/)")
    ap.add_argument("--replay", default=None,
                    help="Use a saved .paddle.json instead of running PaddleOCR")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Error: could not find image at {image_path}")

    from scan_photo import _parse_panel

    from src.core.pipeline import CompliancePipeline
    from src.report.builder import build_report, extract_evidence_crops
    from src.vision.ocr.backends import build_default_ocr
    from src.vision.ocr.vlm_gemini import GeminiReader, VLMUnavailable

    os.environ.setdefault("SIH_OCR_BACKEND", "paddle")
    if args.replay:
        from src.vision.ocr.paddle import ReplayOCR

        ocr = ReplayOCR(args.replay)
    else:
        ocr = build_default_ocr()

    vlm = None
    if not args.no_vlm:
        reader = GeminiReader()
        try:
            reader.check()
            vlm = reader
        except VLMUnavailable as exc:
            print(f"(Gemini off: {exc})")

    print(f"Scanning physical image: {image_path} ...")
    print("PASS 1: PaddleOCR" + (f"  |  PASS 2: {vlm.label()} for fields OCR missed" if vlm else ""))

    pipeline = CompliancePipeline(ocr=ocr, vlm=vlm)
    res = pipeline.scan(str(image_path), panel_size_mm=_parse_panel(args.panel_mm))
    if args.coverage_complete:
        res.coverage_complete = True
        res.findings = []
        pipeline.engine.evaluate(res)

    try:
        extract_evidence_crops(res, Path("data/evidence") / res.scan_id)
    except Exception as exc:
        print(f"(evidence crops failed: {exc})")
    out = Path(args.out) if args.out else Path("data/reports") / f"{image_path.stem}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out = build_report(res, out, fmt="html") or out

    v = (res.ocr_stats or {}).get("vlm")
    if v:
        print(f"\nPASS 2 asked {vlm.label()} for: {', '.join(v.get('asked') or []) or 'nothing'}")
        if v.get("error"):
            print(f"   Gemini call failed: {v['error']}")
        for f in v.get("found") or []:
            tag = "confirmed by OCR" if f in (v.get("confirmed_by_ocr") or []) else "model only - confirm"
            print(f"   found {f}  ({tag})")

    print("=" * 60)
    print(" LMPC COMPLIANCE REPORT (PASS 1 + PASS 2)")
    print("=" * 60)
    verdict = {True: "COMPLIANT", False: "NON-COMPLIANT",
               None: "UNDECIDED (retake, or scan the other panels)"}[res.is_compliant]
    print(f" VERDICT: {verdict}\n")

    print("[EXTRACTED DECLARATIONS]")
    for d in res.declarations:
        if d.present:
            src = "" if d.source == "ocr" else f"   <- {d.source}"
            print(f"  [x] {d.field_id:<22}: {d.raw_text[:70]}{src}")
            for n in d.notes:
                print(f"      {'':<22}  note: {n[:110]}")
        else:
            print(f"  [ ] {d.field_id:<22}: not read in this photo")

    print("\n[GAZETTE VIOLATIONS]")
    if not res.violations:
        print("  None asserted on this panel.")
    for f in res.violations:
        print(f"  - {f.citation}: {f.message[:110]}")
    if res.indeterminate:
        print(f"\n[TO CONFIRM BY EYE] {len(res.indeterminate)} item(s) - see the report")

    print("\n[CALIBRATION]")
    cal = res.calibration
    if cal.available:
        print(f"  {cal.method.value}: {cal.px_per_mm:.2f} px/mm "
              f"(+/- {cal.uncertainty_px_per_mm:.2f})")
    else:
        print(f"  No scale. Millimetre rules (Table-I) abstained. {cal.notes[:120]}")
    print(f"\nReport: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
