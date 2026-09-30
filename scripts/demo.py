#!/usr/bin/env python3
"""
End-to-end demo. Run this first to confirm the install works.

    python scripts/demo.py

Renders a compliant label and several deliberately non-compliant ones,
scans each, and prints what the rules engine found. No network, no GPU,
no model downloads.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.pipeline import CompliancePipeline
from src.core.rules_engine import RuleConfig
from src.synth.generator import LabelRenderer, LabelSpec, apply_violation

OUT = Path("data/demo")
CASES = [
    ("missing_mrp", "No price declaration at all"),
    ("mrp_no_tax_qualifier", "Price present, prescribed wording absent"),
    ("missing_consumer_care", "No consumer care contact"),
    ("missing_date", "No manufacture date"),
    ("banned_phrase", "'approx.' attached to net quantity"),
    ("glyph_aspect_ratio", "Horizontally condensed numerals"),
    ("quantity_clear_space", "Print crowded against the quantity"),
]


def main():
    # Rendered labels can be read by PaddleOCR (default - the real engine)
    # or by SyntheticOCR, which returns the renderer's own text and needs
    # no model:  python scripts/demo.py synthetic
    if len(sys.argv) > 1 and sys.argv[1] in ("paddle", "synthetic"):
        os.environ["SIH_OCR_BACKEND"] = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    audit = RuleConfig().audit()
    print("=" * 72)
    print("RULE CONFIG AUDIT")
    print(f"  verified   : {len(audit['verified'])}")
    print(f"  unverified : {len(audit['unverified'])} -> {', '.join(audit['unverified'])}")
    print(f"  coverage   : {audit['coverage']*100:.0f}%")
    print("  Unverified rules CANNOT emit a violation. They report")
    print("  UNVERIFIED_RULE until someone transcribes the gazette values.")
    print("=" * 72)

    renderer = LabelRenderer(dpi=300)
    pipeline = CompliancePipeline()

    # -- compliant baseline -------------------------------------------
    renderer.render(LabelSpec(), out_path=str(OUT / "compliant.png"))
    res = pipeline.scan(str(OUT / "compliant.png"))
    print("\nBASELINE (compliant label)")
    print(f"  verdict     : {res.is_compliant}   <- None means 'cannot say'")
    print(f"  violations  : {len(res.violations)}")
    print(f"  unverified  : {len(res.unverified)}")
    print(f"  calibration : {res.calibration.method.value} "
          f"({res.calibration.px_per_mm:.2f} px/mm)"
          if res.calibration.available else "  calibration : none")
    print("  Verdict is None, not True, because Table-I is unverified.")
    print("  Refusing to certify compliance you cannot actually check is")
    print("  the point, not a limitation.")

    # -- injected violations ------------------------------------------
    print("\nINJECTED VIOLATIONS")
    print(f"  {'injected':<24}{'detected rule ids'}")
    print("  " + "-" * 68)
    for kind, _desc in CASES:
        spec = apply_violation(LabelSpec(), kind)
        path = OUT / f"{kind}.png"
        renderer.render(spec, out_path=str(path), violations=[kind])
        r = pipeline.scan(str(path))
        ids = ", ".join(f.rule_id for f in r.violations) or "(none)"
        print(f"  {kind:<24}{ids}")

    print(f"\nImages written to {OUT}/")


if __name__ == "__main__":
    main()
