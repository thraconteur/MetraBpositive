#!/usr/bin/env python3
"""
Generate the printable calibration card.

    python scripts/make_marker.py --size-mm 25 --out marker.png

PRINT AT 100% SCALE. Any "fit to page" or "shrink to fit" option
silently rescales the marker, and every millimetre measurement in the
system inherits that error. After printing, measure the marker edge with
a ruler: if it reads 24 mm instead of 25 mm, either reprint or pass the
true measured size with --size-mm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.calibration import generate_marker_card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="marker_card.png")
    ap.add_argument("--size-mm", type=float, default=25.0)
    ap.add_argument("--marker-id", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    p = generate_marker_card(a.out, a.marker_id, a.size_mm, a.dpi)
    print(f"Wrote {p}  ({a.size_mm} mm marker, id={a.marker_id}, {a.dpi} dpi)")
    print("Print at 100% scale, then verify the edge with a ruler.")


if __name__ == "__main__":
    main()
