#!/usr/bin/env python3
"""
Printable calibration cards as an A4 PDF (4 cards per page).

    python scripts/make_card_pdf.py            # -> calibration_cards.pdf

Why a PDF: a PNG opened in a photo viewer prints "fit to frame" by default
and is silently rescaled. A PDF printed at "Actual size" / 100% keeps the
marker exactly 25.0 mm. Check it with a ruler after printing: the black
square must measure 25 mm, and the check line under it 50 mm.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calibration_cards.pdf")
    ap.add_argument("--size-mm", type=float, default=25.0)
    ap.add_argument("--marker-id", type=int, default=0)
    a = ap.parse_args()

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, a.marker_id, 600)
    tmp = Path(tempfile.gettempdir()) / "lmpc_marker.png"
    cv2.imwrite(str(tmp), marker)

    c = canvas.Canvas(a.out, pagesize=A4)
    W, H = A4
    s = a.size_mm * mm
    card_w, card_h = 90 * mm, 80 * mm
    for i in range(4):
        cx = 15 * mm + (i % 2) * (card_w + 10 * mm)
        cy = H - 25 * mm - (i // 2 + 1) * (card_h + 10 * mm)
        c.setDash(2, 3)
        c.rect(cx, cy, card_w, card_h)           # cut line
        c.setDash()
        mx = cx + (card_w - s) / 2
        my = cy + card_h - 12 * mm - s
        c.drawImage(str(tmp), mx, my, s, s)
        # 50 mm check line
        ly = my - 9 * mm
        lx = cx + (card_w - 50 * mm) / 2
        c.setLineWidth(0.6)
        c.line(lx, ly, lx + 50 * mm, ly)
        for t in range(0, 51, 10):
            c.line(lx + t * mm, ly, lx + t * mm, ly + (2.5 if t % 50 else 4) * mm)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(cx + card_w / 2, ly - 4 * mm,
                            f"check: this line = 50 mm, black square = {a.size_mm:g} mm")
        c.drawCentredString(cx + card_w / 2, cy + 4 * mm,
                            f"METRA / LMPC calibration  id={a.marker_id}  - place flat ON the label face")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(15 * mm, H - 15 * mm,
                 "Print at ACTUAL SIZE / 100% (not 'fit to page'). Then check the 50 mm line with a ruler.")
    c.save()
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
