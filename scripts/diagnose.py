"""
Diagnostic tooling: raw OCR export and stability testing.

WHY THIS EXISTS
----------------
A compliance report tells you what the SYSTEM concluded. It does not
tell you what the OCR ENGINE actually saw, so a report alone cannot
distinguish "the rule logic is wrong" from "the OCR misread the label" -
which is exactly the question raised against back_9.jpeg. This module
answers that question directly: dump every raw span with its text,
confidence and box, so the two failure classes can be told apart without
guessing.

It also answers the stability question. Re-running the SAME bytes through
a deterministic OCR engine (PaddleOCR is deterministic on identical pixels) twice
produces IDENTICAL output - that would trivially "prove" stability while
proving nothing, because a photograph is never handed to the pipeline
twice with byte-identical pixels. The perturbation harness instead
applies small rotations and scales - the variation an inspector's second
photo actually has - and measures how much the extracted fields move.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dump_raw_ocr(image_path: str, out_path: Optional[str] = None,
                 ocr=None) -> dict:
    """
    Run OCR alone (no extraction, no rules) and dump every span.

    This is deliberately the OCR stage in isolation: it answers "what did
    the engine read" without any of the extraction or rules-engine logic
    that could mask or explain away a misread. If a field is wrong
    downstream, this file is the first thing to check - it tells you
    whether the token was ever legible at all.
    """
    from src.vision.ocr.backends import build_default_ocr

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    engine = ocr or build_default_ocr()
    spans = engine.recognise(img)

    dump = {
        "image_path": image_path,
        "image_shape": list(img.shape),
        "engine": spans[0].source_engine if spans else "none",
        "span_count": len(spans),
        "spans": [
            {
                "text": s.text,
                "confidence": round(s.confidence, 4),
                "bbox": {"x": round(s.bbox.x, 1), "y": round(s.bbox.y, 1),
                        "w": round(s.bbox.w, 1), "h": round(s.bbox.h, 1)},
                "source_engine": s.source_engine,
            }
            for s in spans
        ],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(dump, indent=2), encoding="utf-8")
    return dump


def _perturb(img: np.ndarray, rotate_deg: float, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_deg, scale)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def stability_report(
    image_path: str,
    rotations: tuple = (-5, -2, 0, 2, 5),
    scales: tuple = (0.95, 1.0, 1.05),
    out_path: Optional[str] = None,
) -> dict:
    """
    Run the FULL pipeline across small rotation/scale perturbations and
    report which fields are stable and which flip.

    This is the honest version of "run it 3-5 times and diff the output".
    Re-running identical bytes through a deterministic engine produces
    identical output every time and would report perfect stability while
    testing nothing. What actually varies between two real photographs of
    the same pack is framing: a few degrees of tilt, a slightly different
    distance. Perturbing the same base image approximates that variation
    without needing new photographs, and is the closest thing to a real
    stability measurement obtainable from one image.
    """
    from src.core.pipeline import CompliancePipeline

    base = cv2.imread(image_path)
    if base is None:
        raise FileNotFoundError(image_path)

    pipeline = CompliancePipeline()
    runs = []
    for rot in rotations:
        for sc in scales:
            variant = _perturb(base, rot, sc)
            tmp = "/tmp/_stability_variant.png"
            cv2.imwrite(tmp, variant)
            try:
                res = pipeline.scan(tmp)
                fields = {
                    d.field_id: {
                        "present": d.present,
                        "raw_text": d.raw_text,
                        "value": d.value,
                        "unit": d.unit,
                    }
                    for d in res.declarations
                }
            except Exception as exc:
                fields = {"_error": str(exc)}
            runs.append({"rotation_deg": rot, "scale": sc, "fields": fields})

    # Per-field stability: does presence and value agree across every run?
    all_field_ids = {
        fid for r in runs for fid in r["fields"] if not fid.startswith("_")
    }
    stability = {}
    for fid in sorted(all_field_ids):
        values = [
            (r["fields"].get(fid, {}).get("present"),
             r["fields"].get(fid, {}).get("value"))
            for r in runs
        ]
        distinct = set(values)
        stability[fid] = {
            "stable": len(distinct) == 1,
            "distinct_outcomes": len(distinct),
            "runs": len(runs),
        }

    report = {
        "image_path": image_path,
        "n_runs": len(runs),
        "perturbations": {"rotations_deg": list(rotations),
                          "scales": list(scales)},
        "field_stability": stability,
        "runs": runs,
    }
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--mode", choices=["raw", "stability"], default="raw")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "raw":
        d = dump_raw_ocr(args.image, args.out)
        print(f"{d['span_count']} spans via {d['engine']}")
    else:
        r = stability_report(args.image, out_path=args.out)
        for fid, s in r["field_stability"].items():
            flag = "STABLE" if s["stable"] else "UNSTABLE"
            print(f"{fid:24s} {flag:9s} {s['distinct_outcomes']} distinct "
                 f"outcome(s) across {s['runs']} runs")
