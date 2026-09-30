#!/usr/bin/env python3
"""
Scan real photos with PaddleOCR and write one compliance report per photo.

    python scripts/scan_photo.py back_9.jpeg
    python scripts/scan_photo.py photos/                     # every image in a folder
    python scripts/scan_photo.py back_9.jpeg --secondary-lang hi   # bilingual label
    python scripts/scan_photo.py back_9.jpeg --replay data/ocr_dumps/back_9.paddle.json

PaddleOCR is the only engine. If it is not installed this stops and
says so - it never quietly reads with something weaker.

Each scan writes three things:
  data/reports/<photo>.html          the inspector report (also shows the
                                     raw OCR lines and what was extracted)
  data/ocr_dumps/<photo>.paddle.json exactly what Paddle read - send this
                                     back when a report looks wrong
  data/.ocr_cache/                   cached OCR, so re-running the same
                                     photo after a code change skips the
                                     model entirely (--no-cache to force)

--replay re-runs everything downstream of OCR from a saved dump without
loading any model, in well under a second. Use it to check an
extraction or rules fix against a photo someone else scanned.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _photos(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            out.append(p)
        else:
            sys.exit(f"No such file or folder: {p}")
    if not out:
        sys.exit("No images found.")
    return out


def _build_ocr(args):
    if args.replay:
        from src.vision.ocr.paddle import ReplayOCR

        return ReplayOCR(args.replay), f"replay of {args.replay}"

    from src.vision.ocr.paddle import PaddleConfig, PaddleOCRBackend

    cfg = dict(
        lang=args.lang,
        ocr_version=args.ocr_version,
        secondary_lang="" if args.no_hindi else args.secondary_lang,
        device=args.device,
        # None = automatic: on with PaddlePaddle 3.2.x, off on 3.3.x (crash).
        enable_mkldnn=(True if (args.mkldnn or os.environ.get("SIH_OCR_MKLDNN", "") == "1")
                       else False if args.no_mkldnn else None),
        text_det_limit_side_len=args.det_side,
    )
    if args.sideways:
        cfg["sideways_reread"] = True
    if args.no_upscale:
        cfg["upscale_short_side_below"] = 0
    if args.no_refine:
        cfg["refine_below_score"] = 0.0
    if args.det_thresh is not None:
        cfg["text_det_thresh"] = args.det_thresh
    if args.box_thresh is not None:
        cfg["text_det_box_thresh"] = args.box_thresh
    if args.unclip is not None:
        cfg["text_det_unclip_ratio"] = args.unclip
    if os.environ.get("SIH_OCR_DET_MAX"):
        cfg["det_max_long_side"] = int(os.environ["SIH_OCR_DET_MAX"])
    config = PaddleConfig(**{k: v for k, v in cfg.items() if v is not None})

    ocr = PaddleOCRBackend(config=config,
                           cache_dir=None if args.no_cache else
                           Path(__file__).resolve().parents[1] / "data" / ".ocr_cache")
    if not ocr.available():
        sys.exit(
            "PaddleOCR is not installed in this Python environment.\n"
            "    pip install paddlepaddle paddleocr\n"
            "(Check with: python -c \"import paddleocr; print(paddleocr.__version__)\")"
        )
    desc = f"paddleocr lang={config.lang} rec={ocr._rec_model_name(config.lang)}"
    try:
        import paddle

        from src.vision.ocr.paddle import _mkldnn

        on = _mkldnn(config)
        desc += (f" | fast CPU mode {'ON' if on else 'OFF'} (paddlepaddle {paddle.__version__}"
                 + ("" if on else "; pip install paddlepaddle==3.2.2 to turn it on") + ")")
    except Exception:
        pass
    if config.secondary_lang:
        desc += f" + {config.secondary_lang} re-read"
        if not _hindi_model_on_disk(ocr, config.secondary_lang):
            desc += ("\n  !! Hindi model NOT downloaded - Hindi lines will be read as English."
                     "\n  !! Fix once:  python scripts/get_hindi_model.py")
    return ocr, desc


def _hindi_model_on_disk(ocr, lang):
    home = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".paddlex")
    return (Path(home) / "official_models" / ocr._rec_model_name(lang)).is_dir()


def _parse_panel(txt):
    if not txt:
        return None
    import re

    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[xX*,]\s*(\d+(?:\.\d+)?)\s*", txt)
    if not m:
        sys.exit("--panel-mm must look like 95x70 (width x height in mm).")
    return float(m.group(1)), float(m.group(2))


def _build_vlm(choice):
    if not choice:
        return None
    from src.vision.ocr.vlm_gemini import GeminiReader, VLMUnavailable

    reader = GeminiReader()
    try:
        reader.check()
    except VLMUnavailable as exc:
        sys.exit(str(exc))
    return reader


def _print_scan(result, out, st):
    print(f"  OCR          : {st.get('regions', len(result.spans))} lines"
          + (" (cached)" if st.get("cache_hit") else
             f" in {st.get('total_s', '?')}s"
             + (f" (model load {st['model_load_s']}s)" if st.get("model_load_s") else "")
             + (f", {st['refined']} weak line(s) re-read" if st.get("refined") else "")
             + (f", {st['curved']} curved line(s) re-read" if st.get("curved") else "")
             + (f", {st['secondary_replaced']} Devanagari" if st.get("secondary_replaced") else "")))
    parts = [f"{k[:-2]} {st[k]}s" for k in ("predict_s", "zoom_s", "contrast_s", "sideways_s", "vertical_s", "refine_s", "block_s", "curved_s", "secondary_s")
             if st.get(k) is not None]
    if parts:
        print(f"  time split   : {', '.join(parts)}")
    if st.get("secondary_unavailable"):
        print("  !! Hindi    : reader could not start (model not downloaded) - read in English only."
              "\n                 Fix once:  python scripts/get_hindi_model.py")
    if st.get("dump"):
        print(f"  OCR dump     : {st['dump']}")
    if st.get("vertical_bands"):
        print(f"  rotated text : {st['vertical_bands']} vertical band(s) re-read")
    v = st.get("vlm")
    if v:
        print(f"  second read  : {v.get('model')} asked for {len(v.get('asked', []))}, "
              f"found {v.get('found', [])}, confirmed by OCR {v.get('confirmed_by_ocr', [])}"
              + (f", ERROR {v['error']}" if v.get("error") else ""))
    cal = result.calibration
    print(f"  scale        : " + (f"{cal.px_per_mm:.2f} px/mm ({cal.method.value})"
                                  if cal.available else f"none - {cal.notes[:90]}"))
    print("  extracted    :")
    for d in result.declarations:
        if not d.present:
            continue
        val = "" if d.value is None else f" -> {d.value}{(' ' + d.unit) if d.unit else ''}"
        src = "" if getattr(d, "source", "ocr") == "ocr" else f"  [{d.source}]"
        print(f"    {d.field_id:22s} {d.raw_text[:60]!r}{val}{src}")
        for n in d.notes or []:
            print(f"    {'':22s} note: {n}")
    verdict = {True: "COMPLIANT", False: "NON-COMPLIANT", None: "UNDECIDED"}[result.is_compliant]
    try:
        from src.core.package import capture_advice
        from src.report.builder import _verdict

        _note = _verdict(result)[2] if result.is_compliant is None else ""
        _advice = capture_advice(result)
    except Exception:
        _note, _advice = "", []
    print(f"  verdict      : {verdict}  ({len(result.violations)} violation(s), "
          f"{len(result.indeterminate)} indeterminate)")
    for f in result.violations:
        print(f"    VIOLATION  {f.citation}: {f.message[:110]}")
    if _note:
        print(f"  why          : {_note}")
    for a in _advice:
        print(f"  next photo   : {a}")
    print(f"  report       : {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", nargs="+", help="Photo(s) or folder(s) to scan")
    ap.add_argument("--out", default=None,
                    help="Output path (single photo only; default data/reports/<name>.<fmt>)")
    ap.add_argument("--fmt", default="html", choices=["html", "pdf", "json"])
    ap.add_argument("--package", default=None, metavar="NAME",
                    help="All the photos given are sides of ONE pack (front, back, top, "
                         "base, crimp ...): decide the pack from all of them together. "
                         "The report is data/reports/NAME.html")
    ap.add_argument("--coverage-complete", action="store_true",
                    help="Assert the photo(s) show EVERY panel, so a missing field is "
                         "a VIOLATION rather than INDETERMINATE.")
    ap.add_argument("--panel-mm", default=None, metavar="WxH",
                    help="Real size of the photographed panel in mm, measured with a "
                         "ruler (e.g. 95x70). Gives a millimetre scale when there is no "
                         "printed marker in the photo. Never paste a marker into a photo.")
    ap.add_argument("--vlm", default=None, choices=["gemini"],
                    help="Ask a vision-language model for declarations OCR missed "
                         "(needs internet and GEMINI_API_KEY). Its reads are marked "
                         "on the report and never decide a verdict alone.")

    g = ap.add_argument_group("OCR")
    g.add_argument("--lang", default="en", choices=["en", "hi"],
                   help="Primary recognition language (default en - the strongest model; "
                        "use --secondary-lang hi for bilingual labels instead)")
    g.add_argument("--secondary-lang", default=None, choices=["hi"],
                   help="Second reader for weakly read lines (default: hi, Hindi)")
    g.add_argument("--sideways", action="store_true",
                   help="Also read the photo turned 90 degrees both ways, for text printed "
                        "sideways along an edge (slower: two extra reads)")
    g.add_argument("--no-hindi", action="store_true",
                   help="Turn the Hindi second reader off (it is on by default)")
    g.add_argument("--ocr-version", default=None,
                   choices=["PP-OCRv3", "PP-OCRv4", "PP-OCRv5", "PP-OCRv6"],
                   help="Model generation (default: PaddleOCR's choice for the "
                        "language - PP-OCRv6 medium for en)")
    g.add_argument("--device", default=None, help="cpu, gpu:0, ... (default: auto)")
    g.add_argument("--no-mkldnn", action="store_true",
                   help="Turn the fast CPU mode off (it is on automatically with "
                        "PaddlePaddle 3.2.x, off on 3.3.x where it crashes)")
    g.add_argument("--mkldnn", action="store_true",
                   help="Enable oneDNN. Faster on CPU, but crashes on some Windows "
                        "machines - off by default")
    g.add_argument("--det-side", type=int, default=None,
                   help="Detection input min side (default 1280)")
    g.add_argument("--det-thresh", type=float, default=None)
    g.add_argument("--box-thresh", type=float, default=None)
    g.add_argument("--unclip", type=float, default=None)
    g.add_argument("--no-upscale", action="store_true",
                   help="Do not upscale small photos before OCR")
    g.add_argument("--no-refine", action="store_true",
                   help="Do not re-read low-confidence lines")
    g.add_argument("--no-cache", action="store_true", help="Ignore the OCR cache")
    g.add_argument("--replay", default=None,
                   help="Use a saved .paddle.json dump instead of running Paddle")
    args = ap.parse_args()

    os.environ.setdefault("SIH_OCR_BACKEND", "paddle")
    photos = _photos(args.photos)
    if args.replay and len(photos) != 1:
        sys.exit("--replay takes exactly one photo (the one the dump was made from).")
    if args.out and len(photos) != 1 and not args.package:
        sys.exit("--out takes exactly one photo; use the default naming for batches.")

    from src.core.pipeline import CompliancePipeline
    from src.report.builder import build_report

    ocr, desc = _build_ocr(args)
    vlm = _build_vlm(args.vlm)
    print(f"engine: {desc}" + (f" + {vlm.label()} for missed fields" if vlm else ""))
    pipeline = CompliancePipeline(ocr=ocr, vlm=vlm)
    panel = _parse_panel(args.panel_mm)

    if args.package:
        t0 = time.perf_counter()
        print(f"\npackage '{args.package}': {len(photos)} photo(s)")
        from src.core.package import merge_package

        parts = []
        for photo in photos:
            print(f"  reading {photo.name} ...")
            parts.append(pipeline.scan(str(photo), panel_size_mm=panel))
        # Every side photographed: the inspector says so by using
        # --package; --coverage-complete is implied.
        result = merge_package(parts, pipeline.engine, coverage_complete=True)
        try:
            from src.report.builder import extract_evidence_crops

            extract_evidence_crops(result, Path("data/evidence") / result.scan_id)
        except Exception as exc:
            print(f"  (evidence crops failed: {exc})")
        out = Path(args.out) if args.out else Path("data/reports") / f"{args.package}.{args.fmt}"
        out.parent.mkdir(parents=True, exist_ok=True)
        out = build_report(result, out, fmt=args.fmt) or out
        _print_scan(result, out, {})
        print(f"  total        : {time.perf_counter() - t0:.1f}s")
        return

    for photo in photos:
        print(f"\n{photo}")
        t0 = time.perf_counter()
        result = pipeline.scan(str(photo), panel_size_mm=panel)
        if args.coverage_complete:
            # Presence findings depend on coverage_complete, which is
            # only known here - so evaluate again with it set.
            result.coverage_complete = True
            result.findings = []
            pipeline.engine.evaluate(result)

        try:
            from src.report.builder import extract_evidence_crops

            extract_evidence_crops(result, Path("data/evidence") / result.scan_id)
        except Exception as exc:          # evidence must never fail the scan
            print(f"  (evidence crops failed: {exc})")

        out = (Path(args.out) if args.out
               else Path("data/reports") / f"{photo.stem}.{args.fmt}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.fmt == "json":
            import json

            out.write_text(json.dumps(result.to_dict(), indent=1, ensure_ascii=False,
                                      default=str), encoding="utf-8")
        else:
            out = build_report(result, out, fmt=args.fmt) or out
        _print_scan(result, out, dict(result.ocr_stats or {}))
        print(f"  total        : {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
