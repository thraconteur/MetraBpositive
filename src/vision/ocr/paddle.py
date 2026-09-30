"""
PaddleOCR backend - the only real OCR engine in this project.

Written against paddleocr 3.7 / paddlex 3.x. Every setting below was
checked against the installed library source, not assumed; the reasons
are recorded next to each one because several defaults are actively
wrong for photographed packaging.

WHAT THIS CHANGES FROM A PLAIN `PaddleOCR(lang="en").predict(img)`
-------------------------------------------------------------------
1. Document preprocessing is OFF.
   PaddleOCR 3.x runs a document orientation classifier and the UVDoc
   unwarping model by default (paddlex/configs/pipelines/OCR.yaml:
   use_doc_orientation_classify: True, use_doc_unwarping: True).
   Detection and cropping then run on the PREPROCESSED image
   (paddlex .../ocr/pipeline.py feeds `doc_preprocessor_images` to the
   detector), so every returned polygon is in the coordinates of a
   rotated / warped copy - not of the image we passed in. Everything
   downstream here measures pixels at those coordinates: glyph height,
   clear space, panel placement, evidence crops. With unwarping on they
   point at the wrong pixels, which is exactly the signature of the
   impossible glyph ratios (0.07, 3.56) seen on real photos. UVDoc is
   also a heavy model, and a large share of per-photo runtime. Geometry
   is this pipeline's job (ArUco rectification), not Paddle's.

2. Detection upscales small inputs.
   The pipeline default is limit_type="min", limit_side_len=64, which
   never enlarges anything. A 1600x954 phone crop keeps 1.5 mm consumer
   care text at a handful of pixels, and small print is missed or
   merged. Raising limit_side_len makes Paddle scale the DETECTION input
   up (max_side_limit=4000 still caps it). Recognition crops still come
   from our image, so coordinates are unaffected.

3. Low-resolution inputs are upscaled on the READ path only.
   Recognition crops are cut from the image we pass in, so a tiny image
   gives the recogniser tiny crops. Below `upscale_short_side_below` the
   image is Lanczos-upscaled for OCR and every polygon is divided back
   to input coordinates. Measurement never sees the upscaled pixels.

4. Weak lines get a second look.
   Regions scoring below `refine_below_score` are re-cropped with extra
   margin (descenders survive: a clipped "g" is how "6 g" becomes "69"),
   upscaled, contrast-normalised, and re-recognised. The higher-scoring
   read wins. Bounded by `refine_max_regions` so runtime stays flat.

5. Bilingual labels.
   lang="hi" does not load the strong model: it resolves to
   devanagari_PP-OCRv5_mobile_rec, while lang="en" gets
   PP-OCRv6_medium_rec. Running Hindi-only therefore reads the English
   half of every bilingual label with a weaker model. Instead, detection
   and recognition run once in English, and each region is re-read by
   the Devanagari recogniser; the Devanagari read replaces the English
   one only where it actually contains Devanagari script.

4b. Curved lines get a curved read.
   A line printed on a bent pouch, a bottle or a can curves; a straight
   detection box cannot follow it, so at one end it takes in the next
   line and the recogniser reads a mix of both ("ADAT0-3-055RRTETOUS..."
   for "AND AT 1800-203-0515 OR WRITE TO US AT paperboat@..." on a real
   Paper Boat pouch). Lines still weak after (4) are detected again
   locally with POLYGON boxes, and each polygon is unwarped into a
   straight strip before recognition - PaddleX's curved (seal) text
   path. The new read replaces the old only when it is clearly better.

4c. Address blocks are detected twice.
   Tightly spaced lines, slightly tilted, come back from detection as ONE
   box, and the recogniser reads one of the two lines: on a real Amul
   carton "Gujarat Co-operative Milk Marketing Federation Ltd.," vanished
   into the box of "Amul Dairy Road, Anand ...". The block under each
   "Marketed by" / "Mfd. by" / "Customer Care" label is detected again
   locally (the polygon detector of 4b), a box that holds several lines is
   split into them, and lines nobody read are added.

4d. A small label is read again, magnified.
   When all the text found covers under a third of the photo, the text
   area is cropped and read again at the detector's preferred scale; new
   lines are merged in and better reads replace worse ones.

6. Vertical text is flagged, not merged.
   Paddle already rotates tall crops before recognition, so inkjet date
   codes printed at 90 degrees are often read correctly. But their boxes
   are tall and narrow, and a row-based line grouper then splices them
   into whatever horizontal line shares their centre. Spans are marked
   `vertical=True` so extraction keeps them on a line of their own.

7. Vertical inkjet bands are re-read rotated.
   A batch code printed at 90 degrees ("60702814YA 04:52 - MAR/26-
   NOV/26-D" down the edge of a real Maggi sachet) comes out of detection
   as a column of tiny fragments that the recogniser reads as 5, 寸,
   ON, 心, M - the manufacture date is lost. A narrow column of such
   fragments is cropped as one band, rotated both ways, and read again as
   a whole; the rotation with the better read replaces the fragments.

8. Chinese glyphs are dropped.
   PP-OCRv6's recogniser is Chinese/English; on Indian packs it turns
   smudges and bar-code edges into 印, 心, 出. They are never real text
   here, so they are removed (a region left empty is skipped).

9. Results are cached on disk.
   Keyed by the exact pixels passed in plus the full config, so
   re-running a photo after an extraction or rules change skips the
   model entirely - model load and inference are the slow part, and
   they no longer repeat while iterating on anything downstream.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ...core.schema import BBox, TextSpan
from .base import OCRBackend

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = _PROJECT_ROOT / "data" / ".ocr_cache"
DEFAULT_DUMP_DIR = _PROJECT_ROOT / "data" / "ocr_dumps"
OCR_CODE_VERSION = "2026-09-30a"

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Printable ASCII plus the rupee sign and a few typographic marks.
_LATIN_ONLY = re.compile(r"[\x20-\x7e₹°®©™–—’‘“”•·×µ]*")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")


def mkldnn_safe() -> bool:
    """oneDNN works on this PaddlePaddle? 3.3.0 / 3.3.1 break it; 3.2.x is
    fine (install it with: pip install paddlepaddle==3.2.2)."""
    try:
        import paddle

        ver = tuple(int(x) for x in re.findall(r"\d+", paddle.__version__)[:2])
    except Exception:
        return False
    return ver < (3, 3)


def _mkldnn(c) -> bool:
    return c.enable_mkldnn if c.enable_mkldnn is not None else mkldnn_safe()


def _overlaps_any(regions: list[dict], new: dict, frac: float) -> bool:
    q = cv2.convexHull(np.asarray(new["poly_ocr"], dtype=np.float32).reshape(-1, 2))
    a = float(cv2.contourArea(q)) or 1.0
    for r in regions:
        o = cv2.convexHull(np.asarray(r["poly_ocr"], dtype=np.float32).reshape(-1, 2))
        try:
            inter, _ = cv2.intersectConvexConvex(q, o)
        except cv2.error:
            inter = 0.0
        if inter / min(a, float(cv2.contourArea(o)) or 1.0) > frac:
            return True
    return False


def _merge_region(regions: list[dict], new: dict) -> int:
    """Add a region from another read of the same image, or let it replace
    the overlapping one(s) it reads better. Returns 1 if the list changed.

    Overlap is measured on the rotated quadrilaterals, not their upright
    bounding boxes: three diagonal inkjet lines on a can base have upright
    boxes that all overlap, and would otherwise collapse into one.

    One better line can replace SEVERAL fragments of itself: a sharp photo
    of a can base read "HRP" and "Rs.125." as two pieces where a softer
    read had the whole "MRP Rs.125"."""
    def quad(r):
        p = np.asarray(r["poly_ocr"], dtype=np.float32).reshape(-1, 2)
        return cv2.convexHull(p)
    if not _visible(new["text"]) or new["score"] < 0.5:
        return 0
    nq_poly = quad(new)
    n_area = float(cv2.contourArea(nq_poly)) or 1.0
    nq = new["score"] * len(_visible(new["text"]))
    same = []
    for i, r in enumerate(regions):
        op = quad(r)
        o_area = float(cv2.contourArea(op)) or 1.0
        try:
            inter, _ = cv2.intersectConvexConvex(nq_poly, op)
        except cv2.error:
            inter = 0.0
        small = min(n_area, o_area)
        if inter / small > 0.3:
            if _different_rows(nq_poly, op):
                # Parallel lines, one above the other: "MFD:03/JUL/26" and
                # "EXP:03/JUL/28" on a can base look alike as text and their
                # slanted quads overlap, but they are two lines.
                continue
            a, b = _visible(new["text"]).lower(), _visible(r["text"]).lower()
            if (a not in b and b not in a
                    and difflib.SequenceMatcher(None, a, b).ratio() < 0.5
                    and inter / small < 0.8):
                # Different words in overlapping quads: two neighbouring
                # diagonal lines ("MRP Rs.125" sits inside the rotated box
                # of the "MFD..." line above it on a can base). Keep both.
                continue
            same.append((i, inter / o_area))
    if not same:
        regions.append(new)
        return 1
    # The fragments mostly inside the new quad are pieces of it; so is
    # the single best overlap.
    pieces = [i for i, frac in same if frac > 0.6] or [max(same, key=lambda t: t[1])[0]]
    old_q = sum(regions[i]["score"] * len(_visible(regions[i]["text"])) for i in pieces)
    old_n = sum(len(_visible(regions[i]["text"])) for i in pieces)
    new_n = len(_visible(new["text"]))
    if len(pieces) == 1 and abs(new_n - old_n) <= 0.25 * max(new_n, old_n):
        # Two reads of the same line, about as long: the surer one wins
        # ("HF0:03JUL/26" at 0.94 over "HF0:03-JUL/260" at 0.91).
        better = new["score"] > regions[pieces[0]]["score"] + 0.02
    else:
        # One line over several of its fragments needs no extra margin:
        # joining them is itself the gain.
        better = nq > old_q * (1.1 if len(pieces) == 1 else 1.0)
    if better:
        keep = [r for k, r in enumerate(regions) if k not in set(pieces)]
        regions[:] = keep + [new]
        return 1
    return 0


# Labels that start an address / consumer-care block (section 4c).
_BLOCK_LABEL = re.compile(
    r"(?:marketed|manufactured|packed|imported|distributed|mfd|mfg|mkt|mktd)\.?\s*(?:&\s*\w+\s*)?by"
    r"|(?:consumer|customer)\s*(?:care|complaints?|relations|helpline)|name\s*&\s*add?r?e?s", re.I)


# Printed pointers to text on an edge or seal (section on sideways reads).
_SIDE_REF = re.compile(r"coding\s+area|\bsee\b.{0,20}?\b(?:edge|seal|side\s+seal)\b", re.I)


def _poly_area(q) -> float:
    return float(cv2.contourArea(np.asarray(q, np.float32).reshape(-1, 2))) or 1.0


def _inter_area(qa, qb) -> float:
    try:
        inter, _ = cv2.intersectConvexConvex(
            cv2.convexHull(np.asarray(qa, np.float32).reshape(-1, 2)),
            cv2.convexHull(np.asarray(qb, np.float32).reshape(-1, 2)))
        return float(inter)
    except cv2.error:
        return 0.0


def _absorb_block(regions: list[dict], new: list[dict]) -> int:
    """
    Fold a local re-detection of a block into the regions:
      * an old box holding two or more new lines was lines merged by the
        detector - it is replaced by those lines (if they read more text);
      * a new line that no old box covers is added;
      * anything else (the same line read again) is left as it was.
    """
    changed = 0
    used = set()
    for i in range(len(regions) - 1, -1, -1):
        r = regions[i]
        inside = [k for k, n in enumerate(new) if k not in used
                  and _inter_area(n["poly_ocr"], r["poly_ocr"]) / _poly_area(n["poly_ocr"]) > 0.6]
        if len(inside) >= 2:
            rows = sorted(inside, key=lambda k: np.asarray(new[k]["poly_ocr"])[:, 1].mean())
            # Distinct rows, not one line cut in two side by side.
            ys = [np.asarray(new[k]["poly_ocr"])[:, 1].mean() for k in rows]
            if max(ys) - min(ys) < 0.5 * min(cv2.minAreaRect(
                    np.asarray(new[k]["poly_ocr"], np.float32))[1][1] or 1 for k in rows):
                continue
            n_old = len(_visible(r["text"]))
            n_new = sum(len(_visible(new[k]["text"])) for k in rows)
            if n_new >= 1.3 * n_old:
                regions[i:i + 1] = [new[k] for k in rows]
                used.update(rows)
                changed += 1
    for k, n in enumerate(new):
        if k in used:
            continue
        a = _visible(n["text"]).lower()
        dup = False
        for r in regions:
            cov = _inter_area(n["poly_ocr"], r["poly_ocr"]) / _poly_area(n["poly_ocr"])
            if cov > 0.6:
                dup = True
                break
            if cov > 0.1:
                b = _visible(r["text"]).lower()
                if a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() > 0.6:
                    dup = True
                    break
        if not dup:
            # A line the first detection never found ("Gujarat Co-operative
            # Milk Marketing Federation Ltd.," - it overlapped the box of the
            # line under it by a third, a different line).
            regions.append(n)
            changed += 1
    return changed


def _different_rows(qa: np.ndarray, qb: np.ndarray) -> bool:
    """Are two text quads on different lines? Their centres are offset
    ACROSS the text direction by more than half the (smaller) text height."""
    (ca, sa, aa) = cv2.minAreaRect(np.asarray(qa, np.float32).reshape(-1, 2))
    (cb, sb, _) = cv2.minAreaRect(np.asarray(qb, np.float32).reshape(-1, 2))
    th = max(2.0, min(min(sa), min(sb)))
    # Direction of the longer side of quad a.
    ang = np.radians(aa if sa[0] >= sa[1] else aa + 90.0)
    d = np.array(cb) - np.array(ca)
    across = abs(-np.sin(ang) * d[0] + np.cos(ang) * d[1])
    return across > 0.6 * th


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Four corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]],
                    dtype=np.float32)


_PRICE_CTX = re.compile(r"\b(?:[uo0]sp|m\.?\s*r\.?\s*p|rs|mrp)\b|/\s*(?:ml|g|kg|l)\b|\bper\s+\d*\s*(?:ml|g|kg)\b", re.I)


def _visible(text: str) -> str:
    """Region text with CJK noise removed and whitespace tidied. In a line
    about prices, a CJK glyph right before a number is the rupee sign
    misread ("29/08/26,天10:USP车 0.071/ml", real Paper Boat inkjet)."""
    t = text or ""
    if _CJK.search(t) and _PRICE_CTX.search(_CJK.sub(" ", t)):
        t = re.sub(r"(?:%s)(?=\s?\d)" % _CJK.pattern, "₹", t)
    return re.sub(r"\s+", " ", _CJK.sub("", t)).strip()

# Loaded models are shared across backend instances in one process. A
# fresh CompliancePipeline() per test or per stability run would
# otherwise reload several hundred MB of weights each time.
_PIPELINE_CACHE: dict[str, object] = {}
_SECONDARY_UNAVAILABLE: set[str] = set()
_RECOGNISER_CACHE: dict[str, object] = {}
_POLY_DET_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class PaddleConfig:
    lang: str = "en"
    ocr_version: Optional[str] = None          # None -> library default
    device: Optional[str] = None               # "cpu", "gpu:0", ... None -> auto
    use_textline_orientation: bool = True      # fixes 180-degree lines
    text_det_limit_type: str = "min"
    text_det_limit_side_len: int = 1280
    text_det_thresh: Optional[float] = None
    text_det_box_thresh: Optional[float] = None
    text_det_unclip_ratio: Optional[float] = None
    upscale_short_side_below: int = 1200
    upscale_factor: float = 2.0
    max_long_side: int = 4000                  # Paddle's own max_side_limit
    # Detection input cap for large photos (recognition keeps full res).
    # 2880 = what a 580x1280 photo reaches after the x2 upscale + Paddle's
    # "min side 1280" - the scale every tuning on real photos was done at.
    det_max_long_side: int = 2880
    refine_below_score: float = 0.80
    refine_max_regions: int = 20
    refine_scale: float = 3.0
    # Curved re-read (section 4b) of lines still below this score; 0 disables.
    curved_reread_below: float = 0.80
    curved_max_regions: int = 12
    # Turn a sideways / upside-down / tilted photo upright before reading
    # (section 7); decided from a quick read of a 960 px copy.
    auto_orient: bool = True
    # Zoomed re-read when all the text found covers less than this share
    # of the photo (4d); 0 disables.
    zoom_below_area: float = 0.35
    # Re-detect the block under each address / consumer-care label (4c).
    block_redetect: bool = True
    # Hindi second reader, ON by default: Indian packs print declarations in
    # Hindi too, and the English model reads Devanagari as junk ("ChRR",
    # "anRif").
    # None / "" turns it off. The English model reads Hindi as confident junk
    # ("aDhkt Kudra" at 0.95; on 84 real photos at most 0.933), so only
    # plain-Latin lines at >= 0.97 skip the Hindi re-read - about 60% of
    # lines, none of the 110 the Hindi read won on those photos.
    secondary_lang: Optional[str] = "hi"
    secondary_below_score: float = 0.97
    reread_vertical_bands: bool = True         # rotated re-read of inkjet codes
    # Second read of a SPARSE photo (a can base, a carton top, a crimp):
    # contrast-equalised, and inverted, with a lower detection threshold.
    contrast_reread_below: int = 6             # strong lines; 0 disables
    # Read the photo turned 90 degrees both ways too, for text printed
    # sideways along an edge (price / date / volume on a pouch's side seal).
    # Off by default: two extra reads per photo.
    sideways_reread: bool = False
    # ...but DO read sideways when the pack itself points to a coding area or
    # a side / edge / seal ("For MRP, USP, NET VOL. ... SEE CODING AREA" on a
    # real Vim pouch, whose code runs up the pouch edge).
    sideways_when_referenced: bool = True
    # oneDNN (MKL-DNN): ~3x faster detection on CPU with IDENTICAL results
    # (all 38 real photos compared, 28 Sept 2026). None = on when the
    # installed PaddlePaddle is not 3.3.x, whose oneDNN path crashes with
    # "ConvertPirAttribute2RuntimeAttribute not support" (Paddle #77340).
    enable_mkldnn: Optional[bool] = None
    cpu_threads: Optional[int] = None

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def _config_from_env(**overrides) -> PaddleConfig:
    """Environment knobs, so the API and scripts can be tuned without edits."""
    env = {}
    if os.environ.get("SIH_OCR_LANG"):
        env["lang"] = os.environ["SIH_OCR_LANG"]
    if os.environ.get("SIH_OCR_SECONDARY_LANG"):
        env["secondary_lang"] = os.environ["SIH_OCR_SECONDARY_LANG"]
    if os.environ.get("SIH_OCR_DEVICE"):
        env["device"] = os.environ["SIH_OCR_DEVICE"]
    if os.environ.get("SIH_OCR_DET_MAX"):
        # Long side the text detector sees (recognition stays full-res).
        # Smaller = faster on big phone photos; see docs/results_log.md.
        env["det_max_long_side"] = int(os.environ["SIH_OCR_DET_MAX"])
    if os.environ.get("SIH_OCR_MKLDNN", "") == "1":
        # Much faster on CPU. Off by default only because some Windows CPUs
        # crash with it (ConvertPirAttribute2RuntimeAttribute) - if yours
        # doesn't, leave it on.
        env["enable_mkldnn"] = True
    env.update({k: v for k, v in overrides.items() if v is not None})
    return PaddleConfig(**env)


class PaddleOCRBackend(OCRBackend):
    name = "paddleocr"
    supports_languages = ("en", "hi")

    def __init__(
        self,
        config: Optional[PaddleConfig] = None,
        lang: Optional[str] = None,
        cache_dir: Optional[str | Path] = DEFAULT_CACHE_DIR,
        dump_dir: Optional[str | Path] = DEFAULT_DUMP_DIR,
        use_textline_orientation: Optional[bool] = None,
    ):
        overrides = {"lang": lang}
        if use_textline_orientation is not None:
            overrides["use_textline_orientation"] = use_textline_orientation
        self.config = config or _config_from_env(**overrides)
        if os.environ.get("SIH_OCR_CACHE", "1") == "0":
            cache_dir = None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self._init_error: Optional[str] = None
        # Timings of the most recent call, for scripts that report them.
        self.last_stats: dict = {}
        # For tests: inject a fake with the same .predict() contract.
        self._pipeline_override = None
        self._recogniser_override: dict[str, object] = {}

    # -----------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------
    def _pipeline(self):
        if self._pipeline_override is not None:
            return self._pipeline_override
        k = self.config.key()
        if k in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[k]
        from paddleocr import PaddleOCR

        c = self.config
        kwargs = dict(
            lang=c.lang,
            ocr_version=c.ocr_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=c.use_textline_orientation,
            text_det_limit_type=c.text_det_limit_type,
            text_det_limit_side_len=c.text_det_limit_side_len,
            text_det_thresh=c.text_det_thresh,
            text_det_box_thresh=c.text_det_box_thresh,
            text_det_unclip_ratio=c.text_det_unclip_ratio,
            text_rec_score_thresh=0.0,
            enable_mkldnn=_mkldnn(c),
        )
        if c.device:
            kwargs["device"] = c.device
        if c.cpu_threads:
            kwargs["cpu_threads"] = c.cpu_threads
        kwargs = {k2: v for k2, v in kwargs.items() if v is not None}
        t0 = time.perf_counter()
        pipe = PaddleOCR(**kwargs)
        self.last_stats["model_load_s"] = round(time.perf_counter() - t0, 2)
        _PIPELINE_CACHE[k] = pipe
        return pipe

    def _rec_model_name(self, lang: str) -> str:
        """The recogniser PaddleOCR itself would pick for `lang`."""
        try:
            from paddleocr._pipelines.ocr import PaddleOCR

            _, rec = PaddleOCR._get_ocr_model_names(
                None, lang, self.config.ocr_version if lang == self.config.lang else None
            )
            if rec:
                return rec
        except Exception:
            pass
        return {"en": "PP-OCRv6_medium_rec",
                "hi": "devanagari_PP-OCRv5_mobile_rec"}.get(lang, "PP-OCRv6_medium_rec")

    def _recogniser(self, lang: str):
        if lang in self._recogniser_override:
            return self._recogniser_override[lang]
        name = self._rec_model_name(lang)
        k = f"{name}|{self.config.device}|{_mkldnn(self.config)}"
        if k in _RECOGNISER_CACHE:
            return _RECOGNISER_CACHE[k]
        from paddleocr import TextRecognition

        kwargs = {"model_name": name, "enable_mkldnn": _mkldnn(self.config)}
        if self.config.device:
            kwargs["device"] = self.config.device
        rec = TextRecognition(**kwargs)
        _RECOGNISER_CACHE[k] = rec
        return rec

    def available(self) -> bool:
        if self._pipeline_override is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            import paddleocr  # noqa: F401
            return True
        except Exception as exc:  # pragma: no cover - environment
            self._init_error = str(exc)
            return False

    def warm_up(self) -> None:
        """Load models now rather than on the first scan."""
        self._pipeline()

    # -----------------------------------------------------------------
    # Recognition
    # -----------------------------------------------------------------
    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        stats: dict = {"cache_hit": False}
        if getattr(self, "_upright", None) is not None:
            stats["upright"] = {"turn": self._upright[0], "tilt": self._upright[1]}
            self._upright = None
        self.last_stats = stats
        t_start = time.perf_counter()

        cache_key = self._cache_key(image)
        cached = self._cache_read(cache_key)
        if cached is not None:
            stats["cache_hit"] = True
            regions = cached
        else:
            regions = self._run(image, stats)
            self._cache_write(cache_key, regions)

        spans = [self._to_span(r) for r in regions if _visible(r["text"])]
        stats["regions"] = len(spans)
        stats["total_s"] = round(time.perf_counter() - t_start, 2)
        if source and self.dump_dir is not None:
            self._dump(source, image, regions, stats)
        return spans

    def _run(self, image: np.ndarray, stats: dict) -> list[dict]:
        c = self.config
        ocr_img, scale = self._ocr_input(image)
        stats["ocr_scale"] = scale

        t0 = time.perf_counter()
        results = self._predict_full(ocr_img)
        stats["predict_s"] = round(time.perf_counter() - t0, 2)

        regions = self._regions_from(results, scale)

        # Sparse photo: stamped / embossed / dot-matrix codes on metal and
        # glossy card. On a real Monster can base the default pass read
        # NOTHING; contrast-equalised and inverted, with a lower detection
        # threshold, it read "MRP", "Rs.125", "USP Rs.0.36/m", "MFD 03 JUL/26".
        strong = sum(1 for r in regions if r["score"] >= 0.8 and len(_visible(r["text"])) >= 3)
        if c.contrast_reread_below and strong < c.contrast_reread_below:
            t0 = time.perf_counter()
            added = 0
            # On the photo as taken, not the upscaled copy: on the can base
            # the upscaled read merged three diagonal lines into one box.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
            img3 = np.ascontiguousarray(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR))
            cap = ({"text_det_limit_type": "max", "text_det_limit_side_len": int(c.det_max_long_side)}
                   if c.det_max_long_side and max(img3.shape[:2]) > c.det_max_long_side else {})
            for kw in ({**cap}, {"text_det_thresh": 0.15, "text_det_box_thresh": 0.3, **cap}):
                try:
                    res2 = self._pipeline().predict(img3, **kw)
                except TypeError:
                    res2 = self._pipeline().predict(img3)
                except Exception:
                    logger.warning("Contrast re-read failed", exc_info=True)
                    continue
                for r in self._regions_from(res2, 1.0, to_ocr=scale):
                    r["engine"] = "paddleocr(contrast)"
                    added += _merge_region(regions, r)
            # Dot-matrix print on a SHARP photo: the recogniser sees separate
            # dots, not strokes. The same can base photographed at 580x1280
            # read "MFD:03 JUL/26 06:43", "MRP Rs.125"; at 1856x4096 it read
            # "HF0:03-JUL/260" and "HRP". A softened copy joins the dots.
            if max(image.shape[:2]) > 2000:
                try:
                    ds = 1280.0 / float(max(image.shape[:2]))
                    small = cv2.resize(image, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
                    s_img, s_up = self._ocr_input(small)
                    res4 = self._pipeline().predict(s_img)
                    for r in self._regions_from(res4, ds * s_up, to_ocr=scale / (ds * s_up)):
                        r["engine"] = "paddleocr(softened)"
                        added += _merge_region(regions, r)
                except Exception:
                    logger.warning("Softened re-read failed", exc_info=True)
            stats["contrast_added"] = added
            stats["contrast_s"] = round(time.perf_counter() - t0, 2)

        # The label is small in the frame (a cup held at arm's length):
        # read the text area again, magnified. On a real Kissan cup the
        # red consumer-care lines were found by NO whole-photo detection
        # scale, and at once in a crop of the text area.
        if c.zoom_below_area and regions:
            t0 = time.perf_counter()
            try:
                stats["zoom_added"] = self._zoom(ocr_img, regions, scale)
            except Exception:
                logger.warning("Zoomed re-read failed", exc_info=True)
            stats["zoom_s"] = round(time.perf_counter() - t0, 2)

        if c.sideways_reread or (c.sideways_when_referenced and _SIDE_REF.search(
                " ".join(r["text"] for r in regions))):
            stats["sideways_triggered"] = not c.sideways_reread
            t0 = time.perf_counter()
            H, W = ocr_img.shape[:2]
            added = 0
            for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                try:
                    res3 = self._pipeline().predict(np.ascontiguousarray(cv2.rotate(ocr_img, rot)))
                except Exception:
                    logger.warning("Sideways re-read failed", exc_info=True)
                    continue
                for r in self._regions_from(res3, 1.0):
                    pts = np.asarray(r["poly_ocr"], dtype=np.float32)
                    if rot == cv2.ROTATE_90_CLOCKWISE:      # rotated (x', y') <- (y', H-1-x')
                        back = np.stack([pts[:, 1], (H - 1) - pts[:, 0]], axis=1)
                    else:                                   # rotated (x', y') <- (W-1-y', x')
                        back = np.stack([(W - 1) - pts[:, 1], pts[:, 0]], axis=1)
                    r["poly_ocr"] = back.tolist()
                    r["poly"] = (back / scale).tolist()
                    r["engine"] = "paddleocr(sideways)"
                    r["vertical"] = True
                    # Only text the upright read did not already cover:
                    # turned sideways, ordinary lines read as junk.
                    if r["score"] >= 0.85 and not _overlaps_any(regions, r, 0.1):
                        regions.append(r)
                        added += 1
            stats["sideways_added"] = added
            stats["sideways_s"] = round(time.perf_counter() - t0, 2)

        if c.reread_vertical_bands and regions:
            t0 = time.perf_counter()
            regions, stats["vertical_bands"] = self._vertical_bands(ocr_img, regions, scale)
            stats["vertical_s"] = round(time.perf_counter() - t0, 2)

        if c.refine_below_score and regions:
            t0 = time.perf_counter()
            stats["refined"] = self._refine(ocr_img, regions)
            stats["refine_s"] = round(time.perf_counter() - t0, 2)

        if c.block_redetect and regions and self._pipeline_override is None:
            t0 = time.perf_counter()
            try:
                stats["block_lines"] = self._block_redetect(ocr_img, regions, scale)
            except Exception:                  # an extra pass must never cost the read
                logger.warning("Block re-detection failed", exc_info=True)
            stats["block_s"] = round(time.perf_counter() - t0, 2)

        if c.curved_reread_below and regions and self._pipeline_override is None:
            t0 = time.perf_counter()
            try:
                stats["curved"] = self._curved(ocr_img, regions, scale)
            except Exception:
                logger.warning("Curved re-read failed", exc_info=True)
            stats["curved_s"] = round(time.perf_counter() - t0, 2)

        if c.secondary_lang and regions:
            t0 = time.perf_counter()
            stats["secondary_replaced"] = self._secondary(ocr_img, regions)
            stats["secondary_s"] = round(time.perf_counter() - t0, 2)

        # The same orientation test on the finished read (every pass): faint dot-matrix on a
        # can base gives the quick 960 px check nothing to go on.
        if c.auto_orient:
            try:
                good = [r for r in regions if r["score"] >= 0.6 and len(_visible(r["text"])) >= 4]
                after = decide_upright(good, ocr_img.shape[:2])
                if after[0] == 180 and not self._flip_by_reading(ocr_img, good):
                    after = (0, decide_upright(good, ocr_img.shape[:2], force_turn=0)[1])
                if after[0] in (90, 270):
                    t = self._side_by_reading(ocr_img, good, after[0])
                    after = (t, decide_upright(good, ocr_img.shape[:2], force_turn=t)[1])
                if after != (0, 0.0):
                    stats["upright_after"] = {"turn": after[0], "tilt": after[1]}
            except Exception:
                logger.warning("Orientation re-check failed", exc_info=True)
        for r in regions:
            r.pop("poly_ocr", None)
        return regions

    def _predict_full(self, img: np.ndarray):
        """
        The main read. On a large photo, DETECTION runs at the scale the
        detector finds lines best (long side <= det_max_long_side), while
        RECOGNITION still crops the lines from the full-resolution photo.

        Measured on real photos: the same Amul carton sent at 1856x4096
        lost two whole address lines ("Gujarat Co-operative Milk Marketing
        Federation Ltd., Amul Dairy Road ...") that the detector found when
        it saw ~1280x2800 - the size a WhatsApp copy reaches after our
        upscaling. The detector is also twice as fast on half the pixels.
        """
        c = self.config
        if c.det_max_long_side and max(img.shape[:2]) > c.det_max_long_side:
            try:
                return self._pipeline().predict(
                    img, text_det_limit_type="max",
                    text_det_limit_side_len=int(c.det_max_long_side))
            except TypeError:                  # a predictor without per-call limits
                pass
        return self._pipeline().predict(img)

    @staticmethod
    def _regions_from(results, scale: float, to_ocr: float = 1.0) -> list[dict]:
        """Paddle output -> region dicts. `scale` maps the coordinates Paddle
        saw to the input photo; `to_ocr` maps them to the upscaled OCR image
        (for a read made on the photo as taken, scale=1, to_ocr=ocr scale)."""
        regions: list[dict] = []
        for res in results or []:
            try:
                texts, scores, polys = res["rec_texts"], res["rec_scores"], res["rec_polys"]
            except (KeyError, TypeError):
                continue
            try:
                angles = list(res["textline_orientation_angles"])
            except (KeyError, TypeError):
                angles = []
            for k, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                if pts.shape[0] < 4:
                    continue
                regions.append({
                    # Paddle returns the textline class: 0 upright, 1 upside down
                    # (180 degrees). Stored as degrees.
                    "tl_angle": (180 if k < len(angles) and angles[k] in (1, 180) else 0),
                    "text": str(text),
                    "score": float(score),
                    "poly_ocr": (pts * to_ocr).tolist(),   # in ocr_img coords
                    "poly": (pts / scale).tolist(),    # in input coords
                    "engine": "paddleocr",
                    "language": "hi" if _DEVANAGARI.search(str(text)) else "en",
                })
        return regions

    def _ocr_input(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        c = self.config
        h, w = image.shape[:2]
        if not c.upscale_short_side_below or min(h, w) >= c.upscale_short_side_below:
            return image, 1.0
        # Up to the target short side, not blindly x2: a 720x1280 phone
        # photo becomes 1200x2133 (2.6 MP), not 1440x2560 (3.7 MP) - the
        # detector's CPU time grows with pixel count.
        scale = min(c.upscale_factor, c.upscale_short_side_below / float(min(h, w)),
                    c.max_long_side / float(max(h, w)))
        if scale <= 1.01:
            return image, 1.0
        up = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                        interpolation=cv2.INTER_LANCZOS4)
        return up, scale

    # -- second look at weak regions ----------------------------------
    def _refine(self, ocr_img: np.ndarray, regions: list[dict]) -> int:
        c = self.config
        weak = sorted(
            (i for i, r in enumerate(regions) if r["score"] < c.refine_below_score),
            key=lambda i: regions[i]["score"],
        )[: c.refine_max_regions]
        if not weak:
            return 0

        crops, owners = [], []
        for i in weak:
            base = crop_region(ocr_img, np.asarray(regions[i]["poly_ocr"]), pad_ratio=0.25)
            if base is None:
                continue
            up = cv2.resize(base, None, fx=c.refine_scale, fy=c.refine_scale,
                            interpolation=cv2.INTER_CUBIC)
            crops.append(up)
            owners.append(i)
            crops.append(_clahe(up))
            owners.append(i)
        if not crops:
            return 0

        try:
            outs = _rec_batched(self._recogniser(c.lang), crops)
        except Exception:
            logger.warning("Refinement recognition failed", exc_info=True)
            return 0

        best: dict[int, tuple[str, float]] = {}
        for i, out in zip(owners, outs):
            text, score = _rec_output(out)
            if text and score > best.get(i, ("", -1.0))[1]:
                best[i] = (text, score)

        replaced = 0
        for i, (text, score) in best.items():
            # A margin, so a coin-flip between two equally weak reads
            # does not churn the output from run to run. And the new read
            # must not LOSE the line: on a real sachet a 0.75 read of
            # "Aniseed, Black pepper, Fenugreek, Ginger, Clove," was
            # "improved" to "A" at 0.95. A read that keeps under 60% of
            # the characters is a different (smaller) crop, not a better
            # read of the same line.
            old_len = len(_visible(regions[i]["text"]))
            new_len = len(_visible(text))
            if new_len == 0 or (regions[i]["score"] >= 0.5 and new_len < 0.6 * old_len):
                continue
            if score >= regions[i]["score"] + 0.05:
                regions[i]["alt_text"] = regions[i]["text"]
                regions[i]["alt_score"] = regions[i]["score"]
                regions[i]["text"], regions[i]["score"] = text, score
                regions[i]["engine"] = "paddleocr(refined)"
                replaced += 1
        return replaced

    # -- rotated re-read of vertical inkjet bands ----------------------
    def _vertical_bands(self, ocr_img: np.ndarray, regions: list[dict], scale: float):
        H, W = ocr_img.shape[:2]
        boxes = []
        for i, r in enumerate(regions):
            pts = np.asarray(r["poly_ocr"], dtype=np.float32).reshape(-1, 2)
            x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            vis = _visible(r["text"])
            suspect = (vis != r["text"].strip() or len(vis) <= 3 or r["score"] < 0.7
                       or (y1 - y0) >= 1.5 * max(1.0, x1 - x0))
            boxes.append((i, x0, y0, x1, y1, suspect))

        # Columns of suspect fragments that overlap horizontally.
        sus = sorted((b for b in boxes if b[5]), key=lambda b: (b[1] + b[3]) / 2)
        clusters: list[list[tuple]] = []
        for b in sus:
            for cl in clusters:
                cx0 = min(c[1] for c in cl); cx1 = max(c[3] for c in cl)
                if min(cx1, b[3]) - max(cx0, b[1]) > 0.3 * min(cx1 - cx0, b[3] - b[1]):
                    cl.append(b)
                    break
            else:
                clusters.append([b])

        replaced_bands = 0
        drop: set[int] = set()
        added: list[dict] = []
        for cl in clusters:
            if len(cl) < 3:
                continue
            x0 = min(c[1] for c in cl); x1 = max(c[3] for c in cl)
            y0 = min(c[2] for c in cl); y1 = max(c[4] for c in cl)
            bw, bh = x1 - x0, y1 - y0
            if bh < 3 * bw or bw > 0.15 * W:
                continue
            pad = int(0.25 * bw) + 4
            X0, Y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            X1, Y1 = min(W, int(x1) + pad), min(H, int(y1) + pad)
            crop = ocr_img[Y0:Y1, X0:X1]
            ch, cw = crop.shape[:2]
            if ch < 8 or cw < 8:
                continue
            old_q = sum(regions[c[0]]["score"] * len(_visible(regions[c[0]]["text"]))
                        for c in cl)
            best = None
            for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                rimg = np.ascontiguousarray(cv2.rotate(crop, rot))
                try:
                    # Detection limits for THIS call only. The band is a thin
                    # strip; with the photo-wide "min side 1280" setting Paddle
                    # blew it up 6-16x and one band cost minutes on a CPU.
                    try:
                        res = self._pipeline().predict(
                            rimg, text_det_limit_type="max", text_det_limit_side_len=960)
                    except TypeError:          # a predictor without per-call limits
                        res = self._pipeline().predict(rimg)
                except Exception:
                    logger.warning("Rotated re-read failed", exc_info=True)
                    continue
                reads = []
                for out in res or []:
                    try:
                        items = zip(out["rec_texts"], out["rec_scores"], out["rec_polys"])
                    except (KeyError, TypeError):
                        continue
                    for text, score, poly in items:
                        vis = _visible(str(text))
                        if not vis:
                            continue
                        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                        if rot == cv2.ROTATE_90_CLOCKWISE:
                            # rotated (x', y') came from crop (y', ch-1-x')
                            back = np.stack([pts[:, 1], (ch - 1) - pts[:, 0]], axis=1)
                        else:
                            # rotated (x', y') came from crop (cw-1-y', x')
                            back = np.stack([(cw - 1) - pts[:, 1], pts[:, 0]], axis=1)
                        back = back + np.float32([X0, Y0])
                        reads.append((vis, float(score), back))
                q = sum(sc * len(t) for t, sc, _ in reads)
                if reads and (best is None or q > best[0]):
                    best = (q, reads)
            if best is None or best[0] <= 1.2 * old_q:
                continue
            if max(len(t) for t, _, _ in best[1]) < 5:
                continue
            drop.update(c[0] for c in cl)
            for text, score, back in best[1]:
                added.append({
                    "text": text,
                    "score": score,
                    "poly_ocr": back.tolist(),
                    "poly": (back / scale).tolist(),
                    "engine": "paddleocr(rotated)",
                    "language": "en",
                })
            replaced_bands += 1

        if not replaced_bands:
            return regions, 0
        return [r for i, r in enumerate(regions) if i not in drop] + added, replaced_bands

    # -- curved re-read (section 4b) ------------------------------------
    def _poly_detector(self):
        c = self.config
        try:
            from paddleocr._pipelines.ocr import PaddleOCR

            det, _ = PaddleOCR._get_ocr_model_names(None, c.lang, c.ocr_version)
        except Exception:
            det = None
        det = det or "PP-OCRv6_medium_det"
        k = f"{det}|{c.device}|{_mkldnn(c)}"
        if k not in _POLY_DET_CACHE:
            from paddlex import create_predictor

            kw = {"device": c.device or "cpu", "enable_mkldnn": _mkldnn(c)}
            pred = create_predictor(det, **kw)
            pred.post_op.box_type = "poly"        # follow the curve
            _POLY_DET_CACHE[k] = pred
        return _POLY_DET_CACHE[k]

    def _curved(self, ocr_img: np.ndarray, regions: list[dict], scale: float) -> int:
        c = self.config
        weak = sorted(
            (i for i, r in enumerate(regions)
             if r["score"] < c.curved_reread_below and len(_visible(r["text"])) >= 4
             and not r.get("vertical")),
            key=lambda i: regions[i]["score"],
        )[: c.curved_max_regions]
        if not weak:
            return 0
        try:
            det = self._poly_detector()
            from paddlex.inference.pipelines.components import CropByPolys
            unwarp = CropByPolys("poly")
        except Exception as exc:
            logger.warning("Curved re-read unavailable (%s)", str(exc).splitlines()[0][:120])
            return 0
        H, W = ocr_img.shape[:2]
        replaced = 0
        for i in weak:
            r = regions[i]
            q = np.asarray(r["poly_ocr"], dtype=np.float32).reshape(-1, 2)
            x0, y0 = q.min(axis=0)
            x1, y1 = q.max(axis=0)
            # Text height = the quad's short side, not its (slanted) box.
            rect = cv2.minAreaRect(q)
            th = max(4.0, min(rect[1]))
            X0, Y0 = int(max(0, x0 - th)), int(max(0, y0 - 0.8 * th))
            X1, Y1 = int(min(W, x1 + th)), int(min(H, y1 + 0.8 * th))
            if X1 - X0 < 16 or Y1 - Y0 < 8:
                continue
            crop = np.ascontiguousarray(ocr_img[Y0:Y1, X0:X1])
            try:
                polys = list(det.predict([crop]))[0]["dt_polys"]
            except Exception:
                logger.warning("Curved detection failed", exc_info=True)
                return replaced
            if polys is None or len(polys) == 0:
                continue
            # Which polygons are THIS line: mostly inside the weak box.
            weak_mask = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillPoly(weak_mask, [np.round(q - [X0, Y0]).astype(np.int32)], 1)
            cands = []
            for pl in polys:
                pa = np.asarray(pl, dtype=np.float32).reshape(-1, 2)
                m = np.zeros_like(weak_mask)
                cv2.fillPoly(m, [np.round(pa).astype(np.int32)], 1)
                area = int(m.sum())
                inter = int((m & weak_mask).sum())
                if area and inter / area >= 0.5:
                    cands.append((inter, pa))
            if not cands:
                continue
            cands.sort(key=lambda t: -t[0])
            main = cands[0][1]
            mh = max(4.0, min(cv2.minAreaRect(main)[1]))
            row = [main] + [pa for _, pa in cands[1:]
                            if abs(pa[:, 1].mean() - main[:, 1].mean()) < 0.6 * mh
                            and (pa[:, 0].max() < main[:, 0].min() or pa[:, 0].min() > main[:, 0].max())]
            row.sort(key=lambda pa: pa[:, 0].min())
            try:
                strips = [x if isinstance(x, np.ndarray) else x["img"]
                          for x in unwarp(crop, [pa for pa in row])]
                outs = _rec_batched(self._recogniser(c.lang), strips)
            except Exception:
                logger.warning("Curved recognition failed", exc_info=True)
                continue
            reads = [_rec_output(o) for o in outs]
            text = " ".join(t for t, _ in reads if t).strip()
            n = sum(len(t) for t, _ in reads) or 1
            score = sum(sc * len(t) for t, sc in reads) / n
            old_len = len(_visible(r["text"]))
            new_len = len(_visible(text))
            # Same acceptance as the weak-line re-read: clearly better,
            # and not a smaller piece of the line.
            if new_len == 0 or (r["score"] >= 0.5 and new_len < 0.6 * old_len):
                continue
            if score < r["score"] + 0.05:
                continue
            pts = np.concatenate(row) + [X0, Y0]
            box = _order_quad(cv2.boxPoints(cv2.minAreaRect(pts.astype(np.float32))))
            r["alt_text"], r["alt_score"] = r["text"], r["score"]
            r["text"], r["score"] = text, float(score)
            r["engine"] = "paddleocr(curved)"
            r["poly_ocr"] = box.tolist()
            r["poly"] = (box / scale).tolist()
            replaced += 1
        return replaced

    # -- which way up is the photo (section 7) ---------------------------
    def note_orientation(self, turn: int, tilt: float) -> None:
        """The pipeline turned the photo itself (after a first read showed it
        on its side); record it with the next read, as upright_rotation does."""
        self._upright = (int(turn), float(tilt))

    def upright_rotation(self, image: np.ndarray, source: Optional[str] = None) -> tuple[int, float]:
        c = self.config
        self._upright = None
        if not c.auto_orient or self._pipeline_override is not None:
            return 0, 0.0
        if self._clearly_upright(image):
            self._upright = (0, 0.0)
            return 0, 0.0
        h, w = image.shape[:2]
        regs, small = [], image
        # 960 px first (quick); a sparse label on a big photo - dot-matrix on
        # a can base - may give too few lines there, so 1800 px next.
        for side in (960.0, 1800.0):
            s = min(1.0, side / float(max(h, w)))
            small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
            try:
                res = self._pipeline().predict(np.ascontiguousarray(small), text_det_limit_type="max",
                                               text_det_limit_side_len=int(side))
            except TypeError:
                res = self._pipeline().predict(np.ascontiguousarray(small))
            except Exception:
                logger.warning("Orientation check failed", exc_info=True)
                return 0, 0.0
            regs = [r for r in self._regions_from(res, 1.0)
                    if r["score"] >= 0.6 and len(_visible(r["text"])) >= 4]
            if len(regs) >= 6 or s >= 1.0:
                break
        turn, tilt = decide_upright(regs, small.shape[:2])
        # The reading tests crop the lines from the FULL photo: at 960 px
        # small print is a few pixels tall and reads as nothing either way.
        k = image.shape[1] / float(small.shape[1])
        full_regs = [dict(r, poly_ocr=(np.asarray(r["poly_ocr"], np.float32) * k).tolist())
                     for r in regs]
        if turn == 180 and not self._flip_by_reading(image, full_regs):
            turn, tilt = 0, decide_upright(regs, small.shape[:2], force_turn=0)[1]
        if turn == 0 and not tilt and len(regs) < 6:
            # Few lines read cleanly in English: a mostly-Hindi label (read as
            # junk either way up by the English model) upside down looked
            # "upright" (audit, 30 Sept: a 580 px Hindi pack at 180 degrees).
            # Read its widest lines both ways in English AND Hindi.
            allr = self._regions_from(res, 1.0)
            if len(allr) >= 4:
                full_all = [dict(r, poly_ocr=(np.asarray(r["poly_ocr"], np.float32) * k).tolist())
                            for r in allr]
                langs = (c.lang,) + ((c.secondary_lang,) if c.secondary_lang
                                     and c.secondary_lang not in _SECONDARY_UNAVAILABLE else ())
                if self._flip_by_reading(image, full_all, langs=langs):
                    turn = 180
        if turn in (90, 270):
            turn = self._side_by_reading(image, full_regs, turn)
            if turn in (90, 270):
                tilt = decide_upright(regs, small.shape[:2], force_turn=turn)[1]
        self._upright = (turn, tilt)
        return turn, tilt

    def _clearly_upright(self, image: np.ndarray) -> bool:
        """
        Fast path of the orientation check. Most photos are taken upright,
        and reading every line (the full check) costs 8-20 s on a label with
        50-130 lines - recognition costs the same per line at 960 px as at
        full size. Detection plus Paddle's line up/down classifier costs
        about 1 s and is enough to say "upright and level" with a wide
        margin. Anything less clear - lines tall, flipped, tilted, or too
        few - goes to the full check, unchanged.
        """
        try:
            inner = self._pipeline().paddlex_pipeline._pipeline
            det_model, tl = inner.text_det_model, inner.textline_orientation_model
        except AttributeError:
            return False
        if tl is None or not self.config.use_textline_orientation:
            return False
        c = self.config
        h, w = image.shape[:2]
        s = min(1.0, 960.0 / float(max(h, w)))
        small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else image
        small = np.ascontiguousarray(small)
        try:
            params = inner.get_text_det_params(960, "max", None, c.text_det_thresh,
                                               c.text_det_box_thresh, c.text_det_unclip_ratio)
            polys = list(inner._sort_boxes(list(det_model([small], **params))[0]["dt_polys"]))
            # Few lines: the up/down classifier's vote is too thin (a mostly-
            # Hindi label upside down had 6 lines, none called flipped).
            if len(polys) < 10:
                return False
            subs, kept = [], []
            for sub, p in zip(inner._crop_by_polys(small, polys), polys):
                if sub is not None and sub.size and sub.shape[0] > 0 and sub.shape[1] > 0:
                    subs.append(sub)
                    kept.append(np.asarray(p, np.float32).reshape(-1, 2))
            angles = [int(np.asarray(o["class_ids"], dtype=np.int64).ravel()[0]) for o in tl(subs)]
        except Exception:
            logger.warning("Quick orientation check failed; using the full one", exc_info=True)
            return False
        tall = wide = flipped = tot = 0.0
        angs, ws = [], []
        for p, a in zip(kept, angles):
            x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
            bw, bh = float(x1 - x0), float(y1 - y0)
            wt = max(bw, bh) / max(1.0, min(bw, bh))       # ~ characters on the line
            tot += wt
            if bh >= 1.5 * max(1.0, bw):
                tall += wt
            elif bw >= 1.5 * max(1.0, bh):
                wide += wt
                flipped += wt if a == 1 else 0.0
                if bw >= 2.0 * max(1.0, bh):
                    angs.append(_line_angle(_order_quad(cv2.boxPoints(cv2.minAreaRect(p)))))
                    ws.append(wt)
        med = abs(float(np.median(angs))) if len(angs) >= 4 else 0.0
        self._quick_upright = dict(n=len(kept), wide=wide / max(tot, 1e-6),
                                   tall=tall / max(tot, 1e-6),
                                   flipped=flipped / max(wide, 1e-6), tilt=med)
        if tot <= 0 or wide < 0.7 * tot or tall > 0.2 * tot or flipped > 0.05 * wide:
            return False
        if med >= 6.0:
            return False
        return True

    def _flip_by_reading(self, img: np.ndarray, regions: list[dict], langs=None) -> bool:
        """
        Upside down? Paddle's per-line classifier says so on decorative
        fonts that are upright (a Maharaja Gulal front was turned 180). Read
        the widest lines as they are and turned 180; turn only if the turned
        read is clearly better.
        """
        wide = []
        for r in regions:
            p = np.asarray(r["poly_ocr"], np.float32).reshape(-1, 2)
            x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
            if (x1 - x0) >= 1.5 * max(1.0, y1 - y0):
                wide.append((x1 - x0, p))
        wide = sorted(wide, key=lambda t: -t[0])[:6]
        same, flip = [], []
        for _, p in wide:
            # A strip cut ALONG the line (as Paddle cuts it): the upright box
            # of a long, slightly slanted line takes in the lines around it.
            crop = crop_region(img, p, pad_ratio=0.15)
            if crop is not None and crop.size:
                same.append(np.ascontiguousarray(crop))
                flip.append(np.ascontiguousarray(cv2.rotate(crop, cv2.ROTATE_180)))
        if not same:
            return False
        best = None
        for lang in (langs or (self.config.lang,)):
            try:
                rec = self._recogniser(lang)
                q = [sum(sc * len(_visible(t)) for t, sc in (_rec_output(o) for o in rec.predict(b)))
                     for b in (same, flip)]
            except Exception:
                logger.warning("Upside-down check failed (%s)", lang, exc_info=True)
                continue
            # the script that reads this label best, either way up, decides
            if best is None or max(q) > max(best):
                best = q
        return bool(best) and best[1] > 1.15 * best[0]

    def _side_by_reading(self, img: np.ndarray, regions: list[dict], guess: int) -> int:
        """
        A photo lying on its side: which side? Read its longest tall lines
        turned clockwise and anticlockwise and keep the direction that reads
        better. Paddle's upside-down classifier is unreliable on sideways
        dot-matrix (a can base turned 270 degrees came back "upright").
        """
        tall = []
        for r in regions:
            p = np.asarray(r["poly_ocr"], np.float32).reshape(-1, 2)
            x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
            if (y1 - y0) >= 1.5 * max(1.0, x1 - x0):
                tall.append((y1 - y0, int(x0), int(y0), int(x1), int(y1)))
        tall = sorted(tall, reverse=True)[:6]
        if not tall:
            return guess
        H, W = img.shape[:2]
        cw, ccw = [], []
        for _, x0, y0, x1, y1 in tall:
            pad = max(2, (x1 - x0) // 4)
            crop = img[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)]
            if crop.size == 0:
                continue
            cw.append(np.ascontiguousarray(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)))
            ccw.append(np.ascontiguousarray(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)))
        if not cw:
            return guess
        try:
            rec = self._recogniser(self.config.lang)
            q = []
            for batch in (cw, ccw):
                outs = [_rec_output(o) for o in rec.predict(batch)]
                q.append(sum(sc * len(_visible(t)) for t, sc in outs))
        except Exception:
            logger.warning("Sideways direction check failed", exc_info=True)
            return guess
        # Turning the CROPS clockwise reads better -> turn the PHOTO clockwise.
        if q[0] > 1.15 * q[1]:
            return 90
        if q[1] > 1.15 * q[0]:
            return 270
        return guess

    # -- zoomed re-read of a small label (section 4d) -------------------
    def _zoom(self, ocr_img: np.ndarray, regions: list[dict], scale: float) -> int:
        H, W = ocr_img.shape[:2]
        pts = np.concatenate([np.asarray(r["poly_ocr"], np.float32).reshape(-1, 2) for r in regions])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        mx, my = 0.08 * (x1 - x0) + 20, 0.08 * (y1 - y0) + 20
        X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        X1, Y1 = int(min(W, x1 + mx)), int(min(H, y1 + my))
        if (X1 - X0) * (Y1 - Y0) >= self.config.zoom_below_area * W * H:
            return 0
        crop = np.ascontiguousarray(ocr_img[Y0:Y1, X0:X1])
        if min(crop.shape[:2]) < 32:
            return 0
        # About 2x: on the Kissan cup the red care lines appeared from 2x
        # (3x found more but cost 26 s). The detector input stays within
        # det_max_long_side like the main read: magnifying a WIDE crop to
        # Paddle's 4000 px cap took 3.5 GB (a real 4096x2304 photo ran the
        # machine out of memory). If that cap leaves little magnification
        # over the main read, the zoom is not worth a pass.
        cap = float(self.config.det_max_long_side or 2880)
        s0 = min(1.0, cap / float(max(H, W)))                 # main read's scale
        hs, ls = min(crop.shape[:2]), max(crop.shape[:2])
        # No more detector pixels than the main read used (memory), and at
        # least twice its magnification (else the pass buys little).
        main_px = (s0 * H) * (s0 * W)
        m = min(2.2, cap / float(ls), (1.25 * main_px / float(hs * ls)) ** 0.5)
        if m < 2.0 * s0:
            return 0
        try:
            res = self._pipeline().predict(crop, text_det_limit_type="min",
                                           text_det_limit_side_len=int(m * hs))
        except TypeError:
            res = self._pipeline().predict(crop)
        added = 0
        for r in self._regions_from(res, 1.0):
            q = np.asarray(r["poly_ocr"], np.float32) + [X0, Y0]
            r["poly_ocr"] = q.tolist()
            r["poly"] = (q / scale).tolist()
            r["engine"] = "paddleocr(zoom)"
            added += _merge_region(regions, r)
        return added

    # -- address blocks detected twice (section 4c) ----------------------
    def _block_redetect(self, ocr_img: np.ndarray, regions: list[dict], scale: float) -> int:
        anchors = [r for r in regions if _BLOCK_LABEL.search(r["text"] or "")
                   and not r.get("vertical")][:4]
        if not anchors:
            return 0
        try:
            det = self._poly_detector()
            from paddlex.inference.pipelines.components import CropByPolys
            unwarp = CropByPolys("poly")
        except Exception as exc:
            logger.warning("Block re-detection unavailable (%s)", str(exc).splitlines()[0][:120])
            return 0
        H, W = ocr_img.shape[:2]
        added = 0
        for a in anchors:
            q = np.asarray(a["poly_ocr"], np.float32).reshape(-1, 2)
            th = max(6.0, min(cv2.minAreaRect(q)[1]))
            x0, y0 = q.min(axis=0)
            x1, _ = q.max(axis=0)
            # The block: from the label down ~6 lines, as wide as the
            # lines under it usually are (they outrun a short label).
            X0 = int(max(0, x0 - 10 * th)); X1 = int(min(W, x1 + 10 * th))
            Y0 = int(max(0, y0 - 0.3 * th)); Y1 = int(min(H, y0 + 6.5 * th))
            if X1 - X0 < 32 or Y1 - Y0 < 16:
                continue
            crop = np.ascontiguousarray(ocr_img[Y0:Y1, X0:X1])
            try:
                polys = list(det.predict([crop]))[0]["dt_polys"]
                if polys is None or len(polys) == 0:
                    continue
                polys = [np.asarray(pl, np.float32).reshape(-1, 2) for pl in polys]
                strips = [x if isinstance(x, np.ndarray) else x["img"] for x in unwarp(crop, polys)]
                outs = _rec_batched(self._recogniser(self.config.lang), strips)
            except Exception:
                logger.warning("Block re-detection failed", exc_info=True)
                continue
            new = []
            for pl, o in zip(polys, outs):
                text, score = _rec_output(o)
                if score < 0.85 or len(_visible(text)) < 4:
                    continue
                box = _order_quad(cv2.boxPoints(cv2.minAreaRect((pl + [X0, Y0]).astype(np.float32))))
                new.append({"text": text, "score": float(score), "poly_ocr": box.tolist(),
                            "poly": (box / scale).tolist(), "engine": "paddleocr(block)",
                            "language": "en"})
            added += _absorb_block(regions, new)
        return added

    # -- Devanagari re-read -------------------------------------------
    def _secondary(self, ocr_img: np.ndarray, regions: list[dict]) -> int:
        lang = self.config.secondary_lang
        if lang in _SECONDARY_UNAVAILABLE:
            return 0
        crops, owners = [], []
        for i, r in enumerate(regions):
            # A line the English model read cleanly, in plain Latin text, is
            # not Devanagari. On 84 real photos, of the 110 lines the Hindi
            # read won, the English read scored at most 0.933 (cut-off 0.97) or was junk in
            # another script ("市", "さ"); 70% of all lines score >= 0.95.
            if (r["score"] >= self.config.secondary_below_score
                    and not _DEVANAGARI.search(r["text"])
                    and _LATIN_ONLY.fullmatch(r["text"] or "")):
                continue
            crop = crop_region(ocr_img, np.asarray(r["poly_ocr"]), pad_ratio=0.08)
            if crop is not None:
                crops.append(crop)
                owners.append(i)
        if not crops:
            return 0
        try:
            outs = _rec_batched(self._recogniser(lang), crops)
        except Exception as exc:
            # Model not downloaded and no internet, or not installed: carry
            # on in English only, and do not try again for every photo.
            _SECONDARY_UNAVAILABLE.add(lang)
            logger.warning("Hindi reader unavailable (%s); reading English only.",
                           str(exc).splitlines()[0][:120])
            self.last_stats["secondary_unavailable"] = True
            return 0

        replaced = 0
        for i, out in zip(owners, outs):
            text, score = _rec_output(out)
            if prefer_devanagari(regions[i]["text"], regions[i]["score"], text, score):
                regions[i]["alt_text"] = regions[i]["text"]
                regions[i]["alt_score"] = regions[i]["score"]
                regions[i]["text"], regions[i]["score"] = text, score
                regions[i]["engine"] = f"paddleocr({lang})"
                regions[i]["language"] = "hi"
                replaced += 1
        return replaced

    # -----------------------------------------------------------------
    @staticmethod
    def _to_span(r: dict) -> TextSpan:
        pts = np.asarray(r["poly"], dtype=np.float32)
        x, y = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        return TextSpan(
            text=_visible(r["text"]),
            bbox=BBox(x, y, x2 - x, y2 - y),
            confidence=float(r["score"]),
            source_engine=r.get("engine", "paddleocr"),
            language=r.get("language", "en"),
            vertical=is_vertical(pts, _visible(r["text"])),
            angle=_line_angle(pts),
        )

    # -- cache and dumps ----------------------------------------------
    def _cache_key(self, image: np.ndarray) -> str:
        try:
            import paddleocr
            ver = getattr(paddleocr, "__version__", "?")
        except Exception:
            ver = "?"
        h = hashlib.sha1()
        h.update(np.ascontiguousarray(image).tobytes())
        h.update(str(image.shape).encode())
        h.update(self.config.key().encode())
        h.update(ver.encode())
        # The read also depends on OUR code around Paddle (re-reads, merges);
        # bump this when that changes, so an old cached read is never reused.
        h.update(OCR_CODE_VERSION.encode())
        return h.hexdigest()

    def _cache_read(self, key: str) -> Optional[list[dict]]:
        if self.cache_dir is None or self._pipeline_override is not None:
            return None
        p = self.cache_dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))["regions"]
        except Exception:
            return None

    def _cache_write(self, key: str, regions: list[dict]) -> None:
        if self.cache_dir is None or self._pipeline_override is not None:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps({"config": asdict(self.config), "regions": regions}), encoding="utf-8")
        except Exception:
            logger.warning("Could not write OCR cache", exc_info=True)

    def _dump(self, source: str, image: np.ndarray, regions: list[dict], stats: dict) -> None:
        """
        A readable copy of exactly what Paddle read, named after the photo.

        This is what to send when a report looks wrong: it separates
        "Paddle misread the label" from "extraction or rules got it
        wrong", and `scripts/scan_photo.py --replay` re-runs everything
        downstream of OCR from it without loading a model.
        """
        try:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            out = self.dump_dir / f"{Path(source).stem}.paddle.json"
            out.write_text(json.dumps({
                "source": str(source),
                "ocr_input_shape": list(image.shape),
                "config": asdict(self.config),
                "stats": stats,
                "regions": regions,
            }, indent=1, ensure_ascii=False), encoding="utf-8")
            stats["dump"] = str(out)
        except Exception:
            logger.warning("Could not write OCR dump", exc_info=True)


# =====================================================================
# Replay: run everything downstream of OCR from a saved Paddle dump
# =====================================================================

class ReplayOCR(OCRBackend):
    """
    Returns the regions saved in a `.paddle.json` dump instead of running
    a model. Lets extraction and rules be debugged against real Paddle
    output on a machine with no Paddle models at all, in milliseconds.
    """

    name = "paddleocr(replay)"
    supports_languages = ("en", "hi")

    def __init__(self, dump_path: str | Path):
        self.dump_path = Path(dump_path)
        raw = self.dump_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Dumps written by builds before the UTF-8 fix used the Windows
            # default (cp1252) - e.g. a degree sign as byte 0xb0.
            text = raw.decode("cp1252", errors="replace")
        self.dump = json.loads(text)

    def upright_rotation(self, image: np.ndarray, source: Optional[str] = None) -> tuple[int, float]:
        u = (self.dump.get("stats") or {}).get("upright") or {}
        return int(u.get("turn", 0) or 0), float(u.get("tilt", 0.0) or 0.0)

    def recognise(self, image: np.ndarray, source: Optional[str] = None) -> list[TextSpan]:
        want = tuple(self.dump.get("ocr_input_shape", ()))[:2]
        if want and tuple(image.shape[:2]) != want:
            logger.warning(
                "Replay dump was taken on a %s image but this run fed OCR a %s "
                "image; boxes may not line up.", want, image.shape[:2],
            )
        spans = []
        for r in self.dump.get("regions", []):
            if _visible(r.get("text", "")):
                span = PaddleOCRBackend._to_span(r)
                span.source_engine = self.name
                spans.append(span)
        return spans


# =====================================================================
# Helpers (pure functions, unit-tested without any model)
# =====================================================================

def _rot_points(pts: np.ndarray, turn: int, shape: tuple) -> np.ndarray:
    """Points of an image of `shape` (h, w) after turning it `turn` degrees clockwise."""
    h, w = shape
    x, y = pts[:, 0], pts[:, 1]
    if turn == 90:
        return np.stack([(h - 1) - y, x], axis=1)
    if turn == 180:
        return np.stack([(w - 1) - x, (h - 1) - y], axis=1)
    if turn == 270:
        return np.stack([y, (w - 1) - x], axis=1)
    return pts


def decide_upright(regions: list[dict], shape: tuple, force_turn: Optional[int] = None) -> tuple[int, float]:
    """
    Which way up is the photo, from one read of it.

    * Most lines tall and narrow -> it lies on its side. Paddle turns every
      tall line crop 90 degrees anticlockwise before reading, and its
      textline classifier then flips the crops that came out upside down:
      unflipped means the photo was turned clockwise (turn it back 270),
      flipped means anticlockwise (turn it 90).
    * Most lines level but flipped -> upside down (180).
    Then the remaining tilt: the median slope of the (now level) lines,
    applied only when the lines agree on it (a curved label does not).
    """
    rs = [r for r in regions if len(np.asarray(r["poly_ocr"]).reshape(-1, 2)) >= 4]
    if len(rs) < 3:
        return 0, 0.0
    def wh(r):
        p = np.asarray(r["poly_ocr"], np.float32).reshape(-1, 2)
        rect = cv2.minAreaRect(p)
        (bw, bh) = rect[1]
        x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
        return (x1 - x0), (y1 - y0)
    weights = [len(_visible(r["text"])) for r in rs]
    tall = [(r, wt) for r, wt in zip(rs, weights) if wh(r)[1] >= 1.5 * max(1.0, wh(r)[0])]
    wide = [(r, wt) for r, wt in zip(rs, weights) if wh(r)[0] >= 1.5 * max(1.0, wh(r)[1])]
    tot = float(sum(weights)) or 1.0
    turn = 0 if force_turn is None else force_turn
    if force_turn is not None:
        pass
    elif sum(wt for _, wt in tall) >= 0.6 * tot:
        flipped = sum(wt for r, wt in tall if r.get("tl_angle") == 180)
        turn = 90 if flipped >= 0.5 * sum(wt for _, wt in tall) else 270
    elif sum(wt for _, wt in wide) >= 0.6 * tot:
        flipped = sum(wt for r, wt in wide if r.get("tl_angle") == 180)
        if flipped >= 0.6 * sum(wt for _, wt in wide):
            turn = 180
    # tilt, measured on the lines as they will stand after the turn
    angs, ws = [], []
    for r, wt in zip(rs, weights):
        p = _rot_points(np.asarray(r["poly_ocr"], np.float32).reshape(-1, 2), turn, shape)
        x0, y0 = p.min(axis=0); x1, y1 = p.max(axis=0)
        if (x1 - x0) < 2.0 * max(1.0, (y1 - y0)) and wt < 8:
            continue                                  # short words say little about slope
        angs.append(_line_angle(_order_quad(cv2.boxPoints(cv2.minAreaRect(p)))))
        ws.append(wt)
    tilt = 0.0
    if len(angs) >= 4:
        a = np.asarray(angs); med = float(np.median(a))
        spread = float(np.median(np.abs(a - med)))
        # 10-30 degrees only. Below 10 the reading already copes (rotated
        # boxes, deskewed measurement) and straightening only resamples fine
        # print - it cost a real Vim pouch (8.9 deg) its edge-printed price.
        # Above 30 it is not told apart from print laid on the diagonal
        # (can-base inkjet at 34 deg lost its MRP when "straightened").
        if 10.0 <= abs(med) <= 30.0 and spread <= 3.0:
            tilt = round(med, 1)
    return turn, tilt


def turn_upright(image: np.ndarray, turn: int, tilt: float) -> np.ndarray:
    """Apply decide_upright's answer: quarter turns exactly, then the tilt
    on an enlarged white canvas (nothing is cut off)."""
    if turn == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif turn == 180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif turn == 270:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if tilt:
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), tilt, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2.0 - w / 2.0
        M[1, 2] += nh / 2.0 - h / 2.0
        border = (255, 255, 255) if image.ndim == 3 else 255
        image = cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)
    return image


def _line_angle(pts: np.ndarray) -> float:
    """Slope of the box's long side in degrees, folded into (-45, 45]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return 0.0
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    v = e1 if float(np.hypot(*e1)) >= float(np.hypot(*e2)) else e2
    a = float(np.degrees(np.arctan2(v[1], v[0])))
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    if a > 45:
        a -= 90
    elif a <= -45:
        a += 90
    return round(a, 2)


def is_vertical(pts: np.ndarray, text: str) -> bool:
    """A region whose long side runs top-to-bottom, holding real text."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    return len(text.strip()) >= 3 and h >= 1.5 * max(w, 1.0)


def crop_region(img: np.ndarray, poly: np.ndarray, pad_ratio: float = 0.15) -> Optional[np.ndarray]:
    """
    Crop a text region the way Paddle does - minimum-area rectangle,
    perspective-corrected, rotated upright when tall - but with margin.

    The margin is the point: Paddle's own crop is tight to the detector
    box, and on small print a tight box shaves descenders and the gap
    between a number and its unit.
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 4:
        return None
    rect = cv2.minAreaRect(poly)
    (cx, cy), (rw, rh), angle = rect
    if rw < 2 or rh < 2:
        return None
    short = min(rw, rh)
    rect = ((cx, cy), (rw + 2 * pad_ratio * short, rh + 2 * pad_ratio * short), angle)
    box = cv2.boxPoints(rect)
    # Order: top-left, top-right, bottom-right, bottom-left.
    s, d = box.sum(axis=1), np.diff(box, axis=1).ravel()
    tl, br = box[np.argmin(s)], box[np.argmax(s)]
    tr, bl = box[np.argmin(d)], box[np.argmax(d)]
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 2 or height < 2:
        return None
    M = cv2.getPerspectiveTransform(
        np.float32([tl, tr, br, bl]),
        np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
    )
    crop = cv2.warpPerspective(img, M, (width, height),
                               borderMode=cv2.BORDER_REPLICATE,
                               flags=cv2.INTER_CUBIC)
    if height >= 1.5 * width:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


def _clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if img.ndim == 3 else img
    if img.ndim == 3:
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(img)



def _rec_batched(rec, crops: list, batch: int = 6) -> list:
    """Recognise crops the way Paddle's own OCR pipeline does: sorted by
    aspect ratio, `batch` at a time (a bare TextRecognition runs one crop
    at a time). 1.3-2x faster on CPU; outputs come back in input order."""
    if not crops:
        return []
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / float(max(1, crops[i].shape[0])))
    sampler = getattr(getattr(rec, "paddlex_predictor", None), "batch_sampler", None)
    old = getattr(sampler, "batch_size", None)
    try:
        if sampler is not None:
            sampler.batch_size = batch
        outs = list(rec.predict([crops[i] for i in order]))
    finally:
        if sampler is not None and old is not None:
            sampler.batch_size = old
    res = [None] * len(crops)
    for k, i in enumerate(order):
        res[i] = outs[k]
    return res

def _rec_output(out) -> tuple[str, float]:
    """TextRecognition.predict yields dict-likes with rec_text/rec_score."""
    try:
        return str(out["rec_text"]).strip(), float(out["rec_score"])
    except (KeyError, TypeError, ValueError):
        return "", 0.0


def prefer_devanagari(en_text: str, en_score: float, hi_text: str, hi_score: float) -> bool:
    """
    Take the Devanagari read only where the region really is Devanagari.

    The Devanagari model also emits Latin letters, so a region reading
    "MRP" in both models must stay with the stronger English read. The
    swap needs the Devanagari output to be mostly Devanagari script and
    to be at least roughly as confident.
    """
    if not hi_text:
        return False
    letters = [ch for ch in hi_text if not ch.isspace() and not ch.isdigit()]
    if not letters:
        return False
    dev_ratio = sum(1 for ch in letters if _DEVANAGARI.match(ch)) / len(letters)
    if dev_ratio < 0.5 or sum(1 for ch in letters if _DEVANAGARI.match(ch)) < 2:
        return False
    return hi_score >= en_score - 0.10 or hi_score >= 0.80
