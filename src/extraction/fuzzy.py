"""
Fuzzy matching for OCR-damaged labels, driven by rules/lexicon.yaml.

WHY
---
Real PaddleOCR output mangles the very labels extraction keys on:
"TINET QUNIIITY:" and "GTUET CUANTTY:" for NET QUANTITY (a curved
bottle), "Tol Free" for Toll Free, "(nd of af taues)" for "(incl. of all
taxes)". An exact regex never sees those. The lexicon (anchors,
thresholds, negative context) lets a near-miss label still be found - and
every match found this way is marked on the declaration, so the report
says the label was matched fuzzily and an inspector can check it.

rapidfuzz is used when installed (it is fast and gives the match
position); otherwise a difflib fallback does the same job more slowly.
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

LEXICON_PATH = Path(__file__).resolve().parents[2] / "rules" / "lexicon.yaml"

try:  # pragma: no cover - depends on the environment
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:  # pragma: no cover
    _rf_fuzz = None


@lru_cache(maxsize=1)
def lexicon() -> dict:
    import yaml

    try:
        return yaml.safe_load(LEXICON_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()


def partial_match(needle: str, hay: str) -> tuple[float, int, int]:
    """
    Best match of `needle` anywhere inside `hay`.
    Returns (score 0-100, start, end) with start/end indexing `hay`.
    """
    if not needle or not hay:
        return 0.0, 0, 0
    if len(hay) < 0.85 * len(needle):
        return 0.0, 0, 0        # a fragment ("QUANTITY") is not the label ("net quantity")
    if len(hay) < len(needle):
        # partial matching would search the SHORTER string inside the
        # longer one, so "QUANTITY" alone scored 100 against "net
        # quantity". A fragment shorter than the anchor is compared whole.
        if _rf_fuzz is not None:
            return float(_rf_fuzz.ratio(needle, hay)), 0, len(hay)
        return difflib.SequenceMatcher(None, needle, hay).ratio() * 100, 0, len(hay)
    if _rf_fuzz is not None:
        al = _rf_fuzz.partial_ratio_alignment(needle, hay)
        if al is None:
            return 0.0, 0, 0
        return float(al.score), al.dest_start, al.dest_end
    n = len(needle)
    best = (0.0, 0, 0)
    for width in {n - 1, n, n + 1}:
        if width <= 0:
            continue
        for i in range(0, max(1, len(hay) - width + 1)):
            r = difflib.SequenceMatcher(None, needle, hay[i:i + width]).ratio() * 100
            if r > best[0]:
                best = (r, i, i + width)
    return best


def threshold_for(anchor: str) -> Optional[float]:
    """
    Minimum score for an anchor of this length, from the lexicon.
    Short anchors ("mrp", "mfg") are too easy to hit by accident and are
    never matched fuzzily: None means exact only.
    """
    t = lexicon().get("thresholds") or {}
    short = int(t.get("short_anchor_len", 4))
    mid = int(t.get("mid_anchor_len", 8))
    if len(anchor) <= short:
        return None
    if len(anchor) <= mid:
        return float(t.get("mid_score", 85))
    return float(t.get("long_score", 78))


def in_negative_context(text: str) -> bool:
    low = _norm(text)
    for words in (lexicon().get("negative_context") or {}).values():
        for w in words or []:
            if re.search(rf"\b{re.escape(w.lower())}\b", low):
                return True
    return False


def find_anchor(text: str, group: str) -> Optional[tuple[str, float, int, int]]:
    """
    Fuzzy-find any anchor of `group` (e.g. "net_qty") in `text`.

    Returns (canonical_anchor, score, start, end), start/end indexing
    `text` itself, or None. Exact containment is handled by the regexes
    already; this is only for OCR-damaged labels.
    """
    anchors = (lexicon().get("anchors") or {}).get(group) or []
    hay = (text or "").lower()
    best = None
    for a in anchors:
        a_n = a.lower()
        thr = threshold_for(_norm(a_n))
        if thr is None:
            continue
        score, s, e = partial_match(a_n, hay)
        if score < thr or not _words_present(a_n, hay[s:e]):
            continue
        if best is None or score > best[1]:
            best = (a, score, s, e)
    return best


def _words_present(anchor: str, span: str, min_ratio: float = 0.6) -> bool:
    """
    Every word of the anchor must be recognisably in the matched span.

    Character similarity alone let "Imported" (a product title) pass for
    "imported by": the score was 90 with no "by" anywhere. The short
    words are what make a label a label. In anchors of three or more
    words, one- and two-letter words may be missing (OCR drops "of").
    """
    words = [w for w in re.findall(r"[a-z]+", anchor.lower())]
    got = re.findall(r"[a-z]+", span.lower())
    if not got:
        return False
    for w in words:
        if len(words) >= 3 and len(w) <= 2:
            continue
        if not any(difflib.SequenceMatcher(None, w, g).ratio() >= min_ratio
                   or (len(w) >= 4 and w in g) for g in got):
            return False
    return True


def resembles(text: str, target: str, min_score: float) -> tuple[bool, float]:
    score, _, _ = partial_match(_norm(target), _norm(text))
    return score >= min_score, score


def commodity_noun(text: str) -> Optional[str]:
    """A generic commodity noun from the lexicon, as a whole word."""
    low = _norm(text)
    for noun in lexicon().get("common_name_nouns") or []:
        if re.search(rf"\b{re.escape(noun.lower())}\b", low):
            return noun
    return None
