"""
Field extraction: OCR text -> bound legal declarations.

THE POINT EVERY SHALLOW SUBMISSION MISSES
-----------------------------------------
Finding the string "Rs 120" on a package proves nothing. The legal
question is whether the package bears a RETAIL SALE PRICE DECLARATION,
which means a price, correctly worded as inclusive of all taxes, on the
principal display panel. Those are different questions and only the
second one is what Rule 6(1)(e) asks.

So extraction is a key-value binding problem, not a keyword search. Each
extractor here answers "which declaration is this text, and what is its
parsed value", and hands the rules engine a typed Declaration object.

TWO IMPLEMENTATIONS
-------------------
`RegexExtractor`   - deterministic, fast, no dependencies, fully
                     debuggable. Handles the majority of real labels
                     because Indian packaging is highly formulaic.
`SchemaVLMExtractor` - constrained JSON extraction for the messy tail.

Start with regex. Add the VLM for what regex misses, and MEASURE the
difference so you can say what it bought you. A hybrid that you can
explain beats a black box that you cannot.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from ..core.schema import BBox, Declaration, TextSpan

# =====================================================================
# Text normalisation
# =====================================================================

# OCR routinely confuses these on packaging print. Applied only inside
# numeric contexts - never to whole strings, or "Sol" becomes "S01".
_DIGIT_CONFUSIONS = {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"}


def normalise_unit(raw: str) -> str:
    """
    Map an OCR-confused unit token back to its intended unit.

    Applied only in unit position, never to free text - "1" means one
    everywhere else, and only here does it stand a real chance of being
    a mangled litre.
    """
    u = (raw or "").strip()
    hu = re.sub(r"\s+", "", u).rstrip(".")
    if hu in HINDI_UNITS:
        return HINDI_UNITS[hu]
    if u in OCR_UNIT_CONFUSIONS:
        return OCR_UNIT_CONFUSIONS[u]
    low = u.lower()
    if low in OCR_UNIT_CONFUSIONS:
        return OCR_UNIT_CONFUSIONS[low]
    return u


def normalise_numeric(text: str) -> str:
    return "".join(_DIGIT_CONFUSIONS.get(c, c) for c in text)


# Bilingual packs print Devanagari numerals (५०० ग्राम). Normalising them
# here means every numeric pattern below works on either script.
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


# CJK glyphs never occur on Indian packaging, but PP-OCRv6's recogniser
# is bilingual Chinese/English and reads smudges, bar-code fragments and
# rotated inkjet dots as 印, 心, 出 ... (every one of these came back on
# real photos). They are noise, never text.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]")
# "(Rs.0.4 0/ml)": OCR splits a decimal with a space. Only re-joined
# directly before a unit-price marker, where the reading is unambiguous.
_SPLIT_DECIMAL = re.compile(r"(\d\.\d) (\d)(?=\s*(?:/|per\b|\)))", re.I)
# A zero read as the letter O in front of a decimal: "USP:O.52/g" (real
# Rajam jar) matched as a stray currency glyph "O." and a price of 52.
# Not inside a word, and only before ".<digit>".
_O_ZERO = re.compile(r"(?<![A-Za-z])[OoDQ](?=[.,]\d)")


# "06/09/2610:USP" - a d/m/yy date with the next number glued on ("₹10" with
# the rupee sign and the comma lost, real Paper Boat inkjet). A four-digit
# "year" outside 1990-2099 right after a day and month is yy + something.
# "70 g + 9 g EXTRA = 79 g" (real Britannia): the pack holds the TOTAL.
_EXTRA_TOTAL = re.compile(
    r"\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l)\s*\+\s*\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l)\s*"
    r"(?:extra|free)\W{0,3}\s*=\s*(\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l))(?![a-z])", re.I)

_GLUED_YEAR = re.compile(r"(?<![\d/])(\d{1,2}/\d{1,2}/)(\d{2})(\d{2})(?!\d)")


def _unglue_year(m: re.Match) -> str:
    if 1990 <= int(m.group(2) + m.group(3)) <= 2099:
        return m.group(0)
    return f"{m.group(1)}{m.group(2)} {m.group(3)}"


# The rupee sign read as a CJK glyph: "29/08/26,天10:USP车 0.071/ml" (real
# Paper Boat inkjet). Only in a line that is about prices, and only right
# before a number; any other CJK glyph is still dropped as noise.
_PRICE_CTX = re.compile(r"\b(?:[uo0]sp|m\.?\s*r\.?\s*p|rs|mrp)\b|/\s*(?:ml|g|kg|l)\b|\bper\s+\d*\s*(?:ml|g|kg)\b", re.I)
_CJK_RUPEE = re.compile(r"(?:%s)(?=\s?\d)" % _CJK.pattern)
# "260.C" / "0.4O": C or O read for a zero AFTER a decimal point (real Rajam
# inkjet MRP). Not followed by a letter, so "12.Oct" stays a date.
_DEC_ZERO = re.compile(r"(?<=\d\.)[CcOo](?![A-Za-z])|(?<=\d\.\d)[CcOo](?![A-Za-z])")


_COUNTRIES = [
    "India", "China", "P\\.?\\s*R\\.?\\s*C", "Thailand", "Vietnam", "Viet Nam", "Indonesia", "Malaysia",
    "Singapore", "Philippines", "Japan", "Korea", "South Korea", "Republic of Korea", "Taiwan",
    "Hong Kong", "Sri Lanka", "Bangladesh", "Nepal", "Bhutan", "Pakistan", "Myanmar", "UAE",
    "United Arab Emirates", "Saudi Arabia", "Oman", "Qatar", "Kuwait", "Bahrain", "Turkey",
    "Turkiye", "Iran", "Israel", "Egypt", "South Africa", "Kenya", "Nigeria", "Morocco",
    "Italy", "France", "Germany", "Spain", "Portugal", "Belgium", "Netherlands", "Holland",
    "Switzerland", "Austria", "Poland", "Czech Republic", "Czechia", "Hungary", "Romania",
    "Greece", "Ireland", "United Kingdom", "UK", "U\\.K", "England", "Scotland", "Denmark",
    "Sweden", "Norway", "Finland", "Russia", "Ukraine", "USA", "U\\.S\\.A", "United States",
    "United States of America", "America", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
    "Peru", "Colombia", "Australia", "New Zealand",
]

# "2026-01-15" / "2026/01": ISO order, rewritten day-first like the rest.
_ISO_DATE = re.compile(r"(?<![\d/.-])((?:19|20)\d{2})([-/])(0[1-9]|1[0-2])(?:\2(0[1-9]|[12]\d|3[01]))?(?![\d/.-])")
# "2 x 100 g" / "6 x 20 g" (multipacks): the pack holds the product; "4 x 25 g
# = 100 g" states it. Read as the total.
_MULTIPACK = re.compile(
    r"(?<![\d.])(\d{1,3})\s*[x×*]\s*(\d+(?:\.\d+)?)\s*(g|gm|kg|ml|l)(?![a-z])"
    r"(\s*=\s*\d+(?:\.\d+)?\s*(?:g|gm|kg|ml|l)(?![a-z]))?", re.I)


def _iso(m: re.Match) -> str:
    y, mo, d = m.group(1), int(m.group(3)), m.group(4)
    return f"{int(d):02d}/{mo:02d}/{y}" if d else f"{mo:02d}/{y}"


def _multipack(m: re.Match) -> str:
    if m.group(4):
        return m.group(4).split("=", 1)[1].strip()
    tot = int(m.group(1)) * float(m.group(2))
    return f"{tot:g} {m.group(3)}"


def clean(text: str) -> str:
    t = (text or "").translate(_DEV_DIGITS)
    if _PRICE_CTX.search(_CJK.sub(" ", t)):
        t = _CJK_RUPEE.sub("₹", t)
    t = _CJK.sub("", t)
    t = _DEC_ZERO.sub("0", t)
    t = re.sub(r"\s+", " ", t.strip())
    t = _O_ZERO.sub("0", t)
    t = _GLUED_YEAR.sub(_unglue_year, t)
    t = _EXTRA_TOTAL.sub(r"\1", t)
    t = _ISO_DATE.sub(_iso, t)
    t = _MULTIPACK.sub(_multipack, t)
    return _SPLIT_DECIMAL.sub(r"\1\2", t)


# =====================================================================
# Patterns
# =====================================================================

CURRENCY = r"(?:₹|Rs\.?|INR|R s\.?|रु\.?|रू\.?)"
# The comma-grouped branch uses + not *, and plain digits come second.
# With * the first branch matched "149" out of "1495.00" - three digits,
# zero comma groups, and the optional decimal could not attach because
# the next character was "5". Regex alternation then returned that
# partial match rather than trying the long-number branch. Every MRP
# above Rs.999 written without a comma was silently read as its first
# three digits: Rs.1495.00 became Rs.149.00. On a real Casio watch box
# that is a tenfold error in the single most important declaration.
NUMBER = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
# Unit sale prices are sometimes printed to three places ("USP ₹0.071/ml",
# real Paper Boat); with NUMBER that read as 0.07 plus a stray "1".
NUMBER3 = r"\d+(?:\.\d{1,3})?"

# OCR confuses a handful of glyphs that matter enormously here, because
# the unit carries the legal meaning: "1 l" of cooking oil read as "1 |"
# extracts no quantity at all, and the package is then reported as
# missing its net quantity declaration - a false violation from a single
# misread character.
#
# The pipe/one/ell family is the worst: |, I, 1 and l are near-identical
# in many packaging faces, and litre is a single lowercase l.
OCR_UNIT_CONFUSIONS = {
    "|": "l", "I": "l", "1": "l", "!": "l",      # litre
    "9": "g", "q": "g", "6": "g", "&": "g",       # gram - "500 g" is
                                                  # very often read "500 6"
    "mI": "ml", "m|": "ml", "rnl": "ml",          # millilitre
    "kq": "kg", "k9": "kg",                       # kilogram
    "0": "g",                                     # rare, but seen on foil
}

# Hindi unit spellings as printed on bilingual packs, trailing dot removed.
HINDI_UNITS = {
    "ग्राम": "g", "ग्रा": "g", "कि.ग्रा": "kg", "किग्रा": "kg", "किलोग्राम": "kg",
    "कि.ग्रा.": "kg", "मि.ली": "ml", "मिली": "ml", "मिलीलीटर": "ml",
    "ली": "l", "लीटर": "l", "नग": "N",
}
_HINDI_UNIT_RE = (
    r"किलोग्राम|कि\.?\s?ग्रा\.?|ग्राम|ग्रा\.?|मिलीलीटर|मि\.?\s?ली\.?|मिली|"
    r"लीटर|ली\.?|नग"
)

QTY_UNITS = (
    r"(?:" + _HINDI_UNIT_RE + r"|"
    r"kg|kgs|k[q96]|g|gm|gms|gram|grams|ml|mls|m[I|]|rnl|"
    r"l|ltr|litre|liter|[|I!69&]|N|U|mm|cm|m|"
    # Rule 13(5)(ii) prescribes the symbol N or U for goods sold by
    # number, but almost nothing on a real shelf obeys that: a Casio
    # watch box reads "NET QUANTITY 1 PIECE". Without these the
    # quantity line fails to match at all and a greedy fallback grabs
    # whatever number it finds first - on the Casio box it captured
    # "MANUFACTURED ON : 03/2026" as the net quantity.
    r"piece|pieces|pc|pcs|no|nos|number|unit|units|"
    r"pkt|pkts|packet|packets|tablet|tablets|cap|caps|capsule|capsules)"
)

# "1 l" frequently comes back as a single token "11" - the space is lost
# and the litre glyph reads as a one. Only applied behind an explicit
# "Net Qty" prefix, where a bare trailing digit cannot plausibly be part
# of the quantity itself.
MERGED_LITRE = re.compile(
    r"(?:net\s*(?:qty|quantity|vol|volume))\s*:?\s*(\d+)\s*[1lI|]\b", re.I
)

# "M.R.P." with its first letter damaged: "A.R.P." (real Drolia Gulal).
# Only the fully dotted form - three dotted capitals are not a word.
_DOTTED_MRP = r"(?<![A-Za-z])[A-Z]\.\s*R\.\s*P\.?"

PATTERNS: dict[str, list[re.Pattern]] = {
    "retail_sale_price": [
        re.compile(
            rf"(?:M\.?R\.?P\.?|{_DOTTED_MRP}|max(?:imum|\.)?\s*retail\s+price|"
            rf"अधिकतम\s*खुदरा\s*मूल्य|अधि\.?\s*खु\.?\s*मू\.?|अ\.?\s*खु\.?\s*मू\.?|"
            rf"एम\.?\s*आर\.?\s*पी\.?)"
            # After the label, the rupee sign is often misread as a stray
            # letter: "Max Retail Price R: 5.00", "A.R.P.R 40.00" (real
            # Drolia packs). Behind a label, one such glyph is skipped.
            # And "₹ 10/-" read as "7 10/-" (real Haldiram, "MRP.7 10/-"):
            # a lone 7 or 2 directly followed by another amount is the ₹.
            rf"\s*:?\s*(?:{CURRENCY}|[^\d\s]{{1,2}}(?=\s*:?\s*\d)|[72](?=\s+\d))?\s*:?\s*({NUMBER})"
            # ...and not a number glued into a word: "MRP08P" is "MRP ₹/USP"
            # with the ₹ read as 0 (real Troovy), not a price of 8.
            # A full stop after a whole-rupee amount is punctuation: "MRP
            # Rs.125." (real Monster can base) - only ".<digit>" continues it.
            rf"(?![\d,]|\.\d)(?![A-Za-z]{{1,2}}(?![A-Za-z]))",
            re.I,
        ),
        # Unlabelled "<currency> <number>" - but never a per-unit price:
        # "Rs.0.29/g" is the unit sale price, and taking it as the MRP
        # raised two false violations on a real Haldiram pack.
        re.compile(rf"{CURRENCY}\s*({NUMBER})(?![\d,]|\.\d)"
                   rf"(?!\s*(?:/|per\b|प्रति))\s*(?:only)?", re.I),
        # Label and ₹ both lost, qualifier kept: "RP 20.00 tad. of all
        # taxes" (real Bingo pack, "M" cut). A rupees-and-paise amount
        # directly followed by "... of all taxes" is the MRP.
        # Comma groups allowed and never started mid-number: "(1,699.00 (incl
        # of all taxes)" (real Zebronics, ₹ read as "(") was taken as 699.00.
        re.compile(r"(?<![\d.,])((?:\d{1,3}(?:,\d{2,3})+|\d{1,6})\.\d{2})\s*\S{0,6}\s*"
                   r"of\s+a[l1iI|]{1,2}\s+tax", re.I),
        # "/-" after an amount is the Indian mark of a rupee price ("₹25/-"):
        # it identifies the price when the ₹ itself is lost ("HB*25/-", an
        # inkjet edge code on a real Vim pouch) - but not a rate: "buy back
        # price ₹10/- per kg" (real Troovy) is not the MRP.
        re.compile(r"(?<![\d.,/])(\d{1,6}(?:\.\d{2})?)\s*/-(?!\s*(?:per\b|/|प्रति))", re.I),
    ],
    "net_quantity": [
        re.compile(
            rf"(?:net\.?\s*(?:qty|quantity|wt|weight|vol|volume)|contents?|"
            rf"शुद्ध\s*(?:मात्रा|भार|वजन)|कुल\s*मात्रा|नेट\s*(?:मात्रा|वजन))[\s:.\-]*"
            # "Net Qty: Pack of 6 (120 g)" - the count, then the quantity
            rf"(?:pack\s+of\s+\d+\s*\(?\s*)?"
            # A qualifier BEFORE the number ("Net Qty: approx 200 g", "Min.
            # 200 g") must not stop the match: the quantity was then "not
            # read" and the banned qualifier never reached the check.
            rf"(?:(?:approx(?:imately|imate)?|apprx|about|min(?:imum)?|max(?:imum)?|avg|average|"
            rf"nearly|around|upto|up\s*to|not\s+less\s+than|~)\.?\s*)?"
            # NOT \b here: a word boundary after a non-word unit glyph
            # such as "|" can never match, so every OCR-mangled litre
            # silently failed to extract.
            rf"({NUMBER}|\d+\.\d{{3}})\s*({QTY_UNITS})(?![A-Za-z0-9])",
            re.I,
        ),
        
        MERGED_LITRE,
    ],
    "unit_sale_price": [
        re.compile(
            # "0SP" / "OSP": USP in dot-matrix with the U read as 0 or O.
            rf"(?:unit\s+sale\s+price|price\s+per\s+unit|per\s+{QTY_UNITS}|\b[uo0]\.?s\.?p(?![a-z])\.?|"
            rf"इकाई\s*(?:बिक्री\s*)?मूल्य)\s*:?\s*"
            rf"(?:{CURRENCY}|[^\d\s]{{1,2}}(?=\s*\d))?\s*({NUMBER3})(?![\d])"
            # ...but not a number split off another that carries the "/unit":
            # "USP10 52/9" is "USP Rs.0.52/g" with the Rs read as 1 and the
            # point lost (real Rajam jar) - unreadable, not "10".
            rf"(?!\s+\d+(?:[.,]\d+)?\s*(?:/|per\b))",
            re.I,
        ),
        re.compile(rf"{CURRENCY}\s*({NUMBER3})\s*(?:/|per|प्रति)\s*({QTY_UNITS})", re.I),
        # The rupee sign is the glyph OCR drops most often: "(₹0.83 per g)"
        # comes back as "(0.83 per g)" and the currency-anchored pattern
        # above then misses a unit price that is plainly printed. Without
        # a currency, require the shape that makes it unambiguous: a
        # two-decimal amount (Rule 6(11) prints them to two places)
        # followed by "per <unit>", optionally with a basis quantity.
        re.compile(
            rf"(?<![\d.])(\d+\.\d{{2}})\s*(?:/|per|प्रति)\s*(?:\d+\s*)?({QTY_UNITS})(?![A-Za-z0-9])",
            re.I,
        ),
    ],
    "manufacture_date": [
        re.compile(
            # "HFD" / "HF0" is MFD in dot-matrix print read with M as H and
            # D as 0 (real Monster can base, sharp and WhatsApp copies).
            r"(?:mfg(?!\.?\s*lic)|(?<![A-Za-z])[mh]f[d0o](?!\.?\s*lic)|pkd|manufactured|packed|packing|"
            r"date\s+of\s+(?:mfg|manufacture|packing|import)|(?:month\s*(?:&|and)\s*)?year\s+of\s+import|"
            r"import(?:ed)?\s*(?:date|on)|"
            r"निर्माण\s*(?:की\s*)?(?:तिथि|तारीख)|उत्पादन\s*तिथि|पैकिंग\s*तिथि|"
            r"पैक\s*करने\s*की\s*तिथि)"
            # month/year, or day/month/year. The 4-digit year is tried
            # first and must not be followed by a digit, or "03/2026"
            # would split as month 03, "year" 20.
            r"[^\d]{0,12}(\d{1,2})[/\-.\s](\d{4}|\d{1,2})(?!\d)(?:[/\-.](\d{4}|\d{2})(?!\d))?",
            re.I,
        ),
        re.compile(
            r"(?:mfg|(?<![A-Za-z])[mh]f[d0o]|pkd|packed|manufactured)\.?(?:\s*(?:date|dt|on))?[^\w]{0,6}"
            # the day may come first, glued on: "Mfg:10MAY26" (real Nat Habit)
            r"(?:\d{1,2}\s*[/\-.]?\s*)?"
            r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)\.?\s*"
            r"[/\-.'\s]?\s*(\d{2,4})",
            re.I,
        ),
    ],
    "manufacturer_details": [
        # Indian packs abbreviate this relentlessly. A Nestle sachet
        # prints "Mkt by:" and "Mfg. by:"; only the fully spelled
        # "Marketed by" matched before, so the manufacturer declaration
        # was reported MISSING on a pack that plainly carries it - the
        # worst kind of false violation, because the label is compliant.
        re.compile(
            r"(?:(?:manufactured|manufacturer|mfg|mfd|mfrd|mftd|"
            r"packed|pkd|marketed|mktd|mkt|imported|imp)\.?\s*"
            r"(?:by|for)\.?|"
            # "Packed & Marketed" with "By" cut off by the fold (real MHM)
            r"(?:packed|pkd|manufactured|mfd)\.?\s*(?:&|and)\s*(?:marketed|mktd|packed|pkd)\.?(?:\s*by\.?)?|"
            # "Imported and Distributed by", "Marketed & Distributed by",
            # "Imported, Marketed & Distributed by", "Mktd. & Distd. by"
            r"(?:imported|marketed|mktd|packed|pkd|manufactured|mfd|mfg)\.?\s*(?:,|&|and)\s*"
            r"(?:(?:marketed|mktd|packed|pkd|imported)\.?\s*(?:,|&|and)\s*)?"
            r"(?:marketed|mktd|distributed|distd|packed|pkd|imported)\.?\s*by\.?|"
            # plain labels: "Manufacturer:", "Packer:", "Importer:", "Mfr:"
            r"(?<![A-Za-z])(?:manufacturer|packer|importer|marketer|mfr)\s*[:\-]|"
            # "MANUFACTURED IN INDIA BY:" (real Indoco toothpaste)
            r"(?:manufactured|mfd|made)\s+in\s+india\s+by\.?|"
            # Multi-unit brands print the unit list by batch code: "For
            # name and address of mfg. unit read the first two characters
            # of batch code & see: (BM) Dabur India Limited, ..." (real).
            r"name\s*(?:and|&)\s*add?r?e?s{1,2}|"
            r"निर्माता|विपणनकर्ता|आयातक|पैककर्ता|द्वारा\s*(?:निर्मित|विपणन|पैक(?:\s*किया)?))"
            r"\s*:?\s*(.{6,160})",
            re.I | re.S,
        ),
    ],
    "consumer_care": [
        re.compile(
            # Real packs word this many ways: "Consumer Care", "Consumer
            # Services", "Customer Support", "For queries/feedback/
            # complaints contact...", "Toll free:". The window is wide
            # (260 chars over up to 4 lines) because the phone / e-mail
            # is usually on the LAST line of the block, and a 160-char
            # cap cut it off on a real Nestle back panel.
            r"(?:consumer\s*(?:care|services?|affairs|relations?|helpline|cel+)|"
            r"customer\s*(?:care|services?|support|helpline|cel+)|complaints?|"
            r"queries|grievance|tol+[\s-]*free|helpline|let'?s\s+talk|"
            r"उपभोक्ता\s*(?:सेवा|शिकायत)|ग्राहक\s*सेवा)"
            r"[ \t]*[:\-]?\s*(.{6,260})",
            re.I | re.S,
        ),
    ],
    "common_name": [
        # Imported packs routinely head this "COMMODITY:" rather than
        # printing a bare descriptive name.
        re.compile(
            r"(?:commodity|generic\s+name|common\s+name|product\s+type|"
            # FSSAI category line, "PROPRIETARY FOOD - NAMKEEN" (real Bingo)
            r"\w{0,2}prietary\s*foo?d?\s*[-–:]|"
            # "Contents: Roli (Kumkum)" - a word, not a quantity (real Drolia)
            r"contents?\s*:(?=\s*[A-Za-z]))"
            r"\s*:?\s*([A-Za-z][A-Za-z \-/&]{2,40}(?:\([A-Za-z \-/&]{2,30}\))?)",
            re.I,
        ),
        # NO unlabelled fallback, PROVEN unsafe by a stability test on a
        # real photo (scripts/diagnose.py --mode stability), not just
        # argued in the abstract. A negative-context gate rejects known
        # disqualifying words - "consumer", "care", "nutrition" - but on
        # a real image OCR noise destroys those exact words before the
        # regex ever sees them: "NESTLE CONSUMER CARE" was read as
        # "HESTECOIINER ARE", which contains no recognisable "consumer"
        # or "care" for the gate to catch. Perturbing the same photo by a
        # few degrees of rotation produced FIFTEEN different values for
        # this field, all garbage: "HESTECOIINER ARE", "NESTLE INDIA",
        # "i My", "te ie". A gate that a small rotation defeats is not a
        # safety net. Same failure class as the unlabelled net_quantity
        # fallback that promoted a nutrition figure to the pack quantity -
        # a wrong fact is worse than a missing one, because nobody
        # rechecks a field that looks answered.
    ],
    "country_of_origin": [
        re.compile(
            r"(?:country\s+of\s+origin|मूल\s*देश|उत्पत्ति\s*(?:का\s*)?देश)"
            r"\s*:?\s*([A-Za-z\u0900-\u097F\s'\u2019.]{2,40})",
            re.I,
        ),
        # "Made in China", "Product of Thailand", "Origin: Italy" - only with
        # a country name after it ("A Product of Nestle SA" and "Made in a
        # facility that ..." are not a country of origin).
        re.compile(
            r"(?:made\s+in|product\s+of|manufactured\s+in|(?<![A-Za-z])origin\s*:)\s*(?:the\s+)?"
            r"((?:" + "|".join(_COUNTRIES) + r")\b\.?)",
            re.I,
        ),
    ],
}

MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )
}


# =====================================================================
# Extractor
# =====================================================================

@dataclass
class ExtractionContext:
    """Optional hooks the extractor uses when they are available."""
    image = None
    calibration = None
    measure_glyphs: Optional[Callable] = None


class RegexExtractor:
    """
    Deterministic extraction over OCR spans.

    Two-pass by design:
      Pass 1 - line-level match against a labelled pattern ("MRP: 120").
               High precision; a labelled field is unambiguous.
      Pass 2 - for fields still missing, unlabelled fallback patterns
               over the whole text ("₹120" on its own). Lower precision,
               so matches get lower confidence and the rules engine can
               weigh them accordingly.

    Multi-line fields (addresses) get a lookahead window, because a
    consumer-care block is nearly always 2-4 lines and the first line
    alone fails the reachability check.
    """

    def __init__(self, window_lines: int = 3):
        self.window_lines = window_lines

    def extract(
        self,
        spans: list[TextSpan],
        image=None,
        calibration=None,
    ) -> list[Declaration]:
        spans = [s for s in spans if clean(s.text)]
        # Reading order via row bucketing (see vision.ocr.base.sort_reading_order);
        # a raw (y, x) sort interleaves words from adjacent lines.
        from ..vision.ocr.base import sort_reading_order
        spans = sort_reading_order(spans)
        lines = self._group_into_lines(spans)

        found: dict[str, Declaration] = {}

        # ---- Pass 1: labelled patterns, line + lookahead window -----
        # The anchor must match on THIS line before the window widens.
        # Searching a multi-line window directly lets a field latch onto
        # text belonging to an earlier line: the MRP pattern matched
        # inside the window opened at the product-name line and cut off
        # the "(inclusive of all taxes)" qualifier two lines below - a
        # confident FALSE VIOLATION on a compliant package.
        for line in lines:
            line_text = self._line_text(line)
            for field_id, patterns in PATTERNS.items():
                if field_id in found:
                    continue
                if not self._anchored_here(field_id, line, lines, line_text):
                    continue
                decl = self._match_labelled(field_id, line, lines, line_text, 0.92)
                if decl:
                    found[field_id] = decl

        # ---- Pass 1b: OCR-damaged labels (rules/lexicon.yaml) -------
        # "TINET QUNIIITY:", "Tol Free", "NESTLÉCONSUMER CARE": the label
        # is there but no exact pattern sees it. A lexicon anchor that
        # matches closely enough is swapped in for the damaged words and
        # the normal pattern runs on the repaired line. Marked on the
        # declaration, at lower confidence.
        from . import fuzzy as _fz

        for line in lines:
            line_text = self._line_text(line)
            if _fz.in_negative_context(line_text):
                continue
            for field_id, group in _FUZZY_GROUPS.items():
                if field_id in found:
                    continue
                hit = _fz.find_anchor(line_text, group)
                if hit is None:
                    continue
                canon, score, a, b = hit
                repaired = f"{_CANONICAL_LABEL.get(field_id, canon)} {line_text[b:]}".strip()
                if not PATTERNS[field_id][0].search(repaired):
                    # A block label on a line of its own ("Menutactures by" over
                    # "Tansukh Herbals (P) Ltd.", real jar): the name is below.
                    below = (self._below(line, lines, limit=1)
                             if field_id in _MULTILINE_FIELDS else [])
                    if not (below and PATTERNS[field_id][0].search(
                            f"{repaired} {self._line_text(below[0])}")):
                        continue
                decl = self._match_labelled(field_id, line, lines, repaired, 0.75)
                if decl:
                    decl.notes.append(
                        f"Label read as '{line_text[a:b].strip()}' and matched to "
                        f"'{canon}' by similarity ({score:.0f}/100); confirm on the pack.")
                    found[field_id] = decl

        # ---- Pass 1c: a net quantity printed off its label's row -----
        # Label | value tables are often misregistered: on a real Haldiram
        # pack "35g" sits half a row above "NET QUANTITY:", level with
        # "BATCH NO.:" (whose own value is a vertical inkjet code). A
        # label with nothing to its right takes a value that is ONLY a
        # quantity with a unit, in the value column, within 1.5 rows.
        if "net_quantity" not in found:
            hit = self._offset_quantity(lines)
            if hit is not None:
                found["net_quantity"] = hit

        # ---- Pass 1d: an MRP printed off its label's row ------------
        # "MRP ₹/ USP ₹ (INCL. OF ALL TAXES)" as one label, and the values
        # stacked in the column to its right, a row out of step: "65.00"
        # over "0.93/g" (real Troovy pouch). The amount directly above
        # the per-unit price is the MRP.
        if "retail_sale_price" not in found:
            hit = self._offset_mrp(lines)
            if hit is not None:
                found["retail_sale_price"] = hit

        # ---- Pass 1e: a row of labels over a row of values ---------
        # "BATCH NO | PKD ON | USE BY | M.R.P. (incl. all taxes)" as column
        # headers and the inkjet values in ONE line under them,
        # "R4N257/NOV 2024/OCT 2026 260.00" (real Rajam jar). Paired left to
        # right, only when the number of values equals the number of labels.
        missing = [f for f in ("manufacture_date", "retail_sale_price") if f not in found]
        if missing:
            for fid, decl in self._header_table(spans, lines).items():
                if fid in missing:
                    found[fid] = decl

        # ---- Pass 1f: a quantity in imperial units only --------------
        # "Net Wt. 7 oz" with no metric figure: read it (as printed), so the
        # unit check can say it is not metric - unread, it passed unseen.
        if "net_quantity" not in found:
            imp = re.compile(
                r"(?:net\.?\s*(?:qty|quantity|wt|weight|vol|volume)|contents?)\s*[:.\-]*\s*"
                r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|oz|lbs?|cc)(?![A-Za-z])", re.I)
            metric = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(g|gm|kg|ml|l)(?![A-Za-z])", re.I)
            for line in lines:
                lt = self._line_text(line)
                m = imp.search(lt)
                mm_ = metric.search(lt[m.end():]) if m else None
                if m and mm_:
                    # "Net Wt. 7.05 oz (200 g)": the metric figure is the declaration
                    found["net_quantity"] = Declaration(
                        field_id="net_quantity", raw_text=clean(lt),
                        value=float(mm_.group(1)), unit=normalise_unit(mm_.group(2)),
                        bbox=self._line_bbox(line), spans=list(line), present=True,
                        extraction_confidence=0.8 * _mean_conf(line),
                        notes=["Metric quantity printed after a non-metric one."])
                    break
                if m:
                    found["net_quantity"] = Declaration(
                        field_id="net_quantity", raw_text=clean(m.group(0)),
                        value=float(m.group(1)), unit=m.group(2).lower(),
                        bbox=self._line_bbox(line), spans=list(line), present=True,
                        extraction_confidence=0.8 * _mean_conf(line),
                        notes=["Quantity printed in a non-metric unit."])
                    break

        # ---- Pass 2: unlabelled fallbacks ---------------------------
        mrp_elsewhere = bool(_MRP_ELSEWHERE.search(
            " ".join(self._line_text(l) for l in lines)))
        for idx, line in enumerate(lines):
            line_text = self._line_text(line)
            for field_id, patterns in PATTERNS.items():
                if field_id in found or len(patterns) < 2:
                    continue
                # Unlabelled patterns are greedy by nature: the MRP
                # fallback is just "<currency> <number>", which happily
                # matches the "Unit Sale Price: Rs. 24.00" line when the
                # real MRP is absent. The package still gets flagged, but
                # under the wrong sub-rule and with the wrong evidence
                # crop - and an inspector who opens a violation report
                # about a missing MRP and sees a photo of the unit price
                # stops trusting the tool. So a fallback never fires on a
                # line that plainly belongs to another declaration.
                if _disqualified(field_id, line_text):
                    continue
                if field_id == "retail_sale_price" and (
                        mrp_elsewhere or _mean_conf(line) < 0.85):
                    # The pack says the MRP is on the cap / base / lid /
                    # crimp, or the "price" is a low-confidence fragment:
                    # "RS" + "15" from a rotated strip on a real MyFitness
                    # tub became an MRP of 15.
                    continue
                for pat in patterns[1:]:
                    m = pat.search(line_text)
                    if not m:
                        continue
                    decl = self._build(field_id, m, line, line_text, confidence=0.68)
                    if decl:
                        found[field_id] = decl
                    break

        # ---- Pass 2b: a bare quantity in the coding area ------------
        # "For MRP, USP, NET VOL., MFD. & BATCH NO. SEE CODING AREA" (real
        # Vim pouch): the pack itself says the quantity is printed there,
        # unlabelled ("200ml" along the edge). Only then is a span that is
        # nothing but a quantity taken - and only if it is the only one.
        if "net_quantity" not in found and _QTY_IN_CODING.search(
                " ".join(self._line_text(l) for l in lines)):
            qty_only = re.compile(rf"^\s*({NUMBER})\s*({QTY_UNITS})\s*$", re.I)
            hits = [(sp, qty_only.match(clean(sp.text))) for l in lines for sp in l]
            hits = [(sp, m) for sp, m in hits if m and m.group(2).strip().isalpha()
                    and sp.confidence >= 0.9]
            if len({(m.group(1), m.group(2).lower()) for _, m in hits}) == 1:
                sp, m = hits[0]
                decl = self._build("net_quantity", m, [sp], clean(sp.text), confidence=0.65)
                if decl:
                    decl.notes.append("Unlabelled quantity in the coding area, which the pack "
                                      "names as where the net quantity is printed.")
                    found["net_quantity"] = decl

        # ---- Pass 3: unlabelled date pairs -------------------------
        # Many packs print "See coding area for MFD / USE BY" (Maggi) or
        # "Mfd. & Use before, See above" (Dabur) and put the dates on
        # their own, unlabelled: one inkjet line "MAR/26-NOV/26-D", or two
        # stacked lines "03/2026 (B4)" / "03/2029". When the pack refers
        # to such dates, two dates close together are MFD and USE BY, and
        # the EARLIER one is the manufacture date - chronology decides,
        # not print order. One date alone is never taken: it could be
        # either.
        md = found.get("manufacture_date")
        future_label = md is not None and _FUTURE_NOTE in (md.notes or [])
        if "manufacture_date" not in found or future_label:
            all_text = " ".join(self._line_text(l) for l in lines)
            if _MFG_CONTEXT.search(all_text):
                hit = self._date_pair(lines)
                if hit is not None and future_label and _is_future(hit[1], hit[0]):
                    hit = None          # nothing better: keep the labelled date
                if hit is not None:
                    yr, mon, tok, used_lines = hit
                    spans_used = [sp for l in used_lines for sp in l]
                    found["manufacture_date"] = Declaration(
                        field_id="manufacture_date",
                        raw_text=" / ".join(self._line_text(l) for l in used_lines),
                        value=f"{mon:02d}/{yr}",
                        bbox=self._line_bbox(spans_used),
                        spans=spans_used,
                        extraction_confidence=0.6 * _mean_conf(spans_used),
                        present=True,
                        notes=[
                            f"Read from an unlabelled date pair; the earlier date "
                            f"({clean(tok)}) is taken as manufacture, since a "
                            f"manufacture date cannot follow its use-by date."
                        ],
                    )

        # ---- Pass 4: consumer care from contact details -------------
        # No heading read, but a toll-free number or an e-mail address is
        # on the pack ("：18002668110", "infoohrlindia.com" on a curved
        # bottle where "Customer Care" wrapped out of view). A toll-free
        # number on a retail pack is the consumer helpline in practice;
        # found this way it is marked as such and carries lower weight.
        if "consumer_care" not in found:
            for line in lines:
                lt = self._line_text(line)
                if _disqualified("consumer_care", lt):
                    continue
                m = _TOLL_FREE.search(lt) or _EMAIL_LIKE.search(lt)
                if not m:
                    continue
                window = [line] + [l for l in self._below(line, lines, limit=2)
                                   if _TOLL_FREE.search(self._line_text(l))
                                   or _EMAIL_LIKE.search(self._line_text(l))]
                spans_used = [sp for l in window for sp in l]
                found["consumer_care"] = Declaration(
                    field_id="consumer_care",
                    raw_text=" ".join(self._line_text(l) for l in window),
                    value=clean(m.group(0)),
                    bbox=self._line_bbox(spans_used),
                    spans=spans_used,
                    extraction_confidence=0.5 * _mean_conf(spans_used),
                    present=True,
                    notes=["Identified from a toll-free number / e-mail address; "
                           "no 'consumer care' heading was read. Confirm on the pack."],
                )
                break

        # ---- Pass 5: expiry (use by / best before) --------------------
        exp = self._expiry(lines, found.get("manufacture_date"))
        if exp is not None:
            found["expiry_date"] = exp

        # ---- Pass 6: a rupee sign read as a digit in the unit price --
        # "(₹0.83 per g)" came back as "20.83 per g" (real Maggi sachet).
        # A price per ONE gram / millilitre cannot exceed the MRP of a pack
        # holding at least that much, so a leading 2 or 7 that makes it do
        # so is the ₹, as it already is for the MRP ("MRP.7 10/-").
        usp, mrp = found.get("unit_sale_price"), found.get("retail_sale_price")
        if usp is not None and mrp is not None and isinstance(usp.value, (int, float)) \
                and isinstance(mrp.value, (int, float)) and usp.value > mrp.value > 0:
            m = re.search(r"(?<![\d.])([27])(\d{1,3}\.\d{2,3})\s*(?:/|per)\s*(?:1\s*)?"
                          r"(g|gm|ml|m[lI]|pc|piece|n|u)\b", usp.raw_text or "", re.I)
            if m and float(m.group(2)) <= mrp.value and abs(
                    float(m.group(1) + m.group(2)) - usp.value) < 1e-6:
                usp.value = float(m.group(2))
                usp.notes.append(f"Read as '{m.group(1)}{m.group(2)}'; the leading "
                                 f"'{m.group(1)}' is the ₹ sign (a price per "
                                 f"{m.group(3)} cannot exceed the MRP). Confirm on the pack.")

        # ---- Attach glyph metrics -----------------------------------
        if image is not None:
            from ..vision.glyph import measure_glyphs
            for decl in found.values():
                if decl.bbox is None:
                    continue
                box, hint = decl.bbox, decl.raw_text
                if decl.field_id in _MULTILINE_FIELDS and decl.spans:
                    # Measure the anchor line, not the whole block: the
                    # block's box can take in a bar code or a logo, whose
                    # bars read as impossibly narrow "letters".
                    box, hint = decl.spans[0].bbox, decl.spans[0].text
                decl.glyph = measure_glyphs(image, box, calibration, text_hint=hint)

        # ---- Emit absent fields explicitly --------------------------
        # The rules engine needs to distinguish "we looked and it is not
        # there" from "we never checked". Absent-but-declared is the
        # former; a missing key would be the latter.
        for field_id in PATTERNS:
            found.setdefault(field_id, Declaration(field_id=field_id, present=False))

        return list(found.values())

    # -- helpers ------------------------------------------------------

    def _group_into_lines(
        self, spans: list[TextSpan], y_tol_ratio: float = 0.5
    ) -> list[list[TextSpan]]:
        """
        One OCR region = one line, plus label/value re-joins.

        PaddleOCR already returns LINES, and where it splits a printed
        row into two regions it is usually because they are in different
        columns. The old grouper (written for Tesseract's word boxes)
        merged every region whose centre fell within half a line height
        into one "row". On a real Maggi back panel - three columns of
        tight 8-point text - that spliced the ingredients, the
        manufacturer address and the consumer-care block into the same
        lines, so the manufacturer "address" began with "Green cardamom
        powder" and consumer care was never found at all.

        The one join that IS needed: a fragment that is only a LABEL
        followed, on the same row, by a value - "NET QUANTITY:" ... "6 g",
        "Net Qty.:" ... "45ml" (both real). A label with nothing of its
        own next to a value is a key-value pair, not two columns.

        Vertical spans (inkjet codes printed at 90 degrees) stay alone.
        """
        if not spans:
            return []

        vertical = [s for s in spans if getattr(s, "vertical", False)]
        horiz = [s for s in spans if not getattr(s, "vertical", False)]
        horiz.sort(key=lambda s: (s.bbox.cy, s.bbox.x))

        heights = sorted(s.bbox.h for s in horiz) or [1.0]
        median_h = heights[len(heights) // 2] or 1.0
        pair_gap = median_h * 14.0

        # Labels claim their values FIRST: sorted by centre, the value
        # "45ml" can come before its label "Net Qty.:" (centres 3 px
        # apart on a real bottle) and would otherwise be taken as a line
        # of its own before the label ever looked for it.
        used: set[int] = set()
        lines: list[list[TextSpan]] = []
        for i, s in enumerate(horiz):
            if i in used or not _is_label_only(s.text):
                continue
            # The NEAREST region to the right on the same row, and only if
            # it is a value: "MANUFACTURED" | "ON" | ":" | "03/2026" must
            # not skip "ON :" to grab the date.
            best, best_gap = None, pair_gap
            for j, t in enumerate(horiz):
                if j in used or j == i:
                    continue
                tol = y_tol_ratio * min(s.bbox.h, t.bbox.h)
                if abs(t.bbox.cy - s.bbox.cy) > tol:
                    continue
                gap = t.bbox.x - s.bbox.x2
                if -0.5 * s.bbox.h <= gap < best_gap:
                    best, best_gap = j, gap
            if best is None:
                continue
            t = horiz[best]
            single = " " not in clean(s.text) and " " not in clean(t.text)
            if single and best_gap <= 0.8 * max(s.bbox.h, t.bbox.h):
                # Word-level input: the chaining below handles it. Two
                # single tokens FAR apart ("USP:" ... "Rs.0.29/g" in a
                # label | value table, real Haldiram) never chain, so
                # they are paired here.
                continue
            if not _is_label_only(t.text) and (
                    _VALUE_START.search(clean(t.text))
                    or (_WORD_VALUE_LABEL.match(clean(s.text))
                        and re.match(r"^\s*[A-Za-z]", clean(t.text)))):
                lines.append([s, t])
                used.update((i, best))
        # Table headers: a label with its value in the cell BELOW it -
        # "Manufactured On" over "04/2026" (real imported-goods box label).
        for i, s in enumerate(horiz):
            if i in used or not _is_label_only(s.text) or " " not in clean(s.text):
                continue
            best, best_dy = None, None
            for j, t in enumerate(horiz):
                if j in used or j == i:
                    continue
                dy = t.bbox.cy - s.bbox.cy
                if dy <= 0.5 * min(s.bbox.h, t.bbox.h) or t.bbox.y - s.bbox.y2 > 1.0 * s.bbox.h:
                    continue
                inter = min(s.bbox.x2, t.bbox.x2) - max(s.bbox.x, t.bbox.x)
                if inter < 0.5 * min(s.bbox.w, t.bbox.w):
                    continue
                if best_dy is None or dy < best_dy:
                    best, best_dy = j, dy
            if best is None:
                continue
            t = horiz[best]
            ct = clean(t.text)
            if (not _is_label_only(t.text) and _VALUE_START.search(ct)
                    and len(ct) <= 24 and re.search(r"\d", ct)):
                lines.append([s, t])
                used.update((i, best))
        # Word-level input (single tokens, as a VLM or ground-truth list
        # may supply) is chained back into lines: two adjacent single
        # words on one row are one line. Paddle LINES are never merged
        # this way - "B) Lic. No. 10012025000032" and "18001031947" sit
        # 3 px apart on a real sachet and belong to different columns.
        singles = [(i, s) for i, s in enumerate(horiz) if i not in used]
        chained: dict[int, list[TextSpan]] = {}
        order = sorted(singles, key=lambda t: t[1].bbox.x)
        for i, s in order:
            if i in chained:
                continue
            line = [s]
            chained[i] = line
            if " " in clean(s.text):
                continue
            while True:
                last = line[-1]
                nxt = None
                for j, t in order:
                    if j in chained or " " in clean(t.text):
                        continue
                    if abs(t.bbox.cy - last.bbox.cy) > y_tol_ratio * min(last.bbox.h, t.bbox.h):
                        continue
                    gap = t.bbox.x - last.bbox.x2
                    if -0.3 * last.bbox.h <= gap <= 0.8 * max(last.bbox.h, t.bbox.h):
                        nxt = (j, t)
                        break
                if nxt is None:
                    break
                chained[nxt[0]] = line
                line.append(nxt[1])
        seen_ids = set()
        for line in chained.values():
            if id(line) not in seen_ids:
                seen_ids.add(id(line))
                lines.append(line)

        lines.sort(key=lambda l: (sum(t.bbox.cy for t in l) / len(l), l[0].bbox.x))
        lines.extend([v] for v in sorted(vertical, key=lambda s: s.bbox.x))
        return lines

    def _anchored_here(self, field_id, line, lines, line_text) -> bool:
        """
        Does this field's label start on THIS line? A heading on a line of
        its own ("Regd. Office & Consumer Cell" - Dabur) has its value on
        the lines below, so for block fields the next line is allowed to
        complete the match, as long as the label itself is on this one.
        """
        pat = PATTERNS[field_id][0]
        if pat.search(line_text):
            return True
        if field_id not in _MULTILINE_FIELDS:
            return False
        below = self._below(line, lines, limit=1)
        if not below:
            return False
        probe = f"{line_text} {self._line_text(below[0])}"
        m = pat.search(probe)
        return bool(m) and m.start() < len(line_text)

    def _match_labelled(self, field_id, line, lines, line_text, confidence):
        """Window + pattern + build for one anchored line (pass 1 and 1b)."""
        pat = PATTERNS[field_id][0]
        if field_id in _MULTILINE_FIELDS:
            # The block continues BELOW, in the same column - not "the
            # next lines in reading order", which on a multi-column back
            # panel belong to other columns.
            window = [line]
            texts = [line_text]
            unread_tail = False
            # Consumer care runs to five lines on real packs (office,
            # address, e-mail, website, toll free - Dabur); a
            # manufacturer block listing units by batch code to five too.
            span_n = self.window_lines + (4 if field_id == "consumer_care" else 2)
            # Sideways text (a strip up the side of a tub or along a
            # pouch's seal): the block continues in the sideways lines
            # stacked BESIDE the label, not in whatever lies below it.
            # On a real MyFitness tub "For Consumer Complaints/Queries:"
            # ran up the side, and "below" it was the manufacturing
            # address in the main column.
            follow = (self._beside(line, lines, limit=span_n - 1)
                      if getattr(line[0], "vertical", False)
                      else self._below(line, lines, limit=span_n - 1))
            for nxt in follow:
                nt = self._line_text(nxt)
                # Stop at OCR garbage (bar code, smudges): not part of
                # the address, and glyphs measured over a bar code read
                # as "width ratio 0.16".
                if _mean_conf(nxt) < 0.6 and not re.search(r"[\d@]", nt):
                    # Remember it: on a real Paper Boat pouch the helpline
                    # number and e-mail are on exactly such a line
                    # ("AAT-5RRETOUSATeEe."), so "no phone read" there is
                    # not "no phone printed".
                    unread_tail = True
                    break
                # A block ends where the next declaration begins: with
                # line-level OCR the manufacturer window ran straight on
                # into "Consumer Care: ..." and reported it as address.
                # A label mentioned mid-sentence is a reference, not a new
                # block: "Contact: Quality Manager at Manufactured by
                # address." (real Paras Spices sachet).
                if any((mo := PATTERNS[o][0].search(nt)) and mo.start() < 12
                       for o in PATTERNS if o != field_id) and not (
                        field_id == "consumer_care" and re.search(r"\bcontact\b", nt, re.I)):
                    break
                # ...or where a date / batch / storage line begins: "Packed
                # by: R. B. Products, Guwahati" was continued with "Best
                # before 90 days from packing date" (real).
                if _BLOCK_STOP.search(nt) or (
                        field_id == "manufacturer_details" and _MFR_STOP.search(nt)):
                    break
                window.append(nxt)
                texts.append(nt)
            # Each printed line of an address is one component of it; the
            # join keeps that visible ("DROLIA CHEMICAL, No. 7 Ramlochan
            # Mullick Street, Kolkata-73"), where a plain space made a
            # four-line address look like a one-part fragment.
            subject = _join_block(texts)
            window_unread = unread_tail
            m = pat.search(subject) or pat.search(line_text)
            span_line = [s for l in window for s in l]
        else:
            # Price qualifiers wrap below ("MRP Rs. 443.74" / "(inclusive
            # of all" / "taxes)") or sit to the RIGHT as their own OCR
            # region ("MRP: Rs. 1495.00" | "INCL. OF ALL TAXES", Casio).
            # The value is matched on this line only; the neighbours are
            # there so the qualifier can be seen.
            subject = line_text
            # A quantity qualifier is looked for on the quantity's OWN line,
            # plus a neighbour that is nothing but a qualifier ("(Approx.)"
            # read as its own box). "NUTRITIONAL INFORMATION (Approx.)" on
            # the next line is not about the net quantity (false violation).
            qual_extra = ""
            for nxt in (self._right_of(line, lines, max_gap_ratio=3.0, limit=1)
                        + self._below(line, lines, limit=2)):
                nt = self._line_text(nxt)
                subject = f"{subject} {nt}"
                if _QTY_QUAL_ONLY.fullmatch(nt.strip()):
                    qual_extra = f"{qual_extra} {nt}"
                if _QUALIFIER.search(subject):
                    break
            m = pat.search(line_text)
            span_line = line
            window_unread = False
            # ...and on that line only right around the quantity: "Net Wt.
            # 200 g Wraparound label" is not "200 g around".
            if m:
                lead = _QTY_QUAL_LEAD.match(line_text[m.end():])
                qual_src = (line_text[max(0, m.start() - 12):m.end()]
                            + (lead.group(0) if lead else "") + qual_extra)
            else:
                qual_src = None
        if not m:
            return None
        decl = self._build(field_id, m, span_line, subject, confidence=confidence,
                           qual_src=qual_src if field_id == "net_quantity" else None)
        if decl is not None and window_unread:
            decl.notes.append(UNREAD_TAIL_NOTE)
        return decl

    def _date_pair(self, lines):
        """Two dates on one line, or on two lines stacked in one column."""
        def dates_in(text: str, date_only: bool):
            out = []
            # Dot-matrix inkjet: "N0V/26", "5EP/26" - a digit read for the
            # O or S of a month name. Only inside a letter-bearing 3-char
            # token directly followed by a year.
            text = re.sub(
                r"(?<![A-Za-z0-9])([A-Za-z0-9]{3})(?=\s*[/\-.'’]?\s*\d{2}(?!\d{3}))",
                lambda m: (m.group(1).replace("0", "O").replace("5", "S")
                           if sum(ch.isalpha() for ch in m.group(1)) >= 2 else m.group(1)),
                text)
            for mt in _MONTH_TOKEN.finditer(text):
                mon = MONTHS.get(mt.group(1)[:3].lower())
                yr = _year(mt.group(2))
                if mon and yr:
                    out.append((yr, mon, mt.group(0), mt.span()))
            squeezed = re.sub(r"(?<=\d) (?=\d)", "", text)
            # Both the squeezed text ("03/202 6" -> 03/2026) and the text as
            # read: squeezing glued "15/06/26 135326F094" (date, then batch)
            # into one digit run and lost a plain date (real Paras sachet).
            seen_spans = set()
            for mt in list(_NUMERIC_DATE.finditer(squeezed)) + list(_NUMERIC_DATE.finditer(text)):
                if mt.group(0) in seen_spans:
                    continue
                seen_spans.add(mt.group(0))
                if mt.group(3):                      # dd/mm/yy
                    mon, yr = int(mt.group(2)), _year(mt.group(3))
                else:                                # mm/yyyy
                    mon, yr = int(mt.group(1)), _year(mt.group(2))
                if 1 <= mon <= 12 and yr:
                    out.append((yr, mon, mt.group(0), mt.span()))
            if date_only and out:
                # The line must be ABOUT the date: "03/2026 (B4)" yes,
                # "Lic. No. 10012025000032" or a nutrition line no.
                alnum = len(re.sub(r"[^0-9A-Za-z]", "", squeezed)) or 1
                covered = sum(len(re.sub(r"[^0-9A-Za-z]", "", d[2])) for d in out)
                # ...unless the date itself is unmistakable: a full d/m/y
                # with slashes ("29/08/26,₹10; USP ₹0.071/ml" - Paper Boat's
                # inkjet line carries its price on the same line).
                full = [d for d in out if re.fullmatch(
                    r"\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})|\d{1,2}\s*/?\s*[A-Za-z]{3}\s*/?\s*\d{2,4}",
                    re.sub(r"\s+", "", d[2]) if "/" in d[2] else d[2].strip())]
                if covered / alnum < 0.6 and not full:
                    return []
            return out

        for line in lines:
            lt = self._line_text(line)
            ds = dates_in(lt, date_only=False)
            if len(ds) >= 2 and len({(d[0], d[1]) for d in ds}) >= 2:
                yr, mon, tok, _ = min(ds)
                return yr, mon, tok, [line]
        for line in lines:
            ds = dates_in(self._line_text(line), date_only=True)
            if len(ds) != 1:
                continue
            for nxt in self._below(line, lines, limit=1):
                ds2 = dates_in(self._line_text(nxt), date_only=True)
                if len(ds2) == 1 and (ds2[0][0], ds2[0][1]) != (ds[0][0], ds[0][1]):
                    yr, mon, tok, _ = min(ds + ds2)
                    return yr, mon, tok, [line, nxt]
        return None

    def _expiry(self, lines, mfd):
        """
        The use-by / best-before date, as "MM/YYYY" (or "DD/MM/YYYY").

        Three ways packs print it:
          * labelled: "USE BY: 16/10/26", "EXP:03/JUL/28", "Exp:18/FEB/27";
          * the later date of an unlabelled MFD / USE BY pair;
          * relative: "Best before 3 years from the date of packing",
            "Use within 3 months of Mfd." - computed from the manufacture date.
        """
        def mk(dt, line_list, note):
            yr, mon, day, tok = dt
            spans_used = [sp for l in line_list for sp in l]
            return Declaration(
                field_id="expiry_date",
                raw_text=" / ".join(self._line_text(l) for l in line_list),
                value=(f"{day:02d}/{mon:02d}/{yr}" if day else f"{mon:02d}/{yr}"),
                bbox=self._line_bbox(spans_used) if spans_used else None,
                spans=spans_used, present=True,
                extraction_confidence=0.8 * _mean_conf(spans_used) if spans_used else 0.5,
                notes=[note] if note else [])

        mfd_ym = None
        if mfd is not None and mfd.present and isinstance(mfd.value, str):
            mm = re.match(r"(\d{2})/(\d{4})$", mfd.value)
            if mm:
                mfd_ym = (int(mm.group(2)), int(mm.group(1)))

        # 1. labelled
        for line in lines:
            lt = self._line_text(line)
            lab = _EXPIRY_LABEL.search(lt)
            if not lab:
                continue
            # "Best before 12 months from date of mfg. 08/2026": the date on
            # the line is the MANUFACTURE date - step 3 computes the expiry.
            # Read as the expiry it made a fresh pack "expired stock".
            if mfd_ym and _REL_EXPIRY.search(lt):
                continue
            all_ds = _dates_in_text(lt[lab.end():])
            if not all_ds:
                # "EXP 12/27", "Use by: 06/27": month/two-digit year, only
                # right after an expiry label (elsewhere 12/27 is ambiguous).
                my = re.match(r"\W{0,4}(0?[1-9]|1[0-2])\s*[/\-.]\s*(\d{2})(?![\d/\-.])", lt[lab.end():])
                if my:
                    all_ds = [(2000 + int(my.group(2)), int(my.group(1)), None, my.group(0).strip(" :.-"))]
            ds = [d for d in all_ds if not mfd_ym or (d[0], d[1]) >= mfd_ym]
            if all_ds and not ds:
                # Only dates EARLIER than the manufacture date: a misread, or a
                # wrongly printed pack - not silently dropped.
                return mk(all_ds[0], [line],
                          f"This date is earlier than the manufacture date {mfd.value}: "
                          f"misread or wrongly printed. Confirm on the pack.")
            if ds:
                # The value column a row out of step: "USE BY" paired with
                # the PACKED date (real Britannia and Mayora packs: "USE BY
                # 20/08/26" where 20/08/26 is PKD and 19/02/27 sits a row
                # lower). An expiry in the manufacture month, with a later
                # date printed close by, is that later date.
                if mfd_ym and (ds[0][0], ds[0][1]) == mfd_ym:
                    i = next(k for k, l in enumerate(lines) if l is line)
                    later = [(abs(k - i), d, l) for k, l in enumerate(lines) if abs(k - i) <= 3
                             for d in _dates_in_text(self._line_text(l))
                             if (d[0], d[1]) > mfd_ym]
                    if later:
                        _, d, l = min(later, key=lambda t: t[0])
                        return mk(d, [line, l] if l is not line else [line],
                                  "The value beside USE BY is the packing date (the "
                                  "value column is a row out of step); the later date "
                                  "is the use-by date. Confirm on the pack.")
                return mk(ds[0], [line], "")
        # 2. the later date of the pair the manufacture date came from
        if mfd is not None and any("date pair" in n for n in (mfd.notes or [])):
            ds = _dates_in_text(mfd.raw_text or "")
            if len(ds) >= 2:
                late = max(ds, key=lambda d: (d[0], d[1], d[2] or 0))
                if (late[0], late[1]) != (ds[0][0], ds[0][1]) or len({(d[0], d[1]) for d in ds}) > 1:
                    return mk(late, [], "The later date of the MFD / USE BY pair.")
        # 3. relative to the manufacture date
        if mfd_ym:
            for line in lines:
                lt = self._line_text(line)
                m = _REL_EXPIRY.search(lt)
                if not m:
                    continue
                n, unit = int(m.group(1)), m.group(2).lower()
                months = n * 12 if unit.startswith("year") else n if unit.startswith("month") else None
                if months is None:                      # days: whole months, rounded up
                    months = (n + 29) // 30
                y, mo = mfd_ym
                tot = y * 12 + (mo - 1) + months
                return mk((tot // 12, tot % 12 + 1, None, lt), [line],
                          f"Computed: {n} {unit} from the manufacture date {mfd.value}.")
        return None

    def _offset_quantity(self, lines):
        # "NET QTY. (g)", and its first letter lost at a fold: "IET QTY. (g)",
        # "NET QTY(9)" (real Troovy stickers).
        label_re = re.compile(
            r"^\s*[NIH]?ET\.?\s*(?:qty|quantity|wt\.?|weight|content)s?\.?\s*"
            r"(?:\(\s*[gq9]\s*\)|\(\s*m?l\s*\))?\s*[:.\-]*\s*"
            # a stray fragment after the label: "NET QUANTITY: 04" (real)
            r"(?:\S{1,3})?\s*$", re.I)
        qty_re = re.compile(rf"^\s*({NUMBER})\s*({QTY_UNITS})\s*$", re.I)
        for line in lines:
            if not label_re.match(clean(line[0].text)) or len(line) > 2:
                continue
            # A label paired with a value that is not a quantity ("NET
            # QTY(9)" + "55.00" on a misaligned sticker) is still unanswered.
            if len(line) == 2 and qty_re.match(clean(line[1].text)):
                continue
            lab = line[0].bbox
            if len(line) == 1 and any(
                    qty_re.match(self._line_text(r))
                    for r in self._right_of(line, lines, max_gap_ratio=14.0, limit=1)):
                continue
            best = None
            for other in lines:
                for sp in other:
                    m = qty_re.match(clean(sp.text))
                    if not m or not m.group(2).strip().isalpha() or sp is line[0]:
                        continue
                    right = (sp.bbox.x >= lab.x2 - 0.5 * lab.h
                             and abs(sp.bbox.cy - lab.cy) <= 1.5 * max(lab.h, sp.bbox.h)
                             and sp.bbox.x - lab.x2 <= 14 * lab.h)
                    # ...or directly UNDER the label: "NET QUANTITY:" with
                    # "140ml" on the next line (real pouch).
                    below = (sp.bbox.cy > lab.cy
                             and sp.bbox.y - lab.y2 <= 1.5 * lab.h
                             and min(sp.bbox.x2, lab.x2) - max(sp.bbox.x, lab.x) > 0)
                    if not (right or below):
                        continue
                    dy = abs(sp.bbox.cy - lab.cy)
                    if best is None or dy < best[0]:
                        best = (dy, sp, m)
            if best is None:
                continue
            _, sp, m = best
            decl = self._build("net_quantity", m, [line[0], sp],
                               f"{clean(line[0].text)} {clean(sp.text)}", confidence=0.7)
            if decl:
                decl.raw_text = f"{clean(line[0].text)} {clean(sp.text)}"
                decl.notes.append("Value printed beside the label but off its row; "
                                  "confirm on the pack.")
            return decl
        return None

    _HDR = [
        ("batch", re.compile(r"^\W*(?:batch|b\.?\s*no|lot)\b[\s.:no]*$", re.I)),
        ("manufacture_date", re.compile(
            r"^\W*(?:pkd|packed|mfd|mfg|manufactured|mfg\.?\s*date|date\s+of\s+(?:mfg|manufacture|packing))"
            r"\.?(?:\s*(?:on|date|dt))?\W*$", re.I)),
        ("expiry_date", re.compile(r"^\W*(?:use\s*by|exp(?:iry)?|best\s*before)\.?(?:\s*(?:date|dt))?\W*$", re.I)),
        ("retail_sale_price", re.compile(r"^\W*m\.?\s*r?\.?\s*p\.?\s*(?:₹|rs\.?)?\W*$", re.I)),
    ]

    def _header_table(self, spans, lines) -> dict:
        labels = []
        for sp in spans:
            if getattr(sp, "vertical", False):
                continue
            t = clean(sp.text)
            for kind, rx in self._HDR:
                if len(t) <= 20 and rx.match(t):
                    labels.append((kind, sp))
                    break
        out: dict = {}
        if len(labels) < 3:
            return out
        labels.sort(key=lambda ks: ks[1].bbox.x)
        h = sorted(sp.bbox.h for _, sp in labels)[len(labels) // 2]
        # one header row: every label within 2 line heights of the median
        cy = sorted(sp.bbox.cy for _, sp in labels)[len(labels) // 2]
        row = [(k, sp) for k, sp in labels if abs(sp.bbox.cy - cy) <= 2.0 * h]
        kinds = [k for k, _ in row]
        if len(row) < 3 or len(set(kinds)) != len(kinds):
            return out
        x0 = min(sp.bbox.x for _, sp in row)
        x1 = max(sp.bbox.x2 for _, sp in row)
        y1 = max(sp.bbox.y2 for _, sp in row)
        for sp in spans:
            b = sp.bbox
            if not (0 <= b.cy - y1 <= 3.0 * h) or b.w < 0.6 * (x1 - x0):
                continue
            # Values found by what they look like (a date, a price), each
            # given to the label column nearest where it STARTS along the
            # line (inkjet is near-monospaced; values are left-aligned under
            # their labels). Only a date under a date label, a price under
            # the MRP label.
            text = clean(sp.text or "")
            if not text:
                continue
            def col_at(pos, kinds):
                wx = b.x + b.w * pos / float(len(text))
                cand = [(abs(lab.bbox.x - wx), k, lab) for k, lab in row if k in kinds]
                return min(cand, key=lambda t: t[0])[1:] if cand else (None, None)
            picks = {}
            found_dates = []
            for seg in re.finditer(r"[^/]+", text):          # "R4N257/NOV 2024/..."
                for d in _dates_in_text(seg.group(0).strip()):
                    at = text.find(d[3], seg.start())
                    if at >= 0:
                        found_dates.append((at, d))
            for at, d in sorted(found_dates, key=lambda t: t[0]):
                k, lab = col_at(at, ("manufacture_date", "expiry_date"))
                if k and k not in picks:
                    picks[k] = (lab, d[3], f"MFD: {d[3]}" if k == "manufacture_date" else None)
            prices = [m for m in re.finditer(r"(?<![\d/.])(\d{1,5}\.\d{1,2})(?![\d/])", text)]
            if prices:
                m = prices[-1]
                k, lab = col_at(m.start(), tuple(kk for kk, _ in row))
                if k == "retail_sale_price":
                    qual = [q for q in spans if q is not lab and q is not sp
                            and 0 < q.bbox.cy - lab.bbox.cy <= 1.8 * h
                            and abs(q.bbox.cx - lab.bbox.cx) <= 3 * h]
                    picks[k] = (lab, m.group(1),
                                f"MRP {m.group(1)} " + " ".join(clean(q.text) for q in qual))
            for kind, (lab, tok, subject) in picks.items():
                if subject is None:
                    continue
                mm = next((mt for pat in PATTERNS[kind] if (mt := pat.search(clean(subject)))), None)
                if not mm:
                    continue
                decl = self._build(kind, mm, [lab, sp], clean(subject), confidence=0.7)
                if decl:
                    decl.raw_text = f"{clean(lab.text)} {tok}"
                    decl.notes.append("Read from a table: label in the header row, value in "
                                      "the row under it; confirm on the pack.")
                    out[kind] = decl
            break
        return out

    def _offset_mrp(self, lines):
        label_re = re.compile(
            r"^\W*(?:M\.?\s*R\.?\s*P\.?|max(?:imum|\.)?\s*retail\s+price)\s*(?:₹|rs\.?|r|z|\?)?\s*"
            r"(?:/\s*(?:[uo0][s8]p|unit\s+sale\s+price)\.?\s*(?:₹|rs\.?|r|z|\?)?)?\s*[:.\-]*\s*"
            r"(?:\(?\s*incl\.?.{0,24}\)?)?\s*$", re.I)
        amount_re = re.compile(r"^\s*(?:₹|rs\.?)?\s*(\d{1,6}(?:\.\d{2})?)\s*(?:/-)?\s*$", re.I)
        per_unit_re = re.compile(rf"\d\s*(?:/|per)\s*(?:\d+\s*)?{QTY_UNITS}\b", re.I)
        spans = [sp for l in lines for sp in l if not getattr(sp, "vertical", False)]
        for line in lines:
            lt = self._line_text(line)
            if len(line) > 1 or not label_re.match(lt):
                continue
            lab = line[0].bbox
            h = max(lab.h, 1.0)
            wants_usp = bool(re.search(r"/\s*(?:[uo0][s8]p|unit)", lt, re.I))
            cands = []
            for sp in spans:
                m = amount_re.match(clean(sp.text))
                if not m or sp is line[0]:
                    continue
                b = sp.bbox
                if not (b.x >= lab.x2 - 0.5 * h and b.x - lab.x2 <= 14 * h
                        and abs(b.cy - lab.cy) <= 2.5 * h):
                    continue
                # Directly above a per-unit price, in the same column?
                under = [u for u in spans if u is not sp and per_unit_re.search(clean(u.text))
                         and 0 < u.bbox.cy - b.cy <= 1.6 * max(b.h, u.bbox.h)
                         and min(u.bbox.x2, b.x2) - max(u.bbox.x, b.x) > 0.3 * min(u.bbox.w, b.w)]
                if wants_usp and not under:
                    continue
                cands.append((abs(b.cy - lab.cy), sp, m))
            if not cands:
                continue
            _, sp, m = min(cands, key=lambda t: t[0])
            below = self._below(line, lines, limit=1)
            subject = f"MRP {m.group(1)} {lt}" + (f" {self._line_text(below[0])}" if below else "")
            mm = PATTERNS["retail_sale_price"][0].search(subject)
            if not mm:
                continue
            decl = self._build("retail_sale_price", mm, [line[0], sp], subject, confidence=0.7)
            if decl:
                decl.raw_text = f"{lt} {clean(sp.text)}"
                decl.notes.append("Value printed beside the label but off its row; "
                                  "confirm on the pack.")
            return decl
        return None

    @staticmethod
    def _ink_h(line: list[TextSpan], fallback: float) -> float:
        hs = [s.ink_bbox.h for s in line
              if getattr(s, "ink_bbox", None) is not None and s.ink_bbox.h < s.bbox.h - 1]
        return max(4.0, sorted(hs)[len(hs) // 2]) if hs else fallback

    @staticmethod
    def _line_h(line: list[TextSpan]) -> float:
        return max(1.0, sorted(t.bbox.h for t in line)[len(line) // 2])

    def _right_of(self, line, lines, max_gap_ratio: float = 14.0, limit: int = 2):
        """Lines on the SAME row, to the right, nearest first."""
        b = self._line_bbox(line)
        h = self._line_h(line)
        out = []
        for other in lines:
            if other is line:
                continue
            ob = self._line_bbox(other)
            if abs(ob.cy - b.cy) > 0.5 * min(h, self._line_h(other)):
                continue
            gap = ob.x - b.x2
            if -0.5 * h <= gap <= max_gap_ratio * h:
                out.append((gap, other))
        return [o for _, o in sorted(out, key=lambda t: t[0])[:limit]]

    def _beside(self, line, lines, limit: int = 3):
        """
        For a sideways (vertical) line: the sideways lines stacked next to
        it, nearest first on each side, in left-to-right order. Which side
        the text continues on depends on which way it was rotated, and
        Paddle turns upside-down crops round before reading, so the box
        corners cannot tell; the neighbours on both sides are taken.
        """
        vert = [l for l in lines if l is not line and getattr(l[0], "vertical", False)]
        cb = self._line_bbox(line)
        th = max(4.0, min(cb.w, cb.h))              # text height, sideways

        def adjacent(a, b):
            ov = min(a.y2, b.y2) - max(a.y, b.y)
            if ov < 0.2 * min(a.h, b.h):
                return False
            gap = max(b.x - a.x2, a.x - b.x2)
            return gap <= 1.0 * th

        out, frontier = [], [line]
        while frontier and len(out) < limit:
            nxt_front = []
            for cur in frontier:
                cbb = self._line_bbox(cur)
                for other in sorted(vert, key=lambda l: abs(self._line_bbox(l).cx - cbb.cx)):
                    if other in out or len(out) >= limit:
                        continue
                    if adjacent(cbb, self._line_bbox(other)):
                        out.append(other)
                        nxt_front.append(other)
            frontier = nxt_front
        return sorted(out, key=lambda l: self._line_bbox(l).cx)

    def _below(self, line, lines, limit: int = 3, max_gap_ratio: float = 1.2):
        """
        The lines directly BELOW, in the same column, top to bottom.

        Same column = left edges within ~1.5 line heights, or at least
        half the narrower line overlapping horizontally. Stops at the
        first vertical gap larger than `max_gap_ratio` line heights, so a
        block ends where the print leaves space.
        """
        out: list[list[TextSpan]] = []
        cur = line
        pool = [l for l in lines if l is not line and not getattr(l[0], "vertical", False)]
        while len(out) < limit:
            cb = self._line_bbox(cur)
            h = self._line_h(cur)
            best, best_dy = None, None
            for other in pool:
                if other in out:
                    continue
                ob = self._line_bbox(other)
                # A mark, not a line: a lone "®"/"™" read as "R" or a dot,
                # far shorter than the text (24 px against 70 px on a real
                # Monster can). Taken as the next line it ended the
                # consumer-care block before the phone and e-mail.
                if (len(re.sub(r"\s", "", self._line_text(other))) <= 2
                        and self._line_h(other) < 0.6 * h):
                    continue
                dy = ob.cy - cb.cy
                # Half the SMALLER height: a bold "DROLIA CHEMICAL" (62 px
                # box) overlaps the address line under it, and half its own
                # height treated that line as the same row and skipped it.
                # Same row? Judged on the INK height where it is known:
                # Paddle's padded boxes on small print are 2-4x the letters,
                # and two address lines 8 px apart (real Paras sachet, 28-40
                # px boxes) looked like one row, so the PIN line was skipped.
                he = min(self._ink_h(cur, h), self._ink_h(other, self._line_h(other)))
                if dy <= min(0.3 * he, 0.2 * he + 2.0):
                    continue
                if ob.y - cb.y2 > max_gap_ratio * h:
                    continue
                inter = max(0.0, min(cb.x2, ob.x2) - max(cb.x, ob.x))
                narrower = max(1.0, min(cb.w, ob.w))
                if abs(ob.x - cb.x) > 1.5 * h and inter / narrower < 0.5:
                    continue
                if best_dy is None or dy < best_dy:
                    best, best_dy = other, dy
            if best is None:
                break
            out.append(best)
            cur = best
        return out

    @staticmethod
    def _same_column(a: list[TextSpan], b: list[TextSpan],
                     min_overlap: float = 0.35) -> bool:
        """Kept for callers outside this class; see _below for the rule used here."""
        ax1 = min(s.bbox.x for s in a); ax2 = max(s.bbox.x2 for s in a)
        bx1 = min(s.bbox.x for s in b); bx2 = max(s.bbox.x2 for s in b)
        inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        narrower = min(ax2 - ax1, bx2 - bx1) or 1.0
        return (inter / narrower) >= min_overlap

    @staticmethod
    def _line_text(line: list[TextSpan]) -> str:
        return clean(" ".join(s.text for s in line))

    @staticmethod
    def _line_bbox(line: list[TextSpan]) -> BBox:
        x = min(s.bbox.x for s in line)
        y = min(s.bbox.y for s in line)
        x2 = max(s.bbox.x2 for s in line)
        y2 = max(s.bbox.y2 for s in line)
        return BBox(x, y, x2 - x, y2 - y)

    def _build(
        self,
        field_id: str,
        match: re.Match,
        line: list[TextSpan],
        subject: str,
        confidence: float,
        qual_src: Optional[str] = None,
    ) -> Optional[Declaration]:
        raw = clean(match.group(0))
        bbox = self._line_bbox(line)
        value, unit = None, None
        notes: list[str] = []

        # A prohibited qualifier sits AFTER the quantity - "500 g
        # (approx.)" - so trimming evidence to the match alone hides the
        # very words Rule 12(6) forbids. Re-attach the qualifier itself,
        # not the rest of the line, so the check can see it without the
        # declaration swallowing its neighbours.
        if field_id == "net_quantity":
            banned = re.search(
                r"\(?\s*(?<![a-z])(?:approx(?:imately|imate)?\.?|apprx\.?|about|minimum|min\.?|"
                r"maximum|max\.?|average|avg\.?|not\s+less\s+than|nearly|around|"
                r"upto|up\s+to|more\s+or\s+less|\+\s*/\s*-|±|~)(?![a-z])\s*\)?"
                # ...but "Max Retail Price" on the next line is the MRP
                # label, not "500 g (max)" (real Drolia / MHM packs).
                r"(?!\s*\.?\s*retail)",
                qual_src if qual_src is not None else subject, re.I,
            )
            if banned:
                q = clean(banned.group(0))
                if q and q.lower() not in raw.lower():
                    raw = f"{raw} {q}"

        if field_id in ("retail_sale_price", "unit_sale_price"):
            try:
                value = float(normalise_numeric(match.group(1)).replace(",", ""))
            except (ValueError, IndexError):
                return None
            # The "inclusive of all taxes" qualifier is part of the legal
            # declaration, so the phrasing check must be able to see it -
            # but taking the WHOLE window as evidence drags in whatever
            # else shares the line. On a multi-column back panel that
            # produced an MRP declaration whose raw_text carried the
            # entire nutrition block, which is useless as evidence: an
            # inspector is shown a paragraph and asked to accept that the
            # price is somewhere in it.
            #
            # Take the match itself, then re-attach ONLY the qualifier,
            # wherever it sits in the window.
            raw = clean(match.group(0))
            qualifier = _QUALIFIER.search(subject)
            if qualifier:
                q = clean(qualifier.group(0))
                if q.lower() not in raw.lower():
                    raw = f"{raw} {q}"

        elif field_id == "net_quantity":
            try:
                value = float(normalise_numeric(match.group(1)).replace(",", ""))
                try:
                    # Normalise BEFORE lowercasing: the confusion map
                    # keys on case ("I" is a mangled litre, "i" is not).
                    unit = normalise_unit(match.group(2)).lower()
                except (IndexError, re.error):
                    # The merged-litre fallback captures only the number;
                    # the unit is implied by the pattern itself.
                    unit = "l"
            except (ValueError, IndexError):
                return None
            try:
                unit_tok = match.group(2)
            except IndexError:
                unit_tok = None
            if unit_tok is None:
                notes.append("Litre inferred from a trailing '1' after the "
                             "quantity (\"1 l\" read as \"11\"); confirm on the pack.")
                confidence *= 0.5
            elif unit_tok.strip() in {"6", "9", "0", "&", "q", "|", "I", "!"}:
                notes.append(f"Unit '{unit}' inferred from a '{unit_tok.strip()}' "
                             f"glyph that OCR commonly confuses with it; confirm "
                             f"on the pack.")
                confidence *= 0.5
            # Do NOT reset raw here. The prohibited-qualifier text was
            # already re-attached above, and overwriting it discarded the
            # very words Rule 12(6) forbids - the check then passed a
            # label reading "500 g (approx.)".

        elif field_id == "manufacture_date":
            try:
                a, b = match.group(1), match.group(2)
                c = match.group(3) if (match.re.groups or 0) >= 3 else None
                if c:
                    # dd/mm/yy: "MFG. DATE: 19/06/26" (a real Haldiram pack)
                    a, b = b, c
                month = MONTHS.get(a[:3].lower()) if a[:1].isalpha() else int(a)
                if len(b.strip()) == 1:
                    # "MFD. 2.9/FR 228..." (real Bingo inkjet) is not
                    # February 2009 - a year has two or four digits.
                    return None
                year = int(b)
                if year < 100:
                    year += 2000
                if not month or not 1 <= month <= 12:
                    return None
                # A manufacture date cannot be in the future. On a real Troovy
                # sticker OCR set the label "MFD." beside the USE BY value
                # (17MAY2027); taking it would report the expiry as the
                # manufacture date. Refused here, the date pair decides.
                value = f"{month:02d}/{year}"
                if _is_future(month, year):
                    notes.append(_FUTURE_NOTE)
            except (ValueError, IndexError):
                return None

        else:
            try:
                value = clean(match.group(1))
            except IndexError:
                value = raw
            if field_id in _MULTILINE_FIELDS:
                value = value.lstrip(" ,;:-")
                # "DEIHI" -> DELHI, "Gurugrom" -> Gurugram: repaired in the
                # value shown to the officer; raw OCR lines stay as read.
                from .spelling import repair
                value, fixes = repair(value)
                if fixes:
                    notes.append("Spelling repaired from the OCR read: "
                                 + ", ".join(fixes[:6]) + ".")
                # Multi-line fields legitimately wrap (an address runs to
                # three lines), so the window IS the evidence here. It is
                # bounded by _same_column at the point the window is
                # built, so it cannot cross into a neighbouring column.
                raw = clean(subject)

        return Declaration(
            field_id=field_id,
            raw_text=raw,
            value=value,
            unit=unit,
            bbox=bbox,
            spans=list(line),
            extraction_confidence=confidence * _mean_conf(line),
            present=True,
            notes=notes,
        )


_MULTILINE_FIELDS = {"manufacturer_details", "consumer_care"}

# A labelled "manufacture" date in the future. Kept (a pack really printed
# that way is a date_plausible violation) unless a date PAIR on the pack
# gives an earlier one: on a real sticker OCR set "MFD." beside the USE BY
# value, and the pair "21AUG2026 / 17MAY2027" is the truth.
_FUTURE_NOTE = "Labelled date is after today; checked against the date pair on the pack."


_EXPIRY_LABEL = re.compile(
    r"\b(?:[ud]se\s*by|use\s*before|best\s*before|exp(?:iry)?(?:\s*date)?|expires?(?:\s*on)?)\b\s*[:.\-]?",
    re.I)
_REL_EXPIRY = re.compile(
    r"(?:best\s*before|use\s*(?:within|before)|consume\s*within)\s*(\d{1,3})\s*"
    r"(months?|days?|years?)\s*(?:from|of|after)", re.I)


def _dates_in_text(text: str) -> list[tuple]:
    """Every date in the text, in order: (year, month, day|None, token)."""
    out = []
    t = text or ""
    for m in re.finditer(
            r"(?<![\d/])(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})(?![\d/])", t):
        d, mo, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
        if y and 1 <= mo <= 12 and 1 <= d <= 31:
            out.append((m.start(), (y, mo, d, m.group(0))))
    for m in re.finditer(
            r"(?<![A-Za-z])(?:(\d{1,2})\s*[/\-.]?\s*)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
            r"[a-z]*\.?\s*[/\-.'’]?\s*(\d{4}|\d{2})(?!\d)", t, re.I):
        y, mo = _year(m.group(3)), MONTHS.get(m.group(2)[:3].lower())
        d = int(m.group(1)) if m.group(1) else None
        if y and mo and (d is None or 1 <= d <= 31):
            out.append((m.start(), (y, mo, d, m.group(0))))
    for m in re.finditer(r"(?<![\d/.])(\d{1,2})\s*/\s*(\d{4})(?![\d/])", t):
        mo, y = int(m.group(1)), _year(m.group(2))
        if y and 1 <= mo <= 12:
            out.append((m.start(), (y, mo, None, m.group(0))))
    seen, res = set(), []
    for _, dt in sorted(out, key=lambda x: x[0]):
        k = (dt[0], dt[1], dt[2])
        if k not in seen:
            seen.add(k)
            res.append(dt)
    return res


def _is_future(month: int, year: int) -> bool:
    from ..core.rules_engine import today_date
    t = today_date()
    nxt = (t.year + (t.month == 12), t.month % 12 + 1)
    return (year, month) > nxt

# Put on a block declaration whose next line could not be read (see the
# rules engine: a contact that may be on that line is not "missing").
UNREAD_TAIL_NOTE = "The block continues into a line OCR could not read; confirm on the pack."

# "For MRP (incl. of all taxes) ... see Cap/Bottom", "SEE LID/BASE", "see
# base of can", "See on Crimp", "see top of the pack": the price is on
# another side of this pack.
# The pack names the coding area (or a side/edge/seal) as where the net
# quantity is printed: "For MRP, USP, NET VOL., #MFD & BATCH NO. SEE CODING AREA".
_QTY_IN_CODING = re.compile(
    r"\bnet\.?\s*(?:qty|quantity|vol|volume|wt|weight|content)\b.{0,120}?\bsee\b.{0,30}?"
    r"(?:coding\s+area|below|side|edge|seal)", re.I | re.S)

_MRP_ELSEWHERE = re.compile(
    # ("see above / below" means THIS panel, so it is not in the list.)
    r"\bmrp\b.{0,160}?\bsee\b.{0,40}?\b(?:cap|bottom|base|lid|crimp|top|side|back|carton)",
    re.I | re.S)

def _join_block(texts: list[str]) -> str:
    """Lines of an address block, comma-separated unless the line already
    ends in punctuation or is only the label ("Mfg. by:", "Packed &
    Marketed")."""
    out = ""
    for i, t in enumerate(texts):
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif re.search(r"[,:;\-]$", out) or (i == 1 and _LABEL_LINE.match(out)):
            out = f"{out} {t}"
        else:
            out = f"{out}, {t}"
    return out


_LABEL_LINE = re.compile(
    r"^\s*(?:(?:manufactured|processed|mfg|mfd|packed|pkd|marketed|mkt|imported)\.?\s*"
    r"(?:(?:&|and)\s*\w+\.?\s*)?(?:by|for)?\.?|name\s*(?:and|&)\s*address.*|"
    r"(?:consumer|customer)\s*(?:care|service)s?.*)\s*:?\s*$", re.I)


# ...and for a manufacturer block, the licence / registration / web lines
# printed under the address ("fssai Lic No.", "CPCB Regn. No.", "Website").
_MFR_STOP = re.compile(
    r"^\W*(?:fs+a|lic\.?\s*no|cpcb|website|www\.|for\s+(?:any\s+quer|consumer|feedback)|"
    r"m[ao]?nufactur\w*\s+(?:by|in)|marketed\s+by|brand\s+owner)", re.I)

# Lines that start a different kind of information and so end an address
# or consumer-care block.
_BLOCK_STOP = re.compile(
    r"^\s*(?:best\s*before|use\s*by|expiry|exp\.?\s*date|batch|lot\s*no|b\.?\s*no\b|"
    r"net\s*(?:wt|weight|qty|quantity)|store\s|storage|ingredients|nutrition|"
    r"mfg\.?\s*date|pkd\.?\s*on|packed\s+on|date\s+of|"
    # "For MRP (incl. of all taxes), Date of Packaging ... see top" (Amul)
    r"for\s+(?:mrp|batch|price|net\s*wt|manufacturing\s+(?:unit|date))|"
    r"keep\s|recommended|no\.?\s*of\s*serv|serving|fs+ai|lic\.?\s*no)", re.I)

# Lexicon anchor group per field, and the label the repaired line gets.
_FUZZY_GROUPS = {
    "net_quantity": "net_qty",
    "retail_sale_price": "mrp",
    "unit_sale_price": "unit_price",
    "manufacture_date": "mfg_date",
    "manufacturer_details": "manufacturer",
    "consumer_care": "consumer_care",
}
# No trailing colon: the damaged label's own ":" follows in the repaired line.
_CANONICAL_LABEL = {
    "net_quantity": "Net Quantity",
    "retail_sale_price": "MRP",
    "unit_sale_price": "Unit Sale Price",
    "manufacture_date": "Mfd",
    "manufacturer_details": "Manufactured by",
    "consumer_care": "Consumer Care",
}

# The "inclusive of all taxes" qualifier in the forms packs actually use,
# tolerant of the spacing OCR loses: "(Incl.of all taxes)", "Inclusive of
# taxes", "INCL OF ALL TAXES", and the Hindi "(सभी करों सहित)".
# A qualifier directly after the quantity ("200 g (approx.)", "500 g upto",
# "200 g ± 5 g"); "Best before upto 9 months" further along is not one.
_QTY_QUAL_LEAD = re.compile(
    r"\W{0,3}(?<![a-z])(?:approx(?:imately|imate)?|apprx|about|min(?:imum)?|max(?:imum)?|avg|"
    r"average|nearly|around|upto|up\s*to|not\s+less\s+than|more\s+or\s+less)(?![a-z])"
    r"(?!\s*\.?\s*retail)\.?\s*\)?|\s{0,3}\(?\s*(?:\+\s*/\s*-|±|~)", re.I)

_QTY_QUAL_ONLY = re.compile(
    r"\W*(?:approx(?:imately|imate)?|apprx|about|min(?:imum)?|max(?:imum)?|avg|average|"
    r"nearly|around|not\s+less\s+than|e|℮)\W*", re.I)

_QUALIFIER = re.compile(
    r"\(?\s*(?:incl(?:usive|\.)?|inc\.?)\s*\.?\s*(?:of)?\s*(?:a[l1iI|]+)?\s*tax(?:es)?\s*\)?"
    r"|\(?\s*(?:सभी\s*)?करों?\s*सहित\s*\)?",
    re.I,
)

# A fragment that is a declaration label and nothing else ("NET QUANTITY",
# "MRP:", "Mfd.") - see _group_into_lines, step 3.
_LABEL_ONLY = re.compile(
    # a commodity word may head it: "BISCUITS NET WEIGHT" (real Britannia)
    r"^\s*(?:(?:[a-z]{3,}\s+)?[nih]?et\.?\s*(?:qty|quantity|wt\.?|weight|vol\.?|volume|content)\.?"
    r"(?:\s*\(\s*(?:[gq9]|m?l)\s*\))?|contents?|"
    r"(?:m\.?\s*r\.?\s*p\.?|[a-z]\.\s*r\.\s*p\.?)(?:\s*(?:₹|rs\.?|r))?|"
    r"max(?:imum|\.)?\s*retail\s+price|"
    # "MFG. DATE:", "Pkd On", "Packed on" (real Haldiram / MHM packs)
    r"(?:mfg|mfd|pkd|manufacturing|manufactured|packing|packed)\.?\s*(?:date|on)?|"
    r"date\s+of\s+(?:mfg|manufacture|packing)|"
    r"unit\s+sale\s+price|u\.?\s*s\.?\s*p\.?|best\s*before|use\s*by|expiry(?:\s+date)?|"
    r"batch\s*no\.?|"
    r"country\s+of\s+origin|generic\s+name|common\s+name|"
    r"शुद्ध\s*(?:मात्रा|भार|वजन)|अधिकतम\s*खुदरा\s*मूल्य|"
    r"निर्माण\s*(?:की\s*)?तिथि)\s*[:.\-]*\s*$",
    re.I,
)
# Labels whose value is WORDS, not a number: a table cell "Country Of Origin"
# | "People's Republic of China", "Generic Name" | "... Ethernet Converter"
# (real Zebronics box).
_WORD_VALUE_LABEL = re.compile(r"^\s*(?:country\s+of\s+origin|generic\s+name|common\s+name)\W*$", re.I)
def _is_label_only(text: str) -> bool:
    """A bare label - exactly ("NET QUANTITY:") or OCR-damaged ("TINET QUNIIITY:")."""
    t = clean(text)
    if _LABEL_ONLY.search(t):
        return True
    if len(t) > 24 or re.search(r"\d", t):
        return False
    from . import fuzzy as _fz

    for group in ("net_qty", "mrp", "mfg_date", "unit_price"):
        hit = _fz.find_anchor(t, group)
        if hit and (hit[3] - hit[2]) >= 0.6 * len(re.sub(r"[^A-Za-z]", "", t)):
            return True
    return False


# "(1,699.00 (incl. of all taxes)": a ₹ read as "(" before the amount.
# A value may follow the column's colon: "PKD" | ":Oct.2024" (real pouch).
_VALUE_START = re.compile(r"^\s*[:\-–]?\s*(?:₹|rs\.?|inr|रु\.?|\d|\(\s*\d|[A-Za-z]{3}\s*[/\-.']?\s*\d)", re.I)

# Coded dates on inkjet batch lines: "MAR/26-NOV/26-D", "JAN 26 DEC 27".
_MONTH_TOKEN = re.compile(
    # The day may be glued on: "21AUG2026", "10MAY26" (real sticker and
    # inkjet codes), so no word boundary on the left - just no letter.
    r"(?<![A-Za-z])(?:\d{1,2}\s*[/\-.]?\s*)?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*"
    r"[/\-.'’]?\s*(\d{4}|\d{2})(?!\d)",
    re.I,
)
_MFG_CONTEXT = re.compile(
    r"\b(?:[mh]f[d0o]|mfg|pkd|packed|manufactured|date\s+of\s+pack|use\s*(?:by|before)|exp|"
    r"best\s+before|expiry|coding\s+area)", re.I)
# mm/yyyy, or dd/mm/yy(yy). Years are checked by _year.
# A decimal like "9.1g" or "Rs.10.00" must never read as a date, so a
# dot separator is only accepted with a 4-digit year or a full d.m.y.
_NUMERIC_DATE = re.compile(
    r"(?<![\d/.])(\d{1,2})(?:/|-|\.(?=\d{4}|\d{1,2}\.\d))(\d{4}|\d{2})"
    r"(?:[/\-.](\d{4}|\d{2}))?(?![\d/.])")
_TOLL_FREE = re.compile(r"(?<!\d)1800[\s\-]?\d{3}[\s\-]?\d{3,4}(?!\d)")
_EMAIL_LIKE = re.compile(r"[\w.\-]+\s?@\s?[\w\-]+\s?\.\s?[a-z]{2,}(?:\.[a-z]{2,})?", re.I)


def _year(tok: str):
    """A plausible manufacture/expiry year, or None."""
    try:
        y = int(tok)
    except (TypeError, ValueError):
        return None
    if y < 100:
        y += 2000
    return y if 2000 <= y <= 2045 else None


def _mean_conf(line: list[TextSpan]) -> float:
    return sum(s.confidence for s in line) / len(line) if line else 0.0


# =====================================================================
# VLM extraction
# =====================================================================

EXTRACTION_SCHEMA = {
    "manufacturer_details": "string or null - full name and address of manufacturer/packer/importer",
    "common_name": "string or null - generic commodity name, NOT the brand name",
    "net_quantity_value": "number or null",
    "net_quantity_unit": "string or null - one of g, kg, ml, l, N, U, mm, cm, m",
    "net_quantity_raw": "string or null - the full quantity line as printed",
    "manufacture_date": "string or null - as MM/YYYY",
    "mrp_value": "number or null",
    "mrp_raw": "string or null - the full price line as printed, including any tax qualifier",
    "consumer_care": "string or null - full consumer care block",
    "unit_sale_price": "number or null",
    "country_of_origin": "string or null",
}

VLM_EXTRACTION_PROMPT = f"""You are reading an Indian packaged-commodity label to check compliance with the Legal Metrology (Packaged Commodities) Rules, 2011.

Extract ONLY what is literally printed on the package. Return a single JSON object, no prose, no markdown fences, with exactly these keys:

{json.dumps(EXTRACTION_SCHEMA, indent=2)}

Critical rules:
- Use null for anything not visibly printed. NEVER infer, complete or guess a value. A null is a correct answer; an invented MRP is a false accusation against a manufacturer.
- For *_raw fields, reproduce the line exactly as printed, including the currency symbol and any wording such as "inclusive of all taxes". The exact wording is legally significant and is checked separately.
- Do not put the brand name in common_name. "Surf Excel" is a brand; "Detergent Powder" is the common name.
- If a character is illegible, use "?" within the raw string rather than omitting it.
"""


class SchemaVLMExtractor:
    """
    Constrained extraction for labels the regex layer cannot parse.

    The schema constraint is not decoration. Free-form VLM output on this
    task hallucinates well-formed prices and addresses that are not on
    the package, and those are exactly the errors that would put a wrong
    violation on a legal report. Constrain, validate, and treat every
    field as null until proven otherwise.
    """

    def __init__(self, client=None, model: str = ""):
        self.client = client
        self.model = model

    def available(self) -> bool:
        return self.client is not None

    def extract(self, image, spans=None, calibration=None) -> list[Declaration]:
        if not self.available():
            return []
        payload = self._call_model(image)
        return self._to_declarations(payload)

    def _call_model(self, image) -> dict:
        raise NotImplementedError("Wire your VLM provider here.")

    def _to_declarations(self, payload: dict) -> list[Declaration]:
        out: list[Declaration] = []

        def add(field_id, value, raw=None, unit=None, conf=0.8):
            present = value is not None and value != ""
            out.append(
                Declaration(
                    field_id=field_id,
                    raw_text=raw if raw is not None else (str(value) if present else ""),
                    value=value,
                    unit=unit,
                    extraction_confidence=conf if present else 0.0,
                    present=present,
                )
            )

        add("manufacturer_details", payload.get("manufacturer_details"))
        add("common_name", payload.get("common_name"))
        add("net_quantity", payload.get("net_quantity_value"),
            raw=payload.get("net_quantity_raw"), unit=payload.get("net_quantity_unit"))
        add("manufacture_date", payload.get("manufacture_date"))
        add("retail_sale_price", payload.get("mrp_value"), raw=payload.get("mrp_raw"))
        add("consumer_care", payload.get("consumer_care"))
        add("unit_sale_price", payload.get("unit_sale_price"))
        add("country_of_origin", payload.get("country_of_origin"))
        return out


# =====================================================================
# Hybrid
# =====================================================================

class HybridExtractor:
    """
    Regex first, VLM only for fields regex could not find.

    Keeps cost and latency low, keeps most results explainable, and
    isolates the black box to the cases that actually need it. Track how
    often the VLM path fires - if it is firing on 80% of packages your
    regex layer needs work, not a bigger model.
    """

    def __init__(self, regex: Optional[RegexExtractor] = None, vlm=None):
        self.regex = regex or RegexExtractor()
        self.vlm = vlm
        self.vlm_invocations = 0

    def extract(self, spans, image=None, calibration=None) -> list[Declaration]:
        decls = self.regex.extract(spans, image=image, calibration=calibration)
        missing = [d.field_id for d in decls if not d.present]

        if missing and self.vlm is not None and self.vlm.available() and image is not None:
            self.vlm_invocations += 1
            try:
                vlm_decls = {d.field_id: d for d in self.vlm.extract(image)}
                merged = []
                for d in decls:
                    if not d.present and d.field_id in vlm_decls and vlm_decls[d.field_id].present:
                        v = vlm_decls[d.field_id]
                        v.extraction_confidence *= 0.9  # discount vs. regex
                        merged.append(v)
                    else:
                        merged.append(d)
                decls = merged
            except Exception as exc:
                print(f"[hybrid] VLM extraction failed: {exc}")

        return decls


# ---------------------------------------------------------------------
# Negative context for unlabelled fallback patterns
# ---------------------------------------------------------------------
# Phrases that mark a line as belonging to a DIFFERENT declaration.
# Consulted only in Pass 2, where patterns are deliberately loose.
_NEGATIVE_CONTEXT: dict[str, tuple[str, ...]] = {
    "retail_sale_price": ("unit sale price", "unit price", "per unit", "per 100",
                          # an offer is not the MRP ("Special Price ₹39", "₹20 OFF")
                          "special price", "our price", "offer", "deal price", "discount",
                          "cashback", "you pay", "you save", "sale price:", "selling price"),
    "unit_sale_price": ("maximum retail price", "m.r.p", "mrp"),
    "net_quantity": (
        "serving size", "per serve", "per 100 g", "per 100 ml",
        # Date lines carry numbers and previously won the fallback race.
        "manufactured", "mfg", "packed on", "expiry", "best before",
        "use by", "batch", "model",
    ),
}


def _disqualified(field_id: str, line_text: str) -> bool:
    low = (line_text or "").lower()
    if field_id == "retail_sale_price" and re.search(r"\boff\b|\bsave\b", low):
        return True
    return any(p in low for p in _NEGATIVE_CONTEXT.get(field_id, ()))