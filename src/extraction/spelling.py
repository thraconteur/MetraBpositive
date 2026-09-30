"""
Spelling repair for address / company text read by OCR.

PaddleOCR gets most words right on a pack, but small print loses single
letters in a way a reader notices at once: "DEIHI" for DELHI (an L read as
I), "Gurugrom", "Industricl", "Privote Limited", "Pvt. Lid". An inspector
reading the report sees a sloppy tool; a reviewer checking the address
against a licence database gets no match.

This repairs ONLY words that are one or two characters away from a word in
a fixed vocabulary of address and company words (Indian states, major
cities and industrial towns, and the words addresses are made of). It is
applied to the extracted manufacturer / consumer-care VALUE shown in the
report, never to the raw OCR lines, which stay exactly as the engine read
them. Every repair is listed in the declaration's notes.

It does not touch numbers, PIN codes, e-mail addresses or web addresses,
and never "repairs" a word that is already in the vocabulary.
"""

from __future__ import annotations

import difflib
import re

# Words that make up Indian postal addresses and company names. Lower case.
_VOCAB = """
private limited pvt ltd llp company corporation industries industry products
foods beverages chemicals cosmetics wellness remedies marketing marketed
manufactured manufacturer packed packer imported importer registered office
plot road street lane marg nagar colony sector phase block floor building
house tower estate industrial area park complex centre center village post
district near opposite behind main cross unit survey gali bazar market
highway chowk mandal taluk tehsil ward extension extn
india delhi new noida greater gurugram gurgaon haryana faridabad ghaziabad
mumbai thane pune nagpur nashik aurangabad maharashtra goa
kolkata howrah bengal west bengaluru bangalore karnataka mysuru mysore
chennai madras tamil nadu tamilnadu salem coimbatore madurai tiruchirappalli
hyderabad secunderabad telangana andhra pradesh vijayawada visakhapatnam
ahmedabad gandhinagar vadodara surat rajkot bhavnagar gujarat anand
jaipur udaipur jodhpur rajasthan lucknow kanpur agra uttar uttarakhand
haridwar dehradun roorkee ludhiana amritsar mohali moga punjab chandigarh
himachal baddi solan shimla jammu kashmir srinagar bihar patna jharkhand
ranchi odisha bhubaneswar assam guwahati sikkim gangtok meghalaya shillong
kerala kochi cochin thiruvananthapuram madhya bhopal indore chhattisgarh
raipur manesar sonipat bahadurgarh kundli sahibabad silvassa daman
consumer customer care helpline complaints queries feedback executive
manager cell toll free email website address contact
marketer marketers packers importers suppliers supplier distributor
distributed distributors brand owner owned licence license registered
""" 
VOCAB = frozenset(_VOCAB.split())

_TOKEN = re.compile(r"[A-Za-z]{4,}")
# Letters OCR confuses in small caps: I/l/1, O/0, rn/m ...
_NEVER = re.compile(r"@|www\.|https?:|\.com|\.in\b|\.coop", re.I)


def _match_case(src: str, word: str) -> str:
    if src.isupper():
        return word.upper()
    if src[:1].isupper():
        return word.capitalize()
    return word


def _best(word: str):
    low = word.lower()
    if low in VOCAB:
        return None
    # A real word built on a vocabulary word ("marketer", "packers",
    # "marketing") is left alone - only damaged words are repaired.
    for suf in ("s", "es", "er", "ers", "ed", "ing"):
        if low.endswith(suf) and low[: -len(suf)] in VOCAB:
            return None
        # ...and the other way round: "PRODUCT" is a word, not a damaged
        # "products" (real Rajam jar, "PRODUCT MANAGER").
        if low + suf in VOCAB:
            return None
    # Same first letter, length within 1 - "DEIHI"->delhi, "Gurugrom"->gurugram.
    cands = [v for v in VOCAB if v[0] == low[0] and abs(len(v) - len(low)) <= 1]
    best, score = None, 0.0
    for v in cands:
        r = difflib.SequenceMatcher(None, low, v).ratio()
        if r > score:
            best, score = v, r
    # 0.8 = one wrong letter in a 5-letter word; two in a 10-letter word.
    need = 0.8 if len(low) < 7 else 0.85
    if best and score >= need:
        return best
    return None


def repair(text: str) -> tuple[str, list[str]]:
    """Return (repaired text, ["Gurugrom -> Gurugram", ...])."""
    if not text:
        return text, []
    fixes: list[str] = []
    out = []
    # "Pvt. Lid" / "Pvt Ld": the one short word worth repairing.
    def _ltd(m):
        fixes.append(f"{m.group(2)} -> {_match_case(m.group(2), 'ltd')}")
        return m.group(1) + _match_case(m.group(2), "ltd")
    text = re.sub(r"(\b(?:pvt|private)\.?\s*)(lid|ld|lt|itd)\b", _ltd, text, flags=re.I)
    for part in re.split(r"(\s+|,)", text):
        if not part or part.isspace() or part == "," or _NEVER.search(part):
            out.append(part)
            continue

        def sub(m):
            w = m.group(0)
            b = _best(w)
            if not b:
                return w
            new = _match_case(w, b)
            if new != w:
                fixes.append(f"{w} -> {new}")
            return new

        out.append(_TOKEN.sub(sub, part))
    return "".join(out), fixes
