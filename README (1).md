# METRA — Legal Metrology compliance checker for packaged goods

**Smart India Hackathon 2026 · Problem statement SIH26034 · Team B+**

METRA checks a photo of a packaged product against the **Legal Metrology
(Packaged Commodities) Rules, 2011**. An inspector photographs the pack
(one side, or every side of one pack). METRA then:

1. reads the label with PaddleOCR (English and Hindi);
2. extracts the mandatory declarations;
3. checks each one against the Rules, citing the rule;
4. produces a report (PDF / Word / web page) with the photo and an evidence
   crop for every finding.

A supervisor reviews anything an inspector flags.

```
photo → orientation → quality gate → calibration card (optional) → PaddleOCR (EN + HI)
      → declarations → exemptions → rules engine → findings + evidence → report
```

## What it checks

| Rule | Check |
|---|---|
| 6(1)(a) | Name and complete address of manufacturer / packer / importer |
| 6(1)(b) | Common or generic name |
| 6(1)(c), 11, 12(6) | Net quantity: present, standard (SI) unit, no "approx." / "min." / "±" qualifiers |
| 6(1)(d) | Month and year of manufacture / packing / import |
| 6(1)(da) | Best-before / use-by date (expired stock is raised as a food-safety alert, not an LMPC violation) |
| 6(1)(e) | MRP "inclusive of all taxes", currency and rounding; a second MRP on the pack |
| 6(1)(f) | Consumer care: phone or email |
| 6(11), 2(bb) | Unit sale price, and that it matches MRP ÷ quantity |
| 6(3) | Price sticker over the printed MRP |
| 6(1)(aa) | Country of origin on imported packs |
| 7(2), Table-I | Minimum numeral height in mm by display-panel area (needs the calibration card) |
| 7(3), 8 | Character shape; clear space around the net quantity |
| 18(2A) | Different MRPs on identical products across scans |

Every threshold lives in `rules/lmpc_2011.yaml` with its citation and a
`verified` flag. A rule that has not been verified against the gazette text
**cannot** produce a violation; it reports `UNVERIFIED_RULE` instead.

Each finding has one of four outcomes:

* **VIOLATION**
* **COMPLIANT**
* **INDETERMINATE** ("confirm by eye")
* **NOT APPLICABLE**

When a photo cannot prove something, METRA says so instead of guessing.
Examples: a declaration on another side of the pack, text under glare, or a
size check without the calibration card.

## Features

* **Bilingual reading.** English and Devanagari labels. Hindi is kept as
  printed in the reports, not translated.
* **Any orientation.** Sideways, upside-down and tilted photos are turned
  upright before reading.
* **Whole-pack scans.** Several photos of one pack are decided together.
  Each declaration records which photo it came from. Photos of different
  products are refused as one pack.
* **Evidence for every finding.** The photo and a crop of the text are in
  the app and in the PDF / Word / HTML report.
* **Millimetre checks.** A printed 25 mm ArUco card placed on the label
  gives a real-world scale for numeral height, character shape and clear
  space. The uncertainty is carried through, so a verdict is given only
  when it holds at both ends of the error range.
* **Guided capture.** Poor photos get specific retake advice, for example
  "text too small, move closer" or "the MRP is on the base, photograph it
  too".
* **Inspector workflow.** Scan history, dashboard, flags, supervisor review
  with an audit trail, and roles that cannot be raised from the app.
* **Safe uploads.** Size, pixel and pack limits. Truncated and HEIC files
  are refused with a clear message.

## Results

Tested on 84 real phone photos of packs. The answer key for each photo was
written by eye from the photo.

| | |
|---|---|
| Declarations read correctly | 261 of 293 (89%) |
| Wrong values | 0 |
| False violations | 0 |
| Violations in the answer key caught | all |
| First-read accuracy on packs never seen during development | 31 of 40 (78%), no wrong values |

The photos cover low- and high-resolution images, English and Hindi labels,
and rotated photos. The declarations still missed are not legible in the
photo, for example smudged stickers or dot-matrix print on a pack's edge.
For those, METRA asks for confirmation instead of reporting a value.

Speed: about 25–30 s per phone photo on a 2-core CPU, faster with an NVIDIA
GPU.

Details: [`docs/results_log.md`](docs/results_log.md). Re-score a set with
`python scripts/real_eval.py --dir tests/fixtures/real_hires`.

## Run it

Python 3.11–3.13.

```bash
pip install -r requirements.txt          # PaddlePaddle 3.2.2 + PaddleOCR 3.7.0, OpenCV, FastAPI ...
python scripts/get_hindi_model.py        # once: the Hindi (Devanagari) reader
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. From a phone on the same Wi-Fi, use
`http://<laptop IP>:8000`.

Demo sign-in tokens are `demo-inspector`, `demo-supervisor`, `demo-guest`
and `demo-admin`; enter one under Settings in the app. You can pick one
photo, or several photos of one pack at once. The first scan also loads the
models.

Without the web app:

```bash
python scripts/scan_photo.py photo.jpg                                     # one photo -> data/reports/photo.html
python scripts/scan_photo.py top.jpg side.jpg base.jpg --package my_pack   # every side of one pack
python scripts/demo.py synthetic                                           # rendered labels, no OCR model needed
pytest -q                                                                  # test suite, no OCR model needed
```

Optional:

* **Calibration card** for the millimetre checks:
  1. Run `python scripts/make_card_pdf.py`.
  2. Print at 100% ("actual size").
  3. Check the 50 mm line with a ruler.
  4. Lay the card flat **on** the label face.
* **Storage cleanup:** `python scripts/cleanup_data.py --days 90 --yes`, or
  set `SIH_RETENTION_DAYS=90` for the server.
* **Connecting a mobile app to the API:** see
  [`docs/app_integration.md`](docs/app_integration.md).
* **OCR settings and troubleshooting:** see
  [`docs/paddleocr_tuning_guide.md`](docs/paddleocr_tuning_guide.md).

## Design choices

* **Absence is not a violation from one photo.** Declarations are often
  split across sides, so only a pack scan (all sides together) decides that
  something is missing.
* **No legal number in the code.** Thresholds are in YAML with citations.
  Unverified rules cannot assert violations.
* **Measure only when the measurement can be trusted.** A millimetre
  verdict needs all of these; otherwise the check says "confirm by eye":
  * the card flat on the label and photographed square-on;
  * enough pixels on the digits;
  * a flat surface;
  * a panel area that falls clearly in one Table-I band.
* **Never guess.** A value the OCR is unsure of is marked for
  confirmation, and a finding on it is never a violation.

## Limits

* Text is read in English and Hindi (Devanagari). Other Indian scripts are
  not read.
* The millimetre checks are validated on generated labels with an exact
  scale (numeral height within 0.08 mm on average). Field validation with
  the printed card is the next step.
* Curved packs (cans, bottles) get no millimetre verdict.
* Sign-in uses static demo tokens to show the role model; it is not
  production authentication.

## Repository layout

```
rules/lmpc_2011.yaml         thresholds, citations, verified flags   ← start here
rules/exemptions.yaml        exemptions, resolved before evaluation
rules/lexicon.yaml           label spellings (incl. Hindi) for fuzzy matching
src/core/pipeline.py         stages, orientation, calibration, classification
src/core/rules_engine.py     the rules
src/core/package.py          several photos of one pack -> one decision
src/extraction/fields.py     text -> declarations
src/vision/ocr/paddle.py     PaddleOCR: re-reads, Hindi, orientation, cache, replay
src/vision/                  calibration card, glyph size, panel, stickers, barcode
src/report/builder.py        PDF / Word / HTML reports with evidence
src/api/main.py              FastAPI: scan, pack scan, history, roles, reports
templates/index.html         the METRA inspector web app (works offline, no CDN)
tests/                       235 tests; tests/fixtures holds the real photos + answer keys
docs/results_log.md          evaluation method and results
docs/app_integration.md      API for a mobile app
docs/paddleocr_tuning_guide.md   OCR pipeline, settings and troubleshooting
```

Fonts: Noto Sans Devanagari and Inter (SIL Open Font License).
