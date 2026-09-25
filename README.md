# SIH26034 — LMPC Compliance Checker (reference scaffold)

A working reference implementation for **SIH26034: Software System to
Check Compliance of Packaged Commodities under the Legal Metrology
(Packaged Commodities) Rules, 2011**.

This is scaffolding to think against, not a submission. It is here so
you can see a whole system working end to end, decide which parts are
right, and rip out the parts that are not. **Read the "Honest scope"
section before you copy anything into a pitch.**

```bash
pip install -r requirements.txt       # includes paddlepaddle + paddleocr
python scripts/scan_photo.py photo.jpg          # real photo -> data/reports/photo.html
python scripts/demo.py synthetic      # end-to-end on rendered labels, no model download
python scripts/scan_real.py photo.jpg           # same, with Gemini for missed fields if GEMINI_API_KEY is set
python scripts/real_eval.py           # scores on REAL photos from saved Paddle output
pytest -q                             # 213 tests, no model needed
```

Connecting the mobile app: `docs/app_integration.md`.

**OCR is PaddleOCR and nothing else.** Tesseract has been removed. If
Paddle is not installed, scanning stops with an error instead of quietly
reading with a weaker engine. The tests use `SyntheticOCR` (the exact
text the label renderer drew, at Paddle's line granularity), so they
check everything downstream of OCR without loading a model. Tuning
Paddle on real photos: `docs/paddleocr_tuning_guide.md`.

---

## What actually works right now

| Capability | State |
|---|---|
| Rules 6, 7(3), 8, 11(6) — presence, wording, units, banned phrases, clear space, glyph ratio | Working, verified against gazette structure |
| Rule 7(2) minimum numeral height in mm | Live. Table-I verified against G.S.R. 629(E) (2017) |
| Rule 6(3) sticker over MRP, incl. the lawful-downward-revision proviso | Working |
| Rule 6(7) 'GM' at top of PDP | Working (GM status is inspector-supplied, not inferred) |
| Rule 18(2A) different MRPs on identical products | Working, across the repository |
| PDP panel area | **1.1% mean error** (was 61% under-reported) |
| Fiducial calibration + plane rectification | Working, scale error < 0.2% |
| Cap-height measurement | **MAE 0.081 mm**, 98% within 0.3 mm (60 renders, noise on) |
| Exemption resolution before evaluation | Working — 14 exemptions, all reachable and effective (see below) |
| OCR: PaddleOCR (PP-OCRv6 en + Devanagari re-read), cache, dumps, replay | Backend logic tested against a fake Paddle module; **accuracy on real photos is measured only by running it on real photos** |
| Synthetic data with exact mm ground truth | Working |
| Real photos (PaddleOCR output replayed) | 3 real packs: **11 of 13 legible declarations read correctly, 0 read wrongly, 0 false violations** - but the Sept 2026 fixes were made looking at these same photos, so this is a floor, not a held-out accuracy. `scripts/real_eval.py` |
| Evaluation harness | Working — **P=1.00, R=0.78, F1=0.88** on 60 synthetic images **with perfect (synthetic) OCR**; run `evaluate.py --ocr paddle` for the real-engine number |
| API: scan, repository, RBAC, dashboard, reports | Working |

Measured on synthetic Latin-script labels. **These are not numbers you
may quote as real-world accuracy.** See "Honest scope".

---

## The one design decision that matters

**No legal threshold is hardcoded anywhere in the Python.** Everything
lives in `rules/lmpc_2011.yaml`, and every rule carries a `verified`
flag:

```yaml
- id: numeral_height
  citation: "Rule 7(2), Table-I and Table-II"
  verified: false          # <-- thresholds are placeholders
```

When `verified: false`, the engine **physically cannot emit a
violation** from that rule. It returns `UNVERIFIED_RULE` instead. A test
enforces this:

```python
def test_unverified_rules_cannot_emit_violations(...):
    for f in res.findings:
        if not f.rule_verified:
            assert f.outcome is not Outcome.VIOLATION
```

Why bother: a confidently wrong legal threshold copied off a blog is
worse than an unimplemented check, because it produces an
authoritative-looking false accusation with a rule citation attached.
I did not transcribe Table-I, so the code refuses to pretend it did.

A consequence you should keep, not fix: a fully compliant label returns
`is_compliant = None`, not `True`. Four outcomes, not two —
`COMPLIANT / VIOLATION / INDETERMINATE / UNVERIFIED_RULE`. "I cannot
tell" is a legitimate and useful answer for an enforcement tool.

---

## Your first task

Open the current consolidated gazette text and fill in this table:

```yaml
table_I:
  bands:
    - max_g_or_ml: 200
      min_height_mm_printed: null      # TODO
      min_height_mm_embossed: null     # TODO
```

Then set `verified: true` and put the notification number in `source`.
`python scripts/demo.py` prints your coverage — currently **18 verified,
7 unverified, 72%**. Walking into the finale able to say "23 of 25 rules
verified against the gazette, and the tool refuses to judge the other
two" is a much stronger claim than silence.

---

## Architecture

```
image → quality gate → calibrate → rectify → preprocess
      → OCR → extract → classify → EXEMPT → rules engine → findings
```

Two properties worth preserving:

**Classify and exempt _before_ evaluating, never filter after.** A 5 g
sachet flagged for a missing unit sale price looks broken to the first
officer who tests it, and you do not recover from that in a live demo.

**Every stage is injectable.** OCR backend, extractor and classifier are
constructor arguments. That is what makes the ablation table possible:
hold everything constant, swap one stage, re-measure.

```python
pipeline = CompliancePipeline(ocr=MyOCR(), extractor=MyExtractor())
```

---

## Four bugs found by running it

Each was silent — the pipeline produced confident, well-formatted, wrong
output. Each now has a regression test. These are the traps, not the
code, and they are the most useful thing in this repo.

**1. Sorting spans by `(y, x)` scrambles reading order.** Words on one
printed line differ by a pixel or two; ascenders start higher. Sorting
on raw `y` interleaves adjacent lines. OCR printed perfectly, extraction
still returned fields, values were quietly wrong. Fixed by bucketing
spans into rows by vertical overlap (`vision/ocr/base.py`).

**2. Deglare destroyed white labels.** Masking every bright,
low-saturation pixel selects the *entire* white carton; inpainting then
smears the print away and OCR returns garbage. Brightness alone cannot
identify glare. Two signals fix it: glare is **local** (bail if the mask
covers >25% of frame) and **textureless** (no Canny edges inside a real
highlight; printed paper is full of them).

**3. Unanchored multi-line windows caused a false violation.** The MRP
pattern matched inside a window opened at the product-name line and cut
off the "(inclusive of all taxes)" qualifier two lines below — flagging
a **fully compliant package**. Fixed by requiring the anchor to match on
the line before widening.

**4. Rule 8 only looked at declarations.** "Other printed matter" is
mostly text that never becomes a declaration — marketing copy,
ingredients, a barcode caption. Judging clear space from declarations
alone reported crowded labels as compliant. `ScanResult.spans` now
carries every OCR span.

---

## Honest scope — read before pitching

**The metrics are synthetic.** Clean Latin-script renders with a
cooperative fiducial. Real packaging is foil, curved, bilingual and
glare-blown. Expect the mm MAE to degrade badly and OCR CER to rise.
Re-measure on real photographs before quoting any number.

**Devanagari is wired but unproven on real labels.** `--secondary-lang hi`
re-reads every line with Paddle's Devanagari recogniser (a weaker mobile
model than the English one) and keeps it only where the line is actually
Devanagari. Extraction understands Hindi labels (अधिकतम खुदरा मूल्य,
शुद्ध मात्रा, सभी करों सहित ...) and Devanagari digits. None of this has
been measured on a real bilingual pack yet. This remains the largest
single unknown.

**Auth is a demo of the role model, not security.** Static tokens in a
dict. The role separation is the point — an inspector proposes, only a
supervisor overrides, every action is logged with who and when — but do
not claim it is secure.

**Untested paths:** PaddleOCR accuracy on real packaging (the backend's
behaviour is tested; what the model reads is not), VLM fallback,
WeasyPrint PDF (falls back to HTML), cylindrical PDP area,
sticker-overlay detection on real stickers.

**Precision 1.00 is with perfect OCR.** On synthetic labels read by
`SyntheticOCR` there are no false violations; the recall gap is mostly
by design (a single photo cannot prove a declaration is absent, so those
come back INDETERMINATE) plus two stickers the visual detector missed.
Real OCR adds errors this number does not contain. Things OCR commonly
loses - the ₹ sign, a qualifier printed on another line, a unit read as
a digit - are reported as INDETERMINATE or carry a note on the report,
never as a violation.

**Do not quote the old P=1.00 figure.** It predated the sticker and GM
rules and, more importantly, it was partly an artefact: the generator
was co-injecting violations that perturb each other, so correct
detections scored as false positives once those rules existed.

---

## What to build next

1. Transcribe Table-I / Table-II. Everything else is downstream.
2. Devanagari: add a font, extend the regexes, re-measure.
3. Shoot 400 real SKUs with the fiducial in frame. Synthetic data
   develops the measurement stage; it does not validate it.
4. Sticker-overlay detection (`sticker_alteration`) — the pasted-MRP
   case is the most demo-friendly violation there is.
5. E-commerce listing mode — a second input path, cheap volume.

---

## Layout

```
rules/lmpc_2011.yaml       legal thresholds + verified flags   ← start here
rules/exemptions.yaml      carve-outs, resolved before evaluation
src/core/schema.py         shared types; four-outcome Finding
src/core/rules_engine.py   config-driven evaluator
src/core/pipeline.py       stage orchestration
src/vision/calibration.py  ArUco → px/mm, plane rectification
src/vision/glyph.py        cap-height via connected components
src/vision/preprocess.py   quality gate, deglare, CLAHE, unsharp
src/vision/ocr/paddle.py   PaddleOCR backend, cache, dumps, ReplayOCR
src/vision/ocr/backends.py SyntheticOCR (tests), StubOCR, VLM hook, factory
src/vision/ink.py          tight ink boxes (Paddle pads its boxes)
src/vision/ocr/vlm_gemini.py  Gemini second reader (marked, OCR-checked, never decides alone)
src/extraction/fuzzy.py    fuzzy label anchors from rules/lexicon.yaml
src/extraction/vlm_recover.py  turns Gemini reads into declarations with provenance
tests/fixtures/real/       real photos + real Paddle output + ground truth
src/extraction/fields.py   regex + VLM-schema extraction
src/synth/generator.py     labels with exact mm ground truth
src/evaluation/harness.py  CER, F1, mm MAE, ablation
src/report/builder.py      HTML/PDF with citations + evidence crops
src/api/main.py            FastAPI: scan, repo, RBAC, dashboard
```

Scripts: `scan_photo.py` (real photos), `paddle_probe.py` (compare
OCR settings), `diagnose.py`, `demo.py`, `make_dataset.py`,
`make_marker.py`, `evaluate.py`.

**Print the marker at 100% scale.** Any "fit to page" silently rescales
it and every measurement inherits the error. Verify with a ruler.
