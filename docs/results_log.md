# Evaluation

How METRA is measured on real photos, and the results.

## Method

* **Photos.** Real photos of real packs, mostly phone photos, in
  `tests/fixtures/`. No synthetic images are used in these scores.
* **Answer key.** For each photo, `ground_truth.yaml` lists:
  * what is legibly printed (MRP, net quantity, dates, address, consumer
    care, ...);
  * which violations a careful inspector would raise.

  The key is written by eye from the photo, not from the OCR output.
  "Possible violations" (judgement calls, such as "Incl GST" instead of
  "inclusive of all taxes") count neither way.
* **Scoring.** `python scripts/real_eval.py --dir tests/fixtures/<set>`
  re-scores every photo in a set from its stored PaddleOCR read
  (`*.paddle.json`), so no model is needed. `scripts/dump_real.py` makes
  those reads with PaddleOCR.

Four numbers are reported:

| Metric | Meaning |
|---|---|
| Read correctly | A declaration in the answer key extracted with the right value |
| Wrong value | A declaration extracted with a value that is not on the pack |
| False violation | A violation raised that the answer key does not contain |
| Violations caught | Answer-key violations that METRA raised |

For an inspector, a wrong value or a false violation is worse than a miss:
a miss is shown as "confirm by eye", while a wrong value is an error in the
report.

## Results

| Test set | Photos | Read correctly | Wrong values | False violations | Violations caught |
|---|---|---|---|---|---|
| `real` (compressed, 580 px wide) | 38 | 103 / 125 | 0 | 0 | 3 / 3 |
| `real_hires` (camera resolution) | 21 | 70 / 76 | 0 | 0 | 1 / 1 |
| `set3` (product images and phone photos) | 13 | 36 / 40 | 0 | 0 | 1 / 1 |
| `rotated` (90°, 180°, 270°, 15°) | 12 | 52 / 52 | 0 | 0 | 4 / 4 |
| **All** | **84** | **261 / 293 (89%)** | **0** | **0** | **all** |

**Packs never seen during development.** The answer key for `set3` was
written before METRA was run on it. On that first read it got 31 of 40
declarations correct (78%) with no wrong values. The table shows the
current engine.

The test suite gates these numbers (`tests/test_real_photos.py`), so a
change that reads fewer declarations, or introduces a wrong value or a false
violation, fails the build.

**What is still missed.** Every declaration still missed on the
camera-resolution photos is not legible in the photo:

* vertical dot-matrix codes;
* dates printed along a pack's edge;
* smudged price stickers;
* print on a crease.

These are reported as "confirm by eye", never as a value.

**Hindi.** The Devanagari reader replaces a line only where the English
model read Hindi print as junk. In testing, it never changed an English
line.

## Speed

On a 2-core cloud CPU, with PaddlePaddle 3.2.2 and oneDNN:

| | Median per photo |
|---|---|
| Whole scan, compressed photos | ~25 s |
| Whole scan, camera-resolution phone photos | ~27 s |

These choices keep scans fast without changing any reading:

* **Fast orientation check.** Text detection plus PaddleOCR's text-line
  classifier decide "upright and level" in about 1.5 s. A full reading test
  runs only when that is unclear.
* **Batched recognition.** Lines sorted by shape and recognised 6 at a
  time.
* **Selective Hindi re-read.** Skipped for lines already read as confident
  plain Latin text.
* **Coarse-to-fine calibration search.** The card is searched for at half
  size first.
* **Detection at ≤ 2880 px, recognition on full-resolution crops.**

An NVIDIA GPU (`paddlepaddle-gpu`) is several times faster. A laptop in
power-saving mode is slower.

## Synthetic labels (millimetre checks)

The millimetre checks need exact ground truth. On 60 rendered labels with
known print sizes:

* violation precision 1.00, recall 0.79;
* numeral height within 0.081 mm on average.

Recall is below 1 by design: a declaration missing from one photo is
reported as "confirm", because one side cannot prove absence.

## Robustness checks

Each of these has a regression test in `tests/`:

* **Rotation.** Sideways, upside-down and tilted photos are turned upright
  before reading. The `rotated` set scores the same as upright photos.
* **Label wording.**
  * Manufacturer, packer and importer labels in their common forms.
  * Import dates and country of origin.
  * Multipacks, imperial units and quantity qualifiers placed before or
    after the number.
  * Offer prices, which are not treated as the MRP.
  * Negated tax wording ("excl. of all taxes").
  * A second MRP on the same pack.
  * Column-style label/value tables.
* **Images.**
  * EXIF rotation, transparent PNGs, truncated JPEGs and motion photos.
  * HEIC is refused with a clear message.
  * Very large images are refused before decoding.
* **Packs.** Photos of two different products (different barcode, quantity
  or MRP) are not merged into one pack.
* **Measurement.** No millimetre verdict in any of these cases:
  * the card is seen at an angle;
  * the digits are too small;
  * the label is curved;
  * the panel area is too close to a Table-I band edge.

  A card printed slightly under-size is covered by the uncertainty term.
