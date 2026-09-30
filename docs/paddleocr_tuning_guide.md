# PaddleOCR in METRA: pipeline, settings, troubleshooting

PaddleOCR is the only OCR engine in METRA. If it is not installed, a scan
stops with a clear error.

---

## 1. Install and check

```bash
pip install -r requirements.txt          # paddlepaddle 3.2.2 + paddleocr 3.7.0
python scripts/get_hindi_model.py        # once: the Devanagari recogniser
python -c "import paddleocr; print(paddleocr.__version__)"
```

The first scan downloads the models (~200 MB) into `~/.paddlex`. After
that, METRA works offline.

**CPU speed.** PaddlePaddle 3.2.2 is pinned because its fast CPU mode
(oneDNN) works with PaddleOCR 3.7.0. On 3.3.0 and 3.3.1 it crashes with
`ConvertPirAttribute2RuntimeAttribute not support` (PaddlePaddle #77340,
PaddleOCR #18162). The first line printed by `scan_photo.py` shows
`fast CPU mode ON/OFF`.

**GPU.**
1. Replace `paddlepaddle` with the `paddlepaddle-gpu` build for your CUDA
   version.
2. Pass `--device gpu:0`.

---

## 2. Scan from the command line

```bash
python scripts/scan_photo.py label.jpg
python scripts/scan_photo.py photos/                          # a whole folder
python scripts/scan_photo.py side.jpg base.jpg --package can  # every side of ONE pack
python scripts/scan_photo.py label.jpg --coverage-complete    # the photo shows EVERY panel
python scripts/scan_photo.py label.jpg --fmt pdf              # html (default), pdf or json
```

Each photo produces:

| File | What it is |
|---|---|
| `data/reports/<name>.html` | The inspector report. It ends with **What was read** (each declaration, with notes) and **Raw OCR lines** (every line PaddleOCR read, with its confidence). |
| `data/ocr_dumps/<name>.paddle.json` | Exactly what PaddleOCR read: text, score, polygon, settings, timings. |
| `data/.ocr_cache/` | Cached OCR. Re-running the same photo skips the model and takes about a second. `--no-cache` forces a fresh read. |

The console prints the model, the OCR time, every extracted field and every
violation.

---

## 3. What METRA adds on top of a plain `PaddleOCR().predict()`

All of this is in `src/vision/ocr/paddle.py` and is on by default.

1. **Boxes in photo coordinates.** Document unwarping and orientation
   classification are off. PaddleOCR 3.x enables them by default and then
   returns boxes in the warped image's coordinates. Glyph heights,
   clear-space measurements and evidence crops need boxes on the real
   photo.
2. **Orientation first.** Sideways, upside-down and tilted photos are
   turned upright before reading:
   * A fast check (detection plus the text-line classifier) passes
     clearly upright photos in about 1.5 s.
   * Anything unclear is decided by reading sample lines each way, in
     English and Hindi.
3. **Scale for detection and recognition.**
   * Detection runs with a long side of at most 2880 px, the scale where
     PP-OCR finds lines best.
   * Recognition runs on full-resolution crops.
   * Small photos (short side under 1200 px) are upscaled 2× for reading
     only.
   * Boxes are always mapped back to the original photo.
4. **Second reads of weak lines.**
   * Lines scoring below 0.80 are re-cropped with extra margin (so a
     clipped `g` does not become `9`), upscaled, contrast-normalised and
     read again.
   * Still-weak curved lines are re-detected with polygon boxes and
     unwarped.
   * A new read replaces the old one only if it scores clearly higher.
5. **Hindi.**
   * The English model (PP-OCRv6 medium) reads first.
   * Lines it did not read as confident plain Latin text are re-read with
     the Devanagari model.
   * The Hindi read is kept only where the line really is Devanagari.
   * `--no-hindi` turns this off.
6. **Hard print.**
   * Vertical inkjet codes are cropped as one band and read rotated.
   * Sparse dot-matrix print is also read softened.
   * A label that is small in the frame is re-read magnified.
   * Address and consumer-care blocks are re-detected locally to recover
     missed lines.
   * A sideways read runs when the pack points to an edge or coding area.
7. **Batched recognition.** Re-reads are sorted by shape and recognised 6
   at a time, as PaddleOCR's own pipeline does.
8. **Noise filtering.**
   * CJK glyphs (smudges read as Chinese characters) are dropped.
   * A rupee sign misread as such a glyph on a price line is restored as
     ₹.
9. **Measurement on ink, not boxes.** Clear space (Rule 8) is measured
   between the actual ink. Glyph height uses only the digits and capitals
   matched to the recognised text, so brackets and descenders do not count.
10. **Models load once per process**, so a folder of photos pays the load
    cost once.

---

## 4. When a report looks wrong

Open the HTML report and expand **Raw OCR lines**.

- **The line is misread there** (e.g. `Net wt. 69` where the pack says
  `6 g`): this is an OCR problem. Try the settings in section 5, or retake
  the photo.
- **The line is read correctly but the finding is wrong**: this is an
  extraction or rules problem. Replay the saved read without the model:

  ```bash
  python scripts/scan_photo.py label.jpg --replay data/ocr_dumps/label.paddle.json
  ```

  This re-runs everything after OCR in under a second, so a fix can be
  checked against the exact photo.

Some findings are INDETERMINATE by design, because OCR can lose these even
when they are printed:

- **No ₹ read in the price.** A missing currency sign is never reported as
  a violation. Check the pack.
- **"Inclusive of all taxes" read, but not on the price line.** Check that
  it belongs to the MRP.
- **A unit inferred from a digit** (`69` read as `6 g`). The declaration
  carries a note saying so.

---

## 5. Settings

Compare several settings on the same photos:

```bash
python scripts/paddle_probe.py label.jpg
python scripts/paddle_probe.py photos/ --out data/probe.json
```

This prints one column per setting with each field's extracted value.

| Symptom in Raw OCR lines | Try |
|---|---|
| Small print missing entirely | `--det-side 1920` (then 2560) |
| Faint or low-contrast print missing | `--det-thresh 0.2 --box-thresh 0.45` |
| Descenders cut off: `6 g`→`69` | `--unclip 2.0` |
| Two lines merged into one | `--unclip 1.3` |
| Print along the pack's edge | `--sideways` |
| Compare model generations | `--ocr-version PP-OCRv5` |
| Too slow on CPU | Check that `fast CPU mode ON` is printed; use `--device gpu:0` if you have a GPU |

`SIH_OCR_DET_MAX=1920` makes detection faster (20–36% less OCR time). The
coarser boxes can shift the millimetre measurements, so it is not the
default.

No setting fixes glare over the text, a label curved away from the camera,
or text under about 12 px tall in the photo. Retake those closer, flatter,
or at an angle that moves the glare.

---

## 6. Measuring without the calibration card

If there is no card in the photo, measure the panel with a ruler and pass
its size:

```bash
python scripts/scan_photo.py label.jpg --panel-mm 95x70
```

The scale then comes from the panel edges found in the photo. The report
says `user_supplied_dimension`, and the uncertainty is wider than with the
card.

Never paste a marker into a photo afterwards. Its size in the picture is
whatever it was drawn at, so every millimetre measured from it would be
invented.

---

## 7. Packs with information on several sides

Cans, cartons, tubes and jars often print the MRP, date and batch on the
top, base or crimp, and say so on the label ("For MRP (incl. of all
taxes) ... see base of can"). Scan the pack as one:

```bash
python scripts/scan_photo.py side.jpg base.jpg --package can
```

How the pack is decided:

* Each declaration is taken from the photo that read it best, and says
  which photo.
* A tax qualifier printed once ("for MRP ... see base") counts for the
  price stamped on the base.
* A declaration no photo shows is a violation only if the pack never names
  it **and** every photo was read clearly. Otherwise it is "confirm by
  eye".
* Photos of different products (different barcode, quantity or MRP) are
  not merged.

A single photo of one side says what to photograph next, for example "The
pack says to see the base - photograph that side too".

---

## 8. Measuring accuracy

```bash
python scripts/real_eval.py --dir tests/fixtures/real_hires   # re-score a photo set from its saved reads
python scripts/dump_real.py --dir tests/fixtures/real_hires   # read photos with no saved read yet, then score
```

The photo sets are `real`, `real_hires`, `set3` and `rotated`.

To add a test photo:
1. Save it in one of these folders.
2. Add its answer key to that folder's `ground_truth.yaml`.
3. Run `dump_real.py --dir <folder>`.

Method and results are in [`results_log.md`](results_log.md).
