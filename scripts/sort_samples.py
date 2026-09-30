#!/usr/bin/env python3
"""
Sort copies of the real test photos into samples/ with readable names:

  samples/1_works_well/   every legible declaration is read - demo photos
  samples/2_needs_work/   printed information the engine still misses
  samples/3_packages/<pack>/   several sides of ONE pack, scanned together

Decided from the current score (scripts/real_eval.py), so re-run this after
any change:  python scripts/sort_samples.py
The originals stay in tests/fixtures/real (photo1..photoN), which the tests use.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

FIX = ROOT / "tests" / "fixtures" / "real"
# Full-resolution copies of photos 18-38 (the phone's own files, not the
# WhatsApp copies). Used for the samples wherever they exist.
HIRES = ROOT / "tests" / "fixtures" / "real_hires"
OUT = ROOT / "samples"

NAMES = {
    1: "streax_hair_serum__side", 2: "dabur_amla_oil_45ml__back",
    3: "haldiram_aloo_bhujia_35g__back", 4: "maggi_noodles__back_cropped",
    5: "maggi_masala_sachet_6g__back", 6: "bingo_mad_angles__back",
    7: "kalash_roli_sachet__back", 8: "kesar_chandan_jar__hindi_side",
    9: "kesar_chandan_jar__mrp_side", 10: "rocket_dhuna__back_close",
    11: "rocket_dhuna__back_far", 12: "maharaja_gulal__back",
    13: "cavins_milkshake_can__side", 14: "rb_dry_fruits__label",
    15: "kesar_chandan_jar__front", 16: "maharaja_gulal__front",
    17: "gopi_chandan_box__side", 18: "troovy_krunchies_40g__back",
    19: "amul_calci_milk__top", 20: "amul_calci_milk__side",
    21: "troovy_chips_70g__back", 22: "amul_calci_milk__front",
    23: "monster_can__side", 24: "rajam_herbal_jar_500g__label",
    25: "monster_can__base", 26: "vim_gel_pouch__back",
    27: "paper_boat_swing__back", 28: "zydus_sweetener_jar__side_inkjet",
    29: "zydus_sweetener_jar__side_quantity", 30: "myfitness_peanut_butter__side",
    31: "kissan_jam_cup__side", 32: "bodywise_lotion__sticker_side",
    33: "zebronics_converter__box_label", 34: "bodywise_lotion__back",
    35: "xerodel_cream_tube__back", 36: "graceness_toothpaste__back",
    37: "nat_habit_hair_mask__back", 38: "oregano_seasoning_sachet__back",
}
PACKS = {
    "amul_calci_milk": [19, 20, 22], "monster_can": [23, 25],
    "kesar_chandan_jar": [8, 9, 15], "zydus_sweetener_jar": [28, 29],
    "bodywise_lotion": [32, 34],
}
# Not a pack set: the Maharaja Gulal photos show only its front and back,
# and a pack scan asserts EVERY side was photographed - its sides may carry
# the consumer care. They are sorted as single photos.


def main():
    from real_eval import evaluate_dir

    rows = {r["photo"]: r for r in evaluate_dir(FIX, verbose=False)["rows"]
            if r["variant"] == "default"}
    if HIRES.exists():
        rows.update({r["photo"]: r for r in evaluate_dir(HIRES, verbose=False)["rows"]
                     if r["variant"] == "default"})

    def src(n):
        h = HIRES / f"photo{n}.jpg"
        return h if h.exists() else FIX / f"photo{n}.jpg"
    shutil.rmtree(OUT, ignore_errors=True)
    in_pack = {n for ns in PACKS.values() for n in ns}
    works, needs = [], []
    for n, name in NAMES.items():
        if n in in_pack or not (FIX / f"photo{n}.jpg").exists():
            continue
        r = rows.get(f"photo{n}")
        miss = (r["missed"] + r["wrong"]) if r else ["not read yet"]
        (needs if miss else works).append((n, name, miss))

    for folder, items in (("1_works_well", works), ("2_needs_work", needs)):
        (OUT / folder).mkdir(parents=True)
        for n, name, _ in items:
            shutil.copy(src(n), OUT / folder / f"{name}.jpg")
    for pack, ns in PACKS.items():
        d = OUT / "3_packages" / pack
        d.mkdir(parents=True)
        for n in ns:
            shutil.copy(src(n), d / f"{NAMES[n]}.jpg")

    md = ["# Sample photos, sorted", "",
          "Copies of the real test photos. The originals stay in `tests/fixtures/real` "
          "as photo1..photoN (the tests use those); photos 18-38 are copied at full "
          "resolution from `tests/fixtures/real_hires`. Re-sort after a change: "
          "`python scripts/sort_samples.py`.", "",
          "## 1_works_well", "", "Every legible declaration on the photo is read. Use these for the demo.", "",
          "| File | Test photo |", "|---|---|"]
    md += [f"| {name}.jpg | photo{n} |" for n, name, _ in works]
    md += ["", "## 2_needs_work", "",
           "Printed information the engine still misses: mostly smudged stickers, inkjet on "
           "curved or metal surfaces, text printed sideways along an edge, red text on foil.", "",
           "| File | Test photo | Still missed |", "|---|---|---|"]
    md += [f"| {name}.jpg | photo{n} | "
           f"{', '.join(m.split('=')[0].replace('_', ' ') for m in miss)} |"
           for n, name, miss in needs]
    md += ["", "## 3_packages", "",
           "Several sides of ONE pack. The MRP, date and batch are often stamped on the top, "
           "base or crimp, with \"For MRP (incl. of all taxes) ... see base\" on the label. An "
           "inspector checks the whole pack, so scan each folder as one pack:", "",
           "```", "python scripts\\scan_photo.py samples\\3_packages\\monster_can --package monster_can",
           "start data\\reports\\monster_can.html", "```", "",
           "| Pack | Photos |", "|---|---|"]
    md += [f"| {p} | " + ", ".join(f"{NAMES[n]}.jpg (photo{n})" for n in ns) + " |"
           for p, ns in PACKS.items()]
    (OUT / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(works)} work well, {len(needs)} need work, {len(PACKS)} packs -> {OUT}")
    for n, name, miss in needs:
        print(f"  needs work: {name}: {', '.join(miss)}")


if __name__ == "__main__":
    main()
