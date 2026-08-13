"""N example images per class, arranged as a labeled contact sheet.

Randomly samples ``--n`` (default 5) train images per class and lays them
out in a row with a "class {y}" caption bar. Also reports per-class train
and val example counts (and flags any class with zero examples), since
knowing whether a class is actually absent is useful before searching for
a name/label mismatch.

NOTE: this does NOT overlay species names. An earlier attempt matched
class_mapping.json's numeric ids (both "new" and "original") against the
public WCS Camera Traps taxonomy (lila.science) and looked plausible, but
spot-checking against the actual photos proved both wrong (e.g. the id
captioned "panthera onca" showed cattle, not a jaguar) — that taxonomy's
id numbering doesn't correspond to this challenge's ids at all. No
verified label->species mapping is available for this dataset; don't
reintroduce one without checking it against real images first.

Usage:
    python student/code/data-exploration/make_class_examples.py --data-root challenge_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 22
BAR_COLOR = (0, 0, 0, 160)
TEXT_COLOR = (255, 255, 255)
THUMB_SIZE = (280, 210)
SEED = 0


def find_image(images_dir: Path, uid: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{uid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for uid {uid} in {images_dir}")


def contact_sheet(thumbs: list[Image.Image], label: str) -> Image.Image:
    n = len(thumbs)
    w, h = THUMB_SIZE
    pad = 4
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    bar_h = FONT_SIZE + 16
    sheet = Image.new("RGB", (n * w + (n + 1) * pad, h + 2 * pad + bar_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), label, font=font, fill=TEXT_COLOR)
    for i, t in enumerate(thumbs):
        sheet.paste(t, (pad + i * (w + pad), bar_h + pad))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=HERE.parents[1] / "results" / "data-exploration" / "images" / "class_examples")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--n", type=int, default=5, help="examples per class")
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_root / "train" / "labels.csv")
    val_df = pd.read_csv(args.data_root / "val" / "labels.csv")

    import json
    with (args.data_root / "class_mapping.json").open() as f:
        K = json.load(f)["num_classes"]

    tr_counts = train_df["y"].value_counts().reindex(range(K), fill_value=0)
    va_counts = val_df["y"].value_counts().reindex(range(K), fill_value=0)
    zero_total = [y for y in range(K) if tr_counts[y] + va_counts[y] == 0]
    zero_train = [y for y in range(K) if tr_counts[y] == 0]
    zero_val = [y for y in range(K) if va_counts[y] == 0]
    print(f"classes with 0 examples in train+val combined: {zero_total}")
    print(f"classes with 0 examples in train: {zero_train}")
    print(f"classes with 0 examples in val:   {zero_val}")
    print(f"min train count per class: {tr_counts.min()} (class {tr_counts.idxmin()})\n")

    df = pd.read_csv(args.data_root / args.split / "labels.csv")
    images_dir = args.data_root / args.split / "images"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(df["y"].unique())
    for y in classes:
        rows = df[df["y"] == y]
        sample = rows.sample(n=min(args.n, len(rows)), random_state=SEED)
        thumbs = []
        for uid in sample["uid"]:
            src = find_image(images_dir, uid)
            img = Image.open(src).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            canvas = Image.new("RGB", THUMB_SIZE, (0, 0, 0))
            canvas.paste(img, ((THUMB_SIZE[0] - img.width) // 2, (THUMB_SIZE[1] - img.height) // 2))
            thumbs.append(canvas)
        sheet = contact_sheet(thumbs, f"class {y}  (n={len(rows)} in {args.split})")
        out_path = args.output_dir / f"{y:02d}.jpg"
        sheet.save(out_path, quality=90)
        print(f"wrote {out_path}")

    print(f"\n{len(classes)} classes -> {args.output_dir}")


if __name__ == "__main__":
    main()
