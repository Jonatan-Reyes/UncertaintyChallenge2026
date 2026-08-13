"""Continent density map of val's OOD-domain animal classes.

Three modes (--mode):
  ood      (default) all val rows with domain=='ood'.
  id-only  only classes that appear in val's id rows but have ZERO ood
           rows (weighted by their id-domain counts).
  ood-only only classes that appear in val's ood rows but have ZERO id
           rows (weighted by their ood-domain counts).

Maps each class to its native continent(s) via class_names.json's
"species" list (splitting a class's count evenly across continents when a
species/domesticate spans several — e.g. cattle counts 1/6 toward each of
its 6 listed continents), and fills each continent's actual landmass (via
continent_mask.build_continent_mask) with a color encoding the weighted
total, choropleth-style.

Non-animal classes (0 "empty", 54 "motorcycle") are excluded.

Caveats inherited from class_names.json: continents are native/domesticated
*range*, not confirmed per-photo GPS locations, and were compiled by manual
research rather than sourced from the dataset itself.

Usage:
    python student/code/evaluation/ood_world_map.py --data-root challenge_data --mode ood
    python student/code/evaluation/ood_world_map.py --data-root challenge_data --mode id-only
    python student/code/evaluation/ood_world_map.py --data-root challenge_data --mode ood-only
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from continent_mask import CONTINENTS, MARKERS, border_mask, build_continent_mask

HERE = Path(__file__).resolve().parent
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_PATH_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Sequential single-hue ramp (light -> dark amber/red) for magnitude;
# continents with zero weight get NO_DATA_COLOR instead (still visibly
# land, but clearly not part of the scale).
COLOR_LOW = (255, 213, 128)
COLOR_HIGH = (165, 30, 15)
NO_DATA_COLOR = (210, 210, 210)

# North/Central/South America are one contiguous landmass, not separated by
# ocean like Africa/Europe/Asia/Australia are from each other and from the
# Americas. class_names.json tags nearly every Neotropical species with all
# three "North America, Central America, South America" entries even when
# its real range is just Central+South America — so a literal
# len(continents) == 1 filter would silently drop the entire Americas.
# Group by landmass instead for the single-region test (still plotted as
# separate North/Central/South bubbles either way).
REGION_OF = {
    "North America": "Americas", "Central America": "Americas", "South America": "Americas",
    "Africa": "Africa", "Europe": "Europe", "Asia": "Asia", "Australia": "Australia",
}


def lerp_color(t: float) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(COLOR_LOW, COLOR_HIGH))


def get_counts(va: pd.DataFrame, mode: str, num_classes: int) -> tuple[pd.Series, str, str]:
    id_rows = va[va["domain"] == "id"]
    ood_rows = va[va["domain"] == "ood"]
    id_counts = id_rows["y"].value_counts().reindex(range(num_classes), fill_value=0)
    ood_counts = ood_rows["y"].value_counts().reindex(range(num_classes), fill_value=0)

    if mode == "ood":
        counts = ood_rows["y"].value_counts().sort_index()
        title = "Density of val 'ood'-domain animals by continent of origin"
        desc = "all val rows with domain='ood'"
    elif mode == "id-only":
        classes = [y for y in range(num_classes) if id_counts[y] > 0 and ood_counts[y] == 0]
        counts = id_counts[classes]
        title = "Density of classes present ONLY in val's 'id' domain (0 ood examples)"
        desc = "val id-domain counts for classes absent from ood"
    elif mode == "ood-only":
        classes = [y for y in range(num_classes) if ood_counts[y] > 0 and id_counts[y] == 0]
        counts = ood_counts[classes]
        title = "Density of classes present ONLY in val's 'ood' domain (0 id examples)"
        desc = "val ood-domain counts for classes absent from id"
    else:
        raise ValueError(mode)
    return counts, title, desc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map", type=Path, default=HERE.parents[2] / "Wmap.jpg")
    parser.add_argument("--class-names", type=Path, default=HERE.parents[1] / "class_names.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mode", choices=["ood", "id-only", "ood-only"], default="ood")
    parser.add_argument("--single-continent-only", action="store_true",
                         help="drop classes whose species/domesticate range spans >1 continent")
    args = parser.parse_args()

    with args.class_names.open() as f:
        cn = json.load(f)
    species = {s["class_id"]: s for s in cn["species"]}

    va = pd.read_csv(args.data_root / "val" / "labels.csv")
    counts, title, desc = get_counts(va, args.mode, cn["num_classes"])
    suffix = "_single-continent" if args.single_continent_only else ""
    output = args.output or (HERE.parents[1] / "results" / "evaluation" / "images" / f"world_map_{args.mode}{suffix}.png")
    if args.single_continent_only:
        title += " — single-continent species only"

    weighted: dict[str, float] = defaultdict(float)
    excluded = []
    for y, n in counts.items():
        conts = species[y]["continents"]
        if not conts:
            excluded.append((int(y), cn["class_names"][y], int(n)))
            continue
        if args.single_continent_only and len({REGION_OF[c] for c in conts}) > 1:
            excluded.append((int(y), cn["class_names"][y], int(n)))
            continue
        share = n / len(conts)
        for c in conts:
            weighted[c] += share

    print(f"mode={args.mode}  rows used: {int(counts.sum())}  (excluded non-animal: {excluded})")
    for c, w in sorted(weighted.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<16} {w:.1f}")

    # --- choropleth fill: color each continent's actual landmass ---
    mask = build_continent_mask(args.map)
    rgb = np.array(Image.open(args.map).convert("RGB"))
    max_w = max(weighted.values())
    for i, cont in enumerate(CONTINENTS):
        w = weighted.get(cont, 0.0)
        color = lerp_color(w / max_w) if w > 0 else NO_DATA_COLOR
        rgb[mask == i] = color
    rgb[border_mask(mask, thickness=3)] = (0, 0, 0)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")

    font_label = ImageFont.truetype(FONT_PATH, 30)
    title_size = 42 if len(title) < 60 else 32
    font_title = ImageFont.truetype(FONT_PATH, title_size)
    font_small = ImageFont.truetype(FONT_PATH_REG, 20)

    for cont, (x, y) in MARKERS.items():
        w = weighted.get(cont, 0.0)
        label = cont if w <= 0 else f"{cont}\n{w:.0f}"
        draw.multiline_text((x, y), label, font=font_label, fill=(20, 10, 5), anchor="mm",
                             align="center", stroke_width=4, stroke_fill=(255, 255, 255), spacing=6)

    draw.text((40, 30), title, font=font_title, fill=(20, 10, 5),
              stroke_width=3, stroke_fill=(255, 255, 255))

    # Legend: sequential ramp low->high, plus the no-data swatch.
    lx, ly, lw, lh = img.width - 340, 30, 260, 22
    for i in range(lw):
        color = lerp_color(i / (lw - 1))
        draw.line([(lx + i, ly), (lx + i, ly + lh)], fill=color)
    draw.rectangle([lx, ly, lx + lw, ly + lh], outline=(40, 20, 10), width=2)
    draw.text((lx, ly + lh + 6), "fewer", font=font_small, fill=(20, 10, 5))
    draw.text((lx + lw, ly + lh + 6), "more", font=font_small, fill=(20, 10, 5), anchor="ra")
    ndx, ndy = lx, ly + lh + 34
    draw.rectangle([ndx, ndy, ndx + 22, ndy + 22], fill=NO_DATA_COLOR, outline=(40, 20, 10), width=2)
    draw.text((ndx + 30, ndy + 3), "no data (0 weight)", font=font_small, fill=(20, 10, 5))

    if args.single_continent_only:
        caption_lines = [
            f"Count of {desc}, restricted to species/domesticates confined to a SINGLE landmass (N/C/S America",
            "counted as one) — cattle/sheep/goat/horse/camel excluded, not split. Range, not confirmed photo GPS.",
        ]
    else:
        caption_lines = [
            f"Weighted count of {desc} (excl. non-animal classes), split evenly across a species' listed",
            "native/domesticated continents (class_names.json) — species range, not confirmed photo GPS.",
        ]
    for i, line in enumerate(caption_lines):
        draw.text((40, img.height - 62 + i * 26), line, font=font_small, fill=(90, 90, 90))

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output, quality=92)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
