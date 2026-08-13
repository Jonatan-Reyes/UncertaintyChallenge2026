"""Classify every landmass pixel of Wmap.jpg into one of 7 continents.

Wmap.jpg is a silhouette with no per-country borders, so continents can't
be separated by color. Africa+Europe+Asia render as one connected blob,
and North+Central+South America as another (see the connected-component
sizes below), which is why this can't just be "recolor each connected
component" — it needs approximate boundary curves through the Old World
blob (Mediterranean/Red Sea for Africa vs Europe/Asia, ~Urals for
Europe vs Asia) and latitude bands through the Americas blob. Small
islands (their own connected components) fall back to nearest-continent-
marker by centroid distance (Madagascar -> Africa, Japan -> Asia, etc).

Produces a (H, W) int8 array: -1 = ocean/background, 0..6 = continent
index into CONTINENTS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent

CONTINENTS = [
    "North America", "Central America", "South America",
    "Europe", "Africa", "Asia", "Australia",
]
MARKERS = {
    "North America": (430, 470),
    "Central America": (520, 700),
    "South America": (650, 950),
    "Europe": (1080, 400),
    "Africa": (1040, 830),
    "Asia": (1500, 480),
    "Australia": (1631, 972),
}

# Control points (x, y) tracing the Africa / Europe+Asia boundary, hand-
# fit against crops of Wmap.jpg: Atlantic -> Gibraltar -> Mediterranean ->
# Sinai land-bridge -> down the Red Sea -> Arabia's south coast, then a
# near-vertical drop so nothing east of Arabia (mainland Iran/Asia, which
# has no Africa to compete with) can be misclassified as Africa.
AFRICA_NORTH_BOUNDARY = [
    (850, 585), (950, 600), (1050, 610), (1125, 600),
    (1150, 660), (1200, 690), (1250, 710), (1300, 730),
    (1305, 1100), (1500, 1100),
]
# Europe / Asia boundary (~Urals), valid north of the Africa boundary's east end.
EUROPE_ASIA_X = 1300

# Latitude bands splitting the Americas blob, hand-fit to the Panama
# isthmus and northern South America (Colombia/Venezuela) coastline so
# the cut doesn't notch into either side.
NORTH_CENTRAL_BOUNDARY = [(178, 640), (420, 640), (560, 660)]
CENTRAL_SOUTH_BOUNDARY = [(420, 700), (480, 735), (560, 750), (620, 758), (700, 775), (785, 790)]

# Reference points used ONLY for nearest-neighbor assignment of small
# islands — distinct from MARKERS (label positions) because Southeast
# Asia's islands sit roughly equidistant between the Asia and Australia
# *labels*, which are placed over mainland China and central Australia
# respectively; a point nearer the Indonesian archipelago keeps Sumatra/
# Java/Borneo/Sulawesi correctly assigned to Asia instead of Australia.
ISLAND_REFERENCE = dict(MARKERS)
ISLAND_REFERENCE["Asia"] = (1650, 760)


def _boundary_y(x: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    return np.interp(x, xs, ys)


def _africa_boundary_y(x: np.ndarray) -> np.ndarray:
    return _boundary_y(x, AFRICA_NORTH_BOUNDARY)


def build_continent_mask(map_path: Path) -> np.ndarray:
    img = Image.open(map_path).convert("L")
    arr = np.array(img)
    land = arr < 128

    labeled, n = ndimage.label(land, structure=np.ones((3, 3)))
    sizes = ndimage.sum(land, labeled, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    old_world_label = order[0] + 1   # biggest: Africa+Europe+Asia
    americas_label = order[1] + 1    # 2nd biggest: N+C+S America

    ys_all, xs_all = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
    cont = np.full(arr.shape, -1, dtype=np.int8)
    idx = {name: i for i, name in enumerate(CONTINENTS)}

    # --- Old World blob: split by boundary curves ---
    ow = labeled == old_world_label
    x_ow, y_ow = xs_all[ow], ys_all[ow]
    af_boundary = _africa_boundary_y(x_ow)
    is_africa = y_ow > af_boundary
    is_europe = (~is_africa) & (x_ow < EUROPE_ASIA_X)
    is_asia = (~is_africa) & (~is_europe)
    sub = np.empty(x_ow.shape, dtype=np.int8)
    sub[is_africa] = idx["Africa"]
    sub[is_europe] = idx["Europe"]
    sub[is_asia] = idx["Asia"]
    cont[ow] = sub

    # --- Americas blob: split by (slanted) latitude band ---
    am = labeled == americas_label
    x_am, y_am = xs_all[am], ys_all[am]
    nc_boundary = _boundary_y(x_am, NORTH_CENTRAL_BOUNDARY)
    cs_boundary = _boundary_y(x_am, CENTRAL_SOUTH_BOUNDARY)
    sub2 = np.empty(y_am.shape, dtype=np.int8)
    sub2[y_am < nc_boundary] = idx["North America"]
    sub2[(y_am >= nc_boundary) & (y_am < cs_boundary)] = idx["Central America"]
    sub2[y_am >= cs_boundary] = idx["South America"]
    cont[am] = sub2

    # --- Every other component (islands, Greenland, Australia, etc): ---
    # nearest continent reference point by centroid distance.
    marker_xy = np.array([ISLAND_REFERENCE[name] for name in CONTINENTS])
    for lbl in range(1, n + 1):
        if lbl in (old_world_label, americas_label):
            continue
        comp = labeled == lbl
        cy, cx = ys_all[comp].mean(), xs_all[comp].mean()
        d2 = (marker_xy[:, 0] - cx) ** 2 + (marker_xy[:, 1] - cy) ** 2
        nearest = int(np.argmin(d2))
        cont[comp] = nearest

    return cont


def border_mask(cont: np.ndarray, thickness: int = 3) -> np.ndarray:
    """Pixels within ``thickness`` of a continent/continent or
    continent/ocean boundary — for drawing a solid outline over a
    choropleth fill."""
    edge = np.zeros(cont.shape, dtype=bool)
    edge[:-1, :] |= cont[:-1, :] != cont[1:, :]
    edge[1:, :] |= cont[:-1, :] != cont[1:, :]
    edge[:, :-1] |= cont[:, :-1] != cont[:, 1:]
    edge[:, 1:] |= cont[:, :-1] != cont[:, 1:]
    return ndimage.binary_dilation(edge, iterations=thickness)


if __name__ == "__main__":
    # Quick visual sanity check: flat-color each continent and save.
    mask = build_continent_mask(HERE.parents[2] / "Wmap.jpg")
    palette = [
        (0, 114, 178), (86, 180, 233), (0, 158, 115),
        (240, 228, 66), (213, 94, 0), (204, 121, 167), (0, 0, 0),
    ]
    out = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    for i, color in enumerate(palette):
        out[mask == i] = color
    Image.fromarray(out).save(HERE / "continent_mask_check.png")
    print("wrote", HERE / "continent_mask_check.png")
    for i, name in enumerate(CONTINENTS):
        print(name, (mask == i).sum())
