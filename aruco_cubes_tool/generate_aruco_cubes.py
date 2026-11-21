#!/usr/bin/env python3
"""Generate printable black-and-white ArUco cube nets (PDF) plus the matching geometry file (YAML).

    python generate_aruco_cubes.py                       # 4 cubes, A4 landscape -> aruco_cubes/
    python generate_aruco_cubes.py --marker-measured-mm 44.8
                                                         # YAML only, from a caliper measurement

Cube frame: origin at the cube centre, axes X, Y, Z.  Net on the page (printed side faces OUT after folding):

                +Y
        +Z  +X  -Z  -X          (+Y sits above +Z, -Y below +Z)
                -Y

Marker IDs: cube k uses IDs 6k .. 6k+5 in the order +Z, +X, -Z, -X, +Y, -Y.
All lengths inside the code are millimetres on the page (y up, like PDF) and metres in the YAML.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import cv2
import numpy as np
import yaml
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfgen.canvas import FILL_NON_ZERO

MM = 72.0 / 25.4                                   # PDF points per millimetre
PAGES_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "LETTER": (279.4, 215.9)}   # landscape (w, h)
FACES = ["+Z", "+X", "-Z", "-X", "+Y", "-Y"]       # also the ID order inside a cube
NET_GRID = {"+Z": (0, 0), "+X": (1, 0), "-Z": (2, 0), "-X": (3, 0), "+Y": (0, 1), "-Y": (0, -1)}   # (col, row), row up
SIDE_DIRS = {"R": (1, 0), "L": (-1, 0), "U": (0, 1), "D": (0, -1)}
SYNTH_K = np.array([[1300.0, 0, 960.0], [0, 1300.0, 540.0], [0, 0, 1]])
SYNTH_SIZE = (1920, 1080)
EPS_OVERLAP_MM = 0.03                              # black modules overlap by this much, so no hairline seams
CLEARANCE_MM = 3.0                                 # free space kept around the net for the label / scale bar


def _v(*a):
    return np.array(a, float)


# marker right axis, marker up axis, outward normal (in the cube frame): exactly the table of the specification
FACE_TABLE = {
    "+Z": (_v(1, 0, 0), _v(0, 1, 0), _v(0, 0, 1)),
    "+X": (_v(0, 0, -1), _v(0, 1, 0), _v(1, 0, 0)),
    "-Z": (_v(-1, 0, 0), _v(0, 1, 0), _v(0, 0, -1)),
    "-X": (_v(0, 0, 1), _v(0, 1, 0), _v(-1, 0, 0)),
    "+Y": (_v(1, 0, 0), _v(0, 0, -1), _v(0, 1, 0)),
    "-Y": (_v(1, 0, 0), _v(0, 0, 1), _v(0, -1, 0)),
}


def axis_of(face: str) -> np.ndarray:
    a = np.zeros(3)
    a["XYZ".index(face[1])] = 1.0 if face[0] == "+" else -1.0
    return a


def assert_face_table() -> None:
    """right x up = normal for every face, and the normal is the face's own axis (right-handed)."""
    for f, (r, u, n) in FACE_TABLE.items():
        assert np.allclose(np.cross(r, u), n), f"face {f}: right x up != normal"
        assert np.allclose(n, axis_of(f)), f"face {f}: normal is not the face axis"


def derive_frames() -> dict:
    """Fold the net in 3D: start with +Z = (X, Y, Z) and cross every shared edge. Returns face -> (right, up, normal)."""
    inv = {v: k for k, v in NET_GRID.items()}
    frames = {"+Z": FACE_TABLE["+Z"]}
    todo = ["+Z"]
    while todo:
        f = todo.pop(0)
        r, u, n = frames[f]
        c, row = NET_GRID[f]
        for side, (dc, dr) in SIDE_DIRS.items():
            g = inv.get((c + dc, row + dr))
            if g is None or g in frames:
                continue
            if side == "R":
                frames[g] = (-n, u, r)
            elif side == "L":
                frames[g] = (n, u, -r)
            elif side == "U":
                frames[g] = (r, -n, u)
            else:
                frames[g] = (r, n, -u)
            todo.append(g)
    return frames


# ----------------------------------------------------------------------------------------------- dictionary
def load_dictionary(name: str):
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        sys.exit(f"ERROR: unknown ArUco dictionary '{name}' (try DICT_5X5_100, DICT_4X4_50, ...)")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    return d, int(d.markerSize), int(d.bytesList.shape[0])


def marker_bitmap(dictionary, marker_id: int, bits: int) -> np.ndarray:
    """(bits+2) x (bits+2) array, True = black module, row 0 = top of the marker (black border included)."""
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, bits + 2, borderBits=1)
    assert img.shape == (bits + 2, bits + 2)
    return img < 128


# ----------------------------------------------------------------------------------------------- geometry (YAML)
def build_yaml_data(dict_name: str, cube_m: float, marker_m: float, num_cubes: int, source: str) -> dict:
    cubes = []
    for k in range(num_cubes):
        markers = []
        for i, face in enumerate(FACES):
            r, u, n = FACE_TABLE[face]
            centre = n * cube_m / 2.0
            h = marker_m / 2.0
            corners = [centre + sx * h * r + sy * h * u for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]   # TL TR BR BL
            markers.append({
                "id": 6 * k + i,
                "face": face,
                "marker_side_m": marker_m,
                "corners_cube_m": [[float(np.round(x, 9)) + 0.0 for x in c] for c in corners],
            })
        cubes.append({
            "cube_id": k,
            "pose_world": np.eye(4).tolist() if k == 0 else None,
            "markers": markers,
        })
    return {"dictionary": dict_name, "cube_side_m": cube_m, "marker_side_m": marker_m,
            "marker_size_source": source, "cubes": cubes}


class _Flow(list):
    pass


yaml.SafeDumper.add_representer(_Flow, lambda d, x: d.represent_sequence("tag:yaml.org,2002:seq", x, flow_style=True))

YAML_HEADER = """\
# ArUco cube geometry (generated by generate_aruco_cubes.py; do not edit by hand).
# Cube frame: origin = cube centre, axes X/Y/Z, units = metres.
# corners_cube_m: 4 corners of the BLACK marker square in the OpenCV ArUco order
#   top-left, top-right, bottom-right, bottom-left, as seen when looking at the marker from outside the cube.
# pose_world: 4x4 cube->world transform. Cube 0 = identity; the others are null until calibrated.
# IDs: cube k uses 6k..6k+5 in the face order +Z, +X, -Z, -X, +Y, -Y.
"""


def write_yaml(path: str, data: dict) -> None:
    data = dict(data)
    cubes = []
    for c in data["cubes"]:
        c = dict(c)
        if c["pose_world"] is not None:
            c["pose_world"] = [_Flow(float(x) for x in row) for row in c["pose_world"]]
        c["markers"] = [dict(m, corners_cube_m=[_Flow(row) for row in m["corners_cube_m"]]) for m in c["markers"]]
        cubes.append(c)
    data["cubes"] = cubes
    with open(path, "w") as f:
        f.write(YAML_HEADER)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


# ----------------------------------------------------------------------------------------------- net layout
def convex_overlap(a, b, eps=1e-6) -> bool:
    """True if two convex polygons overlap with more than eps depth (touching edges do not count)."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            ax = np.array([-(q[1] - p[1]), q[0] - p[0]], float)
            nrm = np.linalg.norm(ax)
            if nrm == 0:
                continue
            ax /= nrm
            pa = [float(np.dot(ax, v)) for v in a]
            pb = [float(np.dot(ax, v)) for v in b]
            if max(pa) <= min(pb) + eps or max(pb) <= min(pa) + eps:
                return False
    return True


def rect_poly(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def build_layout(cube_mm: float, tab_mm: float, page_w: float, page_h: float, margin: float) -> dict:
    s = cube_mm
    frames = derive_frames()
    for f in FACES:                                                  # the folded net must reproduce the table
        assert all(np.allclose(a, b) for a, b in zip(frames[f], FACE_TABLE[f])), f"net folding disagrees with table for {f}"
    inv = {v: k for k, v in NET_GRID.items()}
    faces = {f: {"rect": (c * s, r * s, (c + 1) * s, (r + 1) * s)} for f, (c, r) in NET_GRID.items()}

    boundary, internal = [], []
    for f, (c, r) in NET_GRID.items():
        x0, y0, x1, y1 = faces[f]["rect"]
        rr, uu, nn = frames[f]
        for side, (dc, dr) in SIDE_DIRS.items():
            p0, p1, o = {"R": ((x1, y0), (x1, y1), (1, 0)), "L": ((x0, y0), (x0, y1), (-1, 0)),
                         "U": ((x0, y1), (x1, y1), (0, 1)), "D": ((x0, y0), (x1, y0), (0, -1))}[side]
            if inv.get((c + dc, r + dr)) is None:
                vec = nn + {"R": rr, "L": -rr, "U": uu, "D": -uu}[side]      # identifies the cube edge in 3D
                boundary.append({"face": f, "side": side, "p0": p0, "p1": p1, "o": o, "key": tuple(int(round(x)) for x in vec)})
            elif side in "RU":
                internal.append((p0, p1))

    groups: dict = {}
    for e in boundary:
        groups.setdefault(e["key"], []).append(e)
    pairs = list(groups.values())
    assert len(pairs) == 7 and all(len(p) == 2 for p in pairs), "a cube net must have 7 pairs of free edges"

    # Tab taper: the two tabs on perpendicular edges of a cap fold into the SAME plane and meet at the cube corner.
    # They only stay clear of each other if both slanted sides are steeper than 45 deg, hence inset = 1.25 x tab height.
    inset = 1.25 * tab_mm
    if 2 * inset > 0.7 * s:
        sys.exit(f"ERROR: --cube-mm {s:g} is too small for {tab_mm:g} mm glue tabs (their folded tabs would collide); use smaller --tab-mm.")

    def make_tab(e):
        p0, p1, o = np.array(e["p0"], float), np.array(e["p1"], float), np.array(e["o"], float)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        poly = [p0, p1, p1 + o * tab_mm - d * inset, p0 + o * tab_mm + d * inset]
        return {"edge": e, "poly": [tuple(map(float, p)) for p in poly]}

    squares = [rect_poly(*faces[f]["rect"]) for f in FACES]
    best = None
    for choice in itertools.product((0, 1), repeat=len(pairs)):
        tabs = [make_tab(pairs[i][choice[i]]) for i in range(len(pairs))]
        polys = [t["poly"] for t in tabs]
        if any(convex_overlap(polys[i], polys[j]) for i in range(len(polys)) for j in range(i + 1, len(polys))):
            continue
        if any(convex_overlap(p, q) for p in polys for q in squares):
            continue
        pts = np.array([p for poly in polys + squares for p in poly])
        w, h = pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()
        if w + 2 * margin > page_w + 1e-9 or h + 2 * margin > page_h + 1e-9:
            continue
        score = (round(w * h, 6), choice)
        if best is None or score < best[0]:
            best = (score, tabs, pts)
    if best is None:
        pts = np.array([p for poly in squares for p in poly])
        sys.exit(f"ERROR: the cube net ({pts[:, 0].max() - pts[:, 0].min():.0f} x {pts[:, 1].max() - pts[:, 1].min():.0f} mm) "
                 f"plus {tab_mm:g} mm glue tabs does not fit into the printable area of the page "
                 f"({page_w:g} x {page_h:g} mm minus {margin:g} mm margin on every side). Use a smaller --cube-mm, a larger --page "
                 f"or smaller --tab-mm.")
    _, tabs, pts = best
    sx = (page_w - (pts[:, 0].max() - pts[:, 0].min())) / 2.0 - pts[:, 0].min()
    sy = (page_h - (pts[:, 1].max() - pts[:, 1].min())) / 2.0 - pts[:, 1].min()

    def sh(p):
        return (float(p[0] + sx), float(p[1] + sy))

    tab_edges = {(t["edge"]["face"], t["edge"]["side"]) for t in tabs}
    cut, fold = [], [(sh(a), sh(b)) for a, b in internal]
    for e in boundary:
        (fold if (e["face"], e["side"]) in tab_edges else cut).append((sh(e["p0"]), sh(e["p1"])))
    for t in tabs:                                                   # three outer sides of every tab are cut lines
        p = [sh(q) for q in t["poly"]]
        cut += [(p[1], p[2]), (p[2], p[3]), (p[3], p[0])]
        t["poly"] = p
    out_faces = {}
    for f in FACES:
        x0, y0, x1, y1 = faces[f]["rect"]
        out_faces[f] = {"rect": (x0 + sx, y0 + sy, x1 + sx, y1 + sy), "right": frames[f][0], "up": frames[f][1], "normal": frames[f][2]}
    return {"faces": out_faces, "tabs": tabs, "cut": cut, "fold": fold, "page": (page_w, page_h), "margin": margin,
            "tab_desc": [f"{t['edge']['face']}:{t['edge']['side']}" for t in tabs]}


def place_extras(layout: dict, labels: list[str], info_line: str, cube_mm: float) -> dict:
    """Find free places (outside the net, >= margin from the page edge) for label, print note and 100 mm scale bar."""
    pw, ph = layout["page"]
    m = layout["margin"]
    obstacles = [rect_poly(*f["rect"]) for f in layout["faces"].values()] + [t["poly"] for t in layout["tabs"]]
    note = "Print at 100% / Actual size, do not fit to page"
    mmpt = 1.0 / MM
    label_w = max(stringWidth(t, "Helvetica-Bold", 12) for t in labels) * mmpt
    info_w = stringWidth(info_line, "Helvetica", 7.5) * mmpt
    blocks = [
        ("scale", 100.0, 10.0),
        ("note", stringWidth(note, "Helvetica-Bold", 10) * mmpt, 5.0),
        ("label", max(label_w, info_w), 10.0),
    ]
    placed = {}
    for name, w, h in blocks:
        spot = None
        y = ph - m - h
        while y >= m - 1e-9 and spot is None:
            x = m
            while x + w <= pw - m + 1e-9:
                probe = rect_poly(x - CLEARANCE_MM, y - CLEARANCE_MM, x + w + CLEARANCE_MM, y + h + CLEARANCE_MM)
                if not any(convex_overlap(probe, o) for o in obstacles):
                    spot = (x, y)
                    break
                x += 1.0
            y -= 1.0
        if spot is None:
            sys.exit(f"ERROR: no free space on the page for the {name} block next to the net; use a smaller --cube-mm or a larger --page.")
        placed[name] = (spot[0], spot[1], w, h)
        obstacles.append(rect_poly(spot[0], spot[1], spot[0] + w, spot[1] + h))
    return placed


# ----------------------------------------------------------------------------------------------- PDF drawing
def draw_marker(c, dictionary, bits: int, marker_id: int, cx: float, cy: float, marker_mm: float) -> int:
    """Vector marker: one filled rectangle per black run, plus thin bridges so rows never leave hairline gaps."""
    bm = marker_bitmap(dictionary, marker_id, bits)
    n = bm.shape[0]
    u = marker_mm / n
    x0, ytop = cx - marker_mm / 2.0, cy + marker_mm / 2.0
    rects = []

    def runs(row):
        out, start = [], None
        for i, v in enumerate(list(row) + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                out.append((start, i))
                start = None
        return out

    for r in range(n):
        for a, b in runs(bm[r]):
            rects.append((x0 + a * u, ytop - (r + 1) * u, (b - a) * u, u))
    for r in range(n - 1):                                           # bridges lie fully inside black area
        for a, b in runs(bm[r] & bm[r + 1]):
            rects.append((x0 + a * u, ytop - (r + 1) * u - EPS_OVERLAP_MM, (b - a) * u, 2 * EPS_OVERLAP_MM))
    c.setFillColorRGB(0, 0, 0)
    path = c.beginPath()                                             # ONE path per marker, non-zero fill: the union of
    for x, y, w, h in rects:                                         # all rectangles is rasterised at once -> no seams
        path.rect(x * MM, y * MM, w * MM, h * MM)
    c.drawPath(path, stroke=0, fill=1, fillMode=FILL_NON_ZERO)
    return len(rects)


def draw_page(c, layout, extras, dictionary, bits, k, args, info_line) -> dict:
    pw, ph = layout["page"]
    c.setPageSize((pw * MM, ph * MM))
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)
    n_rects = 0
    for i, f in enumerate(FACES):
        x0, y0, x1, y1 = layout["faces"][f]["rect"]
        n_rects += draw_marker(c, dictionary, bits, 6 * k + i, (x0 + x1) / 2, (y0 + y1) / 2, args.marker_mm)
    c.setDash([])
    c.setLineWidth(0.3)
    for (a, b) in layout["cut"]:
        c.line(a[0] * MM, a[1] * MM, b[0] * MM, b[1] * MM)
    c.setLineWidth(0.25)
    c.setDash([2.0 * MM, 1.5 * MM])
    for (a, b) in layout["fold"]:
        c.line(a[0] * MM, a[1] * MM, b[0] * MM, b[1] * MM)
    c.setDash([])

    sx, sy, sw, sh_ = extras["scale"]                                # 100 mm scale bar with end ticks
    bar_y = sy + 2.0
    p0, p1 = (sx * MM, bar_y * MM), ((sx + 100.0) * MM, bar_y * MM)
    c.setLineWidth(0.3)
    c.line(*p0, *p1)
    c.line(p0[0], (bar_y - 2.0) * MM, p0[0], (bar_y + 2.0) * MM)
    c.line(p1[0], (bar_y - 2.0) * MM, p1[0], (bar_y + 2.0) * MM)
    c.setFont("Helvetica", 8)
    c.drawCentredString((sx + 50.0) * MM, (bar_y + 3.0) * MM, "100 mm - measure with calipers")
    nx, ny, nw, nh = extras["note"]
    c.setFont("Helvetica-Bold", 10)
    c.drawString(nx * MM, (ny + 1.2) * MM, "Print at 100% / Actual size, do not fit to page")
    lx, ly, lw, lh = extras["label"]
    c.setFont("Helvetica-Bold", 12)
    c.drawString(lx * MM, (ly + 5.2) * MM, f"Cube {k}, IDs {6 * k}-{6 * k + 5}")
    c.setFont("Helvetica", 7.5)
    c.drawString(lx * MM, (ly + 1.0) * MM, info_line)
    c.showPage()
    return {"scale_bar_pt": (p0[0], p0[1], p1[0], p1[1]), "n_rects": n_rects}


def write_pdfs(args, layout, extras, dictionary, bits, dict_name) -> tuple[list, list]:
    info_line = f"{dict_name} | cube {args.cube_mm:g} mm | marker {args.marker_mm:g} mm"
    pw, ph = layout["page"]
    paths, infos = [], []
    for k in range(args.num_cubes):
        p = os.path.join(args.out_dir, f"cube_{k:02d}.pdf")
        c = rl_canvas.Canvas(p, pagesize=(pw * MM, ph * MM), invariant=1)
        c.setTitle(f"ArUco cube {k} (IDs {6 * k}-{6 * k + 5})")
        infos.append(draw_page(c, layout, extras, dictionary, bits, k, args, info_line))
        c.save()
        paths.append(p)
    p = os.path.join(args.out_dir, "all_cubes.pdf")
    c = rl_canvas.Canvas(p, pagesize=(pw * MM, ph * MM), invariant=1)
    c.setTitle("ArUco cubes (all)")
    for k in range(args.num_cubes):
        draw_page(c, layout, extras, dictionary, bits, k, args, info_line)
    c.save()
    return paths + [p], infos


# ----------------------------------------------------------------------------------------------- validation
RESULTS: list = []


def report(status: str, name: str, detail: str = "") -> None:
    RESULTS.append(status)
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def info(name: str, detail: str = "") -> None:
    print(f"  [INFO] {name}" + (f" - {detail}" if detail else ""))


def make_detector(dictionary, refine: str = "apriltag"):
    """AprilTag edge-fitting refinement is the one used for the counted checks: it is unbiased on anti-aliased edges."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = {"apriltag": cv2.aruco.CORNER_REFINE_APRILTAG, "subpix": cv2.aruco.CORNER_REFINE_SUBPIX}[refine]
    return cv2.aruco.ArucoDetector(dictionary, params)


def rot_xyz(ax, ay, az):
    a, b, g = np.radians([ax, ay, az])
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return rz @ ry @ rx


SYNTH_POSES = [(32, 38, 8), (-30, 36, -12), (36, -34, 20), (-35, -40, 5)]    # degrees; 3 faces visible each


def render_cube(dictionary, bits, cube, k_mat, rot, t, ss=3):
    """Anti-aliased synthetic view: every visible marker image is warped by the homography of its YAML corners."""
    w, h = SYNTH_SIZE
    img = np.full((h * ss, w * ss), 255, np.uint8)
    shown = []
    side_px = (bits + 2) * 32
    for m in cube["markers"]:
        n = FACE_TABLE[m["face"]][2]
        corners = np.array(m["corners_cube_m"], float)
        centre_cam = rot @ corners.mean(axis=0) + t
        cos_view = float(np.dot(rot @ n, -centre_cam / np.linalg.norm(centre_cam)))
        if cos_view <= 0:
            continue                                                 # back face
        cam = (rot @ corners.T).T + t
        uv = (k_mat @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        marker = cv2.aruco.generateImageMarker(dictionary, m["id"], side_px, borderBits=1)
        src = np.array([[-.5, -.5], [side_px - .5, -.5], [side_px - .5, side_px - .5], [-.5, side_px - .5]])
        dst = uv * ss + (ss - 1) / 2.0                               # pixel-centre convention at the supersampled scale
        hom, _ = cv2.findHomography(src, dst, 0)
        warped = cv2.warpPerspective(marker, hom, (w * ss, h * ss), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        img = np.minimum(img, warped)
        shown.append((m["id"], m["face"], cos_view))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), shown


def check_ids(data, dict_size):
    ids = [m["id"] for c in data["cubes"] for m in c["markers"]]
    ok = len(ids) == len(set(ids)) and all(0 <= i < dict_size for i in ids)
    report("PASS" if ok else "FAIL", f"1. {len(ids)} marker IDs unique and inside the dictionary (size {dict_size})",
           f"range {min(ids)}..{max(ids)}")


def check_handedness(data):
    try:
        assert_face_table()
        frames = derive_frames()
        for f in FACES:
            assert all(np.allclose(a, b) for a, b in zip(frames[f], FACE_TABLE[f])), f"folded net disagrees for {f}"
        # the YAML corners themselves: (TR - TL) = right, (TL - BL) = up, right x up = normal
        for c in data["cubes"]:
            for m in c["markers"]:
                co = np.array(m["corners_cube_m"])
                r, u, n = FACE_TABLE[m["face"]]
                assert np.allclose(co[1] - co[0], r * m["marker_side_m"]) and np.allclose(co[0] - co[3], u * m["marker_side_m"])
                assert np.allclose(np.cross(co[1] - co[0], co[0] - co[3]), n * m["marker_side_m"] ** 2)
        report("PASS", "2. right x up = normal for all 6 faces; folding the net in 3D gives the same table; YAML corners agree")
    except AssertionError as e:
        report("FAIL", "2. face orientation", str(e))


def solve_cube(cube, by_id):
    ms = [m for m in cube["markers"] if m["id"] in by_id]
    obj = np.concatenate([np.array(m["corners_cube_m"]) for m in ms])
    img_pts = np.concatenate([by_id[m["id"]] for m in ms])
    ok, rv, tv = cv2.solvePnP(obj, img_pts, SYNTH_K, None, flags=cv2.SOLVEPNP_ITERATIVE)      # ONE solvePnP, all corners
    rv, tv = cv2.solvePnPRefineLM(obj, img_pts, SYNTH_K, None, rv, tv)
    return ok, cv2.Rodrigues(rv)[0], tv.ravel(), 4 * len(ms)


def pose_errors(rot, t, r_est, t_est):
    cosang = np.clip((np.trace(rot.T @ r_est) - 1) / 2, -1, 1)
    return float(np.linalg.norm(t_est - t) * 1000.0), float(np.degrees(np.arccos(cosang)))


def check_synthetic(data, dictionary, bits):
    det, det_sub = make_detector(dictionary, "apriltag"), make_detector(dictionary, "subpix")
    for ci, cube in enumerate(data["cubes"]):
        rot = rot_xyz(*SYNTH_POSES[ci % len(SYNTH_POSES)])
        t = np.array([0.012, -0.008, 0.5])
        img, shown = render_cube(dictionary, bits, cube, SYNTH_K, rot, t)
        want = sorted(i for i, _, _ in shown)
        faces = ",".join(f for _, f, _ in shown)
        corners, ids, _ = det.detectMarkers(img)
        got = sorted(int(i) for i in ids.ravel()) if ids is not None else []
        if got != want or len(want) < 2:
            report("FAIL", f"3. cube {ci}: synthetic detection", f"drawn {want}, detected {got}")
            continue
        ok, r_est, t_est, n_pts = solve_cube(cube, {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)})
        terr, rerr = pose_errors(rot, t, r_est, t_est)
        good = ok and terr < 0.5 and rerr < 0.1
        report("PASS" if good else "FAIL", f"3. cube {ci}: faces {faces} (IDs {want}) detected, one solvePnP over {n_pts} corners",
               f"translation error {terr:.3f} mm (limit 0.5), rotation error {rerr:.4f} deg (limit 0.1)")
        c2, i2, _ = det_sub.detectMarkers(img)                       # same image, default corner refinement: informational
        if i2 is not None and sorted(int(i) for i in i2.ravel()) == want:
            _, r2, t2, _ = solve_cube(cube, {int(i): c.reshape(4, 2) for i, c in zip(i2.ravel(), c2)})
            te2, re2 = pose_errors(rot, t, r2, t2)
            info(f"3. cube {ci}: same image with CORNER_REFINE_SUBPIX instead", f"{te2:.3f} mm, {re2:.4f} deg (not counted)")


def pdf_checks(args, layout, extras, infos, pdf_paths, dictionary):
    try:
        import pymupdf as fz
    except ImportError:
        try:
            import fitz as fz
        except ImportError:
            fz = None
    # 4. scale bar: the drawn coordinates (generator) ...
    lens = [math.hypot(i["scale_bar_pt"][2] - i["scale_bar_pt"][0], i["scale_bar_pt"][3] - i["scale_bar_pt"][1]) / MM for i in infos]
    ok = all(abs(l - 100.0) < 1e-6 for l in lens)
    report("PASS" if ok else "FAIL", "4. scale bar length from the drawn coordinates", f"{min(lens):.6f}..{max(lens):.6f} mm on {len(lens)} pages")
    if fz is None:
        report("SKIP", "4b/5/6/7. PDF-level checks", "PyMuPDF not installed (pip install pymupdf)")
        return
    # ... and read back from the finished PDF
    doc = fz.open(pdf_paths[0])
    page = doc[0]
    found = 0
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l" and abs(math.hypot(it[2].x - it[1].x, it[2].y - it[1].y) - 100.0 * MM) < 1e-2 and abs(it[2].y - it[1].y) < 1e-3:
                found += 1
    report("PASS" if found == 1 else "FAIL", "4b. scale bar read back from cube_00.pdf (horizontal line, 100 mm = 283.465 pt)", f"{found} line(s) found")
    counts = [len(fz.open(p)) for p in pdf_paths]
    want = [1] * args.num_cubes + [args.num_cubes]
    report("PASS" if counts == want else "FAIL", "4c. page counts (one page per cube_NN.pdf, all_cubes.pdf has one page per cube)", str(counts))
    # 5. render and detect
    dpi = 300
    pix = page.get_pixmap(dpi=dpi, colorspace=fz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    corners, ids, _ = make_detector(dictionary, "apriltag").detectMarkers(img)
    got = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    report("PASS" if got == list(range(6)) else "FAIL", f"5. cube_00.pdf page 1 rendered at {dpi} dpi, markers detected", f"IDs {got}")
    # 5b. detected corner order/position on the printed page == the YAML convention (TL, TR, BR, BL; marker upright on the page)
    pw, ph = layout["page"]
    pxmm = dpi / 25.4
    worst = 0.0
    for c4, i in zip(corners, ids.ravel() if ids is not None else []):
        x0, y0, x1, y1 = layout["faces"][FACES[int(i)]]["rect"]
        cx, cy, hm = (x0 + x1) / 2, (y0 + y1) / 2, args.marker_mm / 2
        expect = np.array([(cx - hm, cy + hm), (cx + hm, cy + hm), (cx + hm, cy - hm), (cx - hm, cy - hm)])
        expect_px = np.stack([expect[:, 0] * pxmm, (ph - expect[:, 1]) * pxmm], axis=1) - 0.5     # pixel-centre convention
        worst = max(worst, float(np.abs(c4.reshape(4, 2) - expect_px).max() / pxmm))
    report("PASS" if worst < 0.15 and ids is not None and len(ids) == 6 else "FAIL",
           "5b. printed marker corners come out as TL, TR, BR, BL at the expected page positions (same convention as the YAML)",
           f"max deviation {worst:.3f} mm (limit 0.15)")
    # 6. white area inside every face stays clean (no lines / text inside the quiet zone)
    px = dpi / 25.4
    bad = 0
    for f, fd in layout["faces"].items():
        x0, y0, x1, y1 = fd["rect"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ix0, ix1 = int((x0 + 0.5) * px), int((x1 - 0.5) * px)
        iy0, iy1 = int((ph - y1 + 0.5) * px), int((ph - y0 - 0.5) * px)
        sub = img[iy0:iy1, ix0:ix1].copy()
        h = args.marker_mm / 2 + 0.25
        mx0, mx1 = int((cx - h) * px) - ix0, int((cx + h) * px) - ix0
        my0, my1 = int((ph - cy - h) * px) - iy0, int((ph - cy + h) * px) - iy0
        sub[max(my0, 0):my1, max(mx0, 0):mx1] = 255
        bad += int((sub < 250).sum())
    report("PASS" if bad == 0 else "FAIL", "6. every face is pure white outside its marker (0.5 mm inside the fold lines)", f"{bad} non-white pixels")
    # 7. vector only, pure black
    off = []
    for d in page.get_drawings():
        for key in ("fill", "color"):
            v = d.get(key)
            if v is not None and tuple(v) != (0.0, 0.0, 0.0):
                off.append((key, v))
    txt_off = [s["color"] for b in page.get_text("dict")["blocks"] if b["type"] == 0 for l in b["lines"] for s in l["spans"] if s["color"] != 0]
    n_img = len(page.get_images(full=True))
    filled = [d for d in page.get_drawings() if d.get("fill") is not None]
    n_fill, n_rect = len(filled), sum(len(d["items"]) for d in filled)
    ok = not off and not txt_off and n_img == 0 and n_fill >= 6 and n_rect >= 6 * 10
    report("PASS" if ok else "FAIL", "7. page is vector-only (no images) and every fill/stroke/text colour is exactly 0,0,0",
           f"{n_fill} filled paths made of {n_rect} rectangles, {n_img} images, {len(off)} non-black paints")


# ----------------------------------------------------------------------------------------------- main
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--num-cubes", type=int, default=4)
    ap.add_argument("--cube-mm", type=float, default=60.0)
    ap.add_argument("--marker-mm", type=float, default=45.0)
    ap.add_argument("--page", default="A4", choices=sorted(PAGES_MM), help="landscape (default A4)")
    ap.add_argument("--out-dir", default="aruco_cubes")
    ap.add_argument("--tab-mm", type=float, default=8.0)
    ap.add_argument("--margin-mm", type=float, default=8.0, help="minimum distance from every page edge")
    ap.add_argument("--marker-measured-mm", type=float, default=None,
                    help="caliper-measured black marker side: regenerate ONLY the YAML (cube_layout_measured.yaml), no PDFs")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    assert_face_table()
    dictionary, bits, dict_size = load_dictionary(args.dict)
    if args.num_cubes < 1 or 6 * args.num_cubes > dict_size:
        sys.exit(f"ERROR: {args.num_cubes} cubes need {6 * args.num_cubes} IDs but {args.dict} only has {dict_size}.")
    if args.cube_mm <= 0 or args.marker_mm <= 0:
        sys.exit("ERROR: sizes must be positive.")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.marker_measured_mm is not None:
        if not 0 < args.marker_measured_mm < args.cube_mm:
            sys.exit("ERROR: --marker-measured-mm must be positive and smaller than --cube-mm.")
        data = build_yaml_data(args.dict, args.cube_mm / 1000.0, args.marker_measured_mm / 1000.0, args.num_cubes, "measured")
        path = os.path.join(args.out_dir, "cube_layout_measured.yaml")
        write_yaml(path, data)
        print(f"YAML only (PDFs untouched): {path}  (marker side {args.marker_measured_mm:g} mm instead of nominal {args.marker_mm:g} mm)")
        print("Validation (checks that need no PDF):")
        data = yaml.safe_load(open(path))
        check_ids(data, dict_size)
        check_handedness(data)
        check_synthetic(data, dictionary, bits)
        return finish()

    module = args.marker_mm / (bits + 2)
    quiet = (args.cube_mm - args.marker_mm) / 2.0
    if args.marker_mm >= args.cube_mm or quiet < module - 1e-9:
        sys.exit(f"ERROR: white margin around the marker is {max(quiet, 0):.2f} mm but one marker module is {module:.2f} mm "
                 f"({args.marker_mm:g} mm / {bits + 2} modules). Use a smaller --marker-mm (<= {args.cube_mm * (bits + 2) / (bits + 4):.1f} mm "
                 f"for --cube-mm {args.cube_mm:g}) or a larger --cube-mm.")

    page_w, page_h = PAGES_MM[args.page]
    layout = build_layout(args.cube_mm, args.tab_mm, page_w, page_h, args.margin_mm)
    labels = [f"Cube {k}, IDs {6 * k}-{6 * k + 5}" for k in range(args.num_cubes)]
    info_line = f"{args.dict} | cube {args.cube_mm:g} mm | marker {args.marker_mm:g} mm"
    extras = place_extras(layout, labels, info_line, args.cube_mm)
    pdfs, infos = write_pdfs(args, layout, extras, dictionary, bits, args.dict)
    data = build_yaml_data(args.dict, args.cube_mm / 1000.0, args.marker_mm / 1000.0, args.num_cubes, "nominal")
    ypath = os.path.join(args.out_dir, "cube_layout_nominal.yaml")
    write_yaml(ypath, data)
    print(f"{args.num_cubes} cubes, {args.dict}, cube {args.cube_mm:g} mm, marker {args.marker_mm:g} mm "
          f"(module {module:.2f} mm, white margin {quiet:.2f} mm), page {args.page} landscape")
    print(f"glue tabs on cube edges of: {', '.join(layout['tab_desc'])}")
    print("wrote:", ", ".join(os.path.basename(p) for p in pdfs + [ypath]), "->", os.path.abspath(args.out_dir))

    print("Validation:")
    report("PASS", f"0. net with glue tabs fits the {args.page} landscape page with >= {args.margin_mm:g} mm margin; "
                   "no tab overlaps another tab or a face; label, note and scale bar placed outside the net")
    data = yaml.safe_load(open(ypath))
    check_ids(data, dict_size)
    check_handedness(data)
    check_synthetic(data, dictionary, bits)
    pdf_checks(args, layout, extras, infos, pdfs, dictionary)
    return finish()


def finish() -> int:
    p, f, s = RESULTS.count("PASS"), RESULTS.count("FAIL"), RESULTS.count("SKIP")
    print(f"Summary: {p} PASS, {f} FAIL, {s} SKIP")
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
