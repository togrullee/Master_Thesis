#!/usr/bin/env python3
"""Virtual print-and-fold test of the generated PDFs.

Paper cannot be printed here, so this script treats each PDF exactly like a printed sheet:
  * reads the PDF vectors only (dashed lines = folds, solid lines = cuts, filled paths = markers) and the 300 dpi raster,
  * detects the printed markers in the raster (IDs and corner order as PRINTED),
  * folds the sheet in 3D by rotating faces 90 deg about the dashed hinges, away from the viewer (print outside),
  * checks the folded cube (closes, print outside, every face where the YAML says it is, YAML corners = folded printed corners),
  * folds the glue tabs and checks where each one lands,
  * "photographs" the folded cube with the page raster as texture, detects the markers and recovers the pose with the YAML.
Nothing from the generator's layout code is used for the geometry; the YAML is only the thing being tested.

    python simulate_print_and_fold.py [--out-dir aruco_cubes] [--cube-mm 60 --marker-mm 45 --dict DICT_5X5_100]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np
import pymupdf
import yaml

import generate_aruco_cubes as g

MM = g.MM
RESULTS: list = []


def report(status, name, detail=""):
    RESULTS.append(status)
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def close(a, b, tol=0.02):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) < tol


# ----------------------------------------------------------------------------------------------- read the PDF
def read_sheet(pdf):
    page = pymupdf.open(pdf)[0]
    h_pt = page.rect.height

    def pt(p):
        return np.array([p.x / MM, (h_pt - p.y) / MM])

    sheet = {"page": page, "w_mm": page.rect.width / MM, "h_mm": page.rect.height / MM, "markers": [], "dashed": [], "solid": [], "extent": []}
    for d in page.get_drawings():
        if d.get("fill") is not None:
            r = d["rect"]
            sheet["markers"].append({"bbox": (r.x0 / MM, (h_pt - r.y1) / MM, r.x1 / MM, (h_pt - r.y0) / MM)})
            sheet["extent"].append(sheet["markers"][-1]["bbox"])
            continue
        dashed = d.get("dashes") not in (None, "", "[] 0")
        for it in d["items"]:
            assert it[0] == "l", it
            a, b = pt(it[1]), pt(it[2])
            (sheet["dashed"] if dashed else sheet["solid"]).append((a, b))
            sheet["extent"].append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
    for blk in page.get_text("dict")["blocks"]:
        if blk["type"] == 0:
            x0, y0, x1, y1 = blk["bbox"]
            sheet["extent"].append((x0 / MM, (h_pt - y1) / MM, x1 / MM, (h_pt - y0) / MM))
    return sheet


def raster(page, dpi):
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


# ----------------------------------------------------------------------------------------------- 3D folding
def rot_about(p, d, theta):
    d = np.asarray(d, float) / np.linalg.norm(d)
    k = np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])
    r = np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = np.asarray(p, float) - r @ np.asarray(p, float)
    return t


def hinge_fold(p0, p1, outward):
    """4x4 rotation of 90 deg about the hinge p0-p1 (page plane, z=0) that swings the side 'outward' BEHIND the page."""
    d = np.array([p1[0] - p0[0], p1[1] - p0[1], 0.0])
    o = np.array([outward[0], outward[1], 0.0])
    sign = -1.0 if np.cross(d / np.linalg.norm(d), o)[2] > 0 else 1.0    # want R*o = -z; R*o = d x o for +90 deg
    return rot_about([p0[0], p0[1], 0.0], d, sign * math.pi / 2)


def app(m, p2):
    return (m @ np.array([p2[0], p2[1], 0.0, 1.0]))[:3]


def fold_sheet(sheet, s_mm, scale=1.0):
    """Faces from the markers (square of side s around each marker centre), hinges from the dashed lines."""
    faces = [{"c": np.array([(m["bbox"][0] + m["bbox"][2]) / 2, (m["bbox"][1] + m["bbox"][3]) / 2])} for m in sheet["markers"]]
    hinges, tab_hinges = [], []
    for a, b in sheet["dashed"]:
        mid, ln = (a + b) / 2, np.linalg.norm(b - a)
        owners = [i for i, f in enumerate(faces) if abs(np.linalg.norm(mid - f["c"]) - s_mm / 2) < 0.02
                  and abs(np.dot((b - a) / ln, mid - f["c"])) < 1e-3]
        if abs(ln - s_mm) > 0.02:
            raise RuntimeError(f"dashed line of {ln:.3f} mm, expected {s_mm:g} mm")
        if len(owners) == 2:
            hinges.append((owners[0], owners[1], a, b))
        elif len(owners) == 1:
            tab_hinges.append((owners[0], a, b))
        else:
            raise RuntimeError("dashed line that belongs to no face")
    return faces, hinges, tab_hinges


def build_fold(faces, hinges, root):
    m = {root: np.eye(4)}
    todo = [root]
    while todo:
        i = todo.pop(0)
        for a, b, p0, p1 in hinges:
            if i not in (a, b):
                continue
            j = b if i == a else a
            if j in m:
                continue
            mid = (p0 + p1) / 2
            o = (faces[j]["c"] - mid)
            o /= np.linalg.norm(o)
            m[j] = m[i] @ hinge_fold(p0, p1, o)
            todo.append(j)
    assert len(m) == len(faces), "the faces are not all connected by fold lines"
    return m


# ----------------------------------------------------------------------------------------------- sheet / fold checks
def tab_polygon(p0, p1, o, solid):
    """The 4-cycle p0 -> p1 -> A -> B -> p0 of solid (cut) lines that leaves the face outward."""
    def nbrs(q):
        return [b if close(a, q) else a for a, b in solid if close(a, q) or close(b, q)]
    chains = []
    for a in nbrs(p1):
        if np.dot(o, a - p1) <= 1e-3:
            continue
        for b in nbrs(a):
            if not close(b, p1) and any(close(q, p0) for q in nbrs(b)) and np.dot(o, b - p0) > 1e-3:
                chains.append([p0, p1, a, b])
    if len(chains) != 1:
        raise RuntimeError(f"cannot trace the tab outline ({len(chains)} candidates)")
    return chains[0]


def analyse(pdf, args, dictionary, bits):
    name = os.path.basename(pdf)
    print(f"\n=== {name} ===")
    sheet = read_sheet(pdf)
    page = sheet["page"]
    wmm, hmm = sheet["w_mm"], sheet["h_mm"]
    report("PASS" if abs(wmm - 297) < 0.1 and abs(hmm - 210) < 0.1 else "FAIL", "P1. sheet is A4 landscape", f"{wmm:.2f} x {hmm:.2f} mm")
    ext = np.array(sheet["extent"])
    dist = min(ext[:, 0].min(), ext[:, 1].min(), wmm - ext[:, 2].max(), hmm - ext[:, 3].max())
    report("PASS" if dist >= args.margin_mm - 0.05 else "FAIL", f"P2. everything printed (lines, markers, text) is >= {args.margin_mm:g} mm from every sheet edge",
           f"closest element {dist:.2f} mm")
    sizes = [(m["bbox"][2] - m["bbox"][0], m["bbox"][3] - m["bbox"][1]) for m in sheet["markers"]]
    ok = len(sizes) == 6 and all(abs(w - args.marker_mm) < 0.02 and abs(h - args.marker_mm) < 0.02 for w, h in sizes)
    report("PASS" if ok else "FAIL", f"P3. 6 printed markers, black square {args.marker_mm:g} mm, module {args.marker_mm / (bits + 2):.3f} mm",
           f"measured {min(min(s) for s in sizes):.3f}..{max(max(s) for s in sizes):.3f} mm")
    bars = [(a, b) for a, b in sheet["solid"] if abs(a[1] - b[1]) < 1e-4 and abs(np.linalg.norm(b - a) - 100.0) < 0.01]
    report("PASS" if len(bars) == 1 else "FAIL", "P4. one 100 mm scale bar in the vectors", f"{len(bars)} found")

    # marker identity and corner order AS PRINTED, from the 300 dpi raster
    dpi = 300
    img = raster(page, dpi)
    corners, ids, _ = g.make_detector(dictionary, "apriltag").detectMarkers(img)
    pxmm = dpi / 25.4
    found = {}
    for c4, i in zip(corners, ids.ravel() if ids is not None else []):
        pts = c4.reshape(4, 2)
        found[int(i)] = np.stack([(pts[:, 0] + 0.5) / pxmm, hmm - (pts[:, 1] + 0.5) / pxmm], axis=1)
    faces, hinges, tab_hinges = fold_sheet(sheet, args.cube_mm)
    report("PASS" if len(faces) == 6 and len(hinges) == 5 and len(tab_hinges) == 7 else "FAIL",
           f"F0. read from the PDF: 6 faces, {len(hinges)} face-to-face folds, {len(tab_hinges)} glue-tab folds; side of a face = {args.cube_mm:g} mm")
    if len(found) != 6:
        report("FAIL", "F0b. all 6 printed markers detected", f"{sorted(found)}")
        return None
    for i, f in enumerate(faces):                                   # which marker is on which flat face
        f["id"] = min(found, key=lambda k: np.linalg.norm(found[k].mean(0) - f["c"]))
    root = min(range(6), key=lambda i: faces[i]["id"])               # lowest ID = the +Z face of the specification (stays in place)
    mats = build_fold(faces, hinges, root)
    centre = np.array([faces[root]["c"][0], faces[root]["c"][1], -args.cube_mm / 2])   # cube sits behind the sheet
    s = args.cube_mm

    names, outward, closed, dev_close = {}, True, True, 0.0
    for i, f in enumerate(faces):
        r = mats[i][:3, :3]
        n = r @ np.array([0, 0, 1.0])
        cen = app(mats[i], f["c"]) - centre
        ax = int(np.argmax(np.abs(n)))
        names[i] = ("+" if n[ax] > 0 else "-") + "XYZ"[ax]
        outward &= bool(np.dot(n, cen) > 0.99 * s / 2)
        dev_close = max(dev_close, float(np.abs(cen - n * s / 2).max()))
        closed &= bool(np.allclose(cen, n * s / 2, atol=0.01))
        f["n"], f["cen"] = n, cen
    report("PASS" if sorted(names.values()) == sorted(g.FACES) else "FAIL",
           "F1. folded, the six faces occupy +X, -X, +Y, -Y, +Z, -Z exactly once",
           ", ".join(f"ID {faces[i]['id']}->{names[i]}" for i in sorted(names, key=lambda k: faces[k]['id'])))
    report("PASS" if closed else "FAIL", "F2. folded face centres sit exactly half a cube side from the centre (the cube closes)",
           f"max deviation {dev_close:.4f} mm (limit 0.01; PDF coordinate rounding)")
    report("PASS" if outward else "FAIL", "F3. the printed side of every face points OUT of the cube")

    cubes = yaml.safe_load(open(os.path.join(args.out_dir, "cube_layout_nominal.yaml")))["cubes"]
    k = faces[root]["id"] // 6
    cube = cubes[k]
    worst, bad_face = 0.0, []
    for i, f in enumerate(faces):
        ym = next(m for m in cube["markers"] if m["id"] == f["id"])
        if ym["face"] != names[i]:
            bad_face.append((f["id"], ym["face"], names[i]))
        folded = np.array([app(mats[i], p) - centre for p in found[f["id"]]])
        worst = max(worst, float(np.abs(folded - np.array(ym["corners_cube_m"]) * 1000).max()))
    report("PASS" if not bad_face else "FAIL", "F4. every printed marker lands on the face the YAML says", f"mismatches: {bad_face}" if bad_face else "6/6")
    report("PASS" if worst < 0.15 else "FAIL", "F5. folded PRINTED marker corners (TL,TR,BR,BL as printed) equal the YAML corners",
           f"max deviation {worst:.3f} mm (limit 0.15; detection noise at 300 dpi)")

    # glue tabs
    solid = [(a, b) for a, b in sheet["solid"] if not (abs(a[1] - b[1]) < 1e-4 and abs(np.linalg.norm(b - a) - 100.0) < 0.01)]
    tabs, tab_ok, planes = [], True, {}
    edges = set()
    for ia, ib, a, b in hinges:
        edges.add(tuple(np.round(app(mats[ia], (a + b) / 2) - centre, 1) + 0.0))
    for fi, p0, p1 in tab_hinges:
        mid = (p0 + p1) / 2
        o = (mid - faces[fi]["c"]) / np.linalg.norm(mid - faces[fi]["c"])
        poly = tab_polygon(p0, p1, o, solid)
        mt = mats[fi] @ hinge_fold(p0, p1, o)
        v3 = np.array([app(mt, p) - centre for p in poly])
        n_t = mt[:3, :3] @ np.array([0, 0, 1.0])
        ax = int(np.argmax(np.abs(n_t)))
        side = np.sign(n_t[ax])
        coplanar = np.allclose(v3[:, ax], side * s / 2, atol=0.01)
        other = [a2 for a2 in range(3) if a2 != ax]
        inside = bool(np.all(np.abs(v3[:, other]) <= s / 2 + 0.01))
        partner = ("+" if side > 0 else "-") + "XYZ"[ax]
        edge = tuple(np.round(app(mats[fi], mid) - centre, 1) + 0.0)
        glue_ok = bool(np.dot(n_t, np.eye(3)[ax] * side) > 0.999)            # front side of the tab touches the partner's back
        tab_ok &= coplanar and inside and glue_ok and edge not in edges
        edges.add(edge)
        tabs.append({"face": names[fi], "partner": partner, "poly2d": [tuple(p) for p in v3[:, other]], "plane": partner,
                     "coplanar": coplanar, "inside": inside, "glue": glue_ok})
    n_edges = len(edges)
    overlap = False
    for p in {t["plane"] for t in tabs}:
        grp = [t["poly2d"] for t in tabs if t["plane"] == p]
        overlap |= any(g.convex_overlap(grp[i], grp[j]) for i in range(len(grp)) for j in range(i + 1, len(grp)))
    report("PASS" if tab_ok and n_edges == 12 else "FAIL",
           "T1. every glue tab folds into the plane of its neighbour face, inside that face's outline, front (blank) side against the neighbour's back; "
           "5 folds + 7 tab edges cover all 12 cube edges exactly once",
           "; ".join(f"{t['face']}->{t['partner']}" for t in tabs) + f"  | distinct cube edges covered: {n_edges}")
    report("PASS" if not overlap else "FAIL", "T2. no two glue tabs overlap each other once folded")
    return {"sheet": sheet, "faces": faces, "mats": mats, "centre": centre, "names": names, "cube": cube, "k": k,
            "raster_dpi": dpi, "page": page}


# ----------------------------------------------------------------------------------------------- virtual photo
def photograph(res, args, dictionary, bits, pose_deg, scale=1.0, ss=3, dpi=150):
    """Fold the sheet, put it at 0.5 m in front of a synthetic 1920x1080 camera and render it with the PDF raster as texture."""
    tex = raster(res["page"], dpi)
    pxmm = dpi / 25.4
    hmm = res["sheet"]["h_mm"]
    w, h = g.SYNTH_SIZE
    rot = g.rot_xyz(*pose_deg)
    t = np.array([0.012, -0.008, 0.5])
    img = np.full((h * ss, w * ss), 255, np.uint8)
    s = args.cube_mm
    shown = []
    for i, f in enumerate(res["faces"]):
        flat = [f["c"] + np.array(d) * s / 2 for d in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
        cube_pts = np.array([(app(res["mats"][i], p) - res["centre"]) for p in flat]) / 1000.0 * scale   # metres, physical size
        cam = (rot @ cube_pts.T).T + t
        n_cam = rot @ f["n"]
        to_cam = -(cam.mean(0)) / np.linalg.norm(cam.mean(0))
        if float(np.dot(n_cam, to_cam)) <= 0:
            continue
        uv = (g.SYNTH_K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        src = np.array([[p[0] * pxmm - 0.5, (hmm - p[1]) * pxmm - 0.5] for p in flat])
        dst = uv * ss + (ss - 1) / 2.0
        hom, _ = cv2.findHomography(src, dst, 0)
        warped = cv2.warpPerspective(tex, hom, (w * ss, h * ss), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        mask = np.zeros((h * ss, w * ss), np.uint8)
        cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255, lineType=cv2.LINE_AA)
        img = np.where(mask > 127, warped, img)
        shown.append((f["id"], res["names"][i]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), shown, rot, t


def pose_from_photo(res, args, dictionary, bits, yaml_cube, pose_deg, scale=1.0):
    img, shown, rot, t = photograph(res, args, dictionary, bits, pose_deg, scale)
    corners, ids, _ = g.make_detector(dictionary, "apriltag").detectMarkers(img)
    got = sorted(int(i) for i in ids.ravel()) if ids is not None else []
    want = sorted(i for i, _ in shown)
    if got != want or len(want) < 2:
        return img, shown, None, f"drawn {want}, detected {got}"
    ok, r_est, t_est, n_pts = g.solve_cube(yaml_cube, {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)})
    terr, rerr = g.pose_errors(rot, t, r_est, t_est)
    return img, shown, (terr, rerr, n_pts), ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="aruco_cubes")
    ap.add_argument("--dict", default="DICT_5X5_100")
    ap.add_argument("--cube-mm", type=float, default=60.0)
    ap.add_argument("--marker-mm", type=float, default=45.0)
    ap.add_argument("--margin-mm", type=float, default=8.0)
    ap.add_argument("--print-scale", type=float, default=0.98, help="printer scaling used for the 'printed too small' scenario")
    args = ap.parse_args(argv)
    dictionary, bits, _ = g.load_dictionary(args.dict)
    pdfs = sorted(glob.glob(os.path.join(args.out_dir, "cube_[0-9][0-9].pdf")))
    if not pdfs:
        sys.exit(f"no cube_NN.pdf in {args.out_dir}; run generate_aruco_cubes.py first")
    shots = os.path.join(args.out_dir, "virtual_photos")
    os.makedirs(shots, exist_ok=True)
    first = None
    for pdf in pdfs:
        res = analyse(pdf, args, dictionary, bits)
        if res is None:
            continue
        first = first or res
        k = res["k"]
        img, shown, out, err = pose_from_photo(res, args, dictionary, bits, res["cube"], g.SYNTH_POSES[k % len(g.SYNTH_POSES)])
        cv2.imwrite(os.path.join(shots, f"cube_{k:02d}_folded.png"), img)
        faces = ",".join(n for _, n in shown)
        if out is None:
            report("FAIL", f"V1. virtual photo of the folded cube (faces {faces})", err)
        else:
            terr, rerr, n_pts = out
            report("PASS" if terr < 0.5 and rerr < 0.1 else "FAIL",
                   f"V1. folded PDF raster photographed at 0.5 m (faces {faces} visible): markers detected, one solvePnP over {n_pts} corners with the nominal YAML",
                   f"translation error {terr:.3f} mm (limit 0.5), rotation error {rerr:.4f} deg (limit 0.1)")
    if first is not None:                                           # printer scaling scenario on cube 0
        f = args.print_scale
        k = first["k"]
        pose = g.SYNTH_POSES[k % len(g.SYNTH_POSES)]
        _, _, nom, e1 = pose_from_photo(first, args, dictionary, bits, first["cube"], pose, scale=f)
        meas = g.build_yaml_data(args.dict, args.cube_mm * f / 1000, args.marker_mm * f / 1000, k + 1, "measured")["cubes"][k]
        _, _, mea, e2 = pose_from_photo(first, args, dictionary, bits, meas, pose, scale=f)
        if nom and mea:
            print(f"\n=== printer scaling {f * 100:g}% (everything printed {100 - f * 100:g}% too small) ===")
            report("INFO", "S1. nominal YAML on that print", f"translation error {nom[0]:.2f} mm, rotation {nom[1]:.3f} deg  (scale mismatch: why you must check the 100 mm bar)")
            report("PASS" if mea[0] < 0.5 and mea[1] < 0.1 else "FAIL",
                   f"S2. YAML regenerated from the measured sizes (marker {args.marker_mm * f:.2f} mm, cube {args.cube_mm * f:.2f} mm)",
                   f"translation error {mea[0]:.3f} mm, rotation {mea[1]:.4f} deg")
            RESULTS.remove("INFO")
        else:
            report("FAIL", "S. scaling scenario", f"{e1} {e2}")
    p, fl = RESULTS.count("PASS"), RESULTS.count("FAIL")
    print(f"\nSummary: {p} PASS, {fl} FAIL")
    return 1 if fl else 0


if __name__ == "__main__":
    sys.exit(main())
