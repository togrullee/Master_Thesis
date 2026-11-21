"""Synthetic RGB-D scenes with two ArUco cubes, a table and an object (KNOWN ground truth) for selftest_aruco.py.

Scene frame S: table plane z = 0, +z up. Each cube is a 60 mm box resting on the table in ANY orientation; its markers are
painted analytically from the marker geometry in cube_layout_*.yaml (so the test follows the same geometry file as the
real pipeline). Cameras use the OpenCV/RealSense convention (x right, y down, z forward), 640x480 like capture_view.py.
"""
import cv2
import numpy as np
import yaml

from common import make_T
from synthetic import K_L515_LIKE, look_at

FACES = ["+Z", "+X", "-Z", "-X", "+Y", "-Y"]


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def load_marker_geometry(layout_yaml):
    raw = yaml.safe_load(open(layout_yaml))
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, raw["dictionary"]))
    bits = int(d.markerSize)
    cubes = {}
    for c in raw["cubes"]:
        faces = {}
        for m in c["markers"]:
            co = np.array(m["corners_cube_m"], float)
            ms = float(m["marker_side_m"])
            bm = cv2.aruco.generateImageMarker(d, int(m["id"]), bits + 2, borderBits=1) < 128
            faces[m["face"]] = {"id": int(m["id"]), "right": (co[1] - co[0]) / ms, "up": (co[0] - co[3]) / ms, "bitmap": bm, "marker": ms}
        cubes[int(c["cube_id"])] = faces
    return float(raw["cube_side_m"]), bits, cubes


def default_scene(layout_yaml, cube1_shift=(0.0, 0.0), with_cube1=True):
    """Two cubes on the table (different yaw, cube 1 lying on another face) and an object between them."""
    side, bits, geo = load_marker_geometry(layout_yaml)
    cubes = [{"id": 0, "T": make_T(rot("z", 25), [0.0, 0.0, side / 2]), "faces": geo[0]}]
    if with_cube1:
        cubes.append({"id": 1, "T": make_T(rot("z", -20) @ rot("x", 90), [0.32 + cube1_shift[0], 0.05 + cube1_shift[1], side / 2]),
                      "faces": geo[1]})
    objects = [  # (pose, half extents, colour)
        (make_T(rot("z", 12), [0.16, 0.03, 0.025]), np.array([0.040, 0.020, 0.025]), (0.85, 0.35, 0.15)),   # main body 80x40x50
        (make_T(rot("z", 12), [0.17, 0.03, 0.065]), np.array([0.015, 0.015, 0.015]), (0.15, 0.45, 0.85)),   # block on top
        (make_T(rot("z", 12), [0.13, 0.045, 0.012]), np.array([0.025, 0.012, 0.012]), (0.25, 0.75, 0.30)),  # low bar
    ]
    return {"side": side, "bits": bits, "cubes": cubes, "objects": objects}


def camera_ring(target=(0.16, 0.03, 0.03), azimuths=(-90, -50, -130, 90, 50, 130), radius=0.50, elevation=50):
    el = np.radians(elevation)
    out = []
    for a in azimuths:
        p = np.array(target) + radius * np.array([np.cos(el) * np.cos(np.radians(a)), np.cos(el) * np.sin(np.radians(a)), np.sin(el)])
        out.append(look_at(p, target))
    return out


def _obb_hit(o_w, d_w, T, half):
    R, t = T[:3, :3], T[:3, 3]
    o = (o_w - t) @ R
    d = d_w @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    t_in, t_out = tmin.max(axis=1), tmax.min(axis=1)
    hit = (t_out >= t_in) & (t_out > 0)
    axis = tmin.argmax(axis=1)
    sgn = -np.sign(d[np.arange(len(d)), axis])                 # outward normal of the entered face (local frame)
    return np.where(hit, np.maximum(t_in, 0), np.inf), axis, sgn, o + d * np.maximum(t_in, 0)[:, None]


def render(T_S_cam, scene, noise_m=0.001, invalid_frac=0.02, seed=0, ss=3, gray_noise=1.5):
    rng = np.random.default_rng(seed)
    k = K_L515_LIKE
    W, H = k["W"], k["H"]
    R, o = T_S_cam[:3, :3], T_S_cam[:3, 3]
    half_cube = np.full(3, scene["side"] / 2)
    bits = scene["bits"]

    def cast(u, v):
        dirs = np.column_stack(((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], np.ones(u.size)))
        dw = dirs @ R.T
        n = u.size
        best = np.full(n, np.inf)
        ok = dw[:, 2] < -1e-9
        best[ok] = -o[2] / dw[ok, 2]
        kind, axis, sgn, ploc = np.full(n, -1), np.zeros(n, int), np.zeros(n), np.zeros((n, 3))
        items = [(c["T"], half_cube) for c in scene["cubes"]] + [(T, h) for T, h, _ in scene["objects"]]
        for i, (T, half) in enumerate(items):
            s, ax, sg, pl = _obb_hit(o, dw, T, half)
            m = s < best
            best[m], kind[m], axis[m], sgn[m], ploc[m] = s[m], i, ax[m], sg[m], pl[m]
        return best, kind, axis, sgn, ploc, o + dw * best[:, None]

    def shade(best, kind, axis, sgn, ploc, P):
        rgb = np.tile(np.array([0.55, 0.52, 0.48]), (best.size, 1))          # table
        for ci, cube in enumerate(scene["cubes"]):
            m = kind == ci
            if not m.any():
                continue
            rgb[m] = 0.97
            for ax in range(3):
                for sg in (-1, 1):
                    f = ("+" if sg > 0 else "-") + "XYZ"[ax]
                    mm = m & (axis == ax) & (sgn == sg)
                    if not mm.any():
                        continue
                    fd = cube["faces"][f]
                    a, b = ploc[mm] @ fd["right"], ploc[mm] @ fd["up"]
                    u = fd["marker"] / (bits + 2)
                    col, row = np.floor((a + fd["marker"] / 2) / u).astype(int), np.floor((fd["marker"] / 2 - b) / u).astype(int)
                    inside = (col >= 0) & (col < bits + 2) & (row >= 0) & (row < bits + 2)
                    black = np.zeros(mm.sum(), bool)
                    black[inside] = fd["bitmap"][row[inside], col[inside]]
                    rgb[np.where(mm)[0][black]] = 0.03
        for j, (_, _, c) in enumerate(scene["objects"]):
            m = kind == len(scene["cubes"]) + j
            rgb[m] = np.array(c)[None, :] * np.array([0.65, 0.8, 1.0])[axis[m]][:, None]
        rgb[~np.isfinite(best)] = 0.3
        return rgb

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    uu, vv = uu.ravel().astype(float), vv.ravel().astype(float)
    acc = np.zeros((uu.size, 3))
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    for dv in offs:
        for du in offs:
            acc += shade(*cast(uu + du, vv + dv))
    color = acc / ss ** 2 * 255 + rng.normal(0, gray_noise, (uu.size, 1))
    color = np.clip(color, 0, 255).astype(np.uint8).reshape(H, W, 3)
    s = cast(uu, vv)[0]
    depth = s.reshape(H, W).astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    depth += rng.normal(0, noise_m, depth.shape).astype(np.float32) * (depth > 0)
    depth[rng.random(depth.shape) < invalid_frac] = 0
    depth[depth > 1.5] = 0
    return dict(depth=depth, color=color, fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"])


def dist_to_objects(P, scene):
    """Distance of each point (scene frame) to the nearest object surface (the table and cubes are not objects)."""
    best = np.full(len(P), np.inf)
    for T, half, _ in scene["objects"]:
        q = (P - T[:3, 3]) @ T[:3, :3]
        outside = np.linalg.norm(np.maximum(np.abs(q) - half, 0), axis=1)
        inside = (half - np.abs(q)).min(axis=1)
        best = np.minimum(best, np.where(outside > 0, outside, inside))
    return best
