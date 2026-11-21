"""Gather a point cloud of an object with ArUco cubes as the world reference, and remember every camera pose.  (WSL)

    python gather_cloud_aruco.py --views ../scans/a.npz ../scans/b.npz ../scans/c.npz ../scans/d.npz
    python gather_cloud_aruco.py --views ... --target 0.15 0.0 0.03      # object position in the table frame (z up, 0 = table)
    python gather_cloud_aruco.py --views ... --recalibrate               # you moved the cubes

What it does
  1. finds the cube markers in the colour image of every view (capture the views with capture_view.py on Windows),
  2. world frame = frame of the first cube (cube 0 unless you pass --cubes 1 2: then cube 1). The pose of every other
     cube in that frame is estimated ONCE from the views that see both cubes (>= 2 markers of each) and then remembered
     in <out>/cube_layout_calibrated.yaml; the next run reuses it (until you pass --recalibrate),
  3. camera pose of each view = one solvePnP over all detected corners of all cubes -> T_world_cam, also stored
     w.r.t. every cube seen (T_cube_cam) in <out>/aruco_poses.json,
  4. back-projects the depth of every accepted view, cuts out the cubes, merges, finds the dominant plane (table / floor) and
     re-expresses everything in a TABLE FRAME (z up, z = 0 on the table, origin below the world cube), removes the plane
     (RANSAC), clusters (DBSCAN) and writes the object cluster. --roi and --target are given in the table frame.

Outputs in --out-dir (default output/aruco):
  aruco_poses.json               camera poses (world and per cube), cube poses, reprojection errors  <- the "memory"
  cube_layout_calibrated.yaml    the cube layout with pose_world of every cube filled in
  scene_cloud.ply                all accepted views merged (cubes cut out)
  object_cloud.ply               the isolated object (with normals)
  aruco_<view>.png               detected markers and cube axes drawn on every image

Nothing here has been run on a real L515 / printed cubes yet; selftest_aruco.py checks it on synthetic scenes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import HERE, backproject, inv_T, load_config, load_view, make_T, resolve, rot_angle_deg, seed_open3d, to_o3d

DEFAULT_LAYOUT = HERE.parent / "aruco_cubes_tool" / "aruco_cubes" / "cube_layout_nominal.yaml"
CALIBRATED_NAME = "cube_layout_calibrated.yaml"


# ---------------------------------------------------------------- layout
def load_layout(path):
    """-> (raw yaml dict, {cube_id: {"pose": 4x4 or None, "markers": {id: {"face", "corners" 4x3 (cube frame, m)}}}})."""
    raw = yaml.safe_load(Path(path).read_text())
    cubes = {}
    for c in raw["cubes"]:
        cubes[int(c["cube_id"])] = {
            "pose": None if c.get("pose_world") is None else np.array(c["pose_world"], float),
            "markers": {int(m["id"]): {"face": m["face"], "corners": np.array(m["corners_cube_m"], float)} for m in c["markers"]},
        }
    return raw, cubes


# ---------------------------------------------------------------- detection and pose
def make_detector(dict_name, refine="apriltag"):
    if not hasattr(cv2.aruco, dict_name):
        raise SystemExit(f"ERROR: unknown ArUco dictionary {dict_name} in the layout file")
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = {"apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
                                     "subpix": cv2.aruco.CORNER_REFINE_SUBPIX}[refine]
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name)), params)


def detect_markers(detector, view, known_ids):
    """-> ({marker id: 4x2 corners (TL,TR,BR,BL) in px}, [ids that are not in the layout])."""
    gray = cv2.cvtColor(view.color, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    found, unknown = {}, []
    for c, i in zip(corners, ids.ravel() if ids is not None else []):
        if int(i) in known_ids:
            found[int(i)] = c.reshape(4, 2).astype(float)
        else:
            unknown.append(int(i))
    return found, unknown


def make_detectors(dict_name, refine):
    """'auto': AprilTag-refined corners first (most accurate), plus markers only the sub-pixel detector finds (small/oblique ones)."""
    return [make_detector(dict_name, n) for n in (["apriltag", "subpix"] if refine == "auto" else [refine])]


def detect_markers_multi(detectors, view, known_ids):
    """-> (found, unknown, extra): extra = ids that only a later detector found."""
    found, unknown, extra = {}, set(), []
    for n, det in enumerate(detectors):
        f, u = detect_markers(det, view, known_ids)
        unknown |= set(u)
        for i, c in f.items():
            if i not in found:
                found[i] = c
                if n > 0:
                    extra.append(i)
    return found, sorted(unknown), extra


def _is_planar(obj, tol=1e-3):
    s = np.linalg.svd(obj - obj.mean(0), compute_uv=False)
    return bool(s[2] < tol * max(s[0], 1e-12))


def _rmse(obj, img, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def solve_pose(obj, img, K, dist):
    """PnP over all points. -> dict(T_cam_obj, rmse px, planar, ambiguity). ambiguity = 2nd best / best RMSE (planar only)."""
    obj, img = np.asarray(obj, float), np.asarray(img, float)
    if _is_planar(obj):
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if n == 0:
            return None
        cands = sorted(((_rmse(obj, img, r, t, K, dist), r, t) for r, t in zip(rvecs, tvecs)), key=lambda x: x[0])
        best = cands[0]
        amb = cands[1][0] / max(best[0], 1e-9) if len(cands) > 1 else np.inf
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, best[1], best[2])
        planar = True
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        planar, amb = False, np.inf
    return {"T": make_T(cv2.Rodrigues(rvec)[0], tvec), "rmse": _rmse(obj, img, rvec, tvec, K, dist),
            "planar": planar, "ambiguity": float(amb), "n_points": len(obj)}


def cube_pose_in_view(cube, dets, K, dist, min_markers=2):
    ids = [i for i in cube["markers"] if i in dets]
    if len(ids) < min_markers:
        return None
    r = solve_pose(np.concatenate([cube["markers"][i]["corners"] for i in ids]), np.concatenate([dets[i] for i in ids]), K, dist)
    if r:
        r["ids"] = ids
    return r


def world_pose_in_view(cubes_world, dets, K, dist, drop_px=3.0, min_markers=2):
    """One solvePnP over all detected markers of all cubes (corners in the world frame). Drops a marker that disagrees."""
    ids = [i for c in cubes_world.values() for i in c["markers"] if i in dets]
    corners_w = {i: (c["T"][:3, :3] @ m["corners"].T).T + c["T"][:3, 3] for c in cubes_world.values() for i, m in c["markers"].items()}
    dropped = []
    while True:
        if len(ids) < min_markers:
            return None
        r = solve_pose(np.concatenate([corners_w[i] for i in ids]), np.concatenate([dets[i] for i in ids]), K, dist)
        if r is None:
            return None
        if r["rmse"] <= drop_px or len(ids) <= max(min_markers, 2):
            r["ids"], r["dropped"] = list(ids), dropped
            return r
        # remove the marker with the largest residual and try again
        worst = max(ids, key=lambda i: _rmse(corners_w[i], dets[i], *_rt(r["T"]), K, dist))
        ids.remove(worst)
        dropped.append(worst)


def _rt(T):
    return cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1)


# ---------------------------------------------------------------- cube-to-cube calibration
def mean_transform(Ts):
    """Chordal mean of rigid transforms. -> (T, translation spread mm, rotation spread deg) (spread = max deviation)."""
    R = np.mean([T[:3, :3] for T in Ts], axis=0)
    u, _, vt = np.linalg.svd(R)
    R = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    dt = max(float(np.linalg.norm(T[:3, 3] - t)) for T in Ts) * 1000
    dr = max(rot_angle_deg(R.T @ T[:3, :3]) for T in Ts)
    return make_T(R, t), dt, dr


def calibrate_cubes(per_view, root=0):
    """per_view: list of {cube_id: T_cam_cube}. -> ({cube_id: T_world_cube}, {cube_id: info}); world = root cube."""
    edges = {}
    for vp in per_view:
        for a, b in itertools.combinations(sorted(vp), 2):
            edges.setdefault((a, b), []).append(inv_T(vp[a]) @ vp[b])            # pose of b in the frame of a
    poses, info = {root: np.eye(4)}, {root: {"source": "reference (identity)"}}
    changed = True
    while changed:
        changed = False
        for (a, b), lst in edges.items():
            if a in poses and b not in poses:
                T, dt, dr = mean_transform(lst)
                poses[b], info[b] = poses[a] @ T, {"source": f"estimated from {len(lst)} view(s) seeing cubes {a} and {b}",
                                                   "n_views": len(lst), "spread_mm": dt, "spread_deg": dr}
                changed = True
            elif b in poses and a not in poses:
                T, dt, dr = mean_transform(lst)
                poses[a], info[a] = poses[b] @ inv_T(T), {"source": f"estimated from {len(lst)} view(s) seeing cubes {a} and {b}",
                                                          "n_views": len(lst), "spread_mm": dt, "spread_deg": dr}
                changed = True
    return poses, info


def _to_vec(T):
    return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])


def _from_vec(x):
    return make_T(cv2.Rodrigues(np.asarray(x[:3], float).reshape(3, 1))[0], x[3:6])


def refine_cubes_ba(cubes, root, init_T, views, iters=40, huber_px=2.0):
    """Bundle adjustment of the cube poses (cube `root` fixed) and of the camera poses of `views` (list of dicts with
    dets, K, dist) over ALL detected marker corners. Uses the views that show markers of >= 2 cubes, also those with only
    one marker per cube, which the closed-form estimate cannot use. Plain numpy Levenberg-Marquardt, Huber loss.
    -> (T_world_cube dict, reprojection RMSE px) or None."""
    free = [k for k in cubes if k != root]
    world = {k: init_T[k] for k in cubes}
    cam0 = []
    for vw in views:
        r = world_pose_in_view({k: {"T": world[k], "markers": cubes[k]["markers"]} for k in cubes}, vw["dets"], vw["K"], vw["dist"], drop_px=1e9)
        if r is None:
            return None
        cam0.append(_to_vec(r["T"]))
    x = np.concatenate([_to_vec(init_T[k]) for k in free] + cam0)
    owner = {m: k for k in cubes for m in cubes[k]["markers"]}

    def residuals(x):
        T = dict(world)
        for j, k in enumerate(free):
            T[k] = _from_vec(x[6 * j:6 * j + 6])
        out = []
        for i, vw in enumerate(views):
            c = x[6 * len(free) + 6 * i: 6 * len(free) + 6 * i + 6]
            for m, img in vw["dets"].items():
                k = owner[m]
                pw = cubes[k]["markers"][m]["corners"] @ T[k][:3, :3].T + T[k][:3, 3]
                proj, _ = cv2.projectPoints(pw, c[:3].reshape(3, 1), c[3:].reshape(3, 1), vw["K"], vw["dist"])
                out.append((proj.reshape(-1, 2) - img).ravel())
        return np.concatenate(out)

    lam, r = 1e-3, residuals(x)
    cost = lambda r: float(np.sum(np.where(np.abs(r) <= huber_px, 0.5 * r ** 2, huber_px * (np.abs(r) - 0.5 * huber_px))))
    for _ in range(iters):
        w = np.sqrt(np.where(np.abs(r) <= huber_px, 1.0, huber_px / np.maximum(np.abs(r), 1e-12)))
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            d = np.zeros(len(x))
            d[j] = 1e-6
            J[:, j] = (residuals(x + d) - r) / 1e-6
        A, g = (J * w[:, None]).T @ (J * w[:, None]), (J * w[:, None]).T @ (r * w)
        for _ in range(10):
            dx = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            r_new = residuals(x + dx)
            if cost(r_new) < cost(r):
                x, r, lam = x + dx, r_new, max(lam / 3, 1e-9)
                break
            lam *= 4
        else:
            break
        if np.linalg.norm(dx) < 1e-9:
            break
    T = dict(world)
    for j, k in enumerate(free):
        T[k] = _from_vec(x[6 * j:6 * j + 6])
    return T, float(np.sqrt(np.mean(r ** 2) * 2))


# ---------------------------------------------------------------- point cloud
def inside_cube(points_w, T_w_cube, half, margin):
    p = (points_w - T_w_cube[:3, 3]) @ T_w_cube[:3, :3]                         # = R^T (p - t)
    return np.all(np.abs(p) < half + margin, axis=1)


def table_frame(scene, T_world_rootcube, cam_positions, plane_cfg):
    """Gravity-like frame from the dominant plane of the scene (the floor / table): z up (towards the cameras), z = 0 on the plane,
    origin below the centre of the world cube, x along the world cube's axis projected on the plane.
    -> (T_table_world 4x4: p_table = T @ p_world, info)."""
    plane, inl = scene.segment_plane(plane_cfg["distance_threshold"], plane_cfg["ransac_n"], plane_cfg["iterations"])
    n, d = np.array(plane[:3]), float(plane[3])
    n, d = n / np.linalg.norm(n), d / np.linalg.norm(np.array(plane[:3]))
    if n @ np.mean(cam_positions, axis=0) + d < 0:                              # make z point towards the cameras
        n, d = -n, -d
    c = T_world_rootcube[:3, 3]
    origin = c - (n @ c + d) * n
    for axis in (0, 1):
        ref = T_world_rootcube[:3, axis]
        x = ref - (ref @ n) * n
        if np.linalg.norm(x) > 0.3:
            break
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n, x), n])                                        # rows = table axes in world coordinates
    return make_T(R, -R @ origin), {"plane_world": [*n, d], "inliers": len(inl), "inlier_fraction": len(inl) / max(len(scene.points), 1)}


def floor_frame_from_cubes(cubes, cube_T, side, cam_positions, seen_ids, root):
    """Floor frame from the VISIBLE marker faces of the cubes, which are assumed to lie on the floor.
    z = direction from the marker towards the camera (outward normal of each cube's top marker face, averaged over the cubes);
    z = 0 is the floor = the top face plane minus one cube side; origin below the centre of cube `root`;
    x = the axis of cube `root` closest to horizontal. -> (T_floor_world, info)."""
    cam = np.mean(cam_positions, axis=0)
    tops, normals = {}, []
    for k, c in cubes.items():
        centre, R = cube_T[k][:3, 3], cube_T[k][:3, :3]
        cand = []
        for mid, m in c["markers"].items():
            co = m["corners"]
            n = np.cross(co[1] - co[0], co[0] - co[3])                           # outward normal of the marker (TL,TR,BR,BL seen from outside)
            cand.append((mid, m["face"], R @ (n / np.linalg.norm(n))))
        pool = [x for x in cand if x[0] in seen_ids] or cand                     # only faces whose marker was actually detected
        to_cam = (cam - centre) / np.linalg.norm(cam - centre)
        tops[k] = max(pool, key=lambda x: x[2] @ to_cam)
        normals.append(tops[k][2])
    n_up = np.mean(normals, axis=0)
    n_up /= np.linalg.norm(n_up)
    spread = max(float(np.degrees(np.arccos(np.clip(n @ n_up, -1, 1)))) for n in normals)
    heights = {k: float(n_up @ cube_T[k][:3, 3]) - side / 2 for k in cubes}      # floor height under each cube along n_up
    h = float(np.mean(list(heights.values())))
    c = cube_T[root][:3, 3]
    origin = c - (n_up @ c - h) * n_up
    for axis in (0, 1, 2):
        ref = cube_T[root][:3, axis]
        x = ref - (ref @ n_up) * n_up
        if np.linalg.norm(x) > 0.5:
            break
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n_up, x), n_up])
    return make_T(R, -R @ origin), {"plane_world": [*n_up, -h], "method": "cube top faces",
                                    "top_faces": {str(k): f"{v[1]} (marker id {v[0]})" for k, v in tops.items()},
                                    "top_normal_disagreement_deg": spread,
                                    "floor_height_disagreement_mm": (max(heights.values()) - min(heights.values())) * 1000}


def view_cloud(view, T_wc, cubes_world, side, cfg, margin):
    import open3d as o3d
    c = cfg["cloud"]
    pts, cols = backproject(view, c["max_depth"], c["min_depth"])
    pts = pts @ T_wc[:3, :3].T + T_wc[:3, 3]
    keep = np.ones(len(pts), bool)
    for cu in cubes_world.values():
        keep &= ~inside_cube(pts, cu["T"], side / 2.0, margin)
    pcd = to_o3d(pts[keep], cols[keep]).voxel_down_sample(c["voxel"])
    if len(pcd.points):
        pcd, _ = pcd.remove_statistical_outlier(c["sor_neighbors"], c["sor_std_ratio"])
    return pcd


def segment_object(scene, cfg, target, mode):
    """RANSAC table removal + DBSCAN. Returns (object cloud, clusters info list, chosen label) or raises SystemExit."""
    p, d = cfg["plane"], cfg["dbscan"]
    if len(scene.points) < 500:
        raise SystemExit("ERROR: the merged cloud has too few points - check the views, the ROI and the depth range.")
    plane, inl = scene.segment_plane(p["distance_threshold"], p["ransac_n"], p["iterations"])
    rest = scene.select_by_index(inl, invert=True)
    labels = np.array(rest.cluster_dbscan(eps=d["eps"], min_points=d["min_points"]))
    if (labels >= 0).sum() == 0:
        raise SystemExit("ERROR: DBSCAN found no cluster after removing the table - try a larger dbscan.eps in config.yaml.")
    pts = np.asarray(rest.points)
    ids = np.arange(labels.max() + 1)
    sizes = np.array([(labels == k).sum() for k in ids])
    cents = np.array([pts[labels == k].mean(0) for k in ids])
    info = [{"label": int(k), "n": int(sizes[k]), "centroid": cents[k].tolist()} for k in np.argsort(-sizes)]
    chosen = int(ids[np.argmax(sizes)]) if (mode == "largest" or target is None) else int(ids[np.argmin(np.linalg.norm(cents - target, axis=1))])
    obj = rest.select_by_index(np.where(labels == chosen)[0])
    t = cfg["target"]
    obj, _ = obj.remove_statistical_outlier(t["outlier_neighbors"], t["outlier_std_ratio"])
    return obj, info, chosen, np.array(plane), len(inl)


# ---------------------------------------------------------------- pipeline
def run(views_files, layout_path, out_dir, cfg, cube_ids=None, refine="auto", min_markers=2, max_rmse=2.0,
        roi=None, target=None, target_mode="between", margin_mm=5.0, recalibrate=False, segment=True, verbose=True, flip_z=False, floor_from="cubes"):
    """cube_ids: the cubes that are in the scene; the FIRST one is the world frame. None = every layout cube found in the images."""
    log = print if verbose else (lambda *a, **k: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, all_cubes = load_layout(layout_path)
    side = float(raw["cube_side_m"])
    detectors = make_detectors(raw["dictionary"], refine)
    known_all = {i: k for k, c in all_cubes.items() for i in c["markers"]}
    log(f"layout {layout_path}  ({raw['dictionary']}, cube {side * 1000:g} mm, marker {raw['marker_side_m'] * 1000:g} mm, "
        f"marker size source: {raw.get('marker_size_source', '?')})")

    # 1) detection of every layout marker in every view
    views = []
    for i, f in enumerate(views_files):
        f = Path(f)
        if not f.exists():
            raise SystemExit(f"ERROR: {f} not found")
        v = load_view(f, f"view{i}")
        dets, unknown, extra = detect_markers_multi(detectors, v, set(known_all))
        views.append({"view": v, "dets_all": dets, "unknown": unknown, "extra": extra})
    if cube_ids is None:
        cube_ids = tuple(sorted({known_all[i] for vw in views for i in vw["dets_all"]}))
        if not cube_ids:
            raise SystemExit("ERROR: no marker of the layout was found in any view (is --layout the file the cubes were printed from?)")
        log(f"cubes found in the images: {list(cube_ids)}")
    missing = [k for k in cube_ids if k not in all_cubes]
    if missing:
        raise SystemExit(f"ERROR: cube(s) {missing} are not in {layout_path}")
    root = int(cube_ids[0])
    cubes = {int(k): all_cubes[int(k)] for k in cube_ids}
    log(f"world frame = frame of cube {root} (origin at its centre)")
    known = {i for c in cubes.values() for i in c["markers"]}
    for i, vw in enumerate(views):
        v = vw["view"]
        vw["dets"] = {m: c for m, c in vw["dets_all"].items() if m in known}
        other = sorted(m for m in vw["dets_all"] if m not in known)
        per_cube = {k: cube_pose_in_view(c, vw["dets"], v.K, v.dist, 2) for k, c in cubes.items()}
        vw["cube_pose"] = {k: r for k, r in per_cube.items() if r}
        seen = ", ".join(f"cube {k}: ids {sorted(m for m in c['markers'] if m in vw['dets'])}" for k, c in cubes.items()
                         if any(m in vw["dets"] for m in c["markers"])) or "no marker"
        notes = []
        if vw["extra"]:
            notes.append(f"ids {sorted(vw['extra'])} only found by the sub-pixel detector")
        if other:
            notes.append(f"ids {other} belong to cubes that are not in --cubes " + str(sorted({known_all[m] for m in other if m in known_all})))
        log(f"[view{i}] {v.path.name}: {seen}" + (f"  ({'; '.join(notes)})" if notes else ""))

    # 2) cube poses: remembered, or estimated from co-visible views
    stored = {k: c["pose"] for k, c in cubes.items() if c["pose"] is not None}
    est, est_info = calibrate_cubes([{k: r["T"] for k, r in vw["cube_pose"].items()} for vw in views if len(vw["cube_pose"]) >= 2], root)
    use_stored = (not recalibrate) and int(raw.get("world_cube", 0)) == root and all(k in stored for k in cubes)
    ba_views = [{"dets": vw["dets"], "K": vw["view"].K, "dist": vw["view"].dist} for vw in views
                if len({kk for kk in cubes if any(m in vw["dets"] for m in cubes[kk]["markers"])}) >= 2]
    chk = {}
    if use_stored and len(cubes) > 1 and ba_views:
        # does the stored layout still explain these views? Same bundle adjustment as the calibration, started from the stored poses
        res_chk = refine_cubes_ba(cubes, root, stored, ba_views)
        chk = res_chk[0] if res_chk is not None else {}
    if not use_stored and len(cubes) > 1 and all(k in est for k in cubes):
        # refine with every view that shows markers of >= 2 cubes (also those with one marker per cube), then check how
        # stable the result is by leaving each view out in turn
        full = refine_cubes_ba(cubes, root, est, ba_views)
        if full is not None:
            jack = []
            for j in range(len(ba_views)):
                sub = refine_cubes_ba(cubes, root, est, ba_views[:j] + ba_views[j + 1:]) if len(ba_views) > 2 else None
                if sub is not None:
                    jack.append(sub[0])
            for k in cubes:
                if k == root:
                    continue
                d0 = inv_T(est[k]) @ full[0][k]
                spread = [inv_T(full[0][k]) @ T[k] for T in jack]
                est_info[k] = {"source": f"estimated from {len(ba_views)} view(s) with markers of several cubes (bundle adjustment, closed form "
                                         f"initialisation from {est_info[k].get('n_views', 0)} view(s))", "n_views": len(ba_views),
                               "ba_reproj_rmse_px": full[1], "change_vs_closed_form_mm": float(np.linalg.norm(d0[:3, 3]) * 1000),
                               "change_vs_closed_form_deg": rot_angle_deg(d0[:3, :3]),
                               "spread_mm": max([float(np.linalg.norm(x[:3, 3]) * 1000) for x in spread] or [float("nan")]),
                               "spread_deg": max([rot_angle_deg(x[:3, :3]) for x in spread] or [float("nan")])}
                est[k] = full[0][k]
    cube_T, cube_info = {}, {}
    for k in cubes:
        if use_stored:
            cube_T[k] = stored[k]
            cube_info[k] = {"source": "remembered from the layout file"}
            if k in chk and k != root:
                dT = inv_T(stored[k]) @ chk[k]
                d_mm, d_deg = float(np.linalg.norm(dT[:3, 3]) * 1000), rot_angle_deg(dT[:3, :3])
                cube_info[k].update(check_vs_these_views_mm=d_mm, check_vs_these_views_deg=d_deg)
                log(f"cube {k}: remembered pose vs these views: {d_mm:.1f} mm / {d_deg:.2f} deg" +
                    ("   <-- WARNING: the cubes were probably moved, use --recalibrate" if d_mm > 10 or d_deg > 3 else ""))
        elif k in est:
            cube_T[k], cube_info[k] = est[k], est_info[k]
            if k != root:
                log(f"cube {k}: pose in the cube-{root} frame {est_info[k]['source']}; spread between views "
                    f"{est_info[k]['spread_mm']:.2f} mm / {est_info[k]['spread_deg']:.2f} deg"
                    + (f"; bundle adjustment moved it {est_info[k]['change_vs_closed_form_mm']:.1f} mm / "
                       f"{est_info[k]['change_vs_closed_form_deg']:.2f} deg vs the closed-form estimate, reprojection RMSE "
                       f"{est_info[k]['ba_reproj_rmse_px']:.2f} px" if "ba_reproj_rmse_px" in est_info[k] else ""))
        else:
            best = {kk: max(len([m for m in cubes[kk]["markers"] if m in vw["dets"]]) for vw in views) for kk in cubes}
            raise SystemExit(f"ERROR: no pose for cube {k}. Calibrating the cubes against each other needs at least one view in which EVERY cube "
                             f"shows >= 2 faces at once. The most markers seen on one cube in a single view: "
                             + ", ".join(f"cube {kk}: {n}" for kk, n in best.items()) +
                             ". Capture views from lower (about 40-50 deg above the table, diagonally to the cubes) and closer "
                             "(< 0.6 m), so that a top and a side face of each cube are visible together.")
    cubes_world = {k: {"T": cube_T[k], "markers": cubes[k]["markers"]} for k in cubes}

    # 3) camera pose of every view w.r.t. all markers
    accepted = []
    for i, vw in enumerate(views):
        v = vw["view"]
        r = world_pose_in_view(cubes_world, vw["dets"], v.K, v.dist, min_markers=min_markers)
        reason = None
        if r is None:
            reason = f"fewer than {min_markers} usable markers"
        elif r["rmse"] > max_rmse:
            reason = f"reprojection RMSE {r['rmse']:.2f} px > {max_rmse:g} px"
        elif r["planar"] and r["ambiguity"] < 1.5:
            reason = f"all markers lie in one plane and the pose is ambiguous (alternative solution only {r['ambiguity']:.2f}x worse)"
        if reason:
            log(f"[view{i}] REJECTED: {reason}")
            vw["rejected"] = reason
            continue
        T_cw = r["T"]                                                           # world -> camera
        T_wc = inv_T(T_cw)
        vw.update(T_wc=T_wc, rmse=r["rmse"], used=r["ids"], dropped=r["dropped"])
        accepted.append(i)
        pos = T_wc[:3, 3]
        log(f"[view{i}] camera at ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m in the cube-{root} frame, reprojection RMSE "
            f"{r['rmse']:.2f} px, {len(r['ids'])} markers" + (f", dropped outlier markers {r['dropped']}" if r["dropped"] else ""))
        dbg = cv2.cvtColor(v.color, cv2.COLOR_RGB2BGR)
        for mid, c4 in vw["dets"].items():
            cv2.polylines(dbg, [c4.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
            cv2.putText(dbg, str(mid), tuple(c4[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        for k, cr in vw["cube_pose"].items():
            cv2.drawFrameAxes(dbg, v.K, v.dist, *_rt(cr["T"]), side * 0.8)
        cv2.imwrite(str(out_dir / f"aruco_{v.name}.png"), dbg)
    if not accepted:
        raise SystemExit("ERROR: no view gave a usable camera pose; see the messages above.")

    # 4) remember the poses
    poses = {"schema": 1, "created": datetime.now().isoformat(timespec="seconds"), "layout": str(layout_path),
             "dictionary": raw["dictionary"], "cube_side_m": side, "marker_side_m": raw["marker_side_m"],
             "world_cube": root,
             "frame": (f"world = frame of cube {root} (origin = centre of cube {root}). Camera frame = OpenCV (x right, y down, z forward). "
                       "T_world_cam maps camera coordinates to world; T_cube_cam maps camera coordinates to that cube's frame."),
             "cubes": {str(k): {"T_world_cube": cube_T[k].tolist(), **cube_info[k]} for k in cubes}, "views": {}}
    for i, vw in enumerate(views):
        v = vw["view"]
        e = {"file": str(v.path), "accepted": i in accepted}
        if i in accepted:
            e.update(T_world_cam=vw["T_wc"].tolist(), reproj_rmse_px=vw["rmse"], markers_used=vw["used"],
                     markers_dropped=vw["dropped"], camera_position_world_m=vw["T_wc"][:3, 3].tolist(), T_cube_cam={})
            for k in cubes:
                e["T_cube_cam"][str(k)] = (inv_T(cube_T[k]) @ vw["T_wc"]).tolist()      # camera pose w.r.t. that cube
        else:
            e["rejected_because"] = vw["rejected"]
        poses["views"][v.name] = e
    (out_dir / "aruco_poses.json").write_text(json.dumps(poses, indent=2))
    calibrated = dict(raw)
    calibrated["world_cube"] = root
    calibrated["cubes"] = [dict(c, pose_world=cube_T[int(c["cube_id"])].tolist() if int(c["cube_id"]) in cube_T else None) for c in raw["cubes"]]
    (out_dir / CALIBRATED_NAME).write_text(
        f"# Cube layout with the pose of every cube in the frame of cube {root} (world_cube) filled in by gather_cloud_aruco.py.\n"
        "# Reused automatically by the next run; pass --recalibrate after moving the cubes.\n" + yaml.safe_dump(calibrated, sort_keys=False))
    log(f"wrote {out_dir / 'aruco_poses.json'} and {out_dir / CALIBRATED_NAME}")

    # 5) fuse, then express everything in a table frame: z up, z = 0 on the table / floor
    import open3d as o3d
    clouds = [view_cloud(views[i]["view"], views[i]["T_wc"], cubes_world, side, cfg, margin_mm / 1000.0) for i in accepted]
    scene_w = o3d.geometry.PointCloud()
    for c in clouds:
        scene_w += c
    scene_w = scene_w.voxel_down_sample(cfg["cloud"]["voxel"])
    cams_w = [views[i]["T_wc"][:3, 3] for i in accepted]
    if floor_from == "cubes":
        seen_ids = {m for vw in views for m in vw["dets"]}
        T_tw, tinfo = floor_frame_from_cubes(cubes, cube_T, side, cams_w, seen_ids, root)
        log("floor frame from the cubes: +z = from the marker towards the camera; top faces used: "
            + ", ".join(f"cube {k}: {v}" for k, v in tinfo["top_faces"].items())
            + f"; z = 0 is the floor under the cubes (cube top face = {side * 1000:g} mm); origin below cube {root}")
        log(f"  the cubes' top normals differ by {tinfo['top_normal_disagreement_deg']:.1f} deg, their floor heights by "
            f"{tinfo['floor_height_disagreement_mm']:.1f} mm" + ("   <-- WARNING: large, is a cube not lying flat?"
                                                                   if tinfo["top_normal_disagreement_deg"] > 8 or tinfo["floor_height_disagreement_mm"] > 15 else ""))
    else:
        T_tw, tinfo = table_frame(scene_w, cube_T[root], cams_w, cfg["plane"])
        log(f"table frame from the dominant plane ({tinfo['inlier_fraction'] * 100:.0f}% of the points): +z towards the cameras, z = 0 on the plane")
    scene = o3d.geometry.PointCloud(scene_w).transform(T_tw)
    F = np.diag([1.0, -1.0, -1.0, 1.0]) if flip_z else np.eye(4)           # 180 deg about x: z (and y) inverted, no mirroring
    if roi is not None:
        lo, hi = np.array(roi[0::2]), np.array(roi[1::2])
        q = np.asarray(scene.points)
        scene = scene.select_by_index(np.where(np.all((q >= lo) & (q <= hi), axis=1))[0])
    o3d.io.write_point_cloud(str(out_dir / "scene_cloud.ply"), o3d.geometry.PointCloud(scene).transform(F))
    log(f"scene cloud: {len(scene.points)} points from {len(accepted)} view(s) (table frame{', z inverted' if flip_z else ''}) -> {out_dir / 'scene_cloud.ply'}")
    poses["table_frame"] = {"T_table_world": (F @ T_tw).tolist(), **tinfo, "z_inverted": bool(flip_z), "floor_from": floor_from,
                            "note": ("p_table = T_table_world @ p_world. scene_cloud.ply / object_cloud.ply are in this frame "
                                     + ("(z INVERTED: the table is z = 0 and things above it have negative z; rotated 180 deg about x, not mirrored)."
                                        if flip_z else "(z up, z = 0 on the table)."))}
    for k in cubes:
        poses["cubes"][str(k)]["T_table_cube"] = (F @ T_tw @ cube_T[k]).tolist()
    for i in accepted:
        poses["views"][views[i]["view"].name]["T_table_cam"] = (F @ T_tw @ views[i]["T_wc"]).tolist()
    (out_dir / "aruco_poses.json").write_text(json.dumps(poses, indent=2))
    consist = []
    for j in range(1, len(clouds)):
        if len(clouds[0].points) and len(clouds[j].points):
            r = o3d.pipelines.registration.evaluate_registration(clouds[j], clouds[0], 0.01, np.eye(4))
            consist.append({"view": views[accepted[j]]["view"].name, "fitness_10mm": float(r.fitness), "inlier_rmse_mm": float(r.inlier_rmse) * 1000})
            log(f"  consistency {views[accepted[j]]['view'].name} vs {views[accepted[0]]['view'].name}: {r.fitness:.2f} of the points within 10 mm, "
                f"inlier RMSE {r.inlier_rmse * 1000:.2f} mm (only meaningful where the views overlap)")
    result = {"poses": poses, "scene": scene, "consistency": consist, "cube_T": cube_T, "T_table_world": F @ T_tw}
    if not segment:
        return result

    # 6) object
    if target is None and target_mode == "between" and len(cube_T) > 1:
        mid = np.mean([T[:3, 3] for T in cube_T.values()], axis=0)
        target = T_tw[:3, :3] @ mid + T_tw[:3, 3]
        log(f"target = midpoint of the cubes {np.round(target, 3).tolist()} (change with --target X Y Z, table frame)")
    obj, clusters, chosen, plane, n_inl = segment_object(scene, cfg, target, target_mode)
    log(f"table plane (RANSAC): {np.round(plane, 3).tolist()}, {n_inl} points removed")
    for c in clusters[:6]:
        log(f"  cluster {c['label']:2d}: {c['n']:6d} pts, centroid ({c['centroid'][0]:+.3f}, {c['centroid'][1]:+.3f}, {c['centroid'][2]:+.3f})"
            + ("   <- chosen" if c["label"] == chosen else ""))
    obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    obj.orient_normals_towards_camera_location((T_tw @ views[accepted[0]]["T_wc"])[:3, 3])
    obj = o3d.geometry.PointCloud(obj).transform(F)                         # normals are rotated with the points
    o3d.io.write_point_cloud(str(out_dir / "object_cloud.ply"), obj)
    box = obj.get_minimal_oriented_bounding_box() if hasattr(obj, "get_minimal_oriented_bounding_box") else obj.get_oriented_bounding_box()
    log(f"object cloud: {len(obj.points)} points -> {out_dir / 'object_cloud.ply'}; minimal oriented box (m), sorted: "
        f"{np.round(np.sort(box.extent), 4).tolist()}, centre {np.round(box.center, 3).tolist()}")
    result.update(object=obj, clusters=clusters, chosen=chosen)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="+", required=True, help=".npz views from capture_view.py (any number >= 1)")
    ap.add_argument("--layout", help="cube_layout_*.yaml (default: <out>/cube_layout_calibrated.yaml if it exists, else aruco_cubes_tool's nominal)")
    ap.add_argument("--cubes", nargs="+", type=int, default=None,
                    help="cube ids in the scene; the FIRST is the world frame (default: every layout cube found in the images)")
    ap.add_argument("--out-dir", default=None, help="default: <paths.output_dir>/aruco")
    ap.add_argument("--config", help="config.yaml (cloud / plane / dbscan / target parameters are used)")
    ap.add_argument("--recalibrate", action="store_true", help="ignore the remembered cube poses and estimate them again")
    ap.add_argument("--refine", choices=["auto", "apriltag", "subpix"], default="auto",
                    help="marker corner refinement; auto = AprilTag-refined plus markers only the sub-pixel detector finds")
    ap.add_argument("--min-markers", type=int, default=2, help="markers needed for a camera pose (default 2)")
    ap.add_argument("--max-rmse", type=float, default=2.0, help="reject a view whose reprojection RMSE exceeds this (px)")
    ap.add_argument("--roi", nargs=6, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="crop box in the TABLE frame (m): x/y around the cube, z = height above the table (0 = table); cuts walls and clutter")
    ap.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), help="object position in the TABLE frame (m)")
    ap.add_argument("--target-mode", choices=["between", "largest"], default="between",
                    help="'between': cluster nearest to the midpoint of the cubes (or --target); 'largest': biggest cluster")
    ap.add_argument("--cube-margin-mm", type=float, default=5.0, help="points this close to a cube are cut out with it")
    ap.add_argument("--plane-threshold", type=float, help="RANSAC table plane distance threshold in m (config.yaml: plane.distance_threshold); "
                                                          "use 0.012-0.02 on carpet")
    ap.add_argument("--dbscan-eps", type=float, help="DBSCAN radius in m (config.yaml: dbscan.eps)")
    ap.add_argument("--floor-from", choices=["cubes", "plane"], default="cubes",
                    help="how the floor / up direction is defined: 'cubes' = the visible top marker faces of the cubes (z from marker to camera, "
                         "z = 0 one cube side below them); 'plane' = RANSAC plane of the point cloud")
    ap.add_argument("--flip-z", action="store_true",
                    help="write the clouds / poses with z inverted (rotation of 180 deg about x, no mirroring): the table stays z = 0, "
                         "things above it get negative z. --roi / --target still use height above the table")
    ap.add_argument("--no-segment", action="store_true", help="only write the fused scene, no table removal / clustering")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.plane_threshold:
        cfg["plane"]["distance_threshold"] = args.plane_threshold
    if args.dbscan_eps:
        cfg["dbscan"]["eps"] = args.dbscan_eps
    seed_open3d()
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg, cfg["paths"]["output_dir"]) / "aruco"
    layout = Path(args.layout) if args.layout else None
    if layout is None:
        remembered = out_dir / CALIBRATED_NAME
        layout = remembered if remembered.exists() and not args.recalibrate else DEFAULT_LAYOUT
        if layout == remembered:
            print(f"reusing the remembered cube layout {remembered}  (pass --recalibrate if you moved the cubes)")
    if not Path(layout).exists():
        print(f"ERROR: layout file {layout} not found - run aruco_cubes_tool/generate_aruco_cubes.py or pass --layout")
        return 1
    run([Path(v) for v in args.views], layout, out_dir, cfg, tuple(args.cubes) if args.cubes else None, args.refine, args.min_markers, args.max_rmse, args.roi,
        None if args.target is None else np.array(args.target), args.target_mode, args.cube_margin_mm,
        args.recalibrate, not args.no_segment, flip_z=args.flip_z, floor_from=args.floor_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
