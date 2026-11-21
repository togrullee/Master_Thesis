"""Estimate the camera pose of every view from the printed chessboard and write poses.json.  (WSL)

    python calibrate_poses.py                                   # scans/view0.npz view1.npz view2.npz
    python calibrate_poses.py --views scans/a.npz scans/b.npz scans/c.npz

World frame = board frame. Origin = the inner corner OpenCV returns as the LAST row's first corner
(index (rows-1)*cols). For an odd x even pattern OpenCV anchors this to a fixed PHYSICAL corner of the
printed board (it does not depend on how you hold the camera), so all views share one frame.
+X runs along the columns, +Y along the rows towards corner 0's row, +Z out of the board.
calib_viewN.png marks the origin with a red circle - put a pen mark on that corner of the paper.
The view named viewN in poses.json is the N-th file given to --views.
"""
import argparse
import json
import sys
from datetime import datetime

import cv2
import numpy as np

from common import (inv_T, load_config, load_view, make_T, meshcat_triad, PALETTE, resolve,
                    rot_angle_deg, seed_open3d, start_meshcat, to_o3d, wait_for_user, backproject)


def board_object_points(cols, rows, square_m):
    """3D corner coordinates in the board frame, in OpenCV detection order (row-major)."""
    pts = np.zeros((rows * cols, 3))
    for r in range(rows):
        for c in range(cols):
            pts[r * cols + c] = (c * square_m, (rows - 1 - r) * square_m, 0.0)
    return pts


def detect_corners(gray, cols, rows):
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags)
    if not ok:                                         # fall back to the classic detector
        ok, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
    return ok, (corners.reshape(-1, 2).astype(np.float64) if ok else None)


def solve_pose(obj, img, K, dist):
    """Return (T_cam_board, reprojection RMSE in px)."""
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rmse = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec), rmse


def depth_plane_of_board(view, corners_px, cfg_cloud):
    """Fit a plane to the depth pixels inside the detected pattern. Returns (n, d) with n.x = d, or None."""
    import open3d as o3d
    mask = np.zeros(view.depth.shape, np.uint8)
    hull = cv2.convexHull(corners_px.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 1)
    masked = view.depth * mask
    sub = type(view)(view.name, view.path, masked, view.color, view.K, view.dist)
    pts, _ = backproject(sub, cfg_cloud["max_depth"], cfg_cloud["min_depth"])
    if len(pts) < 300:
        return None
    plane, inl = to_o3d(pts).segment_plane(0.003, 3, 500)
    if len(inl) < 200:
        return None
    n, d = np.array(plane[:3]), -plane[3]              # open3d: n.x + a = 0  ->  n.x = -a
    return n / np.linalg.norm(n), d


def refine_with_depth_plane(T_cam_board, n_depth, d_depth):
    """Tilt the board so its normal matches the depth plane and put its origin on that plane."""
    R, t = T_cam_board[:3, :3], T_cam_board[:3, 3]
    n_pnp = R[:, 2]
    n = n_depth if n_depth @ n_pnp > 0 else -n_depth
    d = d_depth if n_depth @ n_pnp > 0 else -d_depth
    v = np.cross(n_pnp, n)
    s, c = np.linalg.norm(v), float(n_pnp @ n)
    if s < 1e-9:
        R_new = R
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_new = (np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)) @ R
    t_new = t * (d / (n @ t))                          # slide origin along its viewing ray onto the plane
    return make_T(R_new, t_new)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="+", help=".npz files in order: view0 view1 view2 ...")
    ap.add_argument("--config")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_open3d()
    scans = resolve(cfg, cfg["paths"]["scans_dir"])
    files = [resolve(cfg, f) if args.views else scans / f"view{i}.npz"
             for i, f in enumerate(args.views or ["", "", ""])]
    out_dir = resolve(cfg, cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    b = cfg["board"]
    cols, rows, sq = int(b["cols"]), int(b["rows"]), float(b["square_size_mm"]) / 1000.0
    obj = board_object_points(cols, rows, sq)

    results, failed = {}, []
    for i, f in enumerate(files):
        name = f"view{i}"
        if not f.exists():
            print(f"[{name}] ERROR: {f} not found")
            failed.append(name)
            continue
        view = load_view(f, name)
        gray = cv2.cvtColor(view.color, cv2.COLOR_RGB2GRAY)
        ok, corners = detect_corners(gray, cols, rows)
        dbg = cv2.cvtColor(view.color, cv2.COLOR_RGB2BGR)
        if not ok:
            print(f"[{name}] ERROR: chessboard {cols}x{rows} inner corners not found in {f.name}. "
                  "Check board.cols/rows in config.yaml, that the whole board is visible, in focus "
                  "and evenly lit.")
            cv2.imwrite(str(out_dir / f"calib_{name}_NOT_FOUND.png"), dbg)
            failed.append(name)
            continue

        T_cb, rmse = solve_pose(obj, corners, view.K, view.dist)
        T_wc = inv_T(T_cb)                             # camera pose in the world (= board) frame
        info = {"file": str(f), "reproj_rmse_px": rmse}

        plane = depth_plane_of_board(view, corners, cfg["cloud"])
        if plane is not None:
            n_d, d_d = plane
            n_pnp = T_cb[:3, 2]
            tilt = np.degrees(np.arccos(np.clip(abs(n_d @ n_pnp), 0, 1)))
            off = abs(n_d @ T_cb[:3, 3] - d_d)         # board origin to depth plane, metres
            info.update(depth_plane_tilt_deg=float(tilt), depth_plane_offset_mm=float(off * 1000))
            if b.get("refine_with_depth"):
                T_cb = refine_with_depth_plane(T_cb, n_d, d_d)
                T_wc = inv_T(T_cb)
                info["refined_with_depth"] = True

        cam_pos = T_wc[:3, 3]
        z_toward_cam = float((T_wc[:3, 3] @ np.array([0, 0, 1.0])) > 0)
        msg = (f"[{name}] {f.name}: reprojection RMSE {rmse:.2f} px, camera at "
               f"({cam_pos[0]:+.3f}, {cam_pos[1]:+.3f}, {cam_pos[2]:+.3f}) m")
        if plane is not None:
            msg += (f", depth-plane tilt {info['depth_plane_tilt_deg']:.2f} deg, "
                    f"offset {info['depth_plane_offset_mm']:.1f} mm")
        print(msg)
        if plane is not None and info["depth_plane_tilt_deg"] > 3.0:
            print(f"[{name}] NOTE: the depth inside the board disagrees with the chessboard pose by "
                  f"{info['depth_plane_tilt_deg']:.1f} deg. On a printed board this is usually L515 depth bias on the black "
                  "squares (or a bent sheet), not a pose error, so refine_with_depth stays off. Verify later: "
                  "fuse_and_segment.py must find a HORIZONTAL table plane in the fused cloud.")
        if rmse > 1.0:
            print(f"[{name}] WARNING: reprojection error > 1 px - is square_size / cols / rows right, "
                  "and is the board flat?")
        if not z_toward_cam:
            print(f"[{name}] WARNING: camera is BELOW the board plane (z<0): corner ordering is flipped "
                  "or the board pattern is symmetric (use an odd x even corner count).")

        cv2.drawChessboardCorners(dbg, (cols, rows), corners.astype(np.float32).reshape(-1, 1, 2), True)
        cv2.circle(dbg, tuple(int(x) for x in corners[(rows - 1) * cols]), 9, (0, 0, 255), 2)  # origin
        cv2.imwrite(str(out_dir / f"calib_{name}.png"), dbg)
        info["T_world_cam"] = T_wc.tolist()
        results[name] = info

    if failed:
        print(f"\nCalibration FAILED for {failed}; poses.json NOT written.")
        return 1

    # consistency: the views should all see the same board, so cameras must be on the same side
    heights = [np.array(r["T_world_cam"])[2, 3] for r in results.values()]
    if min(heights) < 0.05:
        print("WARNING: a camera is almost at/below board height - check the poses visually.")

    poses_path = resolve(cfg, cfg["paths"]["poses_file"])
    if poses_path.exists():
        backup = poses_path.with_name(f"{poses_path.stem}_{datetime.now():%Y%m%d_%H%M%S}.json")
        poses_path.rename(backup)
        print(f"existing poses file kept as {backup.name}")
    poses_path.write_text(json.dumps({
        "schema": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "frame": ("world = chessboard frame: origin at inner corner index (rows-1)*cols as ordered by OpenCV "
                  "(a fixed physical corner of the pattern), +X along columns, +Y along rows, +Z out of the board. Camera frame = OpenCV/RealSense "
                  "(x right, y down, z forward). T_world_cam maps camera coordinates to world."),
        "board": {"cols": cols, "rows": rows, "square_size_m": sq},
        "views": results}, indent=2))
    print(f"\nwrote {poses_path}   (annotated detections: {out_dir}/calib_view*.png)")

    if cfg["viz"]["enabled"] and not args.no_viz:
        from pydrake.all import Box, Rgba, RigidTransform, Sphere
        meshcat = start_meshcat(cfg["viz"]["meshcat_port"])
        meshcat_triad(meshcat, "board_frame", None, 0.10)
        w, h = (cols + 1) * sq, (rows + 1) * sq
        meshcat.SetObject("board", Box(w, h, 0.002), Rgba(0.9, 0.9, 0.9, 0.8))
        meshcat.SetTransform("board", RigidTransform([cols * sq / 2, (rows - 1) * sq / 2, -0.001]))
        for i, (name, r) in enumerate(results.items()):
            T = np.array(r["T_world_cam"])
            meshcat_triad(meshcat, f"cameras/{name}", T, 0.08)
            rgb = PALETTE[i % len(PALETTE)]
            meshcat.SetObject(f"cameras/{name}/marker", Sphere(0.015), Rgba(*rgb, 1.0))
            print(f"  Meshcat: cameras/{name} = sphere colour {np.round(rgb * 255).astype(int).tolist()}")
        print("  board_frame triad: red=X, green=Y, blue=Z. Z must point UP out of the board.")
        wait_for_user()
    return 0


if __name__ == "__main__":
    sys.exit(main())
