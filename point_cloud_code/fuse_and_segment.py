"""Fuse the views with the stored poses, refine with ICP, remove the table (RANSAC), cluster (DBSCAN).  (WSL)

    python fuse_and_segment.py                                  # scans/view0.npz view1.npz view2.npz
    python fuse_and_segment.py --views new_a.npz new_b.npz new_c.npz     # re-use poses.json, no board needed

--views are matched to poses.json BY ORDER (1st file <-> view0, 2nd <-> view1, ...). poses.json is only an
initial guess; ICP does the real alignment of views 1.. onto view 0.
"""
import argparse
import sys

import numpy as np
import open3d as o3d

from common import (PALETTE, backproject, load_config, load_view, meshcat_cloud, meshcat_triad, resolve, seed_open3d,
                    start_meshcat, to_o3d, transform_points, wait_for_user)
import json

reg = o3d.pipelines.registration


# ---------------------------------------------------------------- per-view preprocessing
def preprocess(view, T_world_cam, cfg):
    c, roi = cfg["cloud"], cfg["roi"]
    pts, cols = backproject(view, c["max_depth"], c["min_depth"])       # invalid depth dropped here
    pts = transform_points(T_world_cam, pts)                            # initial alignment (calibration)
    n_raw = len(pts)
    if roi["enabled"]:
        keep = ((pts[:, 0] >= roi["x"][0]) & (pts[:, 0] <= roi["x"][1]) &
                (pts[:, 1] >= roi["y"][0]) & (pts[:, 1] <= roi["y"][1]) &
                (pts[:, 2] >= roi["z"][0]) & (pts[:, 2] <= roi["z"][1]))
        pts, cols = pts[keep], cols[keep]
    pcd = to_o3d(pts, cols).voxel_down_sample(c["voxel"])
    n_roi = len(pcd.points)
    if n_roi:
        pcd, _ = pcd.remove_statistical_outlier(c["sor_neighbors"], c["sor_std_ratio"])
    print(f"  {view.name}: {n_raw} valid points -> {n_roi} after ROI+voxel -> {len(pcd.points)} after outlier removal")
    return pcd


def remove_table(pcd, cfg, label=""):
    p = cfg["plane"]
    if len(pcd.points) < 50:
        return pcd, None
    plane, inl = pcd.segment_plane(p["distance_threshold"], p["ransac_n"], p["iterations"])
    return pcd.select_by_index(inl, invert=True), (np.array(plane), len(inl))


# ---------------------------------------------------------------- registration
def measure(src, dst, T, dist):
    r = reg.evaluate_registration(src, dst, dist, T)
    return {"fitness": float(r.fitness), "rmse": float(r.inlier_rmse)}


def better(a, b):
    """True if registration metrics a are better than b."""
    if abs(a["fitness"] - b["fitness"]) > 1e-3:
        return a["fitness"] > b["fitness"]
    return a["rmse"] < b["rmse"] - 1e-6


def clearly_better(new, base, min_gain):
    """ICP/FGR must beat the baseline by a margin; tiny gains are just noise-fitting on partial overlap."""
    if new["fitness"] >= base["fitness"] + min_gain:
        return True
    return new["fitness"] >= base["fitness"] - 0.01 and new["rmse"] <= 0.85 * base["rmse"]


def icp_schedule(src, dst, T0, icfg):
    T = T0
    for vox, iters in zip(icfg["voxels"], icfg["max_iterations"]):
        s, t = src.voxel_down_sample(vox), dst.voxel_down_sample(vox)
        for p in (s, t):
            p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 3, max_nn=30))
        r = reg.registration_icp(s, t, vox * icfg["max_dist_factor"], T,
                                 reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(max_iteration=iters))
        T = r.transformation
    return T


def fgr_init(src, dst, icfg):
    vox = icfg["fgr_voxel"]
    s, t = src.voxel_down_sample(vox), dst.voxel_down_sample(vox)
    feats = []
    for p in (s, t):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 2, max_nn=30))
        feats.append(reg.compute_fpfh_feature(
            p, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 5, max_nn=100)))
    r = reg.registration_fgr_based_on_feature_matching(
        s, t, feats[0], feats[1], reg.FastGlobalRegistrationOption(maximum_correspondence_distance=vox * 1.5))
    return r.transformation


def table_plane(pcd, cfg):
    """RANSAC table plane of a cloud. Returns (unit normal pointing up, offset d with n.x = d, inlier idx)."""
    p = cfg["plane"]
    if len(pcd.points) < 50:
        return None
    plane, inl = pcd.segment_plane(p["distance_threshold"], p["ransac_n"], p["iterations"])
    n, d = np.array(plane[:3]), -plane[3]                  # open3d: n.x + a = 0
    if n[2] < 0:
        n, d = -n, -d
    return n / np.linalg.norm(n), d, inl


def plane_align_T(src_plane, dst_plane, pivot):
    """Rigid transform that makes the src table plane coincide with the dst table plane.

    Fixes tilt (2 DOF) and height (1 DOF) only; rotates about `pivot` so the object does not slide in-plane.
    """
    n_s, d_s, _ = src_plane
    n_d, d_d, _ = dst_plane
    v, c = np.cross(n_s, n_d), float(n_s @ n_d)
    s = np.linalg.norm(v)
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    t = pivot - R @ pivot
    d_s2 = n_d @ (R @ (n_s * d_s) + t)                     # src plane offset along n_d after the rotation
    t = t + (d_d - d_s2) * n_d
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def resnap(T, src_full, src_obj, dst_plane, cfg):
    """After ICP moved the source by T, re-align its table plane to dst_plane (pivot = moved object centroid)."""
    pl = table_plane(o3d.geometry.PointCloud(src_full).transform(T), cfg)
    if pl is None:
        return T
    pivot = T[:3, :3] @ np.asarray(src_obj.points).mean(0) + T[:3, 3]
    return plane_align_T(pl, dst_plane, pivot) @ T


def plane_gap(pl_a, pl_b):
    """(tilt in degrees, offset in mm at the world origin) between two table planes."""
    tilt = np.degrees(np.arccos(np.clip(pl_a[0] @ pl_b[0], -1, 1)))
    return float(tilt), float(abs(pl_a[1] - pl_b[1]) * 1000)


def refine_view(src_full, dst_full, cfg, name):
    """Return (T 4x4 mapping src->dst world, report dict).

    two-stage (icp.plane_align: true):  1) snap table planes together  2) ICP on above-table points
                                         3) snap the planes together again (ICP may have tilted slightly)
    classic  (icp.plane_align: false):  plain coarse-to-fine ICP on the whole clouds
    """
    icfg = cfg["icp"]
    ev = icfg["eval_distance"]
    ps, pd = table_plane(src_full, cfg), table_plane(dst_full, cfg)
    use_planes = bool(icfg["plane_align"] and ps is not None and pd is not None)
    if icfg["plane_align"] and not use_planes:
        print(f"  WARNING {name}: no table plane found - using plain ICP on the whole cloud")
    if use_planes and min(ps[0][2], pd[0][2]) < 0.9:
        print(f"  WARNING {name}: table plane is not horizontal in the world frame "
              f"(nz={min(ps[0][2], pd[0][2]):.2f}) - the poses or ROI are probably wrong; using plain ICP")
        use_planes = False

    if use_planes:
        src = src_full.select_by_index(ps[2], invert=True)
        dst = dst_full.select_by_index(pd[2], invert=True)
        if min(len(src.points), len(dst.points)) < icfg["min_points"]:
            print(f"  WARNING {name}: fewer than {icfg['min_points']} points above the table - "
                  "only the table planes are aligned (no object to run ICP on)")
            T = plane_align_T(ps, pd, np.asarray(src_full.points).mean(0))
            return T, {"before": {"fitness": 0.0, "rmse": 0.0}, "after": {"fitness": 0.0, "rmse": 0.0},
                       "method": "plane alignment only"}
    else:
        src, dst = src_full, dst_full

    before = measure(src, dst, np.eye(4), ev)
    report = {"before": before, "method": "calibration only"}
    if use_planes:
        report["plane_gap_before"] = plane_gap(ps, pd)
    gain = icfg["min_fitness_gain"]

    best_name, best_T, best = "calibration only", np.eye(4), before
    if use_planes:                                          # the views share ONE table: always honour that
        T_a = plane_align_T(ps, pd, np.asarray(src.points).mean(0))
        m_a = measure(src, dst, T_a, ev)
        if m_a["fitness"] >= before["fitness"] - 0.01:
            best_name, best_T, best = "plane alignment", T_a, m_a
    T_init = best_T

    T = icp_schedule(src, dst, T_init, icfg)
    if use_planes:                                          # re-snap the planes after ICP
        T = resnap(T, src_full, src, pd, cfg)
    m_icp = measure(src, dst, T, ev)
    report["icp"] = m_icp
    if clearly_better(m_icp, best, gain):
        best_name, best_T, best = "ICP", T, m_icp
    else:
        print(f"  {name}: ICP gain too small (fitness {best['fitness']:.3f} -> {m_icp['fitness']:.3f}, "
              f"RMSE {best['rmse']*1000:.2f} -> {m_icp['rmse']*1000:.2f} mm) - keeping {best_name}")

    if icfg["fgr_fallback"] and best["fitness"] < icfg["fgr_trigger_fitness"]:
        print(f"  {name}: fitness {best['fitness']:.2f} < {icfg['fgr_trigger_fitness']} -> trying Fast Global Registration")
        try:
            T_f = icp_schedule(src, dst, fgr_init(src, dst, icfg), icfg)
            if use_planes:
                T_f = resnap(T_f, src_full, src, pd, cfg)
            m_f = measure(src, dst, T_f, ev)
            report["fgr+icp"] = m_f
            if clearly_better(m_f, best, gain):
                best_name, best_T, best = "FGR+ICP", T_f, m_f
        except Exception as e:                              # FGR can fail on sparse clouds
            print(f"  {name}: FGR failed ({e})")

    report.update(after=best, method=best_name)
    if use_planes:
        moved = o3d.geometry.PointCloud(src_full).transform(best_T)
        pl = table_plane(moved, cfg)
        if pl is not None:
            report["plane_gap_after"] = plane_gap(pl, pd)
    gap = ""
    if "plane_gap_before" in report and "plane_gap_after" in report:
        gb, ga = report["plane_gap_before"], report["plane_gap_after"]
        gap = f", table gap {gb[0]:.2f} deg/{gb[1]:.1f} mm -> {ga[0]:.2f} deg/{ga[1]:.1f} mm"
    print(f"  {name} -> view0: fitness {before['fitness']:.3f} -> {best['fitness']:.3f}, "
          f"inlier RMSE {before['rmse']*1000:.2f} -> {best['rmse']*1000:.2f} mm{gap}  [{best_name}]")
    if best_name == "calibration only":
        print(f"  WARNING {name}: ICP did not clearly improve the alignment, KEEPING the calibration-only pose. "
              "Check overlap between views, ROI and icp.* settings.")
    return best_T, report


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", nargs="+", help=".npz files in order (default scans/view0.npz view1.npz view2.npz)")
    ap.add_argument("--config")
    ap.add_argument("--poses", help="poses.json (default: paths.poses_file in config)")
    ap.add_argument("--out-dir", help="where to write fused.ply / target_cluster.ply (default: paths.output_dir)")
    ap.add_argument("--no-icp", action="store_true", help="calibration-only alignment")
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_open3d()
    scans = resolve(cfg, cfg["paths"]["scans_dir"])
    files = ([resolve(cfg, f) for f in args.views] if args.views
             else [scans / f"view{i}.npz" for i in range(3)])
    poses_path = resolve(cfg, args.poses or cfg["paths"]["poses_file"])
    out_dir = resolve(cfg, args.out_dir or cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not poses_path.exists():
        print(f"ERROR: {poses_path} not found - run calibrate_poses.py first.")
        return 1
    poses = json.loads(poses_path.read_text())["views"]
    if len(files) > len(poses):
        print(f"ERROR: {len(files)} views given but poses.json only has {len(poses)}")
        return 1
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        print("ERROR: file(s) not found:", *missing, sep="\n  ")
        return 1

    print("1) preprocessing")
    clouds = []
    for i, f in enumerate(files):
        T = np.array(poses[f"view{i}"]["T_world_cam"])
        clouds.append(preprocess(load_view(f, f"view{i}"), T, cfg))
    if any(len(c.points) == 0 for c in clouds):
        print("ERROR: a view has no points inside the ROI - check roi.* in config.yaml and the poses.")
        return 1

    print("2) ICP refinement onto view0")
    aligned, reports = [clouds[0]], {}
    for i in range(1, len(clouds)):
        if args.no_icp or not cfg["icp"]["enabled"]:
            aligned.append(clouds[i])
            continue
        T, rep = refine_view(clouds[i], clouds[0], cfg, f"view{i}")
        reports[f"view{i}"] = rep
        aligned.append(o3d.geometry.PointCloud(clouds[i]).transform(T))

    merged = o3d.geometry.PointCloud()
    for p in aligned:
        merged += p
    merged = merged.voxel_down_sample(cfg["cloud"]["voxel"])
    o3d.io.write_point_cloud(str(out_dir / "fused.ply"), merged)
    print(f"merged cloud: {len(merged.points)} points -> {out_dir/'fused.ply'}")

    print("3) RANSAC plane removal on the merged cloud")
    rest, info = remove_table(merged, cfg)
    if info is None:
        print("ERROR: merged cloud too small for plane segmentation")
        return 1
    plane, n_in = info
    print(f"  plane {np.round(plane, 4).tolist()}  ({n_in} inliers removed, {len(rest.points)} points left)")

    print("4) DBSCAN clustering")
    labels = np.array(rest.cluster_dbscan(eps=cfg["dbscan"]["eps"], min_points=cfg["dbscan"]["min_points"]))
    if (labels >= 0).sum() == 0:
        print("ERROR: DBSCAN found no cluster - try a larger dbscan.eps or smaller min_points.")
        return 1
    ids = np.arange(labels.max() + 1)
    sizes = np.array([(labels == k).sum() for k in ids])
    order = ids[np.argsort(-sizes)]
    rpts = np.asarray(rest.points)
    print(f"  {len(ids)} clusters ({(labels < 0).sum()} noise points); largest first:")
    for k in order[:10]:
        c = rpts[labels == k]
        print(f"   cluster {k:2d}: {len(c):6d} pts, centroid ({c.mean(0)[0]:+.3f},{c.mean(0)[1]:+.3f},{c.mean(0)[2]:+.3f})")
    t = cfg["target"]
    if t["mode"] == "nearest":
        cent = np.array([rpts[labels == k].mean(0) for k in ids])
        target = ids[np.argmin(np.linalg.norm(cent - np.array(t["point"]), axis=1))]
    else:
        target = order[0]
    obj = rest.select_by_index(np.where(labels == target)[0])
    n_before = len(obj.points)
    obj, _ = obj.remove_statistical_outlier(t["outlier_neighbors"], t["outlier_std_ratio"])   # a few stray points ruin the box
    print(f"  target cluster outlier removal: {n_before} -> {len(obj.points)} points")
    cam0 = np.array(poses["view0"]["T_world_cam"])[:3, 3]
    obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    obj.orient_normals_towards_camera_location(cam0)
    o3d.io.write_point_cloud(str(out_dir / "target_cluster.ply"), obj)
    obb = obj.get_minimal_oriented_bounding_box() if hasattr(obj, "get_minimal_oriented_bounding_box") \
        else obj.get_oriented_bounding_box()
    pca = obj.get_oriented_bounding_box()
    print(f"target = cluster {target} ({len(obj.points)} pts) -> {out_dir/'target_cluster.ply'}")
    print("minimal oriented bounding box (m), sorted:", np.round(np.sort(obb.extent), 4).tolist(),
          "  centre:", np.round(obb.center, 3).tolist())
    print("PCA-based bounding box (m), sorted:      ", np.round(np.sort(pca.extent), 4).tolist(),
          "  (loose for non-box shapes)")

    if cfg["viz"]["enabled"] and not args.no_viz:
        meshcat = start_meshcat(cfg["viz"]["meshcat_port"])
        size = cfg["viz"]["point_size"]
        r = cfg["roi"]["x"]
        dx = (r[1] - r[0] + 0.15) if cfg["roi"]["enabled"] else 0.7
        shift = lambda x: np.array([[1, 0, 0, x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        for i, (before, after) in enumerate(zip(clouds, aligned)):
            col = PALETTE[i % len(PALETTE)]
            meshcat_cloud(meshcat, f"1_calibration_only/view{i}", np.asarray(before.points), col, size, shift(0))
            meshcat_cloud(meshcat, f"2_after_ICP/view{i}", np.asarray(after.points), col, size, shift(dx))
        meshcat_cloud(meshcat, "3_target_cluster", np.asarray(obj.points), np.asarray(obj.colors), size, shift(2 * dx))
        meshcat_triad(meshcat, "world_frame", None, 0.1)
        print(f"Meshcat layout along +x: calibration only (x=0), after ICP (x=+{dx:.2f} m), target cluster (x=+{2*dx:.2f} m)")
        print("colours: view0 red, view1 green, view2 blue")
        wait_for_user()
    return 0


if __name__ == "__main__":
    sys.exit(main())
