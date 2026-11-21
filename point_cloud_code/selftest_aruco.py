"""Self test of gather_cloud_aruco.py on synthetic scenes with known ground truth (no camera needed).

    python selftest_aruco.py            # about 3 minutes (rendering is slow)

Scene: table, two ArUco cubes in different orientations (cube 1 lies on another face), an object between them, six
640x480 views around it with 1 mm depth noise and 2 % dropouts. The marker geometry comes from the same YAML file the real
pipeline uses, so this checks the pipeline (detection, cube-to-cube calibration, camera poses, fusion, segmentation,
remembering), not the printed marker orientation (that is covered by aruco_cubes_tool/simulate_print_and_fold.py).
The images are cleaner and sharper than real camera images.
"""
import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np

import gather_cloud_aruco as gca
import synthetic_aruco as sa
from common import HERE, inv_T, load_config, rot_angle_deg

FAILS = []


def check(ok, name, detail=""):
    FAILS.extend([] if ok else [name])
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def save_views(scene, poses, folder, tag, ss=3):
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for i, T in enumerate(poses):
        f = folder / f"{tag}{i}.npz"
        if not f.exists():
            np.savez(f, **sa.render(T, scene, seed=i, ss=ss))
        files.append(f)
    return files


def pose_err(T_est, T_gt):
    d = inv_T(T_gt) @ T_est
    return float(np.linalg.norm(d[:3, 3]) * 1000), rot_angle_deg(d[:3, :3])


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*a, **k), buf.getvalue()
        except SystemExit as e:
            return e, buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=str(HERE / "output" / "aruco_selftest"))
    ap.add_argument("--layout", default=str(gca.DEFAULT_LAYOUT))
    args = ap.parse_args(argv)
    work = Path(args.work_dir)
    cfg = load_config()
    layout = Path(args.layout)
    if not layout.exists():
        sys.exit(f"{layout} not found: run aruco_cubes_tool/generate_aruco_cubes.py first")

    scene = sa.default_scene(layout)
    cams = sa.camera_ring()
    T_S_c0 = scene["cubes"][0]["T"]
    to_c0 = inv_T(T_S_c0)
    print("1) six views around the object, cubes + object + table")
    files = save_views(scene, cams, work / "views", "v")
    res, out = quiet(gca.run, files, layout, work / "run1", cfg, (0, 1))
    print(out.split("wrote")[0].rstrip())
    if isinstance(res, BaseException):
        check(False, "pipeline ran", str(res))
        return 1
    check(True, "pipeline ran on 6 views", f"{sum(v['accepted'] for v in res['poses']['views'].values())}/6 views accepted")

    e = pose_err(res["cube_T"][1], to_c0 @ scene["cubes"][1]["T"])
    check(e[0] < 1.5 and e[1] < 0.3, "cube 1 pose in the cube-0 frame (estimated from the views) vs ground truth",
          f"{e[0]:.2f} mm, {e[1]:.3f} deg (limits 1.5 mm, 0.3 deg)")
    errs = [pose_err(np.array(v["T_world_cam"]), to_c0 @ cams[i]) for i, v in enumerate(res["poses"]["views"].values()) if v["accepted"]]
    check(max(x[0] for x in errs) < 1.5 and max(x[1] for x in errs) < 0.3, "camera pose of every view vs ground truth",
          f"worst {max(x[0] for x in errs):.2f} mm, {max(x[1] for x in errs):.3f} deg over {len(errs)} views")
    rm = max(v["reproj_rmse_px"] for v in res["poses"]["views"].values() if v["accepted"])
    check(rm < 1.0, "reprojection RMSE of every view", f"worst {rm:.2f} px")
    # the camera pose w.r.t. each cube is stored and consistent: T_cube_cam = inv(T_world_cube) @ T_world_cam
    v0 = res["poses"]["views"]["view0"]
    ok = all(np.allclose(np.array(v0["T_cube_cam"][str(k)]), inv_T(res["cube_T"][k]) @ np.array(v0["T_world_cam"])) for k in (0, 1))
    check(ok, "camera pose w.r.t. every cube is stored (T_cube_cam)")

    T_tw = res["T_table_world"]
    up = T_tw[:3, :3] @ (to_c0[:3, :3] @ np.array([0, 0, 1.0]))                  # true "up" expressed in the table frame
    check(np.degrees(np.arccos(np.clip(up[2], -1, 1))) < 1.0, "table frame: z points up (true up direction within 1 deg of +z)",
          f"{np.degrees(np.arccos(np.clip(up[2], -1, 1))):.2f} deg")
    obj_t = np.asarray(res["object"].points)                                       # object cloud is written in the table frame
    check(abs(obj_t[:, 2].min()) < 0.01 and obj_t[:, 2].max() < 0.095, "object cloud sits on z = 0 (table) and is 0.08 m high",
          f"z from {obj_t[:, 2].min() * 1000:.1f} to {obj_t[:, 2].max() * 1000:.1f} mm")
    obj = (obj_t - T_tw[:3, 3]) @ T_tw[:3, :3]                                     # back to the world (cube-0) frame for the other checks
    p_S = obj @ scene["cubes"][0]["T"][:3, :3].T + scene["cubes"][0]["T"][:3, 3]
    d = sa.dist_to_objects(p_S, scene)
    check(len(obj) > 1500 and np.median(d) < 0.0025 and np.percentile(d, 95) < 0.008,
          "object cloud lies on the true object surface (table and cubes removed)",
          f"{len(obj)} points, median {np.median(d) * 1000:.2f} mm, 95th percentile {np.percentile(d, 95) * 1000:.2f} mm")
    cube_pts = [gca.inside_cube(obj, T, scene["side"] / 2, 0.0).sum() for T in res["cube_T"].values()]
    check(sum(cube_pts) == 0, "no cube points left in the object cloud", f"{sum(cube_pts)} points inside the cubes")
    check(len(res["clusters"]) <= 2, "after removing table and cubes, only the object remains as a cluster", f"{len(res['clusters'])} cluster(s)")

    print("2) remembering: second run with the stored cube layout")
    stored = work / "run1" / gca.CALIBRATED_NAME
    check(stored.exists() and (work / "run1" / "aruco_poses.json").exists(), "cube_layout_calibrated.yaml and aruco_poses.json were written")
    res2, out2 = quiet(gca.run, files[:3], stored, work / "run2", cfg, (0, 1))
    ok = not isinstance(res2, BaseException) and res2["poses"]["cubes"]["1"]["source"].startswith("remembered") \
        and np.allclose(res2["cube_T"][1], res["cube_T"][1])
    check(ok, "cube 1 pose is taken from the stored layout (no new estimate)", "reused" if ok else str(res2))
    if ok:
        c1 = res2["poses"]["cubes"]["1"]
        check(c1["check_vs_these_views_mm"] < 2 and c1["check_vs_these_views_deg"] < 0.5, "stored pose agrees with what the new views see",
              f"{c1['check_vs_these_views_mm']:.2f} mm, {c1['check_vs_these_views_deg']:.3f} deg")
    # views that see ONLY cube 1 still get a pose from the remembered cube pose
    only1, _ = quiet(gca.run, files[:1], stored, work / "run2b", cfg, (0, 1), segment=False)
    check(not isinstance(only1, BaseException), "a single view is enough once the cube layout is stored")

    print("3) cube 1 moved by 3 cm after the calibration")
    moved = sa.default_scene(layout, cube1_shift=(0.03, 0.0))
    mfiles = save_views(moved, [cams[0], cams[1], cams[3], cams[4]], work / "views_moved", "m")
    res3, out3 = quiet(gca.run, mfiles, stored, work / "run3", cfg, (0, 1), segment=False)
    c1 = res3["poses"]["cubes"]["1"] if not isinstance(res3, BaseException) else {}
    check(c1.get("check_vs_these_views_mm", 0) > 5 and "WARNING" in out3, "the stale stored pose is detected and reported",
          f"{c1.get('check_vs_these_views_mm', float('nan')):.1f} mm off" + ("; warning printed" if "WARNING" in out3 else "; NO warning"))
    res4, _ = quiet(gca.run, mfiles, stored, work / "run4", cfg, (0, 1), recalibrate=True, segment=False)
    if isinstance(res4, BaseException):
        check(False, "--recalibrate", str(res4))
    else:
        T_c0 = moved["cubes"][0]["T"]
        e = pose_err(res4["cube_T"][1], inv_T(T_c0) @ moved["cubes"][1]["T"])
        check(e[0] < 1.5 and e[1] < 0.3, "--recalibrate finds the new cube position", f"{e[0]:.2f} mm, {e[1]:.3f} deg from ground truth")

    print("4) failure modes")
    solo = sa.default_scene(layout, with_cube1=False)
    sfiles = save_views(solo, cams[:2], work / "views_solo", "s", ss=2)
    r5, _ = quiet(gca.run, sfiles, layout, work / "run5", cfg, (0, 1), segment=False)
    check(isinstance(r5, SystemExit) and "no pose for cube 1" in str(r5), "cube 1 never seen -> clear error", str(r5)[:110])
    r6, _ = quiet(gca.run, sfiles, layout, work / "run6", cfg, (0,), segment=False)
    check(not isinstance(r6, BaseException), "single-cube mode (--cubes 0) works with views that only show cube 0")
    # one marker alone is a planar PnP problem: at 0.5 m the two candidate poses fit equally well, the pose can be flipped
    K = np.array([[610.0, 0, 320.0], [0, 610.0, 240.0], [0, 0, 1]])
    corners = gca.load_layout(layout)[1][0]["markers"][0]["corners"]
    rng, amb = np.random.default_rng(1), []
    for tilt in (10, 30, 60):
        R = sa.rot("x", 180 + tilt) @ sa.rot("z", 20)
        cam = (R @ corners.T).T + np.array([0.01, 0.0, 0.5])
        uv = (K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3] + rng.normal(0, 0.2, (4, 2))
        amb.append(gca.solve_pose(corners, uv, K, np.zeros(5))["ambiguity"])
    check(max(amb) < 1.5, "a single marker is flagged as ambiguous (such views are rejected) at 0.5 m, 0.2 px noise",
          "ambiguity ratios " + ", ".join(f"{a:.2f}" for a in amb) + " (< 1.5 = rejected)")
    r7, _ = quiet(gca.run, files[:2], layout, work / "run7", cfg, (0, 1), min_markers=99, segment=False)
    check(isinstance(r7, SystemExit) and "no view gave a usable camera pose" in str(r7), "views without enough markers are rejected", str(r7)[:90])

    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
