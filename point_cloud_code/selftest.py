"""End-to-end self test on synthetic data (no camera needed).  python selftest.py [--workdir DIR]

Checks, against known ground truth:
  1. capture_view.py with a MOCK camera: file naming (never overwrites), keys, no-device message
  2. calibrate_poses.py recovers the camera poses from the rendered chessboard
  3. fuse_and_segment.py with calibration-only alignment, and with ICP
  4. 'reuse' mode: cameras moved by hand (board NOT in the scans), poses.json used as initial guess
  5. ICP safeguard: keeps the calibration-only pose when ICP cannot improve
  6. old-format .npz (depth,color,fx,fy,cx,cy only) still loads and runs
"""
import argparse
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

import synthetic as S
from common import HERE, inv_T, load_config, load_view, rot_angle_deg

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")


def quiet(fn, *a, **k):
    """Run fn, capturing stdout; return (result, text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ---------------------------------------------------------------- mock RealSense
def make_fake_rs(n_devices=1):
    rs = types.ModuleType("pyrealsense2")
    rs.stream = types.SimpleNamespace(depth="depth", color="color")
    rs.format = types.SimpleNamespace(z16="z16", rgb8="rgb8")
    rs.camera_info = types.SimpleNamespace(serial_number="serial")
    H, W = 480, 640
    rng = np.random.default_rng(1)
    raw = (rng.uniform(1500, 3000, (H, W))).astype(np.uint16)       # 0.375..0.75 m at scale 0.00025
    raw[:50] = 0
    rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    intr = types.SimpleNamespace(fx=610.0, fy=611.0, ppx=321.0, ppy=239.0, coeffs=[0.0] * 5, model="brown_conrady")

    class Frame:
        def __init__(self, a): self.a = a
        def get_data(self): return self.a
        @property
        def profile(self): return types.SimpleNamespace(as_video_stream_profile=lambda: types.SimpleNamespace(intrinsics=intr))

    class Frames:
        def get_depth_frame(self): return Frame(raw)
        def get_color_frame(self): return Frame(rgb)

    class Sensor:
        def get_depth_scale(self): return 0.00025

    class Device:
        def first_depth_sensor(self): return Sensor()
        def get_info(self, k): return "FAKE123"

    rs.context = lambda: types.SimpleNamespace(query_devices=lambda: [object()] * n_devices)
    rs.config = lambda: types.SimpleNamespace(enable_stream=lambda *a: None)
    rs.pipeline = lambda: types.SimpleNamespace(
        start=lambda cfg: types.SimpleNamespace(get_device=lambda: Device()),
        wait_for_frames=lambda t=0: Frames(), stop=lambda: None)
    rs.align = lambda s: types.SimpleNamespace(process=lambda f: f)
    return rs


def test_capture(work):
    print("1) capture_view.py with a mock camera")
    import capture_view
    out = work / "capture"
    capture_view.rs = make_fake_rs(1)
    r1, _ = quiet(capture_view.main, ["view0", "--out", str(out), "--frames", "3", "--warmup", "1"])
    r2, txt = quiet(capture_view.main, ["view0", "--out", str(out), "--frames", "3", "--warmup", "1"])
    names = sorted(p.name for p in out.glob("*.npz"))
    check("first capture saves view0.npz, second does NOT overwrite it", names == ["view0.npz", "view0_2.npz"], str(names))
    check("user is told the renamed file", "view0_2.npz" in txt)
    v = load_view(out / "view0_2.npz")
    check("saved depth is metres, aligned size, zero where invalid", v.depth.shape == (480, 640)
          and 0.3 < v.depth[v.depth > 0].mean() < 0.8 and v.depth[:50].max() == 0)
    d = np.load(out / "view0.npz")
    check("original 6 keys present + new keys", {"depth", "color", "fx", "fy", "cx", "cy"} <= set(d.files)
          and {"depth_scale", "dist_coeffs", "timestamp", "serial"} <= set(d.files), str(sorted(d.files)))
    capture_view.rs = make_fake_rs(0)
    r3, txt = quiet(capture_view.main, ["view9", "--out", str(out)])
    check("no device -> clear message, exit code 3, no file", r3 == 3 and "no RealSense device" in txt
          and not (out / "view9.npz").exists())


# ---------------------------------------------------------------- synthetic scans
def save_scans(folder, poses, board, show_board, seed0, old_format=False):
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for i, T in enumerate(poses):
        r = S.render(T, board, show_board=show_board, seed=seed0 + i)
        f = folder / f"view{i}.npz"
        if old_format:
            np.savez(f, depth=r["depth"], color=r["color"], fx=r["fx"], fy=r["fy"], cx=r["cx"], cy=r["cy"])
        else:
            np.savez(f, **r, depth_scale=0.00025, dist_coeffs=np.zeros(5), timestamp="synthetic")
        files.append(f)
    return files


def write_test_config(work, **dbscan):
    cfg = load_config()
    cfg.pop("_dir")
    cfg["paths"] = {"scans_dir": str(work / "scans"), "poses_file": str(work / "poses.json"),
                    "output_dir": str(work / "output")}
    cfg["viz"]["enabled"] = False
    cfg["board"] = {"cols": 9, "rows": 6, "square_size_mm": 25.0, "refine_with_depth": False}   # the synthetic board
    cfg["dbscan"]["eps"] = 0.025                    # merge the 3 synthetic blocks (gaps are 20 mm) into one object
    path = work / "config_test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def cluster_error(work, sub="output"):
    pts = np.asarray(o3d.io.read_point_cloud(str(work / sub / "target_cluster.ply")).points)
    d = S.dist_to_objects(pts)
    return float(np.sqrt(np.mean(d ** 2))), float(np.percentile(d, 95)), pts


def expected_extents():
    lo = np.min([o[0] for o in S.OBJECTS], axis=0)
    hi = np.max([o[1] for o in S.OBJECTS], axis=0)
    return np.sort(hi - lo)


def run_fuse(cfgp, extra=(), out=None):
    import fuse_and_segment
    o = ["--out-dir", str(Path(yaml.safe_load(Path(cfgp).read_text())["paths"]["output_dir"]).parent / out)] if out else []
    rc, txt = quiet(fuse_and_segment.main, ["--config", cfgp, "--no-viz", *o, *extra])
    return rc, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--skip-stress", action="store_true")
    args = ap.parse_args()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="pcc_selftest_"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"working directory: {work}\n")
    board = dict(cols=9, rows=6, square_size_m=0.025)
    rng = np.random.default_rng(42)

    test_capture(work)

    # ---- calibration
    print("\n2) calibrate_poses.py on rendered chessboard views (old-format .npz: 6 keys only)")
    import calibrate_poses
    truth = S.default_camera_poses()
    save_scans(work / "scans", truth, board, True, 0, old_format=True)
    cfgp = write_test_config(work)
    rc, txt = quiet(calibrate_poses.main, ["--config", cfgp, "--no-viz"])
    print("   " + txt.strip().replace("\n", "\n   "))
    import json
    poses = json.loads((work / "poses.json").read_text())["views"]
    errs = [(np.linalg.norm((inv_T(truth[i]) @ np.array(poses[f"view{i}"]["T_world_cam"]))[:3, 3]) * 1000,
             rot_angle_deg((inv_T(truth[i]) @ np.array(poses[f"view{i}"]["T_world_cam"]))[:3, :3]))
            for i in range(3)]
    check("calibration exit code 0 and poses.json written", rc == 0)
    check("camera positions within 3 mm of ground truth", max(e[0] for e in errs) < 3, f"(worst {max(e[0] for e in errs):.2f} mm)")
    check("camera rotations within 0.3 deg of ground truth", max(e[1] for e in errs) < 0.3, f"(worst {max(e[1] for e in errs):.3f} deg)")
    check("reprojection error < 1 px in every view", all(p["reproj_rmse_px"] < 1 for p in poses.values()))
    rc, txt = quiet(calibrate_poses.main, ["--config", cfgp, "--no-viz"])
    check("re-running calibration keeps the old poses.json as a timestamped backup",
          len(list(work.glob("poses_*.json"))) == 1)

    # ---- fusion on the calibration views
    print("\n3) fuse_and_segment.py, same views")
    rc, txt = run_fuse(cfgp, ["--no-icp"])
    rms0, p95_0, _ = cluster_error(work)
    rc2, txt2 = run_fuse(cfgp)
    rms1, p95_1, pts = cluster_error(work)
    print("   " + txt2.strip().replace("\n", "\n   "))
    ext = np.sort(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).get_minimal_oriented_bounding_box().extent)
    check("pipeline runs (calibration only / with ICP)", rc == 0 and rc2 == 0)
    check("target cluster lies on the true objects: RMS < 3 mm", rms0 < 0.003 and rms1 < 0.003,
          f"(calibration only {rms0*1000:.2f} mm, with ICP {rms1*1000:.2f} mm)")
    check("ICP does not degrade an already good calibration (< 1 mm worse)", rms1 < rms0 + 0.001,
          f"({rms0*1000:.2f} -> {rms1*1000:.2f} mm)")
    check("object size from minimal OBB within 10 mm of truth", np.all(np.abs(ext - expected_extents()) < 0.010),
          f"(measured {np.round(ext*1000).astype(int).tolist()} mm, true {np.round(expected_extents()*1000).astype(int).tolist()} mm)")
    check("fused.ply and target_cluster.ply written", (work / "output/fused.ply").exists() and (work / "output/target_cluster.ply").exists())

    # ---- reuse mode: hand-placed cameras, no board in the scans
    print("\n4) reuse mode: cameras moved by hand (views 1,2 off by ~2 cm / 3 deg), NO chessboard in the scans")
    moved = [truth[0]] + [S.perturb(T, rng, 0.02, 3.0) for T in truth[1:]]
    new_files = save_scans(work / "new_scans", moved, board, False, 100)
    args_v = ["--views", *map(str, new_files)]
    run_fuse(cfgp, ["--no-icp", *args_v])
    rms_c, _, _ = cluster_error(work)
    rc, txt = run_fuse(cfgp, args_v)
    rms_i, _, _ = cluster_error(work)
    print("   " + "\n   ".join(l for l in txt.splitlines() if "view" in l and "->" in l))
    check("pipeline runs without the board", rc == 0)
    check("ICP improves the error over poses.json alone", rms_i < rms_c, f"(RMS {rms_c*1000:.1f} mm -> {rms_i*1000:.2f} mm)")
    check("after ICP the cluster is within 4 mm of truth", rms_i < 0.004)
    pts4 = cluster_error(work)[2]
    ext4 = np.sort(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts4)).get_minimal_oriented_bounding_box().extent)
    check("object size within 15 mm of truth (hand-placed cameras)", np.all(np.abs(ext4 - expected_extents()) < 0.015),
          f"(measured {np.round(ext4*1000).astype(int).tolist()} mm)")

    # ---- safeguard
    print("\n5) ICP safeguard: a deliberately BAD ICP result must be rejected")
    import fuse_and_segment as F
    v = load_view(work / "scans/view0.npz")
    c5 = load_config(cfgp)
    cloud = F.preprocess(v, np.array(poses["view0"]["T_world_cam"]), c5)
    c5["icp"]["fgr_fallback"] = False
    real = F.icp_schedule
    F.icp_schedule = lambda s_, d_, T0, i_: np.array([[1, 0, 0, 0.05], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    try:
        (T, rep), txt = quiet(F.refine_view, cloud, cloud, c5, "badicp")
    finally:
        F.icp_schedule = real
    check("bad ICP (5 cm shift) is not used; pose stays at calibration", rep["method"] != "ICP"
          and np.linalg.norm(T[:3, 3]) < 1e-3 and rot_angle_deg(T[:3, :3]) < 0.01, f"(chose: {rep['method']})")
    check("better() prefers higher fitness, then lower RMSE",
          F.better({"fitness": .9, "rmse": .005}, {"fitness": .8, "rmse": .001})
          and F.better({"fitness": .8, "rmse": .001}, {"fitness": .8, "rmse": .002})
          and not F.better({"fitness": .8, "rmse": .002}, {"fitness": .8, "rmse": .002}))

    # ---- stress (informative)
    if not args.skip_stress:
        print("\n6) stress (informational, not asserted): views 1,2 off by ~6 cm / 10 deg")
        far = [truth[0]] + [S.perturb(T, rng, 0.06, 10.0) for T in truth[1:]]
        far_files = save_scans(work / "far_scans", far, board, False, 200)
        fv = ["--views", *map(str, far_files)]
        run_fuse(cfgp, ["--no-icp", *fv])
        a, _, _ = cluster_error(work)
        rc, txt = run_fuse(cfgp, fv)
        b, _, _ = cluster_error(work) if rc == 0 else (float("nan"),) * 3
        print("   " + "\n   ".join(l for l in txt.splitlines() if "view" in l and ("->" in l or "WARNING" in l or "Fast" in l)))
        print(f"   RMS to truth: poses only {a*1000:.1f} mm -> after ICP/FGR {b*1000:.1f} mm")

    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
