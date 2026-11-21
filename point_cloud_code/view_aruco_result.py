"""Show the result of gather_cloud_aruco.py in Meshcat (open the printed link in your Windows browser).  (WSL)

    python view_aruco_result.py                         # reads output/aruco
    python view_aruco_result.py --dir output/aruco_real

Shown by default: the object cluster (own colours), the cubes as boxes with their frames, and every accepted camera as a
coloured sphere with its frame (X red, Y green, Z blue). The scene (floor, clutter) is hidden unless you pass --show-scene.
Everything is in the table frame: z up, z = 0 is the table / floor (Meshcat's grid).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import HERE, PALETTE, meshcat_cloud, meshcat_triad, start_meshcat, wait_for_user


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(HERE / "output" / "aruco"), help="output folder of gather_cloud_aruco.py")
    ap.add_argument("--port", type=int, default=None, help="Meshcat port (default: Drake picks 7000 or the next free one)")
    ap.add_argument("--max-points", type=int, default=250000, help="randomly thin the scene to this many points for display")
    ap.add_argument("--show-scene", action="store_true", help="also show the whole merged scene (floor and clutter, thinned)")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for Enter (for scripts / tests)")
    args = ap.parse_args(argv)

    import open3d as o3d
    from pydrake.all import Box, Rgba, RigidTransform, Sphere

    d = Path(args.dir)
    poses_file = d / "aruco_poses.json"
    if not poses_file.exists():
        print(f"ERROR: {poses_file} not found - run gather_cloud_aruco.py first (or pass --dir)")
        return 1
    poses = json.loads(poses_file.read_text())
    meshcat = start_meshcat(args.port)

    scene = o3d.io.read_point_cloud(str(d / "scene_cloud.ply")) if (args.show_scene and (d / "scene_cloud.ply").exists()) else None
    if scene is not None and len(scene.points):
        pts, cols = np.asarray(scene.points), np.asarray(scene.colors)
        if len(pts) > args.max_points:
            keep = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
            pts, cols = pts[keep], cols[keep]
        meshcat_cloud(meshcat, "scene", pts, cols, 0.004)
        print(f"scene: {len(pts)} points")
    obj_file = d / "object_cloud.ply"
    if obj_file.exists():
        obj = o3d.io.read_point_cloud(str(obj_file))
        cols = np.asarray(obj.colors) if obj.has_colors() else np.array([0.1, 0.9, 0.2])
        meshcat_cloud(meshcat, "object", np.asarray(obj.points), cols, 0.004)
        print(f"object: {len(obj.points)} points")
    else:
        print("no object_cloud.ply (run without --no-segment to get one)")

    side = float(poses["cube_side_m"])
    meshcat_triad(meshcat, "table_frame", None, 0.15)
    for k, c in poses["cubes"].items():
        T = np.array(c.get("T_table_cube", c["T_world_cube"]))
        meshcat.SetObject(f"cubes/cube{k}", Box(side, side, side), Rgba(0.9, 0.9, 0.9, 0.35))
        meshcat.SetTransform(f"cubes/cube{k}", RigidTransform(T))
        meshcat_triad(meshcat, f"cubes/cube{k}/frame", None, side)
        print(f"cube {k} at {np.round(T[:3, 3], 3).tolist()} m")
    n = 0
    for name, v in poses["views"].items():
        if not v["accepted"]:
            print(f"{name}: rejected ({v['rejected_because']})")
            continue
        T = np.array(v.get("T_table_cam", v["T_world_cam"]))
        rgb = PALETTE[n % len(PALETTE)]
        meshcat_triad(meshcat, f"cameras/{name}", T, 0.06)
        meshcat.SetObject(f"cameras/{name}/ball", Sphere(0.012), Rgba(*rgb, 1.0))
        n += 1
    print(f"{n} cameras drawn")
    print(f"\nMeshcat is running: open the link printed above (usually http://localhost:7000) in your Windows browser.")
    if not args.no_wait:
        wait_for_user()
    return 0


if __name__ == "__main__":
    sys.exit(main())
