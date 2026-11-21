"""Shared helpers: config, .npz loading, back-projection, transforms, Meshcat."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"

PALETTE = np.array([[0.90, 0.20, 0.20],    # view0 red
                    [0.20, 0.70, 0.30],    # view1 green
                    [0.20, 0.40, 0.90],    # view2 blue
                    [0.90, 0.75, 0.10],
                    [0.70, 0.30, 0.80]])


# ---------------------------------------------------------------- config / paths
def load_config(path=None, overrides=None):
    import yaml
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text())
    cfg["_dir"] = path.resolve().parent
    for dotted, value in (overrides or {}).items():       # e.g. {"roi.enabled": False}
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return cfg


def resolve(cfg, p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else (cfg["_dir"] / p).resolve()


# ---------------------------------------------------------------- views
@dataclass
class View:
    name: str
    path: Path
    depth: np.ndarray            # HxW float32, metres, 0 = invalid, aligned to colour
    color: np.ndarray            # HxWx3 uint8, RGB
    K: np.ndarray                # 3x3 colour intrinsics
    dist: np.ndarray             # distortion coefficients (k1,k2,p1,p2,k3), zeros if absent
    extra: dict = field(default_factory=dict)


def load_view(path, name=None):
    """Load a capture .npz. Old files (depth,color,fx,fy,cx,cy) and new ones both work."""
    path = Path(path)
    d = np.load(path, allow_pickle=False)
    depth = np.asarray(d["depth"])
    if depth.dtype.kind in "ui":                       # raw z16 counts instead of metres
        if "depth_scale" not in d.files:
            raise ValueError(f"{path.name}: integer depth but no depth_scale key")
        depth = depth.astype(np.float32) * float(d["depth_scale"])
    depth = depth.astype(np.float32)
    color = np.ascontiguousarray(d["color"])
    K = np.array([[float(d["fx"]), 0, float(d["cx"])],
                  [0, float(d["fy"]), float(d["cy"])],
                  [0, 0, 1.0]])
    dist = (np.asarray(d["dist_coeffs"], float).ravel()[:5]
            if "dist_coeffs" in d.files else np.zeros(5))
    extra = {k: (str(d[k]) if d[k].dtype.kind in "US" else d[k].item() if d[k].ndim == 0 else d[k])
             for k in d.files if k not in ("depth", "color", "dist_coeffs")}
    return View(name or path.stem, path, depth, color, K, dist, extra)


def backproject(view, max_depth=1.5, min_depth=0.0):
    """Depth image -> (points Nx3 in the camera frame, colours Nx3 in 0..1)."""
    z = view.depth
    H, W = z.shape
    valid = np.isfinite(z) & (z > max(min_depth, 1e-6)) & (z <= max_depth)
    v, u = np.nonzero(valid)
    zz = z[v, u].astype(np.float64)
    fx, fy, cx, cy = view.K[0, 0], view.K[1, 1], view.K[0, 2], view.K[1, 2]
    pts = np.column_stack(((u - cx) / fx * zz, (v - cy) / fy * zz, zz))
    cols = view.color[v, u].astype(np.float64) / 255.0
    return pts, cols


def to_o3d(points, colors=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))
    return pcd


def seed_open3d(seed=0):
    import open3d as o3d
    try:
        o3d.utility.random.seed(seed)
    except Exception:
        pass


# ---------------------------------------------------------------- transforms
def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).ravel()
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def transform_points(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- Meshcat
def start_meshcat(port=None):
    from pydrake.all import Meshcat, MeshcatParams, StartMeshcat
    if port is None:
        return StartMeshcat()
    try:
        return Meshcat(MeshcatParams(port=int(port)))
    except Exception as e:
        print(f"[meshcat] could not open port {port} ({e}); letting Drake choose a free one")
        return Meshcat()


def meshcat_cloud(meshcat, path, points, colors, size=0.003, T=None):
    """colors: Nx3 floats 0..1, or a single RGB triple applied to all points."""
    from pydrake.all import BaseField, Fields, PointCloud, RigidTransform
    points = np.asarray(points)
    colors = np.asarray(colors, float)
    if colors.ndim == 1:
        colors = np.tile(colors, (len(points), 1))
    pc = PointCloud(len(points), Fields(BaseField.kXYZs | BaseField.kRGBs))
    pc.mutable_xyzs()[:] = points.T
    pc.mutable_rgbs()[:] = (colors.T * 255).astype(np.uint8)
    meshcat.SetObject(path, pc, point_size=size)
    if T is not None:
        meshcat.SetTransform(path, RigidTransform(T))


def meshcat_triad(meshcat, path, T=None, length=0.1, radius=0.003):
    """Coordinate axes (X red, Y green, Z blue). Meshcat.SetTriad does not exist in all Drake versions."""
    from pydrake.all import Cylinder, Rgba, RigidTransform, RotationMatrix
    axes = [("x", RotationMatrix.MakeYRotation(np.pi / 2), [length / 2, 0, 0], Rgba(1, 0, 0, 1)),
            ("y", RotationMatrix.MakeXRotation(-np.pi / 2), [0, length / 2, 0], Rgba(0, 1, 0, 1)),
            ("z", RotationMatrix(), [0, 0, length / 2], Rgba(0, 0, 1, 1))]
    for name, R, c, col in axes:
        meshcat.SetObject(f"{path}/{name}", Cylinder(radius, length), col)
        meshcat.SetTransform(f"{path}/{name}", RigidTransform(R, c))
    if T is not None:
        meshcat.SetTransform(path, RigidTransform(T))


def wait_for_user(msg="Open the Meshcat link above in your browser. Press Enter to quit..."):
    try:
        input(msg)
    except EOFError:
        pass
