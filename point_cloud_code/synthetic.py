"""Synthetic RGB-D scenes with KNOWN ground truth, used by selftest.py (no camera needed).

Scene (world = board frame): table plane z=0, a printed chessboard on it, and a few boxes beside the board.
Cameras follow the OpenCV/RealSense convention (x right, y down, z forward).
"""
import numpy as np

from common import make_T

K_L515_LIKE = dict(fx=610.0, fy=610.0, cx=320.0, cy=240.0, W=640, H=480)

# axis-aligned boxes: (lo, hi, RGB)
OBJECTS = [
    (np.array([0.30, 0.02, 0.0]), np.array([0.40, 0.08, 0.05]), (0.85, 0.35, 0.15)),   # main block
    (np.array([0.32, 0.03, 0.05]), np.array([0.36, 0.07, 0.09]), (0.15, 0.45, 0.85)),  # small block on top
    (np.array([0.42, 0.04, 0.0]), np.array([0.45, 0.14, 0.03]), (0.25, 0.75, 0.30)),   # low bar, asymmetry
]


def look_at(pos, target, up=(0, 0, 1.0)):
    """Camera-to-world transform for a camera at `pos` looking at `target` (x right, y down, z forward)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return make_T(np.column_stack((r, d, f)), pos)


def perturb(T, rng, trans_m, rot_deg):
    """Apply a random small rigid error to a camera pose (simulates placing the camera by hand)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rot_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    dT = make_T(R, rng.normal(size=3) / np.sqrt(3) * trans_m)
    return dT @ T


def _plane_hit(o, d):
    s = np.full(d.shape[0], np.inf)
    ok = d[:, 2] < -1e-9
    s[ok] = -o[2] / d[ok, 2]
    return s


def _box_hit(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    t_in, t_out = tmin.max(axis=1), tmax.min(axis=1)
    hit = (t_out >= t_in) & (t_out > 0)
    return np.where(hit, np.maximum(t_in, 0), np.inf), tmin.argmax(axis=1)


def render(T_world_cam, board, show_board=True, noise_m=0.001, invalid_frac=0.02, seed=0, ss=3):
    """Return dict with depth (m), color (RGB uint8), fx fy cx cy for one camera pose."""
    rng = np.random.default_rng(seed)
    k = K_L515_LIKE
    W, H = k["W"], k["H"]
    cols, rows, sq = board["cols"], board["rows"], board["square_size_m"]
    R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]

    def cast(u, v):
        dirs = np.column_stack(((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], np.ones(u.size)))
        dw = dirs @ R.T                                 # ray directions in the world (z_cam = ray parameter s)
        s_plane = _plane_hit(o, dw)
        best_s, best_id, best_ax = s_plane.copy(), np.full(u.size, -1), np.zeros(u.size, int)
        for i, (lo, hi, _) in enumerate(OBJECTS):
            s, ax = _box_hit(o, dw, lo, hi)
            m = s < best_s
            best_s[m], best_id[m], best_ax[m] = s[m], i, ax[m]
        return best_s, best_id, best_ax, o + dw * best_s[:, None]

    def shade(best_s, best_id, best_ax, P):
        rgb = np.zeros((best_s.size, 3))
        rgb[:] = (0.55, 0.52, 0.48)                     # table
        on_plane = best_id < 0
        x, y = P[:, 0], P[:, 1]
        # paper sheet around the board, then the squares (squares start one square before inner corner 0)
        paper = on_plane & (x > -2 * sq) & (x < (cols + 2) * sq) & (y > -2 * sq) & (y < (rows + 2) * sq)
        rgb[paper] = (0.97, 0.97, 0.97)
        if show_board:
            on = on_plane & (x >= -sq) & (x < cols * sq) & (y >= -sq) & (y < rows * sq)
            idx = np.floor(x / sq).astype(int) + np.floor(y / sq).astype(int)
            rgb[on & (idx % 2 != 0)] = (0.04, 0.04, 0.04)
            rgb[on & (idx % 2 == 0)] = (0.97, 0.97, 0.97)
        for i, (_, _, c) in enumerate(OBJECTS):
            m = best_id == i
            rgb[m] = np.array(c)[None, :] * np.array([0.65, 0.8, 1.0])[best_ax[m]][:, None]
        rgb[~np.isfinite(best_s)] = 0.3
        return rgb

    # colour: ss x ss sub-pixel supersampling so the printed squares are anti-aliased like a real camera
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    uu, vv = uu.ravel().astype(float), vv.ravel().astype(float)
    acc = np.zeros((uu.size, 3))
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    for dv in offs:
        for du in offs:
            acc += shade(*cast(uu + du, vv + dv))
    color = np.clip(acc / ss ** 2 * 255, 0, 255).astype(np.uint8).reshape(H, W, 3)

    s, _, _, _ = cast(uu, vv)
    depth = s.reshape(H, W).astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    depth += rng.normal(0, noise_m, depth.shape).astype(np.float32) * (depth > 0)
    depth[rng.random(depth.shape) < invalid_frac] = 0
    depth[depth > 1.5] = 0
    return dict(depth=depth, color=color, fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"])


def default_camera_poses():
    target = (0.20, 0.08, 0.02)
    return [look_at((0.20, -0.32, 0.45), target),
            look_at((0.62, 0.30, 0.45), target),
            look_at((-0.12, 0.34, 0.45), target)]


def dist_to_objects(P):
    """Distance from each point to the nearest object surface (ground truth, for the self test)."""
    best = np.full(len(P), np.inf)
    for lo, hi, _ in OBJECTS:
        outside = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0), axis=1)
        inside = np.minimum(P - lo, hi - P).min(axis=1)
        best = np.minimum(best, np.where(outside > 0, outside, inside))
    return best
