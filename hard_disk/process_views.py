import numpy as np, open3d as o3d
from pathlib import Path
from pydrake.all import StartMeshcat, PointCloud, Fields, BaseField

N_VIEWS = 4
STEP_DEG = 90.0    # the object was turned this much between views, counter-clockwise seen from above
MAX_DIST = 1.5
SEED_PIXEL = (192, 358)   # (u, v) of a point on the object in view 0's colour image
ROI_RADIUS = 0.30         # metres around the seed used to find the table plane
OBJ_RADIUS = 0.15         # metres around the seed that can belong to the object
MIN_HEIGHT = 0.006        # metres above the table (drops the sheet of paper)
DATA = Path(__file__).parent / "scans" / "box"


def extract_object(path, seed=None):
    d = np.load(path)
    depth = d["depth"].astype(np.float32)
    color = np.ascontiguousarray(d["color"])
    H, W = depth.shape
    fx, fy, cx, cy = (float(d[k]) for k in ("fx", "fy", "cx", "cy"))
    if seed is None:
        u, v = SEED_PIXEL
        patch = depth[v - 4:v + 5, u - 4:u + 5]
        z = np.median(patch[patch > 0])
        seed = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])

    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color), o3d.geometry.Image(depth),
        depth_scale=1.0, depth_trunc=MAX_DIST, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    pcd = pcd.voxel_down_sample(0.003)

    near = np.linalg.norm(np.asarray(pcd.points) - seed, axis=1) < ROI_RADIUS
    crop = pcd.select_by_index(np.where(near)[0])
    plane, _ = crop.segment_plane(0.004, 3, 1000)
    n, off = np.array(plane[:3]), plane[3]
    if off < 0:                                   # make the normal point up, towards the camera
        n, off = -n, -off

    pts = np.asarray(pcd.points)
    height = pts @ n + off
    keep = ((height > MIN_HEIGHT) & (height < 0.15)
            & (np.linalg.norm(pts - seed, axis=1) < OBJ_RADIUS))
    cand = pcd.select_by_index(np.where(keep)[0])
    labels = np.array(cand.cluster_dbscan(eps=0.01, min_points=20))
    if (labels >= 0).sum() == 0:
        raise RuntimeError(f"no object found in {path.name}")
    biggest = np.bincount(labels[labels >= 0]).argmax()
    obj = cand.select_by_index(np.where(labels == biggest)[0])
    obj, _ = obj.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    obj.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return obj, n, seed


def icp(source, target, init):
    result = None
    for dist in (0.02, 0.005):
        result = o3d.pipelines.registration.registration_icp(
            source, target, dist, init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        init = result.transformation
    return result


def build(objs, axis, sign):
    """Return per-view transforms into view 0's frame, and the total ICP fitness."""
    poses = [np.eye(4)]
    total = 0.0
    undo = o3d.geometry.get_rotation_matrix_from_axis_angle(
        axis * np.deg2rad(-sign * STEP_DEG))
    for k in range(1, len(objs)):
        c_src = np.asarray(objs[k].points).mean(axis=0)
        c_dst = np.asarray(objs[k - 1].points).mean(axis=0)
        init = np.eye(4)
        init[:3, :3] = undo
        init[:3, 3] = c_dst - undo @ c_src
        res = icp(objs[k], objs[k - 1], init)
        print(f"  view {k} -> {k-1}: fitness {res.fitness:.2f}, rmse {res.inlier_rmse*1000:.1f} mm")
        poses.append(poses[-1] @ res.transformation)
        total += res.fitness
    return poses, total


objs, axis, seed = [], None, None
for k in range(N_VIEWS):
    obj, n, seed = extract_object(DATA / f"view_{k:02d}.npz", seed)
    print(f"view {k}: {len(obj.points)} object points")
    objs.append(obj)
    if k == 0:
        axis = n          # table normal, pointing up; the camera is fixed so it is the same in every view
print("table normal (camera frame):", axis)

best = None
for sign in (+1, -1):
    print(f"trying rotation sign {sign:+d}")
    poses, total = build(objs, axis, sign)
    if best is None or total > best[1]:
        best = (poses, total, sign)
poses, total, sign = best
print(f"using sign {sign:+d} (total fitness {total:.2f})")

merged = o3d.geometry.PointCloud()
for obj, T in zip(objs, poses):
    merged += o3d.geometry.PointCloud(obj).transform(T)
merged = merged.voxel_down_sample(0.003)
merged, _ = merged.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
print("merged points:", len(merged.points))

obb = merged.get_oriented_bounding_box()
print("object size (m):", np.sort(obb.extent))   # compare with a tape measure

o3d.io.write_point_cloud(str(DATA.parent / "box_merged.ply"), merged)

meshcat = StartMeshcat()
pts, cols = np.asarray(merged.points), np.asarray(merged.colors)
pc = PointCloud(len(pts), Fields(BaseField.kXYZs | BaseField.kRGBs))
pc.mutable_xyzs()[:] = pts.T
pc.mutable_rgbs()[:] = (cols.T * 255).astype(np.uint8)
meshcat.SetObject("merged", pc, point_size=0.003)

try:
    input("Open the Meshcat link above in your browser. Press Enter to quit...")
except EOFError:
    pass
