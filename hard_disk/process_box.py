import numpy as np, open3d as o3d
from pydrake.all import StartMeshcat, PointCloud, Fields, BaseField

d = np.load("scans/box_view0.npz")
depth = d["depth"].astype(np.float32)
color = np.ascontiguousarray(d["color"])
H, W = depth.shape
print("valid depth pixels:", (depth > 0).mean())

intr = o3d.camera.PinholeCameraIntrinsic(W, H, float(d["fx"]), float(d["fy"]),
                                         float(d["cx"]), float(d["cy"]))
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(color), o3d.geometry.Image(depth),
    depth_scale=1.0, depth_trunc=1.5, convert_rgb_to_intensity=False)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
print("raw points:", len(pcd.points))

pcd = pcd.voxel_down_sample(0.003)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

# remove the table (largest plane), then keep the biggest remaining blob
_, inliers = pcd.segment_plane(0.005, 3, 1000)
rest = pcd.select_by_index(inliers, invert=True)
labels = np.array(rest.cluster_dbscan(eps=0.02, min_points=20))
biggest = np.bincount(labels[labels >= 0]).argmax()
box = rest.select_by_index(np.where(labels == biggest)[0])

box.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
box.orient_normals_towards_camera_location(np.zeros(3))  # camera at origin

obb = box.get_oriented_bounding_box()
print("box size (m):", np.sort(obb.extent))   # compare with a tape measure

# view in Meshcat
meshcat = StartMeshcat()
pts, cols = np.asarray(box.points), np.asarray(box.colors)
pc = PointCloud(len(pts), Fields(BaseField.kXYZs | BaseField.kRGBs))
pc.mutable_xyzs()[:] = pts.T
pc.mutable_rgbs()[:] = (cols.T * 255).astype(np.uint8)
meshcat.SetObject("box", pc, point_size=0.003)
input("Open the Meshcat link in your browser, press Enter to quit")