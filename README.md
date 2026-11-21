# Master_Thesis

Master thesis on **robotic unscrewing of e-waste** (simulation in Drake). The repo has two parts:

| Folder | Part | Owner |
|---|---|---|
| `point_cloud_code/`, `scans/` | **Perception**: one clean 3D point cloud of an object (a drill) from three views of a single Intel RealSense L515, classic geometry only (numpy, Open3D, OpenCV; no learning-based methods) | Toghrul |
| `diana7_drake/` | **Robot model and control in Drake**: Diana 7 arm (URDF + meshes) and first simulation scripts | Emre |

`diana7_drake/` was imported from https://github.com/chocolatecherry-hub/diana7-drake (branch `main`, state of
2026-11-12) with `git subtree`, so Emre's commits and authorship are kept in the history. His own repo is the
original; if he keeps working there, pull his changes with
`git subtree pull --prefix=diana7_drake git@github.com:chocolatecherry-hub/diana7-drake.git main`.

This file is the status/tracking document for both people and their AI assistants. Update the **Work log**
and **Status** sections whenever something changes, and keep claims tied to numbers or files.

---

## 1. Status at a glance (last updated 2026-11-21)

| Item | State |
|---|---|
| Diana 7 URDF with OBJ meshes (Drake-loadable) | Done: `diana7_drake/models/diana7/` (Emre). Checked 2026-11-21: loads in Drake 1.57, 7 positions, 11 bodies |
| Diana 7 viewing + hold-pose simulation | Scripts exist (`diana7_drake/scripts/01_view_robot.py`, `02_hold_pose.py`); **not re-run since the import**, see section 9 |
| Grasping / unscrewing, connecting the perception output to the robot | **Not started** |
| Capture of 3 views of the drill with the L515 | Done: `scans/drillA.npz`, `drillB.npz`, `drillC.npz` |
| Camera pose calibration from a chessboard | Done: `point_cloud_code/poses.json` (board 7x6 inner corners, 25 mm squares) |
| Fusion (plane alignment + ICP, FGR fallback) | Done: `point_cloud_code/output/fused.ply` |
| Table removal (RANSAC) + object isolation (DBSCAN) | Done: `point_cloud_code/output/target_cluster.ply` |
| Object size compared against a ruler | **Not done yet** |

## 2. The canonical perception files (use these, ignore anything else)

| File | What it is |
|---|---|
| `scans/drillA.npz`, `drillB.npz`, `drillC.npz` | Raw RGB-D views 0, 1, 2 (the **only** scans `poses.json` refers to) |
| `point_cloud_code/poses.json` | Calibrated camera poses `T_world_cam` (camera -> chessboard/world frame, metres). Created 2026-10-30 13:11 |
| `point_cloud_code/output/fused.ply` | All three views merged, table still included. 98,146 points, with colours, no normals |
| `point_cloud_code/output/target_cluster.ply` | The isolated drill. 5,600 points, with colours and normals (oriented toward camera 0) |
| `point_cloud_code/output/calib_view{0,1,2}.png` | Chessboard detection overlays (green corners, red circle = world origin) |
| `point_cloud_code/config.yaml` | Every tunable parameter. The values in the repo are the ones that produced the two PLY files above |

`fused.ply` and `target_cluster.ply` were written on 2026-10-30 13:16 by `fuse_and_segment.py` from the three
drill scans and the `poses.json` above, with the committed `config.yaml`. To reproduce:

```bash
cd point_cloud_code
python fuse_and_segment.py --views ../scans/drillA.npz ../scans/drillB.npz ../scans/drillC.npz --no-viz
```

Old test scans (box, earlier view*/drill* retakes), failed-detection images and earlier pose backups were
deliberately **not** uploaded; they are not valid inputs for the current results.

## 3. What the pipeline does (and which algorithm does what)

Code: `point_cloud_code/fuse_and_segment.py`. Full documentation: `point_cloud_code/README.md`.

1. **Acquisition.** `capture_view.py` (Windows only, the camera is not reachable from WSL) saves a median of
   30 depth frames aligned to colour. `calibrate_poses.py` finds the chessboard (`findChessboardCornersSB`),
   runs `solvePnP` and writes the camera pose of each view.
2. **Preprocessing per view.** Back-project depth to 3D, transform with `poses.json`, crop to the ROI,
   voxel downsample (3 mm), statistical outlier removal.
3. **Registration of views 1 and 2 onto view 0.** Table planes are snapped together (RANSAC plane), then
   **point-to-plane ICP** on above-table points (coarse-to-fine voxels 10/5/3 mm), then planes are snapped
   again. ICP is accepted only if it clearly beats the calibration-only pose. **Fast Global Registration
   (FPFH)** is the fallback if fitness stays below 0.5.
4. **Table removal.** RANSAC plane (`segment_plane`, threshold 4 mm) on the merged cloud.
5. **Object isolation.** **DBSCAN** (eps 15 mm, min 30 points). Target mode is `nearest` to the world point
   `[0.135, -0.105, 0.15]` (the drill); `largest` picked the laptop instead.
6. Outlier removal on the cluster, normals, minimal oriented bounding box.

## 4. Numbers that can be checked

From `poses.json` (reprojection error of the chessboard PnP, should be < 1 px):

| View | File | Reprojection RMSE | Camera position in world (x, y, z) m |
|---|---|---|---|
| 0 | drillA | 0.184 px | (0.107, -0.011, 0.719) |
| 1 | drillB | 0.155 px | (0.253, -0.027, 0.708) |
| 2 | drillC | 0.202 px | (0.197, 0.100, 0.686) |

From `target_cluster.ply` (computed from the file):
- 5,600 points; axis-aligned extent x 0.007..0.257, y -0.149..-0.074, z 0.101..0.217 m.
- Minimal oriented bounding box, sorted: **0.070 x 0.089 x 0.256 m**, centre (0.133, -0.109, 0.175).

## 5. Known caveats (read before trusting or building on this)

- `poses.json` also stores a "depth plane" check: tilt 1.4 to 3.2 degrees and offset 17 to 22 mm between the
  chessboard plane and the depth plane. The code documentation says this is a diagnostic only (L515 depth on
  the black squares is biased), but the offsets are large and **have not been investigated further**.
- The per-view ICP report (fitness / inlier RMSE before and after, table-plane gap) was printed to the console
  during the 2026-10-30 run and **was not saved**. Re-run the command in section 2 to regenerate it, and paste
  the numbers into the work log.
- The drill's real dimensions were **not** measured, so the bounding box above is not yet validated. The
  cluster floats at z = 0.10..0.22 m above the board surface, which is plausible only if the drill was held up
  or stood on something; this has not been confirmed.
- The code was validated end-to-end on synthetic scans (`selftest.py`, 21 checks). On the real L515 the
  capture, calibration and fusion ran, but there is no ground-truth accuracy measurement yet.
- The `.npz` and `.ply` files are binary and committed directly (about 11 MB in total). Do not commit the
  `venv/` folder.

## 6. Rules for anyone (human or AI) continuing this work

1. Do not overwrite `scans/*.npz` or `poses.json`; add new files with new names and record them below.
2. Do not guess `board.cols/rows/square_size_mm`; they come from the physical board (7x6 inner corners, 25 mm).
3. Report numbers (reprojection RMSE, fitness before/after, cluster sizes, box size), not just "it worked".
4. Say what was verified and how. Anything not run on real data stays under "Not verified".
5. The camera only works from Windows. Ask the human to capture and to give the exact filenames printed.
6. Meshcat output can only be judged in a browser by a human.

## 7. Work log

| Date | Who | What | Evidence |
|---|---|---|---|
| 2026-10-30 | Toghrul | Captured drill scans A/B/C, calibrated poses, fused and segmented the drill | `scans/drill*.npz`, `poses.json`, `output/*.ply` |
| 2026-11-11 / 11-12 | Emre | Created the Diana 7 Drake repo: URDF converted from the vendor xacro, STL meshes converted to OBJ, view and hold-pose scripts | `diana7_drake/` (his commits `1711ddb`, `0b8eed1`) |
| 2026-11-21 | Toghrul | Uploaded the canonical perception files to this repo and wrote this tracking README | `022561c` |
| 2026-11-21 | Toghrul | Imported Emre's `diana7-drake` repo into `diana7_drake/` (git subtree) and extended this README; loaded the URDF headless in Drake as a check | `fdc96eb` and the commit after it |

## 8. Suggested next steps

0. Emre: confirm the joint-4 pose issue in section 9 and re-run both scripts from this repo.

1. Measure the drill with a ruler and compare with the 0.070 x 0.089 x 0.256 m box.
2. Re-run fusion with the saved console log and record the ICP fitness/RMSE per view.
3. Decide how the cloud becomes a Drake model (mesh / primitive fit) for the unscrewing simulation.

## 9. Diana 7 model (`diana7_drake/`)

Files (from Emre's repo, unchanged by the import):

| File | What it is |
|---|---|
| `diana7_drake/models/diana7/diana7.urdf` | Diana 7 arm, generated from the vendor xacro (`diana_description`). Root link `base`, joints `diana7_joint1..7`. It has no `<transmission>` tags, so the scripts add the actuators themselves |
| `diana7_drake/models/diana7/meshes/robots/diana7/{visual,collision}/link0..7.obj` | Meshes converted from STL to OBJ because Drake cannot load STL. The URDF refers to them by relative path, so the folder can be moved as a whole |
| `diana7_drake/scripts/convert_stl_to_obj.py` | One-time STL to OBJ conversion (needs `trimesh` and the vendor `diana_description` folder, which is **not** in this repo) |
| `diana7_drake/scripts/01_view_robot.py` | Builds the plant, prints joint and effort limits, shows the arm in Meshcat, simulates 5 s with zero torque |
| `diana7_drake/scripts/02_hold_pose.py` | Adds an `InverseDynamicsController` (kp 100, critical damping) and holds a pose; prints `max |q - q_des|` after 4 s |

Checked on 2026-11-21 (headless, `MultibodyPlant`, Drake 1.57 from `thesis/venv`): the URDF loads from its new
location, 7 positions, 11 bodies. Joint limits read from the URDF (rad): joints 1, 3, 5, 6, 7 are +-3.12; joint 2
is +-1.57; **joint 4 is [0, 3.05]**. Drake prints warnings that `safety_controller` tags are ignored and that
non-zero joint friction is unsupported; the friction values are therefore **not** simulated.

Open points:
- **Joint 4 vs the test pose.** Both scripts use `q = [0, 0.5, 0, -1.0, 0, 0.5, 0]`, but joint 4 only allows
  0 to 3.05 rad, so -1.0 is outside its limit. Check whether the sign of the pose or of the URDF limit is wrong
  before trusting the "hold pose" result.
- **Default paths.** `01_view_robot.py` defaults to `~/robots/diana7.urdf` (the old location), and
  `02_hold_pose.py` to `~/diana7-drake/models/diana7/diana7.urdf`. In this repo pass the path explicitly:
  `python diana7_drake/scripts/02_hold_pose.py diana7_drake/models/diana7/diana7.urdf`.
- The scripts were written by Emre and have **not** been run again after the import, so their output is not
  verified here. Meshcat can only be judged by a human in a browser.
- The Diana 7 model files come from the robot vendor's `diana_description` package. Check its licence before
  making this repo public or sharing it outside the thesis.
