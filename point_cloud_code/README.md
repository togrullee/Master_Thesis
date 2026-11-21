# Multi-view point cloud pipeline (RealSense L515 + chessboard calibration)

Goal: simulate a **static 3-camera setup** with one handheld Intel RealSense L515. The camera is placed at
3 poses above the workspace, those poses are calibrated once with a printed chessboard and stored, and the
3 scans are fused into one cloud. The object is then isolated with the classic pipeline
**RANSAC table removal → ICP refinement → DBSCAN clustering**.

Context: master thesis on robotic unscrewing of e-waste (Drake). Everything here is classic geometry
(numpy, Open3D, OpenCV). No learning-based methods.

> **Status / honesty note.** All code was tested end-to-end on **synthetic** scans with known ground truth
> (`selftest.py`, 21 checks) and the capture script was tested against a **mock** camera. It has **not** been
> run on a real L515 or a real printed board yet. See [What is verified](#what-is-verified-and-what-is-not).

---

## 1. The pipeline at a glance

```
WINDOWS (camera only works here)               WSL2 Ubuntu (everything else)
─────────────────────────────────              ───────────────────────────────────────────────
capture_view.py view0/1/2  ──copy .npz──►  calibrate_poses.py ──► poses.json   (done once, board visible)
 (board + object on table)                      │
                                                ▼
                                  fuse_and_segment.py  ──► output/fused.ply
                                    1 depth → cloud per view, crop to ROI, voxel, outlier removal
                                    2 transform with poses.json   (initial alignment)
                                    3 align views 1,2 onto view 0  (table planes → ICP → planes again, FGR fallback)
                                    4 RANSAC table plane removal on the merged cloud
                                    5 DBSCAN, pick the target cluster ──► output/target_cluster.ply
                                    6 bounding-box size + Meshcat viewer
```

Two modes:

| Mode | When | Board needed in scans? |
|---|---|---|
| **Calibration** | first time, or whenever the 3 camera positions change on purpose | **yes** |
| **Reuse** | later: camera moved back *by hand* to roughly the same 3 poses | **no** – `poses.json` is only the initial guess, ICP does the real alignment |

---

## 2. Files

| File | Runs on | Purpose |
|---|---|---|
| `capture_view.py` | **Windows** (`C:\rsenv`) | grab one aligned RGB-D view → `<name>.npz`, never overwrites |
| `calibrate_poses.py` | WSL | chessboard → camera pose per view → `poses.json` (+ annotated PNGs, Meshcat check) |
| `fuse_and_segment.py` | WSL | fuse views, ICP, RANSAC, DBSCAN → `fused.ply`, `target_cluster.ply`, box size, Meshcat |
| `config.yaml` | – | **all tunable parameters** (board, ROI, voxel, ICP, RANSAC, DBSCAN, Meshcat port) |
| `view_npz.py` | WSL | look at scans: PNG of colour + depth (`output/preview_<name>.png`), `--3d` shows the cloud in Meshcat |
| `common.py` | WSL | shared helpers (npz loading, back-projection, Meshcat) |
| `synthetic.py`, `selftest.py` | WSL | synthetic scenes with ground truth + automatic tests (no camera needed) |
| `requirements-wsl.txt`, `requirements-windows.txt` | – | Python packages |

Folder layout expected by default (paths are in `config.yaml → paths`, relative to that file):

```
~/thesis/
├── scans/                    ← .npz files copied from Windows go here  (paths.scans_dir = ../scans)
├── venv/                     ← WSL virtualenv (open3d, numpy, opencv-python, pyyaml, drake)
└── point_cloud_code/         ← this folder
    ├── poses.json            ← written by calibrate_poses.py
    └── output/               ← fused.ply, target_cluster.ply, calib_view*.png
```

---

## 3. One-time setup

**Windows (PowerShell)** – the venv `C:\rsenv` (Python 3.11) already exists:
```powershell
C:\rsenv\Scripts\Activate.ps1
pip install numpy pyrealsense2==2.54.2.5684      # see requirements-windows.txt
mkdir C:\scans -ErrorAction SilentlyContinue
# copy capture_view.py to C:\scans (e.g. from \\wsl$\Ubuntu\home\maqakriti123\thesis\point_cloud_code\)
Copy-Item \\wsl$\Ubuntu\home\maqakriti123\thesis\point_cloud_code\capture_view.py C:\scans\
```
The camera is only reachable from Windows. Close the RealSense Viewer before capturing.

**WSL**:
```bash
cd ~/thesis
source venv/bin/activate
pip install -r point_cloud_code/requirements-wsl.txt     # open3d, opencv-python, pyyaml, numpy, drake
cd point_cloud_code
python selftest.py          # expect "21/21 checks passed" (takes ~1 min). Run this FIRST.
```
Tested with Python 3.14, open3d 0.20.0, opencv-python 5.0.0, numpy 2.5, Drake 1.57 (Meshcat).

---

## 4. Prepare the chessboard  ⚠ confirm these values

`config.yaml → board` holds **7 × 6 inner corners** (= a board of **8 × 7 squares**) and a square size that you must
measure. The first test board was 8 × 6 squares (7 × 5 inner corners, odd × odd): that pattern looks identical when rotated 180°, so
OpenCV cannot tell which corner is the origin and the views can disagree. It only seemed to work as "7 × 6" in two views because the
white paper margin counted as an extra row, and it failed in the third. **Print a board with 8 × 7 squares** (e.g. 25 mm squares on A4: 200 × 175 mm, leaving a white margin of ~17 mm, which the detector needs). Change the values only if you use a different board:

* `cols`, `rows` = number of **inner corners** (where 4 squares meet), *not* squares. A board with 10×7 squares has 9×6 inner corners.
* `square_size_mm` = edge of one printed square, **measured with a ruler** after printing (printers rescale!). A wrong value silently scales every pose and the fused cloud.
* Use an **odd × even** inner-corner count (7×6 or 9×6). Inner corners = squares − 1 in each direction, so an 8×6-square board is 7×5 (odd × odd) and does NOT work. With an even×even or odd×odd board OpenCV cannot tell which corner is "first" and the frame can flip 180° between views.
* To find the real size of an existing board, run `cv2.findChessboardCornersSB` for every (cols, rows) and check which sizes are found, then count the dark squares (inner corners = squares − 1).
* The paper must be **flat and rigid** (glue it to a board/plate). Print at 100 % scale, no "fit to page".
* The board **must not move** between the three captures; it defines the world frame.
* Put the object **next to** the board, not on top of it, so no corner is covered.

**World frame (board frame):** origin = one inner corner of the board (the one OpenCV returns as corner index
`(rows-1)*cols`; for an odd×even board this is a fixed *physical* corner, independent of how you hold the camera),
+X along the columns, +Y along the rows, **+Z out of the board = up**. `calib_viewN.png` draws a red circle on the
origin corner; mark it with a pen so everyone agrees on where the world origin is. Units are metres.

---

## 5. Procedure (calibration mode) – exact commands, in order

### Step 1 – capture 3 views (Windows PowerShell)
Place the board flat on the table and the object next to it. Hold the camera roughly 0.3–0.8 m above and
around the workspace, entire board **and** object in view, camera still, then:
```powershell
C:\rsenv\Scripts\Activate.ps1
cd C:\scans
python capture_view.py view0       # camera at pose 1
python capture_view.py view1       # move camera to pose 2, board and object untouched
python capture_view.py view2       # pose 3
```
Each call: warm-up, median of 30 depth frames (noise reduction), depth aligned to colour (`rs.align`), saves `C:\scans\<name>.npz`.
* **Never overwrites.** If `view0.npz` exists the file becomes `view0_2.npz`, `view0_3.npz`, … and the script prints the name it used. Note the real names.
* If no camera is found it prints a clear message and exits with code 3 (check USB 3 cable, close other apps, run from Windows not WSL).

### Step 2 – copy to WSL (Windows PowerShell)
```powershell
copy C:\scans\view0.npz \\wsl$\Ubuntu\home\maqakriti123\thesis\scans\
copy C:\scans\view1.npz \\wsl$\Ubuntu\home\maqakriti123\thesis\scans\
copy C:\scans\view2.npz \\wsl$\Ubuntu\home\maqakriti123\thesis\scans\
```
(`\\wsl.localhost\Ubuntu\...` also works. PowerShell's `copy` **overwrites silently** – if you captured a retake with a
suffix such as `view1_2.npz`, copy that exact file and pass it with `--views` below. The WSL user name/distro may differ: adjust.)

### Step 3 – calibrate the poses (WSL, once)
```bash
cd ~/thesis && source venv/bin/activate && cd point_cloud_code
python calibrate_poses.py                       # uses ../scans/view0.npz view1.npz view2.npz
# with renamed files:
python calibrate_poses.py --views ../scans/view0.npz ../scans/view1_2.npz ../scans/view2.npz
```
What it does per view: detects the chessboard (`findChessboardCornersSB`, falls back to `findChessboardCorners` + `cornerSubPix`),
runs `solvePnP` (IPPE + LM refinement) with the saved intrinsics (and distortion if present), converts to `T_world_cam`.
It also fits a plane to the *depth* inside the board and reports its tilt/offset against the PnP board plane (a sanity check;
set `board.refine_with_depth: true` to also snap the PnP board plane onto the depth plane).

Check the printout:
* `reprojection RMSE` should be **< 1 px** (typically 0.1–0.5). Larger → wrong `cols/rows/square_size_mm`, blurry image, bent paper.
* `depth-plane tilt` / `offset`: a diagnostic only. On the first real scans it read 10–21° although the reprojection error was 0.3 px, because the L515 depth on the printed black squares is unreliable (few valid pixels, biased). Do not trust it over the chessboard pose; the real check is that the table plane in the fused cloud is horizontal.
* `camera at (x, y, z)`: `z` must be **positive** (above the board) and the three positions plausible in metres. A `WARNING: camera is BELOW the board` means flipped corner ordering.
* Look at `output/calib_view*.png` (green corner overlay, red circle = origin).
* A Meshcat page opens (link printed). Three coloured spheres are the cameras (view0 red, view1 green, view2 blue); the triad at the board is X red / Y green / **Z blue pointing up**. Press Enter in the terminal to quit.

Result: `poses.json` (an existing one is kept as `poses_<timestamp>.json`). Calibration only needs to be redone if the *intended* camera poses change.
Use `--no-viz` to skip the browser.

### Step 4 – fuse and segment (WSL)
```bash
python fuse_and_segment.py                      # scans view0..view2, poses.json
python fuse_and_segment.py --views ../scans/a.npz ../scans/b.npz ../scans/c.npz   # explicit files
```
Console output, in order: points per view after ROI/voxel/outlier removal → per-view registration report
(`fitness` and `inlier RMSE` **before → after**, table-plane gap, method used) → merged size → RANSAC plane → number and sizes of DBSCAN
clusters → target cluster → **bounding-box size** (compare with a ruler).

Outputs in `output/`: `fused.ply` (all views merged, table included), `target_cluster.ply` (the object, with normals),
Meshcat scene laid out along +x: **`1_calibration_only`** (poses.json only), **`2_after_ICP`**, **`3_target_cluster`**; colours view0 red / view1 green / view2 blue.
Useful flags: `--no-icp` (calibration-only alignment), `--no-viz`, `--out-dir DIR`, `--config FILE`, `--poses FILE`.

---

## 6. Reuse mode (camera moved back by hand, board NOT in the scans)

Capture the three new views with `capture_view.py` as before (no board needed; same three rough positions, **same order**). Then:
```bash
python fuse_and_segment.py --views ../scans/new_a.npz ../scans/new_b.npz ../scans/new_c.npz
```
`--views` files are matched to `poses.json` **by order** (1st ↔ view0, 2nd ↔ view1, 3rd ↔ view2). `poses.json` is used purely as the
initial guess; views 1 and 2 are then aligned onto view 0 by the registration step. Consequences to keep in mind:

* The world frame is whatever view 0's stored pose says – it is no longer re-measured from the board. If view 0 itself is off by 2 cm, the whole cloud is off by 2 cm in the world frame (relative geometry between views is still consistent).
* **ROI margins:** the ROI crop uses the stored poses, so keep `roi.*` generous by more than your hand-placement error, especially `roi.z[0]` (< 0), because the table is needed for plane alignment.
* If views are off by more than ~4–5 cm / ~8°, ICP may fail. The Fast-Global-Registration fallback (`icp.fgr_fallback`) then kicks in automatically when fitness stays below `icp.fgr_trigger_fitness`. In synthetic tests (6 cm / 10°) it recovered but ended at ~9 mm error – put the camera back closer if you can.

---

## 7. How the alignment works (and why)

For each view k ≥ 1 against view 0, `fuse_and_segment.refine_view`:

1. **Plane alignment.** The dominant plane of each view (the table) is found with RANSAC; view k is tilted/shifted so the two table planes coincide. This fixes 3 degrees of freedom (tilt, height) that the table constrains very well. The views really do see one table, so this is always applied when both planes are horizontal in the world frame.
2. **ICP on above-table points only** (point-to-plane, estimated normals, coarse-to-fine voxel schedule `icp.voxels`, max correspondence distance = voxel × `icp.max_dist_factor`). It fixes the remaining 3 degrees of freedom (in-plane shift and yaw) from the object itself.
3. **Plane alignment again**, because ICP may have tilted the cloud by a fraction of a degree.
4. **Safeguard.** Registration metrics (`fitness` = fraction of points with a neighbour within `icp.eval_distance`, `inlier RMSE`) are measured before and after on the above-table points. ICP replaces the previous result **only if it clearly helps** (fitness up by `icp.min_fitness_gain`, or RMSE down ≥ 15 %). Otherwise the script says so and keeps the plane-aligned / calibration-only pose; if nothing helped it prints `WARNING ... KEEPING the calibration-only pose`.
5. **FGR fallback.** If fitness is still below `icp.fgr_trigger_fitness`, Fast Global Registration (FPFH features) provides a new initial guess that is refined with the same ICP schedule, and is accepted only if clearly better.

Why not plain ICP on whole clouds? The table is ~90 % of the points and a plane cannot constrain in-plane motion, so ICP happily converged to a wrong yaw with a high fitness number in testing (`icp.plane_align: false` gives that classic behaviour if you want to compare). Why not ICP on the object only? Three small blocks cannot hold the table's tilt – hence the plane stages.

After merging: voxel downsample → **RANSAC plane** (`segment_plane`, `plane.*`) removes the table → **DBSCAN** (`dbscan.*`) → the target cluster is the largest, or the one nearest to `target.point` (`target.mode: nearest`) → statistical outlier removal on the cluster → minimal oriented bounding box (the Open3D PCA-based box is also printed; it is loose for non-box shapes, e.g. about [100, 105, 190] mm instead of the true [90, 120, 150] in testing).

---

## 8. Parameters (`config.yaml`) and how to tune them

| Key | Default | Tune when |
|---|---|---|
| `board.cols/rows/square_size_mm` | 9 / 6 / 25 | **always check** (see §4) |
| `board.refine_with_depth` | false | board tilt estimate from PnP looks poor but the L515 depth plane is good |
| `cloud.max_depth / min_depth` | 1.5 / 0.1 m | workspace is farther/closer than 0.3–0.8 m |
| `cloud.voxel` | 0.003 m | too slow (increase) or object too small (decrease) |
| `cloud.sor_*` | 30, 2.0 | noisy cloud (lower std_ratio = more aggressive) |
| `roi.x/y/z` | [-0.15,0.60] × [-0.15,0.45] × [-0.05,0.30] m in **board coordinates** | always adapt to where your object sits relative to the board origin; must contain the object **and some table**. `z[0]` must stay below 0 by more than the pose error |
| `icp.voxels` / `max_iterations` | [0.010,0.005,0.003] / [50,40,30] | views start far apart → larger first voxel |
| `icp.max_dist_factor` | 2.5 | |
| `icp.eval_distance` | 0.010 m | only affects the reported fitness/RMSE and the accept/reject decision |
| `icp.min_fitness_gain` | 0.05 | ICP is rejected too often (lower) or accepted on noise (raise) |
| `icp.plane_align` | true | see §7 |
| `icp.fgr_fallback`, `fgr_trigger_fitness`, `fgr_voxel` | true, 0.5, 0.006 | |
| `plane.distance_threshold` | 0.004 m | table is thicker/noisier (increase); object is thin like a PCB (decrease) |
| `dbscan.eps / min_points` | 0.015 / 30 | object splits into several clusters (increase eps) or merges with neighbours (decrease) |
| `target.mode / point` | largest / – | the object is not the largest cluster → `nearest` + a world point |
| `viz.meshcat_port` | null | `null` = Drake picks 7000 or the next free port; set e.g. `7001` to pin the port so several viewers can run at once |

---

## 9. Data formats

**View `.npz`** (backward compatible; old files with only the first six keys still load):

| key | meaning |
|---|---|
| `depth` | HxW float32, **metres**, 0 = invalid, aligned to the colour image |
| `color` | HxWx3 uint8, **RGB** |
| `fx, fy, cx, cy` | colour-stream intrinsics (valid for the aligned depth) |
| *added:* `depth_scale`, `dist_coeffs` (5), `dist_model`, `n_frames`, `timestamp`, `serial` | optional; distortion defaults to zeros if absent; if `depth` were integer counts, `depth_scale` converts to metres |

**`poses.json`**
```json
{ "schema": 1, "created": "...", "frame": "<text description>",
  "board": {"cols": 9, "rows": 6, "square_size_m": 0.025},
  "views": { "view0": {"file": "...", "T_world_cam": [[4x4]], "reproj_rmse_px": 0.2,
                       "depth_plane_tilt_deg": 0.1, "depth_plane_offset_mm": 0.3},
             "view1": {...}, "view2": {...} } }
```
`T_world_cam` maps points from the **camera frame** (OpenCV/RealSense: x right, y down, z forward) to the **world/board frame**.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: pydrake / open3d / cv2` | WSL venv not active: `source ~/thesis/venv/bin/activate` |
| capture: "no RealSense device found" | run from Windows PowerShell, USB 3 port/cable, close RealSense Viewer |
| `chessboard ... not found` | `board.cols/rows` wrong (inner corners!), board partly out of frame, glare, blurry; see `output/calib_view*_NOT_FOUND.png`. To find the size of an unknown board, run `cv2.findChessboardCornersSB` for every (cols, rows) from 3 to 13 on a scan: the largest size that is detected is the real one. Seen on real data: a view taken at a steep angle found only 5 of the 6 corner rows → retake it from higher up / more frontal |
| reprojection RMSE > 1 px | wrong `square_size_mm`, bent paper, motion blur |
| "camera is BELOW the board" | corner ordering flipped: use an odd×even board; check the origin circle in `calib_view*.png` |
| fuse: "no points inside the ROI" | `roi.*` does not contain the object, or poses wrong → look at Meshcat `1_calibration_only` |
| "table plane is not horizontal in the world frame" | poses/ROI wrong or the largest plane is not the table |
| "ICP did not clearly improve ... KEEPING calibration-only" | little overlap between views, or already well aligned (this is fine in that case) |
| target cluster is a thin sliver / wrong object | views disagree on table height; check the "table gap" in the report; try `target.mode: nearest` |
| box size off by several cm | check `square_size_mm`; the PCA box is loose, use the minimal box line |
| Meshcat on another port than expected | Drake moves from 7000 to 7001, 7002… if the port is taken: use the printed link or set `viz.meshcat_port` |

---

## What is verified, and what is not

**Verified (synthetic data, `python selftest.py`, 21/21 checks, repeated 3× with the same outcome):**
* Chessboard detection and PnP recover the camera poses to < 1 mm / < 0.1° (noise: 1 mm depth, 2 % dropouts, anti-aliased rendering).
* Calibration-only fusion puts the object within 0.7 mm RMS of the ground truth; box size [88, 122, 152] mm vs true [90, 120, 150] mm.
* Reuse mode with views 1,2 displaced by ~2 cm / 3° and **no board visible**: 228 mm → ~1.8 mm RMS after registration (box within 15 mm).
* ICP safeguard: a deliberately bad ICP result is rejected; on already-good poses ICP is not applied (it would add ~1–2 mm by fitting noise).
* `capture_view.py` against a **mock** `pyrealsense2`: file naming never overwrites (`view0.npz`, `view0_2.npz`), keys written, "no device" message/exit code.
* Old-format `.npz` (6 keys) loads and runs; the Meshcat code paths run without errors.

**Not verified:**
* Anything on the **real L515** or a real printed board: real depth noise/black-square bias, L515 colour distortion model (coefficients are passed to OpenCV as-is; the L515 normally reports zeros), the real `board.*` values (placeholders!), real ROI values.
* Real-hardware capture (`capture_view.py` was never run against pyrealsense2 2.54.2 on Windows).
* The Meshcat pages were not looked at in a browser (only that the code runs); check the triad direction yourself.
* Robustness beyond the tested perturbations: with 6 cm / 10° initial error the synthetic test ends at ~9 mm (informational, not a pass criterion).

---

## Notes for an AI agent reproducing the experiment

1. Run `python selftest.py` first; stop and fix the environment if it does not print `21/21 checks passed`.
2. Ask the human for the real **inner-corner counts and the measured square size** and put them in `config.yaml` before step 3. Do not guess.
3. The camera can only be used from Windows; you cannot capture from WSL. Ask the human to run the Windows steps and to tell you the exact file names that `capture_view.py` printed.
4. Never overwrite `scans/*.npz` or `poses.json` by hand; the scripts already keep backups.
5. After each step compare the printout with the "check the printout" items above and report numbers (reprojection RMSE, fitness before/after, cluster sizes, box size), not just "it worked".
6. Meshcat can only be inspected by the human in a browser; ask them to confirm the board triad's blue axis points up and that the three camera spheres are above the board.

---

## ArUco cube mode (2 cubes instead of the chessboard)

`gather_cloud_aruco.py` replaces `calibrate_poses.py` + `fuse_and_segment.py` when the world reference is the printed
ArUco cubes from `aruco_cubes_tool/`. By default it finds which layout cubes are in the images (IDs 6k..6k+5 belong to
cube k); the first one (lowest id, or the first in `--cubes 1 2`) is the world frame, the others are calibrated against it. Capture still happens on
Windows with `capture_view.py`; everything else runs in WSL.

```bash
# 1. capture N views (any number, 4-8 is good) with the object between the cubes, camera 0.3-0.6 m away
python capture_view.py a1 ; python capture_view.py a2 ; ...        # Windows, C:\rsenv
# 2. WSL
python gather_cloud_aruco.py --views ../scans/a1.npz ../scans/a2.npz ../scans/a3.npz ../scans/a4.npz
python gather_cloud_aruco.py --views ... --target 0.15 0.0 0.03     # if the object is not between the cubes
python gather_cloud_aruco.py --views ... --recalibrate              # after moving a cube
python gather_cloud_aruco.py --views ... --roi -0.7 0.7 -0.7 0.7 0.012 0.4 --plane-threshold 0.02   # carpet: rough floor (roi in the table frame)
```

To look at the result in Meshcat: `python view_aruco_result.py` (or `--dir output/aruco_real`), then open the printed link
(usually http://localhost:7000) in the Windows browser. Shows only the object, the cubes (boxes with frames) and the cameras (coloured spheres with frames),
z up; add `--show-scene` to also see the floor and clutter.

Rules for the views: every marker needs about 6 px per module in the image (640x480: stay within about 0.6 m), and
**at least one view must show two faces of each cube** so that cube 1 can be placed relative to cube 0. A view needs
>= 2 markers; a single small marker gives an ambiguous (possibly flipped) pose and is rejected.

The cube-to-cube pose is a bundle adjustment over all views that show markers of several cubes (initialised from a view
with >= 2 markers on each cube), and the script prints how much it moved and a leave-one-view-out spread. Treat a spread of
several mm as a warning that more / closer / lower views are needed.

What is remembered (in `output/aruco/`):
* `aruco_poses.json`: per view `T_world_cam` and `T_cube_cam` for every cube (the camera pose w.r.t. the markers),
  reprojection error, markers used; per cube `T_world_cube` with how it was obtained.
* `cube_layout_calibrated.yaml`: the cube layout with cube 1's pose filled in. The next run reuses it automatically
  (a single view is then enough) and reports how far the cubes moved if they did (> 5 mm / 2 deg gives a warning).

Outputs: `scene_cloud.ply` (all views, cubes cut out), `object_cloud.ply` (table removed by RANSAC, DBSCAN cluster
nearest to the midpoint of the cubes or `--target`), `aruco_<view>.png` (detections and cube axes). `scene_cloud.ply` and
`object_cloud.ply` are written in a **floor frame** defined by the cubes: +z points from the visible marker faces towards the
camera (the top face of each cube is the face with a detected marker that faces the cameras; their normals are averaged),
z = 0 is the floor, one cube side below those top faces, origin below the world cube. So a cube's top face is at z = 60 mm
and an object's z is its height above the floor (carpet and all). `--floor-from plane` uses the RANSAC plane of the point
cloud instead. `--roi` and `--target` are given in this frame (z = height above the floor). `--flip-z` writes everything with z inverted (a 180 deg rotation about x, not a mirror;
the table stays z = 0, things above it get negative z); `--roi` / `--target` are unaffected. `aruco_poses.json` keeps the camera / cube poses in the world-cube frame
(`T_world_cam`) and in the table frame (`T_table_cam`, `T_table_cube`, `table_frame.T_table_world`). Cloud, plane and DBSCAN
parameters come from `config.yaml`.

`python selftest_aruco.py` (synthetic scenes with ground truth, about 20 s once the views are rendered, 2 min the first
time): cube 1 pose 0.21 mm / 0.03 deg, camera poses worst 0.61 mm / 0.12 deg, object cloud median 0.3 mm from the true
surface, stored-layout reuse, moved-cube detection and the failure modes all pass.

First real data (2026-11-21, 10 views, egg carton on carpet, cubes 1 and 2, 0.6-0.96 m, mostly top-down, markers only
20-47 px wide): 9/10 views got a pose (reprojection RMSE 0.2-1.5 px; one view showed a single marker and was rejected);
cube 2 relative to cube 1: the closed-form estimate from the only view with 2 markers per cube was **29 mm / 4.8 deg off**
compared with the bundle adjustment, which has a leave-one-out spread of 6 mm / 1.2 deg; the two cubes come out at the same
height (2 mm) and 2.6 deg relative tilt, as two cubes on one floor should; views agree to about 5 mm RMS; the object
cluster (plane threshold 0.02 m) is 5.4 x 12.0 x 31 cm, not yet compared with a ruler. The default 4 mm plane threshold
of config.yaml fails on carpet (the floor merges into one cluster). So: usable for a rough object cloud, **not** mm
accurate. Better data: closer (< 0.6 m), 40-50 deg above the floor so top and side faces of both cubes are visible together,
more views with both cubes in them, or larger cubes/markers.

**Not verified:** accuracy against a ruler on real data, the L515 depth bias between views (no ICP refinement is done
here), and the colour-camera lens distortion (the L515 reports k1=0.146, k2=-0.475, k3=0.416, used for the marker poses
but ignored when back-projecting depth, as in the older scripts).
