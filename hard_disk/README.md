# Hard disk point cloud (October 2026)

Early single-camera test: a 3.5-inch hard disk drive lying on a white sheet on a table, captured with the RealSense L515
(`capture_view.py`, 640x480, camera fixed, **the object was turned 90 degrees between the 4 views**). The scripts are the
originals from `~/thesis/`; they use paths relative to their own folder (`scans/...`), so they run unchanged from here once the raw
scans are put back.

| File | What it is |
|---|---|
| `scans/box_merged.ply` | **The result**: the hard disk, 4,494 points with colours, the 4 views merged. Written 2026-10-21 by `process_views.py` |
| `process_views.py` | Reads `scans/box/view_00..03.npz`. Per view: back-project depth (voxel 3 mm), find the table plane around a seed pixel (RANSAC), keep points 6-150 mm above it within 15 cm of the seed, DBSCAN, biggest cluster. Then point-to-plane ICP chains view k onto view k-1 starting from a 90 degree rotation about the table normal (it tries both rotation signs and keeps the one with the higher total ICP fitness), merges, prints the oriented box size, writes `scans/box_merged.ply`, shows it in Meshcat |
| `process_box.py` | The first, single-view version: reads `scans/box_view0.npz`, removes the table plane, keeps the biggest DBSCAN cluster, prints the box size, shows it in Meshcat |

**Raw scans are not in this repo** (10.6 MB; they show the lab and, in `view_00`, a person's arm): `scans/box/view_00..03.npz`
and `scans/box_view0.npz`. They are on Toghrul's machine in `~/thesis/scans/` and `C:\scans`. Put them in `hard_disk/scans/`
to re-run.

Numbers (computed from `box_merged.ply`): oriented bounding box **15.6 x 11.3 x 3.1 cm**. A 3.5-inch drive is normally about
14.7 x 10.2 x 2.6 cm, so the cloud is plausible but each side is 0.5-1 cm larger (noise and outlier points at the edges). **The
actual drive was not recorded or measured**, so this is not a validated size. The ICP fitness / RMSE per view that
`process_views.py` prints was not saved.
