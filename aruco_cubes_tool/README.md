# ArUco cube generator

`generate_aruco_cubes.py` writes printable black-and-white ArUco cube nets (one A4-landscape page per cube) and
the matching geometry file `cube_layout_nominal.yaml`, for camera pose estimation (handheld RealSense L515).
Defaults: `DICT_5X5_100`, 4 cubes, cube 60 mm, marker 45 mm. Cube k uses IDs 6k..6k+5 in the face order
+Z, +X, -Z, -X, +Y, -Y.

## Setup and run (WSL2 Ubuntu)

```bash
cd ~/Master_Thesis/aruco_cubes_tool
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # numpy, opencv-contrib-python, reportlab, pyyaml, pymupdf
python generate_aruco_cubes.py           # -> aruco_cubes/cube_00..03.pdf, all_cubes.pdf, cube_layout_nominal.yaml
```

Options: `--dict`, `--num-cubes`, `--cube-mm`, `--marker-mm`, `--page {A4,A3,LETTER}` (always landscape),
`--out-dir`, `--tab-mm` (8), `--margin-mm` (8), `--marker-measured-mm`. The script stops with an error if the white
margin around the marker is smaller than one marker module, if the net with tabs does not fit the page, or if the
dictionary has too few IDs. Re-running gives byte-identical files.
`pymupdf` is only used by the PDF checks; without it those checks are reported as SKIP.

## Print

1. Open `aruco_cubes/cube_00.pdf` (or `all_cubes.pdf`) and print at **100% / Actual size, never "fit to page"**,
   on plain matte white paper (glossy paper gives glare), black only, highest quality.
2. Measure the printed **100 mm scale bar** with calipers. If it is not 100.0 mm, the printer rescaled: fix the print
   settings first.
3. All lines are 8 mm or more from the page edge, so no printer margin setting should clip anything.

## Assemble

Solid lines = cut, dashed lines = fold. Cut along the solid outline (including the tabs).
1. Crease **every dashed line so that the printed side ends up on the outside** (the paper bends away from you
   when you look at the print). A ruler edge helps.
2. Close the ring of four side faces: glue the tab at the left end of the strip to the inside of the face at the
   other end.
3. The ring now has an open top and bottom. The -Y square (below +Z) is the bottom cap and the +Y square (above +Z)
   is the top cap; each is already attached to +Z. Fold the -Y cap in first: put glue on the three tabs along the
   bottom edges of the other ring faces and press them against the inside of the cap from the open top.
4. Last, the +Y cap: put glue on the three tabs along the top edges, close the cap and press from outside with a flat
   ruler. Use a glue stick or thin double-sided tape (liquid glue wrinkles the paper and bends the faces).
5. Keep the faces flat and the edges sharp: a dented cube gives a bad pose.

Put the glue on the side of the tab that is on the same side of the paper as the printing (the virtual fold below
shows that this side ends up against the back of the neighbouring face). Which edges carry tabs is printed when the
script runs (`glue tabs on cube edges of: ...`).

## Measure the printed marker and regenerate the YAML

Printers scale slightly, and the pose scale depends on the marker size.
1. Measure the **black square** (outer edge of the black border to the other outer edge) with calipers on several
   faces, in both directions, and average, for example 44.8 mm.
2. Regenerate only the geometry (the PDFs are not touched):
   ```bash
   python generate_aruco_cubes.py --marker-measured-mm 44.8
   ```
   This writes `aruco_cubes/cube_layout_measured.yaml` (`marker_size_source: measured`) and leaves
   `cube_layout_nominal.yaml` as it is. Use the same `--num-cubes/--cube-mm/--dict` as for printing.
3. The cube side is not measured by that option. If you measure the folded cube, add it to the same command
   (`--cube-mm 59.8 --marker-measured-mm 44.8`). The face centres are placed at half the cube side.

## YAML format

Top level: `dictionary`, `cube_side_m`, `marker_side_m`, `marker_size_source`, `cubes`. Per cube: `cube_id`,
`pose_world` (4x4, identity for cube 0, `null` for the others until calibrated), `markers`. Per marker: `id`, `face`,
`marker_side_m`, `corners_cube_m` (4x3, metres, cube frame): corners of the black square in the OpenCV ArUco order
top-left, top-right, bottom-right, bottom-left as seen from outside. Face frames (right, up, normal) are
the table in the specification; the script asserts right x up = normal and also folds the net in 3D to confirm it.

## Validation (runs at the end of every generation)

| # | Check | What it proves |
|---|---|---|
| 0 | net + tabs fit the page with 8 mm margin, nothing overlaps | layout is printable |
| 1 | IDs unique and inside the dictionary | |
| 2 | right x up = normal, folded net = table, YAML corners agree | orientations are right-handed and consistent |
| 3 | synthetic image of each cube (marker images warped by the YAML corners, 1920x1080, fx=fy=1300, 0.5 m), detection, one `solvePnP` over all corners; limits 0.5 mm / 0.1 deg | YAML corner order and face orientation match the markers |
| 4, 4b, 4c | scale bar is 100 mm (drawn coordinates, and read back from the PDF); page counts | |
| 5, 5b | cube_00.pdf rendered at 300 dpi: all 6 IDs detected, corner order and position match the YAML convention | the printed markers match the YAML |
| 6 | every face is pure white outside its marker | no lines or text in the quiet zone |
| 7 | vector only, no images, every fill, stroke and text colour is exactly 0,0,0 | |

Check 3 uses `CORNER_REFINE_APRILTAG`. With the default `CORNER_REFINE_SUBPIX` the same images give 0.7 to 1.0 mm
and up to 0.13 deg (printed as `[INFO]`, not counted). This is a property of the detector's corner estimate on
blurred edges. It suggests using `CORNER_REFINE_APRILTAG` in the real pipeline, but that has **not** been tested on
real L515 images.

## Virtual print-and-fold test

```bash
python simulate_print_and_fold.py        # after generate_aruco_cubes.py; also writes aruco_cubes/virtual_photos/*.png
```

Paper cannot be printed by the script, so it treats every `cube_NN.pdf` as a printed sheet. It reads only the PDF
(dashed lines = folds, solid = cuts, filled paths = markers) and its 300 dpi raster, and uses the YAML only as the
thing under test:

| # | Check |
|---|---|
| P1-P4 | sheet is exactly A4 landscape; everything printed is >= 8 mm from the sheet edges; six markers of exactly 45 mm; one 100 mm bar |
| F0-F3 | faces, 5 face folds and 7 tab folds are found in the PDF; the sheet is folded in 3D (90 deg about each dashed line, away from the viewer): the six faces end up on +-X, +-Y, +-Z once each, the cube closes exactly, the print is on the outside |
| F4-F5 | every printed marker (ID and corner order read from the raster) lands on the face the YAML says, and its folded corners equal the YAML corners (max 0.08 mm) |
| T1-T2 | every glue tab folds into the plane of its neighbour face, inside that face's outline; the 5 folds + 7 tabs cover all 12 cube edges exactly once; no two tabs overlap once folded |
| V1 | the folded sheet, with the PDF raster as texture, is photographed at 0.5 m; markers are detected and one `solvePnP` with the nominal YAML recovers the pose |
| S1-S2 | printer-scaling scenario (98%): the nominal YAML is 10.4 mm off, the YAML from the measured sizes 0.29 mm |

Last run: 53 PASS, 0 FAIL. This test found a real defect that the generator's own checks missed: with the first
tab shape, two tabs on perpendicular edges of a cap overlapped once folded (they fold into the same plane). The taper
is now 1.25 x the tab height, and the generator refuses cube sizes too small for it.

## What is not verified

- Anything physical: the actual print (scale, black density, paper, your printer's real margins), the real folded cube
  (paper thickness, glue, how flat the faces stay), the L515 image quality. The virtual test models perfect paper.
- Detection of the printed cubes by the real camera. All detection checks use synthetic or rendered images.
- Pose accuracy on real data. The 0.5 mm / 0.1 deg result is for noise-free synthetic renderings only.
