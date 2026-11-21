"""Look at capture .npz files.  (WSL)

    python view_npz.py ../scans/drill0_2.npz ../scans/drill1_2.npz     # saves output/preview_<name>.png
    python view_npz.py ../scans/drill0_2.npz --3d                      # also shows the point cloud in Meshcat

The PNG shows the colour image (left) and the depth image (right, near = red, far = blue, black = no depth).
Open the PNG in VS Code by clicking it in the Explorer.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from common import HERE, backproject, load_config, load_view, meshcat_cloud, resolve, start_meshcat, wait_for_user


def preview(view, out_dir):
    d = view.depth
    valid = d > 0
    vis = np.zeros((*d.shape, 3), np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = cv2.applyColorMap(((1 - norm) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[~valid] = 0
    img = np.hstack([cv2.cvtColor(view.color, cv2.COLOR_RGB2BGR), vis])
    path = out_dir / f"preview_{view.name}.png"
    cv2.imwrite(str(path), img)
    rng = f"{d[valid].min():.2f}-{d[valid].max():.2f} m" if valid.any() else "none"
    print(f"{view.name}: {d.shape[1]}x{d.shape[0]}, valid depth {valid.mean():.0%}, range {rng}, "
          f"timestamp {view.extra.get('timestamp', '-')}  ->  {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--3d", dest="show3d", action="store_true", help="show the point clouds in Meshcat (camera frame)")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out_dir = resolve(cfg, cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    views = [load_view(f) for f in args.files]
    for v in views:
        preview(v, out_dir)
    if args.show3d:
        meshcat = start_meshcat(cfg["viz"]["meshcat_port"])
        for i, v in enumerate(views):
            pts, cols = backproject(v, cfg["cloud"]["max_depth"], cfg["cloud"]["min_depth"])
            T = np.eye(4)
            T[0, 3] = i * 1.2                                  # place the clouds side by side
            meshcat_cloud(meshcat, f"scan/{v.name}", pts, cols, cfg["viz"]["point_size"], T)
        print("Meshcat: each cloud is in its own CAMERA frame (x right, y down, z forward), 1.2 m apart.")
        wait_for_user()
    return 0


if __name__ == "__main__":
    sys.exit(main())
