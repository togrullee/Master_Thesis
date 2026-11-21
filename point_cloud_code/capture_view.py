"""Capture one RealSense L515 view and save it as <name>.npz  (run on WINDOWS, in C:\\rsenv).

    python capture_view.py view0
    python capture_view.py view1 --out C:\\scans

Never overwrites: if view0.npz exists the file is saved as view0_2.npz, then view0_3.npz, ...
The name actually used is printed at the end; pass it to the WSL scripts with --views.

.npz keys (the first six are the original format, the rest are additions):
    depth        HxW float32, metres, 0 = invalid, ALIGNED to the colour image
    color        HxWx3 uint8, RGB
    fx fy cx cy  colour-stream intrinsics (valid for the aligned depth)
    depth_scale  metres per raw depth unit
    dist_coeffs  5 distortion coefficients of the colour stream
    dist_model   distortion model name
    n_frames     number of depth frames that were median-filtered
    timestamp    ISO-8601 local time of capture
    serial       camera serial number
"""
import argparse
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:                                    # pragma: no cover
    rs = None


def unique_path(out_dir, name):
    """Return (path, file object) for a not-yet-existing <name>.npz, adding _2, _3, ... if needed."""
    n = 1
    while True:
        stem = name if n == 1 else f"{name}_{n}"
        path = os.path.join(out_dir, stem + ".npz")
        try:
            return path, open(path, "xb")              # 'x' = fail if it exists, so no race/overwrite
        except FileExistsError:
            n += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="view name, e.g. view0, view1, view2")
    ap.add_argument("--out", default=r"C:\scans", help=r"output folder (default C:\scans)")
    ap.add_argument("--frames", type=int, default=30, help="depth frames to median-average (default 30)")
    ap.add_argument("--warmup", type=int, default=30, help="frames to discard first (default 30)")
    ap.add_argument("--max-dist", type=float, default=1.5, help="zero out depth beyond this many metres")
    ap.add_argument("--timeout-ms", type=int, default=15000)
    args = ap.parse_args(argv)

    if rs is None:
        print("ERROR: pyrealsense2 is not installed in this Python. Activate C:\\rsenv first:\n"
              "       C:\\rsenv\\Scripts\\Activate.ps1")
        return 2
    if len(rs.context().query_devices()) == 0:
        print("ERROR: no RealSense device found. Check the USB cable (use a USB 3 port), that the\n"
              "       L515 is not open in another program (e.g. RealSense Viewer), and that you are\n"
              "       running this from Windows PowerShell, not from WSL.")
        return 3

    os.makedirs(args.out, exist_ok=True)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        print(f"ERROR: could not start the camera pipeline: {e}")
        return 3

    try:
        dev = profile.get_device()
        depth_scale = dev.first_depth_sensor().get_depth_scale()
        serial = dev.get_info(rs.camera_info.serial_number)
        align = rs.align(rs.stream.color)              # depth pixels now match colour pixels

        for _ in range(args.warmup):
            pipe.wait_for_frames(args.timeout_ms)

        stack, color, intr = [], None, None
        for _ in range(args.frames):
            frames = align.process(pipe.wait_for_frames(args.timeout_ms))
            d, c = frames.get_depth_frame(), frames.get_color_frame()
            stack.append(np.asanyarray(d.get_data()).astype(np.float32) * depth_scale)
            color = np.asanyarray(c.get_data())
            intr = c.profile.as_video_stream_profile().intrinsics   # colour intrinsics = aligned depth
    finally:
        pipe.stop()

    stack = np.array(stack)
    stack[stack == 0] = np.nan                         # 0 = invalid pixel, ignore it in the median
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        depth = np.nan_to_num(np.nanmedian(stack, axis=0)).astype(np.float32)
    depth[depth > args.max_dist] = 0

    path, fh = unique_path(args.out, args.name)
    with fh:
        np.savez(fh, depth=depth, color=color,
                 fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                 depth_scale=depth_scale,
                 dist_coeffs=np.array(intr.coeffs, dtype=np.float64),
                 dist_model=str(intr.model),
                 n_frames=len(stack),
                 timestamp=datetime.now().isoformat(timespec="seconds"),
                 serial=serial)
    print(f"saved {path}  (valid depth pixels: {(depth > 0).mean():.2f})")
    if os.path.basename(path) != args.name + ".npz":
        print(f"NOTE: {args.name}.npz already existed, so this view was saved as "
              f"{os.path.basename(path)}. Use that name in later steps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
