"""
One-time conversion: Diana 7 URDF with .stl meshes -> URDF with .obj meshes (Drake format).

Why: Drake cannot load .stl files, only .obj/.vtk/.gltf. The vendor repo is left
untouched; everything is written to a new output folder that you can keep in your project.

Usage:
    ~/drake-env/bin/pip install trimesh
    ~/drake-env/bin/python convert_stl_to_obj.py \
        [input_urdf] [vendor_package_folder] [output_folder]

Defaults:
    input_urdf            ~/robots/diana7.urdf
    vendor_package_folder ~/robots/diana_description
    output_folder         ~/diana7-drake/models/diana7
"""
import os
import re
import sys
import trimesh

urdf_in = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/robots/diana7.urdf")
pkg_dir = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/robots/diana_description")
out_dir = os.path.expanduser(sys.argv[3] if len(sys.argv) > 3 else "~/diana7-drake/models/diana7")

with open(urdf_in) as f:
    urdf_text = f.read()

# Match every mesh reference of the form package://diana_description/<relative path>.stl
# The capture group keeps the relative path (e.g. meshes/robots/diana7/visual/link0).
pattern = re.compile(r'package://diana_description/(.+?)\.stl', re.IGNORECASE)

converted = {}  # remember files already converted: visual/collision meshes can repeat

def convert(match):
    rel_no_ext = match.group(1)                       # e.g. meshes/robots/diana7/visual/link0
    if rel_no_ext not in converted:
        src = os.path.join(pkg_dir, rel_no_ext + ".stl")
        dst = os.path.join(out_dir, rel_no_ext + ".obj")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # force="mesh" merges multi-part STL scenes into one mesh; export writes .obj
        # (inferred from the file extension). Vertex coordinates are copied unchanged,
        # so units and the <mesh scale=...> attribute in the URDF stay valid.
        mesh = trimesh.load(src, force="mesh")
        mesh.export(dst)
        converted[rel_no_ext] = dst
    # Replace with a path RELATIVE to the URDF file. Drake resolves relative paths
    # against the URDF's own folder, so no package_map entry is needed afterwards.
    return rel_no_ext + ".obj"

new_urdf_text = pattern.sub(convert, urdf_text)

os.makedirs(out_dir, exist_ok=True)
urdf_out = os.path.join(out_dir, "diana7.urdf")
with open(urdf_out, "w") as f:
    f.write(new_urdf_text)

print(f"Converted {len(converted)} meshes to .obj")
print(f"Wrote {urdf_out}")
