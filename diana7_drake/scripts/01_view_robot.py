"""
Step 1: load the Diana 7 URDF into a Drake Diagram, check it, and view it in Meshcat.

Run (from the project folder):
    ~/drake-env/bin/python scripts/01_view_robot.py
Optional arguments: path to urdf, path to the diana_description folder.
"""
import os
import sys
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    StartMeshcat,
    MeshcatVisualizer,
    Simulator,
)

# --- Paths -------------------------------------------------------------------
# The URDF we generated from the vendor's xacro, and the vendor package folder.
# Defaults match the earlier setup; override via command line if yours differ.
urdf_path = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/robots/diana7.urdf")
pkg_path = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/robots/diana_description")

# Meshcat = browser-based 3D viewer (opens at the URL printed in the console).
meshcat = StartMeshcat()

# DiagramBuilder collects all "systems" (blocks) and the wires between them.
builder = DiagramBuilder()

# MultibodyPlant = rigid-body dynamics of the robot (M(q)qdd + C(q,qd) + G(q) = tau).
# SceneGraph = geometry engine (visual + collision shapes), used by the visualizer.
# time_step > 0 -> discrete-time plant, the usual choice in Drake.
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)

# The URDF refers to meshes as package://diana_description/meshes/...
# Drake must be told which folder "diana_description" is, otherwise the meshes
# are not found. This is why we pointed it at the cloned vendor repo.
parser = Parser(plant)
robot = parser.AddModels(urdf_path)[0]  # returns a ModelInstanceIndex

# Weld the root link "base" (from the URDF) to the world. Without this the arm is a
# free-floating body and simply falls under gravity.
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", robot))

# Drake only creates actuators from <transmission> tags. If the URDF has none, we add
# one actuator per revolute joint ourselves; otherwise we could not apply torques later
# (e.g. from a controller). Must be done BEFORE Finalize().
if plant.num_actuators() == 0:
    # Drake does not copy <limit effort=...> from the URDF into actuators we add by hand
    # (the default limit would be infinite), so read the efforts from the URDF ourselves.
    import xml.etree.ElementTree as ET
    effort = {}
    for jt in ET.parse(urdf_path).getroot().iter("joint"):
        lim = jt.find("limit")
        if lim is not None and lim.get("effort") is not None:
            effort[jt.get("name")] = float(lim.get("effort"))
    for j in plant.GetJointIndices(robot):
        joint = plant.get_joint(j)
        if joint.type_name() == "revolute":
            plant.AddJointActuator(f"act_{joint.name()}", joint,
                                   effort_limit=effort.get(joint.name(), np.inf))

# After Finalize() the plant structure is frozen (no more bodies/joints/actuators).
plant.Finalize()

# Adds a visualizer system and wires SceneGraph -> Meshcat for us.
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

diagram = builder.Build()

# --- Sanity checks: do these BEFORE any control work, descriptions are often wrong ----
print("positions (nq):", plant.num_positions(robot), "<- expect 7")
print("actuators:", plant.num_actuators(), "<- expect 7")
# Position limits live on the joint; effort (torque) limits live on the actuator.
for i in plant.GetJointActuatorIndices():
    act = plant.get_joint_actuator(i)
    joint = act.joint()
    print(f"{joint.name():16s} limits [{joint.position_lower_limits()[0]:+.2f}, "
          f"{joint.position_upper_limits()[0]:+.2f}] rad, "
          f"effort limit {act.effort_limit():.0f} Nm")

# Context = numerical state of the whole diagram. Systems are stateless; contexts hold state.
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(context)

# Start from a simple bent pose (rad). Joint limits above tell you what is allowed;
# adjust if this pose violates one.
plant.SetPositions(plant_context, robot, np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0]))

# Draw one frame so the model shows up in the browser right away.
diagram.ForcedPublish(context)

# Block diagram of the system as an SVG file (needs: pip install pydot, plus the
# graphviz program: sudo apt install graphviz).
try:
    import pydot
    svg = pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].create_svg()
    with open("diana7_diagram.svg", "wb") as f:
        f.write(svg)
    print("Wrote diana7_diagram.svg")
except Exception as e:  # missing graphviz is the usual reason
    print("Skipping diagram SVG:", e)

# Zero-torque simulation: with no controller the arm should sag under gravity.
# If it floats, masses/inertias are missing; if it explodes, check the inertias.
sim = Simulator(diagram, context)
sim.set_target_realtime_rate(1.0)
meshcat.StartRecording()
sim.AdvanceTo(5.0)
meshcat.PublishRecording()
input("Meshcat is live (open the URL above). Press Enter to quit.")
