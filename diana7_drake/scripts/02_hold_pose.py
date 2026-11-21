"""
Step 2: add a controller to the Diana 7 diagram.

Diagram built here:

    ConstantVectorSource --desired state--> InverseDynamicsController --torques--> MultibodyPlant
                                                  ^                                    |
                                                  +---------- estimated state ---------+
                                                                  (plant -> SceneGraph -> Meshcat as before)

Run:
    ~/drake-env/bin/python scripts/02_hold_pose.py ~/diana7-drake/models/diana7/diana7.urdf
"""
import os
import sys
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    MultibodyPlant,
    Parser,
    StartMeshcat,
    MeshcatVisualizer,
    Simulator,
    InverseDynamicsController,
    ConstantVectorSource,
)

urdf_path = os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else "~/diana7-drake/models/diana7/diana7.urdf")


def add_diana7(plant):
    """Load the Diana 7 into `plant`, weld it to the world and add joint actuators.

    A function because we need the SAME robot twice: once in the simulated world, and once
    in a stand-alone plant the controller uses as its internal model (see below).
    """
    robot = Parser(plant).AddModels(urdf_path)[0]
    # Fix the root link "base" to the world so the arm cannot float away.
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base", robot))
    # URDF has no <transmission> tags, so we create one actuator per revolute joint.
    # Without actuators the plant would have no torque input to connect the controller to.
    for j in plant.GetJointIndices(robot):
        joint = plant.get_joint(j)
        if joint.type_name() == "revolute":
            plant.AddJointActuator(f"act_{joint.name()}", joint)
    plant.Finalize()
    return robot


meshcat = StartMeshcat()
builder = DiagramBuilder()

# --- 1) The simulated world (the "real robot" stand-in) ----------------------------------
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-3)
robot = add_diana7(plant)
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

# --- 2) The controller --------------------------------------------------------------------
# InverseDynamicsController computes: tau = M(q) * (kp*(q_des-q) + kd*(v_des-v)) + C(q,v) + G(q)
# i.e. it cancels gravity/Coriolis using a MODEL of the robot, then makes the joints behave
# like simple spring-dampers. It needs its own copy of the robot as that model; a plant with
# no SceneGraph and time_step=0 (continuous) is enough, since it only does math.
control_plant = MultibodyPlant(time_step=0.0)
add_diana7(control_plant)

nq = plant.num_positions(robot)  # 7
# With gravity/inertia cancelled, every joint acts like a unit-mass spring-damper, so
# kd = 2*sqrt(kp) gives critical damping (fastest response without overshoot).
# kp = 100 -> settles in roughly half a second. Raise it for stiffer tracking, but
# real hardware has delays: start low there.
kp = np.full(nq, 100.0)
kd = 2.0 * np.sqrt(kp)
ki = np.zeros(nq)  # no integral action: not needed because the model is exact in simulation

controller = builder.AddSystem(InverseDynamicsController(control_plant, kp, ki, kd, False))

# Wires:
#  plant state [q; v] -> controller  (measurement; on hardware this comes from the encoders)
builder.Connect(plant.get_state_output_port(robot),
                controller.get_input_port_estimated_state())
#  controller torques -> plant actuation  (command; on hardware this goes to the motors)
builder.Connect(controller.get_output_port_control(),
                plant.get_actuation_input_port(robot))

# --- 3) The reference (what we want the arm to do) ----------------------------------------
# Desired STATE is [q_des; v_des] (14 numbers). A constant source = "hold this pose".
q_des = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])  # same bent pose as step 1 (rad)
reference = builder.AddSystem(ConstantVectorSource(np.concatenate([q_des, np.zeros(nq)])))
builder.Connect(reference.get_output_port(), controller.get_input_port_desired_state())

diagram = builder.Build()

# Save the block diagram; open diana7_controller_diagram.svg to see the new blocks and wires.
try:
    import pydot
    svg = pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=1))[0].create_svg()
    with open("diana7_controller_diagram.svg", "wb") as f:
        f.write(svg)
    print("Wrote diana7_controller_diagram.svg")
except Exception as e:
    print("Skipping diagram SVG:", e)

# --- 4) Run -------------------------------------------------------------------------------
context = diagram.CreateDefaultContext()
plant_context = plant.GetMyMutableContextFromRoot(context)
# Start with all joints at zero, away from the target, so we SEE the controller move the arm.
plant.SetPositions(plant_context, robot, np.zeros(nq))
plant.SetVelocities(plant_context, robot, np.zeros(nq))

sim = Simulator(diagram, context)
sim.set_target_realtime_rate(1.0)
meshcat.StartRecording()
sim.AdvanceTo(4.0)
meshcat.PublishRecording()

# Report the final error: should be ~0 if the controller holds the pose.
q_final = plant.GetPositions(plant_context, robot)
print("max |q - q_des| after 4 s [rad]:", np.max(np.abs(q_final - q_des)))
input("Meshcat is live. Press Enter to quit.")
