import pybullet as p
import pybullet_data
import time
import numpy as np

# Initialize PyBullet simulation
p.connect(p.GUI)
print(pybullet_data.getDataPath())
breakpoint()
# Load the robot from the URDF file
robot_id = p.loadURDF("./vr_teleop/kbot_urdf/robot.urdf", useFixedBase=True)

# Set gravity
p.setGravity(0, 0, -9.81)

# Define the arm joints we want to control
right_arm_joint_names = [
    "right_shoulder_pitch_03",
    "right_shoulder_roll_03",
    "right_shoulder_yaw_02",
    "right_elbow_02",
    "right_wrist_02"
]

# Identify the end-effector joint
end_effector_joint_name = "right_wrist_02"
end_effector_index = None

# Get joint indices
num_joints = p.getNumJoints(robot_id)
joint_indices = {}
right_arm_indices = []

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode("utf-8")
    joint_indices[joint_name] = i
    
    if joint_name in right_arm_joint_names:
        right_arm_indices.append(i)
    
    if joint_name == end_effector_joint_name:
        end_effector_index = i

if end_effector_index is None:
    raise ValueError(f"End effector joint '{end_effector_joint_name}' not found in URDF.")

# Print joint info for debugging
print("\nRight Arm Joints:")
for name in right_arm_joint_names:
    if name in joint_indices:
        joint_idx = joint_indices[name]
        joint_info = p.getJointInfo(robot_id, joint_idx)
        print(f"{name}: index={joint_idx}, type={joint_info[2]}, limits=[{joint_info[8]}, {joint_info[9]}]")

# Define target position for IK in world coordinates - ensure the arm can reach this
target_position_world = [0.4, 0.3, 0.3]  # X, Y, Z in world space
target_orientation_world = p.getQuaternionFromEuler([0, 0, 0])  # No rotation

# Get robot base position & orientation
base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)

# Get all movable joints (we'll use this for visualization, but IK will use only arm joints)
movable_joints = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED]

# Define an initial pose that forces the elbow to be bent
initial_pose = [0.0] * len(right_arm_indices)
# Set a non-zero angle for the elbow joint to encourage bending
elbow_index = right_arm_joint_names.index("right_elbow_02")
if elbow_index >= 0:
    initial_pose[elbow_index] = 0.7  # Set initial elbow angle to encourage bending

# Apply initial pose
for i, joint_index in enumerate(right_arm_indices):
    p.resetJointState(robot_id, joint_index, initial_pose[i])

# Visual markers
p.addUserDebugText("Target", target_position_world, textColorRGB=[1, 0, 0], textSize=1.5)
target_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1])
target_visual_id = p.createMultiBody(0, baseVisualShapeIndex=target_sphere, basePosition=target_position_world)

# Add a visual marker for the end effector
ee_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 0, 1, 1])
ee_visual_id = p.createMultiBody(0, baseVisualShapeIndex=ee_sphere, basePosition=[0, 0, 0])

# Create a debug parameter to control target position interactively
try:
    target_x = p.addUserDebugParameter("Target X", -0.5, 0.5, target_position_world[0])
    target_y = p.addUserDebugParameter("Target Y", -0.5, 0.5, target_position_world[1])
    target_z = p.addUserDebugParameter("Target Z", 0.0, 1.0, target_position_world[2])
    use_sliders = True
except:
    print("User debug parameters not supported in this environment. Using fixed target.")
    use_sliders = False

# Line ID for debug line
line_id = None

# Run the simulation loop
try:
    while True:
        # Update target position - either from sliders or use default
        if use_sliders:
            try:
                target_position_world = [
                    p.readUserDebugParameter(target_x),
                    p.readUserDebugParameter(target_y),
                    p.readUserDebugParameter(target_z)
                ]
            except Exception as e:
                print(f"Error reading sliders: {e}. Using default target.")
                # Just continue with the current target position
        
        # Update target visual position
        p.resetBasePositionAndOrientation(target_visual_id, target_position_world, [0, 0, 0, 1])
        
        # Important: Transform world coordinates to local coordinates relative to the robot base
        inv_base_position, inv_base_orientation = p.invertTransform(base_position, base_orientation)
        target_position_local, _ = p.multiplyTransforms(inv_base_position, inv_base_orientation, target_position_world, [0, 0, 0, 1])
        
        # Get current joint positions of the arm
        current_positions = [p.getJointState(robot_id, j)[0] for j in right_arm_indices]
        
        # Set preferred angles (rest pose) to encourage a natural pose with bent elbow
        rest_poses = current_positions.copy()  # Start with current positions
        # Bias the elbow to prefer a bent position
        rest_poses[elbow_index] = 0.7  # Prefer a bent elbow
        
        try:
            # Compute inverse kinematics targeting the end effector (wrist) position
            ik_solution = p.calculateInverseKinematics(
                robot_id, 
                end_effector_index,  # Use the wrist joint as the end effector
                targetPosition=target_position_local,  # Target position for the end effector
                targetOrientation=target_orientation_world,
                maxNumIterations=200
            )
            
            # Extract the solution for the arm joints
            arm_solution = []
            for i in range(len(right_arm_indices)):
                if i < len(ik_solution):
                    arm_solution.append(ik_solution[i])
                else:
                    arm_solution.append(current_positions[i])
                    
            # Apply the IK solution only to the arm joints
            for i, joint_idx in enumerate(right_arm_indices):
                # Apply position control
                p.setJointMotorControl2(
                    robot_id, 
                    joint_idx, 
                    p.POSITION_CONTROL, 
                    targetPosition=arm_solution[i],
                    force=100  # Use appropriate force
                )
        except Exception as e:
            print(f"IK calculation error: {e}")
        
        # Get the current end effector position (wrist)
        ee_state = p.getLinkState(robot_id, end_effector_index)
        ee_position = ee_state[0]  # Position in world coordinates
        
        # Update end effector visual marker
        p.resetBasePositionAndOrientation(ee_visual_id, ee_position, [0, 0, 0, 1])
        
        # Draw a line from end effector to target
        if line_id is not None:
            p.removeUserDebugItem(line_id)
        line_id = p.addUserDebugLine(
            ee_position, 
            target_position_world, 
            lineColorRGB=[0, 1, 0], 
            lineWidth=1
        )
        
        # Calculate and display distance error
        distance = np.linalg.norm(np.array(ee_position) - np.array(target_position_world))
        print(f"\rEnd effector distance to target: {distance:.4f} m    ", end="")
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1/240)  # 240 Hz
except KeyboardInterrupt:
    print("\nSimulation stopped by user")
    
p.disconnect()
