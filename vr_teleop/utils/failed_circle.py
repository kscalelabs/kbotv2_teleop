from vr_teleop.utils.ik import *
from vr_teleop.ikrobot import KBot_Robot
from vr_teleop.utils.old_motion_planning import move_actuators_with_trajectory
import numpy as np
import asyncio
import logging
import xml.etree.ElementTree as ET
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # .DEBUG .INFO

def check_joint_limits(joint_name, angle, urdf_path):
    """
    Check if the angle for a joint is within the limits defined in the robot's XML file.
    
    Args:
        joint_name (str): Name of the joint to check
        angle (float): Angle in radians to check
        urdf_path (str): Path to the URDF/XML directory
        
    Returns:
        bool: True if angle is within limits, False otherwise
    """
    base_dir = os.path.dirname(urdf_path)
    robot_xml_path = os.path.join(base_dir, "robot.xml")
    
    if not os.path.exists(robot_xml_path):
        logger.warning(f"Robot XML file not found at {robot_xml_path}. Skipping joint limit check.")
        return True
    
    try:
        tree = ET.parse(robot_xml_path)
        root = tree.getroot()
        
        # Find the joint with the given name
        for joint_elem in root.findall(".//joint"):
            if joint_elem.get("name") == joint_name:
                range_attr = joint_elem.get("range")
                if range_attr:
                    min_angle, max_angle = map(float, range_attr.split())
                    if min_angle <= angle <= max_angle:
                        return True
                    else:
                        logger.error(f"Joint {joint_name} angle {angle:.4f} exceeds limits [{min_angle:.4f}, {max_angle:.4f}]")
                        return False
                else:
                    logger.warning(f"No range attribute found for joint {joint_name}. Skipping limit check.")
                    return True
        
        logger.warning(f"Joint {joint_name} not found in XML. Skipping limit check.")
        return True
    
    except Exception as e:
        logger.error(f"Error checking joint limits: {e}")
        return True  # Default to allowing the movement if there's an error

def check_solution_joint_limits(qpos, joint_names, urdf_path):
    """
    Check if the IK solution's joints are within their limits.
    
    Args:
        qpos (np.array): Joint positions from IK solution
        joint_names (list): List of joint names corresponding to qpos
        urdf_path (str): Path to the URDF directory
    
    Returns:
        bool: True if all joints are within limits, False otherwise
    """
    for i, (joint_name, angle) in enumerate(zip(joint_names, qpos)):
        if not check_joint_limits(joint_name, angle, urdf_path):
            return False
    return True

def calculate_circle_ik_solutions(center_pos=None, radius=0.2, num_points=6, leftside=False, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
    """
    Calculate IK solutions for points in a circle around a center position.
    
    Args:
        center_pos (np.array, optional): Center of the circle in 3D space. If None, uses current position.
        radius (float): Radius of the circle in meters
        num_points (int): Number of points on the circle
        leftside (bool): Whether to use the left or right arm
        urdf_path (str): Path to the robot's URDF
        
    Returns:
        tuple: (ik_solutions, circle_points, joint_names) where:
            - ik_solutions is a list of IK solutions for each point
            - circle_points is a list of 3D positions forming the circle
            - joint_names is a list of joint names corresponding to the arm
    """
    # Initialize the robot
    kbotv2 = KBot_Robot(urdf_path, gravity_enabled=False, timestep=0.001)
    
    # Set up the end effector name based on which arm we're using
    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
        # Joint names for the left arm
        joint_names = ["left_shoulder_pitch_03", "left_shoulder_roll_03", "left_shoulder_yaw_02", 
                      "left_elbow_02", "left_wrist_02"]
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"
        # Joint names for the right arm
        joint_names = ["right_shoulder_pitch_03", "right_shoulder_roll_03", "right_shoulder_yaw_02", 
                      "right_elbow_02", "right_wrist_02"]
    
    # If no center provided, use current end effector position
    if center_pos is None:
        center_pos = kbotv2.data.body(ee_name).xpos.copy()
        logger.info(f"Using current end effector position as circle center: {center_pos}")
    else:
        logger.info(f"Using provided center position: {center_pos}")
    
    # Get the current orientation for consistency
    target_ort = kbotv2.data.body(ee_name).xquat.copy()
    initial_states = kbotv2.get_limit_center(leftside=leftside)
    
    # Define a circle in the x-y plane around the specified center
    circle_points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)
        
        # Create a target position offset from the center
        target_pos = center_pos.copy()
        target_pos[0] += x_offset  # Add offset to x
        target_pos[1] += y_offset  # Add offset to y
        
        circle_points.append(target_pos)
    
    # Solve inverse kinematics for each point on the circle
    kbotv2.set_iksolve_side(leftside)
    kbotv2.reset()
    
    ik_solutions = []
    for i, point in enumerate(circle_points):
        logger.info(f"Solving IK for point {i+1}/{len(circle_points)}: {point}")
        
        calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(
            kbotv2.model, 
            kbotv2.data, 
            point, 
            target_ort, 
            initial_states, 
            leftside, 
            debug=True
        )
        
        if error_norm_pos > 0.01 or error_norm_rot > 0.1:
            logger.warning(f"IK solution has high error: pos_error={error_norm_pos}, rot_error={error_norm_rot}")
        
        if leftside:
            arm_angles = arms_tofullq(kbotv2.model, kbotv2.data, leftarmq=calc_qpos)
        else:
            arm_angles = arms_tofullq(kbotv2.model, kbotv2.data, rightarmq=calc_qpos)
        
        # Check if the solution exceeds joint limits
        if not check_solution_joint_limits(arm_angles, joint_names, urdf_path):
            raise ValueError(f"IK solution for point {i+1} exceeds joint limits")
        
        logger.info(f"Point {i+1}: Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")
        ik_solutions.append(calc_qpos)
    
    return ik_solutions

async def execute_circle_movement(ik_solutions, leftside=False, kbotv2_sim=None):
    """
    Execute robot movement to follow the points of a circle using pre-calculated IK solutions.
    
    Args:
        ik_solutions (list): List of IK solutions for each point
        leftside (bool): Whether to use the left or right arm
        kbotv2_sim (object, optional): KOS simulation instance if available
    """
    if kbotv2_sim is None:
        logger.warning("No simulation instance provided. Movement will not be executed.")
        return
    
    # Determine actuator IDs based on which arm we're using
    if leftside:
        actuator_ids = [11, 12, 13, 14, 15]  # Left arm actuator IDs
    else:
        actuator_ids = [21, 22, 23, 24, 25]  # Right arm actuator IDs
    
    # Loop through each IK solution and move to it
    for i, solution in enumerate(ik_solutions):
        # Extract the arm joint angles from the full qpos
        if leftside:
            arm_angles = arms_tofullq(kbotv2_sim.model, kbotv2_sim.data, leftarmq=solution)
        else:
            arm_angles = arms_tofullq(kbotv2_sim.model, kbotv2_sim.data, rightarmq=solution)
        
        logger.info(f"Moving to point {i+1}/{len(ik_solutions)}")
        
        # Use motion planning to smoothly move to the next point
        await move_actuators_with_trajectory(kbotv2_sim, actuator_ids, arm_angles)

async def move_robot_in_circle(center_pos=None, radius=10, num_points=6, leftside=False, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
    """
    Move the robot's end effector in a circle. This combines calculation and execution.
    
    Args:
        center_pos (np.array, optional): Center of the circle in 3D space. If None, uses current position.
        radius (float): Radius of the circle in meters
        num_points (int): Number of points on the circle
        leftside (bool): Whether to use the left or right arm
        urdf_path (str): Path to the robot's URDF
    """
    # Calculate IK solutions for the circle
    ik_solutions, _, _ = await calculate_circle_ik_solutions(
        center_pos, radius, num_points, leftside, urdf_path
    )
    
    # Create a KOS simulation instance if available
    kbotv2_sim = None  # Replace with your sim KOS instance if available
    
    # Execute the movement
    await execute_circle_movement(ik_solutions, leftside, kbotv2_sim)

def arms_tofullq(model, data, leftarmq=None, rightarmq=None):
    """
    Convert arm joint positions to full robot qpos.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        leftarmq: Left arm joint positions (5 values), can be None
        rightarmq: Right arm joint positions (5 values), can be None
        
    Returns:
        np.array: Full joint position array
    """
    # Create an empty array with model.nq dimensions
    newqpos = data.qpos.copy()
    
    # Process left arm if provided
    if leftarmq is not None:
        left_moves = {
            "left_shoulder_pitch_03": leftarmq[0],
            "left_shoulder_roll_03": leftarmq[1],
            "left_shoulder_yaw_02": leftarmq[2],
            "left_elbow_02": leftarmq[3],
            "left_wrist_02": leftarmq[4]
        }
        
        for joint_name, value in left_moves.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = model.jnt_qposadr[joint_id]

            if model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                raise ValueError(f"Joint {joint_name} is not a hinge joint. This function only works with hinge joints (1 DOF).")
            if joint_id >= 0:
                newqpos[qpos_index] = value
    
    # Process right arm if provided
    if rightarmq is not None:
        right_moves = {
            "right_shoulder_pitch_03": rightarmq[0],
            "right_shoulder_roll_03": rightarmq[1],
            "right_shoulder_yaw_02": rightarmq[2],
            "right_elbow_02": rightarmq[3],
            "right_wrist_02": rightarmq[4]
        }
        
        for joint_name, value in right_moves.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_index = model.jnt_qposadr[joint_id]

            if model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                raise ValueError(f"Joint {joint_name} is not a hinge joint. This function only works with hinge joints (1 DOF).")
            if joint_id >= 0:
                newqpos[qpos_index] = value
    
    return newqpos

async def main():
    """
    Main function that demonstrates the robot moving in a circle
    """

    temproboot = KBot_Robot("vr_teleop/kbot_urdf/scene.mjcf", gravity_enabled=False,timestep=0.001)
    llocs = temproboot.get_limit_center(leftside=True)
    rlocs = temproboot.get_limit_center(leftside=False)
    starting_pos = arms_tofullq(temproboot.model, temproboot.data, llocs, rlocs)
    temproboot.set_qpos(starting_pos)

    target_center = temproboot.data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()

    ik_solutions = calculate_circle_ik_solutions(target_center, leftside=True)
    # breakpoint()

    def runThis(robot, sim_time):
        # Keep track of the last solution index and the time it was applied
        if not hasattr(runThis, "last_index"):
            runThis.last_index = -1
            runThis.last_change_time = 0
        
        current_step = int(sim_time / 2)
        target_time = current_step * 2
        
        if sim_time > target_time and sim_time <= target_time + 0.01 and current_step > runThis.last_index:
            next_index = runThis.last_index + 1
            
            if next_index < len(ik_solutions):
                print(f"Time: {sim_time:.2f}s - Moving to position {next_index}")
                robot.set_qpos(ik_solutions[next_index])
                
                runThis.last_index = next_index
                runThis.last_change_time = sim_time

    temproboot.run_viewer(runThis)

   

    # # Example usage: Move the robot's end effector in a circle with a custom center
    # # If you want to use the current end effector position, leave center_pos as None
    # custom_center = np.array([0.3, 0.0, 1.0])  # Example custom center position
    
    # try:
    #     # Use default center (current end effector position)
    #     # await move_robot_in_circle()
        
    #     # Or use a custom center position
    #     await move_robot_in_circle(center_pos=custom_center, radius=0.1, num_points=8, leftside=False)
    # except ValueError as e:
    #     logger.error(f"Error during execution: {e}")

if __name__ == "__main__":
    asyncio.run(main())

