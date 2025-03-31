import pybullet as p
import numpy as np
import time
import logging
from vr_teleop.utils.logging import setup_logger
import csv
import os

logger = setup_logger(__name__)
logger.setLevel(logging.DEBUG)

def joint_limit_clamp(robot, input_qpos):
    """Clamp joint positions to their limits"""
    clamped_qpos = input_qpos.copy()
    
    for i, joint_idx in enumerate(robot.joint_indices):
        if i < len(input_qpos):
            limit_lower = robot.joint_lower_limits[i]
            limit_upper = robot.joint_upper_limits[i]
            
            # Check if joint has limits (non-zero values)
            if limit_lower != 0 or limit_upper != 0:
                prev_value = clamped_qpos[i]
                clamped_qpos[i] = max(limit_lower, min(clamped_qpos[i], limit_upper))
                new_value = clamped_qpos[i]
                
                if prev_value != new_value:
                    joint_info = p.getJointInfo(robot.robot_id, joint_idx)
                    joint_name = joint_info[1].decode('utf-8')
                    logger.debug(f"Updated joint {joint_name} value from {prev_value:.6f} to {new_value:.6f}, limits: [{limit_lower:.6f}, {limit_upper:.6f}]")
                    if "elbow" in joint_name:
                        logger.warning("Elbow joint was clamped")
    
    return clamped_qpos

def orientation_error(target_quat, current_quat):
    """Calculate orientation error between two quaternions"""
    # PyBullet uses quaternions in [x,y,z,w] format
    # We need to convert if they are in different formats
    
    # Check if quaternions are in [w,x,y,z] format (MuJoCo format)
    # and convert to [x,y,z,w] for processing if needed
    if len(target_quat) == 4 and target_quat[0] > 0.9:  # Likely a [w,x,y,z] format
        target_quat_pb = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
    else:
        target_quat_pb = target_quat
        
    if len(current_quat) == 4 and current_quat[0] > 0.9:  # Likely a [w,x,y,z] format
        current_quat_pb = [current_quat[1], current_quat[2], current_quat[3], current_quat[0]]
    else:
        current_quat_pb = current_quat
    
    # Calculate difference quaternion
    # For a difference quat q, we have: target_quat = q * current_quat
    # In PyBullet, we can compute the difference quaternion as:
    inv_current = p.invertTransform([0, 0, 0], current_quat_pb)[1]  # Get the inverse quaternion
    diff_quat = p.multiplyTransforms([0, 0, 0], target_quat_pb, [0, 0, 0], inv_current)[1]
    
    # Convert to axis-angle representation for the error
    axis, angle = p.getAxisAngleFromQuaternion(diff_quat)
    
    # Normalize and scale by angle to get the orientation error
    if np.linalg.norm(axis) < 1e-10:
        return np.zeros(3)  # No rotation needed
    
    # Scale axis by angle for the error term
    error = np.array(axis) * angle
    return error

def forward_kinematics(robot, joint_angles, leftside: bool):
    """
    Compute forward kinematics using PyBullet
    """
    # Save current state
    state_id = p.saveState()
    
    # Set joint positions
    for i, joint_idx in enumerate(robot.joint_indices):
        if i < len(joint_angles):
            p.resetJointState(robot.robot_id, joint_idx, joint_angles[i])
    
    # Get end effector position and orientation
    if leftside:
        ee_index = robot.left_ee_index
    else:
        ee_index = robot.right_ee_index
    
    if ee_index != -1:
        link_state = p.getLinkState(robot.robot_id, ee_index)
        pos = np.array(link_state[0])  # Position
        ort = np.array(link_state[1])  # Quaternion [x,y,z,w]
        # Convert quaternion to match MuJoCo format [w,x,y,z] if needed
        ort = np.array([ort[3], ort[0], ort[1], ort[2]])  
    else:
        logger.error(f"End effector index not found for {'left' if leftside else 'right'} arm")
        pos = np.zeros(3)
        ort = np.array([1.0, 0.0, 0.0, 0.0])  # Default orientation (no rotation)
    
    # Restore original state
    p.restoreState(state_id)
    
    return pos, ort

def save_arm_positions_to_csv(arm_positions, error_norms, leftside, converged=True):
    """
    Save the tracked arm positions and errors to a CSV file.
    """
    if not arm_positions:
        logger.warning("No arm positions to save")
        return None
        
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    arm_side = 'left' if leftside else 'right'
    status = "" if converged else "_unconverged"
    filename = f'logs/ik_solving.csv'
    
    # Write to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        joint_names = []
        if leftside:
            joint_names = ["left_shoulder_pitch", "left_shoulder_roll", 
                          "left_shoulder_yaw", "left_elbow", "left_wrist"]
        else:
            joint_names = ["right_shoulder_pitch", "right_shoulder_roll", 
                          "right_shoulder_yaw", "right_elbow", "right_wrist"]
        
        header = joint_names + ["pos_error", "rot_error", "iteration"]
        writer.writerow(header)
        
        # Write data
        for idx, (pos, errors) in enumerate(zip(arm_positions, error_norms)):
            row = list(pos) + [errors[0], errors[1], idx]
            writer.writerow(row)
    
    logger.info(f"Arm position tracking saved to {filename}")
    return filename

def pb_inverse_kinematics(robot, target_pos, target_ort=None, leftside=False):
    """
    Use PyBullet's built-in IK solver to compute inverse kinematics
    
    Args:
        robot: The PB_KBot robot instance
        target_pos: Target end effector position as a 3D array [x, y, z]
        target_ort: Optional target orientation as a quaternion [x, y, z, w]
        leftside: Boolean indicating if this is for the left arm
        
    Returns:
        Joint positions for the arm
    """
    # Determine which end effector to use
    if leftside:
        ee_index = robot.left_ee_index
        joint_names = [
            "left_shoulder_pitch_03",
            "left_shoulder_roll_03",
            "left_shoulder_yaw_02",
            "left_elbow_02",
            "left_wrist_02"
        ]
    else:
        ee_index = robot.right_ee_index
        joint_names = [
            "right_shoulder_pitch_03",
            "right_shoulder_roll_03",
            "right_shoulder_yaw_02",
            "right_elbow_02",
            "right_wrist_02"
        ]
    
    if ee_index == -1:
        logger.error(f"End effector not found for {'left' if leftside else 'right'} arm")
        return np.zeros(5)  # Return default joint values
    
    # Find joint indices for the arm
    arm_joint_indices = []
    for name in joint_names:
        for i in range(robot.num_joints):
            joint_info = p.getJointInfo(robot.robot_id, i)
            if joint_info[1].decode('utf-8') == name:
                arm_joint_indices.append(i)
                break
    
    if len(arm_joint_indices) != 5:
        logger.error(f"Could not find all 5 joints for the {'left' if leftside else 'right'} arm")
        return np.zeros(5)  # Return default joint values
    
    # Prepare joint limit arrays specifically for the arm
    lowerLimits = []
    upperLimits = []
    jointRanges = []
    restPoses = []
    
    for joint_idx in arm_joint_indices:
        joint_info = p.getJointInfo(robot.robot_id, joint_idx)
        lowerLimits.append(joint_info[8])
        upperLimits.append(joint_info[9])
        jointRanges.append(joint_info[9] - joint_info[8])
        restPoses.append(p.getJointState(robot.robot_id, joint_idx)[0])
    
    # Format target orientation if provided
    if target_ort is not None:
        # Convert from [w,x,y,z] to [x,y,z,w] if in MuJoCo format
        if len(target_ort) == 4 and target_ort[0] > 0.9:  # Likely a [w,x,y,z] format
            target_ort_pb = [target_ort[1], target_ort[2], target_ort[3], target_ort[0]]
        else:
            target_ort_pb = target_ort
    else:
        # If no orientation specified, use current orientation
        current_state = p.getLinkState(robot.robot_id, ee_index)
        target_ort_pb = current_state[1]  # The current quaternion
    
    # Use PyBullet's IK solver with joint limit parameters
    ik_solution = p.calculateInverseKinematics(
        robot.robot_id,
        ee_index,
        targetPosition=target_pos,
        targetOrientation=target_ort_pb,
        lowerLimits=lowerLimits,
        upperLimits=upperLimits,
        jointRanges=jointRanges,
        restPoses=restPoses,
        maxNumIterations=100,
        residualThreshold=1e-5
    )
    
    # Extract only the arm joint values (first 5 values)
    arm_solution = np.array(ik_solution[:5])
    
    return arm_solution

def accurateIK(robot, target_pos, target_ort=None, leftside=False, maxIter=5, threshold=1e-3):
    """
    More accurate IK that iteratively refines the solution for more precision
    
    Args:
        robot: The PB_KBot robot instance
        target_pos: Target end effector position as a 3D array [x, y, z]
        target_ort: Optional target orientation as a quaternion
        leftside: Boolean indicating if this is for the left arm
        maxIter: Maximum number of iterations for refinement
        threshold: Distance threshold for considering the solution acceptable
        
    Returns:
        (arm_solution, position_error, orientation_error)
    """
    if leftside:
        ee_index = robot.left_ee_index
    else:
        ee_index = robot.right_ee_index
    
    if ee_index == -1:
        logger.error(f"End effector not found for {'left' if leftside else 'right'} arm")
        return np.zeros(5), 1e10, 1e10
    
    # Save current state to restore at the end
    original_state_id = p.saveState()
    
    # Initial IK solution
    arm_solution = pb_inverse_kinematics(robot, target_pos, target_ort, leftside)
    
    # Prepare for iterative refinement
    closeEnough = False
    iteration = 0
    position_error = 1e30
    orientation_error = 0.0
    
    # Get full qpos
    current_qpos = np.zeros(len(robot.joint_indices))
    for i, joint_idx in enumerate(robot.joint_indices):
        current_qpos[i] = p.getJointState(robot.robot_id, joint_idx)[0]
    
    # Convert arm solution to full qpos
    full_qpos = robot.convert_armqpos_to_fullqpos(
        leftarmq=arm_solution if leftside else None,
        rightarmq=None if leftside else arm_solution
    )
    
    # Iterative refinement loop
    while not closeEnough and iteration < maxIter:
        # Apply the current solution to check its accuracy
        for i, joint_idx in enumerate(robot.joint_indices):
            p.resetJointState(robot.robot_id, joint_idx, full_qpos[i])
        
        # Get the resulting end effector position
        link_state = p.getLinkState(robot.robot_id, ee_index)
        current_pos = np.array(link_state[0])
        current_ort = np.array(link_state[1])
        
        # Calculate position difference
        pos_diff = target_pos - current_pos
        position_error = np.linalg.norm(pos_diff)
        
        # Calculate orientation error if target orientation was provided
        if target_ort is not None:
            # Convert current_ort from [x,y,z,w] to [w,x,y,z] for consistency
            current_ort_w_first = np.array([current_ort[3], current_ort[0], current_ort[1], current_ort[2]])
            ort_diff = orientation_error(target_ort, current_ort_w_first)
            orientation_error = np.linalg.norm(ort_diff)
        else:
            orientation_error = 0.0
        
        logger.debug(f"IK iteration {iteration}, position error: {position_error:.6f}, orientation error: {orientation_error:.6f}")
        
        # Check if we're close enough
        closeEnough = (position_error < threshold)
        
        if not closeEnough and iteration < maxIter - 1:
            # Get a refined IK solution
            new_arm_solution = pb_inverse_kinematics(robot, target_pos, target_ort, leftside)
            
            # Update the full qpos with the new arm solution
            full_qpos = robot.convert_armqpos_to_fullqpos(
                leftarmq=new_arm_solution if leftside else None,
                rightarmq=None if leftside else new_arm_solution
            )
            
            # Update the arm solution
            arm_solution = new_arm_solution
        
        iteration += 1
    
    # Restore original state
    p.restoreState(original_state_id)
    
    logger.debug(f"IK solver finished after {iteration} iterations, final position error: {position_error:.6f}")
    
    return arm_solution, position_error, orientation_error

def ik_gradient(robot, target_pos, target_ort=None, leftside=False):
    """
    Calculate IK using PyBullet's built-in solver, then refine using iterative method.
    
    Args:
        robot: The PB_KBot robot instance
        target_pos: Target end effector position as a 3D array [x, y, z]
        target_ort: Optional target orientation as a quaternion
        leftside: Boolean indicating if this is for the left arm
        
    Returns:
        (full_delta_q, error_norm_pos, error_norm_rot)
    """
    start_time = time.time()
    
    # Use our improved accurateIK function instead of the basic pb_inverse_kinematics
    arm_solution, position_error, orientation_error = accurateIK(
        robot, 
        target_pos, 
        target_ort, 
        leftside, 
        maxIter=1,
        threshold=1e-3
    )
    
    # Get the current full joint state
    current_qpos = np.zeros(len(robot.joint_indices))
    for i, joint_idx in enumerate(robot.joint_indices):
        current_qpos[i] = p.getJointState(robot.robot_id, joint_idx)[0]
    
    # Convert arm solution to full delta_q
    full_delta_q = np.zeros_like(current_qpos)
    
    # Identify which joints to update
    if leftside:
        joint_names = [
            "left_shoulder_pitch_03",
            "left_shoulder_roll_03",
            "left_shoulder_yaw_02",
            "left_elbow_02",
            "left_wrist_02"
        ]
    else:
        joint_names = [
            "right_shoulder_pitch_03",
            "right_shoulder_roll_03",
            "right_shoulder_yaw_02",
            "right_elbow_02",
            "right_wrist_02"
        ]
    
    # Map arm solution to full_delta_q
    for i, name in enumerate(joint_names):
        for j, joint_idx in enumerate(robot.joint_indices):
            joint_info = p.getJointInfo(robot.robot_id, joint_idx)
            if joint_info[1].decode('utf-8') == name:
                # Calculate the change needed
                full_delta_q[j] = arm_solution[i] - current_qpos[j]
                break
    
    # Debug logging
    logger.debug(f"IK calculation time: {time.time() - start_time:.4f}s")
    logger.debug(f"Position error: {position_error:.4f}, Orientation error: {orientation_error:.4f}")
    
    return full_delta_q, position_error, orientation_error 