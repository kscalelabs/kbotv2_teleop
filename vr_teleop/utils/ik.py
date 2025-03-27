import mujoco
import mujoco.viewer
import time
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger

import csv
import os

logger = setup_logger(__name__)
logger.setLevel(logging.DEBUG)

pi = np.pi

r_kinematic_chain = ["YOKE_STOP_INNER", "RS03_5", "R_Bicep_Lower_Drive", "R_Forearm_Upper_Structural", "KB_C_501X_Bayonet_Adapter_Hard_Stop"]
l_kinematic_chain = ["YOKE_STOP_INNER_2", "RS03_6", "L_Bicep_Lower_Drive", "L_Forearm_Upper_Drive", "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"]

r_end_effector = ["KB_C_501X_Bayonet_Adapter_Hard_Stop"]
l_end_effector = ["KB_C_501X_Bayonet_Adapter_Hard_Stop_2"]


# * IK LOGIC
def joint_limit_clamp(model, full_qpos):
    # prev_qpos = full_qpos.copy()

    for i in range(model.nq):
        if model.jnt_limited[i]:
            prev_value = full_qpos[i].copy()
            full_qpos[i] = max(model.jnt_range[i][0], min(full_qpos[i], model.jnt_range[i][1]))
            new_value = full_qpos[i].copy()
            if prev_value != new_value:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
                logger.debug(f"Updated joint {joint_name} value from {prev_value:.6f} to {new_value:.6f}, limits: [{model.jnt_range[i][0]:.6f}, {model.jnt_range[i][1]:.6f}]")
                if joint_name == "right_elbow_02":
                    logger.warning("elbowing was clamped")
                    # pass
                    # breakpoint()
    
    return full_qpos

def arms_to_fullqpos(model, data, inputqloc: dict, leftside: bool):
        moves = {}

        newqpos = data.qpos.copy()

        if leftside:
            moves = {
                "left_shoulder_pitch_03": inputqloc[0],
                "left_shoulder_roll_03": inputqloc[1],
                "left_shoulder_yaw_02": inputqloc[2],
                "left_elbow_02": inputqloc[3],
                "left_wrist_02": inputqloc[4]
            }
        else:
            moves = {
                "right_shoulder_pitch_03": inputqloc[0],
                "right_shoulder_roll_03": inputqloc[1],
                "right_shoulder_yaw_02": inputqloc[2],
                "right_elbow_02": inputqloc[3],
                "right_wrist_02": inputqloc[4]
            }

        for key, value in moves.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
            qpos_index = model.jnt_qposadr[joint_id]

            if model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                raise ValueError(f"Joint {key} is not a hinge joint. This function only works with hinge joints (1 DOF).")
                return 
            if joint_id >= 0:
                newqpos[qpos_index] = value
        
        return newqpos

def slice_dofs(model, data, input_list, leftside: bool):
    if len(input_list.shape) == 1:
        input_list = input_list.reshape(1, -1)

    if leftside:
        tjoints = [
            "left_shoulder_pitch_03",
            "left_shoulder_roll_03",
            "left_shoulder_yaw_02",
            "left_elbow_02",
            "left_wrist_02"
        ]
    else:
        tjoints = [
            "right_shoulder_pitch_03",
            "right_shoulder_roll_03",
            "right_shoulder_yaw_02",
            "right_elbow_02",
            "right_wrist_02"
        ]
    
    joint_indices = []
    
    for key in tjoints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
        if joint_id >= 0:  # Check if joint exists
            qpos_index = model.jnt_qposadr[joint_id]
            joint_indices.append(qpos_index)
        else:
            print(f"Warning: Joint '{key}' not found in model")

    x_shape = input_list.shape[0]
    num_joints = len(joint_indices)
    result = np.zeros((x_shape, num_joints))
    
    for i in range(x_shape):
        for j, joint_idx in enumerate(joint_indices):
            result[i, j] = input_list[i, joint_idx]

    return result


def orientation_error(target_quat, current_quat):
    cur_quat_conj = np.zeros(4)
    mujoco.mju_negQuat(cur_quat_conj, current_quat)
    dq = np.zeros(4)
    mujoco.mju_mulQuat(dq, target_quat, cur_quat_conj)

    # Add a small epsilon to prevent division by zero
    norm = np.linalg.norm(dq)
    if norm < 1e-10:  # Check if norm is very small
        # logger.warning('quat norm is small')
        return np.zeros(3)  # Return zero error if quaternions are nearly identical
        
    dq = dq / norm
    if dq[0] < 0:
        dq = -dq
    return 2 * dq[1:4]


def forward_kinematics(model, data, joint_angles, leftside: bool):
    """
    Compute forward kinematics of given joint angles by MuJoCo
    Go to position and read position
    """
    mujoco.mj_resetData(model, data)
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    data.qpos = joint_angles
    mujoco.mj_forward(model, data)

    pos = data.body(ee_name).xpos.copy()

    ort = data.body(ee_name).xquat.copy()

    mujoco.mj_resetData(model, data)
    return pos, ort


def save_arm_positions_to_csv(arm_positions, error_norms, leftside, converged=True):
    """
    Save the tracked arm positions and errors to a CSV file.
    
    Args:
        arm_positions: List of arm joint positions at each iteration
        error_norms: List of (position_error, rotation_error) tuples at each iteration
        leftside: Boolean indicating if this is the left arm
        converged: Boolean indicating if IK successfully converged
    """
    if not arm_positions:
        logger.warning("No arm positions to save")
        return None
        
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a filename with timestamp
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
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

def inverse_kinematics(model, data, target_pos, target_ort, leftside: bool, initialstate=None, debug=False):
    max_iteration = 80;
    tol = 0.05;
    step_size = 0.8
    damping = 0.2

    rot_w = 1
    trans_w = 1

    # For logging to file
    arm_positions = []
    error_norms = []

    if initialstate is None:
        if leftside:
            initialstate = np.array([ 0.2617995, -0.5846855,  0.       , -1.2653635,  0.       ])
        else:
            initialstate = np.array([-0.2617995,  0.5846855,  0.       ,  1.2653635,  0.       ])
    
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"

    mujoco.mj_forward(model, data)
    
    next_pos_arm = initialstate.copy()
    next_pos = arms_to_fullqpos(model, data, next_pos_arm.flatten(), leftside)

    I = np.identity(model.nv)
    
    for i in range(max_iteration):
        # Time full iteration
        iter_start = time.time()
        
        # Time forward kinematics
        fk_start = time.time()
        ee_pos, ee_rot = forward_kinematics(model, data, next_pos, leftside=leftside)
        fk_time = time.time() - fk_start
        
        logger.warning(f"EE pos: {ee_pos}, EE target pos: {target_pos}")
        error = np.subtract(target_pos, ee_pos)
        error_norm_pos = np.linalg.norm(error)

        if target_ort is not None:
            error_rot = orientation_error(target_ort, ee_rot)
            error_norm_rot = np.linalg.norm(error_rot)
        else:
            error_rot = np.zeros(3)
            error_norm_rot = 0

        # Store current arm position and error for tracking
        if debug:
            arm_positions.append(next_pos_arm.copy())
            error_norms.append((error_norm_pos, error_norm_rot))

        #* Find Next Pos
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        data.qpos = next_pos
        
        mj_fwd_time = time.time()
        mujoco.mj_forward(model, data)
        mj_fwd_time = time.time() - mj_fwd_time

        mj_jac_time = time.time()
        mujoco.mj_jac(model, data, jacp, jacr, target_pos, model.body(ee_name).id)
        mj_jac_time = time.time() - mj_jac_time

        solve_start = time.time()
        
        A = jacp.T @ jacp + damping * I
        delta_q = trans_w * np.linalg.solve(A, jacp.T @ error)
        pos_solve_time = time.time() - solve_start

        solve_start = time.time()
        # Time orientation optimization (if needed)
        if target_ort is not None:
            A_rot = jacr.T @ jacr + damping * I
            delta_q += rot_w * np.linalg.solve(A_rot, jacr.T @ error_rot)

        rot_solve_time = time.time() - solve_start
        
        # Time position update and joint limit enforcement
        prev_pos = next_pos.copy()
        next_pos = prev_pos + delta_q * step_size
        next_pos = joint_limit_clamp(model, next_pos)
        next_pos_arm = slice_dofs(model, data, next_pos, leftside)
        next_pos_arm = next_pos_arm.flatten()

        # logger.warning(f"MJ Fwd time: {mj_fwd_time * 1000:.2f} ms, MJ Jac time: {mj_jac_time * 1000:.2f} ms, Pos solve time: {pos_solve_time * 1000:.2f} ms, Rot solve time: {rot_solve_time * 1000:.2f} ms")

        if error_norm_pos < tol and error_norm_rot < tol:
            #* Converged Return
            return next_pos_arm, error_norm_pos, error_norm_rot
    
    return next_pos_arm, error_norm_pos, error_norm_rot

def ik_gradient(model, data, target_pos, target_ort, leftside: bool, initialstate=None, debug=False):
    """
    Calculate a single step gradient for inverse kinematics without iterating.
    Returns the delta_q values that would be used to update joint positions.
    """
    damping = 0.2
    rot_w = 1
    trans_w = 1
    
    if initialstate is None:
        if leftside:
            initialstate = np.array([0.2617995, -0.5846855, 0.0, -1.2653635, 0.0])
        else:
            initialstate = np.array([-0.2617995, 0.5846855, 0.0, 1.2653635, 0.0])
    
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    
    mujoco.mj_forward(model, data)
    
    # Set up initial state
    pos_arm = initialstate.copy()
    full_pos = arms_to_fullqpos(model, data, pos_arm.flatten(), leftside)
    
    # Calculate current end effector position and orientation
    ee_pos, ee_rot = forward_kinematics(model, data, full_pos, leftside=leftside)
    
    # Calculate position error
    error_pos = np.subtract(target_pos, ee_pos)
    error_norm_pos = np.linalg.norm(error_pos)
    
    # Calculate orientation error if target orientation is provided
    if target_ort is not None:
        error_rot = orientation_error(target_ort, ee_rot)
        error_norm_rot = np.linalg.norm(error_rot)
    else:
        error_rot = np.zeros(3)
        error_norm_rot = 0
    
    # Set up Jacobian calculation
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    I = np.identity(model.nv)
    
    start_time = time.time()

    # Setup state for Jacobian calculation
    data.qpos = full_pos
    mujoco.mj_forward(model, data)
    mujoco.mj_jac(model, data, jacp, jacr, ee_pos, model.body(ee_name).id)


    
    # Calculate position gradient
    A = jacp.T @ jacp + damping * I
    delta_q = trans_w * np.linalg.solve(A, jacp.T @ error_pos)
    
    # Add orientation gradient if needed
    if target_ort is not None:
        A_rot = jacr.T @ jacr + damping * I
        delta_q += rot_w * np.linalg.solve(A_rot, jacr.T @ error_rot)
    
    calc_time = time.time() - start_time
    # logger.warning(f"Jacobian time: {calc_time * 1000:.4f} milliseconds")

    logger.warning(f"Error: {error_norm_pos}")

    return delta_q, error_norm_pos, error_norm_rot

