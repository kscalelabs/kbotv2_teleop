import mujoco
import mujoco.viewer
import time
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger

import csv
import os

logger = setup_logger(__name__)
logger.setLevel(logging.INFO) #.DEBUG .INFO

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
                    pass
                    # logger.warning("elbowing was clamped")
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
    max_iteration = 1000;
    tol = 0.01;
    step_size = 0.8
    damping = 0.5

    rot_w = 0.7
    trans_w = 1  

    # For logging to file
    arm_positions = []
    error_norms = []

    if initialstate is None:
        if leftside:
            initialstate = np.array([ 0.2617995, -0.5846855,  0.       , -1.2653635,  0.       ])
        else:
            initialstate = np.array([-0.2617995,  0.5846855,  0.       ,  1.2653635,  0.       ])

    # Define multiple initial states to try when stuck
    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
        initial_states = [
            initialstate.copy(),
            np.array([0, 0, 1.5, -0.5, 0]),
            np.array([2, -1.3, -1.5, -0.5, 0]), 
            np.array([[2, -1.3, 1.5, -0.5, 0]]),
            # np.array([
            #     np.random.uniform(-2.094395, 2.617994),  # left_shoulder_pitch
            #     np.random.uniform(-1.658063, 0.488692),  # left_shoulder_roll
            #     np.random.uniform(-1.745329, 1.745329),  # left_shoulder_yaw
            #     np.random.uniform(-2.530727, 0),         # left_elbow
            #     np.random.uniform(-1.745329, 1.745329)   # left_wrist
            # ])
        ]
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"
        initial_states = [
            initialstate.copy(),
            np.array([0, 0, -1.5, 0.5, 0]),
            np.array([-2, 1.3, -1.5, 0.5, 0]),
            np.array([-2, 1.3, 1.5, 0.5, 0]),
            # np.array([
            #     np.random.uniform(-2.617994, 2.094395),  # right_shoulder_pitch
            #     np.random.uniform(-0.488692, 1.658063),  # right_shoulder_roll
            #     np.random.uniform(-1.745329, 1.745329),  # right_shoulder_yaw
            #     np.random.uniform(0, 2.530727),          # right_elbow
            #     np.random.uniform(-1.745329, 1.745329)   # right_wrist
            # ])
        ]

    mujoco.mj_forward(model, data)
    
    current_init_state_idx = 0
    next_pos_arm = initial_states[current_init_state_idx].copy()
    next_pos = arms_to_fullqpos(model, data, next_pos_arm.flatten(), leftside)
    
    for i in range(max_iteration):
        ee_pos, ee_rot = forward_kinematics(model, data, next_pos, leftside=leftside)
        error = np.subtract(target_pos, ee_pos)
        error_norm_pos = np.linalg.norm(error)

        error_rot = orientation_error(target_ort, ee_rot)
        error_norm_rot = np.linalg.norm(error_rot)

        # Store current arm position and error for tracking
        if debug:
            arm_positions.append(next_pos_arm.copy())
            error_norms.append((error_norm_pos, error_norm_rot))

        # Check if we should try a different initial state
        if i > 0 and i % (max_iteration // len(initial_states)) == 0 and error_norm_pos > 0.1:
            current_init_state_idx = (current_init_state_idx + 1) % len(initial_states)
            logger.info(f"Switching to initial state {current_init_state_idx} after {i} iterations due to high error: {error_norm_pos:.6f}")
            next_pos_arm = initial_states[current_init_state_idx].copy()
            next_pos = arms_to_fullqpos(model, data, next_pos_arm.flatten(), leftside)
            continue
        

        #* Find Next Pos
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        n = jacp.shape[1]
        I = np.identity(n)

        data.qpos = next_pos
        mujoco.mj_forward(model, data)
        mujoco.mj_jac(model, data, jacp, jacr, target_pos, model.body(ee_name).id)
        

        #*Term for Translation:
        J_inv = np.linalg.inv( jacp.T @ jacp + damping * I) @ jacp.T
        #*Term for Rotation:
        JR_inv = np.linalg.inv(jacr.T @ jacr + damping * I) @ jacr.T
        
        delta_q =  trans_w*(J_inv @ error) + rot_w*(JR_inv @ error_rot)
        

        # prev_pos_arm = next_pos_arm.copy()
        prev_pos = next_pos.copy()

        next_pos = prev_pos + delta_q * step_size
        next_pos = joint_limit_clamp(model, next_pos)
        next_pos_arm = slice_dofs(model, data, next_pos, leftside)
        next_pos_arm = next_pos_arm.flatten()


        if i % 88 == 0:
            # Calculate the row-reduced echelon form (RREF) of the Jacobian
            # This helps analyze the linear independence of the Jacobian rows
            # and identify potential singularities
            logger.debug(f"Rank is: {np.linalg.matrix_rank(jacp)}")
            logger.debug(f"Iteration {i}, Error: {error}, Error norm: {error_norm_pos:.6f}")
            logger.debug(f"Jacobian (position): {jacp}, and the Jacobian Transpose: {jacp.T}")
            logger.debug(f"Jacobian Transpose * Error * Alpha (0.8): {np.linalg.norm(delta_q)}, full: {delta_q}")
            jac_det = np.linalg.det(jacp @ jacp.T)
            logger.debug(f"Jacobian determinant: {jac_det:.6f}")
            logger.debug(f"Target position: {target_pos}")
            logger.debug(f"Current end effector position: {ee_pos}")
            logger.debug(f"Current joints: {next_pos_arm}")
            # breakpoint()

        if error_norm_pos < tol and error_norm_rot < tol:
            logger.info(f"Converged in {i} iterations")
            logger.info(f"Final end effector position: {error_norm_pos}")
            logger.info(f"Final orientation error norm: {error_norm_rot}")
            
            # Save tracking data to CSV if debug is enabled
            if debug and arm_positions:
                save_arm_positions_to_csv(arm_positions, error_norms, leftside, converged=True)
            
            mujoco.mj_resetData(model, data)
            return next_pos, error_norm_pos, error_norm_rot
    
    logger.warning(f"Failed to converge after {max_iteration} iterations, error: {error_norm_pos:.6f} and rot: {error_norm_rot}")
    
    # Save tracking data to CSV even if we didn't converge
    if debug and arm_positions:
        save_arm_positions_to_csv(arm_positions, error_norms, leftside, converged=False)
    
    mujoco.mj_resetData(model, data)
    return next_pos, error_norm_pos, error_norm_rot
