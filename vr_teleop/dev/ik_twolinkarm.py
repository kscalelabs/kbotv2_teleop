import mujoco
import mujoco.viewer
import time
import numpy as np
import logging
from vr_teleop.utils.mujoco_helper import *
from vr_teleop.utils.logging import setup_logger


# Set up basic logging
logger = setup_logger(__name__)
logger.setLevel(logging.DEBUG) #.DEBUG


# Load the model and create data
model = mujoco.MjModel.from_xml_path("vr_teleop/twolinkarm/twolinkarm.xml")
data = mujoco.MjData(model)

# Reset data and disable gravity
mujoco.mj_resetData(model, data)
model.opt.timestep = 0.001  # 0.001 = 1000hz
model.opt.gravity = [0, 0, 0]  # Disable gravity

target_time = time.time()
sim_time = 0.0
mujoco.mj_step(model, data)


ansqpos = data.qpos.copy()
ansqpos[0] = np.radians(60)
ansqpos[1] = np.radians(-30)
data.qpos = ansqpos.copy()
mujoco.mj_forward(model, data)
link2_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
link2_pos = data.xpos[link2_body_id].copy()
logger.info(f"Link2 body position: {link2_pos}")
link2_mat = data.xmat[link2_body_id].reshape(3, 3)
endpoint_offset = np.array([0.3, 0, 0])
target = link2_pos + link2_mat @ endpoint_offset
logger.info(f"Target position (end of link2): {target}")

# Reset to initial position
mujoco.mj_resetData(model, data)

# IK Logic
def joint_limit_clamp(full_qpos):
    """Clamp joint values to limits"""
    for i in range(model.nq):
        if model.jnt_limited[i]:
            prev_value = full_qpos[i].copy()
            full_qpos[i] = max(model.jnt_range[i][0], min(full_qpos[i], model.jnt_range[i][1]))
            new_value = full_qpos[i].copy()
            if prev_value != new_value:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
                logger.debug(f"Updated joint {joint_name} value from {prev_value:.6f} to {new_value:.6f}")
    
    return full_qpos

def forward_kinematics(joint_angles):
    """Compute forward kinematics for given joint angles"""
    # Save current state
    qpos_save = data.qpos.copy()
    
    # Set joint angles
    data.qpos[:len(joint_angles)] = joint_angles
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # Get end effector position
    # First get the body position and orientation
    link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
    body_pos = data.xpos[link2_id].copy()
    body_mat = data.xmat[link2_id].reshape(3, 3)
    
    # Calculate endpoint by applying offset in local coordinates
    endpoint_offset = np.array([0.3, 0, 0])  # Offset to end of link2
    ee_pos = body_pos + body_mat @ endpoint_offset
    
    # Restore state
    data.qpos[:] = qpos_save
    mujoco.mj_forward(model, data)
    
    return ee_pos

def inverse_kinematics(target_pos, initialstate, max_iter=1000, tol=0.01):
    """Compute inverse kinematics using damped least squares method"""
    q = initialstate
    
    # Parameters for the algorithm
    damping = 0.1
    step_size = 0.5
    
    # Track best solution
    best_error = float('inf')
    best_q = q.copy()
    
    # Get the body ID for the end effector
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
    
    for i in range(max_iter):
        # Forward kinematics to get current end effector position
        ee_pos = forward_kinematics(q)
        
        # Calculate error
        error = target_pos - ee_pos
        error_norm = np.linalg.norm(error)
        
        # Track best solution
        if error_norm < best_error:
            best_error = error_norm
            best_q = q.copy()
        
        # Check convergence
        if error_norm < tol:
            logger.info(f"IK converged in {i} iterations, error: {error_norm:.6f}")
            return best_q
        
        # Compute the Jacobian
        jacp = np.zeros((3, model.nv))  # Position Jacobian
        jacr = np.zeros((3, model.nv))  # Rotation Jacobian
        
        # Set joints for Jacobian calculation
        qpos_save = data.qpos.copy()
        data.qpos[:len(q)] = q
        mujoco.mj_forward(model, data)
        
        # First get Jacobian at the body center
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        
        # We must apply an offset to get the Jacobian at the end of the link
        # Get rotation matrix for current body orientation
        body_mat = data.xmat[body_id].reshape(3, 3)
        
        # Calculate endpoint position relative to body origin in world coordinates
        endpoint_offset_world = body_mat @ np.array([0.3, 0, 0])
        
        # Adjust the position Jacobian for the offset
        # This accounts for the rotation induced by the offset from body center to endpoint
        jac_adj = np.zeros((3, model.nv))
        for j in range(model.nv):
            # Cross product of angular Jacobian (rotation effect) with the offset
            jac_adj[:, j] = np.cross(jacr[:, j], endpoint_offset_world)
        
        # Add the adjustment to the position Jacobian
        jacp_endpoint = jacp + jac_adj
        
        # Restore state
        data.qpos[:] = qpos_save
        mujoco.mj_forward(model, data)
        
        # We only need the first two columns of the Jacobian for our two joints
        J = jacp_endpoint[:, :2]
        
        # Compute the damped least squares solution (Levenberg-Marquardt)
        J_T = J.T
        lambda_sq = damping * damping
        J_pseudo_inv = J_T @ np.linalg.inv(J @ J_T + lambda_sq * np.eye(3))
        
        # Calculate joint update
        dq = J_pseudo_inv @ error
        
        # Apply the update with step size
        q = q + step_size * dq
        
        # Apply joint limits
        for j in range(len(q)):
            if j < model.nq and model.jnt_limited[j]:
                q[j] = max(model.jnt_range[j][0], min(q[j], model.jnt_range[j][1]))
        
        # Print debug info occasionally
        if i % 100 == 0:
            logger.info(f"Iteration {i}, Error norm: {error_norm:.6f}")
            logger.info(f"Current joints: {np.degrees(q)}")
            jac_det = np.linalg.det(J @ J.T)
            logger.debug(f"Jacobian determinant: {jac_det:.6f}")
    
    logger.warning(f"Failed to converge after {max_iter} iterations, best error: {best_error:.6f}")
    return best_q

calc_qpos = np.zeros_like(data.qpos)

initialstate = np.array([2, 0])

ik_solution = inverse_kinematics(target, initialstate)
calc_qpos[:len(ik_solution)] = ik_solution
logger.info(f"IK solution: joint1={np.degrees(ik_solution[0]):.2f}°, joint2={np.degrees(ik_solution[1]):.2f}°")

# Store original/starting position
startingpos = data.qpos.copy()

def key_cb(key):
    keycode = chr(key)
    if keycode == 'R' or keycode == 'r':
        # Reset the robot
        mujoco.mj_resetData(model, data)
        logger.info("Reset data")
    elif keycode == 'Q' or keycode == 'q':
        # Move to IK solution
        data.qpos = calc_qpos.copy()
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        logger.info("Teleported to Calculated IK Position")
        
        # Show where the endpoint is
        link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
        body_pos = data.xpos[link2_id].copy()
        body_mat = data.xmat[link2_id].reshape(3, 3)
        endpoint = body_pos + body_mat @ np.array([0.3, 0, 0])
        logger.info(f"Current endpoint: {endpoint}")
        logger.info(f"Target position: {target}")
        logger.info(f"Error: {np.linalg.norm(endpoint - target):.6f}")
    elif keycode == 'V' or keycode == 'v':
        # Move to answer position
        data.qpos = ansqpos.copy()
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        logger.info("Teleported to Answer Position")
        
        # Show target position again
        link2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link2")
        body_pos = data.xpos[link2_id].copy()
        body_mat = data.xmat[link2_id].reshape(3, 3)
        endpoint = body_pos + body_mat @ np.array([0.3, 0, 0])
        logger.info(f"Target endpoint: {endpoint}")
    elif keycode == 'P' or keycode == 'p':
        # Move to starting position
        data.qpos = initialstate.copy()
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        logger.info('Teleported to starting position')

with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
    while viewer.is_running():
        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()

        # Real-time synchronization
        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)



