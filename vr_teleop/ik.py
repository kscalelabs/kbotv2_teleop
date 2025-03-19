import mujoco
import mujoco.viewer
import time
import numpy as np
from vr_teleop.utils.mujoco_helper import *
import logging
from vr_teleop.utils.logging import setup_logger

# Set up logger
logger = setup_logger(__name__)
# Set logging level (adjust as needed)
logger.setLevel(logging.DEBUG) #.DEBUG

pi = np.pi


model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
model.opt.timestep = 0.001  # 0.001 = 1000hz
model.opt.gravity = [0, 0, 0]  # Set gravity to zero

target_time = time.time()
sim_time = 0.0
mujoco.mj_step(model, data)

#* Mock Example End Point the arm should go to: TEST
# ansqpos = move_joints(model, data, [0.2, -0.23, 0.4, -2, 0.52], leftside=True)
ansqpos = move_joints(model, data, [1.8, -0.05, 0.8, -1.2, 0.12], leftside=True)
data.qpos = ansqpos.copy()
mujoco.mj_step(model, data)
target = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()


#* Reset to rest.
mujoco.mj_resetData(model, data)

# * IK LOGIC
def joint_limit_clamp(full_qpos):
    # prev_qpos = full_qpos.copy()

    for i in range(model.nq):
        if model.jnt_limited[i]:
            prev_value = full_qpos[i].copy()
            full_qpos[i] = max(model.jnt_range[i][0], min(full_qpos[i], model.jnt_range[i][1]))
            new_value = full_qpos[i].copy()
            if prev_value != new_value:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
                # logger.debug(f"Updated joint {joint_name} value from {prev_value:.6f} to {new_value:.6f}, limits: [{model.jnt_range[i][0]:.6f}, {model.jnt_range[i][1]:.6f}]")
    
    return full_qpos


def forward_kinematics(joint_angles, leftside: bool):
    """
    Compute forward kinematics of given joint angles by MuJoCo
    Go to position and read position
    """
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    newpos = move_joints(model, data, joint_angles.flatten(), leftside)
    data.qpos = newpos
    mujoco.mj_forward(model, data)

    pos = data.body(ee_name).xpos.copy()
    return pos, newpos


def inverse_kinematics(target_pos, leftside: bool):
    max_iteration = 10000;
    tol = 0.01;
    # Alpha controls the step size in the Jacobian transpose method
    # Reduce alpha to avoid overshooting
    # Learning rate further scales the update
    step_size = 0.8
    # alpha = 0.8
    damping = 0.5

    # cur_qpos = get_joints(model, data, leftside, tolimitcenter=False)
    cur_qpos = np.array([0, 0, 0, -0.35, 0])

    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"

    mujoco.mj_forward(model, data)
    
    # Track best solution
    best_error = float('inf')
    best_pos = None
    
    for i in range(max_iteration):
        ee_pos, full_pos = forward_kinematics(cur_qpos, leftside)
        error = np.subtract(target_pos, ee_pos)
        error_norm = np.linalg.norm(error)
        
        # Track best solution
        if error_norm < best_error:
            best_error = error_norm
            best_pos = full_pos.copy()
    
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        n = jacp.shape[1]
        I = np.identity(n)

        mujoco.mj_jac(model, data, jacp, jacr, target_pos, model.body(ee_name).id)
        J_inv = np.linalg.inv(jacp.T @ jacp + damping * I) @ jacp.T
        delta_q = J_inv @ error
            
        full_pos += delta_q * step_size
        joint_limit_clamp(full_pos)

        cur_qpos = slice_dofs(model, data, full_pos, leftside)
        cur_qpos = cur_qpos.flatten()

        if i % 88 == 0:
            # Calculate the row-reduced echelon form (RREF) of the Jacobian
            # This helps analyze the linear independence of the Jacobian rows
            # and identify potential singularities
            logger.debug(f"Rank is: {np.linalg.matrix_rank(jacp)}")
            logger.debug(f"Iteration {i}, Error: {error}, Error norm: {error_norm:.6f}")
            logger.debug(f"Jacobian (position): {jacp}, and the Jacobian Transpose: {jacp.T}")
            # logger.debug(f"Jacobian Transpose * Error * Alpha (0.8): {np.linalg.norm(grad)}, full: {grad}")
            jac_det = np.linalg.det(jacp @ jacp.T)
            logger.debug(f"Jacobian determinant: {jac_det:.6f}")
            logger.debug(f"Target position: {target_pos}")
            logger.debug(f"Current end effector position: {ee_pos}")
            logger.debug(f"Current joints: {cur_qpos}")
            # breakpoint()

        if error_norm < tol:
            logger.info(f"Converged in {i} iterations")
            return full_pos
    
    logger.warning(f"Failed to converge after {max_iteration} iterations, error: {best_error:.6f}")
    return best_pos

startingpos = get_joints(model, data, True, True)
startingpos = move_joints(model, data, startingpos.flatten(), True)

calc_qpos = inverse_kinematics(target, True)

def key_cb(key):
    keycode = chr(key)
    if keycode == 'R':
        mujoco.mj_resetData(model, data)
        logger.info("Reset data")
    elif keycode == 'Q':
        data.qpos = calc_qpos
        # data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        logger.info("Teleported to Calculated Position")
        np.savetxt('./vr_teleop/data/calculated_qpos.txt', calc_qpos)
        logger.info(f"End effector position: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xpos}")
    elif keycode == 'V':
        data.qpos = ansqpos
        mujoco.mj_forward(model, data)
        logger.info("Teleported to Answer Position")
        np.savetxt('./vr_teleop/data/ans_qpos.txt', ansqpos)
        logger.info(f"End effector position: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xpos}")
    elif keycode == 'P':
        data.qpos = startingpos
        mujoco.mj_forward(model, data)
        logger.info('Teleported to Optimization initial condition')
    

with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:

    while viewer.is_running():
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()

        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)



