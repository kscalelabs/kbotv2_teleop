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
# ansqpos = arms_to_fullqpos(model, data, [0.2, -0.23, 0.4, -2, 0.52], leftside=True)
ansqpos = arms_to_fullqpos(model, data, [1.8, -0.05, 0.8, -1.2, 0.12], leftside=True)
data.qpos = ansqpos.copy()
mujoco.mj_step(model, data)
target = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
target_ort = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xquat.copy()


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


def quat_mult(q1, q2):
    # Multiply two quaternions q1 and q2 (assumed [w, x, y, z])
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    # Conjugate of quaternion q = [w, x, y, z]
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def orientation_error(target_quat, current_quat):
    # Compute the relative rotation quaternion:
    dq = quat_mult(target_quat, quat_conjugate(current_quat))
    # Normalize to avoid numerical issues:
    dq = dq / np.linalg.norm(dq)
    # Make sure the scalar part is nonnegative to get the shortest rotation:
    if dq[0] < 0:
        dq = -dq
    # For small angles, the error can be approximated by 2*[x, y, z]
    return 2 * dq[1:4]


def forward_kinematics(joint_angles, leftside: bool):
    """
    Compute forward kinematics of given joint angles by MuJoCo
    Go to position and read position
    """
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    newpos = arms_to_fullqpos(model, data, joint_angles.flatten(), leftside)
    data.qpos = newpos
    mujoco.mj_forward(model, data)

    pos = data.body(ee_name).xpos.copy()

    ort = data.body(ee_name).xquat.copy()

    return pos, ort


def inverse_kinematics(target_pos, target_ort, initialstate, leftside: bool):
    max_iteration = 10000;
    tol = 0.01;
    step_size = 0.8
    damping = 0.5


    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"

    mujoco.mj_forward(model, data)
    
    next_pos_arm = initialstate.copy()
    
    for i in range(max_iteration):
        ee_pos, ee_rot = forward_kinematics(next_pos_arm, leftside=leftside)
        error = np.subtract(target_pos, ee_pos)
        error_norm = np.linalg.norm(error)

        error_rot = orientation_error(target_ort, ee_rot)

        prev_pos_arm = next_pos_arm.copy()
        prev_pos = arms_to_fullqpos(model, data, prev_pos_arm, leftside=leftside)


        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        n = jacp.shape[1]
        I = np.identity(n)

        mujoco.mj_jac(model, data, jacp, jacr, target_pos, model.body(ee_name).id)
        
        #*Term for Translation:
        J_inv = np.linalg.inv(jacp.T @ jacp + damping * I) @ jacp.T
        #*Term for Rotation:
        JR_inv = np.linalg.inv(jacr.T @ jacr + damping * I) @ jacr.T
        delta_q = J_inv @ error + JR_inv @ error_rot
        
        next_pos = prev_pos + delta_q * step_size

        next_pos = joint_limit_clamp(next_pos)
        next_pos_arm = slice_dofs(model, data, next_pos, leftside)
        next_pos_arm = next_pos_arm.flatten()

        if i % 88 == 0:
            # Calculate the row-reduced echelon form (RREF) of the Jacobian
            # This helps analyze the linear independence of the Jacobian rows
            # and identify potential singularities
            logger.debug(f"Rank is: {np.linalg.matrix_rank(jacp)}")
            logger.debug(f"Iteration {i}, Error: {error}, Error norm: {error_norm:.6f}")
            logger.debug(f"Jacobian (position): {jacp}, and the Jacobian Transpose: {jacp.T}")
            logger.debug(f"Jacobian Transpose * Error * Alpha (0.8): {np.linalg.norm(delta_q)}, full: {delta_q}")
            jac_det = np.linalg.det(jacp @ jacp.T)
            logger.debug(f"Jacobian determinant: {jac_det:.6f}")
            logger.debug(f"Target position: {target_pos}")
            logger.debug(f"Current end effector position: {ee_pos}")
            logger.debug(f"Current joints: {next_pos_arm}")
            # breakpoint()

        if error_norm < tol:
            logger.info(f"Converged in {i} iterations")
            return next_pos
    
    logger.warning(f"Failed to converge after {max_iteration} iterations, error: {error_norm:.6f}")
    return next_pos

# startingpos = get_arm_qpos(model, data, True, True)
# startingpos = arms_to_fullqpos(model, data, startingpos.flatten(), True)
initial_states = get_arm_qpos(model, data, leftside=True, tolimitcenter=True)
# initial_states = np.array([0, 0, 0, -0.35, 0])
# initial_states = np.array([0, 0, 0, 0, 0])

calc_qpos = inverse_kinematics(target, target_ort, initial_states, leftside=True)

def key_cb(key):
    global viewer_ref
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
        logger.info(f"End effector orientation: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xquat}")
    elif keycode == 'V':
        data.qpos = ansqpos
        mujoco.mj_forward(model, data)
        logger.info("Teleported to Answer Position")
        np.savetxt('./vr_teleop/data/ans_qpos.txt', ansqpos)
        logger.info(f"End effector position: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xpos}")
        logger.info(f"End effector orientation: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xquat}")
    elif keycode == 'P':
        data.qpos = arms_to_fullqpos(model, data, initial_states, True)
        mujoco.mj_forward(model, data)
        logger.info('Teleported to Optimization initial condition')
    elif keycode == "O":
        if viewer_ref[0].opt.frame == 7:
            viewer_ref[0].opt.frame = 1
        else:
            viewer_ref[0].opt.frame = 7
        viewer_ref[0].sync()
        logger.info("Toggled frame visualization")

viewer_ref = []

with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:

    while viewer.is_running():
        viewer_ref.append(viewer)
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()

        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)



