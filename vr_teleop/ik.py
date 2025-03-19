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
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

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
ansqpos = move_joints(model, data, [0.3, -0.5, pi*(3/2), -2, 0.5], leftside=True)
data.qpos = ansqpos.copy()
mujoco.mj_step(model, data)
target = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()

breakpoint()

#* Reset to rest.
mujoco.mj_resetData(model, data)

# * IK LOGIC
def forward_kinematics(joint_anlges, leftside: bool):
    """
    Compute forward kinematics of given joint angles by MuJoCo
    Go to position and read position
    """
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    newpos = move_joints(model, data, joint_anlges, leftside)
    data.qpos = newpos
    mujoco.mj_forward(model, data)

    pos = data.body(ee_name).xpos.copy()
    return pos, newpos


def inverse_kinematics(target_pos, leftside: bool):
    max_iteration = 5000;
    tol = 0.000001;
    alpha = 0.9
    learning_rate = 1

    cur_qpos = get_joints(model, data, leftside)

    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"

    mujoco.mj_forward(model, data)
    for i in range(max_iteration):
        ee_pos, full_pos = forward_kinematics(cur_qpos, leftside)
        error = np.subtract(target_pos, ee_pos)
        if i % 7 == 0:
            logger.info(f"Iteration {i}, Error: {np.linalg.norm(error):.6f}")
            logger.debug(f"Target position: {target_pos}")
            logger.debug(f"Current end effector position: {ee_pos}")

        if np.linalg.norm(error) < tol:
            logger.info(f"Converged in {i} iterations")
            return full_pos
    
        jacp = np.zeros((3, model.nv)) # 5 for 5DoF arm
        jacr = np.zeros((3, model.nv))
        # breakpoint
        mujoco.mj_jac(model, data, jacp, jacr, target_pos, model.body(ee_name).id)
        # jacp = slice_dofs(model, data, jacp, leftside)
        # jacr = slice_dofs(model, data, jacr, leftside)

        grad = alpha * jacp.T @ error
        full_pos +=  grad * learning_rate
        cur_qpos = slice_dofs(model, data, full_pos, leftside)
        cur_qpos = cur_qpos.flatten()
        # breakpoint()
    
    # mujoco.mj_resetData(model, data)
    return full_pos
    
calc_qpos = inverse_kinematics(target, True)

# ([-0.23233644,  0.07232578,  0.87956266])

def key_cb(key):
    keycode = chr(key)
    if keycode == 'R':
        mujoco.mj_resetData(model, data)
        logger.info("Reset data")
    elif keycode == 'Q':
        data.qpos = calc_qpos
        mujoco.mj_step(model, data)
        logger.info("Going to Calculated Position")
        np.savetxt('./vr_teleop/data/calculated_qpos.txt', calc_qpos)
        logger.info(f"End effector position: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xpos}")
    elif keycode == 'V':
        data.qpos = ansqpos
        mujoco.mj_step(model, data)
        logger.info("Going to Answer Position")
        np.savetxt('./vr_teleop/data/ans_qpos.txt', ansqpos)
        logger.info(f"End effector position: {data.body('KB_C_501X_Bayonet_Adapter_Hard_Stop_2').xpos}")
    

with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:

    while viewer.is_running():
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()

        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)



