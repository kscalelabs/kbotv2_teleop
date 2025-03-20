import mujoco
import logging
from vr_teleop.utils.logging import setup_logger
import numpy as np

# Set up logger
logger = setup_logger(__name__)
logger.setLevel(logging.INFO)

model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
data = mujoco.MjData(model)


def debug_get_ee_pos(qpos_file):
    # Create a new model and data instance for clean calculation
    temp_model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
    temp_data = mujoco.MjData(temp_model)
    
    # Reset data and set model properties
    mujoco.mj_resetData(temp_model, temp_data)
    temp_model.opt.timestep = 0.001
    temp_model.opt.gravity[2] = 0
    
    # Load qpos from file and apply it
    loaded_qpos = np.loadtxt(qpos_file)
    temp_data.qpos = loaded_qpos
    temp_data.qvel[:] = 0
    
    # Forward kinematics to update positions
    mujoco.mj_forward(temp_model, temp_data)
    mujoco.mj_step(temp_model, temp_data)
    
    # Get the position of the adapter
    adapter_position = temp_data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
    
    return adapter_position



final_adapter_pos = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos
logger.info(f"Final adapter position: {final_adapter_pos}")

# Load positions directly from files for verification
calculated_pos = debug_get_ee_pos('./vr_teleop/data/calculated_qpos.txt')
ans_pos = debug_get_ee_pos('./vr_teleop/data/ans_qpos.txt')

position_error = np.subtract(ans_pos, calculated_pos)
error_norm = np.linalg.norm(position_error)

logger.info(f"Position with calculated_qpos: {calculated_pos}")
logger.info(f"Position with ans_qpos: {ans_pos}")
logger.info(f"Position error vector: {position_error}")
logger.info(f"Error norm: {error_norm}")


