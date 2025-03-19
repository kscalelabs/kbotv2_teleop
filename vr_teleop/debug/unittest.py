import mujoco
import logging
from vr_teleop.utils.mujoco_helper import debug_get_ee_pos
from vr_teleop.utils.logging import setup_logger
from vr_teleop.utils.mujoco_helper import debug_get_ee_pos
import numpy as np

# Set up logger
logger = setup_logger(__name__)
logger.setLevel(logging.INFO)

model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
data = mujoco.MjData(model)


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


