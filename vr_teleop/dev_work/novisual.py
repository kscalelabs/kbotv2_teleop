import mujoco
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger

# Use the custom logger setup instead of basic configuration
logger = setup_logger(__name__)
# Set to DEBUG level to see all messages during development
logger.setLevel(logging.DEBUG)

def get_adapter_position(qpos_file):
    # Load the model
    model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
    data = mujoco.MjData(model)

    # Different log levels for different types of information
    logger.debug("Loading model and initializing data")  # Technical details
    logger.info(f"Processing qpos file: {qpos_file}")    # General workflow info
    
    # Print all bodies in the model - debug level since it's verbose
    for i in range(model.nbody):
        body_name = model.body(i).name
        logger.debug(f"Body {i}: {body_name}")
    
    # Reset data and set model properties
    mujoco.mj_resetData(model, data)
    model.opt.timestep = 0.001
    model.opt.gravity[2] = 0
    logger.debug(f"Model timestep set to {model.opt.timestep}, gravity: {model.opt.gravity}")
    
    # Load qpos from file and apply it
    try:
        loaded_qpos = np.loadtxt(qpos_file)
        data.qpos = loaded_qpos
        data.qvel[:] = 0
        logger.debug(f"Loaded qpos with shape {loaded_qpos.shape}")
    except Exception as e:
        logger.error(f"Failed to load qpos file: {e}")  # Error for problems
        raise
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model,data)
    
    # Get the position of the adapter
    adapter_position = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
    logger.debug(f"Raw adapter position: {adapter_position}")
    
    return adapter_position


# Get positions using both qpos files
logger.info("Computing adapter positions from qpos files...")
calculated_position = get_adapter_position('./vr_teleop/data/calculated_qpos.txt')
ans_position = get_adapter_position('./vr_teleop/data/ans_qpos.txt')

# Calculate error
position_error = np.subtract(ans_position, calculated_position)
error_norm = np.linalg.norm(position_error)

# Use appropriate log levels for different information
logger.info(f"Position with calculated_qpos: {calculated_position}")
logger.info(f"Position with ans_qpos: {ans_position}")
logger.info(f"Position error vector: {position_error}")

# Use warning if the error is significant
if error_norm > 0.01:
    logger.warning(f"Large error detected! Error norm: {error_norm}")
else:
    logger.info(f"Error norm: {error_norm}")