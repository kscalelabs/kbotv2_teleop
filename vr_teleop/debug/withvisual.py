import mujoco
import mujoco.viewer
import time
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger

# Set up logger
logger = setup_logger(__name__)
# You can uncomment this to see debug messages during development
logger.setLevel(logging.DEBUG)

model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
data = mujoco.MjData(model)

logger.info("Initializing model and data")
mujoco.mj_resetData(model, data)

model.opt.timestep = 0.001  # 0.001 = 1000hz
model.opt.gravity[2] = 0
logger.debug(f"Model timestep set to {model.opt.timestep}, gravity: {model.opt.gravity}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    target_time = time.time()
    sim_time = 0.0
    
    # Track frame count for reducing log frequency
    frame_count = 0
    log_interval = 100  # Only log every 100 frames to avoid flooding
    
    # Add flag to track if we've loaded qpos
    qpos_loaded = False
    qpos_load_time = 0

    while viewer.is_running():
        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()
        
        frame_count += 1

        if sim_time > 2 and sim_time < 2.005 and not qpos_loaded:
            loaded_qpos = np.loadtxt('./vr_teleop/data/calculated_qpos.txt') # [-0.35868784  0.19986784  0.84827277]
            # loaded_qpos = np.loadtxt('./vr_teleop/data/ans_qpos.txt') # [-0.3135051   0.14240396  0.93119387]

            # Error: 0.110542
            # 0.22312012116158278
            data.qpos = loaded_qpos
            
            # Forward kinematics to update positions
            mujoco.mj_forward(model, data)
            
            # Zero out velocities
            data.qvel[:] = 0
            
            # Log the position immediately after loading
            adapter_pos = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos
            logger.info(f"Adapter position IMMEDIATELY after loading qpos (t={sim_time:.6f}): {adapter_pos}")
            
            qpos_loaded = True
            qpos_load_time = sim_time
  
        # Log the adapter position (throttled to avoid too many messages)
        adapter_pos = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos
        
        # After loading qpos, log more frequently for a short period
        if qpos_loaded and (sim_time - qpos_load_time) <= 0.1:
            if frame_count % 10 == 0:  # Log every 10 frames for 0.1 seconds after loading
                logger.info(f"Adapter position at t={sim_time:.6f} (dt={sim_time-qpos_load_time:.6f}): {adapter_pos}")
        elif frame_count % log_interval == 0:
            logger.debug(f"Adapter position at t={sim_time:.2f}: {adapter_pos}")

        # Wait for real-time synchronization
        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)

logger.info("Simulation completed")

# Add the error calculation with proper logging
adapter_pos = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos
logger.info(f"Final adapter position: {adapter_pos}")

# Calculate error from loaded to final position
if qpos_loaded:
    loaded_pos = np.loadtxt('./vr_teleop/data/ans_qpos.txt')
    error = np.subtract(adapter_pos, loaded_pos)
    error_norm = np.linalg.norm(error)
    logger.info(f"Position change from loaded qpos: {error}")
    logger.info(f"Position change magnitude: {error_norm}")

# >>> error = np.subtract([-0.1739883, 0.135415, 0.92103787], [-0.23478118, 0.10839, 0.8])
# >>> error
# array([0.06079288, 0.027025  , 0.12103787])
# >>> np.linalg.norm(error)
# np.float64(0.13811694630939136)

# Uncomment and adjust if you want to calculate the error in the script
# calculated_pos = np.array([-0.23478118, 0.10839, 0.8])
# error = np.subtract(adapter_pos, calculated_pos)
# error_norm = np.linalg.norm(error)
# logger.info(f"Position error vector: {error}")
# logger.info(f"Error norm: {error_norm}")


