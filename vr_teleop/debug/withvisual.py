import mujoco
import mujoco.viewer
import time
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger

# Set up logger
logger = setup_logger(__name__)
logger.setLevel(logging.INFO)

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
    
    frame_count = 0
    log_interval = 100 
    
    calculated_qpos_load_time = 0
    ans_qpos_load_time = 0
    calculated_position = None

    while viewer.is_running():
        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()
        
        frame_count += 1

        # First load calculated_qpos at sim_time = 2
        if sim_time > 2 and sim_time < 2.005:
            loaded_qpos = np.loadtxt('./vr_teleop/data/calculated_qpos.txt')
            
            data.qpos = loaded_qpos
            
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            data.qvel[:] = 0
            
            calc_ee_pos = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
            logger.info(f"Adapter position after loading calculated_qpos (t={sim_time:.6f}): {calculated_position}")
            
            # calculated_qpos_load_time = sim_time
            

        if sim_time > 4 and sim_time < 4.005:
            loaded_qpos = np.loadtxt('./vr_teleop/data/ans_qpos.txt')
            data.qpos = loaded_qpos
            
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            data.qvel[:] = 0
            

            ans_ee_pos = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
            breakpoint()
            position_error = np.subtract(ans_ee_pos, calculated_position)
            error_norm = np.linalg.norm(position_error)
            
            logger.info(f"Adapter position after loading ans_qpos (t={sim_time:.6f}): {ans_ee_pos}")
            logger.info(f"Position error vector: {position_error}")
            logger.warning(f"Error norm: {error_norm}")
            
            ans_qpos_load_time = sim_time
        
        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)

logger.info("Simulation completed")



