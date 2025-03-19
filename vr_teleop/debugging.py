import mujoco
import numpy as np

def get_adapter_position(qpos_file):
    # Load the model
    model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
    data = mujoco.MjData(model)
    
    # Reset data and set model properties
    mujoco.mj_resetData(model, data)
    model.opt.timestep = 0.001
    model.opt.gravity[2] = 0
    
    # Load qpos from file and apply it
    loaded_qpos = np.loadtxt(qpos_file)
    data.qpos = loaded_qpos
    data.qvel[:] = 0
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model,data)
    
    # Get the position of the adapter
    adapter_position = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
    
    return adapter_position

# Get positions using both qpos files
calculated_position = get_adapter_position('./vr_teleop/data/calculated_qpos.txt')
ans_position = get_adapter_position('./vr_teleop/data/ans_qpos.txt')

# Calculate error
position_error = np.subtract(ans_position, calculated_position)
error_norm = np.linalg.norm(position_error)

# Print results
print(f"Position with calculated_qpos: {calculated_position}")
print(f"Position with ans_qpos: {ans_position}")
print(f"Position error vector: {position_error}")
print(f"Error norm: {error_norm}")