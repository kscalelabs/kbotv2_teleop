from vr_teleop.ik import *
from vr_teleop.KOSSDK_MOCK import Ik_Robot
import numpy as np

# Define path to URDF
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"

# Define target position and orientation (for example)
target_pos = np.array([0.5, 0.0, 0.5])  # Example position
target_ort = np.array([1.0, 0.0, 0.0, 0.0])  # Example orientation (quaternion w,x,y,z)

# Create an instance of Ik_Robot
# Note: The constructor needs to be fixed as it expects 'parent' but we need 'urdf_path'
ik_robot = Ik_Robot(urdf_path, target_pos, target_ort, gravity_enabled=False,timestep=0.001)

# Initialize the simulation
ik_robot.start_sim()

# Run the viewer
ik_robot.run_viewer()

# #* Mock Example End Point the arm should go to: TEST
# # ansqpos = arms_to_fullqpos(model, data, [0.2, -0.23, 0.4, -2, 0.52], leftside=True)
# ansqpos = arms_to_fullqpos(model, data, [1.8, -0.05, 0.8, -1.2, 0.12], leftside=True)
# data.qpos = ansqpos.copy()
# mujoco.mj_step(model, data)
# target = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
# target_ort = data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xquat.copy()


# #* Reset to rest.
# mujoco.mj_resetData(model, data)





