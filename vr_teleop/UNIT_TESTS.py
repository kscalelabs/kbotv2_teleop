from vr_teleop.utils.ik import *
from vr_teleop.mjRobot import MJ_KBot
import numpy as np


# Define path to URDF
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([1.8, -0.05, 0.8, -1.2, 0.12], True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.2, -0.23, 0.4, -2, 0.52], True)
kbotv2 = MJ_KBot(urdf_path, gravity_enabled=False,timestep=0.001)



# Left Arm in edges of reach
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([2.0, -1.0, -1.0, -2.0, -1.0], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-1.5, -0.3, 1.0, -0.5, -0.5], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0, -1.6, 0, -2.5, 0], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([2.5, -1.6, -1.7, -2.5, -1.7], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.0, -0.5, 1.7, -1.0, 0.5], leftside=True)

# Left arm positions for workspace manipulation (leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0, -0.8, 0, -1.5, 0], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.3, -0.6, 0, -1.2, 0], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.5, -0.7, -0.2, -1.4, -0.3], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.5, -0.7, 0.2, -1.4, 0.3], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.2, -0.9, 0, -1.6, -0.7], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.4, -1.0, -0.4, -1.8, 0.5], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.2, -0.5, 0.3, -1.3, -0.2], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.7, -0.8, -0.1, -2.0, 0], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.3, -0.4, 0, -1.0, 0], leftside=True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.5, -0.6, -0.5, -1.5, 0.5], leftside=True)

# Right Arm in edges of reach
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.0, 1.0, 1.0, 2.0, 1.0], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([1.5, 0.3, -1.0, 0.5, 0.5], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0, 1.6, 0, 2.5, 0], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.5, 1.6, 1.7, 2.5, 1.7] , leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([2.0, 0.5, -1.7, 1.0, -0.5], leftside=False)

# Right Arm in front of workspace
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.3, 0.6, 0, 1.2, 0], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.5, 0.7, 0.2, 1.4, 0.3], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.5, 0.7, -0.2, 1.4, -0.3], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.2, 0.9, 0, 1.6, 0.7], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.2, 0.5, -0.3, 1.3, 0.2], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.7, 0.8, 0.1, 2.0, 0], leftside=False)
ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.3, 0.4, 0, 1.0, 0], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.5, 0.6, 0.5, 1.5, -0.5], leftside=False) 
kbotv2.set_qpos(ans_qpos)

target_pos = kbotv2.data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
target_ort = kbotv2.data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xquat.copy()
initial_states = kbotv2.get_limit_center(leftside=True)


kbotv2_for_ik = MJ_KBot(urdf_path, gravity_enabled=False,timestep=0.001)
calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, initial_states, True)
kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)

logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")
# Run the viewer
kbotv2.run_viewer()




