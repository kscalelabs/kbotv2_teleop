from vr_teleop.utils.ik import *
from vr_teleop.ikrobot import KBot_Robot
import numpy as np

logger.setLevel(logging.DEBUG) #.DEBUG .INFO

# Define path to URDF
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([1.8, -0.05, 0.8, -1.2, 0.12], True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.2, -0.23, 0.4, -2, 0.52], True)
kbotv2 = KBot_Robot(urdf_path, gravity_enabled=False,timestep=0.001)


#*----- FAILING TESTS:
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([2.5, -1.6, -1.7, -2.5, -1.7], leftside=True) #0.015
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.0, -0.5, 1.7, -1.0, 0.5], leftside=True) #0.24
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.5, 1.6, 1.7, 2.3, 1.7], leftside=False) #0.015
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([2.0, 0.5, -1.7, 1.0, -0.5], leftside=False) #0.24


# [2.5, -1.6, -1.7, -2.5, -1.7]
# [-2.0, -0.5, 1.7, -1.0, 0.5]
# [-2.0, 1.0, 1.0, 2.0, 1.0]
ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.5, 1.6, 1.7, 2.5, 1.7], leftside=False) #0.24
leftside = False
#*----- FAILING TESTS

# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-0.2, -0.5, 0.3, -1.3, -0.2], leftside=True)
# leftside = True



# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.5, -1.6, -1.7, -2.5, 1.7], leftside=leftside) 
# kbotv2.set_qpos(ans_qpos)

if leftside:
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
else:
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"

if leftside:
    initial_states = np.array([0.2, -0.5, -1.5, -0.5, 0])
else:
    # initial_states = np.array([-0.2, 0.5, -1.5, 0.5, 0])
    initial_states = np.array([0, 0, -1.5, 0.5, 0])


kbotv2.set_qpos(ans_qpos)
target_pos = kbotv2.data.body(ee_name).xpos.copy()
target_ort = kbotv2.data.body(ee_name).xquat.copy()
# initial_states = kbotv2.get_limit_center(leftside=leftside)

# temp = kbotv2.convert_armqpos_to_fullqpos(initial_states, leftside=False)
# kbotv2.set_qpos(temp)

# breakpoint()

kbotv2_for_ik = KBot_Robot(urdf_path, gravity_enabled=False,timestep=0.001)
calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, initial_states, leftside)
kbotv2.set_iksolve_side(leftside)
kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)
logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")

kbotv2.run_viewer()




