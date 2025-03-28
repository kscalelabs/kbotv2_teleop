from vr_teleop.utils.ik import *
from vr_teleop.helpers.mjRobot import MJ_KBot
import numpy as np
import time

logger.setLevel(logging.DEBUG) #.DEBUG .INFO

# Time robot initialization
start_time = time.time()
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
kbotv2 = MJ_KBot(urdf_path, gravity_enabled=False,timestep=0.001)
print(f"Robot initialization time: {time.time() - start_time:.4f} seconds")

# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.5, 1.6, 1.7, 2.5, 1.7], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([2.0, 0.5, -1.7, 1.0, -0.5], leftside=False)
start_time = time.time()
ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.2, -0.9, 0, -1.6, -0.7], leftside=False)
leftside = False
print(f"Initial pose setting time: {time.time() - start_time:.4f} seconds")

# initialstate = np.array([1.5, 1.4, -1.5, 0.8, 0])

kbotv2.set_qpos(ans_qpos)

# Time target pose extraction
start_time = time.time()
ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2" if leftside else "KB_C_501X_Bayonet_Adapter_Hard_Stop"
target_pos = kbotv2.data.body(ee_name).xpos.copy()
target_ort = kbotv2.data.body(ee_name).xquat.copy()
initialstate = kbotv2.get_limit_center(leftside=leftside)
print(f"Target pose extraction time: {time.time() - start_time:.4f} seconds")

# Time IK solving
start_time = time.time()
kbotv2_for_ik = MJ_KBot(urdf_path, gravity_enabled=False,timestep=0.001)

# temp_target_pos = np.array([-0.43841719, 0.14227997, 0.99074782])


# full_delta_q, _, _ = ik_gradient(kbotv2_for_ik.model, kbotv2_for_ik.data, temp_target_pos, target_ort, leftside, initialstate)

start_time = time.time()
pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort=None, leftside=leftside, initialstate=initialstate)
print(f"IK time: {(time.time() - start_time) * 1000:.4f} milliseconds")

# Time final pose setting and validation
start_time = time.time()
calc_qpos = kbotv2.convert_armqpos_to_fullqpos(pos_arm, leftside)
kbotv2.set_iksolve_side(leftside)
kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initialstate)
print(f"Final pose setting and validation time: {time.time() - start_time:.4f} seconds")

logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")

kbotv2.run_viewer()







