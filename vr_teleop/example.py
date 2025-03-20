from vr_teleop.utils.ik import *
from vr_teleop.ikrobot import KBot_Robot
import numpy as np

logger.setLevel(logging.DEBUG) #.DEBUG .INFO

urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
kbotv2 = KBot_Robot(urdf_path, gravity_enabled=False,timestep=0.001)

ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([-2.5, 1.6, 1.7, 2.5, 1.7], leftside=False)
leftside = False

kbotv2.set_qpos(ans_qpos)

if leftside:
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
else:
    ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"

kbotv2.set_qpos(ans_qpos)
target_pos = kbotv2.data.body(ee_name).xpos.copy()
target_ort = kbotv2.data.body(ee_name).xquat.copy()
initial_states = kbotv2.get_limit_center(leftside=leftside)

kbotv2_for_ik = KBot_Robot(urdf_path, gravity_enabled=False,timestep=0.001)
calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, initial_states, leftside, debug=True)
kbotv2.set_iksolve_side(leftside)
kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)
logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")

kbotv2.run_viewer()







