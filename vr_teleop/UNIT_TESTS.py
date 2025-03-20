from vr_teleop.utils.ik import *
from vr_teleop.robot import KBot_Robot
import numpy as np


# Define path to URDF
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"


kbotv2 = KBot_Robot(urdf_path, gravity_enabled=False,timestep=0.001)

ans_qpos = kbotv2.convert_armqpos_to_fullqpos([1.8, -0.05, 0.8, -1.2, 0.12], True)
# ans_qpos = kbotv2.convert_armqpos_to_fullqpos([0.2, -0.23, 0.4, -2, 0.52], True)
kbotv2.set_qpos(ans_qpos)

target_pos = kbotv2.data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
target_ort = kbotv2.data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xquat.copy()
initial_states = kbotv2.get_limit_center(leftside=True)


kbotv2_for_ik = KBot_Robot(urdf_path, gravity_enabled=False,timestep=0.001)
calc_qpos = inverse_kinematics(kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, initial_states, True)

kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)

# Run the viewer
kbotv2.run_viewer()




