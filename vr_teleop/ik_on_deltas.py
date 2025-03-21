
import logging
import mujoco
import time
import math
import numpy as np
from vr_teleop.ikrobot import KBot_Robot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.utils.ik import *



urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
solver = KBot_Robot(urdf_path)

fullq = solver.convert_armqpos_to_fullqpos([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)
ee_pos = solver.get_ee_pos(leftside=False)

motion_plan = Robot_Planner(solver)

llocs = solver.get_limit_center(leftside=True)
rlocs = solver.get_limit_center(leftside=False)

fullq = motion_plan.arms_tofullq(llocs, rlocs)
motion_plan.set_curangles(solver.data.qpos)
motion_plan.set_nextangles(fullq)

all_angles, all_velocities = motion_plan.get_waypoints()


breakpoint()


#* Give KOS-Sim the same start postions for both arm


# # solver.set_qpos(fullq)

# [0.40799871, 0.06046459, 1.18965362]

# calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(solver.model, solver.data, [target_pos], target_ort, initial_states, leftside, debug=True)




