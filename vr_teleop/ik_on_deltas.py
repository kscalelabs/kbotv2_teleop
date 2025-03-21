import logging
import mujoco
import time
import math
import numpy as np
from vr_teleop.ikrobot import KBot_Robot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.utils.ik import *

from scipy.spatial.transform import Rotation as R



urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
solver = KBot_Robot(urdf_path)

fullq = solver.convert_armqpos_to_fullqpos([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)

motion_plan = Robot_Planner(solver)

llocs = solver.get_limit_center(leftside=True)
rlocs = solver.get_limit_center(leftside=False)

fullq = motion_plan.arms_tofullq(llocs, rlocs)

motion_plan.set_curangles(solver.data.qpos)
motion_plan.set_nextangles(fullq)

all_angles, _ = motion_plan.get_waypoints()

#* KOS-SIM MOVE TO STARTING POSITIONS
#* num_dofs, num_waypoints = all_angles.shape

# import matplotlib.pyplot as plt

# # Get the shape of all_angles
# 

# # Create a figure to plot DOFs
# plt.figure(figsize=(12, 8))

# # Generate x-axis values (waypoint indices)
# waypoint_indices = range(num_waypoints)

# # Plot each DOF trajectory
# for i in range(num_dofs):
#     plt.plot(waypoint_indices, all_angles[i, :], label=f'DOF {i+1}')

# # Add labels and legend
# plt.xlabel('Waypoint Index')
# plt.ylabel('Joint Angle (rad)')
# plt.title('Joint Angles Across Waypoints')
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.tight_layout()
# plt.show()

# # breakpoint()


#* Give KOS-Sim the same start postions for both arm

#* Before Deltas:
solver.set_qpos(fullq)
lee_pos, lee_ort = solver.get_ee_pos(leftside=True)
ree_pos, ree_ort = solver.get_ee_pos(leftside=False)


def apply_pose_delta(current_pos, current_quat, delta_pos=[0, 0, 0], delta_euler=[0, 0, 0]):
    """
    Args:
        current_pos: Current position [x, y, z]
        current_quat: Current quaternion [w, x, y, z]
        delta_pos: List/array of [dx, dy, dz] position changes in 
            meters
        delta_euler: List/array of [d_roll, d_pitch, d_yaw] angle changes in
            radians. where roll is around x, pitch around y, and yaw around z
    """
    # Apply position delta
    new_pos = np.array(current_pos) + np.array(delta_pos)
    
    # Convert Mujoco quaternion [w, x, y, z] to scipy quaternion [x, y, z, w]
    scipy_quat = np.array([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
    
    # Create rotation object and get current Euler angles (in 'xyz' convention)
    rot = R.from_quat(scipy_quat)
    current_euler = rot.as_euler('xyz', degrees=False)
    
    # Apply rotation delta
    new_euler = current_euler + np.array(delta_euler)
    
    # Convert back to quaternion
    new_rot = R.from_euler('xyz', new_euler, degrees=False)
    new_scipy_quat = new_rot.as_quat()  # [x, y, z, w]
    
    # Convert back to Mujoco quaternion format [w, x, y, z]
    new_quat = np.array([new_scipy_quat[3], new_scipy_quat[0], new_scipy_quat[1], new_scipy_quat[2]])
    
    return new_pos, new_quat




new_pos, new_quat = apply_pose_delta(lee_pos, lee_ort, [0.1, 0.1, 0.1], [0, 0, 0])
calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(solver.model, solver.data, new_pos, new_quat, leftside=True)



motion_plan.set_nextangles(calc_qpos)
all_angles, _ = motion_plan.get_waypoints()

solver.set_qpos(all_angles[:, 1])


solver.run_viewer()





