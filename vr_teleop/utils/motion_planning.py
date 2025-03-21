import logging
from vr_teleop.ikrobot import KBot_Robot
import mujoco
import time
import math
import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # .DEBUG .INFO


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    flip_sign: bool
    joint_name: str

class Robot_Planner:
    def __init__(self, mjRobot, acc=50.0, vmax=30.0, updaterate=100.0):
        
        self.profile = "scurve"
        self.acc = acc
        self.vmax = vmax
        self.updaterate = updaterate
        self.T_accel_candidate = 1.5 * self.vmax / self.acc  # duration of acceleration phase
        self.d_accel_candidate = 0.5 * self.vmax * self.T_accel_candidate  # distance during acceleration phase

        self.mjRobot = mjRobot
        self.cur_angles = None
        self.next_angles = None
        self.waypoints = []

        self.idx_to_joint_map = None

        # self.act_id_to_joint_name = {
        #     11: "left_shoulder_pitch_03",
        #     12: "left_shoulder_roll_03",
        #     13: "left_shoulder_yaw_02",
        #     14: "left_elbow_02",
        #     15: "left_wrist_02",
        #     21: "right_shoulder_pitch_03",
        #     22: "right_shoulder_roll_03",
        #     23: "right_shoulder_yaw_02",
        #     24: "right_elbow_02",
        #     25: "right_wrist_02",
        #     31: "left_hip_pitch_04",
        #     32: "left_hip_roll_03",
        #     33: "left_hip_yaw_03",
        #     34: "left_knee_04",
        #     35: "left_ankle_02",
        #     41: "right_hip_pitch_04",
        #     42: "right_hip_roll_03",
        #     43: "right_hip_yaw_03",
        #     44: "right_knee_04",
        #     45: "right_ankle_02"
        # }

        self.sim_act_list = {
            # actuator id, nn id, kp, kd, max_torque, flip_sign
            11: Actuator(11, 1, 150.0, 8.0, 60.0, True, "left_shoulder_pitch_03"),
            12: Actuator(12, 5, 150.0, 8.0, 60.0, False, "left_shoulder_roll_03"),
            13: Actuator(13, 9, 50.0, 5.0, 17.0, False, "left_shoulder_yaw_02"),
            14: Actuator(14, 13, 50.0, 5.0, 17.0, False, "left_elbow_02"),
            15: Actuator(15, 17, 20.0, 2.0, 17.0, False, "left_wrist_02"),
            21: Actuator(21, 3, 150.0, 8.0, 60.0, False, "right_shoulder_pitch_03"),
            22: Actuator(22, 7, 150.0, 8.0, 60.0, True, "right_shoulder_roll_03"),
            23: Actuator(23, 11, 50.0, 2.0, 17.0, True, "right_shoulder_yaw_02"),
            24: Actuator(24, 15, 50.0, 5.0, 17.0, True, "right_elbow_02"),
            25: Actuator(25, 19, 20.0, 2.0, 17.0, False, "right_wrist_02"),
            # 31: Actuator(31, 0, 100.0, 6.1504, 80.0, True, "left_hip_pitch_04"),
            # 32: Actuator(32, 4, 50.0, 11.152, 60.0, False, "left_hip_roll_03"),
            # 33: Actuator(33, 8, 50.0, 11.152, 60.0, False, "left_hip_yaw_03"),
            # 34: Actuator(34, 12, 100.0, 6.1504, 80.0, True, "left_knee_04"),
            # 35: Actuator(35, 16, 20.0, 0.6, 17.0, False, "left_ankle_02"),
            # 41: Actuator(41, 2, 100, 7.0, 80.0, False, "right_hip_pitch_04"),
            # 42: Actuator(42, 6, 50.0, 11.152, 60.0, True, "right_hip_roll_03"),
            # 43: Actuator(43, 10, 50.0, 11.152, 60.0, True, "right_hip_yaw_03"),
            # 44: Actuator(44, 14, 100.0, 6.1504, 80.0, False, "right_knee_04"),
            # 45: Actuator(45, 18, 20.0, 0.6, 17.0, True, "right_ankle_02"),
        }

        self.real_act_list = {
            # actuator id, nn id, kp, kd, max_torque, flip_sign
            11: Actuator(11, 1, 150.0, 8.0, 60.0, True, "left_shoulder_pitch_03"),
            12: Actuator(12, 5, 150.0, 8.0, 60.0, False, "left_shoulder_roll_03"),
            13: Actuator(13, 9, 50.0, 5.0, 17.0, False, "left_shoulder_yaw_02"),
            14: Actuator(14, 13, 50.0, 5.0, 17.0, False, "left_elbow_02"),
            15: Actuator(15, 17, 20.0, 2.0, 17.0, False, "left_wrist_02"),
            21: Actuator(21, 3, 150.0, 8.0, 60.0, False, "right_shoulder_pitch_03"),
            22: Actuator(22, 7, 150.0, 8.0, 60.0, True, "right_shoulder_roll_03"),
            23: Actuator(23, 11, 50.0, 2.0, 17.0, True, "right_shoulder_yaw_02"),
            24: Actuator(24, 15, 50.0, 5.0, 17.0, True, "right_elbow_02"),
            25: Actuator(25, 19, 20.0, 2.0, 17.0, False, "right_wrist_02"),
            # 31: Actuator(31, 0, 100.0, 6.1504, 80.0, True, "left_hip_pitch_04"),
            # 32: Actuator(32, 4, 50.0, 11.152, 60.0, True, "left_hip_roll_03"),
            # 33: Actuator(33, 8, 50.0, 11.152, 60.0, False, "left_hip_yaw_03"),
            # 34: Actuator(34, 12, 200.0, 6.1504, 80.0, True, "left_knee_04"),
            # 35: Actuator(35, 16, 50.0, 5, 17.0, True, "left_ankle_02"),
            # 41: Actuator(41, 2, 100, 7.0, 80.0, False, "right_hip_pitch_04"),
            # 42: Actuator(42, 6, 50.0, 11.152, 60.0, False, "right_hip_roll_03"),
            # 43: Actuator(43, 10, 50.0, 11.152, 60.0, True, "right_hip_yaw_03"),
            # 44: Actuator(44, 14, 200.0, 6.1504, 80.0, False, "right_knee_04"),
            # 45: Actuator(45, 18, 50.0, 5, 17.0, False, "right_ankle_02"),
        }

    def set_curangles(self, cur_angles):
        self.cur_angles = cur_angles
    
    def set_nextangles(self, next_angles):
        if self.cur_angles is not None and self.next_angles is not None:
            self.cur_angles = self.next_angles
        self.next_angles = next_angles

    def set_idx_joint_map(self, joint_map):
        self.idx_to_joint_map = joint_map

    def isclose_to_qlimits(self, mjRobot, qpos):
        pass


    def arms_tofullq(self, leftarmq=None, rightarmq=None):
        newqpos = self.mjRobot.data.qpos.copy()
        
        if leftarmq is not None:
            left_moves = {
                "left_shoulder_pitch_03": leftarmq[0],
                "left_shoulder_roll_03": leftarmq[1],
                "left_shoulder_yaw_02": leftarmq[2],
                "left_elbow_02": leftarmq[3],
                "left_wrist_02": leftarmq[4]
            }
            
            for joint_name, value in left_moves.items():
                joint_id = mujoco.mj_name2id(self.mjRobot.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.mjRobot.model.jnt_qposadr[joint_id]

                if self.mjRobot.model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                    raise ValueError(f"Joint {joint_name} is not a hinge joint. This function only works with hinge joints (1 DOF).")
                if joint_id >= 0:
                    newqpos[qpos_index] = value
        
        # Process right arm if provided
        if rightarmq is not None:
            right_moves = {
                "right_shoulder_pitch_03": rightarmq[0],
                "right_shoulder_roll_03": rightarmq[1],
                "right_shoulder_yaw_02": rightarmq[2],
                "right_elbow_02": rightarmq[3],
                "right_wrist_02": rightarmq[4]
            }
            
            for joint_name, value in right_moves.items():
                joint_id = mujoco.mj_name2id(self.mjRobot.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.mjRobot.model.jnt_qposadr[joint_id]

                if self.mjRobot.model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                    raise ValueError(f"Joint {joint_name} is not a hinge joint. This function only works with hinge joints (1 DOF).")
                if joint_id >= 0:
                    newqpos[qpos_index] = value
        
        return newqpos
    

    def get_waypoints(self):
        if self.cur_angles is None or self.next_angles is None:
            raise ValueError("Start point or next point is not set.")
        
        trajectories = []
        max_trajectory_time = 0

        # Generate trajectory for each DoF
        for i, (current_angle, target_angle) in enumerate(zip(self.cur_angles, self.next_angles)):
            angles, velocities, times = self.find_point2target(current_angle,target_angle)
            trajectories.append((angles, velocities, times))
            max_trajectory_time = max(max_trajectory_time, times[-1])
        
        # Create a unified time grid for all trajectories
        dt = 1.0 / self.updaterate
        num_steps = int(max_trajectory_time / dt) + 1
        time_grid = [i * dt for i in range(num_steps)]
        
        # Interpolate all trajectories to the unified time grid
        all_angles = []
        all_velocities = []
        
        for i, (angles, velocities, times) in enumerate(trajectories):
            interpolated_angles = []
            interpolated_velocities = []
            
            for t in time_grid:
                if t <= times[-1]:
                    # Find the closest time point in the trajectory
                    idx = min(range(len(times)), key=lambda j: abs(times[j] - t))
                    
                    # Use the position and velocity at that time point
                    interpolated_angles.append(angles[idx])
                    velocity = velocities[idx-1] if idx > 0 and idx-1 < len(velocities) else 0.0
                    interpolated_velocities.append(velocity)
                else:
                    # After trajectory completion, maintain final position with zero velocity
                    interpolated_angles.append(angles[-1])
                    interpolated_velocities.append(0.0)
            
            all_angles.append(interpolated_angles)
            all_velocities.append(interpolated_velocities)
        
        return np.array(all_angles), np.array(all_velocities), np.array(time_grid)
    
    def find_point2target(self, cur_angle, next_angle):
        displacement = next_angle - cur_angle
        distance = abs(displacement)
        # direction = 1 if displacement >= 0 else -1

        if distance >= 2 * self.d_accel_candidate:
            T_accel = self.T_accel_candidate
            t_flat = (distance - 2 * self.d_accel_candidate) / self.vmax
            total_time = 2 * T_accel + t_flat

            def profile_velocity(t):
                if t < T_accel:
                    x = t / T_accel
                    return self.vmax * (3 * x**2 - 2 * x**3)
                elif t < T_accel + t_flat:
                    return self.vmax
                elif t < 2 * T_accel + t_flat:
                    x = (2 * T_accel + t_flat - t) / T_accel
                    return self.vmax * (3 * x**2 - 2 * x**3)
                else:
                    return 0.0
        else:
            v_peak = math.sqrt(self.acc * distance / 1.5)
            T_accel = 1.5 * v_peak / self.acc
            total_time = 2 * T_accel

            def profile_velocity(t):
                if t < T_accel:
                    x = t / T_accel
                    return v_peak * (3 * x**2 - 2 * x**3)
                else:
                    x = (2 * T_accel - t) / T_accel
                    return v_peak * (3 * x**2 - 2 * x**3)

        dt = 1.0/ self.updaterate
        steps = int(total_time / dt)
        next_tick = time.perf_counter()
        
        # Create lists to store trajectory data
        trajectory_angles = [cur_angle]
        trajectory_velocities = []
        trajectory_times = [0.0]
        
        t = 0.0

        for _ in range(steps):
            velocity = profile_velocity(t)
            signed_velocity = math.copysign(velocity, displacement)
            cur_angle += signed_velocity * dt
            
            # Store trajectory data
            trajectory_velocities.append(signed_velocity)
            trajectory_angles.append(cur_angle)
            t += dt
            trajectory_times.append(t)
            next_tick += dt

        # Set final angle to exactly target
        trajectory_angles[-1] = next_angle

        return trajectory_angles, trajectory_velocities, trajectory_times
    
    def apply_pose_delta(self, current_pos, current_quat, delta_pos=[0, 0, 0], delta_euler=[0, 0, 0]):
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




