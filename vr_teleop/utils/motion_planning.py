import logging
from vr_teleop.ikrobot import KBot_Robot
import mujoco
import time
import math
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # .DEBUG .INFO



class Robot_Planner:
    def __init__(self, robot, acc=50.0, vmax=30.0, updaterate=100.0):
        
        self.profile = "scurve"
        self.acc = acc
        self.vmax = vmax
        self.updaterate = updaterate
        self.T_accel_candidate = 1.5 * self.vmax / self.acc  # duration of acceleration phase
        self.d_accel_candidate = 0.5 * self.vmax * self.T_accel_candidate  # distance during acceleration phase

        self.robot = robot
        self.cur_angles = None
        self.next_angles = None
        self.waypoints = []

    def set_curangles(self, cur_angles):
        self.cur_angles = cur_angles
    
    def set_nextangles(self, next_angles):
        self.next_angles = next_angles

    def isclose_to_qlimits(self, robot, qpos):
        pass

    def arms_tofullq(self, leftarmq=None, rightarmq=None):
        newqpos = self.robot.data.qpos.copy()
        
        if leftarmq is not None:
            left_moves = {
                "left_shoulder_pitch_03": leftarmq[0],
                "left_shoulder_roll_03": leftarmq[1],
                "left_shoulder_yaw_02": leftarmq[2],
                "left_elbow_02": leftarmq[3],
                "left_wrist_02": leftarmq[4]
            }
            
            for joint_name, value in left_moves.items():
                joint_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.robot.model.jnt_qposadr[joint_id]

                if self.robot.model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
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
                joint_id = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_index = self.robot.model.jnt_qposadr[joint_id]

                if self.robot.model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
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
        
        return np.array(all_angles), np.array(all_velocities)
    
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





