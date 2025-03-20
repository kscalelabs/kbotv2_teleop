import math
import time

import asyncio
import logging
import time
from dataclasses import dataclass
import logging
from pykos import KOS
from .actuator_list import ACTUATOR_LIST_SIM

logger = logging.getLogger(__name__)


#* From Scott's basic motion planning
def find_points_to_target(current_angle: float, target: float, acceleration: float = 50.0,
                         V_MAX: float = 30.0, update_rate: float = 100.0, profile: str = "scurve") -> tuple:
    """
    Generate a trajectory for an actuator to move from the current angle to a target angle.
    
    This function creates a motion profile with either a linear (trapezoidal/triangular) or 
    s-curve velocity profile, and returns the trajectory as lists of angles, velocities, and times.
    
    Args:
        current_angle (float): The starting angle in degrees.
        target (float): The target angle in degrees.
        acceleration (float, optional): Maximum acceleration in deg/s². Defaults to 100.0.
        V_MAX (float, optional): Maximum velocity in deg/s. Defaults to 30.0.
        update_rate (float, optional): Frequency of trajectory points in Hz. Defaults to 100.0.
        profile (str, optional): Motion profile type, either "linear" or "scurve". Defaults to "scurve".
            - "linear": Trapezoidal velocity profile (or triangular if distance is short)
            - "scurve": Smoothed velocity profile with continuous acceleration
    
    Returns:
        tuple: A tuple containing three lists:
            - trajectory_angles (list[float]): Angle positions at each time step
            - trajectory_velocities (list[float]): Velocities at each time step
            - trajectory_times (list[float]): Time points for each step in seconds
    
    Raises:
        ValueError: If an unknown profile type is specified.
    """
    displacement = target - current_angle
    distance = abs(displacement)
    direction = 1 if displacement >= 0 else -1

    if profile == "linear":
        # --- Linear (trapezoidal) profile (with triangular fallback) ---
        t_accel = V_MAX / acceleration
        d_accel = 0.5 * acceleration * t_accel**2

        if distance >= 2 * d_accel:
            # Trapezoidal case: accelerate to V_MAX, cruise, then decelerate.
            d_flat = distance - 2 * d_accel
            t_flat = d_flat / V_MAX
            total_time = 2 * t_accel + t_flat
            print(f"Total time: {total_time}")

            def profile_velocity(t):
                if t < t_accel:
                    return acceleration * t
                elif t < t_accel + t_flat:
                    return V_MAX
                else:
                    return V_MAX - acceleration * (t - t_accel - t_flat)
        else:
            print("Triangular case! Time too short to reach V_MAX.")
            # Triangular case: move too short to reach V_MAX.
            t_phase = math.sqrt(distance / acceleration)
            v_peak = acceleration * t_phase
            total_time = 2 * t_phase

            def profile_velocity(t):
                if t < t_phase:
                    return acceleration * t
                else:
                    return v_peak - acceleration * (t - t_phase)

    elif profile == "scurve":
        # --- S-curve profile using smoothstep for continuous acceleration (and jerk) ---
        T_accel_candidate = 1.5 * V_MAX / acceleration  # duration of acceleration phase
        d_accel_candidate = 0.5 * V_MAX * T_accel_candidate  # distance during acceleration phase

        if distance >= 2 * d_accel_candidate:
            T_accel = T_accel_candidate
            t_flat = (distance - 2 * d_accel_candidate) / V_MAX
            total_time = 2 * T_accel + t_flat

            def profile_velocity(t):
                if t < T_accel:
                    x = t / T_accel
                    return V_MAX * (3 * x**2 - 2 * x**3)
                elif t < T_accel + t_flat:
                    return V_MAX
                elif t < 2 * T_accel + t_flat:
                    x = (2 * T_accel + t_flat - t) / T_accel
                    return V_MAX * (3 * x**2 - 2 * x**3)
                else:
                    return 0.0
        else:
            v_peak = math.sqrt(acceleration * distance / 1.5)
            T_accel = 1.5 * v_peak / acceleration
            total_time = 2 * T_accel

            def profile_velocity(t):
                if t < T_accel:
                    x = t / T_accel
                    return v_peak * (3 * x**2 - 2 * x**3)
                else:
                    x = (2 * T_accel - t) / T_accel
                    return v_peak * (3 * x**2 - 2 * x**3)
    else:
        raise ValueError("Unknown profile type. Choose 'linear' or 'scurve'.")

    dt = 1.0/ update_rate
    steps = int(total_time / dt)
    next_tick = time.perf_counter()
    
    # Create lists to store trajectory data
    trajectory_angles = [current_angle]
    trajectory_velocities = []
    trajectory_times = [0.0]
    
    t = 0.0

    for _ in range(steps):
        velocity = profile_velocity(t)
        signed_velocity = math.copysign(velocity, displacement)
        current_angle += signed_velocity * dt
        
        # Store trajectory data
        trajectory_velocities.append(signed_velocity)
        trajectory_angles.append(current_angle)
        t += dt
        trajectory_times.append(t)
        next_tick += dt

    # Set final angle to exactly target
    trajectory_angles[-1] = target

    return trajectory_angles, trajectory_velocities, trajectory_times





async def move_actuators_with_trajectory(sim_kos: KOS, actuator_ids: list[int], target_angles: list[float], 
                                       real_kos: KOS = None) -> None:
    """
    Move multiple actuators from their current positions to target positions using trajectories.
    
    Args:
        sim_kos: KOS simulation instance
        actuator_ids: List of IDs of the actuators to move
        target_angles: List of target angles in degrees (must match length of actuator_ids)
        real_kos: Optional KOS real robot instance
    """
    if len(actuator_ids) != len(target_angles):
        raise ValueError("Number of actuator IDs must match number of target angles")
    
    flip_sign_map = {actuator_id: ACTUATOR_LIST_SIM[actuator_id].flip_sign for actuator_id in actuator_ids if actuator_id in ACTUATOR_LIST_SIM}
    
    adjusted_target_angles = []
    for actuator_id, target_angle in zip(actuator_ids, target_angles):
        if actuator_id in flip_sign_map and flip_sign_map[actuator_id]:
            adjusted_target_angles.append(-target_angle)
        else:
            adjusted_target_angles.append(target_angle)
    
    kos_to_use = sim_kos if sim_kos is not None else real_kos
    if kos_to_use is None:
        raise ValueError("Both sim_kos and real_kos cannot be None")
        
    state_response = await kos_to_use.actuator.get_actuators_state(actuator_ids)
    current_angles = [state.position for state in state_response.states]
    
    trajectories = []
    max_trajectory_length = 0
    max_trajectory_time = 0
    
    for i, (actuator_id, current_angle, target_angle) in enumerate(zip(actuator_ids, current_angles, adjusted_target_angles)):
        logger.info(f"Generating trajectory for actuator {actuator_id} from {current_angle} to {target_angle} degrees...")
        angles, velocities, times = find_points_to_target(
            current_angle=current_angle,
            target=target_angle,
        )
        trajectories.append((angles, velocities, times))
        max_trajectory_length = max(max_trajectory_length, len(angles))
        max_trajectory_time = max(max_trajectory_time, times[-1])
    
    logger.info(f"Executing trajectories for {len(actuator_ids)} actuators...")
    start_time = time.time()
    
    dt = 1.0 / 100.0  # Assuming 100Hz update rate
    num_steps = int(max_trajectory_time / dt) + 1
    time_grid = [i * dt for i in range(num_steps)]
    
    for step, t in enumerate(time_grid):
        current_time = time.time()
        
        command = []
        for i, (actuator_id, (angles, velocities, times)) in enumerate(zip(actuator_ids, trajectories)):
            if t <= times[-1]:
                idx = min(range(len(times)), key=lambda j: abs(times[j] - t))
                
                position = angles[idx]
                velocity = velocities[idx-1] if idx > 0 and idx-1 < len(velocities) else 0.0
                
                command.append({
                    "actuator_id": actuator_id,
                    "position": position,
                    "velocity": velocity
                })
            else:
                command.append({
                    "actuator_id": actuator_id,
                    "position": angles[-1],
                    "velocity": 0.0
                })
        
        command_tasks = []
        if sim_kos:
            command_tasks.append(sim_kos.actuator.command_actuators(command))
        if real_kos:
            command_tasks.append(real_kos.actuator.command_actuators(command))
        
        await asyncio.gather(*command_tasks)
        
        next_time = start_time + t + dt
        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)
    
    final_command = [
        {
            "actuator_id": actuator_id,
            "position": target_angle,
            "velocity": 0.0
        }
        for actuator_id, target_angle in zip(actuator_ids, adjusted_target_angles)
    ]
    
    command_tasks = []
    if sim_kos:
        command_tasks.append(sim_kos.actuator.command_actuators(final_command))
    if real_kos:
        command_tasks.append(real_kos.actuator.command_actuators(final_command))
    
    await asyncio.gather(*command_tasks)
    
    state_response = await kos_to_use.actuator.get_actuators_state(actuator_ids)
    for i, (actuator_id, state) in enumerate(zip(actuator_ids, state_response.states)):
        logger.info(f"Final position of actuator {actuator_id}: {state.position} degrees")