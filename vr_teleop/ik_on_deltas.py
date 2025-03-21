import logging
import mujoco
import time
import math
import numpy as np
from vr_teleop.ikrobot import KBot_Robot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.utils.ik import *

import asyncio
from pykos import KOS


urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
solver = KBot_Robot(urdf_path)

fullq = solver.convert_armqpos_to_fullqpos([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)

motion_plan = Robot_Planner(solver)

llocs = solver.get_limit_center(leftside=True)
rlocs = solver.get_limit_center(leftside=False)

fullq = motion_plan.arms_tofullq(llocs, rlocs)

motion_plan.set_curangles(solver.data.qpos)
motion_plan.set_nextangles(fullq)

idx_to_joint_map = solver.qpos_idx_to_jointname()

planned_angles, _ , time_grid = motion_plan.get_waypoints()
motion_plan.set_idx_joint_map(idx_to_joint_map)


solver.set_qpos(fullq)
lee_pos, lee_ort = solver.get_ee_pos(leftside=True)
ree_pos, ree_ort = solver.get_ee_pos(leftside=False)


#* KOS-SIM MOVE TO STARTING POSITIONS
#* num_dofs, num_waypoints = all_angles.shape


async def send_to_kos(planner, kos_instance, all_angles, time_grid, sim=True):
    all_angles = np.degrees(all_angles)
    start_time = time.time()

    all_actuators_ids = list(planner.sim_act_list.keys())

    if sim:
        flip_sign_map = {actuator_id: planner.sim_act_list[actuator_id].flip_sign for actuator_id in  all_actuators_ids if actuator_id in planner.sim_act_list}
    else:
        flip_sign_map = {actuator_id: planner.real_act_list[actuator_id].flip_sign for actuator_id in all_actuators_ids if actuator_id in planner.real_act_list}
    
    # Store the mapping between actuator_id and corresponding joint index
    actuator_to_joint_idx = {}
    for i, actuator_id in enumerate(all_actuators_ids):
        joint_name = planner.sim_act_list[actuator_id].joint_name
        for idx, joint in planner.idx_to_joint_map.items():
            if joint == joint_name:
                actuator_to_joint_idx[actuator_id] = idx
                break

    adjusted_target_angles = []

    for step, t in enumerate(time_grid):
        current_time = time.time()
        
        # Remove breakpoint
        command = []
        for actuator_id in all_actuators_ids:
            if actuator_id in actuator_to_joint_idx:
                joint_idx = actuator_to_joint_idx[actuator_id]
                # Use joint_idx to access the correct position in all_angles
                command.append({
                    "actuator_id": actuator_id,
                    "position": all_angles[joint_idx][step],
                    # "velocity": all_velocities[i][step]
                })
        
        command_tasks = []

        command_tasks.append(kos_instance.actuator.command_actuators(command))
        await asyncio.gather(*command_tasks)
        
        next_time = start_time + t
        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)

    # Final command to ensure we hit target positions exactly
    final_command = [
        {
            "actuator_id": actuator_id,
            "position": target_angle,
        }
        for actuator_id, target_angle in zip(all_actuators_ids, adjusted_target_angles)
    ]

    command_tasks = []
    command_tasks.append(kos_instance.actuator.command_actuators(final_command))

    await asyncio.gather(*command_tasks)

    state_response = await kos_instance.actuator.get_actuators_state(all_actuators_ids)
    for i, (actuator_id, state) in enumerate(zip(all_actuators_ids, state_response.states)):
        logger.info(f"Final position of actuator {actuator_id}: {state.position} degrees")


async def activate(kos_instance, planner):
    kos_instance.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})

    disable_commands = []
    for cur_act in planner.sim_act_list.keys():
        disable_commands.append(
                kos_instance.actuator.configure_actuator(actuator_id=cur_act, torque_enabled=False)
            )
    
    await asyncio.gather(*disable_commands)
    await asyncio.sleep(1)

    config_commands = []
    for cur_act in planner.sim_act_list.values():
        config_commands.append(
            kos_instance.actuator.configure_actuator(
                actuator_id=cur_act.actuator_id,
                kp=cur_act.kp,
                kd=cur_act.kd,
                torque_enabled=True,
            )
        )
    await asyncio.gather(*config_commands)
    await asyncio.sleep(1)

async def disable(kos_instance, planner):
    kos_instance.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})

    disable_commands = []
    for cur_act in planner.sim_act_list.keys():
        disable_commands.append(
                kos_instance.actuator.configure_actuator(actuator_id=cur_act, torque_enabled=False)
            )
    
    await asyncio.gather(*disable_commands)
    await asyncio.sleep(1)




async def run_kos():
    async with KOS(ip="10.33.12.161", port=50051) as sim_kos:
        await activate(sim_kos, motion_plan)
        await send_to_kos(motion_plan, sim_kos, planned_angles, time_grid)
        await asyncio.sleep(2)
        await disable(sim_kos, motion_plan)


asyncio.run(run_kos())


new_pos, new_quat = motion_plan.apply_pose_delta(lee_pos, lee_ort, [0.3, 0.3, -0.1], [0, 0, 0])
pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(solver.model, solver.data, new_pos, new_quat, leftside=True)
nextqpos = motion_plan.arms_tofullq(leftarmq=pos_arm, rightarmq=solver.get_limit_center(leftside=False))


motion_plan.set_nextangles(nextqpos)
planned_angles, _ , time_grid = motion_plan.get_waypoints()




#* KOS-SIM MOVE TO NEXT POSITIONS
#* num_dofs, num_waypoints = all_angles.shape



async def run_kos():
    async with KOS(ip="10.33.12.161", port=50051) as sim_kos:
        await activate(sim_kos, motion_plan)
        await send_to_kos(motion_plan, sim_kos, planned_angles, time_grid)
        await asyncio.sleep(2)
        await disable(sim_kos, motion_plan)


asyncio.run(run_kos())




# solver.set_qpos(all_angles[:, 1])

# def runThis(robot, sim_time):
#     # Keep track of the last solution index and the time it was applied
#     if not hasattr(runThis, "last_index"):
#         runThis.last_index = -1
#         runThis.last_change_time = 0
    
#     # Calculate the current step based on 0.75-second intervals
#     current_step = int(sim_time / 0.2)
#     target_time = current_step * 0.2
    
#     if sim_time > target_time and sim_time <= target_time + 0.01 and current_step > runThis.last_index:
#         next_index = runThis.last_index + 1
        
#         if next_index < len(all_angles[1, :]):
#             print(f"Time: {sim_time:.2f}s - Moving to position {next_index}")
#             robot.set_qpos(all_angles[:, next_index])
            
#             runThis.last_index = next_index
#             runThis.last_change_time = sim_time


# solver.run_viewer(runThis)

