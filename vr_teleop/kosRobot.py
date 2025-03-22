from pykos import KOS
from dataclasses import dataclass
import numpy as np
import asyncio
import time
import logging

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


class KOS_KBot:
    @staticmethod
    def get_sim_actuators():
        return {
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
    
    @staticmethod
    def get_real_actuators():
        return {
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
    
    def __init__(self, kos, planner, sim: bool):
        self.kos_instance = kos
        self.planner = planner
        self.sim_act_list = KOS_KBot.get_sim_actuators()
        self.real_act_list = KOS_KBot.get_real_actuators()
        self.sim = sim


    async def send_to_kos(self, all_angles, time_grid, idx_to_joint_map):
        all_angles = np.degrees(all_angles)
        start_time = time.time()

        all_actuators_ids = list(self.sim_act_list.keys())

        if self.sim:
            flip_sign_map = {actuator_id: self.sim_act_list[actuator_id].flip_sign for actuator_id in  all_actuators_ids if actuator_id in self.sim_act_list}
        else:
            flip_sign_map = {actuator_id: self.real_act_list[actuator_id].flip_sign for actuator_id in all_actuators_ids if actuator_id in self.real_act_list}
        
        # Store the mapping between actuator_id and corresponding joint index
        actuator_to_joint_idx = {}
        for i, actuator_id in enumerate(all_actuators_ids):
            joint_name = self.sim_act_list[actuator_id].joint_name
            for idx, joint in idx_to_joint_map.items():
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

            command_tasks.append(self.kos_instance.actuator.command_actuators(command))
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
        command_tasks.append(self.kos_instance.actuator.command_actuators(final_command))

        await asyncio.gather(*command_tasks)

        state_response = await self.kos_instance.actuator.get_actuators_state(all_actuators_ids)
        for i, (actuator_id, state) in enumerate(zip(all_actuators_ids, state_response.states)):
            logger.info(f"Final position of actuator {actuator_id}: {state.position} degrees")


    async def activate(self):
        disable_commands = []
        for cur_act in self.sim_act_list.keys():
            disable_commands.append(
                    self.kos_instance.actuator.configure_actuator(actuator_id=cur_act, torque_enabled=False)
                )
        
        await asyncio.gather(*disable_commands)
        await asyncio.sleep(1)

        config_commands = []
        for cur_act in self.sim_act_list.values():
            config_commands.append(
                self.kos_instance.actuator.configure_actuator(
                    actuator_id=cur_act.actuator_id,
                    kp=cur_act.kp,
                    kd=cur_act.kd,
                    torque_enabled=True,
                )
            )
        await asyncio.gather(*config_commands)
        await asyncio.sleep(1)

    async def disable(kos_instance, planner):
        disable_commands = []
        for cur_act in planner.sim_act_list.keys():
            disable_commands.append(
                    kos_instance.actuator.configure_actuator(actuator_id=cur_act, torque_enabled=False)
                )
        
        await asyncio.gather(*disable_commands)
        await asyncio.sleep(1)



