"""Runs basic performance profiling using the KOS SDK.

Currently, we are able to achieve 360 calls per second, but when WebRTC is active, this drops to closer to 120
calls per second. See the technical notes [here](https://notion.kscale.dev/zbot-tech-notes) for more details.

WebRTC portal: http://192.168.42.1:8083/pages/player/webrtc/s1/0
"""

import asyncio
import logging
import time

import colorlogging
import matplotlib.pyplot as plt
import pykos
from typing import Any

logger = logging.getLogger(__name__)



class TeleopController:
    def __init__(self, kos: pykos.KOS):
        self.kos = kos
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Starting test-00")
        self.counter = 0 
        self.actuator_ids = [11, 12, 13, 14, 35, 21, 22, 23, 24, 45]
        self.aidtoname = {
            11: "left_shoulder_pitch_03",
            12: "left_shoulder_roll_03",
            13: "left_shoulder_yaw_02",
            14: "left_elbow_02",
            35: "left_wrist_roll_02",
            21: "right_shoulder_pitch_03",
            22: "right_shoulder_roll_03",
            23: "right_shoulder_yaw_02",
            24: "right_elbow_02",
            45: "right_wrist_roll_02",
        }
        self.aidinv = {
            11: 1,  # left_shoulder_pitch_03
            12: 1,  # left_shoulder_roll_03
            13: 1,  # left_shoulder_yaw_02
            14: 1,  # left_elbow_02
            35: 1,  # left_wrist_roll_02
            21: 1,  # right_shoulder_pitch_03
            22: 1,  # right_shoulder_roll_03
            23: 1,  # right_shoulder_yaw_02
            24: 1,  # right_elbow_02
            45: 1,  # right_wrist_roll_02
        }
        self.current_states = None
        self.current_positions = {
            11: None,  # left_shoulder_pitch_03
            12: None,  # left_shoulder_roll_03
            13: None,  # left_shoulder_yaw_02
            14: None,  # left_elbow_02
            35: None,  # left_wrist_roll_02
            21: None,  # right_shoulder_pitch_03
            22: None,  # right_shoulder_roll_03
            23: None,  # right_shoulder_yaw_02
            24: None,  # right_elbow_02
            45: None,  # right_wrist_roll_02
        }
        self.current_clipped_positions = {
            11: None,  # left_shoulder_pitch_03
            12: None,  # left_shoulder_roll_03
            13: None,  # left_shoulder_yaw_02
            14: None,  # left_elbow_02
            35: None,  # left_wrist_roll_02
            21: None,  # right_shoulder_pitch_03
            22: None,  # right_shoulder_roll_03
            23: None,  # right_shoulder_yaw_02
            24: None,  # right_elbow_02
            45: None,  # right_wrist_roll_02
        }
        self.prev_clipped_positions = {
            11: None,  # left_shoulder_pitch_03
            12: None,  # left_shoulder_roll_03
            13: None,  # left_shoulder_yaw_02
            14: None,  # left_elbow_02
            35: None,  # left_wrist_roll_02
            21: None,  # right_shoulder_pitch_03
            22: None,  # right_shoulder_roll_03
            23: None,  # right_shoulder_yaw_02
            24: None,  # right_elbow_02
            45: None,  # right_wrist_roll_02
        }
        self.joint_limits = {
            11: {"min": -50.0, "max": 50.0},  # left_shoulder_pitch_03
            12: {"min": -50.0, "max": 50.0},  # left_shoulder_roll_03
            13: {"min": -50.0, "max": 50.0},  # left_shoulder_yaw_02
            14: {"min": -50.0, "max": 50.0},  # left_elbow_02
            35: {"min": -50.0, "max": 50.0},  # left_wrist_roll_02
            21: {"min": -50.0, "max": 50.0},  # right_shoulder_pitch_03
            22: {"min": -50.0, "max": 50.0},  # right_shoulder_roll_03
            23: {"min": -50.0, "max": 50.0},  # right_shoulder_yaw_02
            24: {"min": -50.0, "max": 50.0},  # right_elbow_02
            45: {"min": -50.0, "max": 50.0},  # right_wrist_roll_02
        }

    async def zero_joints(self) -> None:
        for i in range(10):
            await asyncio.sleep(0.1)
            self.logger.info("Zeroing joints %d", i)
            for actuator_id in self.actuator_ids:
                await self.kos.actuator.configure_actuator(
                    actuator_id=actuator_id,
                    kp=32.0,
                    kd=32.0,
                    zero_position=True,
                    torque_enabled=False,
                )

    async def configure_controller(self) -> None:
        for actuator_id in self.actuator_ids:
            self.logger.info(
                "Configuring actuator %d name %s",
                actuator_id,
                self.aidtoname[actuator_id],
            )
            await self.kos.actuator.configure_actuator(
                actuator_id=actuator_id,
                kp=32.0,
                kd=32.0,
                zero_position=False,
                torque_enabled=False,
            )

    async def log_actuator_states(self, states: Any) -> None:
        self.logger.info("Actuator states:")
        for actuator_state in states.states:
            self.logger.info(
                "actuator id %2d: %-25s pos: %6.2f",
                actuator_state.actuator_id,
                self.aidtoname[actuator_state.actuator_id],
                actuator_state.position,
            )

    async def log_positions(self, clipped: bool = False) -> None:
        positions = (
            self.current_clipped_positions if clipped else self.current_positions
        )
        clipped_str = "Clipped" if clipped else "Raw"
        self.logger.info("%s positions:", clipped_str)
        for actuator_id, position in positions.items():
            self.logger.info(
                "actuator id %2d: %-25s pos: %6.2f",
                actuator_id,
                self.aidtoname[actuator_id],
                position,
            )

    async def get_states(self) -> list[float]:
        states = await self.kos.actuator.get_actuators_state(self.actuator_ids)
        await self.log_actuator_states(states)
        self.current_states = states
        for state in self.current_states.states:
            self.current_positions[state.actuator_id] = state.position
        return states

    async def clip_current_positions(self) -> list[float]:
        for actuator_id, position in self.current_positions.items():
            self.current_clipped_positions[actuator_id] = max(
                self.joint_limits[actuator_id]["min"],
                min(position, self.joint_limits[actuator_id]["max"]),
            )
        return self.current_clipped_positions

    async def send_joint_positions(self) -> None:
        kbot_kos = None  # pykos.KOS("10.33.87.40")
        commands = []
        for i, actuator_id in enumerate(self.actuator_ids):
            commands.append(
                pykos.ActuatorCommand(
                    actuator_id=actuator_id,
                    position=self.current_states.states[i].position,
                )
            )
        await kbot_kos.actuator.set_actuators_positions(commands)

    # Function to measure number of calls per second
    async def run_teleop_controller(self) -> None:

        while True:
            self.counter += 1
            # Get the current joint positions
            states = await self.get_states()
            if self.counter == 0:
                self.prev_clipped_positions = await self.clip_current_positions()
            clipped_positions = await self.clip_current_positions()
            await self.log_positions(clipped=False)
            await self.log_positions(clipped=True)
            await asyncio.sleep(0.01)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting teleop controller")

    try:
        async with pykos.KOS("10.33.87.40") as kos:
            teleop_controller = TeleopController(kos)
            # await teleop_controller.zero_joints()
            await teleop_controller.configure_controller()
            await teleop_controller.run_teleop_controller()
    except Exception:
        logger.exception(
            "Make sure that the Z-Bot is connected over USB and the IP address is accessible."
        )
        raise


# Run the performance test for 10 seconds
if __name__ == "__main__":
    asyncio.run(main())
