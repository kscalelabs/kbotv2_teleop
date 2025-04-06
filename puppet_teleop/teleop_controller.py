"""Runs basic performance profiling using the KOS SDK.

Currently, we are able to achieve 360 calls per second, but when WebRTC is active, this drops to closer to 120
calls per second. See the technical notes [here](https://notion.kscale.dev/zbot-tech-notes) for more details.

WebRTC portal: http://192.168.42.1:8083/pages/player/webrtc/s1/0
"""

import asyncio
import logging
import time
import signal
import sys

import colorlogging
import matplotlib.pyplot as plt
import pykos
from typing import Any

logger = logging.getLogger(__name__)

# Global variable to store the teleop controller instance
teleop_controller_instance = None

async def disable_kbot_torque():
    """Disable torque on all KBot actuators."""
    if teleop_controller_instance and teleop_controller_instance.kbot_kos:
        logger.warning("Disabling torque on KBot actuators...")
        kbot_actuator_ids = [11, 12, 13, 14, 21, 22, 23, 24]
        for actuator_id in kbot_actuator_ids:
            try:
                await teleop_controller_instance.kbot_kos.actuator.configure_actuator(
                    actuator_id=actuator_id,
                    kp=32.0,
                    kd=32.0,
                    zero_position=False,
                    torque_enabled=False,
                )
                logger.info(f"Disabled torque on KBot actuator {actuator_id}")
            except Exception as e:
                logger.error(f"Failed to disable torque on KBot actuator {actuator_id}: {e}")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal by disabling torque on KBot actuators."""
    logger.warning("Ctrl+C detected. Disabling torque on KBot actuators...")
    asyncio.run(disable_kbot_torque())
    sys.exit(0)

class TeleopController:
    def __init__(self, kos: pykos.KOS):
        self.kos = kos
        self.kbot_kos = pykos.KOS("10.33.13.78")  # Initialize kbot_kos connection here
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
            12: -1,  # left_shoulder_roll_03
            13: 1,  # left_shoulder_yaw_02
            14: -1,  # left_elbow_02
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

            self.kbot_kos.connect()

    async def configure_kbot_controller(self) -> None:
        """Configure the KBot actuators with IDs 11-14 and 21-24 to enable torque."""
        kbot_actuator_ids = [11, 12, 13, 14, 21, 22, 23, 24]
        for actuator_id in kbot_actuator_ids:
            self.logger.info(
                "Configuring KBot actuator %d",
                actuator_id,
            )
            await self.kbot_kos.actuator.configure_actuator(
                actuator_id=actuator_id,
                kp=20.0,
                kd=2.0,
                zero_position=False,
                torque_enabled=True,
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
        commands = []
        for i, actuator_id in enumerate(self.actuator_ids):
            # Map controller positions 35 and 45 to KBot positions 15 and 25
            kbot_actuator_id = actuator_id
            if actuator_id == 35:
                kbot_actuator_id = 15  # Map left wrist roll from 35 to 15
            elif actuator_id == 45:
                kbot_actuator_id = 25  # Map right wrist roll from 45 to 25
                
            # Apply sign inversion based on aidinv mapping
            position = self.current_states.states[i].position
            if self.aidinv[actuator_id] == -1:
                position = -position
                
            commands.append(
                {
                    "actuator_id": kbot_actuator_id,
                    "position": position,
                }
            )
        await self.kbot_kos.actuator.command_actuators(commands)

    async def get_kbot_positions(self) -> None:
        """Get and log the current positions of the KBot actuators for debugging."""
        try:
            # Create a modified list of actuator IDs for KBot (15 and 25 instead of 35 and 45)
            kbot_actuator_ids = []
            for actuator_id in self.actuator_ids:
                if actuator_id == 35:
                    kbot_actuator_ids.append(15)  # Map left wrist roll from 35 to 15
                elif actuator_id == 45:
                    kbot_actuator_ids.append(25)  # Map right wrist roll from 45 to 25
                else:
                    kbot_actuator_ids.append(actuator_id)
            
            kbot_states = await self.kbot_kos.actuator.get_actuators_state(kbot_actuator_ids)
            self.logger.info("KBot current positions:")
            for actuator_state in kbot_states.states:
                # Map KBot actuator IDs 15 and 25 back to controller IDs 35 and 45 for consistent logging
                display_actuator_id = actuator_state.actuator_id
                if actuator_state.actuator_id == 15:
                    display_actuator_id = 35  # Map left wrist roll from 15 to 35
                elif actuator_state.actuator_id == 25:
                    display_actuator_id = 45  # Map right wrist roll from 25 to 45
                
                # Apply sign inversion based on aidinv mapping for display
                position = actuator_state.position
                if self.aidinv[display_actuator_id] == -1:
                    position = -position
                
                self.logger.info(
                    "actuator id %2d: %-25s pos: %6.2f",
                    display_actuator_id,
                    self.aidtoname[display_actuator_id],
                    position,
                )
        except Exception as e:
            self.logger.error(f"Failed to get KBot positions: {e}")

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
            
            # Send joint positions to KBot and get current positions for debugging
            await self.send_joint_positions()
            await self.get_kbot_positions()
            
            await asyncio.sleep(0.01)


async def main() -> None:
    colorlogging.configure()
    logger.warning("Starting teleop controller")

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        async with pykos.KOS("10.33.85.8") as kos:
            global teleop_controller_instance
            teleop_controller = TeleopController(kos)
            teleop_controller_instance = teleop_controller  # Store the instance globally
            # await teleop_controller.zero_joints()
            await teleop_controller.configure_controller()
            await teleop_controller.configure_kbot_controller()
            await teleop_controller.run_teleop_controller()
    except Exception:
        logger.exception(
            "Make sure that the Z-Bot is connected over USB and the IP address is accessible."
        )
        raise
    finally:
        # Ensure torque is disabled even if an exception occurs
        await disable_kbot_torque()


# Run the performance test for 10 seconds
if __name__ == "__main__":
    asyncio.run(main())
