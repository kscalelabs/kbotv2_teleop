import asyncio
import json
import time
import websockets
from pykos import KOS
from vr_teleop.mjRobot import MJ_KBot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.kosRobot import KOS_KBot

from .vr_cont import process_message
from .utils.logging import setup_logger

logger = setup_logger(__name__)


class TeleopSession:
    def __init__(self, kos_instance: KOS, actuator_list, sim: bool, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
        self.mjRobot = MJ_KBot(urdf_path)
        self.planner = Robot_Planner(self.mjRobot)
        self.koskbot = KOS_KBot(kos_instance, self.planner,sim=sim)

        self.relative_pos = None
        self.startpos_lock = asyncio.Lock()
        self.running = True

    async def initialize(self):
        """Initialize robot: reset and enable actuators."""
        await self.kos.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})
        await disable_actuators(self.kos, self.actuators)
        await enable_actuators(self.kos, self.actuators)
        logger.info("Robot actuators enabled")

    async def shutdown(self):
        """Clean shutdown of actuators."""
        logger.info("Disabling actuators...")
        await disable_actuators(self.kos, self.actuators)

    async def update_goal(self, message: str):
        """Process incoming WebSocket message and update goal."""
        try:
            new_goal = await process_message(message)
            async with self.goal_lock:
                self.goal_state = new_goal
            logger.info(f"Updated goal state: {new_goal}")
        except Exception as e:
            logger.error(f"Failed to process message: {e}")

    async def control_loop(self, hz: int = 100):
        """Main robot control loop (runs continuously)."""
        logger.info("Robot control loop started")
        period = 1.0 / hz
        while self.running:
            try:
                async with self.goal_lock:
                    goal = self.goal_state

                if goal is not None:
                    # TODO: Replace with real IK + motion planning
                    state_response = await self.kos.actuator.get_actuators_state(self.actuators.keys())
                    current_joint_states = [state.position for state in state_response.states]

                    # Placeholder motion plan
                    # next_joint_states = inverse_kinematics(goal)
                    # planned_commands = motion_planning(current_joint_states, next_joint_states)

                    # await self.kos.actuator.command_actuators(planned_commands)
                    logger.debug("Control loop: would send commands here")

                await asyncio.sleep(period)
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.1)


async def handle_connection(websocket):
    """Handle each client connection."""
    async with KOS(ip="localhost", port=50051) as kos:
        real_robot = False
        actuator_list = ACTUATOR_LIST_REAL if real_robot else ACTUATOR_LIST_SIM
        session = TeleopSession(kos, actuator_list, real_robot)

        await session.initialize()

        # Start the background control loop
        control_task = asyncio.create_task(session.control_loop())

        try:
            async for message in websocket:
                logger.info("Received WebSocket message")
                await session.update_goal(message)
                await websocket.send(json.dumps({"status": "success"}))
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            session.running = False
            await session.shutdown()
            await control_task


async def main():
    """Start WebSocket server."""
    server = await websockets.serve(handle_connection, "localhost", 8586)
    logger.info("WebSocket server started at ws://localhost:8586")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
