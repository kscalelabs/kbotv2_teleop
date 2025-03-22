import asyncio
import json
import websockets
from pykos import KOS
from .utils.logging import setup_logger

from vr_teleop.utils.ik import *
from vr_teleop.mjRobot import MJ_KBot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.kosRobot import KOS_KBot


logger = setup_logger(__name__)

class TeleopSession:
    def __init__(self, kos_instance: KOS, sim: bool, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
        self.mjRobot = MJ_KBot(urdf_path)
        self.planner = Robot_Planner(self.mjRobot)
        self.koskbot = KOS_KBot(kos_instance, self.planner,sim=sim)

        # Add an event for button state changes
        self.left_btn_event = asyncio.Event() 
        self.left_btn = None
        self.left_last_pos = None
        self.left_last_rot = None
        self.left_lock = asyncio.Lock()
        self.left_goal = None

        self.right_btn_event = asyncio.Event()
        self.right_btn = None
        self.right_last_pos = None
        self.right_last_rot = None
        self.right_lock = asyncio.Lock()
        self.right_goal = None

        self.running = True

        self.logger = setup_logger(__name__)

    async def initialize(self):
        self.logger.info("Robot actuators enabling")
        await self.koskbot.activate()

        #* Move to starting position
        llocs = self.mjRobot.get_limit_center(leftside=True)
        rlocs = self.mjRobot.get_limit_center(leftside=False)

        idx_to_joint_map = self.mjRobot.qpos_idx_to_jointname()
        self.planner.set_idx_joint_map(idx_to_joint_map)


        fullq = self.planner.arms_tofullq(llocs, rlocs)
        self.mjRobot.set_qpos(fullq)

        self.planner.set_curangles(self.mjRobot.data.qpos)
        self.planner.set_nextangles(fullq)
        planned_angles, _ , time_grid = self.planner.get_waypoints()

        await self.koskbot.send_to_kos(planned_angles, time_grid, idx_to_joint_map)

        self.logger.info("Starting positions set")

        await asyncio.sleep(1)


    async def shutdown(self):
        self.logger.info("Disabling actuators...")
        await self.koskbot.disable()

    async def get_vrcont(self, message):
        try:
            data = json.loads(message)

            if 'controller' in data and data['controller'] in ['left', 'right']:
                controller = data['controller']
                
                # Update the appropriate controller attributes based on which controller sent data
                if controller == 'left':
                    async with self.left_lock:
                        if 'buttons' in data:
                            old_btn_state = self.left_btn
                            self.left_btn = data['buttons'][0]['pressed']
                            # If button state changed, set the event
                            if old_btn_state != self.left_btn:
                                self.left_btn_event.set()
                    # self.logger.debug(f"Updated left controller state, buttons={self.left_btn}")
                
                elif controller == 'right':
                    async with self.right_lock:
                        if 'buttons' in data:
                            old_btn_state = self.right_btn
                            self.right_btn = data['buttons'][0]['pressed']
                            # If button state changed, set the event
                            if old_btn_state != self.right_btn:
                                self.right_btn_event.set()
                    # self.logger.debug(f"Updated right controller state, buttons={self.right_btn}")
            else:
                self.logger.warning("No valid controller data found in message")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")


    async def left_control_loop(self, hz: int = 100):
        logger.info("Robot control loop started")
        period = 1.0 / hz
        while self.running:
            try:
                await self.left_btn_event.wait()

                # async with self.goal_lock:
                #     goal = self.goal_state

                # if goal is not None:
                #     # TODO: Replace with real IK + motion planning
                #     state_response = await self.kos.actuator.get_actuators_state(self.actuators.keys())
                #     current_joint_states = [state.position for state in state_response.states]

                #     # Placeholder motion plan
                #     # next_joint_states = inverse_kinematics(goal)
                #     # planned_commands = motion_planning(current_joint_states, next_joint_states)

                #     # await self.kos.actuator.command_actuators(planned_commands)
                #     logger.debug("Control loop: would send commands here")

                await asyncio.sleep(period)


            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.1)


    async def right_control_loop(self, hz: int = 100):
        logger.info("Robot control loop started")
        period = 1.0 / hz
        while self.running:
            try:
                await self.right_btn_event.wait()


                ree_pos, ree_ort = self.mjRobot.get_ee_pos(leftside=False)
                new_pos, new_quat = self.planner.apply_pose_delta(ree_pos, ree_ort, [0.3, 0.3, -0.1], [0, 0, 0])

                pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(self.mjRobot.model, self.mjRobot.data, new_pos, new_quat, leftside=False)
                logger.warning(f"Computed angles {pos_arm[-1]}")
                nextqpos = self.planner.arms_tofullq(leftarmq=self.mjRobot.get_limit_center(leftside=True), rightarmq=pos_arm)
                
                self.planner.set_nextangles(nextqpos)
                planned_angles, _ , time_grid = self.planner.get_waypoints()

                await self.koskbot.send_to_kos(planned_angles, time_grid)
                breakpoint()
                await asyncio.sleep(1)


                # async with self.goal_lock:
                #     goal = self.goal_state

                # if goal is not None:
                #     # TODO: Replace with real IK + motion planning
                #     state_response = await self.kos.actuator.get_actuators_state(self.actuators.keys())
                #     current_joint_states = [state.position for state in state_response.states]

                #     # Placeholder motion plan
                #     # next_joint_states = inverse_kinematics(goal)
                #     # planned_commands = motion_planning(current_joint_states, next_joint_states)

                #     # await self.kos.actuator.command_actuators(planned_commands)
                #     logger.debug("Control loop: would send commands here")

                await asyncio.sleep(period)


            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.1)


async def websocket_handler(websocket, session):
    """Handle each client connection with an existing session."""
    try:
        #* Create VR Controller Tasks if they're not running
        left_arm_task = asyncio.create_task(session.left_control_loop())
        right_arm_task = asyncio.create_task(session.right_control_loop())

        try:
            async for message in websocket:
                logger.info("Received WebSocket message")
                await session.get_vrcont(message)
                await websocket.send(json.dumps({"status": "success"}))
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Cancel the control tasks
            right_arm_task.cancel()
            left_arm_task.cancel()
            
            try:
                await right_arm_task
                await left_arm_task
            except asyncio.CancelledError:
                pass
    except Exception as e:
        logger.error(f"Error in websocket handler: {e}")
        await websocket.send(json.dumps({"status": "error", "message": str(e)}))

async def main():
    """Start KOS and then WebSocket server."""
    try:
        # Initialize KOS connection once
        async with KOS(ip="10.33.12.161", port=50051) as kos:
            # Create and initialize session
            session = TeleopSession(kos, sim=True)
            try:
                logger.info("Initializing robot...")
                await session.initialize()
                logger.info("Robot initialization complete")
                
                # Start WebSocket server with the initialized session
                async def handle_connection(websocket):
                    await websocket_handler(websocket, session)
                
                server = await websockets.serve(handle_connection, "localhost", 8586)
                logger.info("WebSocket server started at ws://localhost:8586")
                
                # Keep server running until manually terminated
                try:
                    await server.wait_closed()
                except asyncio.CancelledError:
                    logger.info("Server shutdown requested")
                finally:
                    # Clean shutdown
                    session.running = False
                    await session.shutdown()
                    
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
    except Exception as e:
        logger.error(f"Error connecting to KOS: {e}")

if __name__ == "__main__":
    asyncio.run(main())
