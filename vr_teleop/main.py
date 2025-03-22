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
logger.setLevel(logging.DEBUG)


class TeleopSession:
    def __init__(self, kos_instance: KOS, sim: bool, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
        self.mjRobot = MJ_KBot(urdf_path)
        self.planner = Robot_Planner(self.mjRobot)
        self.koskbot = KOS_KBot(kos_instance, self.planner,sim=sim)

        # Add an event for button state changes
        self.left_btn_event = asyncio.Event() 
        self.left_btn = False
        self.left_last_pos = None
        self.left_last_rot = None
        self.left_lock = asyncio.Lock()
        self.left_goal = None

        self.right_btn_event = asyncio.Event()
        self.right_btn = False
        self.right_last_pos = None
        self.right_last_rot = None
        self.right_lock = asyncio.Lock()
        self.right_goal = None
        self.right_movement_in_progress = False

        self.running = True

        self.logger = setup_logger(__name__)

        self.left_control_task = None
        self.right_control_task = None

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
                            # Initialize old_btn_state with current state, handling None case
                            old_btn_state = self.left_btn if self.left_btn is not None else False
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
                            
                            # Only set the event if:
                            # 1. Button state changed, AND
                            # 2. We're not in the middle of a movement, OR we're transitioning to button released
                            if old_btn_state != self.right_btn and (not self.right_movement_in_progress or not self.right_btn):
                                logger.info(f"Right button state changed to {self.right_btn}")
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
                # Wait for button event
                await self.right_btn_event.wait()
                
                # Immediately clear the event
                self.right_btn_event.clear()
                
                # Check if button is actually pressed before proceeding
                if not self.right_btn:
                    logger.debug("Right button released, skipping movement")
                    await asyncio.sleep(period)
                    continue
                    
                logger.info("Processing right button press")
                
                # Get current position
                ree_pos, ree_ort = self.mjRobot.get_ee_pos(leftside=False)
                new_pos, new_quat = self.planner.apply_pose_delta(ree_pos, ree_ort, [0.3, 0.3, -0.1], [0, 0, 0])
                logger.warning(f"old pos: {ree_pos} and new post {new_pos}")

                # Apply inverse kinematics
                pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(self.mjRobot.model, self.mjRobot.data, new_pos, new_quat, leftside=False)
                nextqpos = self.planner.arms_tofullq(leftarmq=self.mjRobot.get_limit_center(leftside=True), rightarmq=pos_arm)
                
                self.planner.set_nextangles(nextqpos)
                planned_angles, _ , time_grid = self.planner.get_waypoints()

                # Add a debounce to prevent too-frequent movements
                # Set a flag to indicate movement in progress
                async with self.right_lock:
                    self.right_movement_in_progress = True
                    
                try:
                    await self.koskbot.send_to_kos(planned_angles, time_grid, self.planner.idx_to_joint_map)
                    # Wait to ensure movement completes before accepting more commands
                    await asyncio.sleep(1)
                finally:
                    # Clear the in-progress flag
                    async with self.right_lock:
                        self.right_movement_in_progress = False
                
                # ... existing code ...
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.1)


async def websocket_handler(websocket, session):
    """Handle each client connection with an existing session."""
    try:
        # Get client info and log with debug level
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.debug(f"New connection from {client_info}")
            
        # Only create control tasks if they're not already running
        if session.left_control_task is None or session.left_control_task.done():
            session.left_control_task = asyncio.create_task(session.left_control_loop())
            logger.info("Started left control loop task")
            
        if session.right_control_task is None or session.right_control_task.done():
            session.right_control_task = asyncio.create_task(session.right_control_loop())
            logger.info("Started right control loop task")

        try:
            # Message loop
            async for message in websocket:
                logger.debug(f"Received WebSocket message")
                await session.get_vrcont(message)
                await websocket.send(json.dumps({"status": "success"}))
                
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Client {client_info} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Don't cancel the control tasks when a client disconnects
            # They should keep running to handle future connections
            pass
    except Exception as e:
        logger.error(f"Error in websocket handler: {e}")
        try:
            await websocket.send(json.dumps({"status": "error", "message": str(e)}))
        except:
            pass  # Client might already be disconnected

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
                    
                    # Cancel any running tasks
                    if session.left_control_task:
                        session.left_control_task.cancel()
                    if session.right_control_task:
                        session.right_control_task.cancel()
                    
                    await session.shutdown()
                    
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
    except Exception as e:
        logger.error(f"Error connecting to KOS: {e}")

if __name__ == "__main__":
    asyncio.run(main())
