import asyncio
import json
import websockets
from pykos import KOS
from ..utils.logging import setup_logger
from scipy.spatial.transform import Rotation as R

from vr_teleop.utils.ik import *
from vr_teleop.helpers.mjRobot import MJ_KBot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.helpers.kosRobot import KOS_KBot


logger = setup_logger(__name__, logging.INFO)


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
        
        # Controller position tracking
        self.right_controller_pos = None
        self.right_controller_rot = None
        self.right_rot_delta = [0, 0, 0]  # Initialize with zeros
        self.right_initial_pos = None     # Initialize position tracking variables
        self.right_initial_rot = None
        
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
                        # Track controller position and orientation
                        if 'position' in data:
                            # Convert position data to floats
                            self.right_controller_pos = [float(x) for x in data['position']]
                        if 'rotation' in data:
                            # Convert rotation data to floats
                            self.right_controller_rot = [float(x) for x in data['rotation']]
                            
                            # Reorder quaternion and convert to Euler angles
                            reordered_quat = [
                                self.right_controller_rot[1], 
                                self.right_controller_rot[2], 
                                self.right_controller_rot[3], 
                                self.right_controller_rot[0]
                            ]
                            r = R.from_quat(reordered_quat)
                            euler_angles = r.as_euler('xyz', degrees=False)  # Change to radians
                            
                            # Calculate orientation delta if initial rotation is recorded
                            if self.right_initial_rot:
                                initial_r = R.from_quat([
                                    self.right_initial_rot[1], 
                                    self.right_initial_rot[2], 
                                    self.right_initial_rot[3], 
                                    self.right_initial_rot[0]
                                ])
                                initial_euler = initial_r.as_euler('xyz', degrees=False)  # Change to radians
                                rot_delta = [e - i for e, i in zip(euler_angles, initial_euler)]
                                self.right_rot_delta = rot_delta  # Store the rotation delta as class variable
                                logger.info(f"Orientation delta (rad): {rot_delta}")

                        if 'buttons' in data:
                            old_btn_state = self.right_btn
                            self.right_btn = data['buttons'][0]['pressed']
                            
                            # When button is first pressed, record current position as reference
                            if not old_btn_state and self.right_btn:
                                self.right_initial_pos = self.right_controller_pos.copy()
                                self.right_initial_rot = self.right_controller_rot.copy() if self.right_controller_rot else None
                                logger.info(f"Recorded initial right controller position: {self.right_initial_pos}")
                            
                            # Set event for any button state change
                            if old_btn_state != self.right_btn:
                                logger.info(f"Right button state changed to {self.right_btn}")
                                self.right_btn_event.set()
                            # Set event for controller movement while button is pressed
                            elif self.right_btn and self.right_controller_pos is not None and self.right_initial_pos is not None:
                                # Trigger control loop on controller movement
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
        
        # Add movement throttling variables
        last_movement_time = 0
        movement_throttle = 0.25  # Minimum seconds between movements
        self.pending_movement = False  # Make this an instance variable to share between methods
        movement_queue = asyncio.Queue(maxsize=1)
        
        # Start a separate task for executing movements
        movement_task = asyncio.create_task(self._execute_movement_queue(movement_queue))
        
        while self.running:
            try:
                # Wait for button event
                await self.right_btn_event.wait()
                
                # Immediately clear the event
                self.right_btn_event.clear()
                
                # Get lock to check state
                async with self.right_lock:
                    # If button is not pressed, skip movement
                    if not self.right_btn:
                        logger.debug("Right button released, skipping movement")
                        # Keep initial position but mark that we're not moving
                        self.pending_movement = False
                        continue
                    
                    # Skip if we don't have both current and initial positions
                    if self.right_controller_pos is None or not hasattr(self, 'right_initial_pos') or self.right_initial_pos is None:
                        continue
                    
                    # Calculate position delta relative to the INITIAL position when button was pressed
                    pos_delta = [
                        float(self.right_controller_pos[0]) - float(self.right_initial_pos[0]),
                        float(self.right_controller_pos[1]) - float(self.right_initial_pos[1]),
                        float(self.right_controller_pos[2]) - float(self.right_initial_pos[2])
                    ]
                    
                    # Flip the axes: -Z (VR) to +Y (Robot)
                    remapped_delta = [
                        pos_delta[0],  # X remains X
                        -pos_delta[2], # -Z becomes +Y
                        pos_delta[1]   # Y becomes Z
                    ]
                    
                    # Scale movement for better control
                    scaling_factor = 0.05
                    remapped_delta = [d * scaling_factor for d in remapped_delta]
                
                # Only calculate a new movement if not throttled
                current_time = asyncio.get_event_loop().time()
                delta_mag = sum(abs(d) for d in remapped_delta)
                
                # Add debugging to see if we're detecting movement at all
                logger.warning(f"Movement delta: {remapped_delta}, magnitude: {delta_mag}")
                
                if delta_mag > 0.005:  # Lower threshold to catch smaller movements
                    # Check if we should queue a new movement
                    if not self.pending_movement and (current_time - last_movement_time) >= movement_throttle:
                        # We can send a movement now
                        try:
                            # Get current position once for this movement
                            ree_pos, ree_ort = self.mjRobot.get_ee_pos(leftside=False)
                            
                            # Put the movement parameters in the queue for processing
                            movement_params = {
                                "ree_pos": ree_pos,
                                "ree_ort": ree_ort,
                                "pos_delta": remapped_delta.copy(),
                                "rot_delta": self.right_rot_delta if self.right_rot_delta is not None else [0, 0, 0]
                            }
                            
                            # Try to put in queue without blocking
                            if movement_queue.empty():
                                await movement_queue.put(movement_params)
                                last_movement_time = current_time
                                self.pending_movement = True
                                logger.warning(f"Queued movement with delta: {remapped_delta}")
                            else:
                                logger.warning("Queue is full, skipping movement")
                        except asyncio.QueueFull:
                            # If queue is full, just continue and try again later
                            logger.warning("Queue is full (exception), skipping movement")
                            pass
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(0.01)  # Reduced error recovery sleep
                
        # Cancel the movement task when exiting
        if movement_task and not movement_task.done():
            movement_task.cancel()
            
    async def _execute_movement_queue(self, queue):
        """Process movements from the queue to separate movement execution from event handling"""
        while self.running:
            try:
                # Get the next movement from the queue
                logger.warning("Waiting for movement in queue...")
                movement = await queue.get()
                logger.warning(f"Got movement from queue with delta: {movement['pos_delta']}")
                
                ree_pos = movement["ree_pos"]
                ree_ort = movement["ree_ort"]
                pos_delta = movement["pos_delta"]
                rot_delta = movement["rot_delta"]
                
                # Apply the delta to current robot position
                new_pos, new_quat = self.planner.apply_pose_delta(ree_pos, ree_ort, pos_delta, rot_delta)
                logger.warning(f"Current position: {ree_pos}, New position: {new_pos}")
                logger.warning(f"Using rotation delta: {rot_delta}")
                
                # Skip if delta would move outside reasonable bounds
                if any(abs(new_p - old_p) > 0.3 for new_p, old_p in zip(new_pos, ree_pos)):
                    logger.warning("Skipping movement - delta too large")
                    queue.task_done()
                    self.pending_movement = False
                    continue
                
                # Measure IK solve time
                ik_start_time = asyncio.get_event_loop().time()
                # Apply inverse kinematics with improved parameters
                pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
                    self.mjRobot.model, 
                    self.mjRobot.data, 
                    new_pos, 
                    new_quat, 
                    leftside=False,
                    max_iter=200,      # Reduced from 400 for speed
                    tolerance=0.1      # Increased tolerance for faster convergence
                )
                ik_end_time = asyncio.get_event_loop().time()
                ik_time = ik_end_time - ik_start_time
                
                logger.warning(f"IK solve complete with error: pos={error_norm_pos}, rot={error_norm_rot}")
                
                # Skip if IK error is too large - helps prevent unsafe movements
                if error_norm_pos > 0.3:
                    logger.warning(f"Skipping movement due to high IK error: {error_norm_pos}")
                    queue.task_done()
                    self.pending_movement = False
                    continue
                
                # Measure planning time
                plan_start_time = asyncio.get_event_loop().time()
                nextqpos = self.planner.arms_tofullq(leftarmq=self.mjRobot.get_limit_center(leftside=True), rightarmq=pos_arm)
                self.planner.set_nextangles(nextqpos)
                
                # Use fewer waypoints for faster motion
                planned_angles, _, time_grid = self.planner.get_waypoints(num_waypoints=3)
                plan_end_time = asyncio.get_event_loop().time()
                plan_time = plan_end_time - plan_start_time
                
                # Log timing information
                logger.warning(f"Planning complete - IK time: {ik_time:.4f}s, Planning time: {plan_time:.4f}s")
                
                # Execute movement
                try:
                    # Measure KOS send time
                    kos_start_time = asyncio.get_event_loop().time()
                    logger.warning("Sending movement to KOS...")
                    await self.koskbot.send_to_kos(planned_angles, time_grid, self.planner.idx_to_joint_map)
                    kos_end_time = asyncio.get_event_loop().time()
                    kos_time = kos_end_time - kos_start_time
                    
                    # Log timing information
                    logger.warning(f"KOS send complete: {kos_time:.4f}s, Total execution time: {kos_time + ik_time + plan_time:.4f}s")
                    
                except Exception as e:
                    logger.error(f"Error sending movement to KOS: {e}")
                
                # Mark task as done
                queue.task_done()
                
                # Signal that we can accept new movements
                self.pending_movement = False
                logger.warning("Movement execution complete, ready for next movement")
                
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                logger.warning("Movement execution task cancelled")
                break
            except Exception as e:
                logger.error(f"Movement execution error: {e}")
                # Signal that we can accept new movements even after an error
                self.pending_movement = False


async def websocket_handler(websocket, session):
    """Handle each client connection with an existing session."""
    try:
        # Get client info and log with debug level
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        # logger.warning(f"New connection from {client_info}")
            
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
