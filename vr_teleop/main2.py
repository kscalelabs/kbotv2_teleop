import asyncio
import json
import websockets
from pykos import KOS
from dataclasses import dataclass
from .utils.logging import setup_logger
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

from vr_teleop.utils.ik import *
from vr_teleop.mjRobot import MJ_KBot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.kosRobot import KOS_KBot


logger = setup_logger(__name__, logging.WARNING)

# Shared state for controller positions
controller_states = {
    'left': {'position': np.array([0.01, 0.01, 0.01]), 'rotation': None, 'buttons': None, 'axes': None},
    'right': {'position': np.array([0.01, 0.01, 0.01]), 'rotation': None, 'buttons': None, 'axes': None},
    'updated': False
}

class Controller:
    def __init__(self, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
        self.mjRobot = MJ_KBot(urdf_path)
        self.last_ee_pos = np.array([0.01, 0.01, 0.01])

        # Get initial robot position
        rlocs = self.mjRobot.get_limit_center(leftside=False)
        fullq = self.mjRobot.convert_armqpos_to_fullqpos(rlocs, leftside=False)
        self.mjRobot.set_qpos(fullq)
        
        # For relative motion control
        self.target_ee_pos = self.last_ee_pos.copy()
        self.last_controller_pos = None
        self.squeeze_pressed_prev = False

    def step(self, cur_ee_pos, buttons=None):
        # Determine if squeeze button is pressed
        squeeze_pressed = False
        if buttons is not None:
            # Find the squeeze button in the buttons list
            for button in buttons:
                if isinstance(button, dict) and button.get('name') == 'squeeze':
                    squeeze_pressed = button.get('pressed', False)
                    break
        
        # Handle squeeze button logic
        if squeeze_pressed:
            # If squeeze just got pressed, store current controller position
            if not self.squeeze_pressed_prev:
                self.last_controller_pos = cur_ee_pos.copy()
                logger.debug(f"Squeeze pressed, storing reference position: {self.last_controller_pos}")
            
            # Calculate delta from last_controller_pos to current controller position
            if self.last_controller_pos is not None:
                delta = cur_ee_pos - self.last_controller_pos
                # Apply delta to target end effector position
                self.target_ee_pos = self.target_ee_pos + delta
                logger.warning(f"Delta: {delta}, New target: {self.target_ee_pos}")
                # Update last_controller_pos for next calculation
                self.last_controller_pos = cur_ee_pos.copy()
        else:
            # When not squeezed, don't update the target position
            if self.squeeze_pressed_prev:
                logger.debug("Squeeze released, freezing target position")
                
        # Update squeeze state for next iteration
        self.squeeze_pressed_prev = squeeze_pressed
        
        # Calculate IK based on the target position (not the raw controller position)
        qpos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
                self.mjRobot.model, 
                self.mjRobot.data, 
                self.target_ee_pos, 
                target_ort=None, 
                leftside=False,
            )

        logger.debug(f"IK: {qpos_arm}, {error_norm_pos}, {error_norm_rot}")
        
        # Store the last end effector position
        self.last_ee_pos = self.target_ee_pos.copy()

        qpos_full = self.mjRobot.convert_armqpos_to_fullqpos(qpos_arm, leftside=False)
        self.mjRobot.set_qpos(qpos_full)

        return qpos_arm

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    flip_sign: bool
    joint_name: str


ACTUATORS = {
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
}


async def command_kbot(qnew_pos, kos):
    # Create a direct mapping for right arm joints
    right_arm_mapping = {
        21: 5*np.degrees(qnew_pos[0]),  # right_shoulder_pitch
        22: 5*np.degrees(qnew_pos[1]),  # right_shoulder_roll
        23: 5*np.degrees(qnew_pos[2]),  # right_shoulder_yaw
        24: 5*np.degrees(qnew_pos[3]),  # right_elbow
        25: 5*np.degrees(qnew_pos[4]),  # right_wrist
    }

    command = []
    for actuator_id in ACTUATORS:
        if actuator_id in right_arm_mapping:
            command.append({
                "actuator_id": actuator_id,
                "position": right_arm_mapping[actuator_id],
            })
    
    command_tasks = []
    
    logger.debug(f"Commanding {command}")
    command_tasks.append(kos.actuator.command_actuators(command))
    await asyncio.gather(*command_tasks)


async def websocket_handler(websocket):
    """Handle incoming WebSocket messages and update the shared state."""
    global controller_states
    
    try:
        async for message in websocket:
            try:
                if message == "ping":
                    await websocket.send(json.dumps({"status": "success"}))
                    continue
                data = json.loads(message)

                
                if 'controller' in data and 'position' in data:
                    controller = data['controller']
                    
                    if controller not in ['left', 'right']:
                        continue
                    
                    # Convert position to numpy array
                    position = np.array(data['position'], dtype=np.float64)
                    
                    # Update the controller state
                    controller_states[controller]['position'] = position
                    
                    # Also update rotation, buttons, axes if available
                    for field in ['rotation', 'buttons', 'axes']:
                        if field in data:
                            controller_states[controller][field] = data[field]
                    
                    controller_states['updated'] = True
                    
                    # Send response back
                    await websocket.send(json.dumps({"status": "success"}))

                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")


async def controller_task(controller, kos, rate=100):
    """Run the controller at the specified rate (Hz)."""
    global controller_states
    
    period = 1.0 / rate  # seconds between iterations
    
    logger.info(f"Starting controller task at {rate}Hz (every {period*1000:.1f}ms)")
    
    # For measuring average Hz
    iteration_count = 0
    last_hz_print_time = time.time()
    
    while True:
        start_time = time.time()
        
        try:
            # Get the current right controller position and buttons
            position = controller_states['right']['position']
            buttons = controller_states['right']['buttons']
            
            if position is not None:
                # Run the controller step and get new joint positions
                # Pass both position and buttons to the step function
                qnew_pos = controller.step(position, buttons)
                
                # Send commands to the robot
                await command_kbot(qnew_pos, kos)
                
            # Reset the updated flag
            controller_states['updated'] = False
            
        except Exception as e:
            logger.error(f"Error in controller task: {e}")
        
        # Calculate sleep time to maintain desired frequency
        elapsed = time.time() - start_time
        sleep_time = max(0, period - elapsed)
        
        # Increment iteration counter
        iteration_count += 1
        
        # Check if it's time to print the average Hz (every 1 second)
        current_time = time.time()
        if current_time - last_hz_print_time >= 1.0:
            elapsed_time = current_time - last_hz_print_time
            average_hz = iteration_count / elapsed_time
            logger.error(f"Controller average rate: {average_hz:.2f} Hz over the last {elapsed_time:.2f} seconds ({iteration_count} iterations)")
            # Reset counters
            iteration_count = 0
            last_hz_print_time = current_time
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            # Log if we're not keeping up with the desired rate
            logger.debug(f"Controller iteration took {elapsed*1000:.1f}ms, exceeding period of {period*1000:.1f}ms")
            # Yield to other tasks but don't sleep
            await asyncio.sleep(0)


async def main():
    """Start KOS and then run WebSocket server and controller task concurrently."""
    try:
        # Initialize KOS connection once
        async with KOS(ip="10.33.12.161", port=50051) as kos:
            controller = Controller()
            
            try:
                # Configure actuators
                enable_commands = []
                for cur_act in ACTUATORS.keys():
                    enable_commands.append(
                        kos.actuator.configure_actuator(
                            actuator_id=cur_act,
                            kp=ACTUATORS[cur_act].kp,
                            kd=ACTUATORS[cur_act].kd,
                            torque_enabled=True
                        )
                    )
                logger.warning(f"Enabling {enable_commands}")
                await asyncio.gather(*enable_commands)
                await asyncio.sleep(1)
                
                # Start the WebSocket server
                server = await websockets.serve(websocket_handler, "localhost", 8586)
                logger.info("WebSocket server started at ws://localhost:8586")
                
                # Start the controller task
                control_task = asyncio.create_task(controller_task(controller, kos))
                
                # Wait for server to close or tasks to complete
                try:
                    await server.wait_closed()
                except asyncio.CancelledError:
                    logger.info("Server shutdown requested")
                finally:
                    # Clean up tasks
                    if not control_task.done():
                        control_task.cancel()
                        try:
                            await control_task
                        except asyncio.CancelledError:
                            logger.info("Controller task cancelled")
                
            except Exception as e:
                logger.error(f"Error during initialization: {e}")
    except Exception as e:
        logger.error(f"Error connecting to KOS: {e}")

if __name__ == "__main__":
    asyncio.run(main())





#* Initialize KOS. 
    ##* Initialize websocket client once KOS has been started. 


#! Event listener for websocket messages
#* Get data, get delta (compare with previous data), get IK, send joint commands. 




#* Websocket gets data -> Puts in queue. (later) Two queues, one for each controller?
#* Task wakes up when there is something in queue. 











