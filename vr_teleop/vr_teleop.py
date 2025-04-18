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
from vr_teleop.helpers.mjRobot import MJ_KBot


logger = setup_logger(__name__, logging.WARNING)

# Shared state for controller positions
controller_states = {
    'left': {'position': np.array([-0.43841719, 0.14227997, 0.99074782]), 'rotation': None, 'buttons': None, 'axes': None},
    'right': {'position': np.array([0.39237205, 0.16629315, 0.90654296]), 'rotation': None, 'buttons': None, 'axes': None},
    'updated': False
}

class Controller:
    def __init__(self, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
        self.mjRobot = MJ_KBot(urdf_path)
        self.right_last_ee_pos = np.array([0.39237205, 0.16629315, 0.90654296])
        self.left_last_ee_pos = np.array([-0.43841719, 0.14227997, 0.99074782])

        self.right_target_ee_pos = self.right_last_ee_pos.copy()
        self.left_target_ee_pos = self.left_last_ee_pos.copy()
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
                self.right_last_ee_pos = cur_ee_pos.copy()
                logger.debug(f"Squeeze pressed, storing reference position: {self.right_last_ee_pos}")
            
            # Calculate delta from right_last_ee_pos to current controller position
            if self.right_last_ee_pos is not None:
                delta = cur_ee_pos - self.right_last_ee_pos

                logger.warning(f"Controller: {self.right_last_ee_pos}")
                
                #* Temp hack for sensitivity
                delta[0] = 3*delta[0]
                delta[1] = -3*delta[1]
                delta[2] = -5*delta[2]

                self.right_target_ee_pos = self.right_target_ee_pos + delta
                logger.warning(f"New target: {self.right_target_ee_pos}")
                # Update right_last_ee_pos for next calculation
                self.right_last_ee_pos = cur_ee_pos.copy()
        else:
            # When not squeezed, don't update the target position
            if self.squeeze_pressed_prev:
                logger.debug("Squeeze released, freezing target position")
                
        # Update squeeze state for next iteration
        self.squeeze_pressed_prev = squeeze_pressed
        
        start_time = time.time()
        # Calculate IK based on the target position (not the raw controller position)
        full_delta_q, error_norm_pos, error_norm_rot = ik_gradient(
                self.mjRobot.model, 
                self.mjRobot.data, 
                self.right_target_ee_pos, 
                target_ort=None, 
                leftside=False,
            )
        ik_time = time.time() - start_time
        # logger.warning(f"IK time: {ik_time*1000:.1f}ms")
        
        # Store the last end effector position
        self.last_right_ee_pos = self.right_target_ee_pos.copy()

        prev_qpos = self.mjRobot.data.qpos.copy()
        new_qpos = prev_qpos + full_delta_q
        
        logger.debug(f"IK: {new_qpos}, {error_norm_pos}, {error_norm_rot}")

        self.mjRobot.set_qpos(new_qpos)


        return new_qpos

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
        21: np.degrees(qnew_pos[0]),  # right_shoulder_pitch
        22: np.degrees(qnew_pos[1]),  # right_shoulder_roll
        23: np.degrees(qnew_pos[2]),  # right_shoulder_yaw
        24: np.degrees(qnew_pos[3]),  # right_elbow
        25: np.degrees(qnew_pos[4]),  # right_wrist
    }

    command = []
    for actuator_id in ACTUATORS:
        if actuator_id in right_arm_mapping:
            command.append({
                "actuator_id": actuator_id,
                "position": right_arm_mapping[actuator_id],
            })
    
    await kos.actuator.command_actuators(command)


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
    
    logger.warning(f"Starting controller task at {rate}Hz (every {period*1000:.1f}ms)")
    
    # For measuring average Hz
    iteration_count = 0
    last_hz_print_time = time.time()
    
    while True:
        start_time = time.time()
        
        try:
            # Get the current right controller position and buttons
            r_position = controller_states['right']['position']
            r_buttons = controller_states['right']['buttons']
            
            if r_position is not None:
                # Run the controller step and get new joint positions
                # Pass both position and buttons to the step function
                qnew_pos = controller.step(r_position, r_buttons)
                
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


async def initialize_controller_and_robot(kos):
    """Initialize the controller and move the robot to starting position."""
    logger.info("Initializing controller and positioning robot")
    
    # Create controller
    controller = Controller()
    
    # Extract both left and right arm joint positions
    left_arm_qpos = controller.mjRobot.get_limit_center(leftside=True)
    right_arm_qpos = controller.mjRobot.get_limit_center(leftside=False)

    fullq = controller.mjRobot.convert_armqpos_to_fullqpos(leftarmq=left_arm_qpos, rightarmq=right_arm_qpos)
    controller.mjRobot.set_qpos(fullq)
    
    # Create commands for both arms
    command = []
    
    # Add left arm actuators (IDs 11-15)
    for i, actuator_id in enumerate(range(11, 16)):
        if actuator_id in ACTUATORS:
            position_value = np.degrees(left_arm_qpos[i])

            command.append({
                "actuator_id": actuator_id,
                "position": position_value,
            })
    
    # Add right arm actuators (IDs 21-25)
    for i, actuator_id in enumerate(range(21, 26)):
        if actuator_id in ACTUATORS:
            position_value = np.degrees(right_arm_qpos[i])
            command.append({
                "actuator_id": actuator_id,
                "position": position_value,
            })
    
    # Send commands directly to KOS
    # logger.warning(f"Commanding {command}")
    await asyncio.gather(*[kos.actuator.command_actuators(command)])
    await asyncio.sleep(1)
    
    logger.warning("Robot initialized to starting position")

    return controller


async def main():
    """Start KOS and then run WebSocket server and controller task concurrently."""
    try:
        # Initialize KOS connection once
        async with KOS(ip="10.33.12.161", port=50051) as kos:
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
                
                # Initialize the controller and position the robot in one step
                controller = await initialize_controller_and_robot(kos)
                
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



