import asyncio
import json
import websockets
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
    def __init__(self):
        self.right_last_ee_pos = np.array([0.39237205, 0.16629315, 0.90654296])
        self.left_last_ee_pos = np.array([-0.43841719, 0.14227997, 0.99074782])

        self.right_target_ee_pos = self.right_last_ee_pos.copy()
        self.left_target_ee_pos = self.left_last_ee_pos.copy()

        self.squeeze_pressed_prev = False

    def step(self, cur_ee_pos, buttons=None):
        # Determine if squeeze button is pressed
        squeeze_pressed = False
        if buttons is not None:
            for button in buttons:
                if isinstance(button, dict) and button.get('name') == 'squeeze':
                    squeeze_pressed = button.get('pressed', False)
                    break
        
        if squeeze_pressed:
            if not self.squeeze_pressed_prev:
                self.right_last_ee_pos = cur_ee_pos.copy()
                logger.debug(f"Squeeze pressed, storing reference position: {self.right_last_ee_pos}")
            
            if self.right_last_ee_pos is not None:
                delta = cur_ee_pos - self.right_last_ee_pos
                logger.warning(f"Controller: {self.right_last_ee_pos}")
                
                #* Temp hack for sensitivity
                delta[0] = 3*delta[0]
                delta[1] = -3*delta[1]
                delta[2] = -5*delta[2]

                self.right_target_ee_pos = self.right_target_ee_pos + delta
                logger.warning(f"New target: {self.right_target_ee_pos}")
                self.right_last_ee_pos = cur_ee_pos.copy()
        else:
            if self.squeeze_pressed_prev:
                logger.debug("Squeeze released, freezing target position")
                
        self.squeeze_pressed_prev = squeeze_pressed

        return None


async def controller_task(controller, rate=100):
    """CONTINUOUSLY run the controller at the specified rate (Hz)."""
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
                controller.step(r_position, r_buttons)
                
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

                
                if 'controller' in data and 'position' in data and data['controller'] in ['left', 'right']:
                    controller = data['controller']
                    
                    position = np.array(data['position'], dtype=np.float64)
                    controller_states[controller]['position'] = position
                    
                    for field in ['rotation', 'buttons', 'axes']:
                        if field in data:
                            controller_states[controller][field] = data[field]
                    
                    controller_states['updated'] = True
                    
                    await websocket.send(json.dumps({"status": "success"}))

                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")


async def main():
    try:
        controller = Controller()
       
        server = await websockets.serve(websocket_handler, "localhost", 8586)
        logger.info("WebSocket server started at ws://localhost:8586")
        
        control_task = asyncio.create_task(controller_task(controller))
        
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
                    
if __name__ == "__main__":
    asyncio.run(main())



