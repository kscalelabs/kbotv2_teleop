import asyncio
import websockets
import json
import csv
import signal
import sys
import time
from datetime import datetime
from vr_teleop.utils.logging import setup_logger
from scipy.spatial.transform import Rotation as R

# Set up logger
logger = setup_logger(__name__)

# Store the latest state of both controllers
controller_states = {
    'left': {'position': None, 'rotation': None, 'buttons': None, 'axes': None},
    'right': {'position': None, 'rotation': None, 'buttons': None, 'axes': None}
}

# Simple list to store all data points
data_log = []

# Flag for clean shutdown
running = True
# Connection state
websocket_client = None
reconnect_task = None
process_task = None
last_success_time = time.time()
message_queue = []
reconnect_interval = 0.02  # 20ms

# Add this constant near the top of the file after the imports
REST_POSE_QUAT = [0.402, 0.085, -0.123, 0.903]  # [w, x, y, z]

def signal_handler(sig, frame):
    """Handle CTRL+C shutdown"""
    global running
    logger.warning("Shutting down and saving data...")
    running = False
    
    # Save the data to CSV
    filename = f"./vr_teleop/dev/vr_controller_data.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['timestamp', 'controller', 'pos_x', 'pos_y', 'pos_z', 
                         'rot_x', 'rot_y', 'rot_z', 'button_pressed', 'axes_x', 'axes_y'])
        # Write data
        for entry in data_log:
            writer.writerow(entry)
    
    logger.warning(f"Data saved to {filename}")
    
    # Cancel running tasks
    if reconnect_task:
        reconnect_task.cancel()
    if process_task:
        process_task.cancel()
    
    sys.exit(0)

async def process_message(message):
    """Process received message and log controller data"""
    try:
        data = json.loads(message)

        # logger.info(f"Received message: {data}")
        
        if 'controller' in data:
            controller = data['controller']
            
            # Only process messages for the right controller
            if controller != 'right':
                return json.dumps({"status": "success"})
            
            # Update controller state
            for field in ['position', 'rotation', 'buttons', 'axes']:
                if field in data:
                    controller_states[controller][field] = data[field]
            
            # Log this data point
            timestamp = datetime.now().timestamp()
            position = controller_states[controller]['position']
            rotation = controller_states[controller]['rotation']
            
            button_pressed = False
            if controller_states[controller]['buttons']:
                button_pressed = controller_states[controller]['buttons'][0]['pressed']
            
            # Add axes data
            axes = controller_states[controller]['axes']
            
            # Convert quaternion to Euler angles
            # Assumption: The quaternion is given in the order [w, x, y, z]
            if len(rotation) == 4:
                # Reorder the quaternion from [w, x, y, z] to [x, y, z, w] for scipy
                reordered_quat = [rotation[1], rotation[2], rotation[3], rotation[0]]
                
                # Create rotation object for the current rotation
                r_current = R.from_quat(reordered_quat)
                
                # Create rotation object for the rest pose (reorder it too)
                rest_reordered = [REST_POSE_QUAT[1], REST_POSE_QUAT[2], REST_POSE_QUAT[3], REST_POSE_QUAT[0]]
                r_rest = R.from_quat(rest_reordered)
                
                # Try the opposite multiplication order
                r_final = r_current * r_rest.inv()
                
                # Get the Euler angles from the transformed rotation
                euler_angles = r_final.as_euler('xyz', degrees=True)
            else:
                euler_angles = [0, 0, 0]
            
            # Add to our data log
            data_log.append([
                timestamp,
                controller,
                position[0], position[1], position[2],
                euler_angles[0], euler_angles[1], euler_angles[2],
                button_pressed,
                axes[0], axes[1]
            ])
            
            # Basic logging
            # logger.warning(f"{controller} pos: {position}")
            logger.warning(f"rot: {euler_angles}")
            logger.warning(f"position: {position}")
            logger.warning(f"quat: {rotation}")
            if controller_states[controller]['buttons']:
                logger.warning(f"button: {button_pressed}")
            
            return json.dumps({"status": "success"})
        
    except Exception as e:
        # logger.error(f"Error processing message: {e}") #! Gets triggered a lot
        return json.dumps({"status": "error", "message": str(e)})

async def reconnect_manager():
    """Manages websocket reconnection"""
    global websocket_client, last_success_time
    
    connection_attempts = 0
    max_connection_attempts = 1000  # Safety limit
    
    while running:
        # Check if we need to reconnect
        need_reconnect = (
            websocket_client is None or 
            websocket_client.closed or
            (time.time() - last_success_time > 5)  # Force reconnect if no success for 5 seconds
        )
        
        if need_reconnect:
            if time.time() - last_success_time > 5:
                logger.warning("No successful communication for 5 seconds, forcing reconnection")
                
            if websocket_client and not websocket_client.closed:
                try:
                    await websocket_client.close()
                except Exception:
                    pass
                    
            connection_attempts += 1
            if connection_attempts > max_connection_attempts:
                logger.error(f"Too many connection attempts ({connection_attempts}). Giving up.")
                running = False
                break
                
            try:
                websocket_client = await websockets.connect("ws://localhost:8586")
                connection_attempts = 0  # Reset counter on success
                logger.info("Successfully (re)connected to server")
            except Exception as e:
                # Only log every 10th attempt to reduce spam
                if connection_attempts % 10 == 0:
                    logger.warning(f"Connection attempt {connection_attempts} failed: {e}")
                await asyncio.sleep(reconnect_interval)
        
        await asyncio.sleep(reconnect_interval)

async def process_queue():
    """Process the message queue by sending messages to the server"""
    global message_queue, last_success_time, websocket_client
    
    while running:
        # Only process if we have messages and a valid connection
        if message_queue and websocket_client and not websocket_client.closed:
            try:
                # Get message from queue
                message = message_queue.pop(0)
                
                # Send to server
                await websocket_client.send(message)
                
                # Wait for response with timeout
                response = await asyncio.wait_for(websocket_client.recv(), timeout=0.5)
                
                # Update success time
                last_success_time = time.time()
                
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError, 
                    ConnectionResetError, websockets.exceptions.WebSocketException) as e:
                # Connection issues - quietly handle
                continue
            except Exception as e:
                logger.error(f"Unexpected error processing queue: {e}")
        
        await asyncio.sleep(reconnect_interval)

async def handle_connection(websocket):
    """Handle incoming WebSocket connections."""
    global message_queue
    
    try:
        async for message in websocket:
            # Process the message
            response = await process_message(message)
            
            # Instead of sending directly, add to queue for reliable delivery
            message_queue.append(response)
            
    # except websockets.exceptions.ConnectionClosed:
    #     logger.info("Client disconnected")
    except Exception as e:
        pass
        # logger.error(f"Error handling connection: {e}") #! Gets triggered a lot

async def main():
    """Start the WebSocket server."""
    global reconnect_task, process_task
    
    # Set up signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start reconnection manager
    reconnect_task = asyncio.create_task(reconnect_manager())
    
    # Start queue processor
    process_task = asyncio.create_task(process_queue())
    
    # Start the server
    server = await websockets.serve(
        handle_connection,
        "localhost",
        8586
    )
    logger.info("WebSocket server started on ws://localhost:8586")
    logger.info("Recording VR controller data. Press CTRL+C to save and exit.")
    
    # Keep server running until CTRL+C
    while running:
        await asyncio.sleep(1)
    
    # Clean shutdown
    reconnect_task.cancel()
    process_task.cancel()
    
    server.close()
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())