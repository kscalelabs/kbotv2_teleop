import asyncio
import websockets
import json
import csv
import signal
import sys
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
                r = R.from_quat(reordered_quat)
                euler_angles = r.as_euler('xyz', degrees=True)
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
            logger.warning(f"{controller} pos: {position}")
            if controller_states[controller]['buttons']:
                logger.warning(f"{controller} button: {button_pressed}")
            
            return json.dumps({"status": "success"})
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return json.dumps({"status": "error", "message": str(e)})

async def handle_connection(websocket):
    """Handle incoming WebSocket connections."""
    try:
        async for message in websocket:
            # Process the message
            response = await process_message(message)
            
            # Send response back
            await websocket.send(response)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error handling connection: {e}")

async def main():
    """Start the WebSocket server."""
    # Set up signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
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
    
    # Close server when done
    server.close()
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())