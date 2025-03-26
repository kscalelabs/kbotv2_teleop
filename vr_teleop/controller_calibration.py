import asyncio
import websockets
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import sys
import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Global variables
controller_states = {
    'right': {'position': None, 'rotation': None, 'buttons': None, 'axes': None}
}
running = True
calibration_data = []
websocket_client = None

def signal_handler(sig, frame):
    """Handle CTRL+C shutdown"""
    global running
    print("Shutting down...")
    running = False
    
    if websocket_client and not websocket_client.closed:
        asyncio.run(websocket_client.close())
    
    if len(calibration_data) > 0:
        analyze_calibration_data()
    
    sys.exit(0)

def plot_orientation_vectors(calibration_data):
    """Plot the orientation vectors from calibration data"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Original coordinate system at origin
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')
    
    # Plot each calibration point's orientation vectors
    for i, data_point in enumerate(calibration_data):
        quat = data_point['quaternion']
        pos = data_point['position']
        
        # Create rotation matrix from quaternion
        rot = R.from_quat(quat)
        
        # Get the three basis vectors (columns of rotation matrix)
        rotation_matrix = rot.as_matrix()
        x_vec = rotation_matrix[:, 0]
        y_vec = rotation_matrix[:, 1]
        z_vec = rotation_matrix[:, 2]
        
        # Plot the rotated coordinate system
        scale = 0.2  # Scale the vectors for visibility
        ax.quiver(pos[0], pos[1], pos[2], scale*x_vec[0], scale*x_vec[1], scale*x_vec[2], color='r', alpha=0.5)
        ax.quiver(pos[0], pos[1], pos[2], scale*y_vec[0], scale*y_vec[1], scale*y_vec[2], color='g', alpha=0.5)
        ax.quiver(pos[0], pos[1], pos[2], scale*z_vec[0], scale*z_vec[1], scale*z_vec[2], color='b', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('VR Controller Orientation Vectors')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    plt.legend()
    plt.savefig('controller_orientation.png')
    plt.show()

def find_optimal_rotation(calibration_data):
    """Find the optimal rotation matrix to align controller with desired coordinate system"""
    # For this function, we assume:
    # - When controller is held normally, we want +Z to point forward from the controller
    # - We want +Y to point up
    # - And +X to point to the right
    
    # Get the average orientation when controller is in "reference" position
    reference_points = [point for point in calibration_data if point['label'] == 'reference']
    
    if not reference_points:
        print("Error: No reference orientation data points collected!")
        return None
    
    # Average the quaternions in the reference position
    # This is a simplification - quaternion averaging is more complex in practice
    avg_quat = np.mean([point['quaternion'] for point in reference_points], axis=0)
    avg_quat = avg_quat / np.linalg.norm(avg_quat)  # Normalize
    
    # Create rotation from averaged quaternion
    current_rotation = R.from_quat(avg_quat)
    
    # The goal is to find rotation R such that R * current = desired
    # For simplicity, we'll print the current basis vectors and suggest manual adjustment
    rotation_matrix = current_rotation.as_matrix()
    
    print("\n==== CONTROLLER ORIENTATION ANALYSIS ====")
    print(f"Current X-axis: {rotation_matrix[:, 0]}")
    print(f"Current Y-axis: {rotation_matrix[:, 1]}")
    print(f"Current Z-axis: {rotation_matrix[:, 2]}")
    
    # Determine which controller axis is most aligned with world "up" (assuming +Y is up)
    up_alignment = [np.abs(np.dot(rotation_matrix[:, i], [0, 1, 0])) for i in range(3)]
    most_up_idx = np.argmax(up_alignment)
    
    # Determine which controller axis is most aligned with world "forward" (assuming +Z is forward)
    forward_alignment = [np.abs(np.dot(rotation_matrix[:, i], [0, 0, 1])) for i in range(3)]
    most_forward_idx = np.argmax(forward_alignment)
    
    # Suggestion for transformation
    print("\n==== SUGGESTED TRANSFORMATION ====")
    
    axes = ['X', 'Y', 'Z']
    print(f"Controller's {axes[most_up_idx]} axis is closest to world UP (Y+)")
    print(f"Controller's {axes[most_forward_idx]} axis is closest to world FORWARD (Z+)")
    
    # Generate code for the correction
    print("\n==== CODE FOR CORRECTION ====")
    print("# Add this to your controller processing code:")
    print("def remap_controller_orientation(rotation):")
    print("    # Convert from [w, x, y, z] to [x, y, z, w] for scipy")
    print("    if all(isinstance(val, str) for val in rotation):")
    print("        rotation = [float(val) for val in rotation]")
    print("    reordered_quat = [rotation[1], rotation[2], rotation[3], rotation[0]]")
    print("    r = R.from_quat(reordered_quat)")
    
    # Determine if we need any custom reordering based on the findings
    permutation = [0, 1, 2]  # Default: no change
    if most_up_idx != 1 or most_forward_idx != 2:
        # We need to reorder some axes
        # This is a simplification - proper reordering is more complex
        print("    # Reorder axes based on calibration")
        if most_up_idx == 0:
            if most_forward_idx == 1:
                print("    # X is up, Y is forward -> Need to remap to Y=up, Z=forward")
                print("    # Rotate -90 degrees around X to get Y up, then -90 around Y to get Z forward")
                print("    correction = R.from_euler('xy', [-90, -90], degrees=True)")
                permutation = [0, 2, 1]
            else:  # most_forward_idx == 2
                print("    # X is up, Z is forward -> Need to remap to Y=up")
                print("    # Rotate 90 degrees around Z")
                print("    correction = R.from_euler('z', 90, degrees=True)")
                permutation = [0, 2, 1]
        elif most_up_idx == 1:
            if most_forward_idx == 0:
                print("    # Y is up, X is forward -> Need to remap X=right, Z=forward")
                print("    # Rotate 90 degrees around Y")
                print("    correction = R.from_euler('y', 90, degrees=True)")
                permutation = [2, 1, 0]
            else:  # most_forward_idx already 2, good configuration
                print("    # Y is up, Z is forward -> Already correct orientation")
                print("    correction = R.from_euler('x', 0, degrees=True)  # Identity rotation")
        elif most_up_idx == 2:
            if most_forward_idx == 0:
                print("    # Z is up, X is forward -> Need to remap Y=up, Z=forward")
                print("    # Rotate -90 degrees around X, then 90 around Y")
                print("    correction = R.from_euler('xy', [-90, 90], degrees=True)")
                permutation = [0, 2, 1]
            else:  # most_forward_idx == 1
                print("    # Z is up, Y is forward -> Need to remap Y=up, Z=forward")
                print("    # Rotate -90 degrees around X")
                print("    correction = R.from_euler('x', -90, degrees=True)")
                permutation = [0, 2, 1]
    else:
        print("    # Controller already in good orientation, just need identity correction")
        print("    correction = R.from_euler('x', 0, degrees=True)  # Identity rotation")
    
    print("    # Apply the correction")
    print("    r_corrected = correction * r")
    print("    # Convert to Euler angles in desired sequence")
    print("    euler_angles = r_corrected.as_euler('xyz', degrees=True)")
    print("    return euler_angles")

def analyze_calibration_data():
    """Analyze the collected calibration data"""
    global calibration_data
    
    if len(calibration_data) < 3:
        print("Not enough calibration data points collected!")
        return
    
    # Plot the orientation vectors
    plot_orientation_vectors(calibration_data)
    
    # Find the optimal rotation
    find_optimal_rotation(calibration_data)

async def process_controller_data(message):
    """Process and display controller data for calibration"""
    global controller_states, calibration_data
    
    try:
        data = json.loads(message)
        
        if 'controller' in data and data['controller'] == 'right':
            # Update controller state
            for field in ['position', 'rotation', 'buttons', 'axes']:
                if field in data:
                    controller_states['right'][field] = data[field]
            
            position = controller_states['right']['position']
            rotation = controller_states['right']['rotation']
            
            # Ensure position and rotation are numeric
            if all(isinstance(val, str) for val in position):
                position = [float(val) for val in position]
            
            if all(isinstance(val, str) for val in rotation):
                rotation = [float(val) for val in rotation]
            
            # Convert from [w, x, y, z] to [x, y, z, w] for scipy
            reordered_quat = [rotation[1], rotation[2], rotation[3], rotation[0]]
            
            # Calculate Euler angles for display
            r = R.from_quat(reordered_quat)
            euler_angles = r.as_euler('xyz', degrees=True)
            
            # Display the current orientation
            sys.stdout.write("\033[H\033[J")  # Clear screen
            print("=== VR CONTROLLER CALIBRATION ===")
            print(f"Position: {position}")
            print(f"Quaternion: {rotation} (format: w,x,y,z)")
            print(f"Euler Angles: {euler_angles} (xyz, degrees)\n")
            
            print("Commands:")
            print("1 - Record reference position (controller held normally)")
            print("2 - Record pointing up position")
            print("3 - Record pointing forward position")
            print("4 - Record pointing right position")
            print("a - Analyze collected data")
            print("q - Quit\n")
            
            print(f"Collected data points: {len(calibration_data)}")
            
            # Return success
            return json.dumps({"status": "success"})
    
    except Exception as e:
        print(f"Error: {e}")
        return json.dumps({"status": "error", "message": str(e)})

async def receive_messages():
    """Receive and process WebSocket messages"""
    global websocket_client
    
    while running:
        if not websocket_client or websocket_client.closed:
            try:
                websocket_client = await websockets.connect("ws://localhost:8586")
                print("Connected to VR controller server")
            except Exception as e:
                print(f"Connection error: {e}")
                await asyncio.sleep(1)
                continue
        
        try:
            message = await websocket_client.recv()
            response = await process_controller_data(message)
            await websocket_client.send(response)
        except Exception as e:
            # print(f"Error: {e}")
            await asyncio.sleep(0.1)

async def user_input():
    """Handle user input for calibration"""
    global calibration_data, controller_states, running
    
    while running:
        command = await asyncio.to_thread(input, "")
        
        if command == 'q':
            running = False
            break
        
        elif command == '1':
            # Record reference position
            if controller_states['right']['rotation']:
                rotation = controller_states['right']['rotation']
                position = controller_states['right']['position']
                
                # Ensure numeric values
                if all(isinstance(val, str) for val in rotation):
                    rotation = [float(val) for val in rotation]
                
                if all(isinstance(val, str) for val in position):
                    position = [float(val) for val in position]
                
                # Convert to scipy format [x, y, z, w]
                quat = [rotation[1], rotation[2], rotation[3], rotation[0]]
                
                calibration_data.append({
                    'label': 'reference',
                    'quaternion': quat,
                    'position': position,
                    'timestamp': time.time()
                })
                print("Recorded reference position")
        
        elif command == '2':
            # Record pointing up
            if controller_states['right']['rotation']:
                rotation = controller_states['right']['rotation']
                position = controller_states['right']['position']
                
                # Ensure numeric values
                if all(isinstance(val, str) for val in rotation):
                    rotation = [float(val) for val in rotation]
                
                if all(isinstance(val, str) for val in position):
                    position = [float(val) for val in position]
                
                # Convert to scipy format
                quat = [rotation[1], rotation[2], rotation[3], rotation[0]]
                
                calibration_data.append({
                    'label': 'up',
                    'quaternion': quat,
                    'position': position,
                    'timestamp': time.time()
                })
                print("Recorded pointing up position")
        
        elif command == '3':
            # Record pointing forward
            if controller_states['right']['rotation']:
                rotation = controller_states['right']['rotation']
                position = controller_states['right']['position']
                
                # Ensure numeric values
                if all(isinstance(val, str) for val in rotation):
                    rotation = [float(val) for val in rotation]
                
                if all(isinstance(val, str) for val in position):
                    position = [float(val) for val in position]
                
                # Convert to scipy format
                quat = [rotation[1], rotation[2], rotation[3], rotation[0]]
                
                calibration_data.append({
                    'label': 'forward',
                    'quaternion': quat,
                    'position': position,
                    'timestamp': time.time()
                })
                print("Recorded pointing forward position")
        
        elif command == '4':
            # Record pointing right
            if controller_states['right']['rotation']:
                rotation = controller_states['right']['rotation']
                position = controller_states['right']['position']
                
                # Ensure numeric values
                if all(isinstance(val, str) for val in rotation):
                    rotation = [float(val) for val in rotation]
                
                if all(isinstance(val, str) for val in position):
                    position = [float(val) for val in position]
                
                # Convert to scipy format
                quat = [rotation[1], rotation[2], rotation[3], rotation[0]]
                
                calibration_data.append({
                    'label': 'right',
                    'quaternion': quat,
                    'position': position,
                    'timestamp': time.time()
                })
                print("Recorded pointing right position")
        
        elif command == 'a':
            if len(calibration_data) > 0:
                analyze_calibration_data()
            else:
                print("No calibration data collected yet!")

async def main():
    """Main function to run the calibration tool"""
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start tasks
    receive_task = asyncio.create_task(receive_messages())
    input_task = asyncio.create_task(user_input())
    
    # Wait for completion
    try:
        await asyncio.gather(receive_task, input_task)
    except asyncio.CancelledError:
        pass
    finally:
        # Clean up
        receive_task.cancel()
        input_task.cancel()

if __name__ == "__main__":
    print("Starting VR Controller Calibration Tool")
    print("Hold the controller in different orientations and record the data")
    print("Press Ctrl+C to exit and analyze the data")
    
    asyncio.run(main()) 