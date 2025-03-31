import asyncio
import json
import websockets
from pykos import KOS
from dataclasses import dataclass
from .utils.logging import setup_logger
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import logging
import os
import pybullet as p
import pybullet_data
import xml.etree.ElementTree as ET  # For parsing URDF
import argparse  # For command-line arguments

# Replace MuJoCo imports with PyBullet imports
from vr_teleop.utils.pb_ik import *
from vr_teleop.helpers.pbRobot import PB_KBot


logger = setup_logger(__name__, logging.WARNING)

# Shared state for controller positions
controller_states = {
    'left': {'position': np.array([-0.43841719, 0.14227997, 0.99074782]), 'rotation': None, 'buttons': None, 'axes': None},
    'right': {'position': np.array([0.39237205, 0.16629315, 0.90654296]), 'rotation': None, 'buttons': None, 'axes': None},
    'updated': False
}

def check_joint_limits(robot, urdf_path):
    """
    Compare the joint limits loaded by PyBullet with those defined in the URDF file
    """
    logger.warning("Checking joint limits from URDF vs PyBullet...")
    
    # Parse the URDF file to extract joint limits
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Dictionary to store joint limits from URDF
        urdf_limits = {}
        
        # Extract joint limits from URDF
        for joint in root.findall(".//joint"):
            joint_name = joint.get('name')
            limit_elem = joint.find('limit')
            
            if limit_elem is not None:
                lower = float(limit_elem.get('lower', '0'))
                upper = float(limit_elem.get('upper', '0'))
                urdf_limits[joint_name] = (lower, upper)
        
        # Compare with PyBullet's joint limits
        for i in range(p.getNumJoints(robot.robot_id)):
            joint_info = p.getJointInfo(robot.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Skip fixed joints
            if joint_type == p.JOINT_FIXED:
                continue
                
            pb_lower = joint_info[8]
            pb_upper = joint_info[9]
            
            if joint_name in urdf_limits:
                urdf_lower, urdf_upper = urdf_limits[joint_name]
                
                # Check if limits match approximately
                lower_match = abs(pb_lower - urdf_lower) < 1e-5
                upper_match = abs(pb_upper - urdf_upper) < 1e-5
                
                status = "✓" if lower_match and upper_match else "✗"
                
                logger.warning(f"{status} Joint: {joint_name}")
                logger.warning(f"  URDF limits: [{urdf_lower:.6f}, {urdf_upper:.6f}]")
                logger.warning(f"  PyBullet limits: [{pb_lower:.6f}, {pb_upper:.6f}]")
                
                if not (lower_match and upper_match):
                    logger.error(f"Joint limits mismatch for {joint_name}!")
            else:
                logger.warning(f"? Joint: {joint_name} - Not found in URDF or no limits defined")
                logger.warning(f"  PyBullet limits: [{pb_lower:.6f}, {pb_upper:.6f}]")
        
    except Exception as e:
        logger.error(f"Error checking joint limits: {e}")

class Controller:
    def __init__(self, urdf_path="vr_teleop/kbot_urdf/robot.urdf"):
        # Check if DISPLAY is set to determine if we should use GUI mode
        display_env = os.environ.get('DISPLAY', '')
        use_gui = display_env != ''
        
        if use_gui:
            logger.warning(f"Using PyBullet GUI mode with DISPLAY={display_env}")
            # We need to use our own physics client to see the GUI
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, 0)
            
            # Load a ground plane for reference
            p.loadURDF("plane.urdf")
            
            # Set physics parameters
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setPhysicsEngineParameter(contactBreakingThreshold=0)
            p.setPhysicsEngineParameter(allowedCcdPenetration=0)
        
        # Replace MJ_KBot with PB_KBot, adding a 1 meter vertical offset
        self.robot_offset = [0, 0, 1.0]  # 1 meter up in z-direction
        self.pbRobot = PB_KBot(urdf_path, use_existing_client=(use_gui, self.physics_client if use_gui else None), base_position=self.robot_offset)
        
        # Check and display joint limits
        check_joint_limits(self.pbRobot, urdf_path)
        
        # Get current end effector positions
        right_ee_pos, _ = self.pbRobot.get_ee_pos(leftside=False)
        left_ee_pos, _ = self.pbRobot.get_ee_pos(leftside=True)
        
        # Store these as the last known positions
        self.right_last_ee_pos = right_ee_pos.copy()
        self.left_last_ee_pos = left_ee_pos.copy()

        # Initialize target positions - set right target 1 meter lower (at ground level)
        # This creates a noticeable distance for the IK to solve
        self.right_target_ee_pos = np.array([0.39237205, 0.16629315, 1.10654296])  # No vertical offset for target
        self.left_target_ee_pos = self.left_last_ee_pos.copy()
        self.squeeze_pressed_prev = False
        
        logger.warning(f"Initial right end effector position: {self.right_last_ee_pos}")
        logger.warning(f"Initial right target position: {self.right_target_ee_pos}")
        logger.warning(f"Initial left end effector position: {self.left_last_ee_pos}")
        logger.warning(f"Initial left target position: {self.left_target_ee_pos}")
        
        # Create visual markers for the target position and end effector
        if use_gui:
            self.visual_markers = {}
            
            # Target position marker (red)
            target_visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.7])
            self.visual_markers['target'] = p.createMultiBody(
                baseVisualShapeIndex=target_visual_id, 
                basePosition=self.right_target_ee_pos
            )
            
            # End effector marker (green)
            ee_pos, _ = self.pbRobot.get_ee_pos(leftside=False)
            ee_visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.7])
            self.visual_markers['ee'] = p.createMultiBody(
                baseVisualShapeIndex=ee_visual_id, 
                basePosition=ee_pos
            )
            
            # Draw a line from EE to target
            self.visual_markers['line'] = p.addUserDebugLine(
                ee_pos, 
                self.right_target_ee_pos, 
                lineColorRGB=[0, 0, 1], 
                lineWidth=2
            )
            
            # Add coordinate axes at robot base to show the offset
            axis_length = 0.3
            axis_width = 3
            
            # X-axis (red)
            p.addUserDebugLine(
                self.robot_offset, 
                [self.robot_offset[0] + axis_length, self.robot_offset[1], self.robot_offset[2]], 
                lineColorRGB=[1, 0, 0], 
                lineWidth=axis_width
            )
            
            # Y-axis (green)
            p.addUserDebugLine(
                self.robot_offset, 
                [self.robot_offset[0], self.robot_offset[1] + axis_length, self.robot_offset[2]], 
                lineColorRGB=[0, 1, 0], 
                lineWidth=axis_width
            )
            
            # Z-axis (blue)
            p.addUserDebugLine(
                self.robot_offset, 
                [self.robot_offset[0], self.robot_offset[1], self.robot_offset[2] + axis_length], 
                lineColorRGB=[0, 0, 1], 
                lineWidth=axis_width
            )
            
            logger.warning(f"Robot positioned with offset: {self.robot_offset}")

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

                logger.debug(f"Controller position: {cur_ee_pos}")
                logger.debug(f"Reference position: {self.right_last_ee_pos}")
                logger.debug(f"Raw delta: {delta}")
                
                #* Temp hack for sensitivity
                delta[0] = 1*delta[0]
                delta[1] = -1*delta[1]
                delta[2] = -1*delta[2]
                
                logger.debug(f"Scaled delta: {delta}")

                # Get current end effector position
                current_ee_pos, _ = self.pbRobot.get_ee_pos(leftside=False)
                prev_target = self.right_target_ee_pos.copy()
                
                # Update target with delta
                self.right_target_ee_pos = self.right_target_ee_pos + delta
                
                logger.warning(f"Current EE: {current_ee_pos}")
                logger.warning(f"Previous target: {prev_target}")
                logger.warning(f"New target: {self.right_target_ee_pos}")
                logger.warning(f"Target delta: {self.right_target_ee_pos - prev_target}")
                
                # Update right_last_ee_pos for next calculation
                self.right_last_ee_pos = cur_ee_pos.copy()
                
                # Update target marker position if visualization is enabled
                if hasattr(self, 'visual_markers') and 'target' in self.visual_markers:
                    p.resetBasePositionAndOrientation(
                        self.visual_markers['target'], 
                        self.right_target_ee_pos, 
                        [0, 0, 0, 1]
                    )
        else:
            # When not squeezed, don't update the target position
            if self.squeeze_pressed_prev:
                logger.debug("Squeeze released, freezing target position")
                
        # Update squeeze state for next iteration
        self.squeeze_pressed_prev = squeeze_pressed
        
        start_time = time.time()
        
        # Get current end effector position before IK
        pre_ik_ee_pos, _ = self.pbRobot.get_ee_pos(leftside=False)
        
        # Use the PyBullet IK solver with joint limits
        full_delta_q, error_norm_pos, error_norm_rot = ik_gradient(
                self.pbRobot, 
                self.right_target_ee_pos, 
                target_ort=None, 
                leftside=False,
            )
        ik_time = time.time() - start_time
        
        # Get current qpos
        current_qpos = np.zeros(len(self.pbRobot.joint_indices))
        for i, joint_idx in enumerate(self.pbRobot.joint_indices):
            current_qpos[i] = p.getJointState(self.pbRobot.robot_id, joint_idx)[0]
            
        new_qpos = current_qpos + full_delta_q
        
        # Apply joint limits explicitly
        new_qpos = joint_limit_clamp(self.pbRobot, new_qpos)
        
        # Log detailed information about the IK solution
        logger.debug(f"IK time: {ik_time*1000:.1f}ms")
        logger.debug(f"Pre-IK EE position: {pre_ik_ee_pos}")
        logger.debug(f"Target EE position: {self.right_target_ee_pos}")
        logger.debug(f"Position error norm: {error_norm_pos}")
        logger.debug(f"Rotation error norm: {error_norm_rot}")
        logger.debug(f"Joint delta: {full_delta_q}")
        logger.debug(f"New joint positions: {new_qpos}")

        # Update robot state
        self.pbRobot.set_qpos(new_qpos)
        
        # Get post-IK end effector position to report actual movement
        post_ik_ee_pos, _ = self.pbRobot.get_ee_pos(leftside=False)
        logger.debug(f"Post-IK EE position: {post_ik_ee_pos}")
        logger.debug(f"Actual EE movement: {post_ik_ee_pos - pre_ik_ee_pos}")
        logger.debug(f"Target vs actual position error: {np.linalg.norm(self.right_target_ee_pos - post_ik_ee_pos)}")
        
        # Update end effector visualization if enabled
        if hasattr(self, 'visual_markers'):
            ee_pos, _ = self.pbRobot.get_ee_pos(leftside=False)
            
            # Update end effector marker
            if 'ee' in self.visual_markers:
                p.resetBasePositionAndOrientation(
                    self.visual_markers['ee'], 
                    ee_pos, 
                    [0, 0, 0, 1]
                )
            
            # Update line from EE to target
            if 'line' in self.visual_markers:
                # Remove old line
                p.removeUserDebugItem(self.visual_markers['line'])
                # Add new line
                self.visual_markers['line'] = p.addUserDebugLine(
                    ee_pos, 
                    self.right_target_ee_pos, 
                    lineColorRGB=[0, 0, 1], 
                    lineWidth=2
                )

        # Extract the right arm joint positions from the full qpos 
        right_arm_qpos = self.pbRobot.get_arm_qpos(leftside=False)
        return right_arm_qpos

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
    11: Actuator(11, 1, 150.0, 8.0, 60.0, False, "left_shoulder_pitch_03"),
    12: Actuator(12, 5, 150.0, 8.0, 60.0, False, "left_shoulder_roll_03"),
    13: Actuator(13, 9, 50.0, 5.0, 17.0, False, "left_shoulder_yaw_02"),
    14: Actuator(14, 13, 50.0, 5.0, 17.0, False, "left_elbow_02"),
    15: Actuator(15, 17, 20.0, 2.0, 17.0, False, "left_wrist_02"),
    21: Actuator(21, 3, 150.0, 8.0, 60.0, False, "right_shoulder_pitch_03"),
    22: Actuator(22, 7, 150.0, 8.0, 60.0, False, "right_shoulder_roll_03"),
    23: Actuator(23, 11, 50.0, 2.0, 17.0, False, "right_shoulder_yaw_02"),
    24: Actuator(24, 15, 50.0, 5.0, 17.0, False, "right_elbow_02"),
    25: Actuator(25, 19, 20.0, 2.0, 17.0, False, "right_wrist_02"),
}


async def command_kbot(qnew_pos, kos):
    # Create a direct mapping for right arm joints
    right_arm_mapping = {
        21: np.degrees(qnew_pos[0]),  # right_shoulder_pitch
        22: -np.degrees(qnew_pos[1]),  # right_shoulder_roll
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
    
    # Create controller with PyBullet
    controller = Controller()
    
    # Extract both left and right arm joint positions
    left_arm_qpos = controller.pbRobot.get_limit_center(leftside=True)
    right_arm_qpos = controller.pbRobot.get_limit_center(leftside=False)

    # Set initial positions in PyBullet
    fullq = controller.pbRobot.convert_armqpos_to_fullqpos(leftarmq=left_arm_qpos, rightarmq=right_arm_qpos)
    controller.pbRobot.set_qpos(fullq)
    
    # Get current actuator states from the robot
    actuator_ids = list(range(11, 16)) + list(range(21, 26))  # Left arm: 11-15, Right arm: 21-25
    
    try:
        actuator_states = await kos.actuator.get_actuators_state(actuator_ids)
        logger.warning(f"Current actuator states retrieved: {len(actuator_states.states)} actuators")
        
        # Create a dictionary to store current positions
        current_positions = {}
        for state in actuator_states.states:
            current_positions[state.actuator_id] = state.position
            logger.warning(f"Actuator {state.actuator_id} current position: {state.position}")
    except Exception as e:
        logger.error(f"Failed to get actuator states: {e}")
        # Default to assuming zeros if we can't get current positions
        current_positions = {aid: 0.0 for aid in actuator_ids}
    
    # Import motion planning function
    from vr_teleop.utils.motion_planning_primitive import find_points_to_target
    
    # Define target positions in degrees
    target_positions = {}
    
    # Calculate left arm target positions (in degrees)
    for i, actuator_id in enumerate(range(11, 16)):
        if actuator_id in ACTUATORS:
            target_positions[actuator_id] = np.degrees(left_arm_qpos[i])
    
    # Calculate right arm target positions (in degrees)
    for i, actuator_id in enumerate(range(21, 26)):
        if actuator_id in ACTUATORS:
            target_positions[actuator_id] = np.degrees(right_arm_qpos[i])

    target_positions[22] = -target_positions[22]
    target_positions[12] = -target_positions[12]
    
    # Generate trajectories for each actuator
    trajectories = {}
    for actuator_id in actuator_ids:
        if actuator_id in ACTUATORS and actuator_id in current_positions and actuator_id in target_positions:
            current_pos = current_positions[actuator_id]
            target_pos = target_positions[actuator_id]
            
            # Skip if already at target position (within tolerance)
            if abs(current_pos - target_pos) < 0.5:
                logger.warning(f"Actuator {actuator_id} already at target position: {current_pos} vs {target_pos}")
                continue
                
            # Generate trajectory with S-curve profile
            trajectory_angles, _, trajectory_times = find_points_to_target(
                current_angle=current_pos,
                target=target_pos,
                acceleration=50.0,  # Reduced acceleration for smoother motion
                V_MAX=20.0,         # Reduced max velocity for safety
                update_rate=50.0,   # 50Hz update rate
                profile="scurve"    # Using S-curve for smooth acceleration
            )
            
            trajectories[actuator_id] = {
                "angles": trajectory_angles,
                "times": trajectory_times,
                "target": target_pos
            }
            
            logger.warning(f"Generated trajectory for actuator {actuator_id}: {len(trajectory_angles)} points, duration: {trajectory_times[-1]:.2f}s")
    
    # Execute trajectories
    if trajectories:
        # Find the maximum trajectory duration
        max_duration = max([traj["times"][-1] for traj in trajectories.values()])
        logger.warning(f"Executing trajectories over {max_duration:.2f} seconds")
        
        # Execute trajectories by sending commands at each time step
        start_time = time.time()
        current_step = 0
        
        update_interval = 1.0 / 50.0  # 50Hz update rate
        end_time = start_time + max_duration
        
        while time.time() < end_time:
            current_time = time.time() - start_time
            
            # Prepare command for this time step
            command = []
            
            for actuator_id, traj in trajectories.items():
                # Find the nearest time point in the trajectory
                next_idx = 0
                while next_idx < len(traj["times"]) and traj["times"][next_idx] <= current_time:
                    next_idx += 1
                
                # Use the trajectory point at or just before the current time
                idx = max(0, next_idx - 1)
                
                if idx < len(traj["angles"]):
                    angle = traj["angles"][idx]
                else:
                    angle = traj["target"]  # Use target if we're past the end of the trajectory
                
                # Add to command
                command.append({
                    "actuator_id": actuator_id,
                    "position": angle,
                })
            
            # Send commands to KOS if we have any
            if command:
                await kos.actuator.command_actuators(command)
            
            # Sleep until next update
            elapsed = time.time() - (start_time + current_time)
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
    else:
        # If no trajectories (all actuators at target positions), just send the target positions directly
        logger.warning("No motion planning needed, sending target positions directly")
        command = []
        for actuator_id, target_pos in target_positions.items():
            if actuator_id in ACTUATORS:
                command.append({
                    "actuator_id": actuator_id,
                    "position": target_pos,
                })
        
        if command:
            await kos.actuator.command_actuators(command)
            await asyncio.sleep(1)  # Wait for positions to be reached
    
    logger.warning("Robot initialized to starting position")
    # breakpoint()
    return controller


async def execute_motion_planning(kos, current_positions, target_positions):
    """Execute motion planning from current positions to target positions."""
    from vr_teleop.utils.motion_planning_primitive import find_points_to_target
    
    # Generate trajectories for each actuator
    trajectories = {}
    actuator_ids = list(current_positions.keys())
    
    for actuator_id in actuator_ids:
        if actuator_id in ACTUATORS and actuator_id in current_positions and actuator_id in target_positions:
            current_pos = current_positions[actuator_id]
            target_pos = target_positions[actuator_id]
            
            # Skip if already at target position (within tolerance)
            if abs(current_pos - target_pos) < 0.5:
                logger.warning(f"Actuator {actuator_id} already at target position: {current_pos} vs {target_pos}")
                continue
                
            # Generate trajectory with S-curve profile
            trajectory_angles, _, trajectory_times = find_points_to_target(
                current_angle=current_pos,
                target=target_pos,
                acceleration=50.0,  # Reduced acceleration for smoother motion
                V_MAX=20.0,         # Reduced max velocity for safety
                update_rate=50.0,   # 50Hz update rate
                profile="scurve"    # Using S-curve for smooth acceleration
            )
            
            trajectories[actuator_id] = {
                "angles": trajectory_angles,
                "times": trajectory_times,
                "target": target_pos
            }
            
            logger.warning(f"Generated trajectory for actuator {actuator_id}: {len(trajectory_angles)} points, duration: {trajectory_times[-1]:.2f}s")
    
    # Execute trajectories
    if trajectories:
        # Find the maximum trajectory duration
        max_duration = max([traj["times"][-1] for traj in trajectories.values()])
        logger.warning(f"Executing trajectories over {max_duration:.2f} seconds")
        
        # Execute trajectories by sending commands at each time step
        start_time = time.time()
        current_step = 0
        
        update_interval = 1.0 / 50.0  # 50Hz update rate
        end_time = start_time + max_duration
        
        while time.time() < end_time:
            current_time = time.time() - start_time
            
            # Prepare command for this time step
            command = []
            
            for actuator_id, traj in trajectories.items():
                # Find the nearest time point in the trajectory
                next_idx = 0
                while next_idx < len(traj["times"]) and traj["times"][next_idx] <= current_time:
                    next_idx += 1
                
                # Use the trajectory point at or just before the current time
                idx = max(0, next_idx - 1)
                
                if idx < len(traj["angles"]):
                    angle = traj["angles"][idx]
                else:
                    angle = traj["target"]  # Use target if we're past the end of the trajectory
                
                # Add to command
                command.append({
                    "actuator_id": actuator_id,
                    "position": angle,
                })
            
            # Send commands to KOS if we have any
            if command:
                await kos.actuator.command_actuators(command)
            
            # Sleep until next update
            elapsed = time.time() - (start_time + current_time)
            sleep_time = max(0, update_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.001)  # Small sleep to prevent CPU hogging
    else:
        # If no trajectories (all actuators at target positions), just send the target positions directly
        logger.warning("No motion planning needed, sending target positions directly")
        command = []
        for actuator_id, target_pos in target_positions.items():
            if actuator_id in ACTUATORS:
                command.append({
                    "actuator_id": actuator_id,
                    "position": target_pos,
                })
        
        if command:
            await kos.actuator.command_actuators(command)
            await asyncio.sleep(1)  # Wait for positions to be reached


async def main():
    """Start KOS and then run WebSocket server and controller task concurrently."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='VR Teleop for KBot')
    parser.add_argument('--gui', action='store_true', help='Enable PyBullet GUI visualization mode regardless of DISPLAY setting')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--kos-ip', type=str, default='10.33.11.154', help='KOS IP address')
    parser.add_argument('--kos-port', type=int, default=50051, help='KOS port')
    parser.add_argument('--ws-port', type=int, default=8586, help='WebSocket server port')
    args = parser.parse_args()
    
    # Set environment variables based on arguments
    if args.gui:
        os.environ['DISPLAY'] = 'FORCED:0'
        logger.warning("Forced GUI mode enabled by command-line argument")
    
    # Set log level based on arguments
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.warning("Debug logging enabled")
    
    try:
        # Initialize KOS connection once
        async with KOS(ip=args.kos_ip, port=args.kos_port) as kos:
            try:
                # await kos.sim.reset()
                # await kos.sim.set_paused(True)

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
                server = await websockets.serve(websocket_handler, "localhost", args.ws_port)
                logger.info(f"WebSocket server started at ws://localhost:{args.ws_port}")
                
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



