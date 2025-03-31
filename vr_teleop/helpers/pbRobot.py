import pybullet as p
import pybullet_data
import time
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger


class PyBullet_Robot:
    """
    A general PyBullet robot used across multiple functionality such as ik debugging.
    """

    def __init__(self, urdf_path, gravity_enabled=False, timestep=1/240.0, use_existing_client=None, base_position=None):
        self.logger = setup_logger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Set default base position if not provided
        if base_position is None:
            self.base_position = [0, 0, 0]
        else:
            self.base_position = base_position
            self.logger.info(f"Robot base position set to: {self.base_position}")

        # Use an existing physics client if provided
        if use_existing_client and isinstance(use_existing_client, tuple) and use_existing_client[0]:
            self.using_external_client = True
            self.physics_client = use_existing_client[1]
            self.logger.info("Using existing PyBullet physics client")
        else:
            # Connect to the physics server
            self.using_external_client = False
            self.physics_client = p.connect(p.DIRECT)  # Use DIRECT for headless mode or GUI for visualization
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.gravity_enabled = gravity_enabled
        self.timestep = timestep
        self.urdf_path = urdf_path
        
        # Set gravity based on parameter
        if gravity_enabled:
            p.setGravity(0, 0, -9.81)
        else:
            p.setGravity(0, 0, 0)
            
        # Load URDF with the base position offset
        self.robot_id = p.loadURDF(urdf_path, basePosition=self.base_position, useFixedBase=True)
        
        # Disable dynamics and constraint solving for pure kinematics
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0)
        p.setPhysicsEngineParameter(allowedCcdPenetration=0)
        
        # Disable default joint damping - we only want kinematics, not dynamics
        for i in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(self.robot_id, i, 
                             linearDamping=0, 
                             angularDamping=0, 
                             jointDamping=0,
                             mass=0)  # Setting mass to 0 for pure kinematics
        
        # Get number of joints and other model info
        self.num_joints = p.getNumJoints(self.robot_id)
        
        # Store joint info for later use
        self.joint_indices = []
        self.joint_names = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Only consider movable joints
            if joint_type != p.JOINT_FIXED:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
                self.joint_lower_limits.append(joint_info[8])
                self.joint_upper_limits.append(joint_info[9])
        
        # Initialize data for joint positions
        self.qpos = np.zeros(len(self.joint_indices))
        for i, joint_idx in enumerate(self.joint_indices):
            self.qpos[i] = p.getJointState(self.robot_id, joint_idx)[0]
        
        # Store initial state
        self.initial_state = p.saveState()

    def set_qpos(self, input_qpos):
        """Set joint positions and update the simulation"""
        if len(input_qpos) >= len(self.joint_indices):
            for i, joint_idx in enumerate(self.joint_indices):
                p.resetJointState(self.robot_id, joint_idx, input_qpos[i], targetVelocity=0)
            
            # Update our qpos cache
            for i, joint_idx in enumerate(self.joint_indices):
                self.qpos[i] = p.getJointState(self.robot_id, joint_idx)[0]
            
            # Step simulation once for forward kinematics update only
            # This doesn't add dynamics since we've disabled them
            p.stepSimulation()
        else:
            self.logger.error(f"Incorrect number of joint positions. Expected at least {len(self.joint_indices)}, got {len(input_qpos)}")

    def reset(self):
        """Reset the simulation to initial state"""
        p.restoreState(self.initial_state)
        # Update qpos after reset
        for i, joint_idx in enumerate(self.joint_indices):
            self.qpos[i] = p.getJointState(self.robot_id, joint_idx)[0]

    def run_viewer(self, func_to_call=None, custom_key_callback=None, sim_end_time=30):
        """Run the simulation with visualization"""
        # Skip if we're using an external client that's already in GUI mode
        if self.using_external_client:
            self.logger.info("Using existing PyBullet physics client for visualization")
            return
            
        # Switch to GUI mode if not already
        if p.getConnectionInfo(self.physics_client)["connectionMethod"] != p.GUI:
            p.disconnect()
            self.physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            # Reload URDF with the base position offset
            self.robot_id = p.loadURDF(self.urdf_path, basePosition=self.base_position, useFixedBase=True)
            
            # Reapply dynamics disabling
            p.setGravity(0, 0, 0)
            p.setPhysicsEngineParameter(enableConeFriction=0)
            p.setPhysicsEngineParameter(contactBreakingThreshold=0)
            p.setPhysicsEngineParameter(allowedCcdPenetration=0)
            
            # Disable default joint damping again
            for i in range(p.getNumJoints(self.robot_id)):
                p.changeDynamics(self.robot_id, i, 
                                linearDamping=0, 
                                angularDamping=0, 
                                jointDamping=0,
                                mass=0)
            
        self.sim_time = 0.0
        start_time = time.time()
        
        while self.sim_time < sim_end_time:
            # Process any key events (limited functionality compared to MuJoCo)
            keys = p.getKeyboardEvents()
            if custom_key_callback and keys:
                for key in keys:
                    if keys[key] & p.KEY_WAS_TRIGGERED:
                        custom_key_callback(key)
            
            # Call the provided function
            if func_to_call:
                func_to_call(self, self.sim_time)
            
            # Step simulation (only for kinematics, not dynamics)
            p.stepSimulation()
            self.sim_time += self.timestep
            
            # Sleep to match real-time if possible
            elapsed = time.time() - start_time
            sleep_time = self.sim_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.logger.info(f"Simulation ended after reaching time limit of {sim_end_time}s")

    def __del__(self):
        """Disconnect from the physics server when the object is destroyed"""
        if hasattr(self, 'physics_client') and p.isConnected(self.physics_client) and not self.using_external_client:
            p.disconnect(self.physics_client)


class PB_KBot(PyBullet_Robot):
    def __init__(self, urdf_path, gravity_enabled=False, timestep=1/240.0, use_existing_client=None, base_position=None):
        super().__init__(urdf_path, gravity_enabled, timestep, use_existing_client, base_position)
        self.target_pos = None
        self.target_ort = None
        self.calc_qpos = None
        self.ans_qpos = None
        self.initial_states = None
        self.leftside = False
        
        # Define end effector names
        self.left_ee_name = 'KB_C_501X_Bayonet_Adapter_Hard_Stop_2'
        self.right_ee_name = 'KB_C_501X_Bayonet_Adapter_Hard_Stop'
        
        # Find the link indices for end effectors
        self.left_ee_index = -1
        self.right_ee_index = -1
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == self.left_ee_name:
                self.left_ee_index = i
            elif link_name == self.right_ee_name:
                self.right_ee_index = i
        
        if self.left_ee_index == -1:
            self.logger.warning(f"Left end effector '{self.left_ee_name}' not found in model")
        if self.right_ee_index == -1:
            self.logger.warning(f"Right end effector '{self.right_ee_name}' not found in model")
    
    def ik_test_set(self, target_pos, target_ort, calc_qpos, ans_qpos=None, initial_states=None):
        self.target_pos = target_pos
        self.target_ort = target_ort
        self.calc_qpos = calc_qpos
        self.ans_qpos = ans_qpos
        self.initial_states = initial_states
    
    def set_iksolve_side(self, leftside: bool):
        """
        Set which arm to solve IK for.
        """
        self.leftside = leftside
        if leftside:
            self.ee_name = self.left_ee_name
            self.ee_index = self.left_ee_index
        else:
            self.ee_name = self.right_ee_name
            self.ee_index = self.right_ee_index
    
    def key_callback(self, key):
        """
        Handle keyboard events
        """
        # Map PyBullet key codes to characters (approximation)
        keycode = chr(key) if key < 256 else None
        
        if keycode == 'R':
            self.reset()
            self.logger.info("Reset data")
        elif keycode == 'Q' and self.calc_qpos is not None:
            self.set_qpos(self.calc_qpos)
            self.logger.info("Teleported to Calculated Position")
            ee_pos, ee_ort = self.get_ee_pos(self.leftside)
            self.logger.debug(f"End effector position: {ee_pos}")
            self.logger.debug(f"End effector orientation: {ee_ort}")
            armJoints = self.get_arm_qpos(self.leftside)
            self.logger.debug(f"Arm Joints at: {armJoints}")
        elif keycode == 'V' and self.ans_qpos is not None:
            self.set_qpos(self.ans_qpos)
            self.logger.info("Teleported to Answer Position")
            ee_pos, ee_ort = self.get_ee_pos(self.leftside)
            self.logger.debug(f"End effector position: {ee_pos}")
            self.logger.debug(f"End effector orientation: {ee_ort}")
            armJoints = self.get_arm_qpos(self.leftside)
            self.logger.debug(f"Arm Joints at: {armJoints}")
        elif keycode == 'P' and self.initial_states is not None:
            self.logger.info('Teleported to Optimization initial condition')
            self.logger.debug(f"Initial states: {self.initial_states}, Leftside: {self.leftside}")
            self.set_qpos(self.convert_armqpos_to_fullqpos(self.initial_states, self.leftside))

    def get_ee_pos(self, leftside: bool):
        """Get the end effector position and orientation"""
        if leftside:
            ee_index = self.left_ee_index
        else:
            ee_index = self.right_ee_index
            
        if ee_index != -1:
            link_state = p.getLinkState(self.robot_id, ee_index)
            ee_pos = np.array(link_state[0])  # Position
            ee_orientation = np.array(link_state[1])  # Quaternion [x,y,z,w]
            # Convert quaternion to match MuJoCo format [w,x,y,z]
            ee_orientation = np.array([ee_orientation[3], ee_orientation[0], ee_orientation[1], ee_orientation[2]])
            return ee_pos, ee_orientation
        else:
            self.logger.error(f"End effector index not found for {'left' if leftside else 'right'} arm")
            return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])  # Default values
    
    def get_arm_qpos(self, leftside: bool):
        """
        Returns the current position of the arm joints, an array of length 5.
        """
        if leftside:
            joint_names = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
        else:
            joint_names = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
        
        joint_indices = []
        
        for name in joint_names:
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[1].decode('utf-8') == name:
                    joint_indices.append(i)
                    break
        
        joint_positions = np.array([p.getJointState(self.robot_id, idx)[0] for idx in joint_indices])
        return joint_positions
    
    def get_limit_center(self, leftside: bool):
        """
        Returns the median positions (center of limits) for each arm joint
        """
        if leftside:
            joint_names = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
        else:
            joint_names = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
        
        centered_positions = []
        
        for name in joint_names:
            found = False
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[1].decode('utf-8') == name:
                    lower_limit = joint_info[8]
                    upper_limit = joint_info[9]
                    median_position = (lower_limit + upper_limit) / 2
                    centered_positions.append(median_position)
                    found = True
                    break
            
            if not found:
                self.logger.warning(f"During Initial Condition Setting: Joint '{name}' not found in model")
                centered_positions.append(0.0)  # Default value if joint not found
        
        return np.array(centered_positions)

    def convert_armqpos_to_fullqpos(self, leftarmq=None, rightarmq=None):
        """
        Convert arm-specific joint positions to full qpos array.
        Can handle left arm, right arm, or both arms simultaneously.
        """
        newqpos = np.zeros(len(self.joint_indices))
        
        # Get current joint positions to start with
        for i, joint_idx in enumerate(self.joint_indices):
            newqpos[i] = p.getJointState(self.robot_id, joint_idx)[0]
        
        # Process left arm if provided
        if leftarmq is not None:
            left_joint_names = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
            
            for i, name in enumerate(left_joint_names):
                for j, joint_idx in enumerate(self.joint_indices):
                    joint_info = p.getJointInfo(self.robot_id, joint_idx)
                    if joint_info[1].decode('utf-8') == name:
                        newqpos[j] = leftarmq[i]
                        break
        
        # Process right arm if provided
        if rightarmq is not None:
            right_joint_names = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
            
            for i, name in enumerate(right_joint_names):
                for j, joint_idx in enumerate(self.joint_indices):
                    joint_info = p.getJointInfo(self.robot_id, joint_idx)
                    if joint_info[1].decode('utf-8') == name:
                        newqpos[j] = rightarmq[i]
                        break
        
        return newqpos

    def convert_fullqpos_to_arm(self, fullqpos, leftside: bool):
        """
        Convert full qpos array to arm-specific joint positions.
        """
        if leftside:
            joint_names = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
        else:
            joint_names = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
        
        arm_qpos = np.zeros(5)
        
        for i, name in enumerate(joint_names):
            for j, joint_idx in enumerate(self.joint_indices):
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                if joint_info[1].decode('utf-8') == name:
                    if j < len(fullqpos):  # Safety check
                        arm_qpos[i] = fullqpos[j]
                    break
        
        return arm_qpos 