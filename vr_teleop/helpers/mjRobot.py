import mujoco
import mujoco.viewer
import time
import numpy as np
import logging
from vr_teleop.utils.logging import setup_logger


class MuJoCo_Robot:
    """
    A general MuJoCo robot with MuJoCo Visualizer used across multiple functionality such as ik debugging.
    """

    def __init__(self, urdf_path, gravity_enabled, timestep):
        self.logger = setup_logger(__name__)
        self.logger.setLevel(logging.DEBUG) #.DEBUG .INFO

        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)
        self.gravity_enabled = gravity_enabled
        
        self.timestep = timestep  # 0.001 = 1000hz

        self.target_time = 0.0
        self.sim_time = 0.0
        self.viewer_ref = []

        if gravity_enabled:
            self.model.opt.gravity = [0, 0, -9.81]
        else:
            self.model.opt.gravity = [0, 0, 0]

    def set_qpos(self, input_qpos):
        self.data.qpos = input_qpos.copy()
        mujoco.mj_step(self.model, self.data)

    def key_callback(self, key):
        keycode = chr(key)
        if keycode == 'R':
            mujoco.mj_resetData(self.model, self.data)
            self.logger.info("Reset data")
        # Add more default key handlers here if needed

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)


    def run_viewer(self, func_to_call=None, custom_key_callback=None, sim_end_time=30):
        #* Sim_end_time is duration in simulation time.
        self.sim_time = 0.0
        self.target_time = time.time()
        
        # Use custom key callback if provided, otherwise use the class method
        key_cb = custom_key_callback if custom_key_callback else self.key_callback
        
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_cb) as viewer:
            # Store the viewer reference
            self.viewer_ref.clear()
            self.viewer_ref.append(viewer)
            
            while viewer.is_running():
                # End the simulation if we've reached the maximum time
                if self.sim_time >= sim_end_time:
                    self.logger.info(f"Simulation ended after reaching time limit of {sim_end_time}s")
                    break
                    
                mujoco.mj_step(self.model, self.data)
                self.sim_time += self.model.opt.timestep
                viewer.sync()

                if func_to_call:
                    func_to_call(self, self.sim_time)
                
                self.target_time += self.model.opt.timestep
                current_time = time.time()
                if self.target_time - current_time > 0:
                    time.sleep(self.target_time - current_time)

class MJ_KBot(MuJoCo_Robot):
    def __init__(self, urdf_path, gravity_enabled=False, timestep=0.001):
        super().__init__(urdf_path, gravity_enabled, timestep)
        self.target_pos = None
        self.target_ort = None
        self.calc_qpos = None
        self.ans_qpos = None
        self.initial_states = None
        
        self.ans_qpos = None 

    def ik_test_set(self, target_pos, target_ort, calc_qpos, ans_qpos=None, initial_states=None):
        self.target_pos = target_pos
        self.target_ort = target_ort
        self.calc_qpos = calc_qpos
        self.ans_qpos = ans_qpos
        self.initial_states = initial_states
    
    def set_iksolve_side(self, leftside: bool):
        """
        Only used for logs during keyboard callbacks.
        """
        self.leftside = leftside

        if leftside:
            self.ee_name =  'KB_C_501X_Bayonet_Adapter_Hard_Stop_2'
        else:
            self.ee_name =  'KB_C_501X_Bayonet_Adapter_Hard_Stop'
    
    def key_callback(self, key):
        """
        Override the default key callback with IK-specific functionality
        """
        keycode = chr(key)
        if keycode == 'R':
            mujoco.mj_resetData(self.model, self.data)
            self.logger.info("Reset data")
        elif keycode == 'Q' and self.calc_qpos is not None:
            self.data.qpos = self.calc_qpos
            mujoco.mj_forward(self.model, self.data)
            self.logger.info("Teleported to Calculated Position")
            self.logger.debug(f"End effector position: {self.data.body(self.ee_name).xpos}")
            self.logger.debug(f"End effector orientation: {self.data.body(self.ee_name).xquat}")
            armJoints = self.convert_fullqpos_to_arm(self.calc_qpos.copy(), self.leftside)
            self.logger.debug(f"Arm Joints at: {armJoints}")
        elif keycode == 'V' and self.ans_qpos is not None:
            self.data.qpos = self.ans_qpos
            mujoco.mj_forward(self.model, self.data)
            self.logger.info("Teleported to Answer Position")
            self.logger.debug(f"End effector position: {self.data.body(self.ee_name).xpos}")
            self.logger.debug(f"End effector orientation: {self.data.body(self.ee_name).xquat}")
            armJoints = self.convert_fullqpos_to_arm(self.ans_qpos.copy(), self.leftside)
            self.logger.debug(f"Arm Joints at: {armJoints}")
        elif keycode == 'P' and self.initial_states is not None:
            self.logger.info('Teleported to Optimization initial condition')
            self.logger.debug(f"Initial states: {self.initial_states}, Leftside: {self.leftside}")
            self.data.qpos = self.convert_armqpos_to_fullqpos(self.initial_states, self.leftside)
            mujoco.mj_forward(self.model, self.data)
        elif keycode == "O" and self.viewer_ref:
            if self.viewer_ref[0].opt.frame == 7:
                self.viewer_ref[0].opt.frame = 1
            else:
                self.viewer_ref[0].opt.frame = 7
            self.viewer_ref[0].sync()
            self.logger.info("Toggled frame visualization")

    def get_ee_pos(self, leftside: bool):
        if leftside:
            ee_name = 'KB_C_501X_Bayonet_Adapter_Hard_Stop_2'
        else:
            ee_name = 'KB_C_501X_Bayonet_Adapter_Hard_Stop'
        
        ee_pos = self.data.body(ee_name).xpos.copy()
        ee_orientation = self.data.body(ee_name).xquat.copy()

        return ee_pos, ee_orientation

    def get_arm_qpos(self, leftside: bool):
        """
        Returns the current position of the arm joints, an array of lenght 5.
        """
        if leftside:
            tjoints = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
        else:
            tjoints = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
        
        joint_indices = []
        
        for key in tjoints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, key)
            if joint_id >= 0:  # Check if joint exists
                qpos_index = self.model.jnt_qposadr[joint_id]
                joint_indices.append(qpos_index)
            else:
                self.logger.warning(f"During Get Arm Qpos: Joint '{key}' not found in model")
        
        joint_positions = np.array([self.data.qpos[idx] for idx in joint_indices])
        return joint_positions
    
    def get_limit_center(self, leftside: bool):
        """
        Returns the median positions (center of limits) for each arm joint
        """
        if leftside:
            tjoints = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
        else:
            tjoints = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
        
        centered_positions = []
        
        for key in tjoints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, key)
            if joint_id >= 0:
                # Check if joint has limits
                if self.model.jnt_limited[joint_id]:
                    # Calculate median between upper and lower limits
                    lower_limit = self.model.jnt_range[joint_id, 0]
                    upper_limit = self.model.jnt_range[joint_id, 1]
                    median_position = (lower_limit + upper_limit) / 2
                else:
                    # If joint has no limits, use current position
                    qpos_index = self.model.jnt_qposadr[joint_id]
                    median_position = self.data.qpos[qpos_index]
                
                centered_positions.append(median_position)
            else:
                self.logger.warning(f"During Initial Condition Setting: Joint '{key}' not found in model")
                centered_positions.append(0.0)  # Default value if joint not found
        
        return np.array(centered_positions)

    def convert_armqpos_to_fullqpos(self, leftarmq=None, rightarmq=None):
        """
        Convert arm-specific joint positions to full qpos array.
        Can handle left arm, right arm, or both arms simultaneously.
        
        Args:
            leftarmq: Left arm joint positions (array or list of length 5) or None
            rightarmq: Right arm joint positions (array or list of length 5) or None
        
        Returns:
            Full qpos array updated with the specified arm joint values
        """
        newqpos = self.data.qpos.copy()
        
        # Process left arm if provided
        if leftarmq is not None:
            left_moves = {
                "left_shoulder_pitch_03": leftarmq[0],
                "left_shoulder_roll_03": leftarmq[1],
                "left_shoulder_yaw_02": leftarmq[2],
                "left_elbow_02": leftarmq[3],
                "left_wrist_02": leftarmq[4]
            }
            
            for joint_name, value in left_moves.items():
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    qpos_index = self.model.jnt_qposadr[joint_id]
                    if self.model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                        raise ValueError(f"Joint {joint_name} is not a hinge joint. This function only works with hinge joints (1 DOF).")
                    newqpos[qpos_index] = value
                else:
                    self.logger.warning(f"Joint '{joint_name}' not found in model")
        
        # Process right arm if provided
        if rightarmq is not None:
            right_moves = {
                "right_shoulder_pitch_03": rightarmq[0],
                "right_shoulder_roll_03": rightarmq[1],
                "right_shoulder_yaw_02": rightarmq[2],
                "right_elbow_02": rightarmq[3],
                "right_wrist_02": rightarmq[4]
            }
            
            for joint_name, value in right_moves.items():
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    qpos_index = self.model.jnt_qposadr[joint_id]
                    if self.model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
                        raise ValueError(f"Joint {joint_name} is not a hinge joint. This function only works with hinge joints (1 DOF).")
                    newqpos[qpos_index] = value
                else:
                    self.logger.warning(f"Joint '{joint_name}' not found in model")
        
        return newqpos

    def convert_fullqpos_to_arm(self, fullq, leftside: bool):
        if leftside:
            tjoints = [
                "left_shoulder_pitch_03",
                "left_shoulder_roll_03",
                "left_shoulder_yaw_02",
                "left_elbow_02",
                "left_wrist_02"
            ]
        else:
            tjoints = [
                "right_shoulder_pitch_03",
                "right_shoulder_roll_03",
                "right_shoulder_yaw_02",
                "right_elbow_02",
                "right_wrist_02"
            ]
        
        # Create a dictionary to store joint names and positions
        arm_positions = {}
        
        for key in tjoints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, key)
            if joint_id >= 0:  # Check if joint exists
                qpos_index = self.model.jnt_qposadr[joint_id]
                # Format the float value to 6 decimal places
                arm_positions[key] = round(float(fullq[qpos_index]), 6)
            else:
                self.logger.warning(f"During convert_fullqpos_to_arm: Joint '{key}' not found in model")
                arm_positions[key] = 0.0  # Default value if joint not found
        
        return arm_positions
    
    def qpos_idx_to_jointname(self):
        qpos_to_joint = {}
        
        # Iterate through all joints in the model
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            qpos_index = self.model.jnt_qposadr[i]
            
            # For slide and hinge joints (1 DOF)
            if self.model.jnt_type[i] in [1, 3]:  # 1=slide, 3=hinge
                qpos_to_joint[qpos_index] = joint_name
            # For ball joints (3 DOF) and free joints (7 DOF), multiple qpos indices map to one joint
            elif self.model.jnt_type[i] == 2:  # ball joint
                for offset in range(4):  # quaternion (4 values)
                    qpos_to_joint[qpos_index + offset] = f"{joint_name}[{offset}]"
            elif self.model.jnt_type[i] == 0:  # free joint
                for offset in range(7):  # position (3) + quaternion (4)
                    qpos_to_joint[qpos_index + offset] = f"{joint_name}[{offset}]"
        
        return qpos_to_joint



