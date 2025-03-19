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

    def key_callback(self, key):
        """
        Default key callback function. Override this in subclasses.
        """
        keycode = chr(key)
        if keycode == 'R':
            mujoco.mj_resetData(self.model, self.data)
            self.logger.info("Reset data")
        # Add more default key handlers here if needed

    def start_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        self.model.opt.timestep = 0.001

        if self.gravity_enabled:
            self.model.opt.gravity = [0, 0, -9.81]
        else:
            self.model.opt.gravity = [0, 0, 0]

        
        self.target_time = time.time()
        mujoco.mj_step(self.model, self.data)

    def run_viewer(self, custom_key_callback=None):
        """
        Launches and runs the MuJoCo viewer for this robot.
        
        Args:
            custom_key_callback: Optional custom callback function for keyboard events
                                If provided, will override the default key_callback
        """
        self.sim_time = 0.0
        self.target_time = time.time()
        
        # Use custom key callback if provided, otherwise use the class method
        key_cb = custom_key_callback if custom_key_callback else self.key_callback
        
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_cb) as viewer:
            # Store the viewer reference
            self.viewer_ref.clear()
            self.viewer_ref.append(viewer)
            
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                self.sim_time += self.model.opt.timestep
                viewer.sync()
                
                self.target_time += self.model.opt.timestep
                current_time = time.time()
                if self.target_time - current_time > 0:
                    time.sleep(self.target_time - current_time)

class Ik_Robot(MuJoCo_Robot):
    def __init__(self, urdf_path, target_pos, target_ort, gravity_enabled, timestep):
        super().__init__(urdf_path, gravity_enabled, timestep)  # Make sure to call the parent class constructor
        self.target_pos = target_pos
        self.target_ort = target_ort
        self.calc_qpos = None
        self.initial_states = None
        
        self.ans_qpos = None  # You might want to define this somewhere
        self.ee_name = 'KB_C_501X_Bayonet_Adapter_Hard_Stop_2'

    
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
            np.savetxt('./vr_teleop/data/calculated_qpos.txt', self.calc_qpos)
            self.logger.info(f"End effector position: {self.data.body(self.ee_name).xpos}")
            self.logger.info(f"End effector orientation: {self.data.body(self.ee_name).xquat}")
        elif keycode == 'V' and self.ansqpos is not None:
            self.data.qpos = self.ans_qpos
            mujoco.mj_forward(self.model, self.data)
            self.logger.info("Teleported to Answer Position")
            np.savetxt('./vr_teleop/data/ans_qpos.txt', self.ans_qpos)
            self.logger.info(f"End effector position: {self.data.body(self.ee_name).xpos}")
            self.logger.info(f"End effector orientation: {self.data.body(self.ee_name).xquat}")
        elif keycode == 'P' and self.initial_states is not None:
            from vr_teleop.utils.mujoco_helper import arms_to_fullqpos  # Import where needed
            self.data.qpos = arms_to_fullqpos(self.model, self.data, self.initial_states, True)
            mujoco.mj_forward(self.model, self.data)
            self.logger.info('Teleported to Optimization initial condition')
        elif keycode == "O" and self.viewer_ref:
            if self.viewer_ref[0].opt.frame == 7:
                self.viewer_ref[0].opt.frame = 1
            else:
                self.viewer_ref[0].opt.frame = 7
            self.viewer_ref[0].sync()
            self.logger.info("Toggled frame visualization")
