import pybullet as p
import pybullet_data
import time
import numpy as np

# Initialize PyBullet simulation
p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)


plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("./vr_teleop/kbot_urdf/robot.urdf", [0, 0, 1], p.getQuaternionFromEuler([0,0,0]))

# p.setGravity(0, 0, -9.81)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
