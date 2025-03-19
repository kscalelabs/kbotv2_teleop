import mujoco
import mujoco.viewer
import time

import numpy as np


model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

model.opt.timestep = 0.001  # 0.001 = 1000hz
model.opt.gravity[2] = 0



with mujoco.viewer.launch_passive(model, data) as viewer:

    target_time = time.time()
    sim_time = 0.0

    while viewer.is_running():

        # Step simulation
        mujoco.mj_step(model, data)
        sim_time += model.opt.timestep
        viewer.sync()

        if sim_time > 2 and sim_time < 2.01:
            # loaded_qpos = np.loadtxt('./vr_teleop/data/calculated_qpos.txt') #[-0.23478118  0.10839152  0.80010863]
            loaded_qpos = np.loadtxt('./vr_teleop/data/ans_qpos.txt') #[-0.32186162  0.1665294   0.97334178]
            data.qpos = loaded_qpos
            mujoco.mj_step(model,data)
            mujoco.mj_forward(model, data)

            data.qvel[:] = 0
  
        print(data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos)

        target_time += model.opt.timestep
        current_time = time.time()
        if target_time - current_time > 0:
            time.sleep(target_time - current_time)

# >>> error = np.subtract([-0.1739883, 0.135415, 0.92103787], [-0.23478118, 0.10839, 0.8])
# >>> error
# array([0.06079288, 0.027025  , 0.12103787])
# >>> np.linalg.norm(error)
# np.float64(0.13811694630939136)


