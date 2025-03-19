import mujoco
import numpy as np


r_kinematic_chain = ["YOKE_STOP_INNER", "RS03_5", "R_Bicep_Lower_Drive", "R_Forearm_Upper_Structural", "KB_C_501X_Bayonet_Adapter_Hard_Stop"]
l_kinematic_chain = ["YOKE_STOP_INNER_2", "RS03_6", "L_Bicep_Lower_Drive", "L_Forearm_Upper_Drive", "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"]

r_end_effector = ["KB_C_501X_Bayonet_Adapter_Hard_Stop"]
l_end_effector = ["KB_C_501X_Bayonet_Adapter_Hard_Stop_2"]


# Helper function for final error calculation (similar to novisual.py)
def debug_get_ee_pos(qpos_file):
    # Create a new model and data instance for clean calculation
    temp_model = mujoco.MjModel.from_xml_path("vr_teleop/kbot_urdf/scene.mjcf")
    temp_data = mujoco.MjData(temp_model)
    
    # Reset data and set model properties
    mujoco.mj_resetData(temp_model, temp_data)
    temp_model.opt.timestep = 0.001
    temp_model.opt.gravity[2] = 0
    
    # Load qpos from file and apply it
    loaded_qpos = np.loadtxt(qpos_file)
    temp_data.qpos = loaded_qpos
    temp_data.qvel[:] = 0
    
    # Forward kinematics to update positions
    mujoco.mj_forward(temp_model, temp_data)
    mujoco.mj_step(temp_model, temp_data)
    
    # Get the position of the adapter
    adapter_position = temp_data.body("KB_C_501X_Bayonet_Adapter_Hard_Stop_2").xpos.copy()
    
    return adapter_position


def get_joints(model, data, leftside: bool, tolimitcenter: bool = False):
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
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
        if joint_id >= 0:  # Check if joint exists
            qpos_index = model.jnt_qposadr[joint_id]
            joint_indices.append(qpos_index)
        else:
            print(f"Warning: Joint '{key}' not found in model")
    
    joint_positions = np.array([data.qpos[idx] for idx in joint_indices])
    
    # Move joints to median of limits if tolimitcenter is True
    if tolimitcenter:
        for i, key in enumerate(tjoints):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
            if joint_id >= 0:
                # Check if joint has limits
                if model.jnt_limited[joint_id]:
                    # Calculate median between upper and lower limits
                    lower_limit = model.jnt_range[joint_id, 0]
                    upper_limit = model.jnt_range[joint_id, 1]
                    median_position = (lower_limit + upper_limit) / 2
                    
                    # Update the joint position to the median
                    joint_positions[i] = median_position
    
    return joint_positions


def slice_dofs(model, data, input_list, leftside: bool):
    if len(input_list.shape) == 1:
        input_list = input_list.reshape(1, -1)

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
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
        if joint_id >= 0:  # Check if joint exists
            qpos_index = model.jnt_qposadr[joint_id]
            joint_indices.append(qpos_index)
        else:
            print(f"Warning: Joint '{key}' not found in model")

    x_shape = input_list.shape[0]
    num_joints = len(joint_indices)
    result = np.zeros((x_shape, num_joints))
    
    for i in range(x_shape):
        for j, joint_idx in enumerate(joint_indices):
            result[i, j] = input_list[i, joint_idx]

    return result


def move_joints(model, data, inputqloc: dict, leftside: bool, ):
    moves = {}

    newqpos = data.qpos.copy()

    if leftside:
        moves = {
            "left_shoulder_pitch_03": inputqloc[0],
            "left_shoulder_roll_03": inputqloc[1],
            "left_shoulder_yaw_02": inputqloc[2],
            "left_elbow_02": inputqloc[3],
            "left_wrist_02": inputqloc[4]
        }
    else:
        moves = {
            "right_shoulder_pitch_03": inputqloc[0],
            "right_shoulder_roll_03": inputqloc[1],
            "right_shoulder_yaw_02": inputqloc[2],
            "right_elbow_02": inputqloc[3],
            "right_wrist_02": inputqloc[4]
        }

    for key, value in moves.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, key)
        qpos_index = model.jnt_qposadr[joint_id]

        if model.jnt_type[joint_id] != 3:  # 3 is for hinge joints (1 DOF)
            raise ValueError(f"Joint {key} is not a hinge joint. This function only works with hinge joints (1 DOF).")
            return 
        if joint_id >= 0:
            newqpos[qpos_index] = value
    
    return newqpos


