from dataclasses import dataclass

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    flip_sign: bool = False


ACTUATOR_LIST_SIM = {
    # actuator id, nn id, kp, kd, max_torque, flip_sign
    11: Actuator(11, 1, 150.0, 8.0, 60.0, True),  # left_shoulder_pitch_03
    12: Actuator(12, 5, 150.0, 8.0, 60.0, False),  # left_shoulder_roll_03
    13: Actuator(13, 9, 50.0, 5.0, 17.0, False),  # left_shoulder_yaw_02
    14: Actuator(14, 13, 50.0, 5.0, 17.0, False),  # left_elbow_02
    15: Actuator(15, 17, 20.0, 2.0, 17.0, False),  # left_wrist_02
    21: Actuator(21, 3, 150.0, 8.0, 60.0, False),  # right_shoulder_pitch_03
    22: Actuator(22, 7, 150.0, 8.0, 60.0, True),  # right_shoulder_roll_03
    23: Actuator(23, 11, 50.0, 2.0, 17.0, True),  # right_shoulder_yaw_02
    24: Actuator(24, 15, 50.0, 5.0, 17.0, True),  # right_elbow_02
    25: Actuator(25, 19, 20.0, 2.0, 17.0, False),  # right_wrist_02
    31: Actuator(31, 0, 100.0, 6.1504, 80.0, True),  # left_hip_pitch_04 (RS04_Pitch)
    32: Actuator(32, 4, 50.0, 11.152, 60.0, False),  # left_hip_roll_03 (RS03_Roll)
    33: Actuator(33, 8, 50.0, 11.152, 60.0, False),  # left_hip_yaw_03 (RS03_Yaw)
    34: Actuator(34, 12, 100.0, 6.1504, 80.0, True),  # left_knee_04 (RS04_Knee)
    35: Actuator(35, 16, 20.0, 0.6, 17.0, False),  # left_ankle_02 (RS02)
    41: Actuator(41, 2, 100, 7.0, 80.0, False),  # right_hip_pitch_04 (RS04_Pitch)
    42: Actuator(42, 6, 50.0, 11.152, 60.0, True),  # right_hip_roll_03 (RS03_Roll)
    43: Actuator(43, 10, 50.0, 11.152, 60.0, True),  # right_hip_yaw_03 (RS03_Yaw)
    44: Actuator(44, 14, 100.0, 6.1504, 80.0, False),  # right_knee_04 (RS04_Knee)
    45: Actuator(45, 18, 20.0, 0.6, 17.0, True),  # right_ankle_02 (RS02)
}


ACTUATOR_LIST_REAL = {
    # actuator id, nn id, kp, kd, max_torque, flip_sign
    11: Actuator(11, 1, 150.0, 8.0, 60.0, True),  # left_shoulder_pitch_03
    12: Actuator(12, 5, 150.0, 8.0, 60.0, False),  # left_shoulder_roll_03
    13: Actuator(13, 9, 50.0, 5.0, 17.0, False),  # left_shoulder_yaw_02
    14: Actuator(14, 13, 50.0, 5.0, 17.0, False),  # left_elbow_02
    15: Actuator(15, 17, 20.0, 2.0, 17.0, False),  # left_wrist_02
    21: Actuator(21, 3, 150.0, 8.0, 60.0, False),  # right_shoulder_pitch_03
    22: Actuator(22, 7, 150.0, 8.0, 60.0, True),  # right_shoulder_roll_03
    23: Actuator(23, 11, 50.0, 2.0, 17.0, True),  # right_shoulder_yaw_02
    24: Actuator(24, 15, 50.0, 5.0, 17.0, True),  # right_elbow_02
    25: Actuator(25, 19, 20.0, 2.0, 17.0, False),  # right_wrist_02
    31: Actuator(31, 0, 100.0, 6.1504, 80.0, True),  # left_hip_pitch_04 (RS04_Pitch)
    32: Actuator(32, 4, 50.0, 11.152, 60.0, True),  # left_hip_roll_03 (RS03_Roll)
    33: Actuator(33, 8, 50.0, 11.152, 60.0, False),  # left_hip_yaw_03 (RS03_Yaw)
    34: Actuator(34, 12, 200.0, 6.1504, 80.0, True),  # left_knee_04 (RS04_Knee)
    35: Actuator(35, 16, 50.0, 5, 17.0, True),  # left_ankle_02 (RS02)
    41: Actuator(41, 2, 100, 7.0, 80.0, False),  # right_hip_pitch_04 (RS04_Pitch)
    42: Actuator(42, 6, 50.0, 11.152, 60.0, False),  # right_hip_roll_03 (RS03_Roll)
    43: Actuator(43, 10, 50.0, 11.152, 60.0, True),  # right_hip_yaw_03 (RS03_Yaw)
    44: Actuator(44, 14, 200.0, 6.1504, 80.0, False),  # right_knee_04 (RS04_Knee)
    45: Actuator(45, 18, 50.0, 5, 17.0, False),  # right_ankle_02 (RS02)
}

