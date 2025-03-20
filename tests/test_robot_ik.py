from vr_teleop.utils.ik import *
from vr_teleop.ikrobot import KBot_Robot
import numpy as np
import pytest

# Define path to URDF
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"

def run_ik_test(arm_qpos, leftside):
    """Helper function to run an IK test with the given arm position"""
    kbotv2 = KBot_Robot(urdf_path, gravity_enabled=False, timestep=0.001)
    ans_qpos = kbotv2.convert_armqpos_to_fullqpos(arm_qpos, leftside=leftside)
    kbotv2.set_qpos(ans_qpos)

    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    target_pos = kbotv2.data.body(ee_name).xpos.copy()
    target_ort = kbotv2.data.body(ee_name).xquat.copy()
    initial_states = kbotv2.get_limit_center(leftside=leftside)

    kbotv2_for_ik = KBot_Robot(urdf_path, gravity_enabled=False, timestep=0.001)
    calc_qpos, error_norm_pos, error_norm_rot = inverse_kinematics(
        kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, initial_states, leftside
    )
    kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)
    kbotv2.reset()

    logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")
    assert error_norm_pos < 0.01 and error_norm_rot < 0.01

# Left Arm in edges of reach tests
def test_left_edge_forward():
    run_ik_test([2.0, -1.0, -1.0, -2.0, -1.0], leftside=True)

def test_left_edge_inward():
    run_ik_test([-1.5, -0.3, 1.0, -0.5, -0.5], leftside=True)

def test_left_edge_downward():
    run_ik_test([0, -1.6, 0, -2.5, 0], leftside=True)

def test_left_edge_extended():
    run_ik_test([2.5, -1.6, -1.7, -2.5, -1.7], leftside=True)

def test_left_edge_upward():
    run_ik_test([-2.0, -0.5, 1.7, -1.0, 0.5], leftside=True)

# Left arm positions for workspace manipulation tests
def test_left_workspace_center():
    run_ik_test([0, -0.8, 0, -1.5, 0], leftside=True)

def test_left_workspace_front_mid():
    run_ik_test([0.3, -0.6, 0, -1.2, 0], leftside=True)

def test_left_workspace_front_right():
    run_ik_test([0.5, -0.7, -0.2, -1.4, -0.3], leftside=True)

def test_left_workspace_front_left():
    run_ik_test([0.5, -0.7, 0.2, -1.4, 0.3], leftside=True)

def test_left_workspace_front_low():
    run_ik_test([0.2, -0.9, 0, -1.6, -0.7], leftside=True)

def test_left_workspace_angled():
    run_ik_test([0.4, -1.0, -0.4, -1.8, 0.5], leftside=True)

def test_left_workspace_rear_high():
    run_ik_test([-0.2, -0.5, 0.3, -1.3, -0.2], leftside=True)

def test_left_workspace_far_front():
    run_ik_test([0.7, -0.8, -0.1, -2.0, 0], leftside=True)

def test_left_workspace_high():
    run_ik_test([0.3, -0.4, 0, -1.0, 0], leftside=True)

def test_left_workspace_twisted():
    run_ik_test([0.5, -0.6, -0.5, -1.5, 0.5], leftside=True)

# Right Arm in edges of reach tests
def test_right_edge_forward():
    run_ik_test([-2.0, 1.0, 1.0, 2.0, 1.0], leftside=False)

def test_right_edge_inward():
    run_ik_test([1.5, 0.3, -1.0, 0.5, 0.5], leftside=False)

def test_right_edge_downward():
    run_ik_test([0, 1.6, 0, 2.5, 0], leftside=False)

def test_right_edge_extended():
    run_ik_test([-2.5, 1.6, 1.7, 2.5, 1.7], leftside=False)

def test_right_edge_upward():
    run_ik_test([2.0, 0.5, -1.7, 1.0, -0.5], leftside=False)

# Right Arm in front of workspace tests
def test_right_workspace_front_mid():
    run_ik_test([-0.3, 0.6, 0, 1.2, 0], leftside=False)

def test_right_workspace_front_left():
    run_ik_test([-0.5, 0.7, 0.2, 1.4, 0.3], leftside=False)

def test_right_workspace_front_right():
    run_ik_test([-0.5, 0.7, -0.2, 1.4, -0.3], leftside=False)

def test_right_workspace_front_low():
    run_ik_test([-0.2, 0.9, 0, 1.6, 0.7], leftside=False)

def test_right_workspace_rear_high():
    run_ik_test([0.2, 0.5, -0.3, 1.3, 0.2], leftside=False)

def test_right_workspace_far_front():
    run_ik_test([-0.7, 0.8, 0.1, 2.0, 0], leftside=False)

def test_right_workspace_high():
    run_ik_test([-0.3, 0.4, 0, 1.0, 0], leftside=False)

def test_right_workspace_twisted():
    run_ik_test([-0.5, 0.6, 0.5, 1.5, -0.5], leftside=False)




