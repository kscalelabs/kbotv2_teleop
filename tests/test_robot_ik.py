from vr_teleop.utils.ik import *
from vr_teleop.mjRobot import MJ_KBot
import numpy as np
import pytest
import time
logger.setLevel(logging.INFO) #.DEBUG .INFO

# Define path to URDF
urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"

def run_ik_test(arm_qpos, leftside):
    """Helper function to run an IK test with the given arm position"""
    kbotv2 = MJ_KBot(urdf_path, gravity_enabled=False, timestep=0.001)
    ans_qpos = kbotv2.convert_armqpos_to_fullqpos(arm_qpos, leftside=leftside)
    kbotv2.set_qpos(ans_qpos)

    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    target_pos = kbotv2.data.body(ee_name).xpos.copy()
    target_ort = kbotv2.data.body(ee_name).xquat.copy()
    initial_states = kbotv2.get_limit_center(leftside=leftside)
    kbotv2_for_ik = MJ_KBot(urdf_path, gravity_enabled=False, timestep=0.001)
    pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
        kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, leftside)
    calc_qpos = kbotv2_for_ik.convert_armqpos_to_fullqpos(pos_arm, leftside)
    kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)
    kbotv2.reset()

    logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")
    assert error_norm_pos < 0.01 and error_norm_rot < 0.011

def run_ik_performance_test(arm_qpos, leftside):
    """Helper function to run an IK test with timing measurements"""
    kbotv2 = MJ_KBot(urdf_path, gravity_enabled=False, timestep=0.001)
    ans_qpos = kbotv2.convert_armqpos_to_fullqpos(arm_qpos, leftside=leftside)
    kbotv2.set_qpos(ans_qpos)

    if leftside:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
    else:
        ee_name = "KB_C_501X_Bayonet_Adapter_Hard_Stop"
    target_pos = kbotv2.data.body(ee_name).xpos.copy()
    target_ort = kbotv2.data.body(ee_name).xquat.copy()
    initial_states = kbotv2.get_limit_center(leftside=leftside)
    kbotv2_for_ik = MJ_KBot(urdf_path, gravity_enabled=False, timestep=0.001)
    
    # Measure IK solving time
    start_time = time.time()
    pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
        kbotv2_for_ik.model, kbotv2_for_ik.data, target_pos, target_ort, leftside)
    elapsed_time = time.time() - start_time
    
    calc_qpos = kbotv2_for_ik.convert_armqpos_to_fullqpos(pos_arm, leftside)
    kbotv2.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)
    kbotv2.reset()

    side_name = "Left" if leftside else "Right"
    logger.info(f"{side_name} arm IK computation time: {elapsed_time:.4f} seconds")
    logger.info(f"Error norm position: {error_norm_pos}, Error norm rotation: {error_norm_rot}")
    
    # Verify solution is accurate
    assert elapsed_time < 0.1 # 0.1 seconds 

# Left Arm in edges of reach tests
def test_left_edge_forward():
    run_ik_test([2.0, -1.0, -1.0, -2.0, -1.0], leftside=True)

def test_left_edge_inward():
    run_ik_test([-1.5, -0.3, 1.0, -0.5, -0.5], leftside=True)

def test_left_edge_downward():
    run_ik_test([0, -1.6, 0, -2.5, 0], leftside=True)

def test_left_edge_extended():
    run_ik_test([2.47, -1.57, -1.67, -2.45, -1.68], leftside=True)

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
    run_ik_test([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)

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

# Performance tests
def test_ik_performance_left_workspace_center():
    """Test IK solver performance with left workspace center position"""
    run_ik_performance_test([0, -0.8, 0, -1.5, 0], leftside=True)

def test_ik_performance_left_workspace_front_right():
    """Test IK solver performance with left workspace front right position"""
    run_ik_performance_test([0.5, -0.7, -0.2, -1.4, -0.3], leftside=True)

def test_ik_performance_left_edge_extended():
    """Test IK solver performance with left edge extended position"""
    run_ik_performance_test([2.47, -1.57, -1.67, -2.45, -1.68], leftside=True)

def test_ik_performance_right_workspace_front_mid():
    """Test IK solver performance with right workspace front mid position"""
    run_ik_performance_test([-0.3, 0.6, 0, 1.2, 0], leftside=False)

def test_ik_performance_right_edge_downward():
    """Test IK solver performance with right edge downward position"""
    run_ik_performance_test([0, 1.6, 0, 2.5, 0], leftside=False)

def test_ik_performance_right_edge_extended():
    """Test IK solver performance with right edge extended position"""
    run_ik_performance_test([-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)

def test_right_joint_limit():
    run_ik_test([-2.6, -0.48, -1.7, 0, -1.7], leftside=False)
    
def test_right_joint_limit2():
    run_ik_test([2.0, 1.65, 1.7, 2.53, 1.7], leftside=False)

def test_left_joint_limit():
    run_ik_test([-2, -1.65, -1.74, -2.53, -1.74], leftside=True)

def test_left_joint_limit2():
    run_ik_test([2.6, 0.48, 1.74, 0, 1.74], leftside=True)

def test_hard_test_limit():
    run_ik_test([2.5, -1.6, -1.7, -2.5, -1.7], leftside=True)

def test_hard_test_limit2():
    run_ik_test([-2.5, 1.6, 1.7, 2.5, 1.7], leftside=False)



