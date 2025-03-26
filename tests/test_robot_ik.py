import logging
import time
from typing import Tuple, List, Optional

import numpy as np
import pytest

from vr_teleop.utils.ik import inverse_kinematics
from vr_teleop.mjRobot import MJ_KBot

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Constants
URDF_PATH = "vr_teleop/kbot_urdf/scene.mjcf"
POSITION_ERROR_THRESHOLD = 0.05
ROTATION_ERROR_THRESHOLD = 0.05
PERFORMANCE_THRESHOLD_SECONDS = 0.1
SIMULATION_TIMESTEP = 0.001

# End effector names
LEFT_EE_NAME = "KB_C_501X_Bayonet_Adapter_Hard_Stop_2"
RIGHT_EE_NAME = "KB_C_501X_Bayonet_Adapter_Hard_Stop"


@pytest.fixture(scope="function")
def kbot():
    """Fixture that provides a fresh KBot instance for each test."""
    robot = MJ_KBot(URDF_PATH, gravity_enabled=False, timestep=SIMULATION_TIMESTEP)
    yield robot
    robot.reset()


def run_ik(kbot: MJ_KBot, arm_qpos: List[float], leftside: bool, 
           measure_performance: bool = False) -> Tuple[float, float, Optional[float]]:
    """
    Returns:
        Tuple containing (position_error, rotation_error, elapsed_time)
        where elapsed_time is None if measure_performance is False
    """
    # Convert arm joint positions to full robot positions and set them
    ans_qpos = kbot.convert_armqpos_to_fullqpos(arm_qpos, leftside=leftside)
    kbot.set_qpos(ans_qpos)

    # Get the end effector target position and orientation
    ee_name = LEFT_EE_NAME if leftside else RIGHT_EE_NAME
    target_pos = kbot.data.body(ee_name).xpos.copy()
    target_ort = kbot.data.body(ee_name).xquat.copy()
    
    # Get initial states for the test
    initial_states = kbot.get_limit_center(leftside=leftside)
    
    # Create a separate robot instance for IK calculation
    kbot_for_ik = MJ_KBot(URDF_PATH, gravity_enabled=False, timestep=SIMULATION_TIMESTEP)
    
    # Optionally measure performance
    elapsed_time = None
    if measure_performance:
        start_time = time.time()
        pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
            kbot_for_ik.model, kbot_for_ik.data, target_pos, target_ort, leftside)
        elapsed_time = time.time() - start_time
    else:
        pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
            kbot_for_ik.model, kbot_for_ik.data, target_pos, target_ort, leftside)
    
    # Convert calculated arm positions to full robot positions
    calc_qpos = kbot_for_ik.convert_armqpos_to_fullqpos(pos_arm, leftside)
    
    # Set positions for test validation
    kbot.ik_test_set(target_pos, target_ort, calc_qpos, ans_qpos, initial_states)
    
    # Log results
    side_name = "Left" if leftside else "Right"
    logger.info(f"{side_name} arm position error: {error_norm_pos:.4f}, rotation error: {error_norm_rot:.4f}")
    if measure_performance:
        logger.info(f"{side_name} arm IK computation time: {elapsed_time:.4f} seconds")
    
    return error_norm_pos, error_norm_rot, elapsed_time



def test_left_edge_forward(kbot):
    """Test IK with the left arm in a forward extended position."""
    pos_err, rot_err, _ = run_ik(kbot, [2.0, -1.0, -1.0, -2.0, -1.0], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_edge_inward(kbot):
    """Test IK with the left arm in an inward position."""
    pos_err, rot_err, _ = run_ik(kbot, [-1.5, -0.3, 1.0, -0.5, -0.5], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_edge_downward(kbot):
    """Test IK with the left arm in a downward position."""
    pos_err, rot_err, _ = run_ik(kbot, [0, -1.6, 0, -2.5, 0], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_edge_extended(kbot):
    """Test IK with the left arm in a fully extended position."""
    pos_err, rot_err, _ = run_ik(kbot, [2.47, -1.57, -1.67, -2.45, -1.68], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_edge_upward(kbot):
    """Test IK with the left arm in an upward position."""
    pos_err, rot_err, _ = run_ik(kbot, [-2.0, -0.5, 1.7, -1.0, 0.5], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"


def test_left_workspace_center(kbot):
    """Test IK with the left arm at workspace center."""
    pos_err, rot_err, _ = run_ik(kbot, [0, -0.8, 0, -1.5, 0], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_front_mid(kbot):
    """Test IK with the left arm at front middle position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.3, -0.6, 0, -1.2, 0], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_front_right(kbot):
    """Test IK with the left arm at front right position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.5, -0.7, -0.2, -1.4, -0.3], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_front_left(kbot):
    """Test IK with the left arm at front left position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.5, -0.7, 0.2, -1.4, 0.3], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_front_low(kbot):
    """Test IK with the left arm at front low position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.2, -0.9, 0, -1.6, -0.7], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_angled(kbot):
    """Test IK with the left arm at angled position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.4, -1.0, -0.4, -1.8, 0.5], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_rear_high(kbot):
    """Test IK with the left arm at rear high position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.2, -0.5, 0.3, -1.3, -0.2], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_far_front(kbot):
    """Test IK with the left arm at far front position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.7, -0.8, -0.1, -2.0, 0], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_high(kbot):
    """Test IK with the left arm at high position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.3, -0.4, 0, -1.0, 0], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_workspace_twisted(kbot):
    """Test IK with the left arm in a twisted position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.5, -0.6, -0.5, -1.5, 0.5], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"


def test_right_edge_forward(kbot):
    """Test IK with the right arm in a forward extended position."""
    pos_err, rot_err, _ = run_ik(kbot, [-2.0, 1.0, 1.0, 2.0, 1.0], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_edge_inward(kbot):
    """Test IK with the right arm in an inward position."""
    pos_err, rot_err, _ = run_ik(kbot, [1.5, 0.3, -1.0, 0.5, 0.5], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_edge_downward(kbot):
    """Test IK with the right arm in a downward position."""
    pos_err, rot_err, _ = run_ik(kbot, [0, 1.6, 0, 2.5, 0], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_edge_extended(kbot):
    """Test IK with the right arm in a fully extended position."""
    pos_err, rot_err, _ = run_ik(kbot, [-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_edge_upward(kbot):
    """Test IK with the right arm in an upward position."""
    pos_err, rot_err, _ = run_ik(kbot, [2.0, 0.5, -1.7, 1.0, -0.5], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"


def test_right_workspace_front_mid(kbot):
    """Test IK with the right arm at front middle position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.3, 0.6, 0, 1.2, 0], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_front_left(kbot):
    """Test IK with the right arm at front left position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.5, 0.7, 0.2, 1.4, 0.3], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_front_right(kbot):
    """Test IK with the right arm at front right position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.5, 0.7, -0.2, 1.4, -0.3], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_front_low(kbot):
    """Test IK with the right arm at front low position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.2, 0.9, 0, 1.6, 0.7], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_rear_high(kbot):
    """Test IK with the right arm at rear high position."""
    pos_err, rot_err, _ = run_ik(kbot, [0.2, 0.5, -0.3, 1.3, 0.2], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_far_front(kbot):
    """Test IK with the right arm at far front position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.7, 0.8, 0.1, 2.0, 0], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_high(kbot):
    """Test IK with the right arm at high position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.3, 0.4, 0, 1.0, 0], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_right_workspace_twisted(kbot):
    """Test IK with the right arm in a twisted position."""
    pos_err, rot_err, _ = run_ik(kbot, [-0.5, 0.6, 0.5, 1.5, -0.5], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

#* Performance tests
def test_ik_performance_left_workspace_center(kbot):
    """Test IK solver performance with left workspace center position."""
    _, _, elapsed_time = run_ik(kbot, [0, -0.8, 0, -1.5, 0], leftside=True, measure_performance=True)
    assert elapsed_time < PERFORMANCE_THRESHOLD_SECONDS, f"Elapsed time {elapsed_time}s exceeds threshold"

def test_ik_performance_left_workspace_front_right(kbot):
    """Test IK solver performance with left workspace front right position."""
    _, _, elapsed_time = run_ik(kbot, [0.5, -0.7, -0.2, -1.4, -0.3], leftside=True, measure_performance=True)
    assert elapsed_time < PERFORMANCE_THRESHOLD_SECONDS, f"Elapsed time {elapsed_time}s exceeds threshold"

def test_ik_performance_left_edge_extended(kbot):
    """Test IK solver performance with left edge extended position."""
    _, _, elapsed_time = run_ik(kbot, [2.47, -1.57, -1.67, -2.45, -1.68], leftside=True, measure_performance=True)
    assert elapsed_time < PERFORMANCE_THRESHOLD_SECONDS, f"Elapsed time {elapsed_time}s exceeds threshold"

def test_ik_performance_right_workspace_front_mid(kbot):
    """Test IK solver performance with right workspace front mid position."""
    _, _, elapsed_time = run_ik(kbot, [-0.3, 0.6, 0, 1.2, 0], leftside=False, measure_performance=True)
    assert elapsed_time < PERFORMANCE_THRESHOLD_SECONDS, f"Elapsed time {elapsed_time}s exceeds threshold"

def test_ik_performance_right_edge_downward(kbot):
    """Test IK solver performance with right edge downward position."""
    _, _, elapsed_time = run_ik(kbot, [0, 1.6, 0, 2.5, 0], leftside=False, measure_performance=True)
    assert elapsed_time < PERFORMANCE_THRESHOLD_SECONDS, f"Elapsed time {elapsed_time}s exceeds threshold"

def test_ik_performance_right_edge_extended(kbot):
    """Test IK solver performance with right edge extended position."""
    _, _, elapsed_time = run_ik(kbot, [-2.47, 1.57, 1.67, 2.45, 1.68], leftside=False, measure_performance=True)
    assert elapsed_time < PERFORMANCE_THRESHOLD_SECONDS, f"Elapsed time {elapsed_time}s exceeds threshold"

def test_right_joint_limit(kbot):
    """Test IK with the right arm at joint limit position."""
    pos_err, rot_err, _ = run_ik(kbot, [-2.6, -0.48, -1.7, 0, -1.7], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"
    
def test_right_joint_limit2(kbot):
    """Test IK with the right arm at second joint limit position."""
    pos_err, rot_err, _ = run_ik(kbot, [2.0, 1.65, 1.7, 2.53, 1.7], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_joint_limit(kbot):
    """Test IK with the left arm at joint limit position."""
    pos_err, rot_err, _ = run_ik(kbot, [-2, -1.65, -1.74, -2.53, -1.74], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_left_joint_limit2(kbot):
    """Test IK with the left arm at second joint limit position."""
    pos_err, rot_err, _ = run_ik(kbot, [2.6, 0.48, 1.74, 0, 1.74], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_hard_test_limit(kbot):
    """Test IK with the left arm at a challenging limit position."""
    pos_err, rot_err, _ = run_ik(kbot, [2.5, -1.6, -1.7, -2.5, -1.7], leftside=True)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"

def test_hard_test_limit2(kbot):
    """Test IK with the right arm at a challenging limit position."""
    pos_err, rot_err, _ = run_ik(kbot, [-2.5, 1.6, 1.7, 2.5, 1.7], leftside=False)
    assert pos_err < POSITION_ERROR_THRESHOLD, f"Position error {pos_err} exceeds threshold"
    assert rot_err < ROTATION_ERROR_THRESHOLD, f"Rotation error {rot_err} exceeds threshold"



