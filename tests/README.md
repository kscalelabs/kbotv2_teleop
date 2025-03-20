# Robot Tests

This directory contains tests for the robot inverse kinematics (IK) and orientation functionality.

## Test Structure

The tests have been simplified into two main files:

1. `test_utils.py` - Contains test fixtures, test data, and utility functions for testing
2. `test_robot_ik.py` - Contains the actual tests for orientation error and robot IK functionality

## Running Tests

To run all tests:

```bash
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_robot_ik.py
```

To run tests with more verbose output:

```bash
pytest -v tests/
```

## Test Cases

The tests cover:

1. **Orientation Error Calculation**
   - Tests for orientation error calculation with known quaternions
   - Tests for zero error when orientations are identical
   - Tests for error magnitude with varying rotation amounts

2. **Robot IK Accuracy**
   - Tests for both left and right arms
   - Tests with multiple joint configurations
   - Verifies both position and orientation accuracy

3. **Robot Orientation Tracking**
   - Tests orientation tracking with various target quaternions
   - Uses multiple joint configurations to test different robot poses

## Adding New Tests

To add new test configurations:
1. Add new joint configurations to `TEST_JOINT_CONFIGS` in `test_utils.py`
2. Add new rotation quaternions to `TEST_ROTATIONS` in `test_utils.py`

## Threshold Configuration

Error thresholds for position and orientation accuracy are defined in `test_robot_ik.py`.
Adjust the `ERROR_THRESHOLD` value to make tests more or less strict. 