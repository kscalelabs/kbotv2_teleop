# kbotv2_teleop

`pip install -e .`
`mjpython -m vr_teleop.example`

To run tests
`pytest tests/test_robot_ik.py`

To run an individual test
`pytest tests/test_robot_ik.py::test_hard_test_limit2 -v --no-header`
`pytest tests/test_robot_ik.py::test_left_edge_downward -v --no-header`
`pytest tests/test_robot_ik.py::test_ik_performance_left_workspace_front_right`
