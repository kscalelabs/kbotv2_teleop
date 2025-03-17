# VR Teleop Requirements

1. **VR Input**
   - Track hand position from VR setup
   - Capture deltas on button press
   https://github.com/kscalelabs/gst-rs-webrtc/blob/master/robot_data_channel.py


2. **Robot Control**
   - Convert to joint commands via PyBullet IK
   - Button press for different pre-config hand grasps. Start with open/close
   - Smooth trajectories with motion planning
      https://github.com/kscalelabs/teleop-old
      https://github.com/kscalelabs/kbot-unit-tests/blob/cycle-test/kbot_cycle_tests/motion_planning_primitive.py

3. **Execution**
   - Run on KOS-Sim and real robot, like kbot-unit test
   https://github.com/kscalelabs/kbot-unit-tests/tree/cycle-test
   https://github.com/kscalelabs/gst-rs-webrtc/blob/master/robot_data_channel.py

4. **Data**
   - Store as protobuf or DataClass w/ JSON
   - Enable motion replay
   - Protobuff should have: XML used, robot used. 
   https://github.com/kscalelabs/skillit/blob/master/skillit/record/recorder.py


