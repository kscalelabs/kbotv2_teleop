# VR Teleop Requirements

1. **VR Input**
   - Track hand position from VR setup
   - Capture deltas on button press
   
   https://github.com/kscalelabs/gst-rs-webrtc/blob/master/robot_data_channel.py


2. **Robot Control**
   - Convert to joint commands via PyBullet IK.
   - Button press to control hand for different pre-config hand grasps. Start with open/close
    
    https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics_husky_kuka.py#L67
    https://github.com/kscalelabs/teleop-old

3. **Execution**
   - Run on KOS-Sim and real robot, like kbot-unit test
   - Smooth trajectories
   
    https://github.com/kscalelabs/kbot-unit-tests/tree/cycle-test
    https://github.com/kscalelabs/kbot-unit-tests/blob/cycle-test/kbot_cycle_tests/motion_planning_primitive.py

4. **Data**
   - Store as HDF5. Integrating video from KOS when that is ready
   - Enable motion replay with smooth motion planning

    https://github.com/tonyzhaozh/aloha/blob/main/aloha_scripts/record_episodes.py
    https://github.com/kscalelabs/skillit/blob/master/skillit/record/recorder.py


*decisions*

- robot client only accepts one instance of kos. Would be running in sim and real sequentially rather than same time. This is because have had issues (in kbot unit test) with latency when running sim w/ render and real at same time/
- 

