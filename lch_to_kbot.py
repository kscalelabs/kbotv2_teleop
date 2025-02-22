"""E2E example of recording a teleop session using the LCH.

Run:
    python -m kdatagen.teleop.record_example_lch --sim
"""

import argparse
import logging
import time

from kos_protos.process_manager_pb2 import KClipStartRequest

from kdatagen.teleop.robots import GPR, LCH, MujocoGPR

if __name__ == "__main__":
    # Add argument parser
    parser = argparse.ArgumentParser(description="Record teleop session using LCH.")
    parser.add_argument("--real", action="store_true", help="Run on real robot")
    parser.add_argument("--sim", action="store_true", help="Run in simulation")
    args = parser.parse_args()

    # Configure the logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    lch = LCH(robot_id=1, logger=logger)

    if args.real:
        gpr = GPR(robot_id=0)
        task_command = input("What's the task instruction? ")
        # Start video recording
        request = KClipStartRequest(action=task_command)
        video_response = gpr.start_kclip(request)
        logger.info("Started video recording with UUID: %s", video_response[0])

    logger.info("Starting teleop recording. Press Ctrl+C to stop.")

    # Store initial angles as offset
    initial_angles = lch.get_joint_pos()
    logger.info("Initial angles: %s", initial_angles)

    if args.sim:
        sim_robot = MujocoGPR(robot_id=1, embodiment="gpr", render=True)

    try:
        while True:
            angles = lch.get_joint_pos()

            # Subtract initial angles to get relative movement
            relative_angles = {k: angles[k] - initial_angles[k] for k in angles.keys()}

            if args.real:
                commands = []
                for i, joint in enumerate(lch.joint2id.keys()):
                    # Hack
                    if joint=="right_wrist_roll":
                        commands.append(
                            {"actuator_id": gpr.joint2id[joint], "position": 0}
                        )
                    else:
                        commands.append(
                            {"actuator_id": gpr.joint2id[joint], "position": relative_angles[lch.joint2id[joint]]}
                        )
                gpr.command_actuators(commands)

            if args.sim:
                sim_robot.step(relative_angles)

            # Sleep to make the simulation more stable
            time.sleep(0.025)

    except KeyboardInterrupt:
        logger.info("Stopping recording...")
    finally:
        # Stop video recording and extract frames
        response = gpr.stop_kclip()
        logger.info("Saved teleop recording: %s", response[0])