import asyncio
import json
import websockets
from pykos import KOS
from dataclasses import dataclass
from .utils.logging import setup_logger
from scipy.spatial.transform import Rotation as R

from vr_teleop.utils.ik import *
from vr_teleop.mjRobot import MJ_KBot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.kosRobot import KOS_KBot


logger = setup_logger(__name__, logging.INFO)

class Controller:
    def __init__(self, urdf_path="vr_teleop/kbot_urdf/scene.mjcf"):
        self.mjRobot = MJ_KBot(urdf_path)
        self.last_ee_pos = np.array([0.01, 0.01, 0.01])

        rlocs = self.mjRobot.get_limit_center(leftside=False)

        fullq = self.mjRobot.convert_armqpos_to_fullqpos(rlocs, leftside=False)
        self.mjRobot.set_qpos(fullq)

    def step(self, cur_ee_pos):

        cur_ee_pos = np.array([0.04, 0.04, 0.04])
        qpos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
                self.mjRobot.model, 
                self.mjRobot.data, 
                cur_ee_pos, 
                target_ort=None, 
                leftside=False,
            )

        self.last_ee_pos = cur_ee_pos

        qpos_full = self.mjRobot.convert_armqpos_to_fullqpos(qpos_arm, leftside=False)

        self.mjRobot.set_qpos(qpos_full)

        return qpos_arm

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    flip_sign: bool
    joint_name: str


ACTUATORS = {
    11: Actuator(11, 1, 150.0, 8.0, 60.0, True, "left_shoulder_pitch_03"),
    12: Actuator(12, 5, 150.0, 8.0, 60.0, False, "left_shoulder_roll_03"),
    13: Actuator(13, 9, 50.0, 5.0, 17.0, False, "left_shoulder_yaw_02"),
    14: Actuator(14, 13, 50.0, 5.0, 17.0, False, "left_elbow_02"),
    15: Actuator(15, 17, 20.0, 2.0, 17.0, False, "left_wrist_02"),
    21: Actuator(21, 3, 150.0, 8.0, 60.0, False, "right_shoulder_pitch_03"),
    22: Actuator(22, 7, 150.0, 8.0, 60.0, True, "right_shoulder_roll_03"),
    23: Actuator(23, 11, 50.0, 2.0, 17.0, True, "right_shoulder_yaw_02"),
    24: Actuator(24, 15, 50.0, 5.0, 17.0, True, "right_elbow_02"),
    25: Actuator(25, 19, 20.0, 2.0, 17.0, False, "right_wrist_02"),
}


async def command_kbot(qnew_pos, kos):
    # Create a direct mapping for right arm joints
    right_arm_mapping = {
        21: 5*np.degrees(qnew_pos[0]),  # right_shoulder_pitch
        22: 5*np.degrees(qnew_pos[1]),  # right_shoulder_roll
        23: 5*np.degrees(qnew_pos[2]),  # right_shoulder_yaw
        24: 5*np.degrees(qnew_pos[3]),  # right_elbow
        25: 5*np.degrees(qnew_pos[4]),  # right_wrist
    }

    command = []
    for actuator_id in ACTUATORS:
        if actuator_id in right_arm_mapping:
            command.append({
                "actuator_id": actuator_id,
                "position": right_arm_mapping[actuator_id],
            })
    
    command_tasks = []
    
    logger.warning(f"Commanding {command}")
    command_tasks.append(kos.actuator.command_actuators(command))
    await asyncio.gather(*command_tasks)


async def websocket_handler(websocket, controller, kos):
    message = await websocket.recv()
    data = json.loads(message)
    
    if 'controller' not in data:
        return json.dumps({"status": "success"})
    
    position =  np.array(data['position'], dtype=np.float64)

    qnew_pos = controller.step(position)
    await command_kbot(qnew_pos, kos)

    return json.dumps({"status": "success"})


async def main():
    """Start KOS and then WebSocket server."""
    try:
        # Initialize KOS connection once
        async with KOS(ip="10.33.12.161", port=50051) as kos:
            controller = Controller()
            try:
                
                async def handle_connection(websocket):
                    await websocket_handler(websocket, controller, kos)


                enable_commands = []
                for cur_act in ACTUATORS.keys():
                    enable_commands.append(
                            kos.actuator.configure_actuator(
                        actuator_id=cur_act,
                        kp=ACTUATORS[cur_act].kp,
                        kd=ACTUATORS[cur_act].kd,
                        torque_enabled=True)
                )
                logger.warning(f"Enabling {enable_commands}")
                await asyncio.gather(*enable_commands)
                await asyncio.sleep(1)
                            
                
                server = await websockets.serve(handle_connection, "localhost", 8586)
                logger.info("WebSocket server started at ws://localhost:8586")
                
                try:
                    await server.wait_closed()
                except asyncio.CancelledError:
                    logger.info("Server shutdown requested")
                finally:
                    pass

            except Exception as e:
                logger.error(f"Error during initialization: {e}")
    except Exception as e:
        logger.error(f"Error connecting to KOS: {e}")

if __name__ == "__main__":
    asyncio.run(main())





#* Initialize KOS. 
    ##* Initialize websocket client once KOS has been started. 


#! Event listener for websocket messages
#* Get data, get delta (compare with previous data), get IK, send joint commands. 




#* Websocket gets data -> Puts in queue. (later) Two queues, one for each controller?
#* Task wakes up when there is something in queue. 











