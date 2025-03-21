import asyncio
from pykos import KOS
import websockets
import time
import json
from .vr_cont import process_message
from .utils.logging import setup_logger
from .robot_cont import enable_actuators, disable_actuators
from .utils.actuator_list import ACTUATOR_LIST_SIM, ACTUATOR_LIST_REAL


logger = setup_logger(__name__)

async def control_loop(websocket, kos_instance: KOS, real_robot: bool):
    """Handle incoming WebSocket connections."""
    actuator_list = ACTUATOR_LIST_REAL if real_robot else ACTUATOR_LIST_SIM

    await kos_instance.sim.reset(initial_state={"qpos": [0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0] + [0.0] * 20})

    await disable_actuators(kos_instance, actuator_list)
    await enable_actuators(kos_instance, actuator_list)

    try:
        async for message in websocket:
            logger.info("Received connection")
            
            # Process the message
            # end_eff_locs = await process_message(message)
            end_eff_locs = [0]

            await kbot_control_loop(kos_instance, actuator_list, end_eff_locs)
            
            # Send response back
            await websocket.send(json.dumps({"status": "success"}))
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
        await disable_actuators(kos_instance, actuator_list)
    except Exception as e:
        logger.error(f"Error handling connection: {e}")
        await disable_actuators(kos_instance, actuator_list)


    await disable_actuators(kos_instance, actuator_list)


async def kbot_control_loop(kos_instance, actuator_list, end_eff_locs, duration=100):
    start_time = time.time()
    next_time = start_time + 1 / 100  # 100Hz control rate

    while time.time() - start_time < duration:
        current_time = time.time()

        #* ---- Main ---- *
        state_response = await kos_instance.actuator.get_actuators_state(actuator_list.keys())
        current_joint_states = [state.position for state in state_response.states]

        # next_joint_states = inverse_kinematics(arm_chain, end_eff_locs)

        # planned_commands = motion_planning(current_joint_states, next_joint_states)

        # command_tasks = []
        # command_tasks.append(kos_instance.actuator.command_actuators(planned_commands))


        #* ---- End ---- *

        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)
        next_time += 1 / 100


async def handle_connection(websocket):
    """Handle incoming WebSocket connections."""

    async with KOS(ip="localhost", port=50051) as sim_kos:
        await control_loop(websocket, sim_kos, False)



async def main():
    """Start the WebSocket server."""
    server = await websockets.serve(
        handle_connection,
        "localhost",
        8586
    )
    logger.info("WebSocket server started on ws://localhost:8586")
    
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())

