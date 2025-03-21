import asyncio
import json
import logging
import websockets
import time
import numpy as np
from pykos import KOS
from vr_teleop.ikrobot import KBot_Robot
from vr_teleop.utils.motion_planning import Robot_Planner
from vr_teleop.utils.ik import inverse_kinematics
from .utils.logging import setup_logger

logger = setup_logger(__name__)
logger.setLevel(logging.DEBUG)

class CommandQueue:
    """Queue for passing commands between WebSocket and robot control"""
    def __init__(self):
        # Queue for controller commands
        self.command_queue = asyncio.Queue()
        # Queue for robot results
        self.result_queue = asyncio.Queue()

async def websocket_producer(command_queue, host="localhost", port=8586):
    """WebSocket server that receives VR controller data and produces commands"""
    
    async def handle_client(websocket):
        """Process incoming WebSocket messages"""
        logger.info("[WEBSOCKET] Client connected")
        
        try:
            async for message in websocket:
                try:
                    # Parse the message
                    data = json.loads(message)
                    
                    if 'controller' in data:
                        controller = data['controller']
                        
                        # Log the data
                        if 'position' in data:
                            logger.info(f"[WEBSOCKET] {controller.upper()} position: {data['position']}")
                        
                        # Check for button press on left controller
                        if (controller == 'left' and 'buttons' in data and 
                            data['buttons'] and data['buttons'][0].get('pressed', False)):
                            
                            # Create a movement command from button press + position
                            command = {
                                'type': 'movement',
                                'controller': controller,
                                'position': data.get('position'),
                                'buttons': data.get('buttons'),
                                'timestamp': time.time()
                            }
                            
                            # Add the command to the queue
                            logger.info(f"[WEBSOCKET] Adding movement command to queue")
                            await command_queue.command_queue.put(command)
                        
                        # Send confirmation back to client
                        await websocket.send(json.dumps({"status": "success"}))
                    
                except json.JSONDecodeError as e:
                    logger.error(f"[WEBSOCKET] JSON decode error: {e}")
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"[WEBSOCKET] Error processing message: {e}")
                    await websocket.send(json.dumps({"status": "error", "message": str(e)}))
        
        except websockets.exceptions.ConnectionClosedError as e:
            # This is normal when the client disconnects
            logger.info(f"[WEBSOCKET] Client disconnected: {e}")
        except websockets.exceptions.ConnectionClosedOK as e:
            # This happens with a clean disconnect
            logger.info(f"[WEBSOCKET] Client disconnected cleanly: {e}")
        except Exception as e:
            # This catches any other unexpected errors
            logger.error(f"[WEBSOCKET] Error in WebSocket connection: {e}")
    
    # Start the WebSocket server
    logger.info(f"[WEBSOCKET] Starting WebSocket server on {host}:{port}")
    server = await websockets.serve(
        handle_client, 
        host, 
        port,
        ping_interval=20,    # Send a ping every 20 seconds
        ping_timeout=10      # Wait 10 seconds for a pong response
    )
    
    # Keep the server running until the program exits
    await asyncio.Future()


async def robot_consumer(command_queue, simulation_mode=True):
    """Consumer that processes commands from the queue and controls the robot"""
    logger.info("[ROBOT] Starting robot control consumer")
    
    # Initialize the robot
    try:
        # Setup robot and motion planner
        logger.info("[ROBOT] Initializing robot and motion planner")
        urdf_path = "vr_teleop/kbot_urdf/scene.mjcf"
        solver = KBot_Robot(urdf_path)
        motion_plan = Robot_Planner(solver)
        
        # Get initial positions
        llocs = solver.get_limit_center(leftside=True)
        rlocs = solver.get_limit_center(leftside=False)
        
        # Set initial state
        fullq = motion_plan.arms_tofullq(llocs, rlocs)
        motion_plan.set_curangles(solver.data.qpos)
        motion_plan.set_nextangles(fullq)
        
        # Map joints to indices
        idx_to_joint_map = solver.qpos_idx_to_jointname()
        motion_plan.set_idx_joint_map(idx_to_joint_map)
        
        # Connect to KOS if not in simulation mode
        kos_instance = None
        if not simulation_mode:
            logger.info("[ROBOT] Connecting to KOS...")
            kos_instance = KOS(ip="10.33.12.161", port=50051)
            await kos_instance.connect()
            
            # Activate robot
            logger.info("[ROBOT] Activating robot actuators")
            # Disable all actuators first
            disable_commands = []
            for cur_act in motion_plan.sim_act_list.keys():
                disable_commands.append(
                    kos_instance.actuator.configure_actuator(actuator_id=cur_act, torque_enabled=False)
                )
            await asyncio.gather(*disable_commands)
            await asyncio.sleep(1)
            
            # Then configure and enable them
            config_commands = []
            for cur_act in motion_plan.sim_act_list.values():
                config_commands.append(
                    kos_instance.actuator.configure_actuator(
                        actuator_id=cur_act.actuator_id,
                        kp=cur_act.kp,
                        kd=cur_act.kd,
                        torque_enabled=True,
                    )
                )
            await asyncio.gather(*config_commands)
            await asyncio.sleep(1)
            logger.info("[ROBOT] Robot actuators activated")
        
        # Main processing loop
        logger.info("[ROBOT] Entering main processing loop")
        robot_busy = False
        
        while True:
            if not robot_busy:
                # Get the next command from the queue (if available)
                try:
                    command = await asyncio.wait_for(command_queue.command_queue.get(), timeout=0.1)
                    
                    if command['type'] == 'movement':
                        robot_busy = True
                        logger.info(f"[ROBOT] Processing movement command: {command}")
                        
                        # Calculate a delta movement (simplified for demo)
                        position = command.get('position', [0, 0, 0])
                        position_delta = [4, 10, 2]  # Static delta for demo
                        logger.info(f"[ROBOT] Using position delta: {position_delta}")
                        
                        try:
                            # Get current end effector position and orientation
                            lee_pos, lee_ort = solver.get_ee_pos(leftside=True)
                            
                            # Apply the delta to get new target position
                            new_pos, new_quat = motion_plan.apply_pose_delta(
                                lee_pos, lee_ort, position_delta, [0, 0, 0])
                            
                            # Run inverse kinematics to get joint angles
                            pos_arm, error_norm_pos, error_norm_rot = inverse_kinematics(
                                solver.model, solver.data, new_pos, new_quat, leftside=True)
                            
                            # Get full joint configuration
                            nextqpos = motion_plan.arms_tofullq(
                                leftarmq=pos_arm, 
                                rightarmq=solver.get_limit_center(leftside=False))
                            
                            # Set next target and generate waypoints
                            motion_plan.set_curangles(solver.data.qpos)
                            motion_plan.set_nextangles(nextqpos)
                            planned_angles, _, time_grid = motion_plan.get_waypoints()
                            
                            # Execute the motion (or simulate it)
                            if simulation_mode:
                                # Simulate movement
                                logger.info("[ROBOT] Simulating movement...")
                                await execute_motion(kos_instance, motion_plan, planned_angles, time_grid)
                                await asyncio.sleep(2)  # Just sleep for simulation
                                logger.info("[ROBOT] Movement simulation complete")
                            else:
                                # Execute with real robot
                                logger.info("[ROBOT] Executing movement on robot...")
                                await execute_motion(kos_instance, motion_plan, planned_angles, time_grid)
                                logger.info("[ROBOT] Movement execution complete")
                            
                            # Send result back
                            result = {
                                "status": "success",
                                "message": "Movement completed",
                                "command_time": command['timestamp']
                            }
                            await command_queue.result_queue.put(result)
                            
                        except Exception as e:
                            logger.error(f"[ROBOT] Error executing movement: {e}")
                            # Send error result back
                            error_result = {
                                "status": "error",
                                "message": str(e),
                                "command_time": command['timestamp']
                            }
                            await command_queue.result_queue.put(error_result)
                        
                        robot_busy = False
                
                except asyncio.TimeoutError:
                    # No commands available, just continue the loop
                    pass
                
            # Sleep to prevent CPU spinning
            await asyncio.sleep(0.01)
    
    except Exception as e:
        logger.error(f"[ROBOT] Fatal error in robot consumer: {e}")
        # Try to clean up
        if kos_instance and not simulation_mode:
            try:
                logger.info("[ROBOT] Disabling robot actuators")
                disable_commands = []
                for cur_act in motion_plan.sim_act_list.keys():
                    disable_commands.append(
                        kos_instance.actuator.configure_actuator(actuator_id=cur_act, torque_enabled=False)
                    )
                await asyncio.gather(*disable_commands)
            except:
                pass


async def execute_motion(kos_instance, motion_plan, planned_angles, time_grid):
    """Execute planned motion on the robot"""
    logger.info("[MOTION] Starting execution of planned motion")
    all_angles = np.degrees(planned_angles)
    start_time = time.time()

    all_actuators_ids = list(motion_plan.sim_act_list.keys())
    
    # Map actuator IDs to joint indices
    actuator_to_joint_idx = {}
    for actuator_id in all_actuators_ids:
        joint_name = motion_plan.sim_act_list[actuator_id].joint_name
        for idx, joint in motion_plan.idx_to_joint_map.items():
            if joint == joint_name:
                actuator_to_joint_idx[actuator_id] = idx
                break
    
    # Execute each waypoint
    for step, t in enumerate(time_grid):
        current_time = time.time()
        
        if step % 5 == 0 or step == len(time_grid) - 1:  # Log periodically
            logger.info(f"[MOTION] Executing waypoint {step+1}/{len(time_grid)}")
        
        command = []
        for actuator_id in all_actuators_ids:
            if actuator_id in actuator_to_joint_idx:
                joint_idx = actuator_to_joint_idx[actuator_id]
                command.append({
                    "actuator_id": actuator_id,
                    "position": all_angles[joint_idx][step],
                })
        
        command_tasks = []
        command_tasks.append(kos_instance.actuator.command_actuators(command))
        await asyncio.gather(*command_tasks)
        
        next_time = start_time + t
        if next_time > current_time:
            await asyncio.sleep(next_time - current_time)


async def result_processor(command_queue):
    """Process results from the robot control loop"""
    logger.info("[RESULTS] Starting result processor")
    
    while True:
        # Get the next result
        result = await command_queue.result_queue.get()
        logger.info(f"[RESULTS] Processed result: {result}")
        
        # Sleep briefly
        await asyncio.sleep(0.1)


async def main():
    """Main function to run the producer-consumer system"""
    logger.info("[MAIN] Starting Producer-Consumer VR Robot Control System")
    
    # Create command queue
    command_queue = CommandQueue()
    
    # Set simulation mode (True for testing without real robot)
    simulation_mode = True
    
    try:
        # Create the three main tasks
        producer_task = asyncio.create_task(
            websocket_producer(command_queue))
        consumer_task = asyncio.create_task(
            robot_consumer(command_queue, simulation_mode))
        result_task = asyncio.create_task(
            result_processor(command_queue))
        
        # Wait for all tasks (they should run indefinitely)
        await asyncio.gather(producer_task, consumer_task, result_task)
        
    except KeyboardInterrupt:
        logger.info("[MAIN] Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"[MAIN] Error in main: {e}")
    finally:
        logger.info("[MAIN] Shutting down...")


if __name__ == "__main__":
    asyncio.run(main()) 