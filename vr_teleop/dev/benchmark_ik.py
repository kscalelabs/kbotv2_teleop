import mujoco
import numpy as np
import time
import os
from vr_teleop.utils.ik import inverse_kinematics, forward_kinematics

def benchmark_mj_forward_in_ik(model_path=None, num_iterations=50):
    """
    Benchmark the execution time of mujoco.mj_forward and mujoco.mj_jac functions when used within the IK solver
    
    Args:
        model_path: Path to the MuJoCo model file (MJCF or XML)
        num_iterations: Number of iterations to run the benchmark
    """
    # Load MuJoCo model
    if model_path is None:
        # Try to find model in common locations
        possible_paths = [
            "robot.xml",
            "kbotv2.xml",
            "model/robot.xml",
            "model/kbotv2.xml",
            "robot.mjcf",
            "kbotv2.mjcf",
            "model/robot.mjcf",
            "model/kbotv2.mjcf",
            "vr_teleop/kbot_urdf/scene.mjcf",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/robot.mjcf"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "model/robot.xml"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError("Could not find MuJoCo model file. Please specify the path.")
    
    print(f"Loading model from {model_path}")
    
    # Load model based on file extension
    if model_path.endswith('.mjcf') or model_path.endswith('.xml'):
        model = mujoco.MjModel.from_xml_path(model_path)
    else:
        raise ValueError(f"Unsupported file format for {model_path}, expected .mjcf or .xml")
        
    data = mujoco.MjData(model)
    
    # Setup target positions to solve for
    num_targets = 5
    np.random.seed(42)  # For reproducibility
    
    # Create several random targets within a reasonable workspace
    # These are just example values - adjust based on your robot's workspace
    target_positions = []
    target_orientations = []
    
    # Right arm targets (example positions that might be reachable)
    for i in range(num_targets):
        # Adjust these ranges based on your robot's workspace
        pos = np.array([
            np.random.uniform(0.3, 0.5),    # x - forward
            np.random.uniform(-0.3, -0.1),  # y - right side
            np.random.uniform(0.3, 0.5)     # z - up
        ])
        
        # Random orientation (simplified)
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        target_positions.append(pos)
        target_orientations.append(quat)
    
    # Counter for mj_forward calls
    forward_calls = 0
    forward_time = 0
    
    # Counter for mj_jac calls
    jac_calls = 0
    jac_time = 0
    
    # Patch mj_forward to track calls and timing
    original_mj_forward = mujoco.mj_forward
    
    def tracked_mj_forward(model, data):
        nonlocal forward_calls, forward_time
        forward_calls += 1
        start = time.time()
        result = original_mj_forward(model, data)
        end = time.time()
        forward_time += (end - start) * 1000  # Convert to milliseconds
        return result
    
    # Patch mj_jac to track calls and timing
    original_mj_jac = mujoco.mj_jac
    
    def tracked_mj_jac(model, data, jacp, jacr, point, body):
        nonlocal jac_calls, jac_time
        jac_calls += 1
        start = time.time()
        result = original_mj_jac(model, data, jacp, jacr, point, body)
        end = time.time()
        jac_time += (end - start) * 1000  # Convert to milliseconds
        return result
    
    # Replace the functions temporarily
    mujoco.mj_forward = tracked_mj_forward
    mujoco.mj_jac = tracked_mj_jac
    
    try:
        # Benchmark IK with tracking
        total_ik_time = 0
        ik_times = []
        
        # Test for right arm
        print(f"Benchmarking IK for right arm with {num_iterations} iterations per target...")
        for i, (pos, quat) in enumerate(zip(target_positions, target_orientations)):
            print(f"Target {i+1}/{num_targets}: {pos}")
            
            # Reset counters for this target
            target_forward_calls = 0
            target_forward_time = 0
            target_jac_calls = 0
            target_jac_time = 0
            
            for j in range(num_iterations):
                # Reset counters for this iteration
                iter_start_forward_calls = forward_calls
                iter_start_forward_time = forward_time
                iter_start_jac_calls = jac_calls
                iter_start_jac_time = jac_time
                
                # Run IK
                start_time = time.time()
                joint_angles, pos_error, rot_error = inverse_kinematics(
                    model, data, pos, quat, leftside=False, debug=False
                )
                end_time = time.time()
                
                # Calculate statistics
                iter_time = (end_time - start_time) * 1000  # ms
                iter_forward_calls = forward_calls - iter_start_forward_calls
                iter_forward_time = forward_time - iter_start_forward_time
                iter_jac_calls = jac_calls - iter_start_jac_calls
                iter_jac_time = jac_time - iter_start_jac_time
                
                # Update totals
                total_ik_time += iter_time
                ik_times.append(iter_time)
                target_forward_calls += iter_forward_calls
                target_forward_time += iter_forward_time
                target_jac_calls += iter_jac_calls
                target_jac_time += iter_jac_time
                
                if (j + 1) % 10 == 0:
                    print(f"  Completed {j+1}/{num_iterations} iterations")
            
            # Print statistics for this target
            avg_forward_time = target_forward_time / target_forward_calls if target_forward_calls > 0 else 0
            avg_jac_time = target_jac_time / target_jac_calls if target_jac_calls > 0 else 0
            
            print(f"  Target {i+1} statistics:")
            print(f"    Total forward calls: {target_forward_calls}")
            print(f"    Forward calls per IK solve: {target_forward_calls / num_iterations:.1f}")
            print(f"    Average time per forward call: {avg_forward_time:.4f} ms")
            print(f"    Total forward time: {target_forward_time:.2f} ms")
            print(f"    Total jacobian calls: {target_jac_calls}")
            print(f"    Jacobian calls per IK solve: {target_jac_calls / num_iterations:.1f}")
            print(f"    Average time per jacobian call: {avg_jac_time:.4f} ms")
            print(f"    Total jacobian time: {target_jac_time:.2f} ms")
            print(f"    Position error: {pos_error:.6f}, Rotation error: {rot_error:.6f}")
        
        # Overall statistics
        avg_ik_time = total_ik_time / (num_targets * num_iterations)
        avg_forward_time = forward_time / forward_calls if forward_calls > 0 else 0
        avg_forward_calls = forward_calls / (num_targets * num_iterations)
        avg_jac_time = jac_time / jac_calls if jac_calls > 0 else 0
        avg_jac_calls = jac_calls / (num_targets * num_iterations)
        
        print("\nOverall Results:")
        print(f"Total IK solves: {num_targets * num_iterations}")
        print(f"Average IK solve time: {avg_ik_time:.4f} ms")
        
        print(f"\nmj_forward statistics:")
        print(f"Total mj_forward calls: {forward_calls}")
        print(f"Average mj_forward calls per IK solve: {avg_forward_calls:.1f}")
        print(f"Total mj_forward time: {forward_time:.2f} ms")
        print(f"Average time per mj_forward call: {avg_forward_time:.4f} ms")
        print(f"mj_forward percentage of total time: {(forward_time / total_ik_time) * 100:.1f}%")
        
        print(f"\nmj_jac statistics:")
        print(f"Total mj_jac calls: {jac_calls}")
        print(f"Average mj_jac calls per IK solve: {avg_jac_calls:.1f}")
        print(f"Total mj_jac time: {jac_time:.2f} ms")
        print(f"Average time per mj_jac call: {avg_jac_time:.4f} ms")
        print(f"mj_jac percentage of total time: {(jac_time / total_ik_time) * 100:.1f}%")
        
        return avg_ik_time, avg_forward_time, avg_forward_calls, avg_jac_time, avg_jac_calls
    
    finally:
        # Restore the original functions
        mujoco.mj_forward = original_mj_forward
        mujoco.mj_jac = original_mj_jac

if __name__ == "__main__":
    print("Benchmarking mujoco.mj_forward function in IK context")
    
    # Try to find the model path
    model_path = None
    possible_paths = [
        "robot.xml",
        "kbotv2.xml",
        "model/robot.xml",
        "model/kbotv2.xml",
        "robot.mjcf",
        "kbotv2.mjcf",
        "model/robot.mjcf",
        "model/kbotv2.mjcf",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("Could not find MuJoCo model file automatically.")
        model_path = input("Please enter the path to the MuJoCo model file (MJCF or XML): ")
    
    # Run benchmark
    benchmark_mj_forward_in_ik(model_path, num_iterations=20) 