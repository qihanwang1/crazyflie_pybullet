"""CrazyFlie software-in-the-loop control example with optional QP-CBF safety filter.

Setup
-----
Step 1: Clone pycffirmware from https://github.com/utiasDSL/pycffirmware
Step 2: Follow the install instructions for pycffirmware in its README 

Example
-------
In terminal, run without CBF: 
    python gym_pybullet_drones/examples/cf_velocity_QP.py

In terminal, run with CBF (obstacles enabled):
    python gym_pybullet_drones/examples/cf_velocity_QP.py --use_cbf

"""
import time
import argparse
import numpy as np
import csv

from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CFAviary_QP import CFAviary
from gym_pybullet_drones.control.CTBRControl import CTBRControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 1000
DEFAULT_CONTROL_FREQ_HZ = 25
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_USE_CBF = True  # Re-enable CBF for testing
NUM_DRONES = 1
INIT_XYZ = np.array([[0.0, 0.0, 1.0] for i in range(NUM_DRONES)])  # Start at obstacle height (z=1.0)
INIT_RPY = np.array([[.0, .0, .0] for _ in range(NUM_DRONES)])

def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        use_cbf=DEFAULT_USE_CBF,
        ):
    #### Create the environment with or without video capture and CBF ##
    env = CFAviary(drone_model=drone,
                        num_drones=NUM_DRONES,
                        initial_xyzs=INIT_XYZ,
                        initial_rpys=INIT_RPY,
                        physics=physics,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        user_debug_gui=user_debug_gui,
                        use_cbf=use_cbf,
                        cbf_params={'gamma': 1.5, 'obs_radius': 0.1} if use_cbf else None
                        )

    # ctrl = CTBRControl(drone_model=drone)

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=NUM_DRONES,
                    output_folder=output_folder,
                    )

    #### Define trajectory that goes through obstacles ####
    # Simplified trajectory: Start at z=1.0 (obstacle height), fly straight forward
    delta = 150  # 6s @ 25hz control loop
    
    # Trajectory: Start at z=1.0, fly straight forward toward obstacles
    # No takeoff needed - already at correct height
    trajectory = [[0, 0, 1.0] for i in range(delta//6)] + \
        [[i/(delta//3)*2.0, 0, 1.0] for i in range(delta//3)] + \
        [[2.0 - i/(delta//6), 0, 1.0] for i in range(delta//6)]

    #### Define obstacles (for CBF testing) ####
    # Obstacles are defined as [position, velocity] pairs
    # Format: position = [x, y, z], velocity = [vx, vy, vz]
    import pybullet as p
    
    obstacles = []
    obstacle_visual_ids = []
    
    if use_cbf:
        # Add static obstacles much further away along the forward trajectory path
        obstacles = [
            {'pos': [1.2, 0.0, 1.0], 'vel': [0.0, 0.0, 0.0]},  # First obstacle far ahead
            {'pos': [1.7, 0.0, 1.0], 'vel': [0.0, 0.0, 0.0]},  # Second obstacle even further
        ]
        
        # Create visual markers for obstacles in PyBullet
        obs_radius = 0.1
        for obs in obstacles:
            # Create a red sphere to visualize the obstacle
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=obs_radius,
                rgbaColor=[1, 0, 0, 0.6],  # Red, semi-transparent
                physicsClientId=PYB_CLIENT
            )
            
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs['pos'],
                physicsClientId=PYB_CLIENT
            )
            obstacle_visual_ids.append(obstacle_id)
        
        print(f"\n[INFO] CBF enabled with {len(obstacles)} obstacles")
        print(f"[INFO] Obstacle positions: {[obs['pos'] for obs in obstacles]}")
        print(f"[INFO] Trajectory: Drone will fly from x=0 to x=2.0m")
        print(f"[INFO] Obstacles at x=1.2m and x=1.7m - drone should approach and avoid")
    else:
        print(f"\n[INFO] CBF disabled - running without obstacle avoidance")

    START = time.time()
    for i in range(0, len(trajectory)):
        t = i/env.ctrl_freq
        
        #### Prepare obstacle information for CBF ####
        obstacle_positions = [obs['pos'] for obs in obstacles] if obstacles else None
        obstacle_velocities = [obs['vel'] for obs in obstacles] if obstacles else None
        
        #### Step the simulation with optional CBF safety filter ###
        obs, reward, terminated, truncated, info = env.step(i, obstacle_positions, obstacle_velocities)
        
        for j in range(NUM_DRONES):
            try:
                target = trajectory[i]
                # Target already contains absolute position [x, y, z]
                pos = np.array(target)
                vel = np.zeros(3)
                acc = np.zeros(3)
                yaw = 0.0  # Keep yaw constant for straight flight
                rpy_rate = np.zeros(3)
                
                # Log position commands every 25 steps (1 second)
                if i % 25 == 0:
                    print(f"\n{'='*80}")
                    print(f"[TEST SCRIPT] Step {i}/{len(trajectory)}, time={t:.2f}s")
                    print(f"[TEST SCRIPT] Target trajectory point: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
                    print(f"[TEST SCRIPT] Commanded position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    print(f"[TEST SCRIPT] Current drone position: [{obs[j][0]:.3f}, {obs[j][1]:.3f}, {obs[j][2]:.3f}]")
                    print(f"[TEST SCRIPT] Current drone velocity: [{obs[j][10]:.3f}, {obs[j][11]:.3f}, {obs[j][12]:.3f}]")
                    if use_cbf and obstacle_positions:
                        closest_obs_dist = min([np.linalg.norm(np.array(obs_pos) - np.array([obs[j][0], obs[j][1], obs[j][2]])) 
                                               for obs_pos in obstacle_positions])
                        print(f"[TEST SCRIPT] Distance to closest obstacle: {closest_obs_dist:.3f}m")
                    print(f"{'='*80}")
                
                env.sendFullStateCmd(pos, vel, acc, yaw, rpy_rate, t)
            except:
                break

        #### Log the simulation ####################################
        for j in range(NUM_DRONES):
            logger.log(drone=j,
                        timestamp=i/env.CTRL_FREQ,
                        state=obs[j]
                        )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            pass
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("velocity_qp") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Test flight script with optional QP-CBF safety filter')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 1000)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 25)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--use_cbf',           default=DEFAULT_USE_CBF, type=str2bool,      help='Whether to use QP-CBF safety filter (default: False)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
