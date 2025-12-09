"""CrazyFlie software-in-the-loop control example with optional MPC safety filter.

Setup
-----
Step 1: Clone pycffirmware from https://github.com/utiasDSL/pycffirmware
Step 2: Follow the install instructions for pycffirmware in its README 

Example
-------
In terminal, run without MPC: 
    python gym_pybullet_drones/examples/cf_velocity_MPC.py

In terminal, run with MPC (obstacles enabled):
    python gym_pybullet_drones/examples/cf_velocity_MPC.py --use_mpc

Notes
-----
- This script mirrors the structure of the QP-CBF example but uses an MPC layer
  to filter velocity commands for safety / obstacle avoidance.
- The underlying Crazyflie firmware (Mellinger controller, etc.) is left unchanged.
"""

import time
import argparse
import numpy as np
import csv
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CFAviary_MPC import CFAviary_MPC
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

################################################################################
# Defaults
################################################################################

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 1000
DEFAULT_CONTROL_FREQ_HZ = 25
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_USE_MPC = True

# Simple default MPC parameters (tune as needed)
DEFAULT_MPC_PARAMS = {
    'horizon': 10,
    'dt': 1.0 / DEFAULT_CONTROL_FREQ_HZ,  # outer loop dt by default
    'max_vel': 0.5,          # m/s
    'max_yaw_rate': 1.0,     # rad/s
    'safety_margin': 0.20,   # m
    'slack_weight': 50.0,
}

NUM_DRONES = 1

# Start slightly above ground
INIT_XYZ = np.array([[0.0, 0.0, 0.1] for _ in range(NUM_DRONES)])
INIT_RPY = np.array([[0.0, 0.0, 0.0] for _ in range(NUM_DRONES)])

################################################################################

def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        use_mpc=DEFAULT_USE_MPC,
        mpc_horizon=DEFAULT_MPC_PARAMS['horizon'],
        mpc_max_vel=DEFAULT_MPC_PARAMS['max_vel'],
        mpc_max_yaw_rate=DEFAULT_MPC_PARAMS['max_yaw_rate'],
        mpc_safety_margin=DEFAULT_MPC_PARAMS['safety_margin']
        ):
    """Main entry point for the MPC test flight script.
    """

    ############################################################################
    # Create environment
    ############################################################################

    mpc_params = {
        'horizon': mpc_horizon,
        'dt': 1.0 / control_freq_hz,   # outer loop dt
        'max_vel': mpc_max_vel,
        'max_yaw_rate': mpc_max_yaw_rate,
        'safety_margin': mpc_safety_margin,
    }

    env = CFAviary_MPC(
        drone_model=drone,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=physics,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,      # external control frequency
        gui=gui,
        user_debug_gui=user_debug_gui,
        record=False,
        obstacles=False,
        output_folder=output_folder,
        use_mpc=use_mpc,
        mpc_params=mpc_params,
    )

    # PyBullet client id
    PYB_CLIENT = env.getPyBulletClient()

    ############################################################################
    # Logger
    ############################################################################

    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=NUM_DRONES,
        output_folder=output_folder,
    )

    ############################################################################
    # Define reference trajectory (position + nominal velocity)
    ############################################################################

    # Simulation duration (in seconds) and corresponding steps
    SIM_DURATION_SEC = 8.0
    num_steps = int(SIM_DURATION_SEC * control_freq_hz)
    dt = 1.0 / control_freq_hz

    # Simple reference:
    # t in [0, 1): takeoff to z = 1.0 at x = 0
    # t in [1, 5): move forward along +x at constant velocity
    # t > 5: hold position at x = 2.0
    v_forward = 0.5  # m/s
    x_final = 2.0

    def reference(t):
        """Return (pos, vel, acc, yaw) reference at time t."""
        if t < 1.0:
            # takeoff and hover at (0,0,1)
            pos = np.array([0.0, 0.0, 1.0])
            vel = np.array([0.0, 0.0, 0.0])
        elif t < 5.0:
            # move along x with constant velocity
            x = min(v_forward * (t - 1.0), x_final)
            pos = np.array([x, 0.0, 1.0])
            vel = np.array([v_forward if x < x_final else 0.0, 0.0, 0.0])
        else:
            # hold at final position
            pos = np.array([x_final, 0.0, 1.0])
            vel = np.array([0.0, 0.0, 0.0])

        acc = np.zeros(3)
        yaw = 0.0
        return pos, vel, acc, yaw

    ############################################################################
    # Define obstacles for MPC testing
    ############################################################################

    obstacles = []
    obstacle_visual_ids = []

    if use_mpc:
        # Two static obstacles along the forward path
        # (same style as your QP example but you can move them closer/further)
        obstacles = [
            {'pos': [1.2, 0.0, 1.0], 'vel': [0.0, 0.0, 0.0]},
            {'pos': [1.7, 0.0, 1.0], 'vel': [0.0, 0.0, 0.0]},
        ]

        obs_radius = 0.1

        for obs in obstacles:
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=obs_radius,
                rgbaColor=[1.0, 0.0, 0.0, 0.6],
                physicsClientId=PYB_CLIENT
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=obs['pos'],
                physicsClientId=PYB_CLIENT
            )
            obstacle_visual_ids.append(obstacle_id)

        print(f"\n[INFO] MPC enabled with {len(obstacles)} obstacles")
        print(f"[INFO] Obstacle positions: {[obs['pos'] for obs in obstacles]}")
        print(f"[INFO] Reference: move from x=0 to x={x_final} m at z=1.0")
    else:
        print(f"\n[INFO] MPC disabled - running without MPC safety layer")

    ############################################################################
    # Main simulation loop
    ############################################################################

    START = time.time()

    # Optionally: send a high-level takeoff command once at the beginning
    # (this interacts with the firmware high-level commander; not required if
    # you are happy to drive with full-state commands only)
    env.sendTakeoffCmd(height=1.0, duration=2.0)

    for i in range(num_steps):
        t = i / control_freq_hz

        # Build obstacle arrays for this step
        obstacle_positions = [obs['pos'] for obs in obstacles] if obstacles else None
        obstacle_velocities = [obs['vel'] for obs in obstacles] if obstacles else None

        # Advance environment; MPC layer is called internally if enabled
        obs, reward, terminated, truncated, info = env.step(
            i,
            obstacle_positions=obstacle_positions,
            obstacle_velocities=obstacle_velocities
        )

        # Send reference command (position + nominal velocity)
        for j in range(NUM_DRONES):
            pos_ref, vel_ref, acc_ref, yaw_ref = reference(t)
            rpy_rate = np.zeros(3)

            # Log reference vs. state occasionally
            if i % control_freq_hz == 0:  # approx once per second
                print("\n" + "=" * 80)
                print(f"[MPC TEST] Step {i}/{num_steps}, t={t:.2f}s")
                print(f"[MPC TEST] Reference pos:  [{pos_ref[0]:.3f}, {pos_ref[1]:.3f}, {pos_ref[2]:.3f}]")
                print(f"[MPC TEST] Reference vel:  [{vel_ref[0]:.3f}, {vel_ref[1]:.3f}, {vel_ref[2]:.3f}]")
                print(f"[MPC TEST] Current pos:    [{obs[j][0]:.3f}, {obs[j][1]:.3f}, {obs[j][2]:.3f}]")
                print(f"[MPC TEST] Current vel:    [{obs[j][10]:.3f}, {obs[j][11]:.3f}, {obs[j][12]:.3f}]")
                if use_mpc and obstacle_positions:
                    dists = [
                        np.linalg.norm(np.array(obs_pos) - np.array([obs[j][0], obs[j][1], obs[j][2]]))
                        for obs_pos in obstacle_positions
                    ]
                    print(f"[MPC TEST] Distances to obstacles: {['{:.3f}'.format(d) for d in dists]}")
                print("=" * 80)

            # This enqueues the full state command; it will be picked up
            # by the firmware wrapper at the next inner loop
            env.sendFullStateCmd(
                pos=pos_ref,
                vel=vel_ref,
                acc=acc_ref,
                yaw=yaw_ref,
                rpy_rate=rpy_rate,
                timestep=t
            )

        # Log at outer control rate
        for j in range(NUM_DRONES):
            logger.log(
                drone=j,
                timestamp=t,
                state=obs[j]
            )

        # Render in GUI
        env.render()

        # Sync with real time for nicer visualization
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    ############################################################################
    # Landing and cleanup
    ############################################################################

    # env.sendLandCmd(height=0.0, duration=2.0)
    # # Run a few extra steps to let landing command execute
    # for k in range(int(2.0 * control_freq_hz)):
    #     i = num_steps + k
    #     t = i / control_freq_hz
    #     obs, reward, terminated, truncated, info = env.step(i)
    #     env.render()
    #     if gui:
    #         sync(i, START, env.CTRL_TIMESTEP)

    env.close()

    logger.save()
    logger.save_as_csv("velocity_mpc")  # Optional CSV save

    if plot:
        logger.plot()


################################################################################
# CLI
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test flight script with optional MPC safety filter')

    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,    type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,        type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 1000)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,   type=int,
                        help='Control frequency in Hz (default: 25)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,     type=str,
                        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--use_mpc',            default=DEFAULT_USE_MPC,   type=str2bool,
                        help='Whether to use MPC safety filter (default: True)', metavar='')

    # MPC-specific CLI parameters (optional)
    parser.add_argument('--mpc_horizon',        default=DEFAULT_MPC_PARAMS['horizon'], type=int,
                        help='MPC prediction horizon (default: 10)', metavar='')
    parser.add_argument('--mpc_max_vel',        default=DEFAULT_MPC_PARAMS['max_vel'], type=float,
                        help='Max velocity magnitude used in MPC (default: 0.5 m/s)', metavar='')
    parser.add_argument('--mpc_max_yaw_rate',   default=DEFAULT_MPC_PARAMS['max_yaw_rate'], type=float,
                        help='Max yaw rate used in MPC (default: 1.0 rad/s)', metavar='')
    parser.add_argument('--mpc_safety_margin',  default=DEFAULT_MPC_PARAMS['safety_margin'], type=float,
                        help='Safety margin around obstacles (default: 0.2 m)', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
