"""
Test script for CFCBFAviaryV2 - Step 2: Horizontal trajectory

This script tests horizontal flight from [0,0,1] to [5,0,1] over 20 seconds.
Still with no obstacles (CBF disabled).
"""

import os
import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CFCBFAviaryV2 import CFCBFAviaryV2
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI')
    parser.add_argument('--duration', type=float, default=25.0, help='Simulation duration in seconds')
    args = parser.parse_args()

    # Simulation parameters
    DURATION = args.duration
    GUI = args.gui
    INIT_XYZS = np.array([[0.0, 0.0, 0.1]])  # Start slightly above ground
    CTRL_FREQ = 25  # 25 Hz control

    # Create environment (inherits from CFAviary)
    print("\n" + "="*80)
    print("CFCBFAviaryV2 - Step 2: Horizontal Trajectory [0,0,1] → [5,0,1]")
    print("="*80)
    
    env = CFCBFAviaryV2(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,
        pyb_freq=500,
        ctrl_freq=CTRL_FREQ,
        gui=GUI,
        record=False,
        obstacles=False,
        user_debug_gui=False,
        verbose=True,
        cbf_gamma=1.0,
        cbf_obs_radius=0.1
    )

    # Initialize logger
    logger = Logger(
        logging_freq_hz=CTRL_FREQ,
        num_drones=1,
        output_folder='results',
        colab=False
    )

    # Reset environment
    obs, info = env.reset()
    print(f"\n[INFO] Environment reset complete")
    print(f"[INFO] Initial position: {obs[0][0:3]}")
    print(f"[INFO] CBF active: {info.get('cbf_active', False)}")

    # Obstacle status: None (CBF disabled)
    env.set_obstacle(position=None, velocity=None)
    print(f"\n[INFO] Obstacle: None (CBF disabled)")

    # Define trajectory phases
    print(f"\n[INFO] Trajectory phases:")
    print(f"  Phase 1 (0-3s):      Takeoff to z=1.0m")
    print(f"  Phase 2 (3-20s):     Horizontal flight x: 0→5m at z=1.0m")
    print(f"  Phase 3 (20-{DURATION}s):   Hover at [5, 0, 1]")

    # Simulation loop
    START_TIME = time.time()
    ctrl_step = 0
    num_steps = int(DURATION * CTRL_FREQ)

    # Trajectory parameters
    TAKEOFF_DURATION = 3.0      # seconds
    TAKEOFF_HEIGHT = 1.0        # meters
    HORIZONTAL_START = 3.0      # seconds (when horizontal flight starts)
    HORIZONTAL_END = 20.0       # seconds (when to reach target)
    HOVER_START = 20.0          # seconds (when to start hovering)
    HORIZONTAL_DISTANCE = 5.0   # meters in x-direction

    print(f"\n[INFO] Starting simulation for {DURATION}s ({num_steps} control steps)")
    print("-" * 80)

    for i in range(num_steps):
        # Time
        t = i / CTRL_FREQ
        
        # Define desired trajectory based on phase
        if t < TAKEOFF_DURATION:
            # Phase 1: Takeoff - rise to z=1.0m
            z_progress = min(t / TAKEOFF_DURATION, 1.0)
            pos_des = np.array([0.0, 0.0, 0.1 + (TAKEOFF_HEIGHT - 0.1) * z_progress])
            vel_des = np.array([0.0, 0.0, (TAKEOFF_HEIGHT - 0.1) / TAKEOFF_DURATION])
            acc_des = np.array([0.0, 0.0, 0.0])
            phase_name = "TAKEOFF"
            
        elif t < HORIZONTAL_END:
            # Phase 2: Horizontal flight at constant z=1.0m with smooth deceleration
            t_horizontal = t - HORIZONTAL_START
            horizontal_duration = HORIZONTAL_END - HORIZONTAL_START
            
            # Use smooth cubic spline for position to naturally decelerate
            # This gives smooth velocity profile that goes to zero at the end
            s = t_horizontal / horizontal_duration  # normalized time [0, 1]
            
            # Cubic ease-in-out: smooth acceleration and deceleration
            if s < 0.5:
                x_smooth = 4 * s * s * s  # Accelerate
            else:
                x_smooth = 1 - 4 * (1 - s) * (1 - s) * (1 - s)  # Decelerate
            
            # Velocity is derivative of position
            if s < 0.5:
                vx_smooth = 12 * s * s / horizontal_duration
            else:
                vx_smooth = 12 * (1 - s) * (1 - s) / horizontal_duration
            
            x_des = HORIZONTAL_DISTANCE * x_smooth
            vx_des = HORIZONTAL_DISTANCE * vx_smooth
            
            pos_des = np.array([x_des, 0.0, TAKEOFF_HEIGHT])
            vel_des = np.array([vx_des, 0.0, 0.0])
            acc_des = np.array([0.0, 0.0, 0.0])
            phase_name = "HORIZONTAL"
            
        else:
            # Phase 3: Hover at final position
            pos_des = np.array([HORIZONTAL_DISTANCE, 0.0, TAKEOFF_HEIGHT])
            vel_des = np.array([0.0, 0.0, 0.0])
            acc_des = np.array([0.0, 0.0, 0.0])
            phase_name = "HOVER"
        
        yaw_des = 0.0
        rpy_rate_des = np.array([0.0, 0.0, 0.0])

        # Step environment with CBF wrapper (but CBF is disabled since obstacle=None)
        obs, reward, terminated, truncated, info = env.step_with_cbf(
            i=i,
            pos_des=pos_des,
            vel_des=vel_des,
            acc_des=acc_des,
            yaw_des=yaw_des,
            rpy_rate_des=rpy_rate_des
        )

        # Log state
        control_log = np.zeros(12)
        logger.log(
            drone=0,
            timestamp=t,
            state=obs[0],
            control=control_log
        )

        # Print progress every second
        if i % CTRL_FREQ == 0:
            pos = obs[0][0:3]
            vel = obs[0][10:13]
            pos_err = np.linalg.norm(pos - pos_des)
            vel_err = np.linalg.norm(vel - vel_des)
            
            print(f"t={t:5.1f}s [{phase_name:10s}] | "
                  f"pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"des=[{pos_des[0]:6.3f}, {pos_des[1]:6.3f}, {pos_des[2]:6.3f}] | "
                  f"err={pos_err:6.4f}m | vel={np.linalg.norm(vel):5.3f}m/s")

        # Check for termination
        if terminated or truncated:
            print(f"\n[WARNING] Episode terminated at t={t:.2f}s")
            break

    # Close environment
    env.close()

    # Save log
    logger.save()
    logger.save_as_csv("cfcbfv2_step2_horizontal")
    print(f"\n[INFO] Log saved to results/cfcbfv2_step2_horizontal.csv")

    # Compute statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    final_pos = obs[0][0:3]
    final_vel = obs[0][10:13]
    expected_final_pos = np.array([5.0, 0.0, 1.0])
    expected_final_vel = np.array([0.0, 0.0, 0.0])
    
    final_pos_err = np.linalg.norm(final_pos - expected_final_pos)
    final_vel_err = np.linalg.norm(final_vel - expected_final_vel)
    
    # Individual axis errors
    x_err = abs(final_pos[0] - expected_final_pos[0])
    y_err = abs(final_pos[1] - expected_final_pos[1])
    z_err = abs(final_pos[2] - expected_final_pos[2])
    
    print(f"Final position:     [{final_pos[0]:6.3f}, {final_pos[1]:6.3f}, {final_pos[2]:6.3f}]")
    print(f"Expected position:  [{expected_final_pos[0]:6.3f}, {expected_final_pos[1]:6.3f}, {expected_final_pos[2]:6.3f}]")
    print(f"Position error:     {final_pos_err:6.4f} m")
    print(f"  - X error:        {x_err:6.4f} m")
    print(f"  - Y error:        {y_err:6.4f} m")
    print(f"  - Z error:        {z_err:6.4f} m")
    print(f"Velocity error:     {final_vel_err:6.4f} m/s")
    
    # Success criteria
    if final_pos_err < 0.1 and z_err < 0.05:
        print(f"\n✓ TEST PASSED: Horizontal trajectory tracking successful!")
        print(f"  Position error: {final_pos_err:.4f}m < 0.1m ✓")
        print(f"  Altitude hold: {z_err:.4f}m < 0.05m ✓")
        print(f"\n  Ready for Step 3: Add obstacle!")
    elif final_pos_err < 0.2:
        print(f"\n⚠ TEST MARGINAL: Position error acceptable but could be better")
        print(f"  Position error: {final_pos_err:.4f}m (target: < 0.1m)")
    else:
        print(f"\n✗ TEST FAILED: Position error too large")
        print(f"  Position error: {final_pos_err:.4f}m > 0.2m")
    
    print("="*80)


if __name__ == "__main__":
    main()
