"""
Test script for CFCBFAviaryV2 - Step 1: No obstacles, should fly like CFAviary

This script tests that CFCBFAviaryV2 inheriting from CFAviary produces
stable flight with the Mellinger controller when no obstacles are present.
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
    parser.add_argument('--duration', type=float, default=20.0, help='Simulation duration in seconds')
    args = parser.parse_args()

    # Simulation parameters
    DURATION = args.duration
    GUI = args.gui
    INIT_XYZS = np.array([[0.0, 0.0, 0.1]])  # Start slightly above ground
    CTRL_FREQ = 25  # 25 Hz control

    # Create environment (inherits from CFAviary)
    print("\n" + "="*80)
    print("CFCBFAviaryV2 - Test 1: No Obstacles (Should fly like CFAviary)")
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
    print(f"\n[INFO] Obstacle: None (CBF disabled - should fly exactly like CFAviary)")

    # Define trajectory: Simple vertical climb then hover
    # This is the same trajectory used in test_simple_control.py that worked perfectly
    print(f"\n[INFO] Trajectory: Vertical climb from z=0.1m to z=1.5m over {DURATION}s")

    # Simulation loop
    START_TIME = time.time()
    ctrl_step = 0
    num_steps = int(DURATION * CTRL_FREQ)

    print(f"\n[INFO] Starting simulation for {DURATION}s ({num_steps} control steps)")
    print("-" * 80)

    for i in range(num_steps):
        # Time
        t = i / CTRL_FREQ
        
        # Define desired trajectory (vertical line)
        # Rise from z=0.1m to z=1.5m linearly over duration
        z_start = 0.1
        z_end = 1.5
        z_des = z_start + (z_end - z_start) * min(t / DURATION, 1.0)
        
        pos_des = np.array([0.0, 0.0, z_des])
        vel_des = np.array([0.0, 0.0, (z_end - z_start) / DURATION])
        acc_des = np.array([0.0, 0.0, 0.0])
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
        # Logger expects control array of size 12 (4 RPMs + 8 zeros for BaseAviary compatibility)
        control_log = np.zeros(12)
        # We don't have direct access to RPMs in this wrapper, so leave as zeros
        
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
            
            print(f"t={t:5.1f}s | pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"des=[{pos_des[0]:6.3f}, {pos_des[1]:6.3f}, {pos_des[2]:6.3f}] | "
                  f"pos_err={pos_err:6.4f}m | vel_err={vel_err:6.4f}m/s | "
                  f"CBF={info.get('cbf_active', False)}")

        # Check for termination
        if terminated or truncated:
            print(f"\n[WARNING] Episode terminated at t={t:.2f}s")
            break

    # Close environment
    env.close()

    # Save log
    logger.save()
    logger.save_as_csv("cfcbfv2_test1")
    print(f"\n[INFO] Log saved to results/cfcbfv2_test1.csv")

    # Compute statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    final_pos = obs[0][0:3]
    final_vel = obs[0][10:13]
    expected_final_pos = np.array([0.0, 0.0, 1.5])
    expected_final_vel = np.array([0.0, 0.0, 0.0])
    
    final_pos_err = np.linalg.norm(final_pos - expected_final_pos)
    final_vel_err = np.linalg.norm(final_vel - expected_final_vel)
    
    print(f"Final position:     [{final_pos[0]:6.3f}, {final_pos[1]:6.3f}, {final_pos[2]:6.3f}]")
    print(f"Expected position:  [{expected_final_pos[0]:6.3f}, {expected_final_pos[1]:6.3f}, {expected_final_pos[2]:6.3f}]")
    print(f"Position error:     {final_pos_err:6.4f} m")
    print(f"Velocity error:     {final_vel_err:6.4f} m/s")
    
    # Success criteria (same as test_simple_control.py)
    if final_pos_err < 0.05 and final_vel_err < 0.1:
        print(f"\n✓ TEST PASSED: CFCBFAviaryV2 with no obstacles flies like CFAviary!")
        print(f"  Position tracking: EXCELLENT (< 5cm error)")
        print(f"  Velocity tracking: EXCELLENT (< 0.1 m/s error)")
    elif final_pos_err < 0.15:
        print(f"\n⚠ TEST MARGINAL: Position error is acceptable but not excellent")
    else:
        print(f"\n✗ TEST FAILED: Position error too large")
    
    print("="*80)


if __name__ == "__main__":
    main()
