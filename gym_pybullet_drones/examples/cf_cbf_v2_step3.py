"""
Test script for CFCBFAviaryV2 - Step 3: Add obstacle and activate CBF-QP

This script tests horizontal flight from [0,0,1] to [5,0,1] with an obstacle
at [2,0,1] in the path. CBF-QP should modify the trajectory to avoid collision.
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
    parser.add_argument('--duration', type=float, default=35.0, help='Simulation duration in seconds')
    args = parser.parse_args()

    # Simulation parameters
    DURATION = args.duration
    GUI = args.gui
    INIT_XYZS = np.array([[0.0, 0.0, 0.1]])
    CTRL_FREQ = 25  # 25 Hz control

    # Obstacle parameters
    OBSTACLE_POS = np.array([1.5, 0.0, 0.8])
    OBSTACLE_VEL = np.array([0.0, 0.0, 0.0])
    OBSTACLE_RADIUS = 0.3  # Physical obstacle (red sphere)
    CBF_OBS_RADIUS = 0.5   # CBF safety boundary
    
    # Create environment (inherits from CFAviary)
    print("\n" + "="*80)
    print("CFCBFAviaryV2 - Step 3: Obstacle Avoidance with CBF-QP")
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
        cbf_obs_radius=CBF_OBS_RADIUS  # Use separate CBF radius
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

    # SET OBSTACLE - This activates CBF-QP!
    env.set_obstacle(position=OBSTACLE_POS, velocity=OBSTACLE_VEL)
    print(f"\n[INFO] OBSTACLE SET: position={OBSTACLE_POS}, velocity={OBSTACLE_VEL}")
    print(f"[INFO] CBF-QP is now ACTIVE - drone will avoid obstacle")

    # Visualize obstacle in GUI if enabled
    if GUI:
        # Main obstacle (solid red)
        obstacle_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=OBSTACLE_RADIUS,
            rgbaColor=[1, 0, 0, 0.7]  # Red
        )
        obstacle_collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=OBSTACLE_RADIUS
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_collision,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=OBSTACLE_POS.tolist()
        )
        
        # Safety zone visualization (larger, semi-transparent yellow)
        # Drone encompassing radius ≈ 0.04m, so total CBF radius is larger
        drone_encompassing_r = 0.041  # sqrt(0.028^2 + 0.028^2 + 0.01^2)
        cbf_total_radius = CBF_OBS_RADIUS + drone_encompassing_r
        
        safety_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=cbf_total_radius,
            rgbaColor=[1, 1, 0, 0.2]  # Transparent yellow
        )
        safety_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=safety_visual,
            basePosition=OBSTACLE_POS.tolist()
        )
        
        print(f"[INFO] Obstacle visualized:")
        print(f"  - Red sphere: Physical obstacle (radius={OBSTACLE_RADIUS}m)")
        print(f"  - Yellow sphere: CBF safety zone (radius={cbf_total_radius:.3f}m)")
        print(f"  - Difference: Drone envelope + safety margin = {drone_encompassing_r:.3f}m")

    # Define trajectory phases
    print(f"\n[INFO] Trajectory phases:")
    print(f"  Phase 1 (0-3s):      Takeoff to z=1.0m")
    print(f"  Phase 2 (3-20s):     Horizontal flight x: 0→5m at z=1.0m")
    print(f"  Phase 3 (20-{DURATION}s):   Hover at [5, 0, 1]")
    print(f"\n[INFO] Obstacle will be encountered around x=2.0m (t≈8-9s)")

    # Simulation loop
    num_steps = int(DURATION * CTRL_FREQ)

    # Trajectory parameters - VERY SLOW for CBF to work
    TAKEOFF_DURATION = 3.0
    TAKEOFF_HEIGHT = 1.0
    HORIZONTAL_START = 3.0
    HORIZONTAL_END = 30.0  # Very slow: 5m over 27 seconds = 0.19 m/s avg
    HOVER_START = 30.0
    HORIZONTAL_DISTANCE = 5.0

    print(f"\n[INFO] Starting simulation for {DURATION}s ({num_steps} control steps)")
    print("-" * 80)

    cbf_activations = 0
    min_obstacle_distance = float('inf')

    for i in range(num_steps):
        # Time
        t = i / CTRL_FREQ
        
        # Define desired trajectory based on phase
        if t < TAKEOFF_DURATION:
            # Phase 1: Takeoff
            z_progress = min(t / TAKEOFF_DURATION, 1.0)
            pos_des = np.array([0.0, 0.0, 0.1 + (TAKEOFF_HEIGHT - 0.1) * z_progress])
            vel_des = np.array([0.0, 0.0, (TAKEOFF_HEIGHT - 0.1) / TAKEOFF_DURATION])
            acc_des = np.array([0.0, 0.0, 0.0])
            phase_name = "TAKEOFF"
            
        elif t < HORIZONTAL_END:
            # Phase 2: Horizontal flight with smooth velocity profile
            t_horizontal = t - HORIZONTAL_START
            horizontal_duration = HORIZONTAL_END - HORIZONTAL_START
            s = t_horizontal / horizontal_duration  # normalized time [0, 1]
            
            # Cubic ease-in-out
            if s < 0.5:
                x_smooth = 4 * s * s * s
                vx_smooth = 12 * s * s / horizontal_duration
            else:
                x_smooth = 1 - 4 * (1 - s) * (1 - s) * (1 - s)
                vx_smooth = 12 * (1 - s) * (1 - s) / horizontal_duration
            
            x_des = HORIZONTAL_DISTANCE * x_smooth
            vx_des = HORIZONTAL_DISTANCE * vx_smooth
            
            pos_des = np.array([x_des, 0.0, TAKEOFF_HEIGHT])
            vel_des = np.array([vx_des, 0.0, 0.0])
            acc_des = np.array([0.0, 0.0, 0.0])
            phase_name = "HORIZONTAL"
            
        else:
            # Phase 3: Hover
            pos_des = np.array([HORIZONTAL_DISTANCE, 0.0, TAKEOFF_HEIGHT])
            vel_des = np.array([0.0, 0.0, 0.0])
            acc_des = np.array([0.0, 0.0, 0.0])
            phase_name = "HOVER"
        
        yaw_des = 0.0
        rpy_rate_des = np.array([0.0, 0.0, 0.0])

        # Step environment with CBF wrapper (CBF IS NOW ACTIVE!)
        obs, reward, terminated, truncated, info = env.step_with_cbf(
            i=i,
            pos_des=pos_des,
            vel_des=vel_des,
            acc_des=acc_des,
            yaw_des=yaw_des,
            rpy_rate_des=rpy_rate_des
        )

        # Track CBF activations
        if info.get('cbf_active', False):
            cbf_activations += 1

        # Calculate distance to obstacle
        pos = obs[0][0:3]
        obstacle_dist = np.linalg.norm(pos - OBSTACLE_POS)
        min_obstacle_distance = min(min_obstacle_distance, obstacle_dist)

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
            vel = obs[0][10:13]
            pos_err = np.linalg.norm(pos - pos_des)
            
            cbf_status = "🛡️ CBF-QP" if info.get('cbf_active', False) else "   REF"
            h_value = info.get('cbf_h_value', 0.0)
            
            print(f"t={t:5.1f}s [{phase_name:10s}] [{cbf_status}] | "
                  f"pos=[{pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f}] | "
                  f"obs_dist={obstacle_dist:5.3f}m | "
                  f"h={h_value:7.3f} | "
                  f"err={pos_err:6.4f}m")

        # Check for termination
        if terminated or truncated:
            print(f"\n[WARNING] Episode terminated at t={t:.2f}s")
            break

    # Close environment
    env.close()

    # Save log
    logger.save()
    logger.save_as_csv("cfcbfv2_step3_obstacle")
    print(f"\n[INFO] Log saved to results/cfcbfv2_step3_obstacle.csv")

    # Compute statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS")
    print("="*80)
    
    final_pos = obs[0][0:3]
    expected_final_pos = np.array([5.0, 0.0, 1.0])
    
    final_pos_err = np.linalg.norm(final_pos - expected_final_pos)
    x_err = abs(final_pos[0] - expected_final_pos[0])
    y_err = abs(final_pos[1] - expected_final_pos[1])
    z_err = abs(final_pos[2] - expected_final_pos[2])
    
    print(f"Final position:         [{final_pos[0]:6.3f}, {final_pos[1]:6.3f}, {final_pos[2]:6.3f}]")
    print(f"Expected position:      [{expected_final_pos[0]:6.3f}, {expected_final_pos[1]:6.3f}, {expected_final_pos[2]:6.3f}]")
    print(f"Position error:         {final_pos_err:6.4f} m")
    print(f"  - X error:            {x_err:6.4f} m")
    print(f"  - Y error:            {y_err:6.4f} m")
    print(f"  - Z error:            {z_err:6.4f} m")
    print(f"\nObstacle avoidance:")
    print(f"  - Min distance:       {min_obstacle_distance:6.4f} m")
    print(f"  - Obstacle radius:    {OBSTACLE_RADIUS:6.4f} m")
    print(f"  - Safety margin:      {min_obstacle_distance - OBSTACLE_RADIUS:6.4f} m")
    print(f"  - CBF activations:    {cbf_activations}/{num_steps} steps ({100*cbf_activations/num_steps:.1f}%)")
    
    # Success criteria
    collision = min_obstacle_distance < OBSTACLE_RADIUS
    safe_margin = min_obstacle_distance > OBSTACLE_RADIUS + 0.1  # At least 10cm safety margin
    reached_goal = final_pos_err < 0.15
    
    print(f"\n" + "-"*80)
    if collision:
        print(f"✗ COLLISION DETECTED! Min distance {min_obstacle_distance:.3f}m < radius {OBSTACLE_RADIUS:.3f}m")
    elif safe_margin and reached_goal:
        print(f"✓ TEST PASSED: CBF-QP obstacle avoidance successful!")
        print(f"  - No collision: {min_obstacle_distance:.3f}m > {OBSTACLE_RADIUS:.3f}m ✓")
        print(f"  - Safe margin: {min_obstacle_distance - OBSTACLE_RADIUS:.3f}m > 0.1m ✓")
        print(f"  - Reached goal: {final_pos_err:.3f}m < 0.15m ✓")
    elif not collision and reached_goal:
        print(f"⚠ TEST MARGINAL: Avoided collision but margin is tight")
        print(f"  - Min distance: {min_obstacle_distance:.3f}m")
        print(f"  - Safety margin: {min_obstacle_distance - OBSTACLE_RADIUS:.3f}m")
    elif not collision:
        print(f"⚠ TEST PARTIAL: Avoided collision but didn't reach goal well")
        print(f"  - Min distance: {min_obstacle_distance:.3f}m > {OBSTACLE_RADIUS:.3f}m ✓")
        print(f"  - Final error: {final_pos_err:.3f}m (should be < 0.15m)")
    else:
        print(f"✗ TEST FAILED: Did not avoid collision")
    
    print("="*80)


if __name__ == "__main__":
    main()
