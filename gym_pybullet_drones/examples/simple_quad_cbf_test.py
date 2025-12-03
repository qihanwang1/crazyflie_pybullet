"""
Test script using SimpleQuadCBF with direct acceleration control
This matches the paper's approach: direct acceleration commands modified by CBF-QP
"""

import os
import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.SimpleQuadCBF import SimpleQuadCBF
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI')
    parser.add_argument('--duration', type=float, default=30.0, help='Simulation duration')
    args = parser.parse_args()

    # Parameters
    DURATION = args.duration
    GUI = args.gui
    INIT_XYZS = np.array([[0.0, 0.0, 0.5]])
    CTRL_FREQ = 120  # Hz - must divide pyb_freq (240)
    
    # Obstacle parameters (paper's setup)
    OBSTACLE_POS = np.array([1.5, 0.0, 0.8])
    OBSTACLE_VEL = np.array([0.0, 0.0, 0.0])
    OBSTACLE_RADIUS = 0.3
    SAFETY_RADIUS = 0.5  # Total CBF avoidance zone
    
    print("\n" + "="*80)
    print("SimpleQuadCBF Test - Direct Acceleration Control (Paper's Approach)")
    print("="*80)
    
    # Create environment
    env = SimpleQuadCBF(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=CTRL_FREQ,
        gui=GUI,
        record=False,
        obstacles=False,
        user_debug_gui=False
    )
    
    # Set obstacle
    env.set_obstacle(OBSTACLE_POS, OBSTACLE_VEL)
    
    # Visualize obstacle in GUI
    if GUI:
        # Physical obstacle (red)
        p.loadURDF(
            "sphere2.urdf",
            OBSTACLE_POS.tolist(),
            globalScaling=OBSTACLE_RADIUS,
            physicsClientId=env.CLIENT
        )
        
        # CBF safety zone (yellow, transparent)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=SAFETY_RADIUS,
            rgbaColor=[1, 1, 0, 0.3],
            physicsClientId=env.CLIENT
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=OBSTACLE_POS.tolist(),
            physicsClientId=env.CLIENT
        )
        
        print(f"\n[INFO] Obstacle visualized:")
        print(f"  - Red sphere: Physical obstacle (radius={OBSTACLE_RADIUS}m)")
        print(f"  - Yellow sphere: CBF safety zone (radius={SAFETY_RADIUS}m)")
    
    # Trajectory parameters - VERY SLOW like the paper
    TAKEOFF_HEIGHT = 1.0
    TAKEOFF_DURATION = 3.0
    HORIZONTAL_START = 3.0
    HORIZONTAL_END = 25.0  # Very slow: 3m over 22s ≈ 0.14 m/s
    GOAL_X = 3.0
    
    print(f"\n[INFO] Trajectory:")
    print(f"  Phase 1 (0-{TAKEOFF_DURATION}s): Takeoff to z={TAKEOFF_HEIGHT}m")
    print(f"  Phase 2 ({HORIZONTAL_START}-{HORIZONTAL_END}s): Horizontal to x={GOAL_X}m (SLOW!)")
    print(f"  Phase 3 ({HORIZONTAL_END}-{DURATION}s): Hover")
    
    # Simulation
    num_steps = int(DURATION * CTRL_FREQ)
    dt = 1.0 / CTRL_FREQ
    
    # CBF parameters (from paper)
    gamma_cbf = 1.0  # Paper uses γ=1
    
    # Stats
    min_distance = float('inf')
    cbf_activations = 0
    
    print(f"\n[INFO] Starting simulation: {DURATION}s, {num_steps} steps")
    print("-" * 80)
    
    for i in range(num_steps):
        t = i * dt
        
        # Get current state
        obs = env._computeObs()
        pos = obs[0][0:3]
        vel = obs[0][10:13]
        
        # Track minimum distance
        dist_to_obs = np.linalg.norm(pos - OBSTACLE_POS)
        min_distance = min(min_distance, dist_to_obs)
        
        # Compute desired trajectory
        if t < TAKEOFF_DURATION:
            # Takeoff
            target_pos = np.array([0.0, 0.0, 0.5 + (TAKEOFF_HEIGHT - 0.5) * (t / TAKEOFF_DURATION)])
            target_vel = np.array([0.0, 0.0, (TAKEOFF_HEIGHT - 0.5) / TAKEOFF_DURATION])
            
        elif t < HORIZONTAL_END:
            # Horizontal flight (smooth)
            t_h = (t - HORIZONTAL_START) / (HORIZONTAL_END - HORIZONTAL_START)
            t_h = np.clip(t_h, 0, 1)
            # Cubic smooth
            if t_h < 0.5:
                s = 4 * t_h**3
                v = 12 * t_h**2 / (HORIZONTAL_END - HORIZONTAL_START)
            else:
                s = 1 - 4 * (1 - t_h)**3
                v = 12 * (1 - t_h)**2 / (HORIZONTAL_END - HORIZONTAL_START)
            
            x_target = GOAL_X * s
            vx_target = GOAL_X * v
            
            target_pos = np.array([x_target, 0.0, TAKEOFF_HEIGHT])
            target_vel = np.array([vx_target, 0.0, 0.0])
        else:
            # Hover
            target_pos = np.array([GOAL_X, 0.0, TAKEOFF_HEIGHT])
            target_vel = np.array([0.0, 0.0, 0.0])
        
        # Simple PD controller for desired acceleration (paper's "reference control")
        kp = 3.0
        kd = 3.0
        acc_des = kp * (target_pos - pos) + kd * (target_vel - vel)
        
        # CBF-QP modification (paper's PolyC2BF-QP)
        # Simplified: use distance-based CBF (h = distance - r_safe)
        to_drone = pos - OBSTACLE_POS
        distance = np.linalg.norm(to_drone)
        
        if distance > 1e-6:
            direction = to_drone / distance
            
            # CBF: h = distance - r_safe
            h = distance - SAFETY_RADIUS
            
            # h_dot = direction · relative_velocity
            rel_vel = vel - OBSTACLE_VEL
            h_dot = np.dot(direction, rel_vel)
            
            # CBF constraint: h_ddot + γ * h_dot ≥ 0
            # h_ddot ≈ direction · acc_drone
            # So: direction · acc_drone ≥ -γ * h_dot
            
            min_acc_needed = -gamma_cbf * h_dot
            acc_projection = np.dot(acc_des, direction)
            
            if acc_projection < min_acc_needed:
                # Constraint violated - add correction
                acc_deficit = min_acc_needed - acc_projection
                acc_correction = acc_deficit * direction
                acc_modified = acc_des + acc_correction
                cbf_activations += 1
                
                if i % (CTRL_FREQ // 4) == 0:  # Print 4 times per second
                    print(f"    [CBF] ACTIVE: h={h:.3f}m, h_dot={h_dot:.2f}m/s, "
                          f"dist={distance:.3f}m, correction={np.linalg.norm(acc_correction):.2f}m/s²")
            else:
                acc_modified = acc_des
        else:
            acc_modified = acc_des
        
        # Step with modified acceleration
        obs, reward, terminated, truncated, info = env.step_direct_accel(acc_modified)
        
        # Logging every second
        if i % CTRL_FREQ == 0:
            print(f"t={t:5.1f}s | pos=[{pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}] | "
                  f"dist={dist_to_obs:.3f}m | h={distance - SAFETY_RADIUS:.3f}m")
        
        if terminated:
            print(f"\n[TERMINATED] at t={t:.1f}s")
            break
    
    # Final statistics
    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Minimum distance to obstacle: {min_distance:.3f}m")
    print(f"Safety radius: {SAFETY_RADIUS}m")
    print(f"Physical obstacle radius: {OBSTACLE_RADIUS}m")
    print(f"CBF activations: {cbf_activations}")
    
    if min_distance > SAFETY_RADIUS:
        print("\n✓ SUCCESS: Never violated safety boundary!")
    elif min_distance > OBSTACLE_RADIUS:
        print(f"\n⚠ WARNING: Violated safety zone but avoided collision (margin: {min_distance - OBSTACLE_RADIUS:.3f}m)")
    else:
        print(f"\n✗ FAILURE: COLLISION! Penetrated obstacle by {OBSTACLE_RADIUS - min_distance:.3f}m")
    
    env.close()
    print("\nSimulation ended.")


if __name__ == "__main__":
    main()
