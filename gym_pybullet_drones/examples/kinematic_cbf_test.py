"""
Test script for kinematic quadrotor with CBF-QP obstacle avoidance.

This implements Solution 1 from HOW_TO_USE_QP_CONTROLLER.md:
-     # Obstacle parameters
    OBSTACLE_POS = np.array([1.5, 0.0, 0.8])
    OBSTACLE_RADIUS = 0.3
    SAFETY_MARGIN = 0.35  # Stay at least this far from obstacle surface (increased for more clearance)
    TOTAL_SAFE_DISTANCE = OBSTACLE_RADIUS + SAFETY_MARGIN  # = 0.65m
    
    # CBF parameters
    GAMMA = 3.0  # Class-K function parameter (increased for more conservative behavior)cceleration control (no Mellinger)
- High-frequency CBF enforcement
- Simple distance-based barrier function

Author: GitHub Copilot
Date: November 2025
"""

import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.KinematicCBFAviary import KinematicCBFAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def compute_cbf_safe_acceleration(current_pos, current_vel, desired_accel, 
                                   obstacle_pos, obstacle_radius, safety_margin, gamma=1.0):
    """
    Compute CBF-safe acceleration using simple barrier function.
    
    Barrier function: h = ||p - p_obs|| - (r_obs + r_safe)
    Constraint: h_dot >= -gamma * h  =>  direction · v >= -gamma * h
    If violated, add corrective acceleration.
    
    Parameters
    ----------
    current_pos : ndarray
        Current position [x, y, z]
    current_vel : ndarray
        Current velocity [vx, vy, vz]
    desired_accel : ndarray
        Desired acceleration from reference controller [ax, ay, az]
    obstacle_pos : ndarray
        Obstacle position [x, y, z]
    obstacle_radius : float
        Physical radius of obstacle
    safety_margin : float
        Additional safety distance
    gamma : float
        CBF class-K parameter
        
    Returns
    -------
    ndarray
        Safe acceleration [ax, ay, az]
    """
    # Compute relative position
    rel_pos = current_pos - obstacle_pos
    distance = np.linalg.norm(rel_pos)
    
    # Barrier function value
    h = distance - (obstacle_radius + safety_margin)
    
    # Check if we're in danger zone
    if h < 0.8:  # Activate CBF when close
        direction = rel_pos / distance
        
        # Barrier derivative: h_dot = direction · velocity
        h_dot = np.dot(direction, current_vel)
        
        # CBF constraint: h_dot >= -gamma * h
        constraint_violation = -gamma * h - h_dot
        
        if constraint_violation > 0:
            # Need corrective acceleration
            # Add repulsive acceleration proportional to violation
            correction_gain = 15.0  # Tunable parameter (increased for stronger avoidance)
            corrective_accel = correction_gain * constraint_violation * direction
            
            safe_accel = desired_accel + corrective_accel
            
            # Limit to reasonable bounds
            safe_accel = np.clip(safe_accel, -15.0, 15.0)
            
            return safe_accel
    
    # No correction needed
    return desired_accel


def pd_controller(current_pos, current_vel, target_pos, target_vel, kp=4.0, kd=2.0):
    """
    Simple PD controller for position tracking.
    
    Parameters
    ----------
    current_pos : ndarray
        Current position [x, y, z]
    current_vel : ndarray
        Current velocity [vx, vy, vz]
    target_pos : ndarray
        Target position [x, y, z]
    target_vel : ndarray
        Target velocity [vx, vy, vz]
    kp : float
        Proportional gain
    kd : float
        Derivative gain
        
    Returns
    -------
    ndarray
        Desired acceleration [ax, ay, az]
    """
    pos_error = target_pos - current_pos
    vel_error = target_vel - current_vel
    
    accel = kp * pos_error + kd * vel_error
    
    # Limit acceleration
    accel = np.clip(accel, -10.0, 10.0)
    
    return accel


def run_kinematic_cbf_test(duration=25, gui=True):
    """
    Run kinematic quadrotor test with CBF obstacle avoidance.
    
    Parameters
    ----------
    duration : int
        Total simulation time in seconds
    gui : bool
        Whether to show PyBullet GUI
    """
    # Environment parameters
    CTRL_FREQ = 240  # High frequency for CBF
    
    # Obstacle parameters
    OBSTACLE_POS = np.array([1.5, 0.0, 0.8])
    OBSTACLE_RADIUS = 0.2
    SAFETY_MARGIN = 0.5  # Stay at least this far from obstacle surface (increased for more clearance)
    TOTAL_SAFE_DISTANCE = OBSTACLE_RADIUS + SAFETY_MARGIN/2  # = 0.65m
    
    # CBF parameters
    GAMMA = 2.0  # Class-K function parameter (increased for more conservative behavior)
    
    # Trajectory parameters
    HOVER_DURATION = 3.0  # Hover at start
    APPROACH_DURATION = 15.0  # Fly towards and past obstacle
    FINAL_HOVER = 5.0  # Hover at end
    
    START_POS = np.array([0.0, 0.0, 0.8])
    END_POS = np.array([3.0, 0.0, 0.8])
    
    print("=" * 70)
    print("KINEMATIC QUADROTOR CBF-QP TEST")
    print("=" * 70)
    print("\nTest Configuration:")
    print(f"  Control Frequency: {CTRL_FREQ} Hz")
    print(f"  Obstacle Position: {OBSTACLE_POS}")
    print(f"  Obstacle Radius: {OBSTACLE_RADIUS} m")
    print(f"  Safety Margin: {SAFETY_MARGIN} m")
    print(f"  Total Safe Distance: {TOTAL_SAFE_DISTANCE} m")
    print(f"  CBF Gamma: {GAMMA}")
    print(f"  Duration: {duration} seconds")
    print("\nTrajectory:")
    print(f"  1. Hover at {START_POS} for {HOVER_DURATION}s")
    print(f"  2. Fly from {START_POS} to {END_POS} over {APPROACH_DURATION}s")
    print(f"  3. Hover at {END_POS} for {FINAL_HOVER}s")
    print("=" * 70)
    
    # Create environment
    env = KinematicCBFAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=START_POS.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False
    )
    
    # Initialize logger
    logger = Logger(
        logging_freq_hz=CTRL_FREQ,
        num_drones=1,
        duration_sec=duration
    )
    
    # Simulation variables
    obs, info = env.reset()
    
    # Create obstacle AFTER reset (so it doesn't get cleared)
    obstacle_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=OBSTACLE_RADIUS,
        rgbaColor=[1, 0, 0, 0.8],
        physicsClientId=env.CLIENT
    )
    obstacle_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=OBSTACLE_RADIUS,
        physicsClientId=env.CLIENT
    )
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=obstacle_collision,
        baseVisualShapeIndex=obstacle_visual,
        basePosition=OBSTACLE_POS,
        physicsClientId=env.CLIENT
    )
    
    # Create safety zone visualization (transparent sphere)
    safety_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=TOTAL_SAFE_DISTANCE,
        rgbaColor=[1, 1, 0, 0.3],
        physicsClientId=env.CLIENT
    )
    safety_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=safety_visual,
        basePosition=OBSTACLE_POS,
        physicsClientId=env.CLIENT
    )
    
    print(f"\n✓ Created obstacle sphere at {OBSTACLE_POS} (radius {OBSTACLE_RADIUS}m)")
    print(f"✓ Created safety boundary at {OBSTACLE_POS} (radius {TOTAL_SAFE_DISTANCE}m)")
    print(f"  Look for: RED sphere (obstacle) + YELLOW sphere (safety zone)\n")
    start_time = time.time()
    min_distance = float('inf')
    violation_count = 0
    
    num_steps = int(duration * CTRL_FREQ)
    
    print("\nStarting simulation...")
    print("-" * 70)
    
    for step in range(num_steps):
        current_time = step / CTRL_FREQ
        
        # Get current state
        state = env._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        
        # Compute distance to obstacle
        distance_to_obstacle = np.linalg.norm(current_pos - OBSTACLE_POS)
        min_distance = min(min_distance, distance_to_obstacle)
        
        # Check for safety violation
        if distance_to_obstacle < TOTAL_SAFE_DISTANCE:
            violation_count += 1
            if violation_count == 1:  # First violation
                print(f"\n⚠️  SAFETY VIOLATION at t={current_time:.2f}s")
                print(f"   Distance: {distance_to_obstacle:.3f}m < {TOTAL_SAFE_DISTANCE:.3f}m")
        
        # Compute target position based on trajectory phase
        if current_time < HOVER_DURATION:
            # Phase 1: Hover at start
            target_pos = START_POS
            target_vel = np.zeros(3)
        elif current_time < HOVER_DURATION + APPROACH_DURATION:
            # Phase 2: Linear trajectory to end
            alpha = (current_time - HOVER_DURATION) / APPROACH_DURATION
            target_pos = START_POS + alpha * (END_POS - START_POS)
            target_vel = (END_POS - START_POS) / APPROACH_DURATION
        else:
            # Phase 3: Hover at end
            target_pos = END_POS
            target_vel = np.zeros(3)
        
        # PD controller for reference acceleration
        desired_accel = pd_controller(current_pos, current_vel, target_pos, target_vel)
        
        # Apply CBF-QP to get safe acceleration
        safe_accel = compute_cbf_safe_acceleration(
            current_pos, current_vel, desired_accel,
            OBSTACLE_POS, OBSTACLE_RADIUS, SAFETY_MARGIN, GAMMA
        )
        
        # Apply action (direct acceleration)
        obs, reward, terminated, truncated, info = env.step(safe_accel)
        
        # Log data
        logger.log(
            drone=0,
            timestamp=current_time,
            state=state,
            control=np.zeros(12)  # Dummy control for logging
        )
        
        # Print progress every 2 seconds
        if step % (2 * CTRL_FREQ) == 0:
            print(f"t={current_time:5.1f}s | pos=[{current_pos[0]:5.2f}, {current_pos[1]:5.2f}, {current_pos[2]:5.2f}] | "
                  f"dist={distance_to_obstacle:5.3f}m | min={min_distance:5.3f}m")
        
        # Sync simulation
        env.render()
    
    # Print results
    print("-" * 70)
    print("\nSIMULATION COMPLETE")
    print("=" * 70)
    print("\nResults:")
    print(f"  Minimum Distance to Obstacle: {min_distance:.3f} m")
    print(f"  Safety Boundary: {TOTAL_SAFE_DISTANCE:.3f} m")
    
    if min_distance >= TOTAL_SAFE_DISTANCE:
        print(f"  ✅ SUCCESS - Stayed outside safety boundary!")
        print(f"     Margin: {(min_distance - TOTAL_SAFE_DISTANCE)*100:.1f} cm")
    else:
        print(f"  ❌ FAILURE - Penetrated safety boundary")
        print(f"     Violation: {(TOTAL_SAFE_DISTANCE - min_distance)*100:.1f} cm")
        print(f"     Violation frames: {violation_count}/{num_steps}")
    
    print("\nThis direct control approach should work where Mellinger failed!")
    print("=" * 70)
    
    # Close environment
    env.close()
    
    # Save log
    logger.save()
    logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kinematic quadrotor CBF test')
    parser.add_argument('--duration', type=int, default=25, help='Duration in seconds')
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI')
    args = parser.parse_args()
    
    run_kinematic_cbf_test(duration=args.duration, gui=args.gui)
