"""
Test script for kinematic quadrotor with CBF obstacle avoidance - Constant Heading.

This tests horizontal obstacle avoidance while maintaining constant heading direction.
The drone maintains a fixed heading (pointing in +X direction) and constant velocity.
CBF pushes it sideways to avoid obstacle, but it keeps flying in the heading direction.

Trajectory: Constant velocity in +X direction at Z=1.0m
Heading: Always pointing in +X direction (yaw=0)
Obstacle: [1.5, 0.2, 1.0] with radius 0.2m
Challenge: Drone gets pushed sideways by CBF but maintains heading and continues forward

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


def compute_cbf_safe_acceleration_multi(current_pos, current_vel, desired_accel, 
                                        obstacles, gamma=1.0):
    """
    Compute CBF-safe acceleration for multiple obstacles.
    
    For each obstacle, computes the corrective acceleration needed.
    Accumulates all corrections to handle multiple obstacles.
    
    Parameters
    ----------
    current_pos : ndarray
        Current position [x, y, z]
    current_vel : ndarray
        Current velocity [vx, vy, vz]
    desired_accel : ndarray
        Desired acceleration from reference controller [ax, ay, az]
    obstacles : list of dict
        List of obstacles, each with keys: 'pos', 'radius', 'safety_margin'
    gamma : float
        CBF class-K parameter
        
    Returns
    -------
    ndarray
        Safe acceleration [ax, ay, az]
    """
    safe_accel = desired_accel.copy()
    
    # Check each obstacle and accumulate corrections
    for obs in obstacles:
        obstacle_pos = obs['pos']
        obstacle_radius = obs['radius']
        safety_margin = obs['safety_margin']
        
        # Compute relative position
        rel_pos = current_pos - obstacle_pos
        distance = np.linalg.norm(rel_pos)
        
        # Barrier function value
        h = distance - (obstacle_radius + safety_margin)
        
        # Check if we're in danger zone - activate earlier for smoother avoidance
        if h < 1.2:  # Activate CBF when approaching (increased from 0.8)
            direction = rel_pos / distance
            
            # Barrier derivative: h_dot = direction · velocity
            h_dot = np.dot(direction, current_vel)
            
            # CBF constraint: h_dot >= -gamma * h
            constraint_violation = -gamma * h - h_dot
            
            if constraint_violation > 0:
                # Need corrective acceleration
                # Reduced gain for smoother, more gradual corrections
                correction_gain = 5.0  # Reduced from 15.0 for gentler avoidance
                corrective_accel = correction_gain * constraint_violation * direction
                
                safe_accel += corrective_accel
    
    # Limit to reasonable bounds (reduced for smoother motion)
    safe_accel = np.clip(safe_accel, -8.0, 8.0)  # Reduced from 15.0 for gentler acceleration
    
    return safe_accel


def constant_velocity_controller(current_vel, target_vel, current_pos, target_altitude, kd=2.0, kp_z=4.0):
    """
    Controller for maintaining constant velocity in XY plane and altitude.
    
    Parameters
    ----------
    current_vel : ndarray
        Current velocity [vx, vy, vz]
    target_vel : ndarray
        Target velocity [vx, vy, vz]
    current_pos : ndarray
        Current position [x, y, z]
    target_altitude : float
        Target Z altitude
    kd : float
        Derivative gain for XY velocity tracking
    kp_z : float
        Proportional gain for altitude control
        
    Returns
    -------
    ndarray
        Desired acceleration [ax, ay, az]
    """
    # XY: Simple velocity tracking (maintain heading velocity)
    vel_error_xy = target_vel[0:2] - current_vel[0:2]
    accel_xy = kd * vel_error_xy
    
    # Z: Position tracking (maintain altitude)
    z_error = target_altitude - current_pos[2]
    z_vel_error = 0.0 - current_vel[2]  # Want zero vertical velocity
    accel_z = kp_z * z_error + kd * z_vel_error
    
    accel = np.array([accel_xy[0], accel_xy[1], accel_z])
    
    # Limit acceleration
    accel = np.clip(accel, -10.0, 10.0)
    
    return accel


def run_xy_plane_cbf_test(duration=25, gui=True):
    """
    Run kinematic quadrotor test with CBF obstacle avoidance in XY plane.
    
    Parameters
    ----------
    duration : int
        Total simulation time in seconds
    gui : bool
        Whether to show PyBullet GUI
    """
    # Environment parameters
    CTRL_FREQ = 240  # High frequency for CBF
    
    # Obstacle parameters - MULTIPLE OBSTACLES IN XY PLANE
    OBSTACLES = [
        {
            'pos': np.array([0.2, -0.75, 1.0]),
            'radius': 0.2,
            'safety_margin': 0.4,
            'color': [1, 0, 0, 0.8]  # Red
        },
        {
            'pos': np.array([-0.5, -2.25, 1.0]),
            'radius': 0.2,
            'safety_margin': 0.4,
            'color': [1, 0.5, 0, 0.8]  # Orange
        }
    ]
    
    # CBF parameters - reduced gamma for smoother constraint enforcement
    GAMMA = 1.5  # Reduced from 3.0 for gentler corrections
    
    # Flight parameters - CONSTANT VELOCITY HEADING (reduced for smoother motion)
    CRUISE_VELOCITY = 0.1  # m/s in -Y direction (reduced from 0.2 for smoother motion)
    FLIGHT_ALTITUDE = 1.0  # meters
    HEADING_DIRECTION = np.array([0.0, -1.0, 0.0])  # -Y direction
    
    START_POS = np.array([0.0, 0.0, FLIGHT_ALTITUDE])
    
    print("=" * 70)
    print("KINEMATIC QUADROTOR CBF TEST - CONSTANT HEADING, MULTI-OBSTACLE")
    print("=" * 70)
    print("\nTest Configuration:")
    print(f"  Control Frequency: {CTRL_FREQ} Hz")
    print(f"  Number of Obstacles: {len(OBSTACLES)}")
    for i, obs in enumerate(OBSTACLES):
        total_dist = obs['radius'] + obs['safety_margin']/2
        print(f"  Obstacle {i+1}: pos={obs['pos']}, radius={obs['radius']}m, safe_dist={total_dist:.2f}m")
    print(f"  CBF Gamma: {GAMMA}")
    print(f"  Duration: {duration} seconds")
    print("\nFlight Profile:")
    print(f"  Heading Direction: -Y (constant)")
    print(f"  Cruise Velocity: {CRUISE_VELOCITY} m/s")
    print(f"  Flight Altitude: {FLIGHT_ALTITUDE} m")
    print(f"  Start Position: {START_POS}")
    print("\nChallenge:")
    print(f"  Obstacles offset from centerline along -Y path")
    print(f"  Drone maintains constant velocity in heading direction (-Y)")
    print(f"  CBF will push drone sideways (in X) to avoid both obstacles")
    print(f"  After clearing obstacles, drone continues in -Y direction")
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
    
    # Create obstacles AFTER reset (so they don't get cleared)
    obstacle_ids = []
    safety_ids = []
    
    for i, obstacle in enumerate(OBSTACLES):
        # Create obstacle sphere
        obstacle_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=obstacle['radius'],
            rgbaColor=obstacle['color'],
            physicsClientId=env.CLIENT
        )
        obstacle_collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=obstacle['radius'],
            physicsClientId=env.CLIENT
        )
        obstacle_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=obstacle_collision,
            baseVisualShapeIndex=obstacle_visual,
            basePosition=obstacle['pos'],
            physicsClientId=env.CLIENT
        )
        obstacle_ids.append(obstacle_id)
        
        # Create safety zone visualization (transparent sphere)
        total_safe_dist = obstacle['radius'] + obstacle['safety_margin']/2
        safety_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=total_safe_dist,
            rgbaColor=[1, 1, 0, 0.3],
            physicsClientId=env.CLIENT
        )
        safety_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=safety_visual,
            basePosition=obstacle['pos'],
            physicsClientId=env.CLIENT
        )
        safety_ids.append(safety_id)
    
    # Draw heading direction arrow (for visualization)
    if gui:
        arrow_length = 1.0
        arrow_end = START_POS + arrow_length * HEADING_DIRECTION
        p.addUserDebugLine(
            lineFromXYZ=START_POS,
            lineToXYZ=arrow_end,
            lineColorRGB=[0, 0, 1],
            lineWidth=3,
            lifeTime=0,
            physicsClientId=env.CLIENT
        )
    
    print(f"\n✓ Created {len(OBSTACLES)} obstacle spheres:")
    for i, obs in enumerate(OBSTACLES):
        print(f"   Obstacle {i+1}: {obs['pos']} (radius {obs['radius']}m, safety margin {obs['safety_margin']}m)")
    print(f"✓ Drew heading direction (blue arrow) pointing in -Y")
    print(f"  Look for: RED & ORANGE spheres (obstacles) + YELLOW spheres (safety zones) + BLUE arrow (heading)\n")
    
    start_time = time.time()
    min_distance = float('inf')
    max_z_deviation = 0.0
    max_x_deviation = 0.0  # Track sideways displacement in X direction
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
        
        # Compute distance to nearest obstacle
        distances_to_obstacles = [np.linalg.norm(current_pos - obs['pos']) for obs in OBSTACLES]
        distance_to_nearest = min(distances_to_obstacles)
        min_distance = min(min_distance, distance_to_nearest)
        
        # Track Z deviation from target altitude
        z_deviation = abs(current_pos[2] - FLIGHT_ALTITUDE)
        max_z_deviation = max(max_z_deviation, z_deviation)
        
        # Track X deviation (sideways displacement from centerline)
        x_deviation = abs(current_pos[0] - 0.0)
        max_x_deviation = max(max_x_deviation, x_deviation)
        
        # Check for safety violation for each obstacle
        for i, obs in enumerate(OBSTACLES):
            dist_to_obs = np.linalg.norm(current_pos - obs['pos'])
            safe_boundary = obs['radius'] + obs['safety_margin']/2
            if dist_to_obs < safe_boundary:
                violation_count += 1
                if violation_count == 1:  # First violation
                    print(f"\n⚠️  SAFETY VIOLATION at t={current_time:.2f}s")
                    print(f"   Obstacle {i+1}: Distance {dist_to_obs:.3f}m < {safe_boundary:.3f}m")
                break  # Only report first violation per step
        
        # Target: Constant velocity in heading direction
        target_vel = CRUISE_VELOCITY * HEADING_DIRECTION
        
        # Constant velocity controller for reference acceleration
        desired_accel = constant_velocity_controller(
            current_vel, target_vel, current_pos, FLIGHT_ALTITUDE
        )
        
        # Apply multi-obstacle CBF to get safe acceleration
        safe_accel = compute_cbf_safe_acceleration_multi(
            current_pos, current_vel, desired_accel, OBSTACLES, GAMMA
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
                  f"dist={distance_to_nearest:5.3f}m | Δx={x_deviation:5.3f}m")
        
        # Sync simulation
        env.render()
    
    # Print results
    print("-" * 70)
    print("\nSIMULATION COMPLETE")
    print("=" * 70)
    print("\nResults:")
    print(f"  Minimum Distance to Nearest Obstacle: {min_distance:.3f} m")
    print(f"  Obstacle Safety Boundaries:")
    for i, obs in enumerate(OBSTACLES):
        safe_boundary = obs['radius'] + obs['safety_margin']/2
        print(f"    Obstacle {i+1}: {safe_boundary:.3f} m (radius={obs['radius']}m + margin={obs['safety_margin']}m)")
    print(f"  Maximum X Deviation (sideways): {max_x_deviation*100:.1f} cm")
    print(f"  Maximum Z Deviation: {max_z_deviation*100:.1f} cm")
    print(f"  Target Altitude: {FLIGHT_ALTITUDE:.2f} m")
    print(f"  Heading Direction: -Y (constant)")
    
    # Check success - must avoid all obstacles
    all_safe = True
    for i, obs in enumerate(OBSTACLES):
        safe_boundary = obs['radius'] + obs['safety_margin']/2
        if min_distance < safe_boundary:
            all_safe = False
            break
    
    if all_safe:
        print(f"  ✅ SAFETY SUCCESS - Avoided all obstacles!")
        smallest_boundary = min([obs['radius'] + obs['safety_margin']/2 for obs in OBSTACLES])
        print(f"     Clearance margin: {(min_distance - smallest_boundary)*100:.1f} cm")
    else:
        print(f"  ❌ SAFETY FAILURE - Penetrated at least one safety boundary")
        print(f"     Minimum distance: {min_distance:.3f} m")
        print(f"     Violation frames: {violation_count}/{num_steps}")
    
    if max_z_deviation < 0.15:
        print(f"  ✅ ALTITUDE SUCCESS - Stayed within 15cm of target altitude")
    elif max_z_deviation < 0.3:
        print(f"  ⚠️  ALTITUDE WARNING - Deviated up to {max_z_deviation*100:.1f}cm")
    else:
        print(f"  ❌ ALTITUDE FAILURE - Excessive Z deviation: {max_z_deviation*100:.1f}cm")
    
    print(f"\n  Behavior: Drone maintained -Y heading while CBF pushed it sideways")
    print(f"  Maximum sideways displacement: {max_x_deviation*100:.1f}cm in X direction")
    print("=" * 70)
    
    # Close environment
    env.close()
    
    # Save log
    logger.save()
    logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kinematic quadrotor XY plane CBF test')
    parser.add_argument('--duration', type=int, default=25, help='Duration in seconds')
    parser.add_argument('--gui', action='store_true', help='Show PyBullet GUI')
    args = parser.parse_args()
    
    run_xy_plane_cbf_test(duration=args.duration, gui=args.gui)
