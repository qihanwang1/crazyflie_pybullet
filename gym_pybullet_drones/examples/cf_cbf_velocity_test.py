"""
Test script for Crazyflie with velocity-based CBF-QP control.

This script demonstrates the integrated CBF-QP controller that operates at the
velocity command level (from the PolyC2BF paper). The controller:
  1. Takes velocity commands [vx, vy, vz, yaw_rate]
  2. Applies CBF-QP to ensure safety constraints
  3. Converts safe velocity commands to motor RPMs
  4. Executes on actual Crazyflie dynamics in PyBullet

Uses CFCBFAviaryDynamics which integrates QP_dynamics_controller.py for
velocity-level CBF control (as opposed to the RPM-based QP_controller_drone.py).

Author: GitHub Copilot
Date: November 2025
"""

import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CFCBFAviaryDynamics import CFCBFAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def compute_reference_velocity(current_pos, current_time, waypoints, cruise_velocity=0.3):
    """
    Compute reference velocity command to follow waypoint trajectory.
    
    Parameters
    ----------
    current_pos : ndarray
        Current position [x, y, z]
    current_time : float
        Current simulation time
    waypoints : list of dict
        Waypoints with 'pos', 'time' keys
    cruise_velocity : float
        Desired cruise speed
        
    Returns
    -------
    ndarray
        Reference velocity [vx, vy, vz, yaw_rate]
    """
    # Find current target waypoint
    target_waypoint = waypoints[0]
    for wp in waypoints:
        if current_time < wp['time']:
            target_waypoint = wp
            break
    
    target_pos = target_waypoint['pos']
    
    # Compute direction to target
    direction = target_pos - current_pos
    distance = np.linalg.norm(direction)
    
    if distance > 0.1:  # Not at target yet
        direction_normalized = direction / distance
        # Scale velocity by distance (slow down near target)
        speed = min(cruise_velocity, distance * 2.0)
        vel_cmd = speed * direction_normalized
    else:
        vel_cmd = np.zeros(3)
    
    # Zero yaw rate (maintain heading)
    yaw_rate = 0.0
    
    return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_rate])


def run_velocity_cbf_test(duration=30, gui=True):
    """
    Run Crazyflie test with velocity-based CBF-QP control.
    
    Parameters
    ----------
    duration : int
        Total simulation time in seconds
    gui : bool
        Whether to show PyBullet GUI
    """
    # Environment parameters
    CTRL_FREQ = 30  # 30 Hz control loop (realistic for Crazyflie, divisible into 240)
    
    # Obstacle configuration - REMOVED FOR STABILITY TESTING
    OBSTACLES = []
    
    # CBF parameters
    CBF_GAMMA = 1.5  # Class-K function parameter
    OBS_RADIUS = 0.1  # Base obstacle radius for CBF (will be added to actual radius)
    
    # Flight parameters - Waypoint trajectory
    CRUISE_VELOCITY = 0.3  # m/s
    FLIGHT_ALTITUDE = 1.0  # meters
    
    # Define waypoint trajectory
    WAYPOINTS = [
        {'pos': np.array([0.0, 0.0, FLIGHT_ALTITUDE]), 'time': 0.0},
        {'pos': np.array([1.0, 0.0, FLIGHT_ALTITUDE]), 'time': 5.0},
        {'pos': np.array([2.0, 0.5, FLIGHT_ALTITUDE]), 'time': 10.0},
        {'pos': np.array([3.0, 0.0, FLIGHT_ALTITUDE]), 'time': 15.0},
        {'pos': np.array([3.0, 0.0, FLIGHT_ALTITUDE]), 'time': duration}  # Hold at end
    ]
    
    START_POS = WAYPOINTS[0]['pos']
    
    print("=" * 80)
    print("CRAZYFLIE VELOCITY-BASED CBF-QP TEST")
    print("=" * 80)
    print("\nTest Configuration:")
    print(f"  Control Frequency: {CTRL_FREQ} Hz (Crazyflie realistic)")
    print(f"  Physics Frequency: 240 Hz (PyBullet)")
    print(f"  Control Steps per Physics Step: {240//CTRL_FREQ}")
    print(f"  CBF Gamma: {CBF_GAMMA}")
    print(f"  Number of Obstacles: {len(OBSTACLES)}")
    for i, obs in enumerate(OBSTACLES):
        total_safe_dist = obs['radius'] + obs['safety_margin']
        print(f"  Obstacle {i+1}: pos={obs['pos']}, radius={obs['radius']}m, safe_dist={total_safe_dist:.2f}m")
    print(f"  Duration: {duration} seconds")
    print("\nFlight Profile:")
    print(f"  Cruise Velocity: {CRUISE_VELOCITY} m/s")
    print(f"  Flight Altitude: {FLIGHT_ALTITUDE} m")
    print(f"  Number of Waypoints: {len(WAYPOINTS)}")
    for i, wp in enumerate(WAYPOINTS):
        print(f"    WP{i}: {wp['pos']} @ t={wp['time']}s")
    print("\nControl Architecture:")
    print("  1. High-level: Waypoint → Reference Velocity [vx, vy, vz, yaw_rate]")
    print("  2. Mid-level: CBF-QP → Safe Velocity (obstacle avoidance)")
    print("  3. Low-level: PD Controller → Motor RPMs")
    print("  4. Physics: Full Crazyflie 12-state dynamics in PyBullet")
    print("=" * 80)
    
    # Create environment with CBF parameters
    env = CFCBFAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=START_POS.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=240,  # Physics simulation frequency
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
        cbf_params={
            'gamma': CBF_GAMMA,
            'obs_radius': OBS_RADIUS
        }
    )
    
    # Initialize logger
    logger = Logger(
        logging_freq_hz=CTRL_FREQ,
        num_drones=1,
        duration_sec=duration
    )
    
    # Reset environment
    obs, info = env.reset()
    
    # Create obstacles in PyBullet (after reset so they persist)
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
        total_safe_dist = obstacle['radius'] + obstacle['safety_margin']
        safety_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=total_safe_dist,
            rgbaColor=[1, 1, 0, 0.2],
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
    
    # Draw waypoint trajectory
    if gui:
        for i in range(len(WAYPOINTS) - 1):
            p.addUserDebugLine(
                lineFromXYZ=WAYPOINTS[i]['pos'],
                lineToXYZ=WAYPOINTS[i+1]['pos'],
                lineColorRGB=[0, 1, 0],
                lineWidth=2,
                lifeTime=0,
                physicsClientId=env.CLIENT
            )
    
    print(f"\n✓ Created {len(OBSTACLES)} obstacles")
    print(f"✓ Drew waypoint trajectory (green line)")
    print(f"  Look for: Colored spheres (obstacles) + Yellow zones (safety) + Green line (path)\n")
    
    # Simulation tracking variables
    start_time = time.time()
    min_distance = float('inf')
    max_z_deviation = 0.0
    violation_count = 0
    cbf_activation_count = 0
    
    num_steps = int(duration * CTRL_FREQ)
    
    print("\nStarting simulation...")
    print("-" * 80)
    
    for step in range(num_steps):
        current_time = step / CTRL_FREQ
        
        # Get current state
        state = env._getDroneStateVector(0)
        current_pos = state[0:3]
        current_vel = state[10:13]
        
        # Compute distance to each obstacle (only if obstacles exist)
        if len(OBSTACLES) > 0:
            distances = [np.linalg.norm(current_pos - obs['pos']) for obs in OBSTACLES]
            min_dist = min(distances)
            min_distance = min(min_distance, min_dist)
            
            # Check for safety violations
            for i, obs in enumerate(OBSTACLES):
                dist = distances[i]
                safe_boundary = obs['radius'] + obs['safety_margin']
                if dist < safe_boundary:
                    violation_count += 1
                    if violation_count == 1:
                        print(f"\n⚠️  SAFETY VIOLATION at t={current_time:.2f}s")
                        print(f"   Obstacle {i+1}: Distance {dist:.3f}m < {safe_boundary:.3f}m")
                    break
        else:
            min_dist = float('inf')  # No obstacles
        
        # Track Z deviation
        z_deviation = abs(current_pos[2] - FLIGHT_ALTITUDE)
        max_z_deviation = max(max_z_deviation, z_deviation)
        
        # Compute reference velocity from waypoint tracker
        u_ref_vel = compute_reference_velocity(
            current_pos, current_time, WAYPOINTS, CRUISE_VELOCITY
        )
        
        # Prepare obstacle data for CBF-QP
        # Format: list of positions [[x1,y1,z1], [x2,y2,z2], ...]
        obstacle_positions = [obs['pos'] for obs in OBSTACLES]
        obstacle_velocities = [obs['vel'] for obs in OBSTACLES]
        
        # Execute step with CBF control
        # This will:
        #   1. Apply CBF-QP to get safe velocity
        #   2. Convert to motor RPMs
        #   3. Simulate one step
        obs, reward, terminated, truncated, info = env.step_with_cbf(
            obstacle_pos=obstacle_positions,
            obstacle_vel=obstacle_velocities,
            u_ref_vel=u_ref_vel  # Reference velocity command
        )
        
        # Check if CBF was active
        if 'cbf_control' in info:
            cbf_data = info['cbf_control']
            u_opt = cbf_data['u_opt_vel']
            u_ref = cbf_data['u_ref_vel']
            
            # CBF is active if optimal velocity differs from reference
            if np.linalg.norm(u_opt[:3] - u_ref[:3]) > 0.01:
                cbf_activation_count += 1
        
        # Log data
        logger.log(
            drone=0,
            timestamp=current_time,
            state=state,
            control=np.zeros(12)  # Dummy control
        )
        
        # Print progress every 2 seconds
        if step % (2 * CTRL_FREQ) == 0:
            vel_mag = np.linalg.norm(current_vel)
            print(f"t={current_time:5.1f}s | pos=[{current_pos[0]:5.2f}, {current_pos[1]:5.2f}, {current_pos[2]:5.2f}] | "
                  f"vel={vel_mag:.2f}m/s | min_dist={min_dist:5.3f}m")
        
        # Render
        env.render()
        
        if terminated or truncated:
            print(f"\nSimulation terminated at t={current_time:.2f}s")
            break
    
    # Print results
    print("-" * 80)
    print("\nSIMULATION COMPLETE")
    print("=" * 80)
    print("\nResults:")
    print(f"  Minimum Distance to Any Obstacle: {min_distance:.3f} m")
    
    # Check each obstacle
    all_safe = True
    for i, obs in enumerate(OBSTACLES):
        safe_boundary = obs['radius'] + obs['safety_margin']
        print(f"  Obstacle {i+1} Safety Boundary: {safe_boundary:.3f} m (radius={obs['radius']}m + margin={obs['safety_margin']}m)")
        if min_distance < safe_boundary:
            all_safe = False
    
    print(f"  Maximum Z Deviation: {max_z_deviation*100:.1f} cm")
    print(f"  CBF Activation Frames: {cbf_activation_count}/{num_steps} ({100*cbf_activation_count/num_steps:.1f}%)")
    
    if len(OBSTACLES) > 0:
        if all_safe:
            print(f"\n  ✅ SAFETY SUCCESS - Avoided all obstacles!")
            smallest_boundary = min([obs['radius'] + obs['safety_margin'] for obs in OBSTACLES])
            print(f"     Clearance margin: {(min_distance - smallest_boundary)*100:.1f} cm")
        else:
            print(f"\n  ❌ SAFETY FAILURE - Penetrated at least one safety boundary")
            print(f"     Violation frames: {violation_count}/{num_steps}")
    else:
        print(f"\n  ℹ️  NO OBSTACLES - Stability test only")
    
    if max_z_deviation < 0.2:
        print(f"  ✅ ALTITUDE SUCCESS - Stayed within 20cm of target")
    else:
        print(f"  ⚠️  ALTITUDE WARNING - Deviated up to {max_z_deviation*100:.1f}cm")
    
    print("\nControl Performance:")
    print(f"  Control Architecture: Velocity-based CBF-QP → PD → Motor RPMs")
    print(f"  Physics: Full 12-state Crazyflie dynamics")
    print(f"  CBF activated {100*cbf_activation_count/num_steps:.1f}% of the time")
    print("=" * 80)
    
    # Close environment
    env.close()
    
    # Save and plot logs
    print("\nSaving logs and generating plots...")
    logger.save()
    logger.plot()
    print("✓ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Crazyflie velocity-based CBF-QP control test'
    )
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duration in seconds (default: 30)')
    parser.add_argument('--gui', action='store_true', 
                       help='Show PyBullet GUI')
    args = parser.parse_args()
    
    run_velocity_cbf_test(duration=args.duration, gui=args.gui)
