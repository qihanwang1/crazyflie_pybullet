"""
CBF-QP Control Example for Crazyflie Simulation

This script demonstrates how to use the CFCBFAviary environment with your CBF control code.
It's designed to mimic your hardware interface while running in simulation.
"""
import numpy as np
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.CFCBFAviary import CFCBFAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger

def main():
    """
    Main function to run the Crazyflie simulation with CBF-QP controller.
    Similar structure to your hardware interface.
    """
    # Path to drone configuration JSON file
    """Run the CF-CBF control example."""
    
    # Drone configuration path
    drone_cfg_path = 'gym_pybullet_drones/envs/CBF_CONTROL/drone1.json'
    
    # CBF controller parameters
    cbf_params = {
        'gamma': 1.0,        # Class K function parameter
        'obs_radius': 0.1    # Obstacle radius in meters
    }
    
    # Initial position and orientation (must match drone1.json)
    # drone1.json: x=0, y=0, z=1
    initial_xyz = np.array([[0.0, 0.0, 1.0]])
    initial_rpy = np.array([[0.0, 0.0, 0.0]])
    
    # Create environment
    print("Initializing CFCBFAviary environment...")
    env = CFCBFAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=initial_xyz,
        initial_rpys=initial_rpy,
        physics=Physics.PYB,
        pyb_freq=500,
        ctrl_freq=25,  # 25Hz control loop (40ms)
        gui=True,
        user_debug_gui=True,
        drone_config_path=drone_cfg_path,
        cbf_params=cbf_params,
        verbose=False  # Set to True for detailed debug output
    )
    
    # Initialize logger
    logger = Logger(
        logging_freq_hz=25,
        num_drones=1,
        output_folder='results'
    )
    
    # Create CSV file for CBF control logging
    cbf_log_file = open('results/cbf_control_log.csv', 'w')
    cbf_log_file.write('time,x,y,z,vx,vy,vz,obs_x,obs_y,obs_z,'
                      'u_ref_0,u_ref_1,u_ref_2,u_ref_3,u_ref_sum,'
                      'rpm0,rpm1,rpm2,rpm3,rpm_avg,thrust_avg,'
                      'traj_x,traj_y,traj_z,traj_vx,traj_vy,traj_vz\n')
    
    # Reset environment
    obs, info = env.reset()
    print("\nEnvironment initialized successfully!")
    print(f"Initial state: {env.get_drone_state_dict()}")
    
    # Control loop parameters
    dt = 1 / env.CTRL_FREQ  # Time step (40ms for 25Hz)
    num_steps = 750  # 30 seconds at 25Hz
    
    # Obstacle (static for now - you can make this dynamic)
    # Drone starts at [0, 0, 1], obstacle is at [1, 0.5, 0.5] - separated by ~1.2m
    # obstacle_pos = np.array([5.0, 0.5, 0.5])  # [x, y, z] in meters
    # obstacle_vel = np.array([0.0, 0.0, 0.0])   # [vx, vy, vz] in m/s
    obstacle_pos = None
    obstacle_vel = None
    
    print(f"\nStarting control loop for {num_steps} steps ({num_steps * dt:.1f} seconds)")
    print(f"Obstacle at: {obstacle_pos}")
    print("\nPress Ctrl+C to stop...\n")
    
    # Define a simple trajectory (optional - set to None for hover only)
    # Options:
    # 1. None - just hover at initial position
    # 2. 'circle' - circular trajectory
    # 3. 'line' - straight line trajectory
    # 4. 'figure8' - figure-8 trajectory
    trajectory_type = 'line'  # Change this to enable trajectory tracking
    
    def get_reference_trajectory(t, traj_type=None):
        """
        Get desired position and velocity at time t.
        
        Returns:
            pos_des: [x, y, z] desired position
            vel_des: [vx, vy, vz] desired velocity
        """
        if traj_type is None:
            # Hover at initial position
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])
        
        elif traj_type == 'circle':
            # Circular trajectory in xy-plane at z=1.0
            radius = 0.5  # meters
            omega = 0.5   # rad/s (period = 2*pi/omega ≈ 12.6 seconds)
            x_des = radius * np.cos(omega * t)
            y_des = radius * np.sin(omega * t)
            z_des = 1.0
            vx_des = -radius * omega * np.sin(omega * t)
            vy_des = radius * omega * np.cos(omega * t)
            vz_des = 0.0
            return np.array([x_des, y_des, z_des]), np.array([vx_des, vy_des, vz_des])
        
        elif traj_type == 'line':
            # Straight line - START WITH VERTICAL ONLY for stability
            # Once this works well, gradually add horizontal movement
            duration = 20.0
            t_clamped = min(t, duration)
            alpha = t_clamped / duration
            
            # Option 1: Vertical only (RECOMMENDED TO START)
            x_des = 0.0  # No horizontal movement
            y_des = 0.0  # No horizontal movement
            z_des = 1.0 + alpha * 0.5  # Climb from 1.0m to 1.5m
            vx_des = 0.0
            vy_des = 0.0
            vz_des = 0.5 / duration if t < duration else 0.0
            
            # Option 2: Full 3D (use once vertical works well)
            # x_des = 0.0 + alpha * 1.0
            # y_des = 0.0 + alpha * 1.0
            # z_des = 1.0 + alpha * 0.5
            # vx_des = 1.0 / duration if t < duration else 0.0
            # vy_des = 1.0 / duration if t < duration else 0.0
            # vz_des = 0.5 / duration if t < duration else 0.0
            
            return np.array([x_des, y_des, z_des]), np.array([vx_des, vy_des, vz_des])
        
        elif traj_type == 'figure8':
            # Figure-8 trajectory in xy-plane at z=1.0
            a = 0.5  # semi-major axis
            omega = 0.4  # rad/s
            x_des = a * np.sin(omega * t)
            y_des = a * np.sin(2 * omega * t) / 2
            z_des = 1.0
            vx_des = a * omega * np.cos(omega * t)
            vy_des = a * omega * np.cos(2 * omega * t)
            vz_des = 0.0
            return np.array([x_des, y_des, z_des]), np.array([vx_des, vy_des, vz_des])
        
        else:
            # Default to hover
            return np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])
    
    try:
        start_time = time.time()
        
        for i in range(num_steps):
            current_time = i * dt
            
            # Get current drone state
            state = env.get_drone_state_dict()
            current_pos = np.array([state['x'], state['y'], state['z']])
            current_vel = np.array([state['vx'], state['vy'], state['vz']])
            
            # Get reference trajectory
            pos_des, vel_des = get_reference_trajectory(current_time, trajectory_type)
            
            # Compute position and velocity errors
            pos_error = pos_des - current_pos
            vel_error = vel_des - current_vel
            
            # ========================================================================
            # CONVERT TRAJECTORY TO THRUST COMMANDS
            # Use PD control to compute desired total thrust based on tracking errors
            # ========================================================================
            
            # Controller gains (tune these for your system)
            Kp_z = 2.0   # Vertical position gain (N/m)
            Kd_z = 1.5   # Vertical velocity gain (N·s/m)
            
            # For horizontal control, we'll use simplified gains
            # Note: This assumes small angles (drone stays roughly level)
            # For aggressive maneuvers, you'd need full attitude control
            Kp_xy = 0.05  # Horizontal position gain (N/m) - very small!
            Kd_xy = 0.03  # Horizontal velocity gain (N·s/m) - very small!
            
            # Compute vertical thrust (Z-axis)
            # Feedforward: compensate gravity
            # Feedback: PD control on z position/velocity errors
            # Note: env.GRAVITY is already m*g (weight), not g!
            feedforward_thrust_z = env.GRAVITY  # This is the total weight
            feedback_thrust_z = Kp_z * pos_error[2] + Kd_z * vel_error[2]
            total_thrust = feedforward_thrust_z + feedback_thrust_z
            
            # Compute horizontal corrections (X, Y axes)
            # These will create small thrust imbalances to tilt the drone
            # WARNING: This is a simplified approach and may be unstable!
            thrust_correction_x = Kp_xy * pos_error[0] + Kd_xy * vel_error[0]
            thrust_correction_y = Kp_xy * pos_error[1] + Kd_xy * vel_error[1]
            
            # Distribute thrust to motors with corrections for horizontal control
            # Motor layout (X configuration):
            #   1 (FL)    0 (FR)
            #        \ /
            #        / \
            #   2 (RL)    3 (RR)
            #
            # Pitch (X): increase rear (2,3), decrease front (0,1)
            # Roll (Y): increase left (1,2), decrease right (0,3)
            
            base_thrust = total_thrust / 4
            u_ref = np.array([
                [base_thrust - thrust_correction_x - thrust_correction_y],  # FR: 0
                [base_thrust - thrust_correction_x + thrust_correction_y],  # FL: 1
                [base_thrust + thrust_correction_x + thrust_correction_y],  # RL: 2
                [base_thrust + thrust_correction_x - thrust_correction_y]   # RR: 3
            ])
            
            # Clamp to safe thrust values (prevent negative or excessive thrust)
            # Min: 50% of hover thrust per motor, Max: 150% of hover thrust per motor
            # Note: env.GRAVITY is the total weight (m*g), not g!
            hover_thrust_per_motor = env.GRAVITY / 4
            min_thrust = 0.50 * hover_thrust_per_motor
            max_thrust = 1.50 * hover_thrust_per_motor
            u_ref = np.clip(u_ref, min_thrust, max_thrust)
            
            # Step with CBF control
            # This will:
            # 1. Run your CBF-QP controller
            # 2. Get safe control
            # 3. Apply to simulation
            # 4. Update drone state
            obs, reward, terminated, truncated, info = env.step_with_cbf(
                obstacle_pos, obstacle_vel, u_ref
            )
            
            # Log the simulation (obs is now 20-element state vector)
            logger.log(
                drone=0,
                timestamp=i / env.CTRL_FREQ,
                state=obs
            )
            
            # Log CBF control data to CSV
            if 'cbf_control' in info:
                cbf = info['cbf_control']
                kf = 3.16e-10
                avg_rpm = np.mean(cbf['u_opt_rpm'])
                avg_thrust = avg_rpm**2 * kf
                
                # Extract individual u_ref components (they're in nested array format)
                u_ref_flat = u_ref.flatten()
                
                # Handle None obstacle (when CBF is bypassed)
                if cbf['obstacle_pos'] is None:
                    obs_x, obs_y, obs_z = 0.0, 0.0, 0.0
                else:
                    obs_x, obs_y, obs_z = cbf['obstacle_pos'][0], cbf['obstacle_pos'][1], cbf['obstacle_pos'][2]
                
                cbf_log_file.write(f"{current_time:.3f},"
                                 f"{cbf['drone_pos'][0]:.4f},{cbf['drone_pos'][1]:.4f},{cbf['drone_pos'][2]:.4f},"
                                 f"{cbf['drone_vel'][0]:.4f},{cbf['drone_vel'][1]:.4f},{cbf['drone_vel'][2]:.4f},"
                                 f"{obs_x:.4f},{obs_y:.4f},{obs_z:.4f},"
                                 f"{u_ref_flat[0]:.6f},{u_ref_flat[1]:.6f},{u_ref_flat[2]:.6f},{u_ref_flat[3]:.6f},"
                                 f"{np.sum(u_ref_flat):.6f},"
                                 f"{cbf['u_opt_rpm'][0]:.2f},{cbf['u_opt_rpm'][1]:.2f},"
                                 f"{cbf['u_opt_rpm'][2]:.2f},{cbf['u_opt_rpm'][3]:.2f},"
                                 f"{avg_rpm:.2f},{avg_thrust:.6f},"
                                 f"{pos_des[0]:.4f},{pos_des[1]:.4f},{pos_des[2]:.4f},"
                                 f"{vel_des[0]:.4f},{vel_des[1]:.4f},{vel_des[2]:.4f}\n")
            
            # Print state every second (25 steps)
            if i % 25 == 0:
                elapsed = time.time() - start_time
                rpm_str = ""
                traj_str = ""
                cbf_str = ""
                
                if 'cbf_control' in info:
                    rpms = info['cbf_control']['u_opt_rpm']
                    rpm_str = f"| RPM=[{rpms[0]:.0f}, {rpms[1]:.0f}, {rpms[2]:.0f}, {rpms[3]:.0f}]"
                    
                    # Show if CBF was bypassed
                    if info['cbf_control'].get('cbf_bypassed', False):
                        cbf_str = " [CBF BYPASSED - using reference trajectory directly]"
                
                if trajectory_type is not None:
                    err_pos = np.linalg.norm(current_pos - pos_des)
                    err_vel = np.linalg.norm(current_vel - vel_des)
                    traj_str = f"| pos_err={err_pos:.3f}m vel_err={err_vel:.3f}m/s"
                    traj_str += f" | target_z={pos_des[2]:.2f}m"
                
                print(f"t={current_time:5.2f}s | pos=[{state['x']:5.2f}, {state['y']:5.2f}, {state['z']:5.2f}] | "
                      f"vel=[{state['vx']:5.2f}, {state['vy']:5.2f}, {state['vz']:5.2f}] {rpm_str} {traj_str}{cbf_str}")
            
            # Render the simulation
            env.render()
            
            # Sleep to maintain real-time (optional - remove for faster sim)
            # time.sleep(dt)
            
    except KeyboardInterrupt:
        print("\n\nControl loop interrupted by user")
    finally:
        # Close CBF log file
        cbf_log_file.close()
        
        # Close environment
        env.close()
        
        # Save logs
        logger.save()
        logger.save_as_csv("cf_cbf")
        
        print("\nSimulation complete!")
        print(f"Logs saved to: results/")
        print(f"  - Standard flight logs: results/save-flight-cf_cbf-*.csv")
        print(f"  - CBF control log: results/cbf_control_log.csv")
        
        # Plot results
        try:
            logger.plot()
        except Exception as e:
            print(f"Could not plot results: {e}")

if __name__ == '__main__':
    main()
