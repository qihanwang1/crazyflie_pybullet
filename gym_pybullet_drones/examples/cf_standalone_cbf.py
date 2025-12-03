"""Standalone QP-CBF control example without firmware wrapper.

This demonstrates the original standalone approach where:
1. User provides velocity commands directly
2. QP-CBF filters velocities for safety
3. Commands sent directly to drone

This is simpler than the CFAviary integration and easier to debug.

Example
-------
In terminal, run:
    python gym_pybullet_drones/examples/cf_standalone_cbf.py --gui
"""
import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # Use concrete implementation
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

# Import QP-CBF controller
from gym_pybullet_drones.envs.CBF_CONTROL.QP_dynamics_controller import QP_Controller_Drone
from gym_pybullet_drones.envs.CBF_CONTROL.drone import Drone

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'

NUM_DRONES = 1
INIT_XYZ = np.array([[0.0, 0.0, 1.0]])
INIT_RPY = np.array([[0.0, 0.0, 0.0]])

def compute_velocity_command(current_pos, target_pos, max_vel=0.5):
    """Compute desired velocity to reach target position.
    
    This is what the USER does in standalone mode - calculate velocities.
    In CFAviary mode, Mellinger does this automatically.
    
    Parameters
    ----------
    current_pos : ndarray
        Current position [x, y, z]
    target_pos : ndarray
        Target position [x, y, z]
    max_vel : float
        Maximum velocity magnitude
        
    Returns
    -------
    ndarray
        Desired velocity [vx, vy, vz]
    """
    error = target_pos - current_pos
    distance = np.linalg.norm(error)
    
    if distance < 0.05:  # Close enough
        return np.zeros(3)
    
    # Simple proportional control: velocity proportional to error
    Kp = 1.0
    desired_vel = Kp * error
    
    # Limit velocity magnitude
    vel_magnitude = np.linalg.norm(desired_vel)
    if vel_magnitude > max_vel:
        desired_vel = desired_vel / vel_magnitude * max_vel
    
    return desired_vel

class VelocityController:
    """PID velocity controller that mimics IMU + flow deck stabilization.
    
    This provides the inner-loop stabilization that real drones have from:
    - IMU (attitude stabilization)
    - Flow deck (velocity estimation and control)
    - Mellinger controller (in CFAviary)
    """
    
    def __init__(self, mass=0.027):
        self.mass = mass
        self.g = 9.81
        self.kf = 3.16e-10
        
        # PID gains for velocity tracking (tuned for stability)
        self.kp_vel = np.array([2.5, 2.5, 5.0])  # Proportional (slightly increased)
        self.ki_vel = np.array([0.3, 0.3, 1.0])  # Integral (reduced for less overshoot)
        self.kd_vel = np.array([0.8, 0.8, 1.5])  # Derivative (increased for damping)
        
        # PID gains for attitude (roll/pitch stabilization)
        self.kp_att = 70.0
        self.ki_att = 0.0
        self.kd_att = 20.0
        
        # State tracking
        self.integral_vel_error = np.zeros(3)
        self.prev_vel_error = np.zeros(3)
        self.prev_time = None
        
        # Anti-windup
        self.max_integral = 1.5  # Reduced for better stability
        
        # Limits
        self.max_tilt = np.radians(20)  # 20 degrees max tilt (reduced for stability)
        
    def reset(self):
        """Reset controller state."""
        self.integral_vel_error = np.zeros(3)
        self.prev_vel_error = np.zeros(3)
        self.prev_time = None
        
    def compute_control(self, desired_vel, current_vel, current_rpy, dt):
        """Compute motor RPMs from desired velocity.
        
        This mimics what IMU + flow deck + inner controller does on real drone.
        
        Parameters
        ----------
        desired_vel : ndarray
            Desired velocity [vx, vy, vz] in m/s
        current_vel : ndarray
            Current velocity [vx, vy, vz] in m/s
        current_rpy : ndarray
            Current roll, pitch, yaw in radians
        dt : float
            Time step in seconds
            
        Returns
        -------
        ndarray
            Motor RPMs [rpm1, rpm2, rpm3, rpm4]
        """
        # Velocity error
        vel_error = desired_vel - current_vel
        
        # PID for velocity
        # Proportional
        p_term = self.kp_vel * vel_error
        
        # Integral (with anti-windup)
        self.integral_vel_error += vel_error * dt
        self.integral_vel_error = np.clip(
            self.integral_vel_error, 
            -self.max_integral, 
            self.max_integral
        )
        i_term = self.ki_vel * self.integral_vel_error
        
        # Derivative
        if self.prev_time is not None:
            d_term = self.kd_vel * (vel_error - self.prev_vel_error) / dt
        else:
            d_term = np.zeros(3)
        
        self.prev_vel_error = vel_error
        self.prev_time = dt
        
        # Acceleration command
        acc_cmd = p_term + i_term + d_term
        
        # Convert to thrust and attitude
        # Hover thrust
        thrust = self.mass * self.g + self.mass * acc_cmd[2]  # Z-axis
        
        # Desired tilt angles (small angle approximation)
        # For forward velocity, pitch backward (negative)
        # For rightward velocity, roll right (positive)
        roll, pitch, yaw = current_rpy
        
        desired_pitch = -np.arctan2(acc_cmd[0], self.g)  # Forward/back
        desired_roll = np.arctan2(acc_cmd[1], self.g)    # Left/right
        
        # Limit tilt
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)
        desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
        
        # Attitude error
        roll_error = desired_roll - roll
        pitch_error = desired_pitch - pitch
        
        # PD control for attitude (fast inner loop)
        roll_torque = self.kp_att * roll_error
        pitch_torque = self.kp_att * pitch_error
        yaw_torque = 0  # Keep yaw constant
        
        # Convert to motor commands (X configuration)
        # Motor layout: 0=front-right, 1=rear-left, 2=rear-right, 3=front-left
        thrust_per_motor = thrust / 4.0
        
        # Add attitude corrections
        motor_thrusts = np.array([
            thrust_per_motor - pitch_torque + roll_torque,  # Front-right
            thrust_per_motor + pitch_torque - roll_torque,  # Rear-left
            thrust_per_motor + pitch_torque + roll_torque,  # Rear-right
            thrust_per_motor - pitch_torque - roll_torque,  # Front-left
        ])
        
        # Convert thrust to RPM
        motor_thrusts = np.clip(motor_thrusts, 0, None)
        rpm = np.sqrt(motor_thrusts / self.kf)
        rpm = np.clip(rpm, 0, 21702)  # Crazyflie max
        
        return rpm

def run(
        drone=DEFAULT_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        ):
    
    #### Create simple CtrlAviary environment (no firmware) ####
    env = CtrlAviary(
        drone_model=drone,
        num_drones=NUM_DRONES,
        initial_xyzs=INIT_XYZ,
        initial_rpys=INIT_RPY,
        physics=physics,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
    )
    
    #### Initialize QP-CBF controller ####
    import os
    drone_cfg_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 'envs', 'CBF_CONTROL', 'drone1.json'
    )
    
    cbf_drone = Drone.from_JSON(drone_cfg_path)
    qp_controller = QP_Controller_Drone(gamma=1.5, obs_radius=0.1)
    
    #### Initialize velocity controller (mimics IMU + flow deck) ####
    drone_params = {'mass': 0.027}  # Crazyflie mass
    vel_controller = VelocityController(mass=drone_params['mass'])
    
    print("\n" + "="*80)
    print("STANDALONE QP-CBF CONTROL (With Stabilization)")
    print("="*80)
    print("Architecture:")
    print("  User → Velocity Cmd → QP-CBF Filter → Velocity Controller → Motors")
    print("                                         (PID - mimics IMU/flow deck)")
    print("\nThis mimics real drone hardware:")
    print("  - Velocity Controller = IMU + flow deck stabilization")
    print("  - Provides attitude control and velocity tracking")
    print("  - Makes the system stable like real hardware!")
    print("="*80 + "\n")
    
    #### Get PyBullet client ####
    PYB_CLIENT = env.getPyBulletClient()
    
    #### Initialize logger ####
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=NUM_DRONES,
        output_folder=output_folder,
    )
    
    #### Define trajectory (same as before) ####
    # Waypoints to visit
    waypoints = [
        np.array([0.0, 0.0, 1.0]),    # Start (hover)
        np.array([1.5, 0.0, 1.0]),    # Move forward through obstacles
        np.array([2.0, 0.0, 1.0]),    # Further forward
        np.array([1.5, 0.0, 1.0]),    # Return
        np.array([0.0, 0.0, 1.0]),    # Back to start
    ]
    
    # Time to spend at each waypoint
    waypoint_durations = [2.0, 4.0, 2.0, 2.0, 2.0]  # seconds
    
    #### Create obstacles ####
    obstacles = [
        {'pos': [1.2, 0.0, 1.0], 'vel': [0.0, 0.0, 0.0]},
        {'pos': [1.7, 0.0, 1.0], 'vel': [0.0, 0.0, 0.0]},
    ]
    
    # Visualize obstacles
    obs_radius = 0.1
    for obs in obstacles:
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=obs_radius,
            rgbaColor=[1, 0, 0, 0.6],
            physicsClientId=PYB_CLIENT
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=obs['pos'],
            physicsClientId=PYB_CLIENT
        )
    
    print(f"Created {len(obstacles)} obstacles at:")
    for i, obs in enumerate(obstacles):
        print(f"  Obstacle {i+1}: {obs['pos']}")
    print()
    
    #### Drone parameters for velocity->RPM conversion ####
    drone_params = {
        'mass': 0.027,  # kg (Crazyflie 2.x)
    }
    
    #### Control loop ####
    CTRL_EVERY_N_STEPS = int(np.floor(env.CTRL_FREQ / control_freq_hz))
    action = np.zeros((NUM_DRONES, 4))
    START = time.time()
    
    # Control timestep
    control_dt = 1.0 / control_freq_hz
    
    current_waypoint_idx = 0
    waypoint_start_time = 0
    
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        
        #### Step environment ####
        obs, reward, terminated, truncated, info = env.step(action)
        
        #### Compute control every CTRL_EVERY_N_STEPS ####
        if i % CTRL_EVERY_N_STEPS == 0:
            t = i / env.CTRL_FREQ
            
            #### Get current state ####
            current_pos = obs[0][0:3]
            current_vel = obs[0][10:13]
            current_rpy = obs[0][7:10]
            
            #### Update waypoint if needed ####
            if t - waypoint_start_time > waypoint_durations[current_waypoint_idx]:
                if current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
                    waypoint_start_time = t
            
            target_pos = waypoints[current_waypoint_idx]
            
            #### STEP 1: User computes desired velocity ####
            u_desired = compute_velocity_command(current_pos, target_pos, max_vel=0.5)
            yaw_rate_desired = 0.0  # Keep yaw constant
            
            #### STEP 2: QP-CBF filters velocity for safety ####
            # Update drone state
            cbf_drone.update_state(
                current_pos.tolist(), 
                current_vel.tolist(), 
                current_rpy.tolist(), 
                1.0 / control_freq_hz
            )
            
            # Prepare reference control [vx, vy, vz, yaw_rate]
            u_ref = np.array([
                [u_desired[0]],
                [u_desired[1]],
                [u_desired[2]],
                [yaw_rate_desired]
            ])
            
            # Setup and solve QP
            qp_controller.set_reference_control(u_ref)
            obstacle_positions = [obs['pos'] for obs in obstacles]
            obstacle_velocities = [obs['vel'] for obs in obstacles]
            qp_controller.setup_QP(cbf_drone, obstacle_positions, obstacle_velocities)
            qp_controller.solve_QP(cbf_drone)
            
            # Get safe velocity
            u_safe = qp_controller.get_optimal_control()
            velocity_safe = u_safe[0:3]
            yaw_rate_safe = u_safe[3]
            
            #### STEP 3: Use velocity controller to convert safe velocity to RPM ####
            # This mimics IMU + flow deck stabilization on real drone
            rpm = vel_controller.compute_control(
                desired_vel=velocity_safe,
                current_vel=current_vel,
                current_rpy=current_rpy,
                dt=control_dt
            )
            action[0] = rpm
            
            #### Debug output every second ####
            if i % int(control_freq_hz) == 0:
                distance_to_obstacles = [
                    np.linalg.norm(current_pos - np.array(obs['pos'])) 
                    for obs in obstacles
                ]
                min_dist = min(distance_to_obstacles)
                
                print(f"\n{'='*80}")
                print(f"[STANDALONE] Time: {t:.2f}s, Waypoint: {current_waypoint_idx + 1}/{len(waypoints)}")
                print(f"[STATE] Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
                print(f"[STATE] Velocity: [{current_vel[0]:.3f}, {current_vel[1]:.3f}, {current_vel[2]:.3f}]")
                print(f"[TARGET] Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                print(f"[USER] Desired velocity: [{u_desired[0]:.3f}, {u_desired[1]:.3f}, {u_desired[2]:.3f}]")
                print(f"[CBF] Safe velocity: [{velocity_safe[0]:.3f}, {velocity_safe[1]:.3f}, {velocity_safe[2]:.3f}]")
                print(f"[SAFETY] Min distance to obstacle: {min_dist:.3f}m")
                
                # Show if CBF modified the command
                vel_change = np.linalg.norm(velocity_safe - u_desired)
                if vel_change > 0.01:
                    print(f"[CBF] ⚠️  ACTIVE - Modified velocity by {vel_change:.3f} m/s")
                else:
                    print(f"[CBF] ✓ No constraints - Using desired velocity")
                print(f"{'='*80}")
        
        #### Log ####
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0]
        )
        
        #### Sync ####
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
    
    #### Close environment ####
    env.close()
    
    #### Save logs ####
    logger.save()
    logger.save_as_csv("standalone_cbf")
    
    #### Plot ####
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone QP-CBF control without firmware')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel, 
                       help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics,
                       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                       help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                       help='Whether to plot results (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                       help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                       help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                       help='Duration of the simulation in seconds (default: 12)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                       help='Folder where to save logs (default: "results")', metavar='')
    
    ARGS = parser.parse_args()
    
    run(**vars(ARGS))
