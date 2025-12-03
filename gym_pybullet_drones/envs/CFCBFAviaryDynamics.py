"""
CFCBFAviary.py - Crazyflie Aviary with Control Barrier Function (CBF) control
"""

import math
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from gym_pybullet_drones.envs.CBF_CONTROL.drone import Drone
from gym_pybullet_drones.envs.CBF_CONTROL.QP_controller_drone import QP_Controller_Drone


class CFCBFAviary(BaseAviary):
    """Crazyflie aviary with integrated Control Barrier Function (CBF) control using velocity commands."""

    RAD_TO_DEG = 180 / math.pi

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 25,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 verbose=False,
                 drone_config_path='gym_pybullet_drones/envs/CBF_CONTROL/drone1.json',
                 cbf_params=None
                 ):
        """Initialize CBF-controlled Crazyflie aviary with velocity-based control."""
        
        if cbf_params is None:
            cbf_params = {'gamma': 1.5, 'obs_radius': 0.2}
        
        self.drone_config_path = drone_config_path
        self.cbf_params = cbf_params
        
        super().__init__(drone_model=drone_model,
                        num_drones=num_drones,
                        neighbourhood_radius=neighbourhood_radius,
                        initial_xyzs=initial_xyzs,
                        initial_rpys=initial_rpys,
                        physics=physics,
                        pyb_freq=pyb_freq,
                        ctrl_freq=ctrl_freq,
                        gui=gui,
                        record=record,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui,
                        output_folder=output_folder)
        
        self._initialize_cbf_system()
        self._initialize_velocity_controller()
        
        if verbose:
            print("[CFCBFAviary] Initialized with velocity-based CBF control")

    def _initialize_cbf_system(self):
        """Initialize the CBF control system."""
        try:
            self.drone = Drone.from_JSON(self.drone_config_path)
            print(f"[CFCBFAviary] Loaded Drone config from {self.drone_config_path}")
        except FileNotFoundError:
            print(f"[CFCBFAviary] Warning: Config file not found at {self.drone_config_path}")
            print("[CFCBFAviary] Creating default Drone with CF2X parameters")
            self.drone = Drone(
                name="cbf_drone",
                length_offset_COM=0.0,
                dimensions=(0.092, 0.092, 0.028),
                m=0.027,
                Ixx=1.4e-5,
                Iyy=1.4e-5,
                Izz=2.17e-5,
                x=self.INIT_XYZS[0][0],
                y=self.INIT_XYZS[0][1],
                z=self.INIT_XYZS[0][2],
                x_d=0.0,
                y_d=0.0,
                z_d=0.0,
                phi=0.0,
                theta=0.0,
                psi=0.0,
                w_1=0.0,
                w_2=0.0,
                w_3=0.0
            )
        
        # Add radius attribute needed by QP controller (for both JSON and default)
        self.drone.r = self.drone.encompassing_radius
        print(f"[CFCBFAviary] Drone initialized: m={self.drone.m}, r={self.drone.r:.4f}")
        
        # Initialize velocity-based QP controller
        self.qp_controller = QP_Controller_Drone(
            gamma=self.cbf_params['gamma'],
            obs_radius=self.cbf_params['obs_radius']
        )

    def _initialize_velocity_controller(self):
        """Initialize PD controller for converting velocity commands to motor RPMs."""
        # PD gains for velocity tracking
        self.kp_vel = np.array([2.0, 2.0, 4.0])  # [x, y, z] - stronger Z for altitude
        self.kd_vel = np.array([0.5, 0.5, 1.0])  # Damping
        
        # Yaw control gains
        self.kp_yaw = 1.0
        
        # Motor mixing matrix for thrust allocation
        # [total_thrust, roll_torque, pitch_torque, yaw_torque] = mixing_matrix @ [rpm0^2, rpm1^2, rpm2^2, rpm3^2]
        self.kf = 3.16e-10  # Thrust coefficient
        self.km = 7.94e-12  # Torque coefficient
        L = 0.0397  # Arm length (m)
        
        # Track previous velocity for derivative term
        self.prev_vel_error = np.zeros(3)

    def velocity_to_rpm(self, vel_cmd, current_vel, current_rpy, yaw_rate_cmd):
        """Convert velocity command [vx, vy, vz, yaw_rate] to motor RPMs.
        
        Uses PD control to track velocity, then converts desired accelerations to motor commands.
        
        Args:
            vel_cmd: Desired velocity [vx, vy, vz] in m/s (world frame)
            current_vel: Current velocity [vx, vy, vz] in m/s (world frame)
            current_rpy: Current attitude [roll, pitch, yaw] in radians
            yaw_rate_cmd: Desired yaw rate in rad/s
            
        Returns:
            np.array: Motor RPMs [rpm0, rpm1, rpm2, rpm3]
        """
        # Velocity tracking error
        vel_error = vel_cmd - current_vel
        vel_error_dot = (vel_error - self.prev_vel_error) / self.CTRL_TIMESTEP
        self.prev_vel_error = vel_error.copy()
        
        # PD control for desired acceleration (world frame)
        desired_acc = self.kp_vel * vel_error + self.kd_vel * vel_error_dot
        
        # Add gravity compensation in Z
        desired_acc[2] += self.G
        
        # Convert to body frame
        roll, pitch, yaw = current_rpy
        R_world_to_body = Rotation.from_euler('XYZ', [roll, pitch, yaw]).as_matrix().T
        desired_acc_body = R_world_to_body @ desired_acc
        
        # Compute required thrust (assume small angles, thrust mainly in Z-body)
        total_thrust = self.GRAVITY  # Hover thrust as baseline
        
        # Add Z acceleration (thrust direction)
        if abs(desired_acc_body[2]) > 0.01:
            total_thrust = self.drone.m * desired_acc_body[2]
        
        # Clip total thrust to reasonable range
        max_thrust = 4 * self.MAX_RPM**2 * self.kf  # Max thrust from all 4 motors
        total_thrust = np.clip(total_thrust, 0.5 * self.GRAVITY, 0.9 * max_thrust)
        
        # Desired roll/pitch from XY accelerations (small angle approximation)
        # pitch controls X acceleration, roll controls Y acceleration
        target_pitch = np.arctan2(desired_acc_body[0], self.G)
        target_roll = np.arctan2(-desired_acc_body[1], self.G)
        
        # Clip angles to safe range
        max_angle = np.deg2rad(20)  # 20 degree max tilt
        target_pitch = np.clip(target_pitch, -max_angle, max_angle)
        target_roll = np.clip(target_roll, -max_angle, max_angle)
        
        # Attitude errors (simplified - no rate feedback)
        roll_error = target_roll - roll
        pitch_error = target_pitch - pitch
        
        # Control torques
        torque_roll = 10.0 * roll_error  # Proportional control
        torque_pitch = 10.0 * pitch_error
        torque_yaw = self.kp_yaw * yaw_rate_cmd
        
        # Motor mixing: Solve for motor thrusts
        # Motor layout (X configuration):
        #   0: Front-right (CW)
        #   1: Front-left (CCW)
        #   2: Back-left (CW)
        #   3: Back-right (CCW)
        
        # Thrust per motor (hover baseline)
        base_thrust = total_thrust / 4.0
        
        # Add control torques (simplified mixing)
        L = 0.0397  # Arm length
        motor_thrusts = np.array([
            base_thrust + torque_pitch / (4*L) - torque_roll / (4*L) - torque_yaw / 4.0,  # M0: FR
            base_thrust + torque_pitch / (4*L) + torque_roll / (4*L) + torque_yaw / 4.0,  # M1: FL
            base_thrust - torque_pitch / (4*L) + torque_roll / (4*L) - torque_yaw / 4.0,  # M2: BL
            base_thrust - torque_pitch / (4*L) - torque_roll / (4*L) + torque_yaw / 4.0   # M3: BR
        ])
        
        # Convert thrust to RPM: thrust = kf * rpm^2  =>  rpm = sqrt(thrust / kf)
        motor_rpms = np.sqrt(np.clip(motor_thrusts / self.kf, 0, self.MAX_RPM**2))
        
        return motor_rpms

    def reset(self, seed=None, options=None):
        """Reset the environment and sync drone state."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Reset velocity controller state
        self.prev_vel_error = np.zeros(3)
        
        # Sync drone state with actual PyBullet position after reset
        self._sync_drone_state(obs)
        
        return obs, info

    def _sync_drone_state(self, obs, add_noise=True):
        """Synchronize the Drone object state with PyBullet observations.
        
        obs is 20-element array: [x, y, z, q1, q2, q3, q4, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, rpm0, rpm1, rpm2, rpm3]
        """
        pos = obs[0:3].copy()
        quat = obs[3:7].copy()
        rpy = obs[7:10].copy()
        vel = obs[10:13].copy()
        ang_vel = obs[13:16].copy()
        
        # Add small noise to prevent numerical issues in QP solver
        if add_noise:
            noise_std = 1e-6
            pos += np.random.normal(0, noise_std, 3)
            vel += np.random.normal(0, noise_std, 3)
            rpy += np.random.normal(0, noise_std, 3)
            ang_vel += np.random.normal(0, noise_std, 3)
        
        # Update drone state
        self.drone.update_state(
            p=pos,
            v=vel,
            rpy=rpy,
            delta_t=self.CTRL_TIMESTEP
        )
        
        # Override angular velocities with actual PyBullet values
        self.drone.w_1 = ang_vel[0]
        self.drone.w_2 = ang_vel[1]
        self.drone.w_3 = ang_vel[2]

    def get_drone_state_dict(self, drone_idx=0):
        """Get drone state in dictionary format."""
        return {
            'x': self.drone.x,
            'y': self.drone.y,
            'z': self.drone.z,
            'vx': self.drone.x_dot,
            'vy': self.drone.y_dot,
            'vz': self.drone.z_dot,
            'roll': self.drone.phi,
            'pitch': self.drone.theta,
            'yaw': self.drone.psi,
            'wx': self.drone.w_1,
            'wy': self.drone.w_2,
            'wz': self.drone.w_3
        }

    def apply_cbf_control(self, obstacle_pos, obstacle_vel, u_ref_vel):
        """Apply CBF-QP control to compute safe velocity command.
        
        Args:
            obstacle_pos: List of obstacle positions [[x1,y1,z1], [x2,y2,z2], ...] or []
            obstacle_vel: List of obstacle velocities [[vx1,vy1,vz1], ...] or []
            u_ref_vel: Reference velocity command [vx, vy, vz, yaw_rate]
            
        Returns:
            np.array: Safe velocity command [vx, vy, vz, yaw_rate]
        """
        self.qp_controller.set_reference_control(u_ref_vel)
        self.qp_controller.setup_QP(self.drone, obstacle_pos, obstacle_vel)
        self.qp_controller.solve_QP(self.drone)
        u_opt_vel = self.qp_controller.get_optimal_control()
        return u_opt_vel

    def step_with_cbf(self, obstacle_pos=None, obstacle_vel=None, u_ref_vel=None):
        """Step the simulation with CBF control using velocity commands.
        
        Args:
            obstacle_pos: List of obstacle positions [[x1,y1,z1], ...] or None/[] for no obstacles
            obstacle_vel: List of obstacle velocities [[vx1,vy1,vz1], ...] or None/[] 
            u_ref_vel: Reference velocity command [vx, vy, vz, yaw_rate], or None for hover
        
        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """
        # Handle empty or None obstacles
        if obstacle_pos is None or len(obstacle_pos) == 0:
            obstacle_pos = []
            obstacle_vel = []
        
        # Default to hover velocity command
        if u_ref_vel is None:
            u_ref_vel = np.array([0.0, 0.0, 0.0, 0.0])  # [vx, vy, vz, yaw_rate]
        
        # Update kinematic information and sync state
        self._updateAndStoreKinematicInformation()
        obs = self._computeObs()
        self._sync_drone_state(obs)
        
        # Apply CBF-QP control (returns safe velocity command)
        u_opt_vel = self.apply_cbf_control(obstacle_pos, obstacle_vel, u_ref_vel)
        
        # Convert velocity command to motor RPMs
        current_vel = obs[10:13]  # [vx, vy, vz]
        current_rpy = obs[7:10]   # [roll, pitch, yaw]
        
        motor_rpms = self.velocity_to_rpm(
            vel_cmd=u_opt_vel[:3],
            current_vel=current_vel,
            current_rpy=current_rpy,
            yaw_rate_cmd=u_opt_vel[3]
        )
        
        # Store CBF control data for logging
        self.last_cbf_data = {
            'u_ref_vel': u_ref_vel.copy(),
            'u_opt_vel': u_opt_vel.copy(),
            'motor_rpms': motor_rpms.copy(),
            'obstacle_pos': obstacle_pos,
            'obstacle_vel': obstacle_vel,
            'drone_pos': obs[0:3].copy(),
            'drone_vel': obs[10:13].copy(),
            'cbf_bypassed': len(obstacle_pos) == 0
        }
        
        return self.step(motor_rpms)

    def step(self, action):
        """Step function accepting motor RPMs directly."""
        clipped_action = self._preprocessAction(action)
        
        for _ in range(self.PYB_FREQ // self.CTRL_FREQ):
            self._physics(clipped_action, 0)
            p.stepSimulation(physicsClientId=self.CLIENT)
            if self.RECORD:
                self._updateAndLog()
        
        self.last_clipped_action = np.reshape(clipped_action, (self.NUM_DRONES, 4))
        
        self._updateAndStoreKinematicInformation()
        obs = self._computeObs()
        self._sync_drone_state(obs)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        info = {'drone_state': self.get_drone_state_dict()}
        if hasattr(self, 'last_cbf_data'):
            info['cbf_control'] = self.last_cbf_data
        
        return obs, reward, terminated, truncated, info

    def _actionSpace(self):
        """Returns the action space (motor RPMs)."""
        act_lower_bound = np.array([0.0, 0.0, 0.0, 0.0])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space."""
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0., -1., -1., -1., -1., 
                                     -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, 
                                     -np.inf, -np.inf, -np.inf, 0., 0., 0., 0.]])
        obs_upper_bound = np.array([[np.inf, np.inf, np.inf, 1., 1., 1., 1., 
                                     np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, 
                                     np.inf, np.inf, np.inf, self.MAX_RPM, self.MAX_RPM, 
                                     self.MAX_RPM, self.MAX_RPM]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation (20-element state vector)."""
        return self._getDroneStateVector(0)

    def _preprocessAction(self, action):
        """Pre-processes the action (clips RPMs)."""
        return np.clip(action, 0, self.MAX_RPM)

    def _computeReward(self):
        """Computes reward."""
        return 0.0

    def _computeTerminated(self):
        """Computes termination condition."""
        drone_idx = 0
        pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[drone_idx],
                                                 physicsClientId=self.CLIENT)
        if pos[2] < 0.0 or pos[2] > 3.0:
            return True
        return False

    def _computeTruncated(self):
        """Computes truncation condition."""
        if self.step_counter / self.PYB_FREQ > 10.0:
            return True
        return False

    def _computeInfo(self):
        """Computes info dict."""
        return {'drone_state': self.get_drone_state_dict()}