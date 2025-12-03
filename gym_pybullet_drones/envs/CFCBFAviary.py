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
    """Crazyflie aviary with integrated Control Barrier Function (CBF) control."""

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
        """Initialize CBF-controlled Crazyflie aviary."""
        
        if cbf_params is None:
            cbf_params = {'gamma': 1.0, 'obs_radius': 0.1}
        
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
        
        if verbose:
            print("[CFCBFAviary] Initialized with CBF control")

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
        
        self.qp_controller = QP_Controller_Drone(
            drone=self.drone,
            gamma=self.cbf_params['gamma'],
            obs_radius=self.cbf_params['obs_radius']
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment and sync drone state.
        
        Overrides base reset to ensure Drone object is synced with PyBullet after reset.
        """
        obs, info = super().reset(seed=seed, options=options)
        
        # Sync drone state with actual PyBullet position after reset
        self._sync_drone_state(obs)
        
        return obs, info

    def _sync_drone_state(self, obs, add_noise=True):
        """Synchronize the Drone object state with PyBullet observations.
        
        obs is 20-element array: [x, y, z, q1, q2, q3, q4, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, rpm0, rpm1, rpm2, rpm3]
        
        Parameters:
            obs: 20-element observation array
            add_noise: If True, add small noise to prevent numerical issues in QP solver
        """
        pos = obs[0:3].copy()
        quat = obs[3:7].copy()
        rpy = obs[7:10].copy()  # Already computed by BaseAviary
        vel = obs[10:13].copy()
        ang_vel = obs[13:16].copy()
        
        # Add small noise to prevent NaN issues in QP solver (division by zero, etc.)
        # This is especially important when drone is stationary or near singularities
        if add_noise:
            noise_std = 1e-6  # Very small noise
            pos += np.random.normal(0, noise_std, 3)
            vel += np.random.normal(0, noise_std, 3)
            rpy += np.random.normal(0, noise_std, 3)
            ang_vel += np.random.normal(0, noise_std, 3)
        
        # Update drone state with arrays: p (position), v (velocity), rpy (euler angles), delta_t
        self.drone.update_state(
            p=pos,
            v=vel,
            rpy=rpy,
            delta_t=self.CTRL_TIMESTEP
        )
        
        # CRITICAL: Override angular velocities with actual PyBullet values
        # The Drone.update_state() computes w_1, w_2, w_3 incorrectly from Euler angles
        # We need to use the actual body-frame angular velocities from PyBullet
        self.drone.w_1 = ang_vel[0]  # Body-frame roll rate
        self.drone.w_2 = ang_vel[1]  # Body-frame pitch rate  
        self.drone.w_3 = ang_vel[2]  # Body-frame yaw rate

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
    
    def get_control_diagnostics(self):
        """Get detailed diagnostics for investigating drift and control issues.
        
        Returns dict with:
            - thrust_to_weight_ratio: Total thrust / weight (should be ~1.0 for hover)
            - rpm_balance: Std dev of RPMs (should be ~0 for balanced hover)
            - position_error: Distance from initial position [0,0,1]
            - velocity_magnitude: Speed of drone
            - angular_velocity_magnitude: Rotation rate
            - attitude: Roll, pitch, yaw angles
        """
        kf = 3.16e-10  # Thrust coefficient
        m = self.drone.m
        g = 9.81
        
        # Get last applied RPMs
        last_rpms = self.last_clipped_action[0]
        
        # Calculate thrust per motor
        thrusts = (last_rpms ** 2) * kf
        total_thrust = np.sum(thrusts)
        
        # Calculate metrics
        thrust_to_weight = total_thrust / (m * g)
        rpm_std = np.std(last_rpms)
        rpm_mean = np.mean(last_rpms)
        rpm_imbalance = (np.max(last_rpms) - np.min(last_rpms)) / rpm_mean if rpm_mean > 0 else 0
        
        # Position and velocity
        pos = np.array([self.drone.x, self.drone.y, self.drone.z])
        vel = np.array([self.drone.x_dot, self.drone.y_dot, self.drone.z_dot])
        ang_vel = np.array([self.drone.w_1, self.drone.w_2, self.drone.w_3])
        
        initial_pos = np.array([0.0, 0.0, 1.0])
        pos_error = np.linalg.norm(pos - initial_pos)
        vel_mag = np.linalg.norm(vel)
        ang_vel_mag = np.linalg.norm(ang_vel)
        
        return {
            'thrust_to_weight_ratio': thrust_to_weight,
            'total_thrust_N': total_thrust,
            'weight_N': m * g,
            'rpm_mean': rpm_mean,
            'rpm_std': rpm_std,
            'rpm_imbalance_pct': rpm_imbalance * 100,
            'rpms': last_rpms.copy(),
            'thrusts_N': thrusts.copy(),
            'position': pos.copy(),
            'position_error_m': pos_error,
            'velocity': vel.copy(),
            'velocity_mag_ms': vel_mag,
            'angular_velocity': ang_vel.copy(),
            'angular_velocity_mag_rads': ang_vel_mag,
            'roll_deg': np.rad2deg(self.drone.phi),
            'pitch_deg': np.rad2deg(self.drone.theta),
            'yaw_deg': np.rad2deg(self.drone.psi)
        }

    def apply_cbf_control(self, obstacle_pos, obstacle_vel, u_ref):
        """Apply CBF-QP control to compute safe control input."""
        self.qp_controller.set_reference_control(u_ref)
        self.qp_controller.setup_QP(self.drone, obstacle_pos, obstacle_vel)
        self.qp_controller.solve_QP(self.drone)
        u_opt = self.qp_controller.get_optimal_control()
        return u_opt

    def step_with_cbf(self, obstacle_pos=None, obstacle_vel=None, u_ref=None):
        """Step the simulation with CBF control.
        
        Args:
            obstacle_pos: Obstacle position [x, y, z] in meters, or None if no obstacle
            obstacle_vel: Obstacle velocity [vx, vy, vz] in m/s, or None if no obstacle
            u_ref: Reference control (thrust per motor) [N, N, N, N], or None for hover
        
        Returns:
            tuple: (obs, reward, terminated, truncated, info) where info contains CBF debug data
        """
        # If no obstacle, bypass CBF-QP entirely and use reference trajectory directly
        if obstacle_pos is None and obstacle_vel is None:
            if u_ref is None:
                u_ref = np.full(4, self.GRAVITY / 4.0)  # Hover thrust per motor (GRAVITY is m*g)
            
            # Convert thrust (N) to RPM: RPM = sqrt(thrust / kf)
            u_ref_rpm = np.sqrt(u_ref.flatten() / self.KF)
            
            # Store info for consistency with CBF path
            self._updateAndStoreKinematicInformation()
            obs = self._computeObs()
            self._sync_drone_state(obs)
            
            self.last_cbf_data = {
                'u_ref': u_ref.copy(),
                'u_opt_rpm': u_ref_rpm.copy(),
                'obstacle_pos': None,
                'obstacle_vel': None,
                'drone_pos': obs[0:3].copy(),
                'drone_vel': obs[10:13].copy(),
                'cbf_bypassed': True
            }
            
            return self.step(u_ref_rpm)
            
        # If obstacle exists, use CBF-QP
        if u_ref is None:
            u_ref = np.full(4, self.GRAVITY / 4.0)  # Hover thrust per motor (GRAVITY is m*g)
        
        # CRITICAL: Update kinematic information from PyBullet first
        self._updateAndStoreKinematicInformation()
        
        # Get current observation and sync drone state
        obs = self._computeObs()
        self._sync_drone_state(obs)
        
        # Apply CBF-QP control (returns RPM values directly)
        u_opt_rpm = self.apply_cbf_control(obstacle_pos, obstacle_vel, u_ref)
        
        # u_opt_rpm is already in RPM, no need for conversion
        action = np.array(u_opt_rpm, dtype=float)
        
        # Store CBF control data for logging
        self.last_cbf_data = {
            'u_ref': u_ref.copy(),
            'u_opt_rpm': action.copy(),
            'obstacle_pos': obstacle_pos.copy(),
            'obstacle_vel': obstacle_vel.copy(),
            'drone_pos': obs[0:3].copy(),
            'drone_vel': obs[10:13].copy(),
            'cbf_bypassed': False
        }
        
        return self.step(action)

    def step(self, action):
        """Simplified step function accepting motor RPMs directly."""
        # Preprocess and clip action
        clipped_action = self._preprocessAction(action)
        
        for _ in range(self.PYB_FREQ // self.CTRL_FREQ):
            self._physics(clipped_action, 0)
            p.stepSimulation(physicsClientId=self.CLIENT)
            if self.RECORD:
                self._updateAndLog()
        
        # Save the last clipped action (used by physics like drag)
        self.last_clipped_action = np.reshape(clipped_action, (self.NUM_DRONES, 4))
        
        # CRITICAL: Update kinematic information from PyBullet before computing obs
        self._updateAndStoreKinematicInformation()
        
        obs = self._computeObs()
        self._sync_drone_state(obs)
        
        reward = 0.0
        terminated = False
        truncated = False
        
        # Include CBF debug data in info if available
        info = {'drone_state': self.get_drone_state_dict()}
        if hasattr(self, 'last_cbf_data'):
            info['cbf_control'] = self.last_cbf_data
        
        return obs, reward, terminated, truncated, info

    def _actionSpace(self):
        """Returns the action space of the environment."""
        act_lower_bound = np.array([0.0, 0.0, 0.0, 0.0])
        act_upper_bound = np.array([self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        # Standard 20-element observation: pos(3) + quat(4) + rpy(3) + vel(3) + ang_vel(3) + rpms(4)
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0., -1., -1., -1., -1., 
                                     -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, 
                                     -np.inf, -np.inf, -np.inf, 0., 0., 0., 0.]])
        obs_upper_bound = np.array([[np.inf, np.inf, np.inf, 1., 1., 1., 1., 
                                     np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, 
                                     np.inf, np.inf, np.inf, self.MAX_RPM, self.MAX_RPM, 
                                     self.MAX_RPM, self.MAX_RPM]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment.
        
        Returns 20-element state vector compatible with Logger:
        [x, y, z, q1, q2, q3, q4, roll, pitch, yaw, vx, vy, vz, wx, wy, wz, rpm0, rpm1, rpm2, rpm3]
        """
        return self._getDroneStateVector(0)

    def _preprocessAction(self, action):
        """Pre-processes the action passed to .step() into motors' RPMs."""
        return np.clip(action, 0, self.MAX_RPM)

    def _computeReward(self):
        """Computes the current reward value(s)."""
        return 0.0

    def _computeTerminated(self):
        """Computes the current terminated value(s)."""
        drone_idx = 0
        pos, _ = p.getBasePositionAndOrientation(self.DRONE_IDS[drone_idx],
                                                 physicsClientId=self.CLIENT)
        if pos[2] < 0.0 or pos[2] > 3.0:
            return True
        return False

    def _computeTruncated(self):
        """Computes the current truncated value(s)."""
        if self.step_counter / self.PYB_FREQ > 10.0:
            return True
        return False

    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {'drone_state': self.get_drone_state_dict()}
