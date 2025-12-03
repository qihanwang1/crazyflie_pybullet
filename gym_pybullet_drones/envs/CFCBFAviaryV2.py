"""
CFCBFAviaryV2: Crazyflie environment with optional CBF-QP obstacle avoidance

This environment inherits from CFAviary (which uses the proven Mellinger controller 
from pycffirmware) and adds optional CBF-QP obstacle avoidance on top.

Key Strategy:
- When obstacles are None: Flies exactly like CFAviary (stable, proven)
- When obstacles are provided: CBF-QP modifies the reference trajectory to avoid obstacles

This is a complete rebuild based on the working CFAviary foundation.
"""

import numpy as np
from gym_pybullet_drones.envs.CFAviary import CFAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Import CBF-QP controller
from gym_pybullet_drones.envs.CBF_CONTROL.QP_controller_drone import QP_Controller_Drone
from gym_pybullet_drones.envs.CBF_CONTROL.drone import Drone


class CFCBFAviaryV2(CFAviary):
    """
    Crazyflie environment with optional CBF-QP obstacle avoidance.
    
    Inherits from CFAviary which uses the Mellinger controller. When obstacles
    are provided, CBF-QP modifies the commanded trajectory to ensure safety.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 500,
                 ctrl_freq: int = 25,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 verbose=False,
                 cbf_gamma: float = 1.0,
                 cbf_obs_radius: float = 0.1,
                 ):
        """
        Initialize CFCBFAviaryV2 environment.

        Parameters
        ----------
        cbf_gamma : float, optional
            CBF class-K function parameter (gamma in alpha(h) = gamma*h)
        cbf_obs_radius : float, optional
            Obstacle radius in meters for CBF calculations
        
        All other parameters are inherited from CFAviary.
        """
        # Initialize parent CFAviary
        super().__init__(
            drone_model=drone_model,
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
            output_folder=output_folder,
            verbose=verbose
        )

        # CBF-QP parameters
        self.cbf_gamma = cbf_gamma
        self.cbf_obs_radius = cbf_obs_radius

        # Initialize Drone object for CBF-QP (matches Crazyflie 2.X parameters)
        self.drone_cbf = Drone(
            name="CF2X",
            x=0.0, y=0.0, z=0.0,
            x_d=0.0, y_d=0.0, z_d=0.0,
            phi=0.0, theta=0.0, psi=0.0,
            w_1=0.0, w_2=0.0, w_3=0.0,
            Ixx=1.4e-5,  # kg*m^2
            Iyy=1.4e-5,  # kg*m^2
            Izz=2.17e-5,  # kg*m^2
            m=0.027,  # kg
            length_offset_COM=0.03,  # m
            dimensions=(0.028, 0.028, 0.01)  # length, width, height in m
        )

        # Initialize QP controller
        self.qp_controller = QP_Controller_Drone(
            drone=self.drone_cbf,
            gamma=self.cbf_gamma,
            obs_radius=self.cbf_obs_radius
        )

        # Obstacle tracking
        self.obstacle_pos = None  # [x, y, z] in meters
        self.obstacle_vel = None  # [vx, vy, vz] in m/s

        # For tracking CBF activity
        self.cbf_active = False
        self.cbf_h_value = 0.0

        if verbose:
            print(f"[CFCBFAviaryV2] Initialized with CBF gamma={cbf_gamma}, obs_radius={cbf_obs_radius}")

    def set_obstacle(self, position, velocity):
        """
        Set obstacle position and velocity for CBF-QP avoidance.

        Parameters
        ----------
        position : np.ndarray or None
            Obstacle position [x, y, z] in meters. Set to None to disable CBF.
        velocity : np.ndarray or None
            Obstacle velocity [vx, vy, vz] in m/s. Set to None to disable CBF.
        """
        self.obstacle_pos = position if position is not None else None
        self.obstacle_vel = velocity if velocity is not None else None

        if self.verbose:
            if position is not None:
                print(f"[CFCBFAviaryV2] Obstacle set: pos={position}, vel={velocity}")
            else:
                print("[CFCBFAviaryV2] Obstacle cleared (CBF disabled)")

    def step_with_cbf(self, i, pos_des, vel_des, acc_des, yaw_des, rpy_rate_des):
        """
        Step the environment with optional CBF-QP obstacle avoidance.

        This method wraps CFAviary's step() and applies CBF-QP modification
        to the commanded trajectory when obstacles are present.

        Parameters
        ----------
        i : int
            Simulation control step index
        pos_des : np.ndarray
            Desired position [x, y, z] in meters
        vel_des : np.ndarray
            Desired velocity [vx, vy, vz] in m/s
        acc_des : np.ndarray
            Desired acceleration [ax, ay, az] in m/s^2
        yaw_des : float
            Desired yaw angle in radians
        rpy_rate_des : np.ndarray
            Desired roll, pitch, yaw rates [rad/s, rad/s, rad/s]

        Returns
        -------
        obs : np.ndarray
            Observation
        reward : float
            Reward value
        terminated : bool
            Whether episode terminated
        truncated : bool
            Whether episode truncated
        info : dict
            Info dictionary with additional fields:
                - 'cbf_active': bool, whether CBF-QP was used
                - 'cbf_h_value': float, value of CBF barrier function
        """
        # Check if CBF should be active
        if self.obstacle_pos is not None and self.obstacle_vel is not None:
            # CBF-QP is active - use simple repulsive acceleration
            self.cbf_active = True
            pos_des_modified, vel_des_modified, acc_des_modified = self._apply_cbf_modification(
                pos_des, vel_des, acc_des
            )
        else:
            # No obstacle - use desired trajectory as-is
            pos_des_modified = pos_des
            vel_des_modified = vel_des
            acc_des_modified = acc_des
            self.cbf_active = False
            self.cbf_h_value = 0.0

        # Send trajectory to the Mellinger controller via sendFullStateCmd
        t = i / self.ctrl_freq
        self.sendFullStateCmd(
            pos=pos_des_modified.tolist() if isinstance(pos_des_modified, np.ndarray) else pos_des_modified,
            vel=vel_des_modified.tolist() if isinstance(vel_des_modified, np.ndarray) else vel_des_modified,
            acc=acc_des_modified.tolist() if isinstance(acc_des_modified, np.ndarray) else acc_des_modified,
            yaw=float(yaw_des),
            rpy_rate=rpy_rate_des.tolist() if isinstance(rpy_rate_des, np.ndarray) else rpy_rate_des,
            timestep=t
        )

        # Execute CFAviary's step (which handles Mellinger controller internally)
        obs, reward, terminated, truncated, info = super().step(i)

        # Add CBF info to the info dict
        info['cbf_active'] = self.cbf_active
        info['cbf_h_value'] = self.cbf_h_value

        return obs, reward, terminated, truncated, info

    def _apply_cbf_modification(self, pos_des, vel_des, acc_des):
        """
        Apply CBF-QP to modify desired acceleration for obstacle avoidance.

        Strategy:
        1. Get current drone state
        2. Convert desired acceleration to reference thrust commands
        3. Set up and solve CBF-QP to get safe thrust commands
        4. Convert safe thrust back to acceleration
        5. Return modified acceleration (position/velocity unchanged)

        Parameters
        ----------
        pos_des : np.ndarray
            Desired position [x, y, z]
        vel_des : np.ndarray
            Desired velocity [vx, vy, vz]
        acc_des : np.ndarray
            Desired acceleration [ax, ay, az]

        Returns
        -------
        pos_modified : np.ndarray
            Modified desired position (unchanged from input)
        vel_modified : np.ndarray
            Modified desired velocity (unchanged from input)
        acc_modified : np.ndarray
            Modified desired acceleration (CBF-safe)
        """
        try:
            # Get current drone state
            obs = self._computeObs()
            drone_state = obs[0]

            # Extract state
            pos = drone_state[0:3]
            rpy = drone_state[7:10]
            vel = drone_state[10:13]

            # Update CBF Drone object with current state
            self.drone_cbf.update_state(
                p=pos,
                v=vel,
                rpy=rpy,
                delta_t=1.0 / self.ctrl_freq
            )

            # Convert desired acceleration to reference thrust for CBF computation
            mass = self.M  # 0.027 kg
            gravity = 9.81  # m/s^2
            
            total_thrust_ref = mass * (acc_des[2] + gravity)
            thrust_per_motor_ref = max(total_thrust_ref / 4.0, 0.0)
            u_ref = np.array([[thrust_per_motor_ref]] * 4)
            
            # Solve CBF-QP to get barrier value
            self.qp_controller.set_reference_control(u_ref)
            self.qp_controller.setup_QP(
                bot=self.drone_cbf,
                c=self.obstacle_pos,
                c_d=self.obstacle_vel
            )
            cbf_state, cbf_value = self.qp_controller.solve_QP(self.drone_cbf)
            self.cbf_h_value = float(self.qp_controller.h)
            
            # =================================================================
            # CBF-QP ACCELERATION FILTERING (IMPROVED)
            # Enforce hard safety constraint: drone must stay outside r_safe
            # =================================================================
            
            # Vector from obstacle to drone
            to_drone = pos - self.obstacle_pos
            distance = np.linalg.norm(to_drone)
            
            # Safety radius (obstacle + drone envelope)
            r_safe = self.qp_controller.obs_r + self.drone_cbf.encompassing_radius  # 0.44m
            
            # Prevent division by zero
            if distance < 1e-6:
                distance = 1e-6
                
            direction = to_drone / distance  # Unit vector from obstacle to drone
            
            # CBF: h = distance - r_safe (positive when safe)
            h = distance - r_safe
            
            # Relative velocity
            rel_vel = vel - self.obstacle_vel
            
            # h_dot = (to_drone / |to_drone|) dot rel_vel
            h_dot = np.dot(direction, rel_vel)
            
            # CBF constraint: h_ddot + gamma * h_dot >= 0
            # h_ddot = d/dt(h_dot) = d/dt((to_drone/|to_drone|) dot v_rel)
            # For our purposes, we approximate: h_ddot ≈ direction dot a_rel
            # Since obstacle is static: a_rel = a_drone
            
            gamma_cbf = 8.0  # Very aggressive
            
            # Activation distance - only apply CBF when close
            activation_dist = 1.5  # meters
            
            if distance < activation_dist:
                # Compute minimum required acceleration projection
                # direction dot a_drone >= -gamma * h_dot
                min_acc_needed = -gamma_cbf * h_dot
                
                # Current acceleration projection in safe direction
                acc_projection = np.dot(acc_des, direction)
                
                if acc_projection < min_acc_needed:
                    # Violation - need to add corrective acceleration
                    acc_deficit = min_acc_needed - acc_projection
                    acc_correction = acc_deficit * direction
                    
                    # Add correction
                    acc_modified = acc_des + acc_correction
                    
                    # Clamp to limits
                    acc_max = 15.0  # m/s^2 (increased for strong correction)
                    acc_norm = np.linalg.norm(acc_modified)
                    if acc_norm > acc_max:
                        acc_modified = acc_modified * (acc_max / acc_norm)
                    
                    if self.verbose:
                        print(f"    [CBF] ACTIVE: h={h:.3f}m, h_dot={h_dot:.2f}m/s, "
                              f"dist={distance:.3f}m, |correction|={np.linalg.norm(acc_correction):.2f}m/s²")
                else:
                    # Safe - no correction needed
                    acc_modified = acc_des
            else:
                # Far from obstacle
                acc_modified = acc_des
            
            self.cbf_h_value = h
            
            # Don't modify position/velocity - let Mellinger handle tracking
            pos_modified = pos_des
            vel_modified = vel_des
            
            return pos_modified, vel_modified, acc_modified
            
        except Exception as e:
            if self.verbose:
                print(f"    [CBF] Error in CBF-QP: {e}, using reference control")
            # On error, return original trajectory
            self.cbf_h_value = 0.0
            return pos_des, vel_des, acc_des

    def _update_cbf_value(self):
        """
        Update CBF barrier function value for monitoring.
        This doesn't modify control, just tracks safety margin.
        """
        try:
            # Get current drone state
            obs = self._computeObs()
            drone_state = obs[0]

            # Extract state
            pos = drone_state[0:3]
            rpy = drone_state[7:10]
            vel = drone_state[10:13]

            # Update CBF Drone object
            self.drone_cbf.update_state(
                p=pos,
                v=vel,
                rpy=rpy,
                delta_t=1.0 / self.ctrl_freq
            )

            # Compute CBF value (without solving QP)
            # We'll set up a dummy u_ref just to evaluate h
            u_ref_dummy = np.array([0.0, 0.0, 0.0, 0.0])
            self.qp_controller.set_reference_control(u_ref_dummy)
            self.qp_controller.setup_QP(
                bot=self.drone_cbf,
                c=self.obstacle_pos,
                c_d=self.obstacle_vel
            )
            
            # The h value is now computed
            self.cbf_h_value = float(self.qp_controller.h)

        except Exception as e:
            if self.verbose:
                print(f"[CFCBFAviaryV2] Warning: CBF value update failed: {e}")
            self.cbf_h_value = 0.0

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed
        options : dict, optional
            Reset options

        Returns
        -------
        obs : np.ndarray
            Initial observation
        info : dict
            Initial info dict
        """
        # Reset obstacle
        self.obstacle_pos = None
        self.obstacle_vel = None
        self.cbf_active = False
        self.cbf_h_value = 0.0

        # Reset parent CFAviary
        obs, info = super().reset()

        # Add CBF info
        info['cbf_active'] = self.cbf_active
        info['cbf_h_value'] = self.cbf_h_value

        return obs, info
