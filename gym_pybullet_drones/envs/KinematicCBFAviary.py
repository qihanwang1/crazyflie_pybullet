"""
Kinematic quadrotor environment with direct acceleration control and CBF-QP.

This environment implements the control architecture from the PolyC2BF paper:
- Direct acceleration commands (no Mellinger layer)
- High-frequency CBF-QP filtering
- Simple point-mass dynamics matching paper's Equation 18

Author: GitHub Copilot
Date: November 2025
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class KinematicCBFAviary(BaseAviary):
    """
    Kinematic quadrotor environment with direct acceleration control.
    
    Bypasses Mellinger controller entirely - applies forces directly to drone
    based on acceleration commands. Suitable for CBF-QP control matching the
    PolyC2BF paper's architecture.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,  # High frequency for CBF
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 ):
        """
        Initialization of kinematic CBF aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obstacles : bool, optional
            Whether to add obstacles to the environment.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        """
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         )
        
        # Store drone mass for force calculations
        self.MASS = self.M
        
        # Add MASSIVE angular damping to drone physics to suppress wild rotations
        # This acts like the drone is moving through thick air that resists rotation
        p.changeDynamics(
            self.DRONE_IDS[0],
            -1,  # Base link
            angularDamping=50.0,  # Very high angular damping (default is ~0.04)
            physicsClientId=self.CLIENT
        )

    def _actionSpace(self):
        """
        Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            3D acceleration command space: [ax, ay, az] in m/s^2
            Limited to ±20 m/s^2 (about 2g, reasonable for quadrotors)
        """
        return spaces.Box(low=np.array([-20.0, -20.0, -20.0]),
                         high=np.array([20.0, 20.0, 20.0]),
                         dtype=np.float32)

    def _observationSpace(self):
        """
        Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., a 12D vector:
            [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        """
        return spaces.Box(low=np.array([-np.inf] * 12),
                         high=np.array([np.inf] * 12),
                         dtype=np.float32)

    def _computeObs(self):
        """
        Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A 12-dimensional array containing the drone's state:
            [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        """
        state = self._getDroneStateVector(0)
        
        # Extract: position (0:3), velocity (10:13), orientation (7:10), angular velocity (13:16)
        obs = np.hstack([
            state[0:3],   # position
            state[10:13], # velocity
            state[7:10],  # orientation (roll, pitch, yaw)
            state[13:16]  # angular velocity
        ]).astype(np.float32)
        
        return obs

    def _preprocessAction(self, action):
        """
        Pre-processes the action passed to `.step()` into motors' RPMs.

        Since we're doing direct force control, we convert acceleration to force
        and apply it directly. This returns dummy RPM values to satisfy BaseAviary.

        Parameters
        ----------
        action : ndarray
            The 3D acceleration command [ax, ay, az] in m/s^2.

        Returns
        -------
        ndarray
            Dummy RPM values (4,) array - not actually used for control.
        """
        # Apply direct force based on acceleration command
        self._applyDirectAcceleration(action)
        
        # Return dummy RPMs to satisfy BaseAviary interface
        # (actual control is via direct force application)
        return np.array([0.0, 0.0, 0.0, 0.0])

    def _applyDirectAcceleration(self, acceleration):
        """
        Apply acceleration command directly as external force with attitude stabilization.

        This is the core of the kinematic control - bypassing Mellinger entirely.
        Implements F = m * (a_desired + g) where g compensates for gravity.
        Also applies torques to stabilize attitude (keep drone level).

        Parameters
        ----------
        acceleration : ndarray
            Desired 3D acceleration [ax, ay, az] in m/s^2.
        """
        # Ensure acceleration is numpy array
        accel = np.array(acceleration, dtype=np.float64)
        
        # Add gravity compensation: F = m * (a + g)
        # Gravity in PyBullet is -9.81 in z, so we add [0, 0, 9.81]
        gravity_compensation = np.array([0.0, 0.0, 9.81])
        total_accel = accel + gravity_compensation
        
        # Convert acceleration to force: F = m * a
        force = self.MASS * total_accel
        
        # Apply force at center of mass
        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,  # -1 means the base/center of mass
            forceObj=force.tolist(),
            posObj=[0, 0, 0],  # Applied at COM
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT
        )
        
        # ATTITUDE STABILIZATION: Apply corrective torques to keep drone level
        state = self._getDroneStateVector(0)
        orientation = state[7:10]  # [roll, pitch, yaw]
        angular_vel = state[13:16]  # [wx, wy, wz]
        
        # STRATEGY: Massive damping + direct angular velocity clamping
        # The problem is that external forces create huge uncontrollable torques
        # Solution: Use aggressive velocity-based damping (like moving through molasses)
        
        # First, directly clamp angular velocities to prevent runaway
        MAX_ANG_VEL = 1.0  # rad/s - maximum allowed angular velocity
        clamped_ang_vel = np.clip(angular_vel, -MAX_ANG_VEL, MAX_ANG_VEL)
        
        # If velocities were clamped, apply strong counter-torque
        vel_error = angular_vel - clamped_ang_vel
        damping_torque = -100.0 * vel_error  # Very strong damping on excess velocity
        
        # PD control for attitude with MUCH stronger damping
        Kp_attitude = 2.0   # Moderate position control
        Kd_attitude = 20.0  # VERY strong velocity damping (increased 20x)
        
        # Compute corrective torques
        torque_roll = -Kp_attitude * orientation[0] - Kd_attitude * angular_vel[0] + damping_torque[0]
        torque_pitch = -Kp_attitude * orientation[1] - Kd_attitude * angular_vel[1] + damping_torque[1]
        torque_yaw = -Kp_attitude * 0.1 * orientation[2] - Kd_attitude * 0.1 * angular_vel[2] + damping_torque[2] * 0.1

        torque = np.array([torque_roll, torque_pitch, torque_yaw])
        
        # Limit torques to prevent violent corrections
        MAX_TORQUE = 0.0001  # N·m - moderate limit to prevent jerky motion
        torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
        
        # Apply torque in body frame (LINK_FRAME) for proper attitude control
        p.applyExternalTorque(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,
            torqueObj=torque.tolist(),
            flags=p.LINK_FRAME,  # Changed from WORLD_FRAME to LINK_FRAME
            physicsClientId=self.CLIENT
        )

    def _computeReward(self):
        """
        Computes the current reward value(s).

        Returns
        -------
        float
            Dummy reward (not used in this environment).
        """
        return 0.0

    def _computeTerminated(self):
        """
        Computes the current terminated value(s).

        Returns
        -------
        bool
            Always False (termination handled externally).
        """
        return False

    def _computeTruncated(self):
        """
        Computes the current truncated value(s).

        Returns
        -------
        bool
            Always False (truncation handled externally).
        """
        return False

    def _computeInfo(self):
        """
        Computes the current info dict(s).

        Returns
        -------
        dict
            Empty dict (no additional info needed).
        """
        return {}
