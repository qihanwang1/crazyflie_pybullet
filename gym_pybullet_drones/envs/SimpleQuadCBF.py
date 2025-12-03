"""
Simple Quadrotor with Direct CBF Control
Matches the paper's point-mass model for proper CBF-QP integration
"""

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class SimpleQuadCBF(BaseAviary):
    """
    Simple quadrotor environment with direct acceleration control.
    
    This bypasses Mellinger and applies CBF-corrected accelerations 
    directly as forces in PyBullet, matching the paper's approach.
    """
    
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,  # Same as physics for direct control
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=False,
                 vision_attributes=False
                 ):
        """Initialize with direct force control (no Mellinger)."""
        
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
            vision_attributes=vision_attributes
        )
        
        # CBF obstacle tracking
        self.cbf_obstacle_active = False
        self.obstacle_pos = None
        self.obstacle_vel = None
        
        print(f"[SimpleQuadCBF] Initialized with direct acceleration control at {ctrl_freq}Hz")
    
    def set_obstacle(self, pos, vel):
        """Set obstacle for CBF avoidance."""
        self.obstacle_pos = np.array(pos)
        self.obstacle_vel = np.array(vel)
        self.cbf_obstacle_active = True
        print(f"[SimpleQuadCBF] Obstacle set at {pos} with velocity {vel}")
    
    def step_direct_accel(self, acceleration_command):
        """
        Step with direct acceleration command (paper's approach).
        
        Args:
            acceleration_command: [ax, ay, az] in m/s^2 (world frame)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Convert acceleration to force: F = m * a
        mass = self.M  # kg
        gravity_compensation = np.array([0, 0, 9.81])
        
        # Total acceleration including gravity compensation
        total_accel = acceleration_command + gravity_compensation
        force = mass * total_accel
        
        # Apply force directly to drone COM
        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[0],
            linkIndex=-1,  # Base/COM
            forceObj=force.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT
        )
        
        # Step physics
        p.stepSimulation(physicsClientId=self.CLIENT)
        
        # Compute observation
        obs = self._computeObs()
        reward = 0  # Implement reward if needed
        terminated = False
        truncated = False
        info = {}
        
        # Advance step counter
        self.step_counter += 1
        
        return obs, reward, terminated, truncated, info
    
    def _actionSpace(self):
        """Acceleration command space: [ax, ay, az]."""
        # ±10 m/s^2 acceleration limits
        import gymnasium as gym
        return gym.spaces.Box(
            low=np.array([-10, -10, -10]),
            high=np.array([10, 10, 10]),
            dtype=np.float32
        )
    
    def _observationSpace(self):
        """Same as BaseAviary."""
        return super()._observationSpace()
    
    def _computeObs(self):
        """Same as BaseAviary."""
        return super()._computeObs()
    
    def _preprocessAction(self, action):
        """No preprocessing needed for direct acceleration."""
        return action
    
    def _computeReward(self):
        """Implement task-specific reward."""
        return 0
    
    def _computeTerminated(self):
        """Terminate if drone crashes."""
        obs = self._computeObs()
        pos = obs[0][0:3]
        
        # Terminate if too low or too high
        if pos[2] < 0.05 or pos[2] > 3.0:
            return True
        return False
    
    def _computeTruncated(self):
        """Truncate if time limit reached."""
        return False
    
    def _computeInfo(self):
        """Return empty info dict."""
        return {}
