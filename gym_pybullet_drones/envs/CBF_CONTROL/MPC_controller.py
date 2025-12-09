# MPC_controller.py

import numpy as np
from abc import ABC, abstractmethod


class MPC_Controller(ABC):
    """
    Abstract base class for MPC-style controllers.

    This is intentionally similar in spirit to QP_Controller, but oriented
    around receding-horizon optimization. It keeps generic MPC parameters
    (horizon, dt) and defines the standard interface:

        - set_reference_control(...)
        - get_reference_control()
        - set_goal_position(...)
        - setup_QP(...)
        - solve_QP(...)
        - get_optimal_control()

    Subclasses are expected to implement the MPC-specific formulation.
    """

    def __init__(self, horizon: int, dt: float):
        """
        Parameters
        ----------
        horizon : int
            Number of prediction steps N in the MPC horizon.
        dt : float
            Discrete time step used in the prediction model.
        """
        self.horizon = int(horizon)
        self.dt = float(dt)
        self._u_ref = None  # generic storage; semantics depend on subclass

    @abstractmethod
    def set_reference_control(self, u_ref: np.ndarray):
        """
        Set a nominal/reference control. The meaning of u_ref is
        subclass-specific (e.g., nominal velocity or a tracking controller
        output), but the interface matches QP_Controller.

        Parameters
        ----------
        u_ref : np.ndarray
            Typically a 4D vector [vx, vy, vz, yaw_rate] for the drone case.
        """
        raise NotImplementedError

    @abstractmethod
    def get_reference_control(self) -> np.ndarray:
        """
        Return the stored reference control.

        Returns
        -------
        np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def set_goal_position(self, p_goal: np.ndarray):
        """
        Set the position goal used in the MPC stage cost.

        Parameters
        ----------
        p_goal : np.ndarray
            3D position [x, y, z] in world frame.
        """
        raise NotImplementedError

    @abstractmethod
    def setup_QP(self, bot, c, c_d):
        """
        Prepare any data needed for the next MPC QP solve (e.g., current
        state, obstacle geometry, precomputed matrices).

        Parameters
        ----------
        bot : object
            System model / state holder (Drone object here).
        c : list or np.ndarray
            Obstacle information; semantics are subclass-specific.
        c_d : list or np.ndarray
            Obstacle velocity information; semantics are subclass-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def solve_QP(self, bot):
        """
        Solve the MPC QP and store the optimal first-step control.

        Parameters
        ----------
        bot : object
            System model / state holder (Drone object here).

        Returns
        -------
        Any
            Can return diagnostic information if desired.
        """
        raise NotImplementedError

    @abstractmethod
    def get_optimal_control(self) -> np.ndarray:
        """
        Return the control to apply at the current time step (typically the
        first element of the MPC control sequence).

        Returns
        -------
        np.ndarray
            Control vector to send to the low-level system. For the drone
            case: [vx, vy, vz, yaw_rate].
        """
        raise NotImplementedError
