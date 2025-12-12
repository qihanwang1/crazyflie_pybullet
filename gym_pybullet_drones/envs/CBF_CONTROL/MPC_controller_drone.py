import numpy as np
from cvxopt import matrix, solvers

from .MPC_controller import MPC_Controller
from .drone import Drone  

solvers.options['show_progress'] = False


class MPC_Controller_Drone(MPC_Controller):
    """
    MPC controller for a single drone with CBF-style first-step obstacle avoidance.

    - Decision variables: translational velocities v_k in R^3 for k = 0,...,N-1
    - Dynamics: p_{k+1} = p_k + dt * v_k  (discrete-time integrator)
    - Cost:
        * Quadratic penalty on velocities (smooth control)
        * Quadratic penalty on position tracking w.r.t. p_goal over the horizon
    - Constraints:
        * Box constraints on v_k: |v_k,i| <= max_vel
        * For each spherical obstacle j, enforce a CBF condition on v_0:
              2 (p0 - c_j)^T v_0 + alpha * h_j(p0) >= 0,
          where h_j(p) = ||p - c_j||^2 - R_total^2.
    """

    def __init__(self,
                 horizon: int = 10,
                 dt: float = 0.1,
                 max_vel: float = 5.0,
                 max_yaw_rate: float = None,
                 safety_margin: float = 0.2,
                 Q_p_diag=(5, 5, 0.2),
                 R_v_diag=(0.5, 0.5, 0.5),
                 obstacle_radius: float = 0.1,
                 cbf_alpha: float = 1.0,
                 **unused_kwargs):
        super().__init__(horizon=horizon, dt=dt)

        # Basic parameters
        self.max_vel = float(max_vel)
        self.max_yaw_rate = float(max_yaw_rate) if max_yaw_rate is not None else None
        self.safety_margin = float(safety_margin)

        # Cost weights
        self.Q_p = np.diag(Q_p_diag).astype(float)
        self.R_v = np.diag(R_v_diag).astype(float)

        # Obstacle geometry (all spherical, same radius here)
        self.obstacle_radius = float(obstacle_radius)

        # CBF gain
        self.cbf_alpha = float(cbf_alpha)

        # Reference control (mainly to pass yaw rate through)
        self.u_ref = np.zeros(4)

        # Goal position for the MPC tracking cost
        self.p_goal = None

        # Last computed optimal control [vx, vy, vz, yaw_rate]
        self.u_star = np.zeros(4)

        # State at the current MPC solve
        self.p0 = None  # current drone position

        # Obstacle centers for current solve
        self.obstacle_centers = []

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def set_reference_control(self, u_ref: np.ndarray):
        """
        Set the nominal control [vx, vy, vz, yaw_rate].

        For this MPC we ignore the nominal translational velocity in the cost
        and only pass yaw_rate through to the final control output, but we keep
        the full vector for possible future extensions.
        """
        u_ref = np.asarray(u_ref, dtype=float).flatten()
        if u_ref.size != 4:
            raise ValueError("u_ref must have length 4: [vx, vy, vz, yaw_rate]")
        self.u_ref = u_ref.copy()

    def get_reference_control(self) -> np.ndarray:
        return self.u_ref.copy()

    def set_goal_position(self, p_goal: np.ndarray):
        """
        Set the goal position for the tracking cost.

        If this is never called, the controller defaults to "stay where you are"
        by setting p_goal = current position in setup_QP().
        """
        p_goal = np.asarray(p_goal, dtype=float).flatten()
        if p_goal.size != 3:
            raise ValueError("p_goal must have length 3: [x, y, z]")
        self.p_goal = p_goal.copy()

    # -------------------------------------------------------------------------
    # Obstacle setup
    # -------------------------------------------------------------------------

    def _set_obstacle_centers(self, centers):
        """
        Normalize obstacle centers for current solve.

        `centers` can be:
            - a single 3D point [x, y, z],
            - or a list of such points.
        """
        self.obstacle_centers = []

        if centers is None:
            return

        # Single obstacle as [x, y, z]
        if isinstance(centers, (list, tuple, np.ndarray)) and np.array(centers).ndim == 1:
            c = np.asarray(centers, dtype=float).flatten()
            if c.size == 3:
                self.obstacle_centers.append(c)
            return

        # List of centers
        for ci in centers:
            ci = np.asarray(ci, dtype=float).flatten()
            if ci.size == 3:
                self.obstacle_centers.append(ci)

    # -------------------------------------------------------------------------
    # MPC setup and solve
    # -------------------------------------------------------------------------

    def setup_QP(self, bot: Drone, c, c_d):
        """
        Prepare data for the next MPC QP solve.

        Parameters
        ----------
        bot : Drone
            Holds the current state of the drone.
        c : list or np.ndarray
            Obstacle centers. Each element is [x, y, z].
        c_d : list or np.ndarray
            Obstacle velocities (not used here; obstacles are treated as static).
        """
        # Current position
        self.p0 = np.array([bot.x, bot.y, bot.z], dtype=float)

        # If no goal has been set, default to "stay here"
        if self.p_goal is None:
            self.p_goal = self.p0.copy()

        # Store obstacle centers for CBF in solve_QP
        self._set_obstacle_centers(c)

    def solve_QP(self, bot: Drone):
        """
        Build and solve the MPC QP.

        Decision vector:
            v = [v_0, v_1, ..., v_{N-1}] ∈ R^{3N},

        where v_k are translational velocities in world frame.
        """
        N = self.horizon
        dt = self.dt
        max_vel = self.max_vel
        Q_p = self.Q_p
        R_v = self.R_v

        p0 = self.p0
        p_goal = self.p_goal

        if p0 is None or p_goal is None:
            raise RuntimeError("MPC_Controller_Drone: p0 or p_goal not set before solve_QP().")

        # Number of decision variables
        n_v = 3 * N

        # Quadratic cost: 0.5 v^T P v + q^T v
        H = np.zeros((n_v, n_v), dtype=float)
        f = np.zeros(n_v, dtype=float)

        # 1) Quadratic cost on velocities: sum v_k^T R_v v_k
        for k in range(N):
            idx = 3 * k
            H[idx:idx+3, idx:idx+3] += R_v

        # 2) Position tracking cost:
        #    p_k = p0 + dt * sum_{j=0}^{k-1} v_j
        #    cost += (p_k - p_goal)^T Q_p (p_k - p_goal),  k = 1,...,N
        for k in range(1, N + 1):
            A_k = np.zeros((3, n_v), dtype=float)
            # p_k depends on v_0,...,v_{k-1}
            for j in range(k):
                j_idx = 3 * j
                A_k[:, j_idx:j_idx+3] += np.eye(3) * dt

            diff_pg = (p0 - p_goal)
            H += A_k.T @ Q_p @ A_k
            f += A_k.T @ Q_p @ diff_pg

        # Convert to cvxopt form
        P = 2.0 * H
        q = 2.0 * f

        P_cvx = matrix(P, tc='d')
        q_cvx = matrix(q, tc='d')

        # Inequality constraints G v <= h
        G_list = []
        h_list = []

        # 1) Velocity box constraints: |v_k,i| <= max_vel
        for k in range(N):
            idx = 3 * k

            # v_k <= max_vel
            for i in range(3):
                row = np.zeros(n_v, dtype=float)
                row[idx + i] = 1.0
                G_list.append(row)
                h_list.append(max_vel)

            # -v_k <= max_vel  -> v_k >= -max_vel
            for i in range(3):
                row = np.zeros(n_v, dtype=float)
                row[idx + i] = -1.0
                G_list.append(row)
                h_list.append(max_vel)

        # 2) CBF constraints on the first control v_0
        #
        # For each spherical obstacle j with center c_j and effective safety
        # radius R_total, we define
        #
        #   h_j(p) = ||p - c_j||^2 - R_total^2,
        #
        # and enforce the discrete-time CBF condition at the current state p0:
        #
        #   dh_j/dt + alpha * h_j(p0) >= 0,
        #
        # where dh_j/dt = 2 (p0 - c_j)^T v_0 for dynamics \dot p = v.
        #
        # This yields a linear inequality in v_0:
        #
        #   2 (p0 - c_j)^T v_0 + alpha * h_j(p0) >= 0.
        #
        # In G v <= h form:
        #
        #   -2 (p0 - c_j)^T v_0 <= alpha * h_j(p0).
        #
        if self.obstacle_centers:
            R_drone = float(bot.encompassing_radius)
            R_total = self.obstacle_radius + R_drone + self.safety_margin

            for c_j in self.obstacle_centers:
                c_j = np.asarray(c_j, dtype=float).flatten()
                r0 = p0 - c_j
                dist_sq = float(r0.dot(r0))
                h0_j = dist_sq - R_total**2

                if h0_j < 0.0:
                    h0_j = 0.0

                g_cbf = 2.0 * r0  # ∇_p h_j(p0)

                row = np.zeros(n_v, dtype=float)
                row[0:3] = -g_cbf  # -2 (p0 - c_j)^T v_0

                G_list.append(row)
                h_list.append(self.cbf_alpha * h0_j)

        if len(G_list) > 0:
            G = matrix(np.vstack(G_list), tc='d')
            h = matrix(np.array(h_list, dtype=float), tc='d')
        else:
            G = None
            h = None

        # Solve the QP
        try:
            sol = solvers.qp(P_cvx, q_cvx, G, h)
        except Exception as e:
            print(f"[MPC QP ERROR] Exception in solver: {e}")
            v0_star = np.zeros(3)
            self.u_star = np.concatenate([v0_star, [self.u_ref[3]]])
            return "error", float("inf")

        status = sol['status']
        if status != "optimal":
            print(f"[MPC QP] Solver status: {status}")
            v0_star = np.zeros(3)
            self.u_star = np.concatenate([v0_star, [self.u_ref[3]]])
            return status, float("inf")

        v_opt = np.array(sol['x']).flatten()
        v0_star = v_opt[0:3]

        # Yaw rate: pass nominal through with optional clipping
        if self.max_yaw_rate is None:
            yaw_rate = self.u_ref[3]
        else:
            yaw_rate = np.clip(self.u_ref[3], -self.max_yaw_rate, self.max_yaw_rate)

        self.u_star = np.concatenate([v0_star, [yaw_rate]])

        obj_val = 0.5 * float(v_opt @ (P @ v_opt) + q @ v_opt)
        return status, obj_val

    def get_optimal_control(self) -> np.ndarray:
        """
        Return the control to apply at the current step:
            [vx_0, vy_0, vz_0, yaw_rate].
        """
        return self.u_star.copy()
