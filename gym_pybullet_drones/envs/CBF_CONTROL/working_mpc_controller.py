import numpy as np
from cvxopt import matrix, solvers

from .MPC_controller import MPC_Controller
from .drone import Drone  # for type clarity; not strictly required at runtime

solvers.options['show_progress'] = False


class MPC_Controller_Drone(MPC_Controller):
    """
    MPC controller for a single drone with hard spherical obstacle avoidance.

    - Decision variables: translational velocities v_k in R^3 for k = 0,...,N-1
    - Dynamics: p_{k+1} = p_k + dt * v_k  (simple discrete-time integrator)
    - Cost:
        * Quadratic penalty on velocities (smooth control)
        * Quadratic penalty on position tracking w.r.t. p_goal over the horizon
    - Constraints:
        * Box constraints on v_k: |v_k,i| <= max_vel
        * For each spherical obstacle j and each step k,
          enforce that the predicted position p_k stays in the
          "outside" half-space of a tangent plane to the sphere,
          which guarantees that the trajectory never enters
          the interior of the sphere.
    """

    def __init__(self,
                 horizon: int = 10,
                 dt: float = 0.1,
                 max_vel: float = 5.0,
                 max_yaw_rate: float = None,
                 safety_margin: float = 0.2,
                 Q_p_diag=(1.0, 1.0, 1.0),
                 R_v_diag=(0.5, 0.5, 0.5),
                 obstacle_radius: float = 0.1,
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

        # Reference control (mainly to pass yaw rate through)
        self.u_ref = np.zeros(4)

        # Goal position for the MPC tracking cost
        self.p_goal = None

        # Last computed optimal control [vx, vy, vz, yaw_rate]
        self.u_star = np.zeros(4)

        # State at the current MPC solve
        self.p0 = None  # current drone position

        # Cached obstacle data for the current solve
        self._obs_centers = None
        self._obs_normals = None
        self._obs_rhs = None  # right-hand side of position half-space constraints

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
    # Obstacle handling
    # -------------------------------------------------------------------------

    def _build_obstacle_halfspaces(self, bot: Drone, centers):
        """
        For each spherical obstacle with center c_j and total safety radius R_total,
        build a separating half-space of the form

            n_j^T (p - c_j) >= R_total

        where n_j is the unit vector from the obstacle center c_j to the
        current drone position p0. All points satisfying this inequality are
        outside (or on the tangent) of the closed ball of radius R_total.

        At the QP level we will enforce this constraint at each predicted
        position p_k along the horizon.
        """
        if centers is None or len(centers) == 0:
            self._obs_centers = None
            self._obs_normals = None
            self._obs_rhs = None
            return

        p0 = np.array([bot.x, bot.y, bot.z], dtype=float)
        self.p0 = p0

        # Normalize to list of 3D centers
        if isinstance(centers[0], (list, tuple, np.ndarray)):
            obs_positions = [np.asarray(ci, dtype=float).flatten() for ci in centers]
        else:
            obs_positions = [np.asarray(centers, dtype=float).flatten()]

        # Effective safety radius: obstacle sphere + drone radius + margin
        R_drone = float(bot.encompassing_radius)
        R_total = self.obstacle_radius + R_drone + self.safety_margin

        normals = []
        rhs_list = []
        centers_list = []

        for c in obs_positions:
            r_vec = p0 - c
            dist = np.linalg.norm(r_vec)

            if dist < 1e-6:
                # Degenerate case: drone essentially at the obstacle center
                n = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                n = r_vec / dist

            # Position-space inequality: n^T (p - c) >= R_total
            # Equivalent to n^T p >= R_total + n^T c.
            normals.append(n)
            centers_list.append(c)
            rhs_list.append(R_total + float(n.dot(c)))

        self._obs_centers = centers_list
        self._obs_normals = normals
        self._obs_rhs = rhs_list

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

        # Build half-space representation for all spherical obstacles
        self._build_obstacle_halfspaces(bot, c)

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

        # 2) Collision-cone CBF on ALL steps v_k:
        #    g0^T v_k >= d0_cbf  for k = 0,...,N-1
        #    -> -g0^T v_k <= -d0_cbf
        if self.has_cbf and self.g0 is not None and self.d0_cbf is not None:
            for k in range(N):
                row = np.zeros(n_v, dtype=float)
                row[3 * k:3 * k + 3] = -self.g0
                G_list.append(row)
                h_list.append(-self.d0_cbf)

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
