# MPC_controller_drone.py

import numpy as np
from cvxopt import matrix, solvers

from .MPC_controller import MPC_Controller
from .drone import Drone  # for type clarity; not strictly required at runtime

solvers.options['show_progress'] = False


class MPC_Controller_Drone(MPC_Controller):
    """
    N-step velocity MPC for the Drone model with a first-step collision-cone
    CBF constraint.
    
    High-level model:
        p_{k+1} = p_k + v_k * dt,     k = 0,...,N-1
        p_0     = current COM position

    Decision variables:
        v_k ∈ R^3,  k = 0,...,N-1   (world-frame velocities)

    Cost:
        sum_{k=1}^N   ||p_k - p_goal||_{Q_p}^2
      + sum_{k=0}^{N-1} ||v_k||_{R_v}^2

    Constraints:
        -max_vel <= v_{k,i} <= max_vel          (box on each component)
        first-step C3CBF:  g_0^T v_0 >= d_0^{CBF}

    The C3CBF is constructed from the current local obstacle geometry, which we
    approximate as a virtual sphere around the closest obstacle point.
    """

    def __init__(self,
                 horizon: int = 10,
                 dt: float = 0.1,
                 max_vel: float = 0.5,
                 max_yaw_rate: float = None,
                 safety_margin: float = 0.1,
                 Q_p_diag=(1.0, 1.0, 1.0),
                 R_v_diag=(0.5, 0.5, 0.5),
                 **unused_kwargs):
        """
        Parameters
        ----------
        horizon : int
            MPC prediction horizon N.
        dt : float
            Discrete time step.
        max_vel : float
            Max translational speed in each axis (symmetric bounds).
        safety_margin : float
            Extra margin added on top of the drone's encompassing radius
            to define the virtual obstacle radius R = r_drone + safety_margin.
        Q_p_diag : tuple
            Diagonal of position tracking weight matrix Q_p (3 entries).
        R_v_diag : tuple
            Diagonal of velocity effort weight matrix R_v (3 entries).
        """
        super().__init__(horizon=horizon, dt=dt)

        self.max_vel = float(max_vel)
        self.max_yaw_rate = float(max_yaw_rate) if max_yaw_rate is not None else None
        self.safety_margin = float(safety_margin)

        self.Q_p = np.diag(Q_p_diag).astype(float)
        self.R_v = np.diag(R_v_diag).astype(float)

        # Reference control (used mainly for yaw_rate and as nominal v for CBF)
        self.u_ref = np.zeros(4)
        # Goal position (default: None -> no position tracking)
        self.p_goal = None

        # Storage for MPC solution
        self.u_star = np.zeros(4)

        # Precomputed for each setup_QP call
        self.p0 = None          # current position
        self.has_cbf = False
        self.g0 = None          # gradient of CBF at v_ref
        self.d0_cbf = None      # right-hand side of linearized CBF
        self.R_virtual = None   # virtual obstacle radius

    # -------------------------------------------------------------------------
    # Interface methods
    # -------------------------------------------------------------------------

    def set_reference_control(self, u_ref: np.ndarray):
        """
        Interpret u_ref as nominal [vx, vy, vz, yaw_rate].

        For the MPC:
          - translational velocities v_k are optimized from scratch, but we can
            use u_ref[:3] as the nominal velocity for CBF linearization.
          - yaw_rate is not optimized and is passed through as u_ref[3].

        Parameters
        ----------
        u_ref : np.ndarray
            4D array-like [vx, vy, vz, yaw_rate].
        """
        u_ref = np.asarray(u_ref, dtype=float).flatten()
        if u_ref.size != 4:
            raise ValueError("u_ref must have length 4: [vx, vy, vz, yaw_rate]")
        self.u_ref = u_ref.copy()

    def get_reference_control(self) -> np.ndarray:
        return self.u_ref.copy()

    def set_goal_position(self, p_goal: np.ndarray):
        """
        Set the goal position p_goal ∈ R^3.

        Parameters
        ----------
        p_goal : np.ndarray
            3D array-like [x, y, z].
        """
        p_goal = np.asarray(p_goal, dtype=float).flatten()
        if p_goal.size != 3:
            raise ValueError("p_goal must have length 3: [x, y, z]")
        self.p_goal = p_goal.copy()

    # -------------------------------------------------------------------------
    # Geometry / CBF setup
    # -------------------------------------------------------------------------

    def _extract_closest_obstacle(self, bot: Drone, c):
        """
        Given obstacle positions `c`, pick the closest one to the drone COM.

        Parameters
        ----------
        bot : Drone
            Drone object holding the current state.
        c : list or np.ndarray
            Obstacle positions. Allowed forms:
                - [] (no obstacles)
                - [x, y, z] (single obstacle)
                - [[x1,y1,z1], [x2,y2,z2], ...]

        Returns
        -------
        (c_closest, dist_center)
            c_closest : np.ndarray (3,)
                Position of the closest obstacle center.
            dist_center : float
                Euclidean distance from drone COM to this obstacle center.
        """
        p0 = np.array([bot.x, bot.y, bot.z], dtype=float)

        if c is None or len(c) == 0:
            return None, None

        # Normalize to list of 3D points
        if isinstance(c[0], (list, tuple, np.ndarray)):
            obs_positions = [np.asarray(ci, dtype=float).flatten() for ci in c]
        else:
            obs_positions = [np.asarray(c, dtype=float).flatten()]

        dists = [np.linalg.norm(p0 - ci) for ci in obs_positions]
        idx = int(np.argmin(dists))
        return obs_positions[idx], dists[idx]

    def _setup_cbf_from_geometry(self, bot: Drone, c):
        """
        Build the linearized first-step C3CBF constraint based on current
        geometry and nominal velocity.

        This computes (and stores):
            - self.has_cbf
            - self.g0
            - self.d0_cbf

        If no valid obstacle is found, has_cbf = False.
        """
        self.has_cbf = False
        self.g0 = None
        self.d0_cbf = None

        p0 = np.array([bot.x, bot.y, bot.z], dtype=float)
        self.p0 = p0

        c_closest, dist_center = self._extract_closest_obstacle(bot, c)
        if c_closest is None:
            return  # no obstacle -> no CBF

        # Virtual spherical obstacle
        R_drone = bot.encompassing_radius
        R_virtual = R_drone + self.safety_margin
        self.R_virtual = float(R_virtual)

        # Center-to-drone vector
        r0 = p0 - c_closest   # from obstacle center to drone COM
        r0_norm = np.linalg.norm(r0)
        if r0_norm < 1e-6:
            # Pathological: drone essentially at the center; fall back to no CBF
            return

        # Surface distance (center-to-drone minus radius)
        d_surface = r0_norm - R_virtual

        # If we're already inside or too close, we can still define a CBF but
        # clamp d_surface to avoid numerical issues. In practice, a higher-level
        # emergency policy might be preferable.
        d_eff = max(d_surface, 0.0)

        # Unit normal from obstacle toward drone
        n0 = r0 / r0_norm

        # Collision-cone C3CBF parameters
        alpha0 = np.sqrt(d_eff**2 + 2.0 * d_eff * R_virtual)  # = sqrt((d+R)^2 - R^2)
        v_ref0 = self.u_ref[:3].copy()
        v_ref_norm = np.linalg.norm(v_ref0)
        eps = 1e-3

        # h0(v_ref0) = (d+R) n0^T v_ref0 + alpha0 ||v_ref0||
        h_ref = (d_eff + R_virtual) * float(n0.dot(v_ref0)) + alpha0 * v_ref_norm

        # gradient: g0 = (d+R) n0 + alpha0 * v_ref / ||v_ref||
        if v_ref_norm > eps:
            g0 = (d_eff + R_virtual) * n0 + alpha0 * v_ref0 / v_ref_norm
        else:
            # If nominal velocity is near zero, the gradient of the norm term
            # is undefined; approximate by dropping the directional part.
            g0 = (d_eff + R_virtual) * n0

        # Linearized inequality: g0^T v_0 >= d0_cbf
        d0_cbf = -h_ref + float(g0.dot(v_ref0))

        self.has_cbf = True
        self.g0 = g0
        self.d0_cbf = d0_cbf

    # -------------------------------------------------------------------------
    # MPC setup and solve
    # -------------------------------------------------------------------------

    def setup_QP(self, bot: Drone, c, c_d):
        """
        Store the current state and build the CBF linearization for the
        next MPC QP solve.

        Parameters
        ----------
        bot : Drone
            Drone object holding current state.
        c : list or np.ndarray
            Obstacle positions (see _extract_closest_obstacle for format).
        c_d : list or np.ndarray
            Obstacle velocities (unused here; static obstacles assumed for now).
        """
        # Current position
        self.p0 = np.array([bot.x, bot.y, bot.z], dtype=float)

        # Ensure we have some notion of goal; if not set, default to
        # "stay where you are" so that Q_p doesn't blow things up.
        if self.p_goal is None:
            self.p_goal = self.p0.copy()

        # Build first-step CBF based on closest obstacle
        self._setup_cbf_from_geometry(bot, c)

    def solve_QP(self, bot: Drone):
        """
        Build and solve the MPC QP with first-step C3CBF constraint.

        Parameters
        ----------
        bot : Drone
            Drone object; not strictly needed here beyond debug/logging.

        Returns
        -------
        (status, obj_value) : (str, float)
            status : 'optimal', 'infeasible', etc.
            obj_value : objective value if optimal, else +inf.
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

        # Decision variables: v = [v_0; v_1; ...; v_{N-1}] ∈ R^{3N}
        n_v = 3 * N

        # Quadratic cost: 0.5 v^T P v + q^T v
        H = np.zeros((n_v, n_v), dtype=float)
        f = np.zeros(n_v, dtype=float)

        # 1) Control effort cost: sum v_k^T R_v v_k
        for k in range(N):
            idx = 3 * k
            H[idx:idx+3, idx:idx+3] += R_v

        # 2) Position tracking cost: sum_{k=1}^N ||p_k(v) - p_goal||_{Q_p}^2
        #    where p_k = p0 + dt * sum_{j=0}^{k-1} v_j.
        for k in range(1, N + 1):
            A_k = np.zeros((3, n_v), dtype=float)
            # p_k depends on v_0,...,v_{k-1}
            for j in range(k):
                j_idx = 3 * j
                A_k[:, j_idx:j_idx+3] += np.eye(3) * dt
            diff_pg = (p0 - p_goal)  # 3D
            H += A_k.T @ Q_p @ A_k
            f += A_k.T @ Q_p @ diff_pg

        # Convert H,f to cvxopt form (P = 2H, q = 2f)
        P = 2.0 * H
        q = 2.0 * f

        P_cvx = matrix(P, tc='d')
        q_cvx = matrix(q, tc='d')

        # Inequality constraints: G v <= h
        G_list = []
        h_list = []

        # 1) Velocity box constraints: -max_vel <= v_i <= max_vel
        #    -> v_i <= max_vel,  -v_i <= max_vel
        for k in range(N):
            idx = 3 * k

            # v_k <= max_vel
            for i in range(3):
                row = np.zeros(n_v, dtype=float)
                row[idx + i] = 1.0
                G_list.append(row)
                h_list.append(max_vel)

            # -v_k <= max_vel  → v_k >= -max_vel
            for i in range(3):
                row = np.zeros(n_v, dtype=float)
                row[idx + i] = -1.0
                G_list.append(row)
                h_list.append(max_vel)

        # 2) First-step CBF: g0^T v_0 >= d0_cbf  →  -g0^T v_0 <= -d0_cbf
        if self.has_cbf and self.g0 is not None and self.d0_cbf is not None:
            row = np.zeros(n_v, dtype=float)
            row[0:3] = -self.g0  # -g0^T v_0
            G_list.append(row)
            h_list.append(-self.d0_cbf)

        if len(G_list) > 0:
            G = matrix(np.vstack(G_list), tc='d')
            h = matrix(np.array(h_list, dtype=float), tc='d')
        else:
            # No inequality constraints (should not happen in practice)
            G = None
            h = None

        # Solve the QP
        try:
            sol = solvers.qp(P_cvx, q_cvx, G, h)
        except Exception as e:
            print(f"[MPC QP ERROR] Exception in solver: {e}")
            # Fallback: emergency stop (zero velocity) but keep yaw_rate
            v0_star = np.zeros(3)
            self.u_star = np.concatenate([v0_star, [self.u_ref[3]]])
            return "error", float('inf')

        status = sol['status']
        if status != 'optimal':
            print(f"[MPC QP] Solver status: {status}")
            # Fallback: emergency stop (zero velocity) but keep yaw_rate
            v0_star = np.zeros(3)
            self.u_star = np.concatenate([v0_star, [self.u_ref[3]]])
            return status, float('inf')

        v_opt = np.array(sol['x']).flatten()
        v0_star = v_opt[0:3]  # first-step velocity

        # Construct 4D control: [vx, vy, vz, yaw_rate]
        self.u_star = np.concatenate([v0_star, [self.u_ref[3]]])

        # Objective value: 0.5 v^T P v + q^T v
        obj_val = 0.5 * float(v_opt @ (P @ v_opt) + q @ v_opt)
        return status, obj_val

    def get_optimal_control(self) -> np.ndarray:
        """
        Return the control to apply at the current time step:
            [vx_0, vy_0, vz_0, yaw_rate_ref]

        vx,vy,vz are the optimized first-step velocities,
        yaw_rate is simply passed through from u_ref.
        """
        return self.u_star.copy()