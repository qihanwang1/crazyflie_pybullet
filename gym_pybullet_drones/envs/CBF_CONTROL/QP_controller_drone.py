import sympy
import numpy as np
from sympy import symbols, diff, Matrix, simplify
from math import cos, sin, tan, pi, sqrt
from sympy import *
from cvxopt import matrix, solvers
import time

# custom imports
from .QP_controller import QP_Controller

from .drone import Drone

solvers.options['show_progress'] = False

def norm(x, y, z):
    return sqrt(x**2 + y**2 + z**2)

class QP_Controller_Drone(QP_Controller):
    def __init__(self, gamma:float, obs_radius=0.1):
        """
        Constructor creates QP_Controller object for Multi Agent system where
        each agent is a Drone model bot.

        Parameters
        ----------
        gamma : float
            class k function is taken as simple function f(x) = gamma x. The
            valueo f gamma is assumed to 1 for all the agents for time being.
        no_of_agents : int
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gamma = gamma
        self.u_ref = None
        self.u_star = None
        self.G = 9.81
        self.kf = 3.16e-10
        self.obs_r = obs_radius

    def set_reference_control(self, u_ref:np.ndarray):
        """
        sets the reference controller

        Parameters
        ----------
        u_ref : numpy.ndarray
            numpy array of size (4, 1) containing the reference velocity controls [vx, vy, vz, yaw_rate].

        Returns
        -------
        None.

        """
        self.u_ref = u_ref
        self.u_star = np.copy(u_ref)

    def get_reference_control(self):
        """
        Returns the reference control input (required by abstract base class)

        Returns
        -------
        numpy.ndarray
            Array containing reference control [vx, vy, vz, yaw_rate]
        """
        if self.u_ref is None:
            return np.zeros(4)
        return self.u_ref.flatten()

    def get_optimal_control(self):
        """
        Returns the optimal velocity commands from QP-CBF

        Returns
        -------
        numpy.ndarray
            Array of shape (4,) containing [vx, vy, vz, yaw_rate] velocity commands
        """
        # For velocity-based control, directly return the optimal velocity commands
        return self.u_star.flatten()


    def setup_QP(self, bot, c, c_d):
        """
        the function takes bot and obstacle info and creates symbolic variables
        associated with the bot required for computation in QP. The function also
        precomputes the required symbolic expressions for QP such as L_f, L_g,
        etc...

        Parameters
        ----------
        bot : Drone object
            Drone bot object
        c : list or list of lists
            Obstacle position(s). Can be:
            - Empty list: [] (no obstacles)
            - Single obstacle: [x, y, z]
            - Multiple obstacles: [[x1, y1, z1], [x2, y2, z2], ...]
        c_d : list or list of lists
            Obstacle velocity(ies). Same format as c

        Returns
        -------
        None.

        """
        # Handle empty obstacle list
        if len(c) == 0:
            self.obstacles = []
            self.num_obstacles = 0
        # Handle both single obstacle and multiple obstacles
        elif isinstance(c[0], (list, np.ndarray)):
            # Multiple obstacles
            self.obstacles = [(c[i], c_d[i]) for i in range(len(c))]
            self.num_obstacles = len(self.obstacles)
        else:
            # Single obstacle - convert to list format
            self.obstacles = [(c, c_d)]
            self.num_obstacles = 1
        self.u_ref = self.u_ref.reshape((4,1))

        # Create placeholders for symbolic expressions
        self.f = [0] # f Matrix in control system
        self.g = [0] # g Matrix in control system
        self.h_list = []  # List of C3BF expressions (one per obstacle)

        # Create placeholder for terms in QP
        self.Psi_list = []  # List of Psi values (one per obstacle)
        self.B = [[]]
        self.C_list = []  # List of C matrices (one per obstacle)

        # If no obstacles, skip all symbolic computation
        if self.num_obstacles == 0:
            return

        # create state and parameter symbolic varaibles for each bot

        symbols_string = 'x y z x_d y_d z_d phi theta psi w_1 w_2 w_3 L Ixx Iyy Izz m l r'
        bot.sym_x, bot.sym_y, bot.sym_z, bot.sym_x_d, bot.sym_y_d, bot.sym_z_d, bot.sym_phi, bot.sym_theta, bot.sym_psi, bot.sym_w_1, bot.sym_w_2, bot.sym_w_3, bot.sym_L, bot.sym_Ixx, bot.sym_Iyy, bot.sym_Izz, bot.sym_m, bot.sym_l, bot.sym_r =  symbols(symbols_string)


        # f-matrix for direct velocity control (full 12-state dynamics)
        self.f = Matrix([
            bot.sym_x_d,  # x_dot = current x velocity
            bot.sym_y_d,  # y_dot = current y velocity  
            bot.sym_z_d,  # z_dot = current z velocity
            0,            # x_ddot = 0 (no acceleration without control)
            0,            # y_ddot = 0
            0,            # z_ddot = 0  
            bot.sym_w_1 + bot.sym_w_2*sin(bot.sym_phi)*tan(bot.sym_theta) + bot.sym_w_3*cos(bot.sym_phi)*tan(bot.sym_theta),  # phi_dot
            bot.sym_w_2*cos(bot.sym_phi) - bot.sym_w_3*sin(bot.sym_phi),  # theta_dot
            (bot.sym_w_2*sin(bot.sym_phi) + bot.sym_w_3*cos(bot.sym_phi))/cos(bot.sym_theta), # psi_dot (using full kinematics)
            0, 0, 0       # w_1_dot, w_2_dot, w_3_dot (no drift in angular accelerations)
        ])

        # f-matrix for thrust inputs
        # self.f = Matrix([bot.sym_x_d,
        #                 bot.sym_y_d,
        #                 bot.sym_z_d,
        #                 0,
        #                 0,
        #                 - self.G,
        #                 bot.sym_w_1 + bot.sym_w_2*sin(bot.sym_phi)*tan(bot.sym_theta) + bot.sym_w_3*cos(bot.sym_phi)*tan(bot.sym_theta),
        #                 bot.sym_w_2*cos(bot.sym_phi) - bot.sym_w_3*sin(bot.sym_phi),
        #                 (bot.sym_w_2*sin(bot.sym_phi) + bot.sym_w_3*cos(bot.sym_phi))/cos(bot.sym_theta),
        #                 (bot.sym_Iyy - bot.sym_Izz)*bot.sym_w_2*bot.sym_w_3/bot.sym_Ixx,
        #                 (bot.sym_Izz - bot.sym_Ixx)*bot.sym_w_1*bot.sym_w_3/bot.sym_Iyy,
        #                 (bot.sym_Ixx - bot.sym_Iyy)*bot.sym_w_1*bot.sym_w_2/bot.sym_Izz])

        # g-matrix for direct velocity control
        self.g = Matrix([
            [0, 0, 0, 0],  # x_dot: already determined by x_d in f
            [0, 0, 0, 0],  # y_dot: already determined by y_d in f
            [0, 0, 0, 0],  # z_dot: already determined by z_d in f
            [1, 0, 0, 0],  # x_ddot = vx_cmd (control sets velocity directly)
            [0, 1, 0, 0],  # y_ddot = vy_cmd
            [0, 0, 1, 0],  # z_ddot = vz_cmd
            [0, 0, 0, 0],  # phi_dot (attitude kinematics from f)
            [0, 0, 0, 0],  # theta_dot
            [0, 0, 0, 0],  # psi_dot: keep using kinematics from f, not direct yaw rate
            [0, 0, 0, 0],  # w_1_dot
            [0, 0, 0, 0],  # w_2_dot
            [0, 0, 0, 0]   # w_3_dot
        ])
        # g-matrix for thrust inputs
        # p = (cos(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) + sin(bot.sym_psi)*sin(bot.sym_phi))/bot.sym_m
        # q = (sin(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) - cos(bot.sym_psi)*sin(bot.sym_phi))/bot.sym_m
        # r = (cos(bot.sym_theta)*cos(bot.sym_phi))/bot.sym_m
        # self.g = Matrix([[0, 0, 0, 0],
        #                 [0, 0, 0, 0],
        #                 [0, 0, 0, 0],
        #                 [p, p, p, p],
        #                 [q, q, q, q],
        #                 [r, r, r, r],
        #                 [0, 0, 0, 0],
        #                 [0, 0, 0, 0],
        #                 [0, 0, 0, 0],
        #                 [0, bot.sym_L/bot.sym_Iyy, 0, -bot.sym_L/bot.sym_Iyy],
        #                 [bot.sym_L/bot.sym_Ixx, 0, -bot.sym_L/bot.sym_Ixx, 0],
        #                 [0, 0, 0, 0]])



        # for CBF h - compute for each obstacle
        # Bot rotation matrices (computed once, used for all obstacles)
        r_1_x = (cos(bot.sym_psi)*cos(bot.sym_theta))
        r_1_y = (sin(bot.sym_psi)*cos(bot.sym_theta))
        r_1_z = (-sin(bot.sym_theta))

        r_2_x = (cos(bot.sym_psi)*sin(bot.sym_theta)*sin(bot.sym_phi) - sin(bot.sym_psi)*cos(bot.sym_phi))
        r_2_y = (sin(bot.sym_psi)*sin(bot.sym_theta)*sin(bot.sym_phi) - cos(bot.sym_psi)*cos(bot.sym_phi))
        r_2_z = (cos(bot.sym_theta)*sin(bot.sym_phi))

        self.x_i = r_1_x
        self.y_i = r_1_y
        self.z_i = r_1_z

        self.x_j = r_2_x
        self.y_j = r_2_y
        self.z_j = r_2_z

        self.x_d_i = -bot.sym_w_3*r_1_y + bot.sym_w_2*r_1_z
        self.y_d_i = -bot.sym_w_1*r_1_z + bot.sym_w_3*r_1_x
        self.z_d_i = -bot.sym_w_2*r_1_x + bot.sym_w_1*r_1_y

        self.x_d_j = -bot.sym_w_3*r_2_y + bot.sym_w_2*r_2_z
        self.y_d_j = -bot.sym_w_1*r_2_z + bot.sym_w_3*r_2_x
        self.z_d_j = -bot.sym_w_2*r_2_x + bot.sym_w_1*r_2_y

        r_x = (cos(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) + sin(bot.sym_psi)*sin(bot.sym_phi))
        r_y = (sin(bot.sym_psi)*sin(bot.sym_theta)*cos(bot.sym_phi) - cos(bot.sym_psi)*sin(bot.sym_phi))
        r_z = (cos(bot.sym_theta)*cos(bot.sym_phi))

        # Loop through each obstacle to create CBF constraints
        for obs_pos, obs_vel in self.obstacles:
            c_x, c_y, c_z = obs_pos
            c_x_d, c_y_d, c_z_d = obs_vel

            # Relative position terms
            p_rel_x = c_x - (bot.sym_x + bot.sym_l*r_x)
            p_rel_y = c_y - (bot.sym_y + bot.sym_l*r_y)
            p_rel_z = c_z - (bot.sym_z + bot.sym_l*r_z)

            # Relative velocity terms
            v_rel_x = c_x_d - (bot.sym_x_d + bot.sym_l*(-bot.sym_w_3*r_y + bot.sym_w_2*r_z))
            v_rel_y = c_y_d - (bot.sym_y_d + bot.sym_l*(-bot.sym_w_1*r_z + bot.sym_w_3*r_x))
            v_rel_z = c_z_d - (bot.sym_z_d + bot.sym_l*(-bot.sym_w_2*r_x + bot.sym_w_1*r_y))

            # 3-D C3BF Candidate for this obstacle
            # Modified to properly handle static obstacles
            epsilon = 0.01  # Small safety margin to prevent domain errors

            # For static/slow obstacles, modify C3BF to use proper collision geometry
            # The standard C3BF: h = p·v + ||v||*sqrt(||p||² - r²)
            # For static obstacles (v_obs = 0), this becomes:
            # h = -p·v_drone + ||v_drone||*sqrt(||p||² - r²)
            #
            # The issue: the second term ||v_drone||*sqrt(...) is always positive,
            # which can make h > 0 even when approaching. We need to ensure h < 0 when approaching.
            #
            # Fix: Use the signed version for static obstacles:
            # h = p·v + sign(p·v)*||v||*sqrt(||p||² - r²)
            # This makes h negative when approaching (p·v < 0)

            dot_product = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z
            v_rel_norm = norm(v_rel_x, v_rel_y, v_rel_z)
            p_rel_norm_sq = norm(p_rel_x, p_rel_y, p_rel_z)**2

            # Add epsilon² to prevent sqrt of negative (when inside safety radius)
            sqrt_term = sqrt(p_rel_norm_sq - bot.sym_r**2 + epsilon**2)

            # For better static obstacle handling: use signed collision cone
            # This ensures h < 0 when approaching the obstacle
            h = dot_product - v_rel_norm * sqrt_term

            self.h_list.append(h)

        # # HO-CBF Candidate
        # gamma = 1
        # p = 0.5
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y + p_rel_z*v_rel_z \
        #     + gamma*(norm(p_rel_x, p_rel_y, p_rel_z)**2 - bot.sym_r**2)**p

        # self.h =1

        # # 2D HO-CBF Candidate - z
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
        #     + norm(1, 1, 1)*sqrt(norm(p_rel_x, p_rel_y, 0.0001)**2 - bot.sym_r**2)

        # # 2D HO-CBF Candidate - x
        # self.h = p_rel_z*v_rel_z + p_rel_y*v_rel_y \
        #     + sqrt(norm(p_rel_z, p_rel_y, 0.0001)**2 - bot.sym_r**2)

        # # 2-D C3BF Candidate - z
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
        #     + norm(v_rel_x, v_rel_y,0.001)*sqrt(norm(p_rel_x, p_rel_y,0.001)**2 - bot.sym_r**2)

        # # 2-D C3BF Candidate - x
        # self.h = p_rel_z*v_rel_z + p_rel_y*v_rel_y \
        #     + norm(0.001, v_rel_y,v_rel_z)*sqrt(norm(0.001, p_rel_y,p_rel_z)**2 - bot.sym_r**2)

        # # 2-D HO-CBF Candidate
        # gamma = 1
        # p = 0.5
        # self.h = p_rel_x*v_rel_x + p_rel_y*v_rel_y \
        #     + gamma*(norm(p_rel_x, p_rel_y, 0.00001)**2 - bot.sym_r**2)**p

        # # # Classical CBF
        # self.h = norm(c_x - bot.sym_x, c_y - bot.sym_y, c_z - bot.sym_z)**2/bot.sym_r**2 -1

        # Compute gradient and constraint matrices for each obstacle
        # For full 12-state system, compute all partial derivatives
        for h in self.h_list:
            rho_h_by_rho_x = diff(h, bot.sym_x)
            rho_h_by_rho_y = diff(h, bot.sym_y)
            rho_h_by_rho_z = diff(h, bot.sym_z)
            rho_h_by_rho_x_d = diff(h, bot.sym_x_d)
            rho_h_by_rho_y_d = diff(h, bot.sym_y_d)
            rho_h_by_rho_z_d = diff(h, bot.sym_z_d)
            rho_h_by_rho_phi = diff(h, bot.sym_phi)
            rho_h_by_rho_theta = diff(h, bot.sym_theta)
            rho_h_by_rho_psi = diff(h, bot.sym_psi)
            rho_h_by_rho_w_1 = diff(h, bot.sym_w_1)
            rho_h_by_rho_w_2 = diff(h, bot.sym_w_2)
            rho_h_by_rho_w_3 = diff(h, bot.sym_w_3)

            # Jacobian: 1x12 for full 12-state system [x, y, z, x_d, y_d, z_d, phi, theta, psi, w_1, w_2, w_3]
            Delta_h_wrt_bot = Matrix([[rho_h_by_rho_x,
                                        rho_h_by_rho_y,
                                        rho_h_by_rho_z,
                                        rho_h_by_rho_x_d,
                                        rho_h_by_rho_y_d,
                                        rho_h_by_rho_z_d,
                                        rho_h_by_rho_phi,
                                        rho_h_by_rho_theta,
                                        rho_h_by_rho_psi,
                                        rho_h_by_rho_w_1,
                                        rho_h_by_rho_w_2,
                                        rho_h_by_rho_w_3]])

            C = Delta_h_wrt_bot*self.g
            self.C_list.append(C)

            n = C * self.u_ref
            n_f = Delta_h_wrt_bot*self.f
            Psi = self.gamma*h
            Psi += n_f[0] + n[0]
            self.Psi_list.append(Psi)

    def solve_QP(self, bot):
        """
        Solving Quadratic Program to set the optimal controls. This function
        substitutes the values in symbolic expressions and solves the QP using
        cvxopt for multiple CBF constraints.

        QP formulation:
        minimize:   (1/2) * ||u - u_ref||^2
        subject to: C_i * u + Psi_i >= 0  for each obstacle i

        Parameters
        ----------
        bot : Drone object
            Drone bot object

        Returns
        -------
        TYPE: tuple of 2 numpy arrays
            first numpy array in the tuple returns of state of CBF if they are
            active or inactive 1 denotes active CBF and 0 denotes inactive CBF.
            second numpy array in the tuple returns the value of CBF since in
            C3BF the value of the function is directly proportional to how
            unsafe system is.
        """
        # If no obstacles, just use reference control
        if self.num_obstacles == 0:
            self.u_star = self.u_ref
            return ((0, 0), (0, 0))

        # build value substitution list
        uk_vs = [bot.sym_x, bot.sym_y, bot.sym_z,
                bot.sym_x_d, bot.sym_y_d, bot.sym_z_d,
                bot.sym_phi, bot.sym_theta, bot.sym_psi,
                bot.sym_w_1, bot.sym_w_2, bot.sym_w_3,
                bot.sym_L, bot.sym_Ixx, bot.sym_Iyy, bot.sym_Izz,
                bot.sym_m, bot.sym_l, bot.sym_r]

        uk_gs = [bot.x, bot.y, bot.z,
                 bot.x_dot, bot.y_dot, bot.z_dot,
                 bot.phi, bot.theta, bot.psi,
                 bot.w_1, bot.w_2, bot.w_3,
                 bot.L, bot.Ixx, bot.Iyy, bot.Izz,
                 bot.m, bot.l, bot.encompassing_radius + self.obs_r]

        d = {uk: uk_gs[i] for i, uk in enumerate(uk_vs)}

        # Evaluate all CBF constraints
        h_values = []
        Psi_values = []
        C_matrices = []

        for i in range(self.num_obstacles):
            h_val = np.array(re(self.h_list[i].xreplace(d)), dtype=np.float64)
            Psi_val = np.array(re(self.Psi_list[i].xreplace(d)))
            C_val = np.array(re(self.C_list[i].xreplace(d)))

            h_values.append(h_val)
            Psi_values.append(float(Psi_val))  # Convert to float
            C_matrices.append(C_val)

        # Check if any constraints are violated
        # Threshold for CBF activation (Psi < threshold means getting close to unsafe)
        # Recommended values:
        #   0.001 = very late activation (theoretical boundary)
        #   0.01  = moderate activation
        #   0.1   = early activation (more conservative) <-- CURRENT
        #   0.5   = very early activation (very conservative)
        CBF_THRESHOLD = 0.1
        active_constraints = [Psi < CBF_THRESHOLD for Psi in Psi_values]

        # DEBUG: Print CBF values only when constraints are active
        if len(Psi_values) > 0 and any(active_constraints):
            timestamp = time.time()
            active_indices = [i for i, active in enumerate(active_constraints) if active]
            print(f"\n[CBF TRIGGERED @ t={timestamp:.3f}] Threshold={CBF_THRESHOLD}, {len(active_indices)}/{len(Psi_values)} obstacles active")
            print(f"[CBF] Active obstacles: {active_indices}")
            print(f"[CBF] Psi values: {[f'{p:.4f}' for p in Psi_values]} (< {CBF_THRESHOLD} triggers)")
            print(f"[CBF] h values: {[f'{float(h):.4f}' for h in h_values]}")
            print(f"[CBF] u_ref: [{self.u_ref[0,0]:.3f}, {self.u_ref[1,0]:.3f}, {self.u_ref[2,0]:.3f}, {self.u_ref[3,0]:.3f}]")

        if not any(active_constraints):
            # No constraints violated, use reference control
            self.u_star = self.u_ref
        else:

            # P matrix for quadratic cost (diagonal weights)
            # Control vector: u = [vx, vy, vz, yaw_rate]
            # Higher weight = penalize deviation more (stay closer to reference)
            # Lower weight = allow more deviation for safety
            #
            # Weight preferences for obstacle avoidance:
            # - vx, vy (horizontal): Lower weight = easier to deviate horizontally (preferred)
            # - vz (vertical): Higher weight = harder to deviate vertically (discouraged)
            # - yaw_rate: Medium weight = allow some rotation for avoidance
            P = matrix(np.diag([0.75, 0.75, 5.0, 1.0]), tc='d')  # [vx, vy, vz, yaw_rate]

            # q vector for linear cost
            q = matrix(-self.u_ref.flatten(), tc='d')

            # Stack inequality constraints: -C_i * u <= Psi_i for active constraints
            G_list = []
            h_list = []
            for i, is_active in enumerate(active_constraints):
                if is_active:
                    G_list.append(-C_matrices[i].reshape(1, 4))
                    h_list.append(Psi_values[i])

            # Add velocity bounds: -v_max <= u <= v_max
            # For Crazyflie: v_max = 0.5 m/s for safety
            v_max = 0.5  # m/s
            yaw_rate_max = 1.0  # rad/s

            # Add box constraints: u <= [v_max, v_max, v_max, yaw_rate_max]
            G_list.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
            h_list.extend([v_max, v_max, v_max, yaw_rate_max])

            # Add box constraints: -u <= v_max (i.e., u >= -v_max)
            G_list.append(np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]))
            h_list.extend([v_max, v_max, v_max, yaw_rate_max])

            if G_list:
                # Ensure proper dtype for cvxopt
                G = matrix(np.vstack(G_list).astype(np.float64), tc='d')
                h = matrix(np.array(h_list, dtype=np.float64).flatten(), tc='d')

                # Solve QP
                try:
                    sol = solvers.qp(P, q, G, h)
                    if sol['status'] == 'optimal':
                        self.u_star = np.array(sol['x']).reshape((4, 1))
                        print(f"[CBF] u_star (QP solution): [{self.u_star[0,0]:.3f}, {self.u_star[1,0]:.3f}, {self.u_star[2,0]:.3f}, {self.u_star[3,0]:.3f}]")
                    else:
                        print(f"[CBF EMERGENCY STOP] QP solver status: {sol['status']}, stopping for safety")
                        self.u_star = np.zeros((4, 1))  # Emergency stop instead of continuing
                except Exception as e:
                    print(f"[CBF EMERGENCY STOP] QP solve error: {e}, stopping for safety")
                    self.u_star = np.zeros((4, 1))  # Emergency stop instead of continuing
            else:
                self.u_star = self.u_ref

        # Return constraint status information
        state_of_h = tuple(active_constraints[:2] if len(active_constraints) >= 2 else (0, 0))
        term_h = tuple(h_values[:2] if len(h_values) >= 2 else (0, 0))
        return (state_of_h, term_h)

