import numpy as np
from gymnasium import spaces
import socket
import math
import os
import sys

from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Import MPC controller for velocity-based safety / planning
from gym_pybullet_drones.envs.CBF_CONTROL.MPC_controller_drone import MPC_Controller_Drone
from gym_pybullet_drones.envs.CBF_CONTROL.drone import Drone

try:
    import pycffirmware as firm
except ImportError:
    raise "PyCFFirmware required for CF Aviary. Please install it from https://github.com/utiasDSL/pycffirmware or use a different aviary class."


class CFAviary_MPC(BaseAviary):
    """Multi-drone environment class for use of Crazyflie controller with MPC safety layer."""
    ACTION_DELAY = 0  # how many firmware loops run between the controller commanding an action and the drone motors responding to it
    SENSOR_DELAY = 0  # how many firmware loops run between experiencing a motion and the sensors registering it
    STATE_DELAY = 0   # not yet supported, keep 0
    CONTROLLER = 'mellinger'  # specifies controller type

    # Configurations to match firmware. Not recommended to change
    GYRO_LPF_CUTOFF_FREQ = 80
    ACCEL_LPF_CUTOFF_FREQ = 30
    QUAD_FORMATION_X = True
    MOTOR_SET_ENABLE = True

    RAD_TO_DEG = 180 / math.pi

    ################################################################################

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
                 use_mpc=False,
                 mpc_params=None,
                 ):
        """Initialization of an aviary environment for use of Crazyflie controller with optional MPC filter.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The *external* control frequency (used for trajectories / logging etc).
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        output_folder : str, optional
            Folder to store logs.
        verbose : bool, optional
            Verbosity flag.
        use_mpc : bool, optional
            Whether to enable the MPC safety / planning layer.
        mpc_params : dict, optional
            Parameters passed to MPC_Controller_Drone, e.g.
            {
                'horizon': 10,
                'dt': 0.04,           # if not provided, uses 1/ctrl_freq
                'max_vel': 0.5,
                'max_yaw_rate': 1.0,
                'safety_margin': 0.2,
            }
        """
        firmware_freq = 500 if self.CONTROLLER == "mellinger" else 1000
        assert (pyb_freq % firmware_freq == 0), f"pyb_freq ({pyb_freq}) must be a multiple of firmware_freq ({firmware_freq}) for CFAviary_MPC."
        if num_drones != 1:
            raise NotImplementedError("Multi-agent support for CFAviary_MPC is not yet implemented.")

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=firmware_freq,  # self.CTRL_FREQ = firmware_freq
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

        # Initialize connection to Crazyflie controller
        self.firmware_freq = firmware_freq
        self.ctrl_freq = ctrl_freq

        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.verbose = verbose

        # MPC usage flag
        self.use_mpc = use_mpc
        self.mpc_params = mpc_params if mpc_params is not None else {}

        # Initialize firmware (sets ctrl_dt, firmware_dt later)
        self._initalize_cffirmware()

        # Initialize MPC controller if requested
        if self.use_mpc:
            self._initialize_mpc_controller(self.mpc_params)

    def _initialize_mpc_controller(self, mpc_params):
        """Initialize the MPC controller for velocity-based safety / planning.

        Parameters
        ----------
        mpc_params : dict
            Parameters for MPC_Controller_Drone.
        """
        # Load drone configuration for geometric / margin info
        drone_cfg_path = os.path.join(os.path.dirname(__file__), 'CBF_CONTROL', 'drone1.json')
        if not os.path.exists(drone_cfg_path):
            print(f"[WARNING] Drone config not found at {drone_cfg_path}, disabling MPC.")
            self.drone = None
            self.mpc_controller = None
            self.use_mpc = False
            return

        self.drone = Drone.from_JSON(drone_cfg_path)

        # Use ctrl_freq for MPC dt by default (outer control step)
        default_dt = 1.0 / self.ctrl_freq if self.ctrl_freq > 0 else 0.04

        self.mpc_controller = MPC_Controller_Drone(
            horizon=mpc_params.get('horizon', 10),
            dt=mpc_params.get('dt', default_dt),
            max_vel=mpc_params.get('max_vel', 5),
            max_yaw_rate=mpc_params.get('max_yaw_rate', 1.0),
            safety_margin=mpc_params.get('safety_margin', 0.5),
            obstacle_radius=0.1,              
        )

        print(f"[CFAviary_MPC] MPC controller initialized with params: "
              f"horizon={self.mpc_controller.horizon}, dt={self.mpc_controller.dt}, "
              f"max_vel={self.mpc_controller.max_vel}, max_yaw_rate={self.mpc_controller.max_yaw_rate}, "
              f"safety_margin={self.mpc_controller.safety_margin}")

    def _initalize_cffirmware(self):
        """Resets the firmware_wrapper object.

        Todo:
            * Add support for state estimation
        """
        self.states = []
        self.takeoff_sent = False

        # Initialize history
        self.action_history = [[0, 0, 0, 0] for _ in range(self.ACTION_DELAY)]
        self.sensor_history = [[[0, 0, 0], [0, 0, 0]] for _ in range(self.SENSOR_DELAY)]
        self.state_history = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] for _ in range(self.STATE_DELAY)]

        # Initialize gyro lpf
        self.acclpf = [firm.lpf2pData() for _ in range(3)]
        self.gyrolpf = [firm.lpf2pData() for _ in range(3)]
        for i in range(3):
            firm.lpf2pInit(self.acclpf[i], self.firmware_freq, self.GYRO_LPF_CUTOFF_FREQ)
            firm.lpf2pInit(self.gyrolpf[i], self.firmware_freq, self.ACCEL_LPF_CUTOFF_FREQ)

        # Initialize state objects
        self.control = firm.control_t()
        self.setpoint = firm.setpoint_t()
        self.sensorData = firm.sensorData_t()
        self.state = firm.state_t()
        self.tick = 0
        self.pwms = [0, 0, 0, 0]
        self.action = np.array([[0, 0, 0, 0]])
        self.command_queue = []

        self.tumble_counter = 0
        self.prev_vel = np.array([0, 0, 0])
        self.prev_rpy = np.array([0, 0, 0])
        self.prev_time_s = None
        self.last_pos_pid_call = 0
        self.last_att_pid_call = 0

        # Initialize state flags
        self._error = False
        self.sensorData_set = False
        self.state_set = False
        self.full_state_cmd_override = True  # When true, high level commander is not called

        # Initialize controller
        if self.CONTROLLER == 'pid':
            firm.controllerPidInit()
            print('PID controller init test:', firm.controllerPidTest())
        elif self.CONTROLLER == 'mellinger':
            firm.controllerMellingerInit()
            assert (self.firmware_freq == 500), "Mellinger controller requires a firmware frequency of 500Hz."
            print('Mellinger controller init test:', firm.controllerMellingerTest())

        # Reset environment
        init_obs, init_info = super().reset()
        init_pos = np.array([init_obs[0][0], init_obs[0][1], init_obs[0][2]])   # global coord, m
        init_vel = np.array([init_obs[0][10], init_obs[0][11], init_obs[0][12]])  # global coord, m/s
        init_rpy = np.array([init_obs[0][7], init_obs[0][8], init_obs[0][9]])  # body coord, rad
        if self.NUM_DRONES > 1:
            raise NotImplementedError("Firmware controller wrapper does not support multiple drones.")

        # Initialize high level commander
        firm.crtpCommanderHighLevelInit()
        self._update_state(0, init_pos, init_vel, np.array([0.0, 0.0, 1.0]), init_rpy * self.RAD_TO_DEG)
        self._update_initial_state(init_obs[0])
        firm.crtpCommanderHighLevelTellState(self.state)

        self.ctrl_dt = 1 / self.ctrl_freq
        self.firmware_dt = 1 / self.firmware_freq

        # Visualization
        self.first_motor_killed_print = True

        return init_obs, init_info

    def step(self, i, obstacle_positions=None, obstacle_velocities=None):
        """Step the firmware_wrapper class and its environment with optional MPC safety filter.

        This function should be called once at the rate of ctrl_freq. Step processes high level commands, 
        and runs the firmware loop and simulator according to the frequencies set.

        Args
        ----
            i : int
                The simulation control step index.
            obstacle_positions : list | None
                [[x1,y1,z1], [x2,y2,z2], ...] (optional, used by MPC).
            obstacle_velocities : list | None
                [[vx1,vy1,vz1], [vx2,vy2,vz2], ...] (optional, used by MPC).

        Returns
        -------
            obs, reward, terminated, truncated, info
        """
        t = i / self.ctrl_freq

        self._process_command_queue(t)

        while self.tick / self.firmware_freq < t + self.ctrl_dt:
            # Step the environment
            obs, reward, terminated, truncated, info = super().step(self.action)

            # Get state values from pybullet
            cur_pos = np.array([obs[0][0], obs[0][1], obs[0][2]])   # global coord, m
            cur_vel = np.array([obs[0][10], obs[0][11], obs[0][12]])  # global coord, m/s
            cur_rpy = np.array([obs[0][7], obs[0][8], obs[0][9]])   # body coord, rad
            body_rot = R.from_euler('XYZ', cur_rpy).inv()

            if self.takeoff_sent:
                self.states += [[self.tick / self.firmware_freq, cur_pos[0], cur_pos[1], cur_pos[2]]]

            # Estimate rates
            cur_rotation_rates = (cur_rpy - self.prev_rpy) / self.firmware_dt  # body coord, rad/s
            self.prev_rpy = cur_rpy
            cur_acc = (cur_vel - self.prev_vel) / self.firmware_dt / 9.8 + np.array([0, 0, 1])  # global coord
            self.prev_vel = cur_vel

            # Update state
            state_timestamp = int(self.tick / self.firmware_freq * 1e3)
            if self.STATE_DELAY:
                raise NotImplementedError("State delay is not yet implemented. Leave at 0.")
            else:
                self._update_state(state_timestamp, cur_pos, cur_vel, cur_acc, cur_rpy * self.RAD_TO_DEG)

            # Update sensor data
            sensor_timestamp = int(self.tick / self.firmware_freq * 1e6)
            if self.SENSOR_DELAY:
                self._update_sensorData(sensor_timestamp, *self.sensor_history[0])
                self.sensor_history = self.sensor_history[1:] + [[body_rot.apply(cur_acc), cur_rotation_rates * self.RAD_TO_DEG]]
            else:
                self._update_sensorData(sensor_timestamp, body_rot.apply(cur_acc), cur_rotation_rates * self.RAD_TO_DEG)

            # Update setpoint from firmware's high-level commander unless overridden
            self._updateSetpoint(self.tick / self.firmware_freq)

            # DEBUG: Print setpoint velocities BEFORE MPC
            if self.tick % 500 == 0:  # every ~1s at 500 Hz
                print(f"\n[CFAviary_MPC - BEFORE MPC] tick={self.tick}, time={self.tick / self.firmware_freq:.2f}s")
                print(f"[CFAviary_MPC] Current position: [{cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f}]")
                print(f"[CFAviary_MPC] Current velocity: [{cur_vel[0]:.3f}, {cur_vel[1]:.3f}, {cur_vel[2]:.3f}]")
                print(f"[CFAviary_MPC] Setpoint position: [{self.setpoint.position.x:.3f}, {self.setpoint.position.y:.3f}, {self.setpoint.position.z:.3f}]")
                print(f"[CFAviary_MPC] Mellinger velocity output (BEFORE MPC): "
                      f"vx={self.setpoint.velocity.x:.3f}, vy={self.setpoint.velocity.y:.3f}, vz={self.setpoint.velocity.z:.3f}")

            # Apply MPC safety / planning filter if enabled
            if self.use_mpc and obstacle_positions is not None:
                self._apply_mpc_filter(cur_pos, cur_vel, cur_rpy, obstacle_positions, obstacle_velocities)

                if self.tick % 500 == 0:
                    print(f"[CFAviary_MPC - AFTER MPC] Filtered velocity: "
                          f"vx={self.setpoint.velocity.x:.3f}, vy={self.setpoint.velocity.y:.3f}, vz={self.setpoint.velocity.z:.3f}")

            # Step firmware controller
            self._step_controller()

            # Get action
            new_action = self.PWM2RPM_SCALE * np.clip(np.array(self.pwms), self.MIN_PWM, self.MAX_PWM) + self.PWM2RPM_CONST

            if self.ACTION_DELAY:
                action = self.action_history[0]
                self.action_history = self.action_history[1:] + [new_action]
            else:
                action = new_action

            if self._error:
                action = np.zeros(4)
                if self.first_motor_killed_print:
                    print("Drone firmware error. Motors are killed.")
                    self.first_motor_killed_print = False

            self.action = action

        return obs, reward, terminated, truncated, info

    def _update_initial_state(self, obs):
        self.prev_vel = np.array([obs[10], obs[11], obs[12]])
        self.prev_rpy = np.array([obs[7], obs[8], obs[9]])

    def _apply_mpc_filter(self, pos, vel, rpy, obstacle_positions, obstacle_velocities):
        """Apply MPC-based planner to generate safe velocity setpoints.

        - The MPC sees the full position/velocity state of the drone.
        - The goal position for the MPC is taken from the current high-level
          position setpoint (what the outer trajectory wants).
        - The yaw-rate reference is passed through from the Crazyflie setpoint.
        """

        if self.drone is None or self.mpc_controller is None:
            return

        # Update internal drone state used by MPC
        self.drone.update_state(pos.tolist(), vel.tolist(), rpy.tolist(), self.firmware_dt)

        # Goal position for MPC: track the current high-level position setpoint
        p_goal = np.array([
            self.setpoint.position.x,
            self.setpoint.position.y,
            self.setpoint.position.z
        ], dtype=float)
        self.mpc_controller.set_goal_position(p_goal)

        # Nominal control: only yaw_rate is really used by the MPC controller,
        # translational part is kept for potential extensions.
        u_ref = np.array([
            self.setpoint.velocity.x,
            self.setpoint.velocity.y,
            self.setpoint.velocity.z,
            self.setpoint.attitudeRate.yaw / self.RAD_TO_DEG  # deg/s -> rad/s
        ], dtype=float).reshape(-1,)
        self.mpc_controller.set_reference_control(u_ref)

        # Normalize obstacle inputs
        if obstacle_positions is None:
            obstacle_positions = []
        if obstacle_velocities is None:
            obstacle_velocities = [[0.0, 0.0, 0.0] for _ in range(len(obstacle_positions))]

        # Setup and solve MPC QP
        self.mpc_controller.setup_QP(self.drone, obstacle_positions, obstacle_velocities)
        status, obj_val = self.mpc_controller.solve_QP(self.drone)

        u_safe = self.mpc_controller.get_optimal_control()  # [vx, vy, vz, yaw_rate]

        # Overwrite Crazyflie velocity setpoint with MPC-safe command
        self.setpoint.velocity.x = float(u_safe[0])
        self.setpoint.velocity.y = float(u_safe[1])
        self.setpoint.velocity.z = float(u_safe[2])
        self.setpoint.attitudeRate.yaw = float(u_safe[3]) * self.RAD_TO_DEG  # rad/s -> deg/s


    ##################################
    ########## Sensor Data ###########
    ##################################

    def _update_sensorData(self, timestamp, acc_vals, gyro_vals, baro_vals=[1013.25, 25]):
        """
        Axis3f acc;               // Gs
        Axis3f gyro;              // deg/s
        Axis3f mag;               // gauss
        baro_t baro;              // C, Pa
        #ifdef LOG_SEC_IMU
            Axis3f accSec;            // Gs
            Axis3f gyroSec;           // deg/s
        #endif
        uint64_t interruptTimestamp;   // microseconds
        """
        # Only gyro and acc are used in controller. Mag and baro used in state estimation (not yet supported)
        self._update_acc(*acc_vals)
        self._update_gyro(*gyro_vals)
        # self._update_baro(self.sensorData.baro, *baro_vals)

        self.sensorData.interruptTimestamp = timestamp
        self.sensorData_set = True

    def _update_gyro(self, x, y, z):
        self.sensorData.gyro.x = firm.lpf2pApply(self.gyrolpf[0], x)
        self.sensorData.gyro.y = firm.lpf2pApply(self.gyrolpf[1], y)
        self.sensorData.gyro.z = firm.lpf2pApply(self.gyrolpf[2], z)

    def _update_acc(self, x, y, z):
        self.sensorData.acc.x = firm.lpf2pApply(self.acclpf[0], x)
        self.sensorData.acc.y = firm.lpf2pApply(self.acclpf[1], y)
        self.sensorData.acc.z = firm.lpf2pApply(self.acclpf[2], z)

    def _update_baro(self, baro, pressure, temperature):
        """
        pressure: hPa
        temp: C
        asl = m
        """
        baro.pressure = pressure
        baro.temperature = temperature
        baro.asl = (((1015.7 / baro.pressure) ** 0.1902630958 - 1) * (25 + 273.15)) / 0.0065

    ##################################
    ######### State Update ###########
    ##################################

    def _update_state(self, timestamp, pos, vel, acc, rpy, quat=None):
        """
        attitude_t attitude;      // deg (legacy CF2 body coordinate system, where pitch is inverted)
        quaternion_t attitudeQuaternion;
        point_t position;         // m
        velocity_t velocity;      // m/s
        acc_t acc;                // Gs (but acc.z without considering gravity)
        """
        self._update_attitude_t(self.state.attitude, timestamp, *rpy)
        if self.CONTROLLER == 'mellinger':
            self._update_attitudeQuaternion(self.state.attitudeQuaternion, timestamp, *rpy)

        self._update_3D_vec(self.state.position, timestamp, *pos)
        self._update_3D_vec(self.state.velocity, timestamp, *vel)
        self._update_3D_vec(self.state.acc, timestamp, *acc)
        self.state_set = True

    def _update_3D_vec(self, point, timestamp, x, y, z):
        point.x = x
        point.y = y
        point.z = z
        point.timestamp = timestamp

    def _update_attitudeQuaternion(self, quaternion_t, timestamp, qx, qy, qz, qw=None):
        """Updates attitude quaternion.

        Note:
            if qw is present, input is taken as a quat. Else, as roll, pitch, and yaw in deg
        """
        quaternion_t.timestamp = timestamp

        if qw is None:  # passed roll, pitch, yaw
            qx, qy, qz, qw = _get_quaternion_from_euler(qx / self.RAD_TO_DEG, qy / self.RAD_TO_DEG, qz / self.RAD_TO_DEG)

        quaternion_t.x = qx
        quaternion_t.y = qy
        quaternion_t.z = qz
        quaternion_t.w = qw

    def _update_attitude_t(self, attitude_t, timestamp, roll, pitch, yaw):
        attitude_t.timestamp = timestamp
        attitude_t.roll = roll
        attitude_t.pitch = -pitch  # Legacy representation in CF firmware
        attitude_t.yaw = yaw

    ##################################
    ########### Controller ###########
    ##################################

    def _step_controller(self):
        if not (self.sensorData_set):
            print("WARNING: sensorData has not been updated since last controller call.")
        if not (self.state_set):
            print("WARNING: state has not been updated since last controller call.")
        self.sensorData_set = False
        self.state_set = False

        # Check for tumbling crazyflie
        if self.state.acc.z < -0.5:
            self.tumble_counter += 1
        else:
            self.tumble_counter = 0
        if self.tumble_counter >= 30:
            print('WARNING: CrazyFlie is tumbling. Killing motors to save propellers.')
            self.pwms = [0, 0, 0, 0]
            self.tick += 1
            self._error = True
            return

        # Determine tick based on time passed, allowing us to run pid slower than design
        cur_time = self.tick / self.firmware_freq
        if (cur_time - self.last_att_pid_call > 0.002) and (cur_time - self.last_pos_pid_call > 0.01):
            _tick = 0  # Runs position and attitude controller
            self.last_pos_pid_call = cur_time
            self.last_att_pid_call = cur_time
        elif (cur_time - self.last_att_pid_call > 0.002):
            self.last_att_pid_call = cur_time
            _tick = 2  # Runs attitude controller
        else:
            _tick = 1  # Runs neither controller

        # Step the chosen controller
        if self.CONTROLLER == 'pid':
            firm.controllerPid(self.control, self.setpoint, self.sensorData, self.state, _tick)
        elif self.CONTROLLER == 'mellinger':
            firm.controllerMellinger(self.control, self.setpoint, self.sensorData, self.state, _tick)

        # Get pwm values from control object
        self._powerDistribution(self.control)
        self.tick += 1

    def _updateSetpoint(self, timestep):
        if not self.full_state_cmd_override:
            firm.crtpCommanderHighLevelTellState(self.state)
            firm.crtpCommanderHighLevelUpdateTime(timestep)
            firm.crtpCommanderHighLevelGetSetpoint(self.setpoint, self.state)

    def _process_command_queue(self, sim_time):
        if len(self.command_queue) > 0:
            firm.crtpCommanderHighLevelStop()
            firm.crtpCommanderHighLevelUpdateTime(sim_time)
            command, args = self.command_queue.pop(0)
            getattr(self, command)(*args)

    def sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        """Adds a sendFullState command to command processing queue."""
        self.command_queue += [['_sendFullStateCmd', [pos, vel, acc, yaw, rpy_rate, timestep]]]

    def _sendFullStateCmd(self, pos, vel, acc, yaw, rpy_rate, timestep):
        self.setpoint.position.x = pos[0]
        self.setpoint.position.y = pos[1]
        self.setpoint.position.z = pos[2]
        self.setpoint.velocity.x = vel[0]
        self.setpoint.velocity.y = vel[1]
        self.setpoint.velocity.z = vel[2]
        self.setpoint.acceleration.x = acc[0]
        self.setpoint.acceleration.y = acc[1]
        self.setpoint.acceleration.z = acc[2]

        self.setpoint.attitudeRate.roll = rpy_rate[0] * self.RAD_TO_DEG
        self.setpoint.attitudeRate.pitch = rpy_rate[1] * self.RAD_TO_DEG
        self.setpoint.attitudeRate.yaw = rpy_rate[2] * self.RAD_TO_DEG

        quat = _get_quaternion_from_euler(0, 0, yaw)
        self.setpoint.attitudeQuaternion.x = quat[0]
        self.setpoint.attitudeQuaternion.y = quat[1]
        self.setpoint.attitudeQuaternion.z = quat[2]
        self.setpoint.attitudeQuaternion.w = quat[3]

        self.setpoint.mode.x = firm.modeAbs
        self.setpoint.mode.y = firm.modeAbs
        self.setpoint.mode.z = firm.modeAbs

        self.setpoint.mode.quat = firm.modeAbs
        self.setpoint.mode.roll = firm.modeDisable
        self.setpoint.mode.pitch = firm.modeDisable
        self.setpoint.mode.yaw = firm.modeDisable

        self.setpoint.timestamp = int(timestep * 1000)
        self.full_state_cmd_override = True

    # All the takeoff / land / goto methods are identical to your QP aviary,
    # just kept verbatim:

    def sendTakeoffCmd(self, height, duration):
        self.command_queue += [['_sendTakeoffCmd', [height, duration]]]

    def _sendTakeoffCmd(self, height, duration):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        self.takeoff_sent = True
        firm.crtpCommanderHighLevelTakeoff(height, duration)
        self.full_state_cmd_override = False

    def sendTakeoffYawCmd(self, height, duration, yaw):
        self.command_queue += [['_sendTakeoffYawCmd', [height, duration, yaw]]]

    def _sendTakeoffYawCmd(self, height, duration, yaw):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendTakeoffVelCmd(self, height, vel, relative):
        self.command_queue += [['_sendTakeoffVelCmd', [height, vel, relative]]]

    def _sendTakeoffVelCmd(self, height, vel, relative):
        print(f"INFO_{self.tick}: Takeoff command sent.")
        firm.crtpCommanderHighLevelTakeoffWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendLandCmd(self, height, duration):
        self.command_queue += [['_sendLandCmd', [height, duration]]]

    def _sendLandCmd(self, height, duration):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLand(height, duration)
        self.full_state_cmd_override = False

    def sendLandYawCmd(self, height, duration, yaw):
        self.command_queue += [['_sendLandYawCmd', [height, duration, yaw]]]

    def _sendLandYawCmd(self, height, duration, yaw):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandYaw(height, duration, yaw)
        self.full_state_cmd_override = False

    def sendLandVelCmd(self, height, vel, relative):
        self.command_queue += [['_sendLandVelCmd', [height, vel, relative]]]

    def _sendLandVelCmd(self, height, vel, relative):
        print(f"INFO_{self.tick}: Land command sent.")
        firm.crtpCommanderHighLevelLandWithVelocity(height, vel, relative)
        self.full_state_cmd_override = False

    def sendStopCmd(self):
        self.command_queue += [['_sendStopCmd', []]]

    def _sendStopCmd(self):
        print(f"INFO_{self.tick}: Stop command sent.")
        firm.crtpCommanderHighLevelStop()
        self.full_state_cmd_override = False

    def sendGotoCmd(self, pos, yaw, duration_s, relative):
        self.command_queue += [['_sendGotoCmd', [pos, yaw, duration_s, relative]]]

    def _sendGotoCmd(self, pos, yaw, duration_s, relative):
        print(f"INFO_{self.tick}: Go to command sent.")
        firm.crtpCommanderHighLevelGoTo(*pos, yaw, duration_s, relative)
        self.full_state_cmd_override = False

    def notifySetpointStop(self):
        self.command_queue += [['_notifySetpointStop', []]]

    def _notifySetpointStop(self):
        print(f"INFO_{self.tick}: Notify setpoint stop command sent.")
        firm.crtpCommanderHighLevelTellState(self.state)
        self.full_state_cmd_override = False

    ##################################
    ###### Hardware Functions ########
    ##################################

    BRUSHED = True
    SUPPLY_VOLTAGE = 3

    def _motorsGetPWM(self, thrust):
        if self.BRUSHED:
            thrust = thrust / 65536 * 60
            volts = -0.0006239 * thrust**2 + 0.088 * thrust
            percentage = min(1, volts / self.SUPPLY_VOLTAGE)
            ratio = percentage * self.MAX_PWM
            return ratio
        else:
            raise NotImplementedError("Emulator does not support the brushless motor configuration at this time.")

    def _limitThrust(self, val):
        if val > self.MAX_PWM:
            return self.MAX_PWM
        elif val < 0:
            return 0
        return val

    def _powerDistribution(self, control_t):
        motor_pwms = []
        if self.QUAD_FORMATION_X:
            r = control_t.roll / 2
            p = control_t.pitch / 2

            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - r + p + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - r - p - control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + r - p + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + r + p - control_t.yaw))]
        else:
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.pitch + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.roll - control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust - control_t.pitch + control_t.yaw))]
            motor_pwms += [self._motorsGetPWM(self._limitThrust(control_t.thrust + control_t.roll - control_t.yaw))]

        if self.MOTOR_SET_ENABLE:
            self.pwms = motor_pwms
        else:
            self.pwms = np.clip(motor_pwms, self.MIN_PWM).tolist()

    ##################################
    ##### Base Aviary Overrides ######
    ##################################

    def _actionSpace(self):
        """Returns the action space of the environment."""
        act_lower_bound = np.array([[0., 0., 0., 0.] for _ in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self):
        """Returns the observation space of the environment."""
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,
                                     -1., -1., -1., -1.,
                                     -np.pi, -np.pi, -np.pi,
                                     -np.inf, -np.inf, -np.inf,
                                     -np.inf, -np.inf, -np.inf,
                                     0., 0., 0., 0.]
                                    for _ in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf, np.inf, np.inf,
                                     1., 1., 1., 1.,
                                     np.pi, np.pi, np.pi,
                                     np.inf, np.inf, np.inf,
                                     np.inf, np.inf, np.inf,
                                     self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM]
                                    for _ in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self):
        """Returns the current observation of the environment."""
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    def _preprocessAction(self, action):
        """Pre-processes the action passed to `.step()` into motors' RPMs."""
        return action.reshape((1, 4))

    def _computeReward(self):
        """Dummy reward (not used)."""
        return -1

    def _computeTerminated(self):
        """Dummy termination flag (not used)."""
        return False

    def _computeTruncated(self):
        """Dummy truncated flag (not used)."""
        return False

    def _computeInfo(self):
        """Dummy info dict (not used)."""
        return {"answer": 42}


def _get_quaternion_from_euler(roll, pitch, yaw):
    """Convert an Euler angle to a quaternion.

    Args:
        roll (float): rotation around x-axis [rad].
        pitch (float): rotation around y-axis [rad].
        yaw (float): rotation around z-axis [rad].

    Returns
    -------
    list[float]
        Quaternion [x, y, z, w].
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]
