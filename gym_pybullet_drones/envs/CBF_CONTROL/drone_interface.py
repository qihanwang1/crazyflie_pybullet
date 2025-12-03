import numpy as np
import threading
import time
import sys
import os
import inspect
import copy
import warnings

# Add project root to path (go up 2 directories from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
# from cflib.utils.multiranger import Multiranger  # Not using - has timing issues


from mpc_controller.controllers.QP_controller_drone import QP_Controller_Drone
from mpc_controller.bots.drone import Drone
from mpc_controller.simple_line_trajectory import get_forward_trajectory, get_hover_trajectory, get_path_tracking_trajectory


class Crazyflie_Interface:
    """
    Hardware interface for Crazyflie drones using QP-CBF controller
    """
    def __init__(self, uri, drone_cfg_pth, controller_params):
        """
        Initialize the Crazyflie interface.

        Args:
            uri (str): URI of the CrazyRadio URI.
            drone_cfg_pth (str): Path to the drone configuration file.
            controller_params (dict): Parameters for the QP-CBF controller.
        """
        self.uri = uri
        self.drone_cfg_pth = drone_cfg_pth
        self.controller_params = controller_params

        # Initialize driver
        cflib.crtp.init_drivers()

        # CrazyRadio connection objects (initialized on connect)
        self.scf = None
        self.cf = None
        self.mc = None  # MotionCommander instance

        # Multiranger sensor readings (populated by logging callback)
        self.multiranger_data = {
            'front': None,
            'back': None,
            'left': None,
            'right': None,
            'up': None,
            'down': None
        }

        # Drone model and controller
        self.drone = Drone.from_JSON(self.drone_cfg_pth)
        self.controller = QP_Controller_Drone(**controller_params)

        self.lock = threading.Lock()
        self.state_buffer = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'wx': 0.0, 'wy': 0.0, 'wz': 0.0,  # Angular velocities from gyro
            'timestamp': 0,
            'updated': False
        }

        self.dt = 0.01
        self.last_command_time = 0

        self.obs_pos = {}
        self.obs_vel = {}
        self.prev_pos = {}

        # Velocity threshold for static vs dynamic obstacle classification (m/s)
        # If estimated velocity is below this threshold, treat as static obstacle (vel = 0)
        # Typical values:
        #   0.1 m/s = very sensitive (may still have noise)
        #   0.3 m/s = moderate threshold (recommended)
        #   0.5 m/s = conservative (only fast-moving objects considered dynamic)
        #   1.0 m/s = very conservative (rejects most noise)
        self.VELOCITY_THRESHOLD = 9999.0  # m/s

        # Maximum reasonable obstacle velocity (m/s) - velocities above this are likely noise
        # and will be clamped to static
        self.MAX_OBSTACLE_VELOCITY = 0.1  # m/s (walking speed is ~1.5 m/s)

    def sync_states(self, dt):
        with self.lock:
            pos = [self.state_buffer['x'], self.state_buffer['y'], self.state_buffer['z']]
            vel = [self.state_buffer['vx'], self.state_buffer['vy'], self.state_buffer['vz']]
            rot = [self.state_buffer['roll'], self.state_buffer['pitch'], self.state_buffer['yaw']]
            w = [self.state_buffer['wx'], self.state_buffer['wy'], self.state_buffer['wz']]

        self.drone.update_state(pos, vel, rot, dt, w)



    def connect(self):
        """
        Connect to the Crazyflie and set up logging.
        """
        print("[INFO] Connecting to Crazyflie at", self.uri, "...")
        cflib.crtp.init_drivers()

    # Establish connection
        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache'))
        self.scf.open_link()
        self.cf = self.scf.cf
        print("[INFO] Connection established successfully.")

    # --- LogConfig 1: position & velocity ---
        log_cfg1 = LogConfig(name='posvel', period_in_ms=10)
        log_cfg1.add_variable('stateEstimate.x', 'float')
        log_cfg1.add_variable('stateEstimate.y', 'float')
        log_cfg1.add_variable('stateEstimate.z', 'float')
        log_cfg1.add_variable('stateEstimate.vx', 'float')
        log_cfg1.add_variable('stateEstimate.vy', 'float')
        log_cfg1.add_variable('stateEstimate.vz', 'float')

        self.cf.log.add_config(log_cfg1)
        log_cfg1.data_received_cb.add_callback(self._log_data_callback)
        log_cfg1.start()

    # --- LogConfig 2: attitude ---
        log_cfg2 = LogConfig(name='attitude', period_in_ms=10)
        log_cfg2.add_variable('stateEstimate.roll', 'float')
        log_cfg2.add_variable('stateEstimate.pitch', 'float')
        log_cfg2.add_variable('stateEstimate.yaw', 'float')
        log_cfg2.add_variable('gyro.x', 'float')
        log_cfg2.add_variable('gyro.y', 'float')
        log_cfg2.add_variable('gyro.z', 'float')

        self.cf.log.add_config(log_cfg2)
        log_cfg2.data_received_cb.add_callback(self._log_data_callback)
        log_cfg2.start()

        # Wait for parameters to sync
        print("[INFO] Waiting for parameter sync...")
        time.sleep(1.0)

        # --- LogConfig 3: Multiranger (using direct logging instead of helper class) ---
        print("[INFO] Setting up Multiranger logging...")
        try:
            log_cfg3 = LogConfig(name='range', period_in_ms=100)
            log_cfg3.add_variable('range.front', 'uint16_t')
            log_cfg3.add_variable('range.back', 'uint16_t')
            log_cfg3.add_variable('range.left', 'uint16_t')
            log_cfg3.add_variable('range.right', 'uint16_t')
            log_cfg3.add_variable('range.up', 'uint16_t')
            log_cfg3.add_variable('range.zrange', 'uint16_t')

            self.cf.log.add_config(log_cfg3)
            log_cfg3.data_received_cb.add_callback(self._multiranger_callback)

            print("[INFO] Starting Multiranger logging...")
            log_cfg3.start()
            print("[INFO] Multiranger logging started, waiting for data...")

            # Wait for sensors to start providing data
            timeout = 3.0
            start_wait = time.time()
            sensors_ready = False

            while (time.time() - start_wait) < timeout:
                time.sleep(0.1)

                with self.lock:
                    # Check if ANY sensor is providing valid data
                    if any(v is not None for v in self.multiranger_data.values()):
                        sensors_ready = True
                        break

            if not sensors_ready:
                print("[WARNING] Multiranger sensors not providing data after 3 seconds!")
                print("[WARNING] Continuing anyway - obstacle avoidance may not work.")
            else:
                elapsed = time.time() - start_wait
                print(f"[INFO] Multiranger ready! (took {elapsed:.2f}s)")
                with self.lock:
                    print(f"[INFO] Initial readings (mm) - front: {self.multiranger_data['front']}, "
                          f"back: {self.multiranger_data['back']}, left: {self.multiranger_data['left']}, "
                          f"right: {self.multiranger_data['right']}, up: {self.multiranger_data['up']}, "
                          f"down: {self.multiranger_data['down']}")

        except KeyError as e:
            print(f"[ERROR] Multiranger log variable not found: {e}")
            print("[WARNING] Multiranger deck not detected by firmware - continuing without it.")
        except Exception as e:
            print(f"[ERROR] Failed to start Multiranger logging: {e}")
            import traceback
            traceback.print_exc()
            print("[WARNING] Continuing without Multiranger.")

        # Initialize MotionCommander for velocity control
        print("[INFO] Initializing MotionCommander and taking off...")
        self.mc = MotionCommander(self.scf, default_height=0.3)
        self.mc.take_off()
        print("[INFO] Takeoff complete.")

    def disconnect(self):
        """
        Safely disconnect from Crazyflie and land the drone.
        """
        print("[INFO] Landing and disconnecting...")
        if self.mc is not None:
            try:
                self.mc.land()
                print("[INFO] Landed successfully.")
            except Exception as e:
                print(f"[ERROR] Landing failed: {e}")
            self.mc = None

        if self.scf is not None:
            self.scf.close_link()
            print("[INFO] Connection closed.")
            self.scf = None
            self.cf = None

    def _log_data_callback(self, timestamp, data, logconf):
        with self.lock:
        # Use .get() with a default value if the key doesn't exist
            self.state_buffer['x'] = data.get('stateEstimate.x', 0.0)
            self.state_buffer['y'] = data.get('stateEstimate.y', 0.0)
            self.state_buffer['z'] = data.get('stateEstimate.z', 0.0)
            self.state_buffer['vx'] = data.get('stateEstimate.vx', 0.0)
            self.state_buffer['vy'] = data.get('stateEstimate.vy', 0.0)
            self.state_buffer['vz'] = data.get('stateEstimate.vz', 0.0)
            self.state_buffer['roll'] = np.deg2rad(data.get('stateEstimate.roll', 0.0))
            self.state_buffer['pitch'] = np.deg2rad(data.get('stateEstimate.pitch', 0.0))
            self.state_buffer['yaw'] = np.deg2rad(data.get('stateEstimate.yaw', 0.0))
            self.state_buffer['wx'] = np.deg2rad(data.get('gyro.x', 0.0))
            self.state_buffer['wy'] = np.deg2rad(data.get('gyro.y', 0.0))
            self.state_buffer['wz'] = np.deg2rad(data.get('gyro.z', 0.0))

    def _multiranger_callback(self, timestamp, data, logconf):
        """Callback for Multiranger sensor data."""
        with self.lock:
            # Convert from mm to meters and store
            self.multiranger_data['front'] = data.get('range.front', None)
            self.multiranger_data['back'] = data.get('range.back', None)
            self.multiranger_data['left'] = data.get('range.left', None)
            self.multiranger_data['right'] = data.get('range.right', None)
            self.multiranger_data['up'] = data.get('range.up', None)
            self.multiranger_data['down'] = data.get('range.zrange', None)


    def control_loop(self, u_ref, obstacle_positions=None, obstacle_velocities=None):
        """
        Run the CBF-QP control loop with multiple obstacles.

        Parameters:
            u_ref: reference velocity control input [vx, vy, vz, yaw_rate]
            obstacle_positions: optional list of [x, y, z] positions or dict {direction: [x,y,z]}
                               If None, uses self.obs_pos
            obstacle_velocities: optional list of [vx, vy, vz] velocities or dict {direction: [vx,vy,vz]}
                                If None, uses self.obs_vel

        Returns:
            vel_cmd: optimal velocity commands [vx, vy, vz, yaw_rate] from QP-CBF
        """
        # Use stored obstacle data if not provided
        if obstacle_positions is None:
            obstacle_positions = self.obs_pos
        if obstacle_velocities is None:
            obstacle_velocities = self.obs_vel

        # Convert dict format to list format if needed
        if isinstance(obstacle_positions, dict):
            if len(obstacle_positions) > 0:
                # Use explicit key ordering to ensure positions and velocities match
                keys = list(obstacle_positions.keys())
                obs_pos_list = [np.array(obstacle_positions[k]).tolist() if isinstance(obstacle_positions[k], np.ndarray) else obstacle_positions[k]
                               for k in keys]
                obs_vel_list = [np.array(obstacle_velocities.get(k, [0.0, 0.0, 0.0])).tolist() if isinstance(obstacle_velocities.get(k, [0.0, 0.0, 0.0]), np.ndarray) else obstacle_velocities.get(k, [0.0, 0.0, 0.0])
                               for k in keys]
            else:
                # No obstacles detected - empty list
                obs_pos_list = []
                obs_vel_list = []
        elif isinstance(obstacle_positions, list) and len(obstacle_positions) > 0:
            # Check if it's a list of lists or a single position
            if isinstance(obstacle_positions[0], (list, np.ndarray)):
                obs_pos_list = obstacle_positions
                obs_vel_list = obstacle_velocities
            else:
                # Single obstacle [x, y, z] format
                obs_pos_list = [obstacle_positions]
                obs_vel_list = [obstacle_velocities]
        else:
            # Empty or invalid input - empty list
            obs_pos_list = []
            obs_vel_list = []

        self.controller.set_reference_control(u_ref)
        self.controller.setup_QP(self.drone, obs_pos_list, obs_vel_list)
        state_h, value_h = self.controller.solve_QP(self.drone)
        vel_cmd = self.controller.get_optimal_control()  # Returns u_star from QP

        return vel_cmd


    def send_control(self, vel_array):
        """
        Send velocity control commands to Crazyflie using MotionCommander.

        Parameters:
            vel_array: numpy array of shape (4,) containing [vx, vy, vz, yaw_rate]
                      This comes directly from QP controller's u_star in WORLD FRAME
        """
        if self.mc is None:
            print("[WARN] MotionCommander not initialized. Cannot send control.")
            return

        # Extract velocity commands from QP output (world frame)
        vx_world = float(vel_array[0])
        vy_world = float(vel_array[1])
        vz_world = float(vel_array[2])
        yaw_rate_cmd = float(vel_array[3])  # rad/s from QP controller

        # Get current yaw angle for frame transformation
        with self.lock:
            yaw = self.state_buffer['yaw']

        # Transform from world frame to body frame
        # World frame: [vx_world, vy_world] -> Body frame: [vx_body, vy_body]
        # Rotation: [vx_body]   [cos(yaw)  sin(yaw)] [vx_world]
        #           [vy_body] = [-sin(yaw) cos(yaw)] [vy_world]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        vx_body = cos_yaw * vx_world + sin_yaw * vy_world
        vy_body = -sin_yaw * vx_world + cos_yaw * vy_world
        vz_body = vz_world  # Z-axis is same in both frames

        # Convert yaw rate from rad/s to deg/s for MotionCommander
        yaw_rate_deg = np.rad2deg(yaw_rate_cmd)

        # Send velocity commands using MotionCommander API
        # Note: MotionCommander expects velocities in m/s and yaw rate in deg/s
        self.mc.start_linear_motion(vx_body, vy_body, vz_body, yaw_rate_deg)


    def get_obs_pos(self):
        """
        Uses the multiranger to get obstacle position (center of object assumption for large safety margin)
        Returns:
            dict: {direction: [x, y, z]} positions of detected obstacles.
        """
        if self.scf is None:
            print("Crazyflie not connected. Cannot get obstacle position.")
            return {}

        # Get distances from multiranger data (in mm, need to convert to meters)
        with self.lock:
            distances = {
                'front': self.multiranger_data['front'] / 1000.0 if self.multiranger_data['front'] is not None else None,
                'back':  self.multiranger_data['back'] / 1000.0 if self.multiranger_data['back'] is not None else None,
                'left':  self.multiranger_data['left'] / 1000.0 if self.multiranger_data['left'] is not None else None,
                'right': self.multiranger_data['right'] / 1000.0 if self.multiranger_data['right'] is not None else None,
                'up':    self.multiranger_data['up'] / 1000.0 if self.multiranger_data['up'] is not None else None,
                'down':  self.multiranger_data['down'] / 1000.0 if self.multiranger_data['down'] is not None else None
            }
            # Current state
            x, y, z = self.state_buffer['x'], self.state_buffer['y'], self.state_buffer['z']
            yaw = self.state_buffer['yaw']

        # Rotation matrix for body-to-world frame transformation
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]])

        # Direction vectors in body frame
        directions = {
            'front': np.array([1, 0, 0]),
            'back':  np.array([-1, 0, 0]),
            'left':  np.array([0, 1, 0]),
            'right': np.array([0, -1, 0]),
            'up':    np.array([0, 0, 1]),
            'down':  np.array([0, 0, -1])
        }

        self.obs_pos = {}

        for key, d in distances.items():
            # Skip 'down' and 'up' sensors - they just detect ground/ceiling and cause false obstacles
            # We only care about horizontal obstacles (front, back, left, right)
            if key in ['down', 'up']:
                continue

            if d is not None and 0.03 < d < 0.4:   # ignore noise (< 3cm) and far obstacles (> 0.4m)
                self.obs_pos[key] = np.array([x, y, z]) + Rz @ (d * directions[key])

        return self.obs_pos

    def get_obs_vel(self, dt):
        """
        Finite difference to estimate obstacle velocity using the obstacle position.
        Classifies obstacles as static or dynamic based on velocity threshold.

        IMPORTANT: This computes the obstacle's velocity in the WORLD FRAME by accounting
        for the drone's own motion. When the drone moves, static obstacles appear to move
        in the world frame, so we subtract the drone's velocity to get the true obstacle velocity.

        This prevents false dynamic obstacle detection when:
        - The sensor detects a different object at a similar distance
        - Measurement noise creates artificial velocity
        - The same obstacle ID switches between different physical objects
        - The drone's own motion makes static obstacles appear dynamic

        Parameters:
            dt: Time step for finite difference calculation

        Returns:
            dict: Dictionary of obstacle velocities {direction: [vx, vy, vz]}
                  Velocities below VELOCITY_THRESHOLD are set to [0, 0, 0] (static)
        """
        new_obs_vel = {}

        # Get drone's current velocity in world frame
        drone_vel = np.array([self.drone.x_dot, self.drone.y_dot, self.drone.z_dot])

        for key, curr_pos in self.obs_pos.items():
            if key in self.prev_pos:
                # Compute apparent velocity in world frame (includes drone motion)
                apparent_vel = (curr_pos - self.prev_pos[key]) / dt

                # Subtract drone's velocity to get obstacle's true velocity
                # If obstacle is static, its velocity should be zero in world frame
                # But it appears to move opposite to drone's motion, so we add drone_vel back
                true_obs_vel = apparent_vel - drone_vel

                vel_magnitude = np.linalg.norm(true_obs_vel)

                # Sanity check: if velocity is unreasonably high, treat as static
                # This handles object switching (sensor seeing different objects at similar distance)
                if vel_magnitude > self.MAX_OBSTACLE_VELOCITY:
                    # Unrealistic velocity - likely sensor noise or object switching
                    new_obs_vel[key] = np.zeros(3)
                # Classify as static or dynamic based on threshold
                elif vel_magnitude < self.VELOCITY_THRESHOLD:
                    # Static obstacle: set velocity to zero
                    new_obs_vel[key] = np.zeros(3)
                else:
                    # Dynamic obstacle: use computed velocity
                    new_obs_vel[key] = true_obs_vel
            else:
                # First detection - assume static until we have velocity data
                new_obs_vel[key] = np.zeros(3)

        self.obs_vel = new_obs_vel
        self.prev_pos = {k: v.copy() for k, v in self.obs_pos.items()}
        return self.obs_vel


def main():
    # Filter out Crazyflie firmware deprecation warnings
    warnings.filterwarnings('ignore', message='Using legacy TYPE_HOVER_LEGACY')

    uri = "radio://0/80/2M"  # Your Crazyflie URI
    # Build path relative to project root
    drone_cfg_path = os.path.join(project_root, "mpc_controller", "bots", "bot_config", "drone1.json")

    # Check config file
    if not os.path.isfile(drone_cfg_path):
        raise FileNotFoundError(f"Drone config not found at {drone_cfg_path}")

    # Example controller parameters
    controller_params = {
        "gamma": 1.5,
        "obs_radius": 0.15
    }

    # Filter params to match QP_Controller_Drone constructor
    sig = inspect.signature(QP_Controller_Drone.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in controller_params.items() if k in valid_keys}

    # Initialize interface
    interface = Crazyflie_Interface(uri, drone_cfg_path, filtered_params)

    # Connect to Crazyflie
    interface.connect()

    # Initialize loop parameters
    dt = 0.01

    print("[INFO] Starting control loop with QP-CBF obstacle avoidance.")
    print("[INFO] Press Ctrl+C to stop at any time.")

    try:
        # ===== HOVER AFTER TAKEOFF =====
        # Let the drone stabilize after takeoff by actively hovering
        stabilization_time = 4.0  # seconds
        print(f"[INFO] Hovering for {stabilization_time}s to stabilize after takeoff...")
        print(time.time())
        hover_traj = get_hover_trajectory()
        start_stabilization = time.time()

        while (time.time() - start_stabilization) < stabilization_time:
            interface.sync_states(dt)
            interface.get_obs_pos()
            interface.get_obs_vel(dt)

            u_ref_hover = hover_traj.get_reference_control()
            vel_cmd_hover = interface.control_loop(u_ref_hover)
            interface.send_control(vel_cmd_hover)

            time.sleep(dt)

        print("[INFO] Drone stabilized. Starting trajectory control.")

        # ===== TRAJECTORY SELECTION =====
        # Choose trajectory type:
        # Option 1: Simple open-loop (no return to path)
        # trajectory = get_forward_trajectory(speed=0.2)

        # Option 2: Path-tracking with feedback (returns to path after obstacles)
        trajectory = get_path_tracking_trajectory(
            forward_speed=0.2,  # Forward velocity (m/s)
            target_y=0.0,       # Stay on centerline y=0
            target_z=0.3,       # Maintain takeoff altitude (MotionCommander default is ~0.3m)
            k_p=0.5             # Position feedback gain (0.3=gentle, 1.0=aggressive)
        )
        print("[INFO] Using path-tracking trajectory with position feedback")
        print(f"[INFO] Target path: y={trajectory.target_y}m, z={trajectory.target_z}m, k_p={trajectory.k_p}")

        step = 0
        start_time = time.time()
        flight_duration = 10.0  # seconds
        while (time.time() - start_time) < flight_duration:
            # Update drone state from logs
            interface.sync_states(dt)

            # Get obstacle positions and velocities from multiranger sensors
            interface.get_obs_pos()      # Updates self.obs_pos
            interface.get_obs_vel(dt)    # Updates self.obs_vel

            # Get current drone position for trajectory feedback
            current_pos = [interface.drone.x, interface.drone.y, interface.drone.z]

            # Get reference control from trajectory (with position feedback)
            u_ref = trajectory.get_reference_control(current_pos)

            # Run CBF-QP control loop
            vel_cmd = interface.control_loop(u_ref)

            # Print status every second
            if step % 100 == 0:
                num_obs = len(interface.obs_pos)
                # Show if command was modified
                u_ref_flat = u_ref.flatten()
                cmd_modified = not np.allclose(vel_cmd[:3], u_ref_flat[:3], atol=0.01)
                status = "CBF ACTIVE" if cmd_modified else "nominal"

                # Check if multiranger is working
                with interface.lock:
                    mr_working = any(v is not None for v in interface.multiranger_data.values())

                mr_status = "[WARNING] MULTIRANGER NOT WORKING" if not mr_working else "sensors OK"

                # Calculate path deviation
                y_error = interface.drone.y - trajectory.target_y
                z_error = interface.drone.z - trajectory.target_z

                print(f"[INFO] {mr_status} | Obstacles: {num_obs} | u_cmd: [{vel_cmd[0]:.2f}, {vel_cmd[1]:.2f}, {vel_cmd[2]:.2f}] | {status}")
                print(f"       Pos: [{interface.drone.x:.2f}, {interface.drone.y:.2f}, {interface.drone.z:.2f}] | Path error: y={y_error:.3f}m, z={z_error:.3f}m")

                # Print obstacle details if any detected
                if num_obs > 0:
                    for direction, pos in interface.obs_pos.items():
                        dist = np.linalg.norm(pos - np.array([interface.drone.x, interface.drone.y, interface.drone.z]))
                        vel = interface.obs_vel.get(direction, np.zeros(3))
                        vel_mag = np.linalg.norm(vel)
                        obs_type = "DYNAMIC" if vel_mag >= interface.VELOCITY_THRESHOLD else "STATIC"
                        print(f"      Obs {direction} [{obs_type}]: dist={dist:.2f}m, vel_mag={vel_mag:.3f}m/s")

            # Send velocity commands to Crazyflie
            interface.send_control(vel_cmd)

            # Wait for next iteration
            time.sleep(dt)
            step += 1
    except KeyboardInterrupt:
        print("\n[INFO] Control loop stopped by user.")
    finally:
        # Always disconnect safely
        interface.disconnect()

if __name__ == '__main__':
    main()