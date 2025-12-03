"""
Drift Analysis Tool for CBF-Controlled Crazyflie

This script runs a short hover test and collects detailed diagnostics to help
identify the cause of drift.
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_pybullet_drones.envs.CFCBFAviary import CFCBFAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def analyze_drift():
    """Run hover test with detailed diagnostics."""
    
    # Configuration
    drone_cfg_path = 'gym_pybullet_drones/envs/CBF_CONTROL/drone1.json'
    cbf_params = {'gamma': 1.0, 'obs_radius': 0.1}
    initial_xyz = np.array([[0.0, 0.0, 1.0]])
    initial_rpy = np.array([[0.0, 0.0, 0.0]])
    
    # Create environment
    print("="*70)
    print("DRIFT ANALYSIS TEST")
    print("="*70)
    print("\nInitializing environment...")
    
    env = CFCBFAviary(
        drone_model=DroneModel.CF2X,
        initial_xyzs=initial_xyz,
        initial_rpys=initial_rpy,
        physics=Physics.PYB,
        pyb_freq=500,
        ctrl_freq=25,
        gui=True,
        user_debug_gui=False,
        drone_config_path=drone_cfg_path,
        cbf_params=cbf_params,
        verbose=False
    )
    
    obs, info = env.reset()
    
    # Test parameters
    num_steps = 125  # 5 seconds at 25Hz
    dt = 1 / env.CTRL_FREQ
    
    # Obstacle far away (no avoidance)
    obstacle_pos = np.array([999.0, 999.0, 999.0])
    obstacle_vel = np.array([0.0, 0.0, 0.0])
    
    # Hover reference
    hover_thrust = env.drone.m * env.drone.g / 4
    u_ref = np.array([[hover_thrust], [hover_thrust], [hover_thrust], [hover_thrust]])
    
    # Data collection
    diagnostics_log = []
    
    print("\nRunning 5-second hover test...\n")
    print("Time | Pos Error | Vel Mag | T/W Ratio | RPM Imbal | Attitude (deg)")
    print("-" * 80)
    
    for i in range(num_steps):
        # Step simulation
        obs, reward, terminated, truncated, info = env.step_with_cbf(
            obstacle_pos, obstacle_vel, u_ref
        )
        
        # Get diagnostics
        diag = env.get_control_diagnostics()
        diag['time'] = i * dt
        diagnostics_log.append(diag)
        
        # Print every 0.5 seconds
        if i % 12 == 0:
            print(f"{diag['time']:4.1f}s | "
                  f"{diag['position_error_m']:8.4f}m | "
                  f"{diag['velocity_mag_ms']:7.4f}m/s | "
                  f"{diag['thrust_to_weight_ratio']:9.4f} | "
                  f"{diag['rpm_imbalance_pct']:8.2f}% | "
                  f"R:{diag['roll_deg']:5.1f} P:{diag['pitch_deg']:5.1f} Y:{diag['yaw_deg']:5.1f}")
    
    env.close()
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    # Convert to arrays for analysis
    pos_errors = np.array([d['position_error_m'] for d in diagnostics_log])
    vel_mags = np.array([d['velocity_mag_ms'] for d in diagnostics_log])
    thrust_ratios = np.array([d['thrust_to_weight_ratio'] for d in diagnostics_log])
    rpm_imbalances = np.array([d['rpm_imbalance_pct'] for d in diagnostics_log])
    rolls = np.array([d['roll_deg'] for d in diagnostics_log])
    pitches = np.array([d['pitch_deg'] for d in diagnostics_log])
    yaws = np.array([d['yaw_deg'] for d in diagnostics_log])
    
    print("\n1. POSITION DRIFT:")
    print(f"   Initial position error: {pos_errors[0]:.4f} m")
    print(f"   Final position error:   {pos_errors[-1]:.4f} m")
    print(f"   Max drift:              {np.max(pos_errors):.4f} m")
    print(f"   Average drift:          {np.mean(pos_errors):.4f} m")
    print(f"   Drift rate:             {(pos_errors[-1] - pos_errors[0]) / (num_steps * dt):.4f} m/s")
    
    print("\n2. VELOCITY:")
    print(f"   Average velocity:       {np.mean(vel_mags):.4f} m/s")
    print(f"   Max velocity:           {np.max(vel_mags):.4f} m/s")
    print(f"   Final velocity:         {vel_mags[-1]:.4f} m/s")
    
    print("\n3. THRUST-TO-WEIGHT RATIO:")
    print(f"   Average T/W:            {np.mean(thrust_ratios):.4f}")
    print(f"   Std dev T/W:            {np.std(thrust_ratios):.4f}")
    print(f"   Target T/W:             1.0000 (for hover)")
    if np.abs(np.mean(thrust_ratios) - 1.0) > 0.01:
        print(f"   ⚠️  WARNING: T/W ratio off by {(np.mean(thrust_ratios) - 1.0)*100:.2f}%")
    
    print("\n4. RPM BALANCE:")
    print(f"   Average imbalance:      {np.mean(rpm_imbalances):.2f}%")
    print(f"   Max imbalance:          {np.max(rpm_imbalances):.2f}%")
    if np.mean(rpm_imbalances) > 1.0:
        print(f"   ⚠️  WARNING: RPM imbalance >1% suggests asymmetric thrust")
    
    print("\n5. ATTITUDE:")
    print(f"   Roll:   mean={np.mean(rolls):5.2f}°, std={np.std(rolls):5.2f}°, max={np.max(np.abs(rolls)):5.2f}°")
    print(f"   Pitch:  mean={np.mean(pitches):5.2f}°, std={np.std(pitches):5.2f}°, max={np.max(np.abs(pitches)):5.2f}°")
    print(f"   Yaw:    mean={np.mean(yaws):5.2f}°, std={np.std(yaws):5.2f}°, max={np.max(np.abs(yaws)):5.2f}°")
    
    if np.max(np.abs(rolls)) > 5 or np.max(np.abs(pitches)) > 5:
        print(f"   ⚠️  WARNING: Large attitude angles suggest instability")
    
    print("\n6. POTENTIAL DRIFT CAUSES:")
    causes = []
    
    if np.abs(np.mean(thrust_ratios) - 1.0) > 0.05:
        causes.append(f"   • Thrust/weight mismatch ({(np.mean(thrust_ratios) - 1.0)*100:+.1f}%) → Check u_ref calculation")
    
    if np.mean(rpm_imbalances) > 2.0:
        causes.append(f"   • High RPM imbalance ({np.mean(rpm_imbalances):.1f}%) → Check CBF control output")
    
    if np.mean(vel_mags) > 0.1:
        causes.append(f"   • Persistent velocity ({np.mean(vel_mags):.3f} m/s) → Check if controller is rejecting disturbances")
    
    if (pos_errors[-1] - pos_errors[0]) / (num_steps * dt) > 0.05:
        causes.append(f"   • Growing position error → Controller not tracking setpoint")
    
    if np.mean(np.abs(rolls)) > 2 or np.mean(np.abs(pitches)) > 2:
        causes.append(f"   • Non-zero mean attitude → Check initial trim or wind disturbance")
    
    if len(causes) == 0:
        print("   ✓ No obvious issues detected. Drift may be due to:")
        print("     - Numerical integration errors (expected <1cm over 5s)")
        print("     - Small unmodeled disturbances")
        print("     - PyBullet simulation tolerances")
    else:
        for cause in causes:
            print(cause)
    
    print("\n7. RECOMMENDED ACTIONS:")
    
    if np.abs(np.mean(thrust_ratios) - 1.0) > 0.05:
        print("   1. Verify u_ref calculation: m * g / 4 per motor")
        print("      Current: m={:.4f} kg, g=9.81 m/s²".format(env.drone.m))
        print("      Expected thrust per motor: {:.6f} N".format(env.drone.m * 9.81 / 4))
    
    if np.mean(rpm_imbalances) > 2.0:
        print("   2. Check CBF-QP solver output for asymmetry")
        print("      Log individual motor thrusts to identify pattern")
    
    if vel_mags[-1] > 0.05:
        print("   3. Add velocity feedback to u_ref:")
        print("      u_ref += K_d * (vel_desired - vel_current)")
    
    print("\n" + "="*70)
    print("Analysis complete. Check results above for drift diagnosis.")
    print("="*70)

if __name__ == '__main__':
    analyze_drift()
