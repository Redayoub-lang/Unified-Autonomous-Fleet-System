import numpy as np
import csv
import time


class QuadcopterPhysicsSO3:
    """Rigid Body Newton-Euler Quadcopter Dynamics with SO(3) Control."""
    def __init__(self, drone_id=1, initial_pos=None):
        self.drone_id = drone_id
        
        # Industrial Physical Parameters
        self.m = 0.917            # Mass [kg]
        self.g = 9.81             # Gravity [m/s^2]
        self.L = 0.39             # Arm length [m]
        self.J = np.diag([0.022, 0.022, 0.043])  # Inertia tensor [kg*m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.kT = 2.4e-6          # Thrust coefficient
        self.b = 1.0e-4           # Drag coefficient

        # State Variables
        self.p = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        # Geometric Controller Gains
        self.Kp_pos = np.diag([3.5, 3.5, 6.5])
        self.Kd_pos = np.diag([3.0, 3.0, 4.5])
        self.Kr_att = np.diag([12.0, 12.0, 6.0])
        self.Kw_att = np.diag([2.2, 2.2, 1.4])

        self.max_tilt_rad = np.radians(12.0)  # Max 12 deg tilt

    def step(self, target_pos, target_vel, is_landing_phase=False, dt=0.01):
        # Ground Disarm Protection Logic
        if is_landing_phase and self.p[2] <= 0.02 and target_pos[2] <= 0.05:
            self.p[2] = 0.0
            self.v = np.zeros(3)
            self.omega = np.zeros(3)
            self.R = np.eye(3)
            return self.p.copy(), 0.0, 0.0

        # Position Controller
        e_p = target_pos - self.p
        e_v = target_vel - self.v
        a_des = self.Kp_pos @ e_p + self.Kd_pos @ e_v
        F_des = self.m * (a_des + np.array([0.0, 0.0, self.g]))

        # Tilt Clamping
        z_b_des = F_des / (np.linalg.norm(F_des) + 1e-6)
        max_xy = np.sin(self.max_tilt_rad)
        xy_norm = np.linalg.norm(z_b_des[:2])
        if xy_norm > max_xy:
            z_b_des[:2] = z_b_des[:2] * (max_xy / xy_norm)
            z_b_des[2] = np.sqrt(1.0 - max_xy**2)

        z_b = self.R[:, 2]
        F = max(0.0, np.dot(F_des, z_b))

        # SO(3) Attitude Control
        x_c = np.array([1.0, 0.0, 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_b_des /= (np.linalg.norm(y_b_des) + 1e-6)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack((x_b_des, y_b_des, z_b_des))

        e_R_matrix = 0.5 * (R_d.T @ self.R - self.R.T @ R_d)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        
        M = -self.Kr_att @ e_R - self.Kw_att @ self.omega + np.cross(self.omega, self.J @ self.omega)

        # Rigid Body Dynamics Integration
        e3 = np.array([0.0, 0.0, 1.0])
        v_dot = (self.R @ (F * e3) - self.m * self.g * e3) / self.m
        self.v += v_dot * dt
        self.p += self.v * dt

        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])

        omega_dot = self.J_inv @ (M - np.cross(self.omega, self.J @ self.omega))
        self.omega += omega_dot * dt

        omega_hat = np.array([
            [0, -self.omega[2], self.omega[1]],
            [self.omega[2], 0, -self.omega[0]],
            [-self.omega[1], self.omega[0], 0]
        ])
        self.R += self.R @ omega_hat * dt
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        # Stable Rotor RPM
        f_per_motor = np.clip(F / 4.0, 0.0, 8.0)
        rotor_rpm = np.sqrt(f_per_motor / self.kT)
        tilt_deg = np.arccos(np.clip(self.R[2, 2], -1.0, 1.0)) * (180.0 / np.pi)

        return self.p.copy(), tilt_deg, rotor_rpm


def run_physics_headless_simulation():
    print("[INFO] Executing Quadcopter Physics Telemetry Generation...")
    drone = QuadcopterPhysicsSO3()
    dt = 0.01
    times = np.arange(0, 14.0, dt)

    with open("fleet_quadcopter_physics_telemetry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time_s", "Pos_X", "Pos_Y", "Pos_Z", "Tilt_deg", "Rotor_RPM"])

        for t in times:
            if t < 2.5:
                ref = np.array([0.0, 0.0, 1.0])
            elif t < 5.5:
                ref = np.array([0.5, 0.0, 1.0])
            elif t < 8.5:
                ref = np.array([-0.91, 0.0, 1.0])
            elif t < 11.5:
                ref = np.array([0.0, 0.0, 1.0])
            else:
                ref = np.array([0.0, 0.0, 0.0])

            pos, tilt, rpm = drone.step(ref, np.zeros(3), is_landing_phase=(t >= 11.5), dt=dt)
            writer.writerow([round(t, 2), round(pos[0], 3), round(pos[1], 3), round(pos[2], 3), round(tilt, 2), int(rpm)])

    print("[INFO] Telemetry output saved to 'fleet_quadcopter_physics_telemetry.csv'.")


if __name__ == "__main__":
    run_physics_headless_simulation()