import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class IndustrialQuadcopterSO3:
    """Rigid Body Quadcopter Simulator with Tilt Clamping & Ground Disarm Logic."""
    def __init__(self):
        self.m = 0.917            # Mass [kg]
        self.g = 9.81             # Gravity [m/s^2]
        self.L = 0.39             # Arm length [m]
        self.J = np.diag([0.022, 0.022, 0.043])  # Inertia tensor [kg*m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.kT = 2.4e-6          # Thrust coefficient
        self.b = 1.0e-4           # Drag coefficient

        self.p = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        self.Kp_pos = np.diag([3.5, 3.5, 6.5])
        self.Kd_pos = np.diag([3.0, 3.0, 4.5])
        self.Kr_att = np.diag([12.0, 12.0, 6.0])
        self.Kw_att = np.diag([2.2, 2.2, 1.4])

        self.max_tilt_rad = np.radians(12.0)

    def step(self, target_pos, target_vel, is_landing_phase=False, dt=0.025):
        # 1. Ground Disarm Protection Logic
        if is_landing_phase and self.p[2] <= 0.02 and target_pos[2] <= 0.05:
            self.p[2] = 0.0
            self.v = np.zeros(3)
            self.omega = np.zeros(3)
            self.R = np.eye(3)
            return self.p.copy(), 0.0, 0.0

        # 2. Position Control
        e_p = target_pos - self.p
        e_v = target_vel - self.v
        a_des = self.Kp_pos @ e_p + self.Kd_pos @ e_v
        F_des = self.m * (a_des + np.array([0.0, 0.0, self.g]))

        # 3. Tilt Angle Clamping
        z_b_des = F_des / (np.linalg.norm(F_des) + 1e-6)
        max_xy = np.sin(self.max_tilt_rad)
        xy_norm = np.linalg.norm(z_b_des[:2])
        if xy_norm > max_xy:
            z_b_des[:2] = z_b_des[:2] * (max_xy / xy_norm)
            z_b_des[2] = np.sqrt(1.0 - max_xy**2)

        z_b = self.R[:, 2]
        F = max(0.0, np.dot(F_des, z_b))

        # 4. Attitude Control
        x_c = np.array([1.0, 0.0, 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_b_des /= (np.linalg.norm(y_b_des) + 1e-6)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack((x_b_des, y_b_des, z_b_des))

        e_R_matrix = 0.5 * (R_d.T @ self.R - self.R.T @ R_d)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        
        M = -self.Kr_att @ e_R - self.Kw_att @ self.omega + np.cross(self.omega, self.J @ self.omega)

        # 5. Dynamics Integration
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

        f_per_motor = np.clip(F / 4.0, 0.0, 8.0)
        rotor_rpm = np.sqrt(f_per_motor / self.kT)
        tilt_deg = np.arccos(np.clip(self.R[2, 2], -1.0, 1.0)) * (180.0 / np.pi)

        return self.p.copy(), tilt_deg, rotor_rpm


def quintic_smooth(t, t0, t1, x0, x1):
    if t <= t0: return x0
    if t >= t1: return x1
    tau = (t - t0) / (t1 - t0)
    s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
    return x0 + s * (x1 - x0)


def get_smooth_trajectory(t):
    if t < 2.5:
        z_ref = quintic_smooth(t, 0.0, 2.5, 0.0, 1.0)
        mode = "TAKEOFF"
    elif t < 11.5:
        z_ref = 1.0
        mode = "HOVER / MANEUVER"
    else:
        z_ref = quintic_smooth(t, 11.5, 14.0, 1.0, 0.0)
        mode = "LANDING"

    if t < 3.0: x_ref = 0.0
    elif t < 5.5: x_ref = quintic_smooth(t, 3.0, 5.5, 0.0, 0.5)
    elif t < 8.5: x_ref = quintic_smooth(t, 5.5, 8.5, 0.5, -0.91)
    elif t < 11.5: x_ref = quintic_smooth(t, 8.5, 11.5, -0.91, 0.0)
    else: x_ref = 0.0

    return np.array([x_ref, 0.0, z_ref]), mode


def generate_cad_mesh():
    w, h, d = 0.15, 0.10, 0.05
    fuselage_v = np.array([
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h, d],  [w, -h, d],  [w, h, d],  [-w, h, d]
    ])
    fuselage_f = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]
    ]

    angles = np.linspace(0, 2 * np.pi, 16)
    r_rad = 0.13
    rotor_circle = np.stack([r_rad * np.cos(angles), r_rad * np.sin(angles), np.zeros(16)], axis=1)

    return fuselage_v, fuselage_f, rotor_circle


def launch_3d_simulation():
    drone = IndustrialQuadcopterSO3()
    fuse_v, fuse_f, r_circle = generate_cad_mesh()

    dt = 0.025
    t_max = 14.0
    frames = int(t_max / dt)

    fig = plt.figure(figsize=(12, 8.5), facecolor='#ffffff')
    
    # 3D View Axis
    ax3d = fig.add_subplot(2, 1, 1, projection='3d')
    ax3d.set_facecolor('#ffffff')
    ax3d.set_xlim([-1.5, 1.5])
    ax3d.set_ylim([-1.5, 1.5])
    ax3d.set_zlim([0.0, 1.4])
    ax3d.view_init(elev=20, azim=-45)
    ax3d.set_title("Advanced CAD Quadcopter Flight Simulation\nNewton-Euler Dynamics · SO(3) Control · Virtual 3D View", fontsize=11, fontweight='bold')

    # Floor Surface
    xg, yg = np.meshgrid(np.linspace(-2, 2, 6), np.linspace(-2, 2, 6))
    ax3d.plot_surface(xg, yg, np.zeros_like(xg), color='#10b981', alpha=0.85, edgecolor='#047857', lw=0.4)

    # Telemetry HUD Box
    hud_box = fig.text(0.5, 0.47, "", ha='center', va='center', fontsize=10.5, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', edgecolor='#0284c7'))

    # 2D Graph Axis
    ax_plot = fig.add_subplot(2, 1, 2)
    ax_plot.set_xlim([0, t_max])
    ax_plot.set_ylim([-1.2, 1.2])
    ax_plot.set_xlabel("Time [s]")
    ax_plot.set_ylabel("Position [m]")
    ax_plot.grid(True, linestyle='--', alpha=0.5)

    line_ref_z, = ax_plot.plot([], [], 'r--', lw=1.8, label='Reference Z (Altitude)')
    line_act_z, = ax_plot.plot([], [], 'b-', lw=2.2, label='Actual Z (Altitude)')
    line_act_x, = ax_plot.plot([], [], 'g-', lw=2.2, label='Lateral X')
    ax_plot.legend(loc='upper right')

    time_h, x_h, z_h, ref_z_h, trail_h = [], [], [], [], []

    # 3D Artists
    body_poly = Poly3DCollection([], facecolors='#0f172a', edgecolors='#38bdf8', alpha=0.95, linewidths=1.2)
    ax3d.add_collection3d(body_poly)

    arm_lines = [ax3d.plot([], [], [], color='#0284c7', lw=4.5)[0] for _ in range(4)]
    # Front Rotors (Red) & Rear Rotors (Blue)
    rotor_lines = [
        ax3d.plot([], [], [], color='#ef4444', lw=2)[0],
        ax3d.plot([], [], [], color='#3b82f6', lw=2)[0],
        ax3d.plot([], [], [], color='#ef4444', lw=2)[0],
        ax3d.plot([], [], [], color='#3b82f6', lw=2)[0]
    ]
    trail_line, = ax3d.plot([], [], [], '--', color='#16a34a', lw=1.5, alpha=0.7)
    gimbal_led, = ax3d.plot([], [], [], 'ro', ms=7)

    def update(frame):
        t = frame * dt
        target_pos, mode_str = get_smooth_trajectory(t)
        is_landing = (mode_str == "LANDING")

        pos, tilt, rpm = drone.step(target_pos, np.zeros(3), is_landing_phase=is_landing, dt=dt)
        trail_h.append(pos.copy())

        # 1. Update Fuselage Mesh
        tf_fuse = pos + (drone.R @ fuse_v.T).T
        poly_verts = [[tf_fuse[idx] for idx in face] for face in fuse_f]
        body_poly.set_verts(poly_verts)

        # 2. Update Arms & Rotors
        L = drone.L
        rotor_centers = np.array([[L, L, 0], [-L, -L, 0], [-L, L, 0], [L, -L, 0]])
        for i in range(4):
            arm_w = np.stack([pos, pos + drone.R @ rotor_centers[i]])
            arm_lines[i].set_data(arm_w[:, 0], arm_w[:, 1])
            arm_lines[i].set_3d_properties(arm_w[:, 2])

            rotor_w = pos + (drone.R @ (rotor_centers[i] + r_circle).T).T
            rotor_lines[i].set_data(rotor_w[:, 0], rotor_w[:, 1])
            rotor_lines[i].set_3d_properties(rotor_w[:, 2])

        # 3. Front Gimbal Camera Marker (+X direction)
        g_pos = pos + drone.R @ np.array([0.20, 0, 0])
        gimbal_led.set_data([g_pos[0]], [g_pos[1]])
        gimbal_led.set_3d_properties([g_pos[2]])

        # 4. Trail
        pts = np.array(trail_h)
        trail_line.set_data(pts[:, 0], pts[:, 1])
        trail_line.set_3d_properties(pts[:, 2])

        # HUD Text Box
        hud_box.set_text(f"MODE: {mode_str:<16} |  x lateral: {pos[0]:+.2f} m  |  z altitude: {pos[2]:+.2f} m  |  tilt: {tilt:.1f}°  |  rotor Ω: {int(rpm)} rad/s")

        # Telemetry Graph Update
        time_h.append(t)
        x_h.append(pos[0])
        z_h.append(pos[2])
        ref_z_h.append(target_pos[2])

        line_ref_z.set_data(time_h, ref_z_h)
        line_act_z.set_data(time_h, z_h)
        line_act_x.set_data(time_h, x_h)

        return body_poly, trail_line, gimbal_led, line_act_z, line_act_x, hud_box

    ani = FuncAnimation(fig, update, frames=frames, interval=25, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    launch_3d_simulation()