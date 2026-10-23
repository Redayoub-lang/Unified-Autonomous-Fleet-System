import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class SwarmQuadcopterSO3:
    """Rigid Body Newton-Euler Quadcopter Engine with SO(3) Control."""
    def __init__(self, drone_id=1, initial_pos=None):
        self.drone_id = drone_id
        self.m = 0.917            # Mass [kg]
        self.g = 9.81             # Gravity [m/s^2]
        self.L = 0.35             # Arm length [m]
        self.J = np.diag([0.022, 0.022, 0.043])  # Inertia tensor [kg*m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.kT = 2.4e-6          # Thrust coefficient
        self.b = 1.0e-4           # Drag coefficient

        self.p = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        self.Kp_pos = np.diag([4.2, 4.2, 7.5])
        self.Kd_pos = np.diag([3.2, 3.2, 5.0])
        self.Kr_att = np.diag([12.0, 12.0, 6.0])
        self.Kw_att = np.diag([2.2, 2.2, 1.4])
        self.max_tilt_rad = np.radians(14.0)

    def step(self, target_pos, target_vel, is_landing_phase=False, dt=0.03):
        if is_landing_phase and self.p[2] <= 0.02 and target_pos[2] <= 0.05:
            self.p[2] = 0.0
            self.v = np.zeros(3)
            self.omega = np.zeros(3)
            self.R = np.eye(3)
            return self.p.copy(), 0.0, 0.0

        e_p = target_pos - self.p
        e_v = target_vel - self.v
        a_des = self.Kp_pos @ e_p + self.Kd_pos @ e_v
        F_des = self.m * (a_des + np.array([0.0, 0.0, self.g]))

        z_b_des = F_des / (np.linalg.norm(F_des) + 1e-6)
        max_xy = np.sin(self.max_tilt_rad)
        xy_norm = np.linalg.norm(z_b_des[:2])
        if xy_norm > max_xy:
            z_b_des[:2] = z_b_des[:2] * (max_xy / xy_norm)
            z_b_des[2] = np.sqrt(1.0 - max_xy**2)

        z_b = self.R[:, 2]
        F = max(0.0, np.dot(F_des, z_b))

        x_c = np.array([1.0, 0.0, 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_b_des /= (np.linalg.norm(y_b_des) + 1e-6)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack((x_b_des, y_b_des, z_b_des))

        e_R_matrix = 0.5 * (R_d.T @ self.R - self.R.T @ R_d)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        
        M = -self.Kr_att @ e_R - self.Kw_att @ self.omega + np.cross(self.omega, self.J @ self.omega)

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


def get_expanded_leader_trajectory(t):
    if t < 3.0:
        z_ref = quintic_smooth(t, 0.0, 3.0, 0.0, 1.8)
        mode = "TAKEOFF"
    elif t < 13.0:
        z_ref = 1.8
        mode = "SWARM MANEUVER"
    else:
        z_ref = quintic_smooth(t, 13.0, 16.0, 1.8, 0.0)
        mode = "LANDING"

    if t < 3.0: x_ref = 0.0
    elif t < 7.0: x_ref = quintic_smooth(t, 3.0, 7.0, 0.0, 2.2)
    elif t < 10.0: x_ref = quintic_smooth(t, 7.0, 10.0, 2.2, -2.2)
    elif t < 13.0: x_ref = quintic_smooth(t, 10.0, 13.0, -2.2, 0.0)
    else: x_ref = 0.0

    if t < 4.0: y_ref = 0.0
    elif t < 8.0: y_ref = quintic_smooth(t, 4.0, 8.0, 0.0, 1.5)
    elif t < 11.0: y_ref = quintic_smooth(t, 8.0, 11.0, 1.5, -1.5)
    elif t < 13.0: y_ref = quintic_smooth(t, 11.0, 13.0, -1.5, 0.0)
    else: y_ref = 0.0

    return np.array([x_ref, y_ref, z_ref]), mode


def generate_cad_mesh():
    w, h, d = 0.16, 0.10, 0.05
    fuselage_v = np.array([
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d],
        [-w, -h, d],  [w, -h, d],  [w, h, d],  [-w, h, d]
    ])
    fuselage_f = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]
    ]

    angles = np.linspace(0, 2 * np.pi, 12)
    r_rad = 0.12
    rotor_circle = np.stack([r_rad * np.cos(angles), r_rad * np.sin(angles), np.zeros(12)], axis=1)

    return fuselage_v, fuselage_f, rotor_circle


def launch_expanded_5_drone_swarm():
    NUM_DRONES = 5

    swarm_offsets = np.array([
        [ 0.0,  0.0,  0.00],
        [ 1.2,  1.0,  0.10],
        [ 1.2, -1.0,  0.10],
        [-1.2,  1.2, -0.10],
        [-1.2, -1.2, -0.10]
    ])

    drone_colors = ['#0284c7', '#dc2626', '#16a34a', '#d97706', '#9333ea']

    drones = [SwarmQuadcopterSO3(drone_id=i+1, initial_pos=swarm_offsets[i] * [1, 1, 0]) for i in range(NUM_DRONES)]
    fuse_v, fuse_f, r_circle = generate_cad_mesh()

    dt = 0.03
    t_max = 16.0
    frames_count = int(t_max / dt)

    fig = plt.figure(figsize=(13, 8.5), facecolor='#ffffff')
    
    ax3d = fig.add_subplot(2, 1, 1, projection='3d')
    ax3d.set_facecolor('#ffffff')
    ax3d.set_xlim([-4.0, 4.0])
    ax3d.set_ylim([-4.0, 4.0])
    ax3d.set_zlim([0.0, 2.8])
    ax3d.view_init(elev=24, azim=-48)
    ax3d.set_title("Autonomous 5-Quadcopter Fleet Swarm Simulation\nNewton-Euler Dynamics · SO(3) Control · Expanded 3D Airspace", fontsize=11, fontweight='bold')

    xg, yg = np.meshgrid(np.linspace(-4.5, 4.5, 10), np.linspace(-4.5, 4.5, 10))
    ax3d.plot_surface(xg, yg, np.zeros_like(xg), color='#10b981', alpha=0.75, edgecolor='#047857', lw=0.3)

    hud_box = fig.text(0.5, 0.47, "", ha='center', va='center', fontsize=10.5, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f1f5f9', edgecolor='#0284c7'))

    ax_plot = fig.add_subplot(2, 1, 2)
    ax_plot.set_xlim([0, t_max])
    ax_plot.set_ylim([-2.5, 2.5])
    ax_plot.set_xlabel("Time [s]")
    ax_plot.set_ylabel("Position [m]")
    ax_plot.grid(True, linestyle='--', alpha=0.5)

    line_ref_z, = ax_plot.plot([], [], 'r--', lw=1.8, label='Leader Target Z')
    drone_lines_z = [ax_plot.plot([], [], color=drone_colors[i], lw=1.8, label=f'Drone {i+1} Z')[0] for i in range(NUM_DRONES)]
    ax_plot.legend(loc='upper right', ncol=3)

    time_h, ref_z_h = [], []
    drones_z_h = [[] for _ in range(NUM_DRONES)]

    body_polys, arm_lines_list, rotor_lines_list, trail_lines = [], [], [], []

    for i in range(NUM_DRONES):
        poly = Poly3DCollection([], facecolors=drone_colors[i], edgecolors='#0f172a', alpha=0.9, linewidths=1.0)
        ax3d.add_collection3d(poly)
        body_polys.append(poly)

        arms = [ax3d.plot([], [], [], color='#1e293b', lw=3.5)[0] for _ in range(4)]
        arm_lines_list.append(arms)

        rotors = [ax3d.plot([], [], [], color=drone_colors[i], lw=1.5)[0] for _ in range(4)]
        rotor_lines_list.append(rotors)

        trail, = ax3d.plot([], [], [], '--', color=drone_colors[i], lw=1.2, alpha=0.5)
        trail_lines.append(trail)

    drone_trails = [[] for _ in range(NUM_DRONES)]

    def update(frame):
        t = frame * dt
        leader_target, mode_str = get_expanded_leader_trajectory(t)
        is_landing = (mode_str == "LANDING")

        time_h.append(t)
        ref_z_h.append(leader_target[2])

        avg_tilt, avg_rpm = [], []

        for i in range(NUM_DRONES):
            drone_target = leader_target + swarm_offsets[i]
            if is_landing:
                drone_target[2] = leader_target[2]

            pos, tilt, rpm = drones[i].step(drone_target, np.zeros(3), is_landing_phase=is_landing, dt=dt)
            drone_trails[i].append(pos.copy())

            avg_tilt.append(tilt)
            avg_rpm.append(rpm)

            tf_fuse = pos + (drones[i].R @ fuse_v.T).T
            poly_verts = [[tf_fuse[idx] for idx in face] for face in fuse_f]
            body_polys[i].set_verts(poly_verts)

            L = drones[i].L
            rotor_centers = np.array([[L, L, 0], [-L, -L, 0], [-L, L, 0], [L, -L, 0]])
            for r in range(4):
                arm_w = np.stack([pos, pos + drones[i].R @ rotor_centers[r]])
                arm_lines_list[i][r].set_data(arm_w[:, 0], arm_w[:, 1])
                arm_lines_list[i][r].set_3d_properties(arm_w[:, 2])

                rotor_w = pos + (drones[i].R @ (rotor_centers[r] + r_circle).T).T
                rotor_lines_list[i][r].set_data(rotor_w[:, 0], rotor_w[:, 1])
                rotor_lines_list[i][r].set_3d_properties(rotor_w[:, 2])

            pts = np.array(drone_trails[i])
            trail_lines[i].set_data(pts[:, 0], pts[:, 1])
            trail_lines[i].set_3d_properties(pts[:, 2])

            drones_z_h[i].append(pos[2])
            drone_lines_z[i].set_data(time_h, drones_z_h[i])

        hud_box.set_text(f"MODE: {mode_str:<20} | Fleet Position: ({leader_target[0]:+.1f}, {leader_target[1]:+.1f}, {leader_target[2]:.1f})m | Avg Tilt: {np.mean(avg_tilt):.1f}° | Avg Rotor Ω: {int(np.mean(avg_rpm))} rad/s")
        line_ref_z.set_data(time_h, ref_z_h)

        return []

    ani = FuncAnimation(fig, update, frames=frames_count, interval=30, blit=False)
    fig.ani = ani

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    launch_expanded_5_drone_swarm()