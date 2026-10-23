import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class StableSwarmQuadcopterSO3:
    """Enterprise Quadcopter Dynamics Engine with Saturated Avoidance and Smooth SO(3) Attitude."""
    def __init__(self, drone_id=1, initial_pos=None):
        self.drone_id = drone_id
        self.m = 0.917            # Mass [kg]
        self.g = 9.81             # Gravity [m/s^2]
        self.L = 0.35             # Arm length [m]
        self.J = np.diag([0.022, 0.022, 0.043])  # Inertia tensor [kg*m^2]
        self.J_inv = np.linalg.inv(self.J)
        self.kT = 2.4e-6          # Thrust coefficient

        self.p = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        # Well-Damped Smooth Controllers Gains
        self.Kp_pos = np.diag([3.2, 3.2, 6.0])
        self.Kd_pos = np.diag([2.8, 2.8, 4.0])
        self.Kr_att = np.diag([8.0, 8.0, 4.0])
        self.Kw_att = np.diag([1.8, 1.8, 1.0])

        self.max_tilt_rad = np.radians(12.0)  # Strictly clamped to 12 deg
        self.d_safe = 0.90
        self.k_avoid = 2.5                    # Smoothed repulsion gain

    def step(self, target_pos, target_vel, neighbors_pos, wind_force, is_landing_phase=False, dt=0.02):
        # Ground Disarm Protection
        if is_landing_phase and self.p[2] <= 0.02 and target_pos[2] <= 0.05:
            self.p[2] = 0.0
            self.v = np.zeros(3)
            self.omega = np.zeros(3)
            self.R = np.eye(3)
            return self.p.copy(), 0.0, 0.0, False

        # 1. Position Control
        e_p = target_pos - self.p
        e_v = target_vel - self.v

        # 2. Smooth Saturated Inter-Agent Avoidance
        a_avoid = np.zeros(3)
        avoidance_active = False
        if self.p[2] > 0.1:  # Only activate avoidance mid-air
            for other_p in neighbors_pos:
                diff = self.p - other_p
                dist = np.linalg.norm(diff)
                if 1e-3 < dist < self.d_safe:
                    avoidance_active = True
                    repulsion = (self.d_safe - dist) / (dist + 1e-3)
                    a_avoid += self.k_avoid * repulsion * (diff / dist)

        # Clamp max avoidance acceleration to prevent sharp jerks
        avoid_norm = np.linalg.norm(a_avoid)
        if avoid_norm > 2.0:
            a_avoid = a_avoid * (2.0 / avoid_norm)

        a_des = self.Kp_pos @ e_p + self.Kd_pos @ e_v + a_avoid
        F_des = self.m * (a_des + np.array([0.0, 0.0, self.g])) - wind_force

        # 3. Strict Tilt Angle Saturation (Clamping)
        z_b_des = F_des / (np.linalg.norm(F_des) + 1e-6)
        max_xy = np.sin(self.max_tilt_rad)
        xy_norm = np.linalg.norm(z_b_des[:2])
        if xy_norm > max_xy:
            z_b_des[:2] = z_b_des[:2] * (max_xy / xy_norm)
            z_b_des[2] = np.sqrt(1.0 - max_xy**2)

        z_b = self.R[:, 2]
        F = max(0.0, np.dot(F_des, z_b))

        # 4. SO(3) Attitude Control
        x_c = np.array([1.0, 0.0, 0.0])
        y_b_des = np.cross(z_b_des, x_c)
        y_b_des /= (np.linalg.norm(y_b_des) + 1e-6)
        x_b_des = np.cross(y_b_des, z_b_des)
        R_d = np.column_stack((x_b_des, y_b_des, z_b_des))

        e_R_matrix = 0.5 * (R_d.T @ self.R - self.R.T @ R_d)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])

        M = -self.Kr_att @ e_R - self.Kw_att @ self.omega + np.cross(self.omega, self.J @ self.omega)

        # 5. Physics Integration
        e3 = np.array([0.0, 0.0, 1.0])
        v_dot = (self.R @ (F * e3) - self.m * self.g * e3 + wind_force) / self.m
        self.v += v_dot * dt
        self.p += self.v * dt

        if self.p[2] < 0.0:
            self.p[2] = 0.0
            self.v[2] = max(0.0, self.v[2])

        omega_dot = self.J_inv @ (M - np.cross(self.omega, self.J @ self.omega))
        self.omega += omega_dot * dt
        self.omega = np.clip(self.omega, -6.0, 6.0)  # Clamp angular velocities for stability

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

        return self.p.copy(), tilt_deg, rotor_rpm, avoidance_active


def quintic_smooth(t, t0, t1, x0, x1):
    if t <= t0: return x0
    if t >= t1: return x1
    tau = (t - t0) / (t1 - t0)
    s = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
    return x0 + s * (x1 - x0)


def get_adaptive_fleet_profile(t):
    if t < 3.0:
        z_ref = quintic_smooth(t, 0.0, 3.0, 0.0, 1.8)
        mode = "TAKEOFF (V-FORMATION)"
        formation_type = "V_SHAPE"
    elif t < 7.0:
        z_ref = 1.8
        mode = "SWARM MANEUVER (V-SHAPE)"
        formation_type = "V_SHAPE"
    elif t < 11.0:
        z_ref = 1.8
        mode = "RECONFIGURATION (LINE)"
        formation_type = "LINE"
    elif t < 15.0:
        z_ref = 1.8
        mode = "RECONFIGURATION (DIAMOND)"
        formation_type = "DIAMOND"
    else:
        z_ref = quintic_smooth(t, 15.0, 18.0, 1.8, 0.0)
        mode = "LANDING PHASE"
        formation_type = "V_SHAPE"

    if t < 3.0: x_ref, y_ref = 0.0, 0.0
    elif t < 8.0:
        x_ref = quintic_smooth(t, 3.0, 8.0, 0.0, 2.0)
        y_ref = quintic_smooth(t, 3.0, 8.0, 0.0, 1.2)
    elif t < 13.0:
        x_ref = quintic_smooth(t, 8.0, 13.0, 2.0, -2.0)
        y_ref = quintic_smooth(t, 8.0, 13.0, 1.2, -1.2)
    elif t < 15.0:
        x_ref = quintic_smooth(t, 13.0, 15.0, -2.0, 0.0)
        y_ref = quintic_smooth(t, 13.0, 15.0, -1.2, 0.0)
    else:
        x_ref, y_ref = 0.0, 0.0

    return np.array([x_ref, y_ref, z_ref]), mode, formation_type


def get_formation_offsets(formation_type):
    if formation_type == "LINE":
        return np.array([
            [ 0.0,  0.0,  0.00],
            [ 0.0,  1.3,  0.05],
            [ 0.0, -1.3,  0.05],
            [ 0.0,  2.6, -0.05],
            [ 0.0, -2.6, -0.05]
        ])
    elif formation_type == "DIAMOND":
        return np.array([
            [ 0.0,   0.0,  0.00],
            [ 1.4,   0.0,  0.08],
            [-1.4,   0.0, -0.08],
            [ 0.0,   1.4,  0.05],
            [ 0.0,  -1.4, -0.05]
        ])
    else:  # V_SHAPE
        return np.array([
            [ 0.0,  0.0,  0.00],
            [ 1.3,  1.2,  0.10],
            [ 1.3, -1.2,  0.10],
            [-1.3,  1.3, -0.10],
            [-1.3, -1.3, -0.10]
        ])


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


def launch_stable_swarm_sim():
    NUM_DRONES = 5
    drone_colors = ['#0284c7', '#dc2626', '#16a34a', '#d97706', '#9333ea']

    init_offsets = get_formation_offsets("V_SHAPE")
    drones = [StableSwarmQuadcopterSO3(drone_id=i+1, initial_pos=init_offsets[i] * [1, 1, 0]) for i in range(NUM_DRONES)]
    fuse_v, fuse_f, r_circle = generate_cad_mesh()

    dt = 0.02
    t_max = 18.0
    frames_count = int(t_max / dt)

    fig = plt.figure(figsize=(13, 8.5), facecolor='#ffffff')

    ax3d = fig.add_subplot(2, 1, 1, projection='3d')
    ax3d.set_facecolor('#ffffff')
    ax3d.set_xlim([-4.0, 4.0])
    ax3d.set_ylim([-4.0, 4.0])
    ax3d.set_zlim([0.0, 2.8])
    ax3d.view_init(elev=24, azim=-48)
    ax3d.set_title("Enterprise 5-Quadcopter Fleet Swarm Simulation\nNewton-Euler Dynamics · SO(3) Geometric Control · Stable Formation", fontsize=11, fontweight='bold')

    xg, yg = np.meshgrid(np.linspace(-4.5, 4.5, 10), np.linspace(-4.5, 4.5, 10))
    ax3d.plot_surface(xg, yg, np.zeros_like(xg), color='#10b981', alpha=0.75, edgecolor='#047857', lw=0.3)

    hud_box = fig.text(0.5, 0.47, "", ha='center', va='center', fontsize=10.0, fontweight='bold',
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
        leader_target, mode_str, form_type = get_adaptive_fleet_profile(t)
        is_landing = (mode_str.startswith("LANDING"))
        current_offsets = get_formation_offsets(form_type)

        wind_force = np.array([
            0.08 * np.sin(0.5 * t),
            0.08 * np.cos(0.4 * t),
            0.0
        ])

        time_h.append(t)
        ref_z_h.append(leader_target[2])

        all_current_positions = [d.p.copy() for d in drones]
        avg_tilt, avg_rpm = [], []
        avoid_flags = []

        for i in range(NUM_DRONES):
            drone_target = leader_target + current_offsets[i]
            if is_landing:
                drone_target[2] = leader_target[2]

            neighbors = [all_current_positions[j] for j in range(NUM_DRONES) if j != i]
            pos, tilt, rpm, is_avoiding = drones[i].step(drone_target, np.zeros(3), neighbors, wind_force, is_landing_phase=is_landing, dt=dt)
            drone_trails[i].append(pos.copy())

            avg_tilt.append(tilt)
            avg_rpm.append(rpm)
            avoid_flags.append(is_avoiding)

            # Update 3D Geometry
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

        avoid_status = "ACTIVE" if any(avoid_flags) else "CLEAR"
        hud_box.set_text(f"MODE: {mode_str:<26} | Wind: ({wind_force[0]:+.2f}, {wind_force[1]:+.2f}) N | Avoidance: {avoid_status} | Avg Tilt: {np.mean(avg_tilt):.1f}°")
        line_ref_z.set_data(time_h, ref_z_h)

        return []

    ani = FuncAnimation(fig, update, frames=frames_count, interval=20, blit=False)
    fig.ani = ani

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    launch_stable_swarm_sim()