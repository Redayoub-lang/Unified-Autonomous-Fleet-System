Markdown# 🛸 Unified Autonomous Fleet System (5-Quadcopter Swarm)

An enterprise-grade, high-fidelity 3D multi-agent quadcopter swarm simulator built with **Rigid-Body Newton-Euler Dynamics**, **Cascaded Geometric $SO(3)$ Control**, **Control Barrier Function (CBF) Collision Avoidance**, and **Adaptive Dynamic Formations**.

---

## 🌟 Key Features

* **Physically Accurate 6-DOF Rigid Body Dynamics:** Implements authentic Newton-Euler equations of motion, rotational dynamics, rotor torque reaction, and gyroscopic moments.
* **Geometric $SO(3)$ Attitude Controller:** Non-linear attitude control with active **Tilt Saturation Clamping ($12^\circ$ max limit)** to preserve vertical thrust equilibrium.
* **Inter-Agent Collision Avoidance (CBF):** Saturated potential field / Control Barrier Functions for autonomous collision avoidance between swarm agents.
* **Adaptive Dynamic Formations:** Smooth mid-air reconfiguration between **V-Shape**, **Line**, and **Diamond** formations.
* **Environmental Turbulence:** Integrated stochastic wind gust perturbation model to evaluate controller robustness.
* **Ground Disarm & Safety Logic:** Automatic rotor disarming and rotational speed decay upon ground contact.
* **S-Curve Trajectory Generation:** Smooth Quintic Polynomial trajectories ($s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$) for jerk-free flight profiles.
* **High-Fidelity 3D CAD Visualization:** Real-time 3D Matplotlib animation featuring 3D Octagonal fuselages, colored directional rotor meshes, real-time Telemetry HUD, and flight trails.

---

## 📂 Repository Structure

Unified-Autonomous-Fleet-System/├── visual_3d_quadcopter_sim.py       # Main interactive 3D Swarm simulation & visualizer├── quadcopter_physics_engine.py      # Headless physics engine & CSV telemetry logger├── requirements.txt                  # Python package dependencies└── README.md                         # Project documentation
---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/Unified-Autonomous-Fleet-System.git](https://github.com/your-username/Unified-Autonomous-Fleet-System.git)
cd Unified-Autonomous-Fleet-System
2. Install DependenciesBashpip install -r requirements.txt
3. Launch 3D Swarm SimulationRun the interactive 3D visualization showcasing the 5-drone swarm maneuver:Bashpython visual_3d_quadcopter_sim.py
4. Run Headless Telemetry Data LoggingGenerate high-frequency CSV telemetry records (fleet_quadcopter_physics_telemetry.csv):Bashpython quadcopter_physics_engine.py
📐 Mathematical & Physics OverviewRigid Body Equations of Motion$$\dot{p} = v$$$$m \dot{v} = R f_z e_3 - m g e_3 + F_{\text{wind}}$$$$\dot{R} = R \hat{\omega}$$$$J \dot{\omega} = M - \omega \times J \omega$$Geometric Attitude Error on $SO(3)$$$e_R = \frac{1}{2} \left( R_d^T R - R^T R_d \right)^\vee$$$$M = -K_R e_R - K_\omega e_\omega + \omega \times J \omega$$📊 Telemetry & HUD ParametersThe real-time Heads-Up Display monitors essential parameters across the fleet:Flight Mode: TAKEOFF, SWARM MANEUVER, FORMATION RECONFIGURATION, LANDINGLeader Telemetry: Position $(x, y, z)$ coordinates in meters.Average Fleet Tilt Angle: Clamped within safe boundary ($0.0^\circ - 12.0^\circ$).Rotor RPM ($\Omega$): Motor rotational speeds ($\sim 968\text{ rad/s}$ hover equilibrium).Collision Avoidance Status: Real-time indicator (CLEAR / ACTIVE).📜 LicenseDistributed under the MIT License. See LICENSE for more information.