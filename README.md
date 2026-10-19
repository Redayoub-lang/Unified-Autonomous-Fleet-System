# Unified Autonomous Fleet System (v1.0 Golden Integration)
## Integrated Autonomous Fleet Command & Control Suite

An enterprise-grade unified autonomous robotics suite integrating **AMR Nav2 Behavior Trees**, **Factor Graph VIO 3D OctoMap SLAM**, **3D Multi-UAV Swarm DMPC Orchestration**, and **Edge AI INT8 DeepOCSORT Re-ID Target Tracking**.

---

## Architecture & Subsystems Integration

1. **Ground Base (`AMR-Nav2-Suite`):** Kinematically constrained ground base navigation utilizing behavior trees and TEB planners.
2. **Perception & Mapping (`SLAM3D-Vision-Suite`):** Continuous 3D OctoMap environment reconstruction powered by VIO state estimation.
3. **Aerial Escort (`DroneSwarm-Suite`):** Multi-UAV 3D swarm performing automated air-to-ground reconnaissance using DMPC and MAVLink telemetry streams.
4. **Target Tracking (`EdgeAI-Vision-Optimization`):** INT8 TensorRT inference engine extracting $\mathbb{R}^{128}$ Re-ID appearance embeddings to identify and track targets across the fleet coordinate system.

---

## Joint State Space Formulation

Global Fleet Joint State Matrix $\mathcal{X}_{\text{Fleet}}(t)$:

$$\mathcal{X}_{\text{Fleet}}(t) = \begin{bmatrix} \mathbf{x}_{\text{AMR}}(t) \\ \mathbf{x}_{\text{SLAM\_Map}}(t) \\ \mathbf{x}_{\text{UAV\_1:N}}(t) \\ \mathbf{f}_{\text{ReID\_Targets}}(t) \end{bmatrix}$$

---

## Quick Execution Guide

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Unified Fleet Command Center
python unified_fleet_system_engine.py