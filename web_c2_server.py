import asyncio
import json
import math
import numpy as np
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Unified Fleet Web-C2 Gateway v2.0")

# Active WebSocket Clients
connected_clients = set()


class FleetSimulationEnginev2:
    """Enterprise Fleet Physics & State Synthesizer for Web C2 Stream."""
    def __init__(self):
        self.amr_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "theta": 0.0}
        self.target_goal = {"x": 25.0, "y": 18.0, "z": 0.0}
        
        self.drones = [
            {"id": 1, "x": 0.0, "y": 0.0, "z": 3.0},
            {"id": 2, "x": -2.0, "y": 2.0, "z": 4.0},
            {"id": 3, "x": 2.0, "y": -2.0, "z": 3.5}
        ]
        
        self.reid_targets = [
            {"id": 901, "x": 12.0, "y": 8.0, "z": 0.0, "label": "Suspicious Target"}
        ]
        
        self.voxels = []
        self._generate_static_map()

    def _generate_static_map(self):
        """Generates 3D Environment Obstacle Voxels for SLAM Stream."""
        for x in range(-5, 30, 2):
            for y in range(-5, 25, 2):
                if np.random.rand() < 0.12:
                    self.voxels.append({"x": float(x), "y": float(y), "z": float(np.random.uniform(0, 2))})

    def step(self, dt=0.05, t_sec=0.0):
        # 1. Update AMR Kinematics
        dx = self.target_goal["x"] - self.amr_pose["x"]
        dy = self.target_goal["y"] - self.amr_pose["y"]
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0.5:
            heading = math.atan2(dy, dx)
            v = min(2.5, 0.8 * dist)
            self.amr_pose["theta"] = heading
            self.amr_pose["x"] += v * math.cos(heading) * dt
            self.amr_pose["y"] += v * math.sin(heading) * dt

        # 2. Update Drone Swarm DMPC Escort Positions
        offsets = [
            (2.0, 2.0, 3.5),
            (-2.0, 2.0, 4.0),
            (0.0, -3.0, 3.0)
        ]
        for i, drone in enumerate(self.drones):
            ox, oy, oz = offsets[i]
            target_x = self.amr_pose["x"] + ox
            target_y = self.amr_pose["y"] + oy
            target_z = oz + 0.5 * math.sin(t_sec * 2.0 + i)

            drone["x"] += 3.0 * (target_x - drone["x"]) * dt
            drone["y"] += 3.0 * (target_y - drone["y"]) * dt
            drone["z"] += 3.0 * (target_z - drone["z"]) * dt

        # 3. Dynamic Re-ID Target Path
        self.reid_targets[0]["x"] = 12.0 + 4.0 * math.cos(t_sec * 0.5)
        self.reid_targets[0]["y"] = 8.0 + 3.0 * math.sin(t_sec * 0.5)

    def get_telemetry_payload(self, elapsed_time):
        return {
            "timestamp": round(elapsed_time, 2),
            "amr": self.amr_pose,
            "goal": self.target_goal,
            "drones": self.drones,
            "reid_targets": self.reid_targets,
            "voxels": self.voxels[:120]  # Transmit map slice
        }


fleet_engine = FleetSimulationEnginev2()


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    print(f"[INFO] New Web C2 Client Connected. Active clients: {len(connected_clients)}")
    
    start_time = time.time()
    try:
        while True:
            elapsed = time.time() - start_time
            fleet_engine.step(dt=0.05, t_sec=elapsed)
            
            payload = fleet_engine.get_telemetry_payload(elapsed)
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.05)  # 20 Hz Broadcast Rate
            
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        print(f"[INFO] Client Disconnected. Remaining clients: {len(connected_clients)}")


@app.get("/")
async def get_web_c2_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    print("[INFO] Starting Unified Fleet Web C2 Gateway v2.0 on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)