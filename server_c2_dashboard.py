import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


class EnterpriseSwarmQuadcopterSO3:
    """Core Physics Engine with Responsive SO(3) Dynamic Trajectory Control."""
    def __init__(self, drone_id=1, initial_pos=None):
        self.drone_id = drone_id
        self.m = 0.917
        self.g = 9.81
        self.J = np.diag([0.022, 0.022, 0.043])
        self.J_inv = np.linalg.inv(self.J)
        self.kT = 2.4e-6

        self.p = np.array(initial_pos if initial_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.v = np.zeros(3, dtype=np.float64)
        self.R = np.eye(3, dtype=np.float64)
        self.omega = np.zeros(3, dtype=np.float64)

        self.Kp_pos = np.diag([3.8, 3.8, 6.5])
        self.Kd_pos = np.diag([2.8, 2.8, 4.0])
        self.Kr_att = np.diag([9.0, 9.0, 5.0])
        self.Kw_att = np.diag([1.8, 1.8, 1.0])
        self.max_tilt_rad = np.radians(12.0)

    def step(self, target_pos, target_vel, is_landing=False, dt=0.03):
        if is_landing and self.p[2] <= 0.02 and target_pos[2] <= 0.05:
            self.p[2] = 0.0
            self.v = np.zeros(3)
            self.omega = np.zeros(3)
            self.R = np.eye(3)
            return self.p.tolist(), 0.0, 0.0

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
        self.omega = np.clip(self.omega, -6.0, 6.0)

        omega_hat = np.array([
            [0, -self.omega[2], self.omega[1]],
            [self.omega[2], 0, -self.omega[0]],
            [-self.omega[1], self.omega[0], 0]
        ])
        self.R += self.R @ omega_hat * dt
        u, _, vh = np.linalg.svd(self.R)
        self.R = u @ vh

        f_per_motor = np.clip(F / 4.0, 0.0, 8.0)
        rotor_rpm = float(np.sqrt(f_per_motor / self.kT))
        tilt_deg = float(np.arccos(np.clip(self.R[2, 2], -1.0, 1.0)) * (180.0 / np.pi))

        return self.p.tolist(), tilt_deg, rotor_rpm


app = FastAPI()


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.get("/")
async def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Fleet C2 Command & Control Center</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <style>
            body { margin: 0; background: #0b0f19; color: #f8fafc; font-family: 'Segoe UI', monospace; overflow: hidden; }
            #hud { position: absolute; top: 15px; left: 15px; background: rgba(15, 23, 42, 0.90); padding: 18px; border-radius: 10px; border: 1px solid #0284c7; z-index: 10; width: 330px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
            h2 { margin: 0 0 10px 0; color: #38bdf8; font-size: 15px; letter-spacing: 1px; }
            .metric { margin-bottom: 6px; font-size: 12.5px; }
            .status-online { color: #4ade80; font-weight: bold; }
            .btn-group { display: flex; gap: 8px; margin-top: 12px; }
            button { flex: 1; padding: 8px; background: #0369a1; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 11px; transition: 0.2s; }
            button:hover { background: #0284c7; }
            button.land { background: #b91c1c; }
            button.land:hover { background: #dc2626; }
        </style>
    </head>
    <body>
        <div id="hud">
            <h2>🛸 FLEET C2 COMMAND CENTER</h2>
            <div class="metric">Status: <span class="status-online" id="net-status">CONNECTING...</span></div>
            <div class="metric">Flight Mode: <span id="flight-mode" style="color:#f59e0b; font-weight:bold;">TAKEOFF</span></div>
            <div class="metric">Sim Time: <span id="sim-time">0.00</span> s</div>
            
            <div class="btn-group">
                <button onclick="sendCommand('TAKEOFF')">TAKEOFF</button>
                <button onclick="sendCommand('MANEUVER')">PATROL</button>
                <button class="land" onclick="sendCommand('LAND')">LAND</button>
            </div>

            <hr style="border-color: #334155; margin: 12px 0;">
            <div id="drone-list"></div>
        </div>

        <script>
            let scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0b0f19);

            let camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, -6.5, 3.8);

            let renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            let controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 0, 1.2);

            let grid = new THREE.GridHelper(12, 24, 0x0284c7, 0x1e293b);
            grid.rotation.x = Math.PI / 2;
            scene.add(grid);

            let dirLight = new THREE.DirectionalLight(0xffffff, 1.3);
            dirLight.position.set(5, -5, 10);
            scene.add(dirLight);
            scene.add(new THREE.AmbientLight(0x64748b));

            let droneColors = [0x0284c7, 0xdc2626, 0x16a34a, 0xd97706, 0x9333ea];
            let droneGroups = [];

            for(let i = 0; i < 5; i++) {
                let group = new THREE.Group();
                
                // Fuselage (Larger Scale)
                let bodyGeo = new THREE.BoxGeometry(0.55, 0.35, 0.12);
                let bodyMat = new THREE.MeshLambertMaterial({ color: droneColors[i] });
                let body = new THREE.Mesh(bodyGeo, bodyMat);
                group.add(body);

                // Arms
                let armGeo = new THREE.BoxGeometry(1.1, 0.06, 0.03);
                let armMat = new THREE.MeshBasicMaterial({ color: 0x475569 });
                let arm1 = new THREE.Mesh(armGeo, armMat);
                let arm2 = new THREE.Mesh(armGeo, armMat);
                arm2.rotation.z = Math.PI / 2;
                group.add(arm1);
                group.add(arm2);

                // Rotors
                let rotorGeo = new THREE.CylinderGeometry(0.2, 0.2, 0.02, 16);
                let rotorMat = new THREE.MeshBasicMaterial({ color: droneColors[i], transparent: true, opacity: 0.7 });
                let positions = [[0.5,0.5,0.08], [-0.5,-0.5,0.08], [-0.5,0.5,0.08], [0.5,-0.5,0.08]];
                positions.forEach(pos => {
                    let rMesh = new THREE.Mesh(rotorGeo, rotorMat);
                    rMesh.position.set(pos[0], pos[1], pos[2]);
                    rMesh.rotation.x = Math.PI / 2;
                    group.add(rMesh);
                });

                scene.add(group);
                droneGroups.push(group);
            }

            let ws = new WebSocket(`ws://${location.host}/ws/telemetry`);
            
            ws.onopen = () => { document.getElementById('net-status').innerText = 'LIVE ONLINE'; };
            ws.onclose = () => { document.getElementById('net-status').innerText = 'OFFLINE'; document.getElementById('net-status').style.color = '#ef4444'; };

            ws.onmessage = (event) => {
                let data = JSON.parse(event.data);
                document.getElementById('sim-time').innerText = data.time.toFixed(2);
                document.getElementById('flight-mode').innerText = data.mode;

                let listHtml = '';
                data.drones.forEach((d, i) => {
                    droneGroups[i].position.set(d.pos[0], d.pos[1], d.pos[2]);
                    listHtml += `<div class="metric">Drone ${i+1}: Z=${d.pos[2].toFixed(2)}m | Tilt=${d.tilt.toFixed(1)}°</div>`;
                });
                document.getElementById('drone-list').innerHTML = listHtml;
            };

            function sendCommand(cmd) {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ command: cmd }));
                }
            }

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await manager.connect(websocket)
    swarm_offsets = np.array([
        [0.0, 0.0, 0.0],
        [1.3, 1.1, 0.08],
        [1.3, -1.1, 0.08],
        [-1.3, 1.2, -0.05],
        [-1.3, -1.2, -0.05]
    ])
    drones = [EnterpriseSwarmQuadcopterSO3(drone_id=i+1, initial_pos=swarm_offsets[i]) for i in range(5)]

    t = 0.0
    dt = 0.03
    current_mode = "TAKEOFF"

    try:
        while True:
            try:
                raw_msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                cmd_data = json.loads(raw_msg)
                if "command" in cmd_data:
                    current_mode = cmd_data["command"]
            except asyncio.TimeoutError:
                pass

            t += dt

            if current_mode == "TAKEOFF":
                z_target = 1.8
                leader_pos = np.array([0.0, 0.0, z_target])
            elif current_mode == "MANEUVER":
                z_target = 1.8
                leader_pos = np.array([1.5 * np.sin(0.4 * t), 1.2 * np.cos(0.3 * t), z_target])
            elif current_mode == "LAND":
                z_target = 0.0
                leader_pos = np.array([drones[0].p[0], drones[0].p[1], z_target])

            drone_payload = []
            for i in range(5):
                target = leader_pos + swarm_offsets[i]
                if current_mode == "LAND":
                    target[2] = z_target

                pos, tilt, rpm = drones[i].step(target, np.zeros(3), is_landing=(current_mode == "LAND"), dt=dt)
                drone_payload.append({
                    "id": i + 1,
                    "pos": pos,
                    "tilt": tilt,
                    "rpm": rpm
                })

            telemetry_data = {
                "time": t,
                "mode": current_mode,
                "drones": drone_payload
            }

            await websocket.send_text(json.dumps(telemetry_data))
            await asyncio.sleep(dt)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)