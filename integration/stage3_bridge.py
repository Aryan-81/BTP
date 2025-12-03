"""
Stage-3 Integration: connect DECA rig (Stage 1) with EMOCA coefficients (Stage 2)
"""

import json, threading, time
import numpy as np
import websocket
import torch
import trimesh

# try to import FLAME from the packaged DECA inside this repo, fall back to generic import
try:
    from st1.decalib.models.FLAME import FLAME
except Exception:
    try:
        from deca.FLAME import FLAME
    except Exception:
        FLAME = None

# ---------- Load stored rig parameters ----------
rig_data = np.load("rig_output/flame_params.npz")
alpha = np.array(rig_data["shape"])     # identity
beta0 = np.array(rig_data["expression"])# neutral exp
delta = np.array(rig_data["texture"])   # texture

if FLAME is None:
    raise RuntimeError("FLAME decoder not available. Make sure DECA (st1) is importable.")

# Create a flame model instance using default config in st1 if available
try:
    flame = FLAME()
except TypeError:
    # some FLAME constructors require a model_cfg; try importing config from st1
    try:
        from st1.decalib.utils.config import cfg
        flame = FLAME(cfg.model)
    except Exception:
        # Last resort: create with no args and hope it works
        flame = FLAME()

# ---------- WebSocket listener for EMOCA output ----------
class EmotionReceiver:
    def __init__(self, url="ws://localhost:8080"):
        self.ws = websocket.WebSocketApp(url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close)
        self.latest_beta = beta0
        self.lock = threading.Lock()
        t = threading.Thread(target=self.ws.run_forever)
        t.daemon = True
        t.start()

    def on_message(self, ws, msg):
        data = json.loads(msg)
        with self.lock:
            self.latest_beta = np.array(data["blendshape"])

    def on_error(self, ws, err): print("WebSocket error:", err)
    def on_close(self, ws, *a): print("WebSocket closed")

receiver = EmotionReceiver()

# ---------- Apply expression updates to rig ----------
def apply_expression(flame, alpha, beta, delta):
    v = flame.forward(torch.tensor(alpha),
                      torch.tensor(beta),
                      torch.zeros(1,3))
    v = v.detach().cpu().numpy().squeeze()
    mesh = trimesh.Trimesh(vertices=v, process=False)
    return mesh

# ---------- Main update loop ----------
while True:
    time.sleep(0.04)  # ~25 FPS
    with receiver.lock:
        beta = receiver.latest_beta
    # combine with base identity
    mesh = apply_expression(flame, alpha, beta, delta)
    mesh.export("live_frame.obj")   # Unity/Blender can reload or read
    print("Updated mesh frame exported.")
