"""
integration/integrate_deca_emoca.py

Simple integration helper that:
- runs DECA (stage1) on a single image and saves FLAME params + obj
- optionally loads an EMOCA model (stage2) to stream expression/blendshape updates
- runs a lightweight websocket/TCP server that accepts JSON messages with
  a `blendshape` array (or `expression`) and exports an updated OBJ

Usage examples:
  # run DECA on a single image and save rig
  python integration/integrate_deca_emoca.py --mode deca_only --image inputs/my_face.jpg

  # start a websocket server that accepts JSON {"blendshape": [...]}
  python integration/integrate_deca_emoca.py --mode serve_ws --port 8080

  # run EMOCA local inference (requires EMOCA run files and models)
  python integration/integrate_deca_emoca.py --mode emoca_local --image inputs/my_face.jpg \
    --path_to_models /path/to/emoca/models --run_name my_run --stage detail

Notes:
- This script assumes the repository root is on Python path when run from the repo root.
- It tries to be robust if GPU or some optional deps are missing.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh
import torch

# Ensure repo root is importable when running from the repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def safe_import_deca():
    try:
        from st1.decalib.deca import DECA
        return DECA
    except Exception as e:
        raise ImportError(f"Could not import DECA from st1: {e}")

def run_deca_on_image(image_path, device='cpu', out_dir='rig_output'):
    DECA = safe_import_deca()
    device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) or device == 'cuda' else 'cpu'
    deca = DECA(device=device)

    codedict, opdict, visdict = deca.run(image_path)

    shape = codedict['shape'].detach().cpu().numpy().squeeze()
    exp = codedict['exp'].detach().cpu().numpy().squeeze()
    tex = codedict['tex'].detach().cpu().numpy().squeeze()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(out_dir) / 'flame_params.npz', shape=shape, expression=exp, texture=tex)

    # save coarse and detailed obj if available
    try:
        deca.save_obj(str(Path(out_dir) / 'initial.obj'), opdict)
    except Exception:
        # fallback: try create mesh from verts
        try:
            verts = opdict['verts'][0].detach().cpu().numpy()
            faces = deca.render.faces[0].cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            mesh.export(str(Path(out_dir) / 'initial.obj'))
        except Exception:
            print("Warning: could not export OBJ (missing renderer or faces)")

    print(f"Saved rig params to {out_dir}/flame_params.npz")
    return deca, shape, exp, tex


def apply_expression_and_export(deca, shape, expression, outpath='live_frame.obj'):
    # build tensors with correct shapes and device
    device = next(deca.parameters()).device if any(True for _ in deca.parameters()) else torch.device('cpu')
    shape_t = torch.tensor(shape, dtype=torch.float32, device=device)[None, ...]
    exp_t = torch.tensor(expression, dtype=torch.float32, device=device)[None, ...]
    pose_dim = deca.param_dict.get('pose', 15) if hasattr(deca, 'param_dict') else 15
    pose_t = torch.zeros((1, int(pose_dim)), dtype=torch.float32, device=device)

    with torch.no_grad():
        verts, trans_verts, landmarks3d = deca.flame(shape_t, exp_t, pose_t)
    verts = verts[0].detach().cpu().numpy()

    # try to get faces from the renderer if available, otherwise try deca.flame model buffer
    faces = None
    try:
        faces = deca.render.faces[0].cpu().numpy()
    except Exception:
        # try reading faces tensor from FLAME module if present
        try:
            faces = deca.flame.faces_tensor.cpu().numpy()
        except Exception:
            faces = None

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False) if faces is not None else trimesh.Trimesh(vertices=verts, process=False)
    mesh.export(outpath)
    print(f"Exported updated mesh to {outpath}")
    return outpath


class BlendshapeServer:
    """Simple server that accepts JSON messages and applies expression updates.

    The server first expects that `rig_output/flame_params.npz` exists (produced by DECA).
    Incoming message examples:
      {"blendshape": [0.0, 0.1, ...]}
      {"expression": [ ... ]}
    """
    def __init__(self, deca, shape, expression, outdir='rig_output', port=8080):
        self.deca = deca
        self.shape = shape
        self.expression = expression
        self.outdir = Path(outdir)
        self.port = port
        self.running = False

    def handle_update(self, data):
        # accept multiple possible keys
        arr = None
        for k in ('blendshape', 'blendshape_coeffs', 'expression', 'exp'):
            if k in data:
                arr = np.array(data[k])
                break
        if arr is None:
            print('Received JSON without recognizable expression key')
            return
        # apply scaling if values are in [0,100]
        if arr.max() > 10:
            arr = arr / 100.0
        timestamp = int(time.time()*1000)
        outpath = str(self.outdir / f'live_frame_{timestamp}.obj')
        apply_expression_and_export(self.deca, self.shape, arr, outpath)

    def serve_websocket(self):
        # try to import websockets, if not available fall back to simple TCP
        try:
            import asyncio
            import websockets

            async def handler(ws, path):
                print('client connected')
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        self.handle_update(data)
                    except Exception as e:
                        print('Error parsing message:', e)

            async def runner():
                async with websockets.serve(handler, '0.0.0.0', self.port):
                    print(f'WebSocket server listening on :{self.port}')
                    await asyncio.Future()  # run forever

            asyncio.get_event_loop().run_until_complete(runner())
        except Exception:
            print('`websockets` package not available or failed — starting simple TCP JSON server on port', self.port)
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            s.listen(1)
            print(f'TCP server listening on :{self.port} — send JSON per-line')
            while True:
                conn, addr = s.accept()
                print('Connection from', addr)
                with conn:
                    buf = b''
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            try:
                                obj = json.loads(line.decode('utf8'))
                                self.handle_update(obj)
                            except Exception as e:
                                print('JSON parse error:', e)


def load_emoca_and_stream(path_to_models, run_name, stage, image_path, outdir='rig_output'):
    # lazy import for EMOCA loader
    try:
        from st2.utils.load import load_deca_and_data
    except Exception as e:
        raise ImportError(f'Could not import EMOCA loader from st2: {e}')

    deca_emoca, dm = load_deca_and_data(path_to_models, run_name, stage)
    print('EMOCA loaded — running inference and streaming expression updates to DECA mesh')

    # run DECA on the input image to get base shape
    deca_local, shape, base_exp, tex = run_deca_on_image(image_path, device='cpu', out_dir=outdir)

    server = BlendshapeServer(deca_local, shape, base_exp, outdir)

    # iterate test set and stream expression tensors
    testset = dm.test_set if hasattr(dm, 'test_set') else None
    if testset is None:
        print('No test set available in EMOCA data manager')
        return

    for idx, batch in enumerate(testset):
        # batch might be dict with tensors — move to cuda if needed
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda() if torch.cuda.is_available() else v
        with torch.no_grad():
            values = deca_emoca.encode(batch, training=False)
            # try decode to get final outputs
            try:
                values = deca_emoca.decode(values, training=False)
            except Exception:
                pass

        # try multiple possible keys for expressions
        exp = None
        for k in ('exp', 'expression', 'blendshape'):
            if k in values:
                exp = values[k].detach().cpu().numpy().squeeze()
                break
        if exp is None and 'ops' in values and 'exp' in values['ops']:
            exp = values['ops']['exp'].detach().cpu().numpy().squeeze()

        if exp is not None:
            server.handle_update({'expression': exp.tolist()})
        else:
            print('Could not find expression in EMOCA output for batch', idx)

        time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['deca_only', 'serve_ws', 'emoca_local'], default='deca_only')
    parser.add_argument('--image', type=str, help='Input image for DECA')
    parser.add_argument('--outdir', type=str, default='rig_output')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--path_to_models', type=str, help='EMOCA models path (for emoca_local)')
    parser.add_argument('--run_name', type=str, help='EMOCA run name')
    parser.add_argument('--stage', type=str, default='detail')
    args = parser.parse_args()

    if args.mode == 'deca_only':
        if not args.image:
            parser.error('--image required for deca_only')
        run_deca_on_image(args.image, device='auto', out_dir=args.outdir)

    elif args.mode == 'serve_ws':
        # require that rig_output/flame_params.npz exists
        rig_path = Path(args.outdir) / 'flame_params.npz'
        if not rig_path.exists():
            print(f'Could not find {rig_path}. Run DECA first (mode deca_only)')
            sys.exit(1)
        data = np.load(str(rig_path))
        shape = data['shape']
        expr = data['expression']
        deca_mod = safe_import_deca()(device='cpu')
        server = BlendshapeServer(deca_mod, shape, expr, outdir=args.outdir, port=args.port)
        server.serve_websocket()

    elif args.mode == 'emoca_local':
        if not args.path_to_models or not args.run_name or not args.image:
            parser.error('--path_to_models, --run_name and --image required for emoca_local')
        load_emoca_and_stream(args.path_to_models, args.run_name, args.stage, args.image, outdir=args.outdir)


if __name__ == '__main__':
    main()
