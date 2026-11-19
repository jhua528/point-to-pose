#!/usr/bin/env python3
"""Send a captured request (README.md) to a pose server, part for part in the
order pose_client sends it, and print the reply.

    python3 send.py PP-S1-2 init  [--server tcp://172.26.177.251:5555]
    python3 send.py PP-S1-2 init_located     # the same init, the VLM's points left out
    python3 send.py PP-S1-2 init_clean       # the same init, the VLM's points off the part (by truth) left out
    python3 send.py PP-S1-2 track
    python3 send.py PP-S1-2 init --no-truth  # without the comparison with truth.json

After the reply, each object's model_pose and model_linear are compared with
the capture's truth.json (Gazebo's, README.md): where the mesh origin is off,
how far it is turned, and the mean distance of the mesh's vertices
(ADD, each at its own scale) -- to the nearest of the poses the part looks
alike in.

An init rebuilds the server's pipeline: a tracking session running on that
server is replaced by this one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zmq

ap = argparse.ArgumentParser()
ap.add_argument("capture", help="a folder beside this script, e.g. PP-S1-2")
ap.add_argument("cmd", choices=("init", "init_located", "init_clean", "track"))
ap.add_argument("--server", default="tcp://172.26.177.251:5555")
ap.add_argument("--timeout-s", type=float, default=300.0)
ap.add_argument("--no-truth", action="store_true", help="do not compare the reply with truth.json")
args = ap.parse_args()

folder = Path(__file__).resolve().parent / args.capture
order = json.loads((folder / "order.json").read_text())[args.cmd]
meta = json.dumps(json.loads((folder / order[0]).read_text())).encode()
parts = [meta] + [(folder / name).read_bytes() for name in order[1:]]

sock = zmq.Context().socket(zmq.REQ)
sock.setsockopt(zmq.LINGER, 0)
sock.setsockopt(zmq.RCVTIMEO, int(args.timeout_s * 1000))
sock.connect(args.server)
sock.send_multipart(parts)
try:
    reply = sock.recv_json()
except zmq.Again:
    raise SystemExit(f"no reply in {args.timeout_s:.0f} s")
for box in reply.get("bboxes") or []:
    if isinstance(box, dict):
        box.pop("surface", None)
        box.pop("surface_views", None)
print(json.dumps(reply, indent=1))


def mesh_vertices(path, count):
    import zlib
    return np.frombuffer(zlib.decompress(path.read_bytes())[:count * 12], np.float32).reshape(-1, 3).astype(float)


def against_truth(reply):
    """Each object's fit in `reply` against truth.json, printed."""
    truth_path = folder / "truth.json"
    if args.no_truth or not truth_path.exists():
        return
    truth = json.loads(truth_path.read_text())
    meta_init = json.loads((folder / "meta_init.json").read_text())
    boxes = reply.get("bboxes") or []
    print("\n--- against truth.json (Gazebo's, offline only) ---")
    for entry in truth["objects"]:
        name, index = entry["name"], entry["index"]
        if "missing" in entry:
            print(f"{name}: no truth ({entry['missing']})")
            continue
        box = boxes[index] if index < len(boxes) and isinstance(boxes[index], dict) else {}
        if box.get("model_pose") is None:
            print(f"{name}: no model_pose in the reply")
            continue
        est = np.asarray(box["model_pose"], float)
        lin_est = np.asarray(box.get("model_linear") or np.eye(3) * float(box.get("model_scale", 1.0)), float)
        lin_true = np.asarray(entry["model_linear"], float)
        x = mesh_vertices(folder / entry["mesh_file"], meta_init["models"][index]["vertices"])
        P_est = (x @ lin_est.T) @ est[:3, :3].T + est[:3, 3]
        rows = []
        for k, T in enumerate([entry["model_pose"]] + entry.get("alike_model_poses", [])):
            T = np.asarray(T, float)
            P_true = (x @ lin_true.T) @ T[:3, :3].T + T[:3, 3]
            R = T[:3, :3].T @ est[:3, :3]
            deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
            rows.append((float(np.linalg.norm(P_est - P_true, axis=1).mean() * 1000),
                         float(np.linalg.norm(est[:3, 3] - T[:3, 3]) * 1000), deg, k))
        add, trans, deg, k = min(rows)
        scale = float(np.cbrt(abs(np.linalg.det(lin_est)))) / entry["model_scale"]
        print(f"{name}: ADD {add:.1f} mm, origin off {trans:.1f} mm, turned {deg:.1f} deg, "
              f"scale x{scale:.3f} of the truth's" + (f" (nearest: alike pose {k})" if k else ""))


against_truth(reply)
