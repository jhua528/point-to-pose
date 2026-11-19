"""Replay each scene in assets/test/scene through pose_server.py and render the poses.

Each scene folder holds what pose_client.py would send: order.json lists the
init request's parts (meta_init.json, the frame, each mesh) and the track
request's (meta_track.json, the frame). For each scene: one init, then
--track-frames track requests on the same frame, so the server's background
mesh retries get a chance; the last reply is rendered into OUT/<scene>.png
(each mesh at its published pose, with its x/y/z axes), and the replies and
timings go to OUT/<scene>.json. The scene folders are only read.

Start a server of its own on a spare port first (5555 is the live one), then
replay, from the repository root:
  python -u scripts/remote/pose_server.py -b tcp://127.0.0.1:5599 &
  python scripts/remote/replay_scenes.py --out /tmp/replay --only PP-S1-1 \
      --same-scale --timeout-ms 15000
--same-scale: the tuned scenes' meshes come from one reconstruction, as
pose_client.py --same-scale-meshes says. --timeout-ms: what pose_client.py
sends with each track request (15000 by default there); the server then works
the frames on its track thread, as it does for the client. Without it each
frame is worked on the server's main thread.
"""

import argparse
import json
import os
import sys
import time
import zlib

import cv2
import numpy as np
import zmq

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOX_EDGES = [(i, i | 1 << k) for i in range(8) for k in range(3) if not i >> k & 1]
OBSTACLE_COLOR = (200, 200, 200)
COLORS = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (220, 60, 220), (60, 60, 255), (255, 255, 0)]
# BGR: x red, y green, z blue. An axis is half the object's longest side, within these.
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
AXIS_MIN_M, AXIS_MAX_M = 0.04, 0.12


def read_mesh(path, spec):
    raw = zlib.decompress(open(path, "rb").read())
    nv, nf = int(spec["vertices"]), int(spec["faces"])
    v = np.frombuffer(raw, "<f4", 3 * nv).reshape(nv, 3).astype(np.float64)
    f = np.frombuffer(raw, "<i4", 3 * nf, offset=12 * nv).reshape(nf, 3)
    return v, f


def request(sock, parts, timeout_ms):
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.send_multipart(parts)
    return sock.recv_json()


def project(P, K):
    z = np.maximum(P[:, 2], 1e-6)
    return np.stack([P[:, 0] / z * K[0, 0] + K[0, 2], P[:, 1] / z * K[1, 1] + K[1, 2]], -1)


def draw_axes(img, T, K, length):
    """The frame T's x, y and z axes (red, green, blue), `length` metres long."""
    if T[2, 3] <= 0.01:
        return
    ends = T[:3, 3] + length * T[:3, :3].T
    pts = np.vstack([T[:3, 3], ends])
    if np.any(pts[:, 2] <= 0.01):
        return
    uv = np.round(project(pts, K)).astype(np.int32)
    o = tuple(int(v) for v in uv[0])
    for k, (colour, label) in enumerate(zip(AXIS_COLORS, "xyz")):
        tip = tuple(int(v) for v in uv[k + 1])
        cv2.arrowedLine(img, o, tip, (0, 0, 0), 4, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(img, o, tip, colour, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(img, label, (tip[0] + 3, tip[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (tip[0] + 3, tip[1] + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    cv2.circle(img, o, 3, (255, 255, 255), -1, cv2.LINE_AA)


def render(scene_dir, names, meshes, K, reply, title, out_dir):
    rgb = cv2.imread(os.path.join(scene_dir, "rgb.jpg"))
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    lines = [title, "axes at each published pose (mesh frame, else box frame): x red, y green, z blue"]
    bboxes = (reply or {}).get("bboxes") or [None] * len(names)
    lost = (reply or {}).get("lost") or [None] * len(names)
    errors = (reply or {}).get("model_errors") or [None] * len(names)

    # Painter's algorithm over every object's triangles, far to near.
    tris, tri_color = [], []
    for i, (name, box) in enumerate(zip(names, bboxes)):
        if box and box.get("model_pose") is not None and meshes[i] is not None:
            v, f = meshes[i]
            T = np.asarray(box["model_pose"], np.float64)
            L = np.asarray(box.get("model_linear", np.eye(3)), np.float64)
            P = (v @ L.T) @ T[:3, :3].T + T[:3, 3]
            tri = P[f]
            keep = np.all(tri[:, :, 2] > 0.01, axis=1)
            tris.append(tri[keep])
            tri_color.append(np.full(keep.sum(), i))
    mask_by_obj = {}
    if tris:
        tri = np.concatenate(tris)
        col = np.concatenate(tri_color)
        order = np.argsort(-tri[:, :, 2].mean(1))
        shade = np.zeros((h, w, 3), np.uint8)
        owner = np.full((h, w), -1, np.int32)
        for t, c in zip(tri[order], col[order]):
            uv = np.round(project(t, K)).astype(np.int32)
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n /= max(np.linalg.norm(n), 1e-12)
            light = 0.55 + 0.45 * abs(n @ (-t.mean(0) / np.linalg.norm(t.mean(0))))
            colour = tuple(int(x * light) for x in COLORS[c % len(COLORS)])
            cv2.fillConvexPoly(shade, uv, colour)
            cv2.fillConvexPoly(owner, uv, int(c))
        on = owner >= 0
        overlay[on] = (0.45 * overlay[on] + 0.55 * shade[on]).astype(np.uint8)
        for c in np.unique(col):
            mask_by_obj[int(c)] = (owner == c).astype(np.uint8)

    for i, name in enumerate(names):
        colour = COLORS[i % len(COLORS)]
        box = bboxes[i] if i < len(bboxes) else None
        if i in mask_by_obj:
            contours, _ = cv2.findContours(mask_by_obj[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, colour, 2)
        if box:
            # The published pose: the mesh's frame where there is one, else the box's.
            T = np.asarray(box["model_pose"] if box.get("model_pose") is not None else box["pose"], np.float64)
            length = float(np.clip(0.5 * max(box["extent"]), AXIS_MIN_M, AXIS_MAX_M))
            draw_axes(overlay, T, K, length)
        if box and box.get("corners"):
            C = np.asarray(box["corners"], np.float64)
            if np.all(C[:, 2] > 0.01):
                uv = np.round(project(C, K)).astype(np.int32)
                for a, b in BOX_EDGES:
                    cv2.line(overlay, tuple(uv[a]), tuple(uv[b]), colour, 1, cv2.LINE_AA)
                top = uv[np.argmin(uv[:, 1])]
                cv2.putText(overlay, name, (int(top[0]) - 20, max(int(top[1]) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, name, (int(top[0]) - 20, max(int(top[1]) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        if box is None:
            status = "no box"
        elif box.get("model_pose") is not None:
            ext = " x ".join(f"{e * 1000:.0f}" for e in box["extent"])
            status = (f"mesh {box.get('model_quality', 'fit')}, scale {box.get('model_scale', 1):.3f}"
                      f"{'' if box.get('model_measured') else ' (unmeasured)'}, box {ext} mm")
        elif meshes[i] is not None:
            status = "mesh pose not localized; points' box only"
        else:
            ext = " x ".join(f"{e * 1000:.0f}" for e in box["extent"])
            status = f"points' box {ext} mm (no mesh)"
        if i < len(lost) and lost[i]:
            status += " [lost]"
        if i < len(errors) and errors[i]:
            status += f" | {errors[i]}"
        lines.append((f"{name}: {status}", colour))

    # Obstacles the init was asked to find (meta "discover"): their boxes, grey, named as pose_client.py names them.
    for obstacle in (reply or {}).get("obstacles") or []:
        box, name = obstacle.get("bbox"), f"obstacle_{obstacle.get('id')}"
        colour = (0, 0, 255) if obstacle.get("lost") else OBSTACLE_COLOR
        if box and box.get("corners"):
            C = np.asarray(box["corners"], np.float64)
            if np.all(C[:, 2] > 0.01):
                uv = np.round(project(C, K)).astype(np.int32)
                for a, b in BOX_EDGES:
                    cv2.line(overlay, tuple(uv[a]), tuple(uv[b]), colour, 1, cv2.LINE_AA)
                top = uv[np.argmin(uv[:, 1])]
                cv2.putText(overlay, name, (int(top[0]) - 30, max(int(top[1]) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, name, (int(top[0]) - 30, max(int(top[1]) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        ext = " x ".join(f"{e * 1000:.0f}" for e in box["extent"]) if box else "?"
        lines.append((f"{name}: box {ext} mm" + (" [lost]" if obstacle.get("lost") else ""), colour))

    legend_h = 18 * len(lines) + 10
    canvas = np.full((h + legend_h, w, 3), 30, np.uint8)
    canvas[:h] = overlay
    y = h + 18
    for line in lines:
        text, colour = (line, (255, 255, 255)) if isinstance(line, str) else line
        cv2.putText(canvas, text[:105], (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        y += 18
    cv2.imwrite(os.path.join(out_dir, os.path.basename(scene_dir.rstrip("/")) + ".png"), canvas)


def person_clicks(scene_dir, names, meshes, K, answer_from):
    """What a person asked to click each object would click: three pixels on
    where it truly is and shows -- truth.json's pose, else the final mesh pose
    of the replies in `answer_from` (a run trusted to be right) -- {name: clicks}."""
    meta = json.load(open(os.path.join(scene_dir, "meta_init.json")))
    h, w = meta["depth_shape"] if "depth_shape" in meta else cv2.imread(os.path.join(scene_dir, "rgb.jpg")).shape[:2]
    raw = zlib.decompress(open(os.path.join(scene_dir, "depth.f32.zlib"), "rb").read())
    depth = np.frombuffer(raw, "<f4").reshape(h, w) if len(raw) == 4 * h * w else None
    poses = {}
    truth_path = os.path.join(scene_dir, "truth.json")
    if os.path.exists(truth_path):
        for t in json.load(open(truth_path)):
            v = np.load(os.path.join(scene_dir, f"true_{t['object']}.npy")).astype(np.float64)
            T = np.asarray(t["T"], np.float64)
            poses[t["object"]] = v @ T[:3, :3].T + T[:3, 3]
    elif answer_from:
        record = json.load(open(os.path.join(answer_from, os.path.basename(scene_dir.rstrip("/")) + ".json")))
        oks = [r for r in record.get("track", []) if r.get("ok")]
        last = oks[-1] if oks else record.get("init") or {}
        for i, (name, box) in enumerate(zip(names, last.get("bboxes") or [])):
            if box and box.get("model_pose") is not None and meshes[i] is not None:
                T = np.asarray(box["model_pose"], np.float64)
                P = meshes[i][0] @ np.asarray(box["model_linear"], np.float64).T
                poses[name] = P @ T[:3, :3].T + T[:3, 3]
    out = {}
    for name, P in poses.items():
        P = P[P[:, 2] > 0.05]
        uv = np.round(project(P, K)).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        uv, z = uv[ok], P[ok, 2]
        if depth is not None:
            d = depth[uv[:, 1], uv[:, 0]]
            uv = uv[(d > 0) & (np.abs(d - z) < 0.01)]          # the part's own, shown
        if len(uv) < 3:
            continue
        mid = np.median(uv, axis=0)
        first = uv[np.argmin(np.linalg.norm(uv - mid, axis=1))]
        axis = np.linalg.svd(uv - uv.mean(0), full_matrices=False)[2][0]
        along = (uv - first) @ axis
        picks = [first, uv[np.argmin(np.abs(along - np.percentile(along, 25)))],
                 uv[np.argmin(np.abs(along - np.percentile(along, 75)))]]
        out[name] = [[int(u), int(v)] for u, v in picks]
    return out


def load_scene(scene_dir, init_list="init"):
    order = json.load(open(os.path.join(scene_dir, "order.json")))
    order["init"] = order.get(init_list) or order["init"]
    meta_init = json.load(open(os.path.join(scene_dir, order["init"][0])))
    names = order["objects"]
    K = np.asarray(meta_init["K"], np.float64).reshape(3, 3)
    specs = meta_init.get("models") or [None] * len(names)
    mesh_files = iter(order["init"][3:])
    meshes = [read_mesh(os.path.join(scene_dir, next(mesh_files)), s) if s else None for s in specs]
    return order, names, K, meshes


def render_saved(scene_dir, out_dir):
    """Re-render a scene from its saved replies, without the server."""
    name = os.path.basename(scene_dir.rstrip("/"))
    _, names, K, meshes = load_scene(scene_dir)
    record = json.load(open(os.path.join(out_dir, f"{name}.json")))
    init = record.get("init") or {}
    if not init.get("ok"):
        title = f"{name}: init failed ({record.get('init_s')} s): {init.get('error', record.get('error'))}"
        render(scene_dir, names, meshes, K, None, title, out_dir)
        return title
    oks = [r for r in record.get("track", []) if r.get("ok")]
    reply = oks[-1] if oks else init
    title = (f"{name}: init {record['init_s']} s, then {len(record.get('track', []))} track frames; "
             f"frame {reply.get('frame_id')}")
    render(scene_dir, names, meshes, K, reply, title, out_dir)
    return title


def run_scene(ctx, server, scene_dir, track_frames, out_dir, same_scale=False, true_size=False, drop_colours=False,
              answer=None, timeout_ms=None, init_list="init"):
    name = os.path.basename(scene_dir.rstrip("/"))
    order, names, K, meshes = load_scene(scene_dir, init_list)

    init_parts = [open(os.path.join(scene_dir, p), "rb").read() for p in order["init"]]
    track_parts = [open(os.path.join(scene_dir, p), "rb").read() for p in order["track"]]
    if timeout_ms:
        # What pose_client.py sends once initialized: the server then works the
        # frame on its track thread and answers within SERVER_REPLY_SHARE of it.
        track_parts[0] = json.dumps({**json.loads(track_parts[0]), "timeout_ms": int(timeout_ms)}).encode()
    if same_scale:
        # What pose_client.py --same-scale-meshes sends.
        init_parts[0] = json.dumps({**json.loads(init_parts[0]), "same_scale_meshes": True}).encode()
    if drop_colours:
        # The same meshes without their colours: the bytes after the faces dropped.
        meta = json.loads(init_parts[0])
        k = 3
        for spec in meta.get("models") or []:
            if spec is None:
                continue
            if spec.pop("colours", False):
                raw = zlib.decompress(init_parts[k])
                init_parts[k] = zlib.compress(raw[:12 * (int(spec["vertices"]) + int(spec["faces"]))])
            k += 1
        init_parts[0] = json.dumps(meta).encode()
    if true_size:
        # What pose_client.py --true-size-meshes sends, at the scale truth.json says the mesh was sent at.
        meta = json.loads(init_parts[0])
        truth = json.load(open(os.path.join(scene_dir, "truth.json")))
        for spec, t in zip(meta["models"], truth):
            if spec is not None:
                spec["known_scale"] = float(t["k"])
        init_parts[0] = json.dumps(meta).encode()
    record = {"scene": name, "objects": names}
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(server)
    reply, title = None, ""
    try:
        t0 = time.time()
        reply = request(sock, init_parts, 900000)
        record["init_s"] = round(time.time() - t0, 1)
        record["init"] = reply
        if not reply.get("ok") and reply.get("need_click") and answer is not None:
            # The person asked to click: as pose_client.py's on_click() sends it.
            record["asked"] = reply
            clicks = person_clicks(scene_dir, names, meshes, K, answer)
            meta = json.loads(init_parts[0])
            sure = reply.get("points") or [[] for _ in names]
            meta["points"] = [list(p) for p in sure]
            meta["labels"] = [[1] * len(p) for p in sure]
            for req in reply["need_click"]:
                asked = names[int(req["object"])]
                meta["points"][int(req["object"])] = clicks.get(asked, [])
                meta["labels"][int(req["object"])] = [1] * len(clicks.get(asked, []))
            record["answered"] = {names[int(r["object"])]: clicks.get(names[int(r["object"])]) for r in reply["need_click"]}
            t1 = time.time()
            reply = request(sock, [json.dumps(meta).encode()] + init_parts[1:], 900000)
            record["init_s"] = round(time.time() - t0, 1)
            record["answer_s"] = round(time.time() - t1, 1)
            record["init"] = reply
        if not reply.get("ok"):
            title = f"{name}: init failed ({record['init_s']} s): {reply.get('error')}"
        else:
            last = reply
            for k in range(track_frames):
                r = request(sock, track_parts, 120000)
                record.setdefault("track", []).append(r)
                if r.get("ok"):
                    last = r
            reply = last
            title = (f"{name}: init {record['init_s']} s, then {track_frames} track frames; "
                     f"frame {reply.get('frame_id')}")
    except zmq.Again:
        title = f"{name}: timed out"
        record["error"] = "timeout"
    finally:
        sock.close()
    render(scene_dir, names, meshes, K, reply if reply and reply.get("ok") else None, title, out_dir)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump(record, fh)
    return title, reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="tcp://127.0.0.1:5599")
    ap.add_argument("--scenes", default=os.path.join(REPO, "assets", "test", "scene"))
    ap.add_argument("--track-frames", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--same-scale", action="store_true", help="tell the server the meshes come from one reconstruction")
    ap.add_argument("--answer-clicks", nargs="?", const="", default=None,
                    help="answer the server's need_click as a person would: truth.json's poses, else those of "
                         "the replies in this folder")
    ap.add_argument("--skip-done", action="store_true", help="leave out scenes already replied to in --out")
    ap.add_argument("--drop-colours", action="store_true", help="send the meshes without their colours")
    ap.add_argument("--true-size", action="store_true", help="tell the server each mesh's scale (truth.json's k)")
    ap.add_argument("--render-only", action="store_true", help="re-render from the saved replies")
    ap.add_argument("--timeout-ms", type=int, default=None,
                    help="send timeout_ms with each track request, as pose_client.py does (its default 15000)")
    ap.add_argument("--init-list", default="init",
                    help="which of order.json's init lists to send (assets/test/scene/README.md): 'init' (the VLM's "
                         "clicks, as the runs send them), 'init_located' (no clicks: located by the meshes) or "
                         "'init_clean' (the VLM's clicks off the part by Gazebo's truth left out); 'init' where "
                         "the scene has no such list")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ctx = zmq.Context()
    scenes = sorted(d for d in os.listdir(args.scenes) if os.path.isdir(os.path.join(args.scenes, d)))
    for scene in scenes:
        if args.only and scene not in args.only:
            continue
        if args.skip_done and os.path.exists(os.path.join(args.out, f"{scene}.json")):
            continue
        if args.render_only:
            print(f"[render] {render_saved(os.path.join(args.scenes, scene), args.out)}", flush=True)
            continue
        title, reply = run_scene(ctx, args.server, os.path.join(args.scenes, scene), args.track_frames, args.out, args.same_scale,
                                 args.true_size, args.drop_colours, args.answer_clicks, args.timeout_ms,
                                 args.init_list)
        print(f"[replay] {title}", flush=True)
        if reply and reply.get("ok"):
            for n, b, l in zip(json.load(open(os.path.join(args.scenes, scene, "order.json")))["objects"],
                               reply["bboxes"], reply["lost"]):
                print(f"[replay]   {n}: " + ("no box" if b is None else
                      f"{b.get('model_quality') or 'points box'} lost={l}"), flush=True)


if __name__ == "__main__":
    sys.exit(main())

