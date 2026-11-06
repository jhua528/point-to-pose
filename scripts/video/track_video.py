"""Click objects on a recorded RGB-D sequence, track them with Point2Pose, and
write each object's per-frame 6D pose and its reconstructed mesh.

The sequence is a folder with rgb/, depth/ (uint16 mm) and cam_K.txt -- what
prepare_video_rgbd.py makes from an RGB video, record_rgbd.py from a
RealSense, and the YCBMultiTrack layout. The pipeline is the RealSense demo's
(ModularPipeline, configs/pipeline/pipeline_test2.yaml), run frame by frame
from --start-frame to the end.

Clicking, on the start frame (SAM2's mask previewed as you click):
  left click   a point on the current object
  right click  a point that is not on it (background)
  n            done with this object; the next clicks are the next object's
  u            undo the last point          r   start over
  s            start tracking               q   quit
Or give the clicks with --points/--labels and no window opens. The clicks
used are saved to OUT/clicks.json, to run again the same way.

Output (OUT):
  obj_<i>/poses.txt     frame time tx ty tz qx qy qz qw lost -- the object's
                        pose in that frame's camera (T_cam_obj, metres)
  obj_<i>/poses.npy     the same, (frames, 4, 4); lost.npy, bool per frame
  obj_<i>/mesh.ply      the object's TSDF mesh in its own frame (mesh.glb too):
                        T_cam_obj maps it into a frame's camera. Pieces under
                        --mesh-keep-share of it dropped; mesh_raw.ply has all
  tracking.mp4          masks, boxes and axes as tracked
  mesh_overlay.mp4      the final mesh drawn with each frame's pose
  summary.json          intrinsics, clicks, per-object box, losses, mesh size

The object's frame is the box Point2Pose fits to its first view (the
oriented box of the clicked mask's depth): its centre and axes.

Run from the repository root:
  python scripts/video/track_video.py --data data/location --out results/location
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from point2pose.data_types.frame import Frame  # noqa: E402
from point2pose.pipeline.modular_pipeline import ModularPipeline  # noqa: E402
from point2pose.utils.mesh_eval import export_final_meshes_from_pipeline  # noqa: E402

WINDOW = "Point2Pose - click objects"
COLORS = [(60, 220, 60), (255, 140, 40), (60, 140, 255), (220, 60, 220), (60, 220, 220), (220, 220, 60)]
# Pieces of a fused mesh smaller than this share of its vertices are dropped:
# the TSDF fuses whatever depth the mask let in -- specks along the edges,
# a hand on the part -- and those float off the part as separate pieces.
MESH_KEEP_SHARE = 0.05
BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


class Sequence:
    def __init__(self, folder):
        self.rgb = sorted(glob.glob(os.path.join(folder, "rgb", "*.png")) + glob.glob(os.path.join(folder, "rgb", "*.jpg")))
        self.depth = sorted(glob.glob(os.path.join(folder, "depth", "*.png")))
        if not self.rgb or len(self.rgb) != len(self.depth):
            raise SystemExit(f"{folder}: needs rgb/ and depth/ with as many frames ({len(self.rgb)} and {len(self.depth)})")
        self.K = np.loadtxt(os.path.join(folder, "cam_K.txt")).reshape(3, 3)
        meta = os.path.join(folder, "meta.json")
        self.fps = json.load(open(meta)).get("fps", 30.0) if os.path.exists(meta) else 30.0

    def __len__(self):
        return len(self.rgb)

    def frame(self, index, frame_id):
        rgb = cv2.cvtColor(cv2.imread(self.rgb[index]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(self.depth[index], cv2.IMREAD_UNCHANGED).astype(np.float32)
        return Frame(id=frame_id, rgb=rgb, depth=depth, intrinsics=self.K.copy(), depth_factor=1000.0,
                     timestamp=index / self.fps)


def draw_masks(bgr, masks):
    if masks is None:
        return bgr
    masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    out = bgr.copy()
    for i in range(masks.shape[0]):
        m = masks[i, 0] > 0
        if m.shape != bgr.shape[:2] or not m.any():
            continue
        out[m] = (0.5 * out[m] + 0.5 * np.array(COLORS[i % len(COLORS)])).astype(np.uint8)
    return out


def project(points, K):
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack([K[0, 0] * points[:, 0] / z + K[0, 2], K[1, 1] * points[:, 1] / z + K[1, 2]], 1), points[:, 2] > 1e-3


def draw_box(bgr, T, extent, K, color, lost):
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    corners = (T[:3, :3] @ (signs * np.asarray(extent) / 2).T).T + T[:3, 3]
    uv, ok = project(corners, K)
    color = (0, 0, 255) if lost else color
    for a, b in BOX_EDGES:
        if ok[a] and ok[b]:
            cv2.line(bgr, tuple(np.int32(uv[a])), tuple(np.int32(uv[b])), color, 2, cv2.LINE_AA)
    axis = 0.5 * float(np.min(extent)) if np.min(extent) > 0 else 0.03
    ends = np.vstack([T[:3, 3]] + [T[:3, 3] + axis * T[:3, k] for k in range(3)])
    uv, ok = project(ends, K)
    if ok.all():
        for k, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
            cv2.line(bgr, tuple(np.int32(uv[0])), tuple(np.int32(uv[k + 1])), c, 2, cv2.LINE_AA)


def click_objects(pipeline, rgb):
    """The person's clicks on `rgb`, per object: (points, labels), or None if quit."""
    state = {"points": [[]], "labels": [[]], "dirty": True, "masks": None}

    def on_mouse(event, x, y, _flags, _param):
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            state["points"][-1].append([int(x), int(y)])
            state["labels"][-1].append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)
            state["dirty"] = True

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        raise SystemExit(f"no window can open here ({exc}); give the clicks with --points and --labels")
    cv2.setMouseCallback(WINDOW, on_mouse)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    while True:
        groups = [(p, l) for p, l in zip(state["points"], state["labels"]) if p and any(l)]
        if state["dirty"]:
            state["masks"] = (pipeline.preview_user_masks(rgb, [p for p, _ in groups], [l for _, l in groups])
                              if groups else None)
            state["dirty"] = False
        view = draw_masks(bgr, state["masks"])
        for i, (pts, lbls) in enumerate(zip(state["points"], state["labels"])):
            for (x, y), lbl in zip(pts, lbls):
                cv2.circle(view, (x, y), 5, COLORS[i % len(COLORS)] if lbl else (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(view, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
        text = (f"object {len(state['points']) - 1}: left = on it, right = not on it | "
                "n next object  u undo  r reset  s start  q quit")
        cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            cv2.destroyWindow(WINDOW)
            return None
        if key == ord("n") and state["points"][-1]:
            if any(state["labels"][-1]):
                state["points"].append([])
                state["labels"].append([])
            else:
                print("this object needs a left click on it first")
        elif key == ord("u"):
            if not state["points"][-1] and len(state["points"]) > 1:
                state["points"].pop()
                state["labels"].pop()
            if state["points"][-1]:
                state["points"][-1].pop()
                state["labels"][-1].pop()
                state["dirty"] = True
        elif key == ord("r"):
            state.update(points=[[]], labels=[[]], dirty=True)
        elif key == ord("s"):
            if groups:
                cv2.destroyWindow(WINDOW)
                return [p for p, _ in groups], [l for _, l in groups]
            print("click at least one object first")


def clean_mesh(mesh, keep_share=MESH_KEEP_SHARE):
    """`mesh` without its pieces smaller than `keep_share` of its vertices."""
    import trimesh
    pieces = mesh.split(only_watertight=False)
    keep = [p for p in pieces if len(p.vertices) >= keep_share * len(mesh.vertices)]
    if not keep or len(keep) == len(pieces):
        return mesh
    return trimesh.util.concatenate(keep)


def write_video(path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with rgb/, depth/ (uint16 mm), cam_K.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("-c", "--config", default=os.path.join(ROOT, "configs", "pipeline", "pipeline_test2.yaml"))
    ap.add_argument("--start-frame", type=int, default=0, help="the frame clicked on; tracking runs from it to the end")
    ap.add_argument("--max-frames", type=int, default=0, help="track at most this many frames (0: to the end)")
    ap.add_argument("--points", default=None,
                    help='clicks per object, JSON, e.g. \'[[[310,120],[300,90]],[[150,240]]]\' (no window then)')
    ap.add_argument("--labels", default=None, help="1 (on the object) / 0 (not) per point, JSON; default all 1")
    ap.add_argument("--show", action="store_true", help="show the tracking as it runs")
    ap.add_argument("--mesh-keep-share", type=float, default=MESH_KEEP_SHARE,
                    help="drop mesh pieces with fewer than this share of its vertices (0: keep all; "
                         "the uncleaned mesh is kept as mesh_raw.ply)")
    args = ap.parse_args()

    seq = Sequence(args.data)
    if not 0 <= args.start_frame < len(seq):
        raise SystemExit(f"--start-frame must be within 0-{len(seq) - 1}")
    os.makedirs(args.out, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.pipeline.params.save_pose = False          # this script writes the poses, in the camera
    cfg.pipeline.params.save_meta_data = False
    cfg.pipeline.params.debug_level = 0
    cfg.pipeline.params.debug_dir = os.path.join(os.path.abspath(args.out), "debug")
    pipeline = ModularPipeline(cfg)

    first = seq.frame(args.start_frame, 0)
    if args.points:
        points = json.loads(args.points)
        labels = json.loads(args.labels) if args.labels else [[1] * len(p) for p in points]
    else:
        clicked = click_objects(pipeline, first.rgb)
        if clicked is None:
            print("quit")
            return
        points, labels = clicked
    with open(os.path.join(args.out, "clicks.json"), "w") as f:
        json.dump({"start_frame": args.start_frame, "points": points, "labels": labels}, f)
    print(f"clicks: --start-frame {args.start_frame} --points '{json.dumps(points)}' --labels '{json.dumps(labels)}'")

    pipeline.add_user_points(points, labels)
    end = len(seq) if not args.max_frames else min(len(seq), args.start_frame + args.max_frames)
    poses, lost, overlay = [], [], []
    started = time.time()
    for index in range(args.start_frame, end):
        frame = first if index == args.start_frame else seq.frame(index, index - args.start_frame)
        pipeline.step(frame)
        objects = pipeline.objects
        poses.append(np.stack([np.asarray(o.pose, dtype=np.float64) @ np.asarray(o.init_pose, dtype=np.float64)
                               for o in objects]))
        lost.append([bool(getattr(o, "lost", False)) for o in objects])
        view = draw_masks(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), getattr(frame, "mask", None))
        for i, o in enumerate(objects):
            extent = np.asarray(o.init_bbox.extent) if getattr(o, "init_bbox", None) is not None else [0.05] * 3
            draw_box(view, poses[-1][i], extent, seq.K, COLORS[i % len(COLORS)], lost[-1][i])
        cv2.putText(view, f"frame {index}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        overlay.append(view)
        if args.show:
            cv2.imshow("Point2Pose - tracking", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        done = index - args.start_frame + 1
        if done % 10 == 0 or index == end - 1:
            rate = done / (time.time() - started)
            print(f"[track] frame {index} ({done}/{end - args.start_frame}, {rate:.2f} fps); lost {lost[-1]}",
                  flush=True)
    if args.show:
        cv2.destroyAllWindows()
    write_video(os.path.join(args.out, "tracking.mp4"), overlay, seq.fps)

    poses = np.stack(poses)          # (frames, objects, 4, 4)
    lost = np.asarray(lost)
    mesh_dir = os.path.join(args.out, "debug", "mesh_cam0")
    exported = export_final_meshes_from_pipeline(pipeline, mesh_dir)
    summary = {"data": os.path.abspath(args.data), "K": seq.K.tolist(), "fps": seq.fps,
               "start_frame": args.start_frame, "frames": int(len(poses)), "points": points, "labels": labels,
               "objects": []}
    mesh_points = {}
    import trimesh
    for i, o in enumerate(pipeline.objects):
        folder = os.path.join(args.out, f"obj_{i}")
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "poses.npy"), poses[:, i])
        np.save(os.path.join(folder, "lost.npy"), lost[:, i])
        with open(os.path.join(folder, "poses.txt"), "w") as f:
            f.write("# frame time tx ty tz qx qy qz qw lost  (T_cam_obj: object frame -> this frame's camera, m)\n")
            for k, T in enumerate(poses[:, i]):
                q = Rotation.from_matrix(T[:3, :3]).as_quat()
                f.write(f"{args.start_frame + k} {(args.start_frame + k) / seq.fps:.4f} "
                        + " ".join(f"{v:.6f}" for v in (*T[:3, 3], *q)) + f" {int(lost[k, i])}\n")
        entry = {"index": i, "init_pose_cam": np.asarray(o.init_pose).tolist(),
                 "box_extent_m": (np.asarray(o.init_bbox.extent).tolist()
                                  if getattr(o, "init_bbox", None) is not None else None),
                 "lost_frames": int(lost[:, i].sum()), "mesh": None}
        ply = exported.get(i)
        if ply and os.path.exists(ply):
            # The TSDF is kept in the start frame's camera; into the object's own frame.
            raw = trimesh.load(ply, process=False)
            raw.apply_transform(np.linalg.inv(np.asarray(o.init_pose, dtype=np.float64)))
            raw.export(os.path.join(folder, "mesh_raw.ply"))
            mesh = clean_mesh(raw, args.mesh_keep_share)
            mesh.export(os.path.join(folder, "mesh.ply"))
            mesh.export(os.path.join(folder, "mesh.glb"))
            entry["mesh"] = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
                             "extent_m": mesh.extents.tolist(), "raw_vertices": int(len(raw.vertices))}
            step = max(1, len(mesh.vertices) // 3000)
            mesh_points[i] = np.asarray(mesh.vertices)[::step]
        summary["objects"].append(entry)
        print(f"[out] object {i}: {len(poses)} poses, lost on {entry['lost_frames']}, "
              + (f"mesh {entry['mesh']['vertices']} vertices, {np.round(entry['mesh']['extent_m'], 3).tolist()} m"
                 if entry["mesh"] else "no mesh (too little was fused)"), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # The final meshes drawn with each frame's pose: where they sit is where the poses put them.
    if mesh_points:
        frames = []
        for k in range(len(poses)):
            view = cv2.cvtColor(seq.frame(args.start_frame + k, k).rgb, cv2.COLOR_RGB2BGR)
            for i, pts in mesh_points.items():
                T = poses[k, i]
                uv, ok = project((T[:3, :3] @ pts.T).T + T[:3, 3], seq.K)
                color = (0, 0, 255) if lost[k, i] else COLORS[i % len(COLORS)]
                for u, v in uv[ok].astype(int):
                    if 0 <= u < view.shape[1] and 0 <= v < view.shape[0]:
                        cv2.circle(view, (u, v), 1, color, -1)
            frames.append(view)
        write_video(os.path.join(args.out, "mesh_overlay.mp4"), frames, seq.fps)
    print(f"[out] written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
