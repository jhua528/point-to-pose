"""ZeroMQ inference server exposing ModularPipeline to a remote ROS2 client.

Each object's bounding box is fit here, once, on the init frame, and then
carried by the tracked motion. The fit uses nothing about the scene -- no bench
height, no gravity, no object shape -- only the mask and the depth:

  1. The surface the object sits on is fit (RANSAC) to a ring of pixels just
     outside the mask. Mask pixels on that surface are the background the mask
     spilled onto at the object's edge, and are dropped.
  2. What is left gets a minimal-volume box. PCA, the usual shortcut, follows
     how the points are spread; one view sees an L-shaped cloud (a top and a
     side), and PCA tilts the box across it, making it too wide and askew.
"""

import argparse
import json
import time
import zlib
import traceback

import cv2
import numpy as np
import zmq
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull

from point2pose.data_types.frame import Frame
from point2pose.pipeline.modular_pipeline import ModularPipeline


def decode_rgb(buf):
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_depth(buf, meta):
    # Both encodings are lossless; float32 metres travel raw because PNG is integer-only.
    if meta.get("depth_enc") == "raw_f32":
        h, w = meta["depth_shape"]
        return np.frombuffer(zlib.decompress(buf), np.float32).reshape(h, w)
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)


def anchor_from_points(points, labels, depth, K, depth_factor, radius=3):
    """Object origin on the init frame: the clicked surface point in camera coordinates.

    The pipeline reports each object's motion since the init frame, with the
    object model stored in camera coordinates, so its origin is the camera
    centre. Composing that motion with this anchor turns it into the object's
    pose in the camera frame.
    """
    metres = depth / depth_factor
    xyz = []
    for (u, v), label in zip(points, labels):
        if not label:
            continue
        u, v = int(round(u)), int(round(v))
        patch = metres[max(v - radius, 0):v + radius + 1, max(u - radius, 0):u + radius + 1]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size:
            z = float(np.median(valid))
            xyz.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    if not xyz:
        raise ValueError(f"no valid depth under the picked points {points}")
    anchor = np.eye(4)
    anchor[:3, 3] = np.median(xyz, axis=0)
    return anchor


# Unit-cube corner signs, ordered so bit 0/1/2 of the index flips x/y/z.
_CORNER_SIGNS = np.array([[(i >> k & 1) * 2 - 1 for k in range(3)] for i in range(8)], np.float64)


def as_numpy(value):
    """float64 array from numpy or a (possibly CUDA) torch tensor."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, np.float64)


# Box fitting. Camera frame and pixels throughout; nothing about this scene.
# The ring the support surface is fit to starts clear of the band at the mask
# edge where spill lives, and is wide enough to hold a plane's worth of pixels.
PLANE_RING_PX = (8, 25)
# RANSAC inlier distance; also how far in front of the surface a point must be
# to count as the object. Parts thinner than this fall back to the raw mask.
PLANE_TOL_M = 0.005
PLANE_RANSAC_ITERS = 300
# Below this share of ring pixels on the best plane there is no clear surface
# (clutter, an object held in the air), and nothing is dropped.
PLANE_MIN_INLIERS = 0.3
MIN_BOX_POINTS = 30
# Boxes within this much of the smallest volume count as ties; see min_obb().
BOX_VOLUME_SLACK = 0.1
# Fewer points than this share of the mask left in front of the surface means
# the part lies flat on it: depth cannot tell it from its support.
FLAT_KEEP_FRACTION = 0.2


def backproject(metres, K):
    """Camera-frame point for every pixel, and which pixels have depth."""
    h, w = metres.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = np.isfinite(metres) & (metres > 0)
    z = np.where(valid, metres, 0.0)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1), valid


def support_plane(mask, cam, valid, rng):
    """(normal, point) of the surface around the mask, normal facing away from
    the camera; None when the ring holds no clear plane."""
    def grown(radius):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    inner, outer = PLANE_RING_PX
    pts = cam[grown(outer) & ~grown(inner) & valid]
    if len(pts) < MIN_BOX_POINTS:
        return None
    best_count, best = 0, None
    for _ in range(PLANE_RANSAC_ITERS):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        count = int(np.sum(np.abs((pts - a) @ n) < PLANE_TOL_M))
        if count > best_count:
            best_count, best = count, (n, a)
    if best is None or best_count < PLANE_MIN_INLIERS * len(pts):
        return None
    n, a = best
    inliers = pts[np.abs((pts - a) @ n) < PLANE_TOL_M]
    centre = inliers.mean(0)
    n = np.linalg.svd(inliers - centre, full_matrices=False)[2][2]
    return (n if n @ centre > 0 else -n), centre


def min_obb(points):
    """Minimal-volume oriented box: (R, centre, extent), R's columns the box axes.

    Each convex-hull face normal is tried as one axis, the smallest rectangle
    of the points projected across it (rotating calipers) giving the other two:
    the usual approximation of the exact minimum.

    Volume alone is not enough. One view usually shows two faces of a part, an
    L in cross-section, and a box flush with the L's diagonal encloses it in
    the same volume as the box flush with its faces (as for a right triangle);
    sampling noise then picks one, and the diagonal is 30 degrees out. So among
    near-smallest boxes the one whose faces carry the most points wins: the
    surfaces the camera saw are the part's faces, the diagonal carries none.
    """
    try:
        hull = points[ConvexHull(points).vertices]
        normals = ConvexHull(hull).equations[:, :3]
    except Exception:
        # Coplanar points (a flat part seen face on): qhull cannot build a 3D
        # hull, and the one axis worth trying is the plane's normal.
        hull = points
        normals = np.linalg.eigh(np.cov((points - points.mean(0)).T))[1][:, :1].T
    boxes = []
    for n in np.unique(np.round(normals, 4), axis=0):
        n = n / np.linalg.norm(n)
        a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(n, a)
        _, _, angle = cv2.minAreaRect((hull @ np.column_stack([a, b])).astype(np.float32))
        u = np.cos(np.radians(angle)) * a + np.sin(np.radians(angle)) * b
        R = np.column_stack([u, np.cross(n, u), n])
        proj = hull @ R
        boxes.append((np.prod(proj.max(0) - proj.min(0)), R))
    smallest = min(volume for volume, _ in boxes)

    def on_faces(R):
        proj = points @ R
        lo, hi = proj.min(0), proj.max(0)
        return np.mean(np.minimum(proj - lo, hi - proj).min(1) < PLANE_TOL_M)

    R = max((R for volume, R in boxes if volume <= smallest * (1 + BOX_VOLUME_SLACK)),
            key=on_faces)
    proj = points @ R
    lo, hi = proj.min(0), proj.max(0)
    return R, R @ ((lo + hi) / 2), hi - lo


def fit_box(mask, metres, K, rng=None):
    """Box of one object on one frame: (T_cam_box, extent, note), or None.

    T_cam_box's rotation is the box axes and its translation the box centre.
    """
    cam, valid = backproject(metres, K)
    mask = mask & valid
    if mask.sum() < MIN_BOX_POINTS:
        return None
    keep, note = mask, "no clear support surface; raw mask"
    plane = support_plane(mask, cam, valid, rng or np.random.default_rng(0))
    if plane is not None:
        n, p = plane
        in_front = mask & ((cam - p) @ n < -PLANE_TOL_M)
        if in_front.sum() >= max(MIN_BOX_POINTS, FLAT_KEEP_FRACTION * mask.sum()):
            keep, note = in_front, "support surface removed"
        else:
            note = "flat on its support; raw mask"
    R, centre, extent = min_obb(cam[keep])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, centre
    return T, extent, f"{int(keep.sum())} of {int(mask.sum())} mask px, {note}"


def box_in_camera(motion, box0, extent, K, image_shape):
    """The init-frame box carried into the current camera by the object's motion.

    ``motion`` is the pipeline's motion since the init frame, the same one the
    anchors ride on. ``rect`` is the image-space [x0, y0, x1, y1] of the
    projected corners, None if any corner is behind the camera or the box lies
    fully outside the image.
    """
    motion = as_numpy(motion)
    if motion.shape != (4, 4) or not np.all(np.isfinite(motion)):
        return None
    T_cam_box = motion @ box0
    extent = as_numpy(extent)
    corners = (_CORNER_SIGNS * 0.5 * extent) @ T_cam_box[:3, :3].T + T_cam_box[:3, 3]

    rect = None
    if np.all(corners[:, 2] > 0):
        uv = corners @ K.T
        uv = uv[:, :2] / uv[:, 2:]
        h, w = image_shape
        x0, y0 = np.clip(uv.min(0), 0, [w - 1, h - 1])
        x1, y1 = np.clip(uv.max(0), 0, [w - 1, h - 1])
        if x1 > x0 and y1 > y0:
            rect = [float(x0), float(y0), float(x1), float(y1)]
    return {
        "pose": T_cam_box.tolist(),
        "extent": extent.tolist(),
        "corners": corners.tolist(),
        "rect": rect,
    }


class PoseServer:
    def __init__(self, cfg_path, bind):
        self.cfg = OmegaConf.load(cfg_path)
        if self.cfg.pipeline.type != "modular":
            raise ValueError(f"expected modular pipeline, got {self.cfg.pipeline.type}")
        self.cfg.pipeline.params.save_pose = False
        self.cfg.pipeline.params.save_meta_data = False

        self.pipeline = None
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.bbox_warned = False

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind)
        print(f"[server] listening on {bind}", flush=True)

    def build_pipeline(self):
        self.pipeline = ModularPipeline(self.cfg)
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.bbox_warned = False

    def fit_boxes(self, rgb, metres, K, points, labels):
        """Each object's box on the init frame, fit to its mask; None where it cannot be.

        A failure here costs the box, never the tracking.
        """
        try:
            # Same call as a preview, made on the freshly built pipeline before
            # add_user_points(), which is the state a preview sees.
            logits = self.pipeline.preview_user_masks(rgb, points, labels)
            masks = (as_numpy(logits) > 0).reshape(len(points), *metres.shape)
        except Exception:
            print("[server] no init masks; tracking without boxes", flush=True)
            traceback.print_exc()
            return [None] * len(points)
        boxes = []
        for index, mask in enumerate(masks):
            try:
                fit = fit_box(mask, metres, K)
            except Exception:
                traceback.print_exc()
                fit = None
            if fit is None:
                print(f"[server] object {index}: no box", flush=True)
                boxes.append(None)
                continue
            T, extent, note = fit
            size = " x ".join(f"{v * 1000:.0f}" for v in extent)
            print(f"[server] object {index}: box {size} mm ({note})", flush=True)
            boxes.append((T, extent))
        return boxes

    def bboxes(self, motions, frame):
        """Per-object boxes; a failure yields None for that box, never a failed reply.

        The box is only drawn and measured, so it must not take the poses down.
        """
        out = []
        for index, motion in enumerate(motions):
            box = self.boxes[index] if self.boxes and index < len(self.boxes) else None
            if box is None:
                out.append(None)
                continue
            try:
                out.append(box_in_camera(motion, *box, frame.intrinsics, frame.rgb.shape[:2]))
            except Exception:
                if not self.bbox_warned:
                    self.bbox_warned = True
                    print("[server] bbox failed; sending None (logged once per session)",
                          flush=True)
                    traceback.print_exc()
                out.append(None)
        return out

    def handle_init(self, meta, parts):
        self.build_pipeline()
        rgb = decode_rgb(parts[0])
        depth = decode_depth(parts[1], meta).astype(np.float32)
        K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3)
        depth_factor = float(meta["depth_factor"])
        self.anchors = np.stack([
            anchor_from_points(points, labels, depth, K, depth_factor)
            for points, labels in zip(meta["points"], meta["labels"])
        ])
        self.boxes = self.fit_boxes(rgb, depth / depth_factor, K, meta["points"], meta["labels"])
        self.pipeline.add_user_points(meta["points"], meta["labels"])
        frame = Frame(
            id=self.frame_id,
            rgb=rgb,
            depth=depth,
            intrinsics=K,
            depth_factor=depth_factor,
            timestamp=meta.get("timestamp", time.time()),
        )
        motions = np.asarray(self.pipeline.step(frame), dtype=np.float64)
        self.frame_id += 1
        return motions @ self.anchors, frame, motions

    def handle_track(self, meta, parts):
        if self.pipeline is None or self.anchors is None:
            raise RuntimeError("pipeline not initialized; send an 'init' request first")
        frame = Frame(
            id=self.frame_id,
            rgb=decode_rgb(parts[0]),
            depth=decode_depth(parts[1], meta).astype(np.float32),
            intrinsics=np.asarray(meta["K"], dtype=np.float64).reshape(3, 3),
            depth_factor=float(meta["depth_factor"]),
            timestamp=meta.get("timestamp", time.time()),
        )
        motions = np.asarray(self.pipeline.step(frame), dtype=np.float64)
        self.frame_id += 1
        return motions @ self.anchors, frame, motions

    def handle_preview(self, meta, parts):
        if self.pipeline is None:
            self.build_pipeline()
        rgb = decode_rgb(parts[0])
        logits = self.pipeline.preview_user_masks(rgb, meta["points"], meta["labels"])
        mask = (logits.squeeze(1) > 0).cpu().numpy().astype(np.uint8)
        out = {"mask_shape": list(mask.shape), "mask_sum": mask.sum(-1).sum(-1).tolist()}
        if meta.get("dump_dir"):
            import os

            os.makedirs(meta["dump_dir"], exist_ok=True)
            cv2.imwrite(f"{meta['dump_dir']}/rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay[mask[0] > 0] = (0, 0, 255)
            cv2.imwrite(f"{meta['dump_dir']}/mask_overlay.png", overlay)
            out["dumped"] = True
        return out

    def run(self):
        while True:
            msg = self.sock.recv_multipart()
            meta = json.loads(msg[0])
            cmd = meta.get("cmd")
            t0 = time.time()
            try:
                if cmd == "ping":
                    self.sock.send_json({"ok": True, "pong": True})
                    continue
                if cmd == "reset":
                    self.pipeline = None
                    self.sock.send_json({"ok": True})
                    continue
                if cmd == "preview":
                    self.sock.send_json({"ok": True, **self.handle_preview(meta, msg[1:])})
                    continue

                if cmd == "init":
                    poses, frame, motions = self.handle_init(meta, msg[1:])
                elif cmd == "track":
                    poses, frame, motions = self.handle_track(meta, msg[1:])
                else:
                    raise ValueError(f"unknown cmd: {cmd}")

                lost = [bool(getattr(o, "lost", False)) for o in self.pipeline.objects]
                bboxes = self.bboxes(motions, frame)
                self.sock.send_json(
                    {
                        "ok": True,
                        "frame_id": self.frame_id - 1,
                        "poses": np.asarray(poses, dtype=np.float64).tolist(),
                        "lost": lost,
                        "bboxes": bboxes,
                        "latency_ms": round((time.time() - t0) * 1000, 1),
                    }
                )
            except Exception as exc:
                traceback.print_exc()
                self.sock.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/pipeline/pipeline_test2.yaml")
    ap.add_argument("-b", "--bind", default="tcp://0.0.0.0:5555")
    args = ap.parse_args()
    PoseServer(args.config, args.bind).run()
