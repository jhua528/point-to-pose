"""ZeroMQ inference server exposing ModularPipeline to a remote ROS2 client."""

import argparse
import json
import time
import zlib
import traceback

import cv2
import numpy as np
import zmq
from omegaconf import OmegaConf

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


class PoseServer:
    def __init__(self, cfg_path, bind):
        self.cfg = OmegaConf.load(cfg_path)
        if self.cfg.pipeline.type != "modular":
            raise ValueError(f"expected modular pipeline, got {self.cfg.pipeline.type}")
        self.cfg.pipeline.params.save_pose = False
        self.cfg.pipeline.params.save_meta_data = False

        self.pipeline = None
        self.frame_id = 0

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind)
        print(f"[server] listening on {bind}", flush=True)

    def build_pipeline(self):
        self.pipeline = ModularPipeline(self.cfg)
        self.frame_id = 0

    def handle_init(self, meta, parts):
        self.build_pipeline()
        rgb = decode_rgb(parts[0])
        self.pipeline.add_user_points(meta["points"], meta["labels"])
        frame = Frame(
            id=self.frame_id,
            rgb=rgb,
            depth=decode_depth(parts[1], meta).astype(np.float32),
            intrinsics=np.asarray(meta["K"], dtype=np.float64).reshape(3, 3),
            depth_factor=float(meta["depth_factor"]),
            timestamp=meta.get("timestamp", time.time()),
        )
        poses = self.pipeline.step(frame)
        self.frame_id += 1
        return poses

    def handle_track(self, meta, parts):
        if self.pipeline is None:
            raise RuntimeError("pipeline not initialized; send an 'init' request first")
        frame = Frame(
            id=self.frame_id,
            rgb=decode_rgb(parts[0]),
            depth=decode_depth(parts[1], meta).astype(np.float32),
            intrinsics=np.asarray(meta["K"], dtype=np.float64).reshape(3, 3),
            depth_factor=float(meta["depth_factor"]),
            timestamp=meta.get("timestamp", time.time()),
        )
        poses = self.pipeline.step(frame)
        self.frame_id += 1
        return poses

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
                    poses = self.handle_init(meta, msg[1:])
                elif cmd == "track":
                    poses = self.handle_track(meta, msg[1:])
                else:
                    raise ValueError(f"unknown cmd: {cmd}")

                lost = [bool(getattr(o, "lost", False)) for o in self.pipeline.objects]
                self.sock.send_json(
                    {
                        "ok": True,
                        "frame_id": self.frame_id - 1,
                        "poses": np.asarray(poses, dtype=np.float64).tolist(),
                        "lost": lost,
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
