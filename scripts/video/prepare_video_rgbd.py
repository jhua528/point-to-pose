"""Turn an RGB-only video into the RGB-D sequence Point2Pose runs on.

Point2Pose needs, per frame, a depth map in metres and the camera's
intrinsics; a phone or webcam video has neither. MoGe-2 (third_party/MoGe,
commit 925b8ed, the last MoGe-2 version) estimates both from one image:
metric depth, and the horizontal field of view.

  1. The field of view is estimated on --fov-frames frames spread over the
     video and their median is kept (or --fov-x is used as given), so every
     frame shares one camera.
  2. Each frame's depth is estimated with that field of view fixed.
  3. Monocular depth jitters in overall scale from frame to frame, and the
     tracker reads that jitter as motion. With --smooth N (default 9) each
     frame's depth is rescaled so that its median follows the running median
     over N frames: slow changes (the camera coming closer) stay, the frame
     to frame jitter goes. --smooth 0 leaves the depth as MoGe gives it.

Output, in the layout Point2Pose's dataset readers use:
  OUT/rgb/000000.png ...      the frames
  OUT/depth/000000.png ...    uint16 depth in millimetres (0 = no depth)
  OUT/cam_K.txt               3x3 intrinsics
  OUT/meta.json               fps, size, field of view, per-frame scale
  OUT/depth_preview.mp4       the depth, colour-mapped, beside the frame

Run from the repository root:
  python scripts/video/prepare_video_rgbd.py \\
      --video assets/test/video/location.mp4 --out data/location
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "MoGe"))


def read_frames(path, start, stride, max_frames, width):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, index = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index >= start and (index - start) % stride == 0:
            if width and bgr.shape[1] != width:
                height = int(round(bgr.shape[0] * width / bgr.shape[1]))
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        index += 1
    cap.release()
    return frames, fps / stride


def infer(model, rgb, device, fov_x, resolution_level):
    image = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = model.infer(image, fov_x=fov_x, resolution_level=resolution_level, use_fp16=True)
    depth = out["depth"].float().cpu().numpy()
    mask = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(depth)
    depth[~mask | ~np.isfinite(depth)] = 0.0
    return depth, out["intrinsics"].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "checkpoints", "moge", "moge-2-vitl-normal", "model.pt"),
                    help="MoGe-2 checkpoint file, or a Hugging Face repo id (Ruicheng/moge-2-vitl-normal)")
    ap.add_argument("--start", type=int, default=0, help="first video frame to keep")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="at most this many frames (0: all)")
    ap.add_argument("--width", type=int, default=0, help="resize frames to this width (0: as they are)")
    ap.add_argument("--fov-x", type=float, default=None,
                    help="horizontal field of view in degrees, if known; otherwise estimated")
    ap.add_argument("--fov-frames", type=int, default=12, help="frames the field of view is estimated on")
    ap.add_argument("--smooth", type=int, default=9,
                    help="running-median window (frames) for the depth's overall scale; 0: off")
    ap.add_argument("--resolution-level", type=int, default=9, help="MoGe's resolution level, 0-9")
    ap.add_argument("--max-depth", type=float, default=10.0, help="depth beyond this (m) is dropped")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from moge.model.v2 import MoGeModel

    frames, fps = read_frames(args.video, args.start, args.stride, args.max_frames, args.width)
    if not frames:
        raise SystemExit("no frames read")
    h, w = frames[0].shape[:2]
    print(f"{len(frames)} frames, {w}x{h}, {fps:.2f} fps", flush=True)

    model = MoGeModel.from_pretrained(args.model).to(args.device).eval()

    fov_x = args.fov_x
    if fov_x is None:
        picks = np.unique(np.linspace(0, len(frames) - 1, min(args.fov_frames, len(frames))).round().astype(int))
        fovs = []
        for i in picks:
            _, intr = infer(model, frames[i], args.device, None, args.resolution_level)
            fovs.append(math.degrees(2 * math.atan(0.5 / float(intr[0, 0]))))
        fov_x = float(np.median(fovs))
        print(f"field of view {fov_x:.1f} deg (median of {len(fovs)}: {min(fovs):.1f}-{max(fovs):.1f})", flush=True)

    fx = 0.5 * w / math.tan(math.radians(fov_x) / 2)
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)

    depths, medians = [], []
    for i, rgb in enumerate(frames):
        depth, _ = infer(model, rgb, args.device, fov_x, args.resolution_level)
        depth[depth > args.max_depth] = 0.0
        depths.append(depth.astype(np.float32))
        valid = depth[depth > 0]
        medians.append(float(np.median(valid)) if valid.size else float("nan"))
        if i % 20 == 0:
            print(f"  depth {i}/{len(frames)}: median {medians[-1]:.3f} m", flush=True)

    medians = np.asarray(medians)
    scales = np.ones(len(frames))
    if args.smooth > 1:
        half = args.smooth // 2
        for i in range(len(frames)):
            window = medians[max(0, i - half): i + half + 1]
            window = window[np.isfinite(window)]
            if window.size and np.isfinite(medians[i]) and medians[i] > 0:
                scales[i] = float(np.median(window)) / medians[i]
        print(f"scale jitter removed: {np.std(scales) * 100:.1f}% (std), "
              f"{np.min(scales):.3f}-{np.max(scales):.3f}", flush=True)

    for name in ("rgb", "depth"):
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    np.savetxt(os.path.join(args.out, "cam_K.txt"), K)
    writer = cv2.VideoWriter(os.path.join(args.out, "depth_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (2 * w, h))
    top = np.nanpercentile(medians, 50) * 2.5
    for i, (rgb, depth) in enumerate(zip(frames, depths)):
        depth = depth * scales[i]
        cv2.imwrite(os.path.join(args.out, "rgb", f"{i:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.out, "depth", f"{i:06d}.png"),
                    np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16))
        shade = cv2.applyColorMap(np.clip(depth / top * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        shade[depth <= 0] = 0
        writer.write(np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), shade]))
    writer.release()

    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"video": os.path.abspath(args.video), "fps": fps, "width": w, "height": h,
                   "fov_x_deg": fov_x, "K": K.tolist(), "depth_scale": scales.tolist(),
                   "median_depth_m": [None if not np.isfinite(m) else m for m in medians.tolist()],
                   "smooth": args.smooth, "start": args.start, "stride": args.stride}, f, indent=1)
    print(f"wrote {len(frames)} frames to {args.out}", flush=True)


if __name__ == "__main__":
    main()
