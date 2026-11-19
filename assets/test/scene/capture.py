"""Capture what pose_client sends the pose server for each trial, exactly:
the client's own main() -- its arguments as experiment/run.py starts it
(Run.client_args(): rate, obstacles, the trial's picks, and with run.py's
--first-clicks vlm, its default, the parts' points clicked by a VLM on the
first image, Run.first_clicks()) -- builds an init and a track request from the live
camera frame, robot description and TF, into a socket that only records.
Beside the init as sent, meta_init_located.json is the same init with the
VLM's points left out (the server locates those parts by their meshes),
meta_init_clean.json the same init with the VLM's points that are off the
part by Gazebo's truth left out -- for offline comparison only: no run knows
that truth -- and first_clicks.json what the VLM said and how it was checked.
Nothing reaches the server. Before each, the arm is opened and taken home
and the trial's scene set up, as run.py does at the start of a trial.

    python3 tests/server_inputs/capture.py [TRIAL ...]     (default: every trial with picks)

Source ROS and the workspace first; Gazebo and MoveIt must be running.
Each trial's folder is replaced only once its new capture is complete. The
capture itself runs in a process of its own (`--one TRIAL DIR`): the
client's main() starts and shuts down ROS, the arm's reset runs on it too.
"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiment"), str(ROOT)]
import picks as picks_store  # noqa: E402

# After the scene is set up, for the camera's frames to show it.
SETTLE_S = 5.0
# For the camera, the robot description and its transforms to arrive.
READY_S = 40.0
# For one capture, all told.
CAPTURE_TIMEOUT_S = 150.0


class Captured(Exception):
    pass


class RecordingSocket:
    """A REQ socket that keeps what is sent and answers nothing."""

    def __init__(self):
        self.parts = None

    def setsockopt(self, *args):
        pass

    def send_multipart(self, parts):
        self.parts = list(parts)

    def recv_json(self):
        raise Captured()

    def close(self):
        pass


def capture(trial, out, client_args):
    """Run pose_client's main() for `trial`, with `client_args` (run.py's
    Run.client_args()), into `out`; the files written."""
    import pose_client as pc
    import rclpy

    entry = picks_store.load()[trial]
    objects = [o["name"] for o in entry["objects"]]
    requests = {}
    client = pc.PoseClient

    class Recording(client):
        def _connect(self):
            self.sock = RecordingSocket()

        def on_frame(self, rgb, depth):
            self.frames = (rgb, depth)          # the synchronised pair; sent below, not queued

    def build(node):
        deadline = time.time() + READY_S
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.K is not None and node.robot_links and getattr(node, "frames", None):
                rgb, depth = node.frames
                metres = node.bridge.imgmsg_to_cv2(depth, "passthrough").astype(np.float64)
                if node.robot_mask(metres, rgb.header.frame_id) is not None:
                    break
        else:
            raise RuntimeError("no camera frames, robot description or robot transforms")
        for _ in range(10):                     # a frame after the transforms
            rclpy.spin_once(node, timeout_sec=0.1)
        for cmd in ("init", "track"):
            node.initialized, node.server_busy, node.sock.parts = cmd == "track", False, None
            try:
                client.track_frame(node, *node.frames)
            except Captured:
                pass
            if node.sock.parts is None:
                raise RuntimeError(f"no {cmd} request was built")
            requests[cmd] = node.sock.parts

    class Capturing:
        """main()'s executor: builds the two requests in place of spinning."""

        def __init__(self, *args, **kwargs):
            self.node = None

        def add_node(self, node):
            self.node = node

        def spin(self):
            build(self.node)

        def shutdown(self):
            pass

    argv = ["pose_client.py", *client_args]
    saved = sys.argv, pc.PoseClient, pc.MultiThreadedExecutor
    sys.argv, pc.PoseClient, pc.MultiThreadedExecutor = argv, Recording, Capturing
    try:
        pc.main()
    finally:
        sys.argv, pc.PoseClient, pc.MultiThreadedExecutor = saved
    if set(requests) != {"init", "track"}:
        raise RuntimeError("pose_client's main() ended without building both requests")

    out.mkdir(parents=True)
    meta = json.loads(requests["init"][0])
    names = ["meta_init.json", "rgb.jpg", "depth.f32.zlib" if meta.get("depth_enc") == "raw_f32" else "depth.png"]
    names += [f"mesh_{name}.f32i32.zlib" for name, spec in zip(objects, meta.get("models") or []) if spec]
    if len(names) != len(requests["init"]):
        raise RuntimeError(f"the init has {len(requests['init'])} parts, not the {len(names)} expected")
    for name, part in zip(names, requests["init"]):
        (out / name).write_bytes(part if name != "meta_init.json" else
                                 json.dumps(json.loads(part), indent=1).encode())
    (out / "meta_track.json").write_text(json.dumps(json.loads(requests["track"][0]), indent=1))
    # For looking at, not sent: the depth as an array, the robot mask as an image.
    if meta.get("depth_enc") == "raw_f32":
        h, w = meta["depth_shape"]
        np.save(out / "depth_m.npy", np.frombuffer(zlib.decompress(requests["init"][2]), np.float32).reshape(h, w))
    if meta.get("robot_mask"):
        h, w = meta["robot_mask"]["shape"]
        bits = np.frombuffer(zlib.decompress(base64.b64decode(meta["robot_mask"]["bits"])), np.uint8)
        cv2.imwrite(str(out / "robot_mask.png"), np.unpackbits(bits, count=h * w).reshape(h, w) * 255)
    from agent.core.demo_mesh import current
    # The files the meshes sent were read from: as picks.mesh_args() gives
    # them to the client, a confirmed refinement over the picks' scan.
    meshes = {o["name"]: current(o["mesh"]) for o in entry["objects"] if o.get("mesh")}
    (out / "order.json").write_text(json.dumps({"init": names, "track": ["meta_track.json", "rgb.jpg", names[2]],
                                                "objects": objects, "meshes": meshes}, indent=1))
    return sorted(p.name for p in out.iterdir())


def _rigid_change(A, B):
    """How far rigid pose B is from A: (mm, deg)."""
    R = np.asarray(A, float)[:3, :3].T @ np.asarray(B, float)[:3, :3]
    deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
    return float(np.linalg.norm(np.asarray(B, float)[:3, 3] - np.asarray(A, float)[:3, 3]) * 1000), deg


def truth_file(folder, session, before, after, camera):
    """truth.json: Gazebo's truth of the capture in `folder`, in the server's
    own terms -- each object's mesh (the one sent, mesh_<object>) posed in
    the camera's frame as a reply's model_pose and model_linear are:
    x_cam = model_pose[:3, :3] @ (model_linear @ x) + model_pose[:3, 3].
    From each part's Gazebo model pose (`before`: read just before the
    capture; `after`: just after, for how much it moved meanwhile) and
    scripts/sim_alignment.json (T_link_model, linear, sim_alike). `camera`:
    the recorder's colour_view() at the capture, (rgb, K, T_world_camera).
    Checked: the mesh sent is the one the alignment calibrates, its K the
    request's, and the posed mesh against the captured depth."""
    import sim_parts as sp
    from recorder import ROBOT_MODEL

    order = json.loads((folder / "order.json").read_text())
    meta = json.loads((folder / "meta_init.json").read_text())
    _, K, T_world_cam = camera
    if not np.allclose(np.asarray(meta["K"], float).reshape(3, 3), K, atol=1e-6):
        raise RuntimeError("the camera_info's K is not the request's")
    T_cam_world = np.linalg.inv(T_world_cam)
    depth = np.load(folder / "depth_m.npy").astype(np.float64)
    objects, overlay = [], []
    for index, name in enumerate(order["objects"]):
        part = session.alignment.get(name)
        spec = (meta.get("models") or [None] * (index + 1))[index]
        entry = {"index": index, "name": name}
        if part is None or name not in before or spec is None:
            entry["missing"] = ("no mesh sent" if spec is None else "no alignment" if part is None
                                else "not in Gazebo")
            objects.append(entry)
            continue
        mesh_file = f"mesh_{name}.f32i32.zlib"
        raw = zlib.decompress((folder / mesh_file).read_bytes())
        sent = np.frombuffer(raw[:spec["vertices"] * 12], np.float32).reshape(-1, 3)
        vertices, faces = sp.read_obj(sp.ROOT / part["mesh"])
        if sent.shape != vertices.shape or not np.array_equal(sent, vertices):
            raise RuntimeError(f"{name}: the mesh sent is not {part['mesh']}, the one the alignment calibrates")
        linear = np.asarray(part["linear"], float)
        T_link_model = np.asarray(part["T_link_model"], float)
        pose = T_cam_world @ before[name] @ T_link_model
        # The check: the mesh so posed, rendered into the camera, against the depth sent.
        samples = sp.surface_samples(vertices.astype(np.float64) @ linear.T, faces, 120_000)
        rendered = sp.render(samples, pose, K, depth.shape)
        seen = sp.visible_mask(rendered, depth)
        residual = np.abs(depth[seen] - rendered[seen]) * 1000
        overlay.append((name, np.isfinite(rendered) & ~seen, seen))
        moved_mm, moved_deg = _rigid_change(before[name], after.get(name, before[name]))
        scaled = vertices.astype(np.float64) @ linear.T
        entry.update(
            mesh_file=mesh_file, mesh_source=part["mesh"],
            model_pose=np.round(pose, 7).tolist(),
            model_linear=np.round(linear, 6).tolist(),
            model_scale=round(float(np.cbrt(np.linalg.det(linear))), 6),
            alike_model_poses=[np.round(pose @ np.asarray(S, float), 7).tolist() for S in part.get("sim_alike") or []],
            scaled_extent_m=np.round(scaled.max(0) - scaled.min(0), 5).tolist(),
            calibration_chamfer_mm=part.get("chamfer_mm"),
            gazebo={"T_world_link": np.round(before[name], 7).tolist(),
                    "T_link_model": T_link_model.tolist()},
            moved_during_capture={"mm": round(moved_mm, 3), "deg": round(moved_deg, 3)},
            depth_check={"rendered_px": int(np.isfinite(rendered).sum()), "seen_px": int(seen.sum()),
                         "median_abs_mm": round(float(np.median(residual)), 2) if residual.size else None,
                         "p90_abs_mm": round(float(np.percentile(residual, 90)), 2) if residual.size else None})
        objects.append(entry)
    # truth_overlay.png, to look at: each mesh at its truth on the colour
    # image sent -- green where the depth agrees, red where it is hidden or off.
    image = cv2.imread(str(folder / "rgb.jpg"))
    for name, hidden, seen in overlay:
        image[seen] = (0.45 * image[seen] + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
        image[hidden] = (0.45 * image[hidden] + 0.55 * np.array([60, 60, 230])).astype(np.uint8)
        ys, xs = np.nonzero(seen | hidden)
        if len(xs):
            cv2.putText(image, name, (int(xs.min()), max(12, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(folder / "truth_overlay.png"), image)
    names = set(order["objects"])
    others = {model: np.round(T_cam_world @ T, 6).tolist() for model, T in sorted(before.items())
              if model not in names and model != ROBOT_MODEL and model not in ("ground_plane",)}
    (folder / "truth.json").write_text(json.dumps({
        "note": "GROUND TRUTH from Gazebo, for offline debugging only: no run, and nothing pose_client "
                "sends, knows it. Each object's model_pose/model_linear are in the server's reply terms: "
                "the mesh sent (mesh_file, its own vertex frame) is at "
                "x_cam = model_pose[:3,:3] @ (model_linear @ x) + model_pose[:3,3], in the camera's "
                "optical frame the request's K and depth are in (x right, y down, z forward, metres). "
                "alike_model_poses: poses the simulated part looks the same in (as true as model_pose; "
                "the sample either side up). model_linear is the calibrated scale of the demonstration "
                "mesh to the simulated part (sim_alignment.json), anisotropic; model_scale its cube-root "
                "volume scale. depth_check: the mesh so posed rendered into the camera against "
                "depth_m.npy, where it is seen (within 10 mm) -- the mesh is the demonstration's "
                "reconstruction, not the simulated part's own, so a few mm is its shape error "
                "(calibration_chamfer_mm). others: every other Gazebo model's link pose in the "
                "camera frame (props, obstacles, the person; not the robot).",
        "frame": "camera optical frame of the captured images (the request's K, depth_shape)",
        "T_world_camera": np.round(T_world_cam, 7).tolist(),
        "objects": objects, "others": others}, indent=1))
    return objects


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trials", nargs="*")
    ap.add_argument("--one", nargs=3, metavar=("TRIAL", "DIR", "CLIENT_ARGS"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        trial, out, client_args = args.one
        print("  " + ", ".join(capture(trial, Path(out), json.loads(client_args))), flush=True)
        return
    trials = args.trials or sorted(picks_store.load())
    import run as runner
    work = HERE / ".capture"
    shutil.rmtree(work, ignore_errors=True)
    # run.py's own defaults, as a point2pose session: what a trial's run starts pose_client with.
    session = runner.Session(runner.parser().parse_args(["--perception", "point2pose", "--out", str(work / "session")]),
                             [])
    failed = []
    for trial in trials:
        print(f"=== {trial} {time.strftime('%H:%M:%S')}", flush=True)
        run = runner.Run(session, {"id": trial}, 1)
        try:
            ok, why = run.reset_arm()
            if ok:
                ok, why = run.run_and_wait("scene", f"python3 -u experiment/scene.py setup {trial}",
                                           runner.SETUP_TIMEOUT_S)
            if not ok:
                raise RuntimeError(f"arm reset or scene setup failed: {why}")
            time.sleep(SETTLE_S)
            staged = work / trial
            entry = session.picks[trial]
            if getattr(session.args, "first_clicks", None) == "vlm":
                # As run.py's start_perception(): the VLM's clicks first, what it says shown.
                note = run.note
                run.note = lambda text, *rest: (print(f"  {text}", flush=True), note(text, *rest))
                entry = runner.Run.first_clicks(run, entry)
                run.note = note
            client_args = json.dumps(run.client_args(entry))
            # Gazebo's truth and the camera's pose, just before the capture and after it (truth_file()).
            camera = session.recorder.colour_view()
            if camera is None:
                raise RuntimeError(f"no camera view for the truth: {getattr(session.recorder, 'colour_why', '?')}")
            before = session.recorder.snapshot()[0]
            done = subprocess.run([sys.executable, __file__, "--one", trial, str(staged), client_args], cwd=ROOT,
                                  capture_output=True, text=True, timeout=CAPTURE_TIMEOUT_S)
            if done.returncode != 0:
                raise RuntimeError((done.stderr.strip().splitlines() or ["no output"])[-1])
            after = session.recorder.snapshot()[0]
            for o in truth_file(staged, session, before, after, camera):
                check = o.get("depth_check") or {}
                print(f"  truth {o['name']}: " + (o["missing"] if "missing" in o else
                      f"seen {check['seen_px']} of {check['rendered_px']} px, depth off {check['median_abs_mm']} mm "
                      f"(p90 {check['p90_abs_mm']}), moved {o['moved_during_capture']['mm']} mm"), flush=True)
            clicked = [index for index, (before, now) in
                       enumerate(zip(session.picks[trial]["objects"], entry["objects"]))
                       if not before["points"] and now["points"]]
            meta = json.loads((staged / "meta_init.json").read_text())
            for index in clicked:
                meta["points"][index], meta["labels"][index] = [], []
            (staged / "meta_init_located.json").write_text(json.dumps(meta, indent=1))
            # Off the part by Gazebo's truth (first_clicks.json's on_part): left out.
            record = getattr(run, "first_click_record", None) or {}
            off = {name: {tuple(p) for p, on in zip(part["points"], part["on_part"] or []) if not on}
                   for name, part in (record.get("parts") or {}).items()}
            clean = json.loads((staged / "meta_init.json").read_text())
            for index, name in enumerate(o["name"] for o in entry["objects"]):
                keep = [k for k, p in enumerate(clean["points"][index]) if tuple(p) not in off.get(name, set())]
                clean["points"][index] = [clean["points"][index][k] for k in keep]
                clean["labels"][index] = [clean["labels"][index][k] for k in keep]
            (staged / "meta_init_clean.json").write_text(json.dumps(clean, indent=1))
            (staged / "first_clicks.json").write_text(json.dumps(
                {"vlm_clicked": [entry["objects"][i]["name"] for i in clicked],
                 **(getattr(run, "first_click_record", None) or {"note": "no VLM clicks (off, or no image/answer)"})},
                indent=1))
            order = json.loads((staged / "order.json").read_text())
            order["init_located"] = ["meta_init_located.json"] + order["init"][1:]
            order["init_clean"] = ["meta_init_clean.json"] + order["init"][1:]
            (staged / "order.json").write_text(json.dumps(order, indent=1))
            shutil.rmtree(HERE / trial, ignore_errors=True)
            shutil.move(str(staged), str(HERE / trial))
            print(done.stdout.strip().splitlines()[-1], flush=True)
        except Exception as exc:                            # noqa: BLE001
            failed.append(trial)
            print(f"  FAILED: {exc}", flush=True)
        finally:
            run.log.close()
    session.close()
    if failed:
        print(f"the runs' logs (reset, scene): {work / 'session'}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    print(f"captured {len(trials) - len(failed)} of {len(trials)}" + (f"; failed: {' '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
