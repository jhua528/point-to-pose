# What pose_client sends the pose server

Each folder is one trial's requests, captured from the live camera, robot
description and TF by pose_client's own `on_frame()` into a socket that only
recorded them (nothing was sent). A request is one ZeroMQ multipart message,
its parts in the order `order.json` lists.

To capture them again (after a change to pose_client, the scenes or the
picks), with Gazebo and MoveIt running and ROS and the workspace sourced:

    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp   # as ~/.bashrc and the launcher's Gazebo have it
    python3 tests/server_inputs/capture.py [TRIAL ...]

From a shell on another DDS than Gazebo's, the topics still come but the
arm's actions are never answered: every reset fails with "reset took over
180 s".

It runs pose_client's `main()` with the arguments `experiment/run.py` starts it
with for that trial (its picks), after opening the gripper, taking the arm
home and setting up the trial's scene as `run.py` does at a trial's start.
The person holding the bar is in every RI, RR and HS scene (`truth.json`'s `others`: `human`).

Captured again 2026-11-19, every trial, with the meshes the runs now send: a demonstration part's confirmed refined model where it has one (tester, sample, ring: `refined_model/model.obj`, the tester's probe moved and joined on), else its recorded scan (bar: `obj_2/scaled_mesh.obj`) -- `agent.core.demo_mesh.current()`, which `picks.mesh_args()` applies to the picks' paths -- and the calibration redone for them (`scripts/sim_alignment.json`, `truth.json`). As `run.py` runs now: before pose_client starts, a VLM
(qwen3.8-flash, `run.py --first-clicks vlm`, the default) is shown the
camera's first colour image with every part of the trial named, and gives a
few points on each (`agent/core/first_clicks.py`). Points are checked one by
one, never a part's all at once for touching another:

- dropped upstream (`run.py`): no depth there, under the bench, far (25 cm)
  from the part's other points, or on another part's point (2 px: one pixel
  of the image, either part's)
- dropped by pose_client before the init goes out: on its robot mask (none
  were, the arm at home)

Parts in contact (the ring on the bar in RR and HS) keep their points: the
server tells them apart by each other's clicks (`separate_parts()`). The
points left go to pose_client's `--points`, so the server does not locate
those parts by their meshes. Three inits per trial, the other parts the same
(`order.json`: `init`, `init_located`, `init_clean`):

- `meta_init.json`: as sent now, the VLM's points in `points`/`labels`
- `meta_init_located.json`: the VLM's points left out, the parts with meshes
  and no click (`[]`) located by the server as before
- `meta_init_clean.json`: the VLM's points that are off the part by
  Gazebo's truth left out (a part left with none is located by the server).
  For offline comparison only: no run knows that truth.

`first_clicks.json` is what the VLM said and how it was checked: per part
its points kept, those dropped and why, and `on_part`, whether each kept
point is on the part by Gazebo's truth (the simulated part's mesh at its pose
against the depth).

| trial | VLM's points sent (on the part, by Gazebo's truth) | clean |
|---|---|---|
| PP-S1-1 .. PP-S4-2 | tester 3 (3), sample 3 (3), every one | the same |
| HS-S1-1 | ring 3 (3), bar 3 (3), tester 3 (3), sample 3 (3) | the same |
| RI-S1-2, RI-S2-1, RI-S2-2, RI-S3-1, RI-S3-2, RR-S1-1 | ring 3 (3), bar 3 (3) | the same |
| RI-S1-1 | ring 3 (2), bar 3 (3) | ring 2 |
| RI-S4-1 | ring 1 (1), bar 3 (2) | bar 2 |
| RI-S4-2 | ring 1 (1), bar 3 (3) | the same |
| RR-S1-2 | ring 2 (1), bar 2 (2); one of each dropped upstream | ring 1 |

The ring is small in the image (250-900 px) and half behind the box in S4:
the VLM gave it one point there, and its misses fell onto the bench or the bar
beside it, a few pixels off, which no check upstream tells. `send.py TRIAL
init` sends what the runs send; `init_located` and `init_clean` are the
comparisons.

## init (`cmd: "init"`)

| part | file | content |
|---|---|---|
| 1 | `meta_init.json` | `cmd`, `K` (3x3, row-major), `depth_factor` (1.0: metres), `timestamp`, per object `points` and `labels` (empty: located by its mesh), `models` (one entry per object, null without a mesh: `vertices` and `faces` counts; `known_scale` with `--true-size-meshes`, `colours` with `--mesh-colours`), `discover` (how many obstacles beyond the objects the server looks for at init: `--obstacles`; 0 in these, pose_client's default and `run.py`'s without `--discover` -- the Agent keeps what is in the way from the depth itself), `same_scale_meshes` (the meshes are of one reconstruction, so their scales are held to each other's: `picks.mesh_args()`), `robot_mask` (`shape`, `bits`: base64 of zlib of `np.packbits` of the robot's pixels), `depth_enc` (`raw_f32`), `depth_shape` |
| 2 | `rgb.jpg` | the colour frame, JPEG quality 90, BGR as OpenCV writes it |
| 3 | `depth.f32.zlib` | the depth frame: zlib of little-endian float32 metres, `depth_shape` rows by columns, 0 where there is none |
| 4.. | `mesh_<object>.f32i32.zlib` | one per object with a mesh, in object order: zlib of float32 vertices (x, y, z, metres) then int32 triangles (then an RGB byte triple per vertex where `colours`) |

The trials' picks name only the task's two parts (all four in HS); the
obstacles of S3 and the occluder of S4 are neither clicked nor, with
`discover` 0, looked for by the server.

## track (`cmd: "track"`)

| part | file | content |
|---|---|---|
| 1 | `meta_track.json` | `cmd`, `K`, `depth_factor`, `timestamp`, `timeout_ms` (how long the client waits: the server answers within part of it, or `stale`), `robot_mask` (as in init, this frame's), `depth_enc`, `depth_shape` |
| 2 | `rgb.jpg` | as in init |
| 3 | `depth.f32.zlib` | as in init |

Sent with a track only when there is something to send, so not in these
captures:

- `priors`: `[{"object": index, "pose": 4x4 mesh pose in the camera's frame, "source": "robot_set_down"}]`, the robot's pose of a part it set down (`/point2pose/pose_prior`), until a reply takes it.
- `bind`: `[{"object": index, "hypothesis": id, "by": "person" | "camera" | "robot"}]`, which way round a part lies, as a person, the Agent's own look at the camera's view, or the robot said (`/point2pose/binding`), until a reply takes it. A server older than `by` reads it as the person's.

Also sent: `{"cmd": "ping"}` alone, after a timeout, to wait until the server
has caught up. The server also reads a `dump_dir` the client never sends.

## Ground truth (`truth.json`) -- offline debugging only

**This is Gazebo's truth. No run knows it and pose_client never sends it.**
It is here to measure the server's replies against. It is read from
`/gazebo/model_states` just before the capture and again just after, and the
two reads must agree (`moved_during_capture`). The camera's pose comes from
`/tf_static`. It is given in the server's own terms, per object in request
order:

| field | content |
|---|---|
| `model_pose` | 4x4 rigid pose of the mesh sent (`mesh_file`, in its own vertex frame), in the camera's optical frame the request's `K` and depth are in (x right, y down, z forward, metres): what a reply's `model_pose` should be |
| `model_linear` | 3x3: the mesh's scale to the simulated part (anisotropic, from the calibration). A vertex `x` lies at `model_pose[:3,:3] @ (model_linear @ x) + model_pose[:3,3]`, as with a reply's `model_linear` |
| `model_scale` | the cube root of `det(model_linear)`: one number for comparing with a reply's `model_scale` |
| `alike_model_poses` | the poses in which the simulated part looks identical. Each is as true as `model_pose`; for example, the sample either side up. Empty for the tester |
| `scaled_extent_m` | the scaled mesh's extent along its own axes |
| `calibration_chamfer_mm` | how well the demonstration mesh fits the simulated part's own shape. A pose this good is the best any fit can reach |
| `depth_check` | the mesh at its truth pose, rendered into the camera and compared with `depth_m.npy`: pixels rendered, pixels seen (within 10 mm), and the median and p90 depth difference |
| `moved_during_capture` | how far the part moved between the two truth reads (mm, deg) |
| `gazebo` | `T_world_link` and `T_link_model` (`scripts/sim_alignment.json`): where `model_pose` comes from, with the top-level `T_world_camera` |

`others` holds every other Gazebo model in the camera frame, as link poses
(not mesh poses): props, the S3 obstacles, the S4 occluder and the person.
The robot is not included. `truth_overlay.png` shows each mesh at its truth
pose on `rgb.jpg`: green where the depth agrees, red where the part is hidden
or the depth is off.

In this capture, every part's median `depth_check` difference is 1.1-2.5 mm,
which is the demonstration mesh's own shape error. The one exception is the
bar in RR-S1-2, at 4.6 mm: there the person's hand grips it, and the fingers
lie within 10 mm of its surface. No part moved more than 0.01 mm during a
capture.

After printing the reply, `send.py` compares each object's `model_pose` and
`model_linear` with the truth. It uses the nearest of the alike poses and
prints:

- ADD over the mesh's vertices, each mesh at its own scale;
- how far the mesh origin is off;
- how far the pose is turned;
- the reply's scale relative to the truth's.

Use `--no-truth` to skip this comparison.

## Not sent, to look at

- `depth_m.npy`: the depth as a 480 x 640 float32 array, metres
- `robot_mask.png`: the robot mask, white on the robot
- `truth.json`, `truth_overlay.png`: Gazebo's truth, offline only (see above)
- `order.json`: the parts of each request (`init`, `init_located`, `track`), and each object's mesh file
- `first_clicks.json`: the VLM's points and their checks (see above)
- `meta_init_located.json`, `meta_init_clean.json`: the comparison inits (see above)

The JSON files are indented here; the client sends them compact. `send.py`
sends a capture's request to a server (an init replaces the session running
there):

    python3 send.py PP-S1-1 init            # as the runs send it, with the VLM's points
    python3 send.py PP-S1-1 init_located    # the same, the server locating the parts
    python3 send.py PP-S1-1 init_clean      # the same, the VLM's points off the part (by truth) left out
    python3 send.py PP-S1-1 track           # after an init
