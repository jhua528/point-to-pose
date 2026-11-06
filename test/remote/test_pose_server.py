"""Tests of scripts/remote/pose_server.py's mesh pose handling, without the GPU pipeline: rendered depth (ray cast
with open3d), the demonstration meshes of assets/test, and a PoseServer stub whose tracked motion each test sets.

Slow (each mesh fit takes seconds): pick with -k, e.g.
    python -m pytest test/remote/test_pose_server.py -k "set_down or prior"
"""

import importlib.util
import json
import threading
import time
import types
import zlib
from concurrent.futures import Future
from pathlib import Path

import cv2
import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pose_server_under_test", ROOT / "scripts/remote/pose_server.py")
ps = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ps)

# The test scenes' camera: K, the bench 0.469 m below, tilted as measured on them.
W, H = 640, 480
K = np.array([[348.0045527399691, 0, 320.5], [0, 348.0045527399691, 240.5], [0, 0, 1.0]])
NORMAL = np.array([0.0, 0.808, 0.589]) / np.linalg.norm([0.0, 0.808, 0.589])
UP = -NORMAL
EX = np.array([1.0, 0.0, 0.0])
TABLE = np.column_stack([EX, np.cross(UP, EX), UP])        # bench (a, b, up) to camera
ORIGIN = np.array([0.0, 0.0, 0.469 / NORMAL[2]])
DEPTH_NOISE_M = 0.001


def rot(axis, deg):
    a = np.asarray(axis, float)
    return cv2.Rodrigues((np.radians(deg) * a / np.linalg.norm(a)).reshape(3, 1))[0]


def scene_mesh(scene, name):
    """A demonstration mesh of assets/test, about its box's centre."""
    d = ROOT / "assets/test" / scene
    meta = json.loads((d / "meta_init.json").read_text())
    order = json.loads((d / "order.json").read_text())
    spec = meta["models"][order["objects"].index(name)]
    raw = zlib.decompress((d / f"mesh_{name}.f32i32.zlib").read_bytes())
    nv, nf = int(spec["vertices"]), int(spec["faces"])
    v = np.frombuffer(raw, "<f4", 3 * nv).reshape(nv, 3).astype(np.float64)
    f = np.frombuffer(raw, "<i4", 3 * nf, offset=12 * nv).reshape(nf, 3).astype(np.int64)
    return v - (v.min(0) + v.max(0)) / 2, f


def box_mesh(ex, ey, ez):
    v = np.array([[x, y, z] for x in (-ex / 2, ex / 2) for y in (-ey / 2, ey / 2) for z in (-ez / 2, ez / 2)])
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    return v, f


def resting(v, R_body, a=0.0, b=0.0, lift=0.0):
    """Mesh frame -> camera for a part turned by R_body (bench frame), on the bench at (a, b)."""
    low = (v @ R_body.T)[:, 2].min()
    T = np.eye(4)
    T[:3, :3] = TABLE @ R_body
    T[:3, 3] = TABLE @ np.array([a, b, -low + lift]) + ORIGIN
    return T


def render(parts, seed=0):
    """(depth, ids): the bench and `parts` ([(vertices in the camera frame, faces)]) ray cast; ids index parts."""
    scene = o3d.t.geometry.RaycastingScene()
    bench = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0.0]]) @ TABLE.T + ORIGIN
    scene.add_triangles(o3d.core.Tensor(bench.astype(np.float32)), o3d.core.Tensor(np.array([[0, 1, 2], [0, 2, 3]], np.uint32)))
    for P, f in parts:
        scene.add_triangles(o3d.core.Tensor(P.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    hit = scene.cast_rays(o3d.core.Tensor(np.concatenate([np.zeros_like(d), d], -1).astype(np.float32)))
    z = hit["t_hit"].numpy().astype(np.float64)
    gid = hit["geometry_ids"].numpy().astype(np.int64)
    depth = np.where(np.isfinite(z), z + np.random.default_rng(seed).normal(0, DEPTH_NOISE_M, z.shape), 0.0)
    ids = np.where(np.isfinite(z) & (gid >= 1), gid - 1, -1)
    return depth.astype(np.float32), ids


class SyncExecutor:
    """mesh_fit_executor's stand-in: a background fit done before submit() returns."""

    def submit(self, fn, *args, **kwargs):
        fut = Future()
        fut.set_result(fn(*args, **kwargs))
        return fut


def stub_server(models):
    """A PoseServer as fit_boxes() leaves it, without the pipeline."""
    s = object.__new__(ps.PoseServer)
    n = len(models)
    s.pipeline = types.SimpleNamespace(objects=[types.SimpleNamespace(lost=False) for _ in range(n)])
    s.models = list(models)
    s.fitted_models = [None] * n
    s.model_required = [True] * n
    s.model_errors = [None] * n
    s.model_retry_after = [0.0] * n
    s.model_check_fails = [0] * n
    s.model_withheld = [False] * n
    s.model_corrections = [None] * n
    s.sent_model_poses = [None] * n
    s.bindings = [None] * n
    s.still_motions, s.holding = [None] * n, [False] * n
    s.pose_priors, s.last_fitted, s.pending_binds = {}, {}, {}
    s.scale_hints = [None] * n
    s.locate_shapes, s.known_scales = {}, {}
    s.mesh_session, s.mesh_fit_job, s.mesh_fit_executor = 1, None, SyncExecutor()
    s.track_robot, s.init_report, s.archive_dir = None, None, None
    s.boxes = [None] * n
    s.surfaces, s.frame_id = [], 0
    return s


def frame(fid, depth, masks):
    return types.SimpleNamespace(id=fid, depth=depth, depth_factor=1.0, intrinsics=K,
                                 rgb=np.zeros((H, W, 3), np.uint8), mask=np.asarray(masks, np.float32)[:, None])


def init_fit(s, index, depth, mask):
    """fit_boxes()'s fit of one object on the init frame."""
    model = s.models[index]
    view = ps.observe(mask, depth.astype(np.float64), K)
    fit, reason = ps.fit_model_box(model, view, K)
    assert fit is not None, reason
    s.fitted_models[index] = model
    s.boxes[index] = fit[:2]
    s.still_motions[index] = np.eye(4)        # handle_init(): the init's motion


def turned(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def mesh_error(model, pose, v_true_cam):
    """Mean distance (m) of the fitted mesh's points from the same points of the part."""
    P = (model.points_source @ np.asarray(model.fitted_linear).T) @ pose[:3, :3].T + pose[:3, 3]
    return float(np.linalg.norm(P - v_true_cam, axis=1).mean())


TESTER = scene_mesh("PP-S1-1", "tester")
TESTER_K = 2.44                         # the simulated tester is about this many times its reconstruction


def make_tester(R_init, R_down, a=0.04, b=0.03):
    v, f = TESTER
    true_v = v * TESTER_K
    model = ps.ObjectModel(v, f)
    model.points_source = v             # the mesh's own vertices, for mesh_error()
    T_A = resting(true_v, R_init)
    T_B = resting(true_v, R_down, a, b)
    return model, true_v, f, T_A, T_B


def run_set_down(model, true_v, f, T_A, T_B, frames, motion_of, prior_on=None):
    """Init on T_A, then the part lies at T_B; motion_of(t) is the tracker's motion on frame t. Per frame: None
    (nothing sent) or (turn deg, mean mm) of the pose sent."""
    s = stub_server([model])
    depth, ids = render([(true_v @ T_A[:3, :3].T + T_A[:3, 3], f)])
    init_fit(s, 0, depth, ids == 0)
    depth, ids = render([(true_v @ T_B[:3, :3].T + T_B[:3, 3], f)], seed=1)
    v_true = true_v @ T_B[:3, :3].T + T_B[:3, 3]
    out = []
    for t in range(1, frames + 1):
        fr = frame(t, depth, [ids == 0])
        motions = s.check_carried_meshes(motion_of(t)[None], fr, None)
        if prior_on == t:
            s.take_priors({"priors": [{"object": 0, "pose": T_B.tolist(), "source": "robot_set_down"}]}, t)
        motions = s.apply_priors(motions, fr, None)
        s.update_bindings(motions, fr, None, None, observe=True)
        s.retry_pending_meshes(motions, fr, None)
        m = s.fitted_models[0]
        if m is None or s.model_withheld[0]:
            out.append(None)
            continue
        pose = motions[0] @ m.fitted_pose
        m.points_source = model.points_source
        out.append((turned(pose[:3, :3], T_B[:3, :3]), 1000 * mesh_error(m, pose, v_true)))
    return s, out


def wrong_motion(T_A, T_B, seed=3, deg=6.0, mm=6.0):
    """The tracker after a set-down: the true motion, off by a turn and a shift that grow every frame."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)

    def motion(t):
        M = T_B @ np.linalg.inv(T_A)
        bad = rot(axis + 0.3 * rng.normal(size=3), deg * (t + 2))
        c = T_B[:3, 3]
        out = np.eye(4)
        out[:3, :3] = bad @ M[:3, :3]
        out[:3, 3] = bad @ (M[:3, 3] - c) + c + mm / 1000 * (t + 2) * np.array([1.0, -0.5, 0.3])
        return out
    return motion


# 1. A part set down turned over, the tracked motion wrong (MODEL_CHECK_FAILS).

@pytest.mark.parametrize("R_down", [rot([0, 0, 1], 70) @ rot([1, 0, 0], 180), rot([0, 0, 1], 40) @ rot([0, 1, 0], 90)],
                         ids=["turned-over", "stood-up"])
def test_set_down_refit_checked_and_held(R_down):
    model, true_v, f, T_A, T_B = make_tester(rot([0, 0, 1], 30), R_down)
    _, out = run_set_down(model, true_v, f, T_A, T_B, 12, wrong_motion(T_A, T_B))
    sent = [e for e in out if e is not None]
    # Nothing sent turned another way round; the motion's first frames off it by what the carried check lets
    # through, then the part held where it lies.
    assert len(sent) >= len(out) - 2
    assert all(deg < 30 for deg, _ in sent), out
    assert all(e is not None and e[0] < 15 and e[1] < 15 for e in out[3:]), out


def test_set_down_prior_taken_at_once():
    model, true_v, f, T_A, T_B = make_tester(rot([0, 0, 1], 30), rot([0, 0, 1], 70) @ rot([1, 0, 0], 180))
    s, out = run_set_down(model, true_v, f, T_A, T_B, 8, wrong_motion(T_A, T_B), prior_on=1)
    assert all(e is not None and e[0] < 15 and e[1] < 15 for e in out), out
    assert s.pose_priors[0]["status"] in ("adopted", "agrees")


def test_set_down_prior_refused_where_the_view_says_otherwise():
    model, true_v, f, T_A, T_B = make_tester(rot([0, 0, 1], 30), rot([0, 0, 1], 70))
    s = stub_server([model])
    depth, ids = render([(true_v @ T_A[:3, :3].T + T_A[:3, 3], f)])
    init_fit(s, 0, depth, ids == 0)
    fr = frame(1, depth, [ids == 0])
    motions = s.check_carried_meshes(np.eye(4)[None], fr, None)
    wrong = T_A.copy()
    wrong[:3, 3] += 0.15 * TABLE[:, 0]                       # 15 cm off: nothing there
    s.take_priors({"priors": [{"object": 0, "pose": wrong.tolist(), "source": "robot_set_down"}]}, 1)
    s.apply_priors(motions, fr, None)
    assert s.pose_priors[0]["status"] == "pending" and "off the part" in s.pose_priors[0]["why"]
    assert turned((motions[0] @ s.fitted_models[0].fitted_pose)[:3, :3], T_A[:3, :3]) < 10


def test_held_part_released_when_the_robot_moves_it():
    model, true_v, f, T_A, _ = make_tester(rot([0, 0, 1], 30), np.eye(3))
    s = stub_server([model])
    depth, ids = render([(true_v @ T_A[:3, :3].T + T_A[:3, 3], f)])
    init_fit(s, 0, depth, ids == 0)
    bad = wrong_motion(T_A, T_A)
    for t in range(1, 13):
        if t <= 5:
            T, motion = T_A, bad(t)                          # still, the tracker wrong
        else:
            M = np.eye(4)
            M[:3, :3] = rot(UP, 6.0 * (t - 5))
            M[:3, 3] = T_A[:3, 3] - M[:3, :3] @ T_A[:3, 3] + TABLE @ np.array([0.008, 0.004, 0.012]) * (t - 5)
            T = M @ T_A
            motion = M                                       # moved, the tracker right
        depth, ids = render([(true_v @ T[:3, :3].T + T[:3, 3], f)], seed=t)
        motions = s.check_carried_meshes(motion[None], frame(t, depth, [ids == 0]), None)
        assert not s.model_withheld[0]
        pose = motions[0] @ s.fitted_models[0].fitted_pose
        assert turned(pose[:3, :3], T[:3, :3]) < 15, t
        assert np.linalg.norm(pose[:3, 3] - T[:3, 3]) < 0.02, t
        if t > 7:
            assert not s.holding[0], t


# 2. The binding's evidence counted in votes (BINDING_CONFIDENCE).

def turning_binding(R_body, frames, step_deg, lift=0.0):
    """A tester turning about the bench's up by step_deg a frame, its fit published turned end for end, every
    box turn a hypothesis: the binding after `frames` frames, and the truth's index in it."""
    v, f = TESTER
    true_v = v * TESTER_K
    model = ps.ObjectModel(v, f)
    T0 = resting(true_v, R_body, lift=lift)
    s = stub_server([model])
    depth, ids = render([(true_v @ T0[:3, :3].T + T0[:3, 3], f)])
    init_fit(s, 0, depth, ids == 0)
    B, c = model.box[:3, :3], model.box[:3, 3]
    fit = ps._ModelFit(model, ps.observe(ids == 0, depth.astype(np.float64), K), K, np.random.default_rng(0))
    scale = ps.fitted_axis_scale(model)
    ways = []
    for P in [np.eye(3)] + list(model.box_turns):
        D = np.eye(4)
        D[:3, :3] = B @ P @ B.T
        D[:3, 3] = model.fitted_linear @ c - D[:3, :3] @ (model.fitted_linear @ c)
        start = model.fitted_pose @ D
        R2, t2, terms = fit.icp(scale, start[:3, :3], start[:3, 3], ps.MODEL_FINE_ROUNDS, ps.MODEL_SAMPLES,
                                ps.MODEL_FINE_POINTS, 2 * fit.tol)
        if terms is not None:
            refit = np.eye(4)
            refit[:3, :3], refit[:3, 3] = R2, t2
            if not any(ps.same_way(model, R2, w[:3, :3]) for w in ways):
                ways.append(refit)
    truth = next(i for i, w in enumerate(ways) if ps.same_way(model, w[:3, :3], T0[:3, :3]))
    wrong = next(i for i, w in enumerate(ways) if i != truth)
    model.fitted_pose = ways[wrong]
    model.fitted_doubts = [np.linalg.inv(ways[wrong]) @ w for i, w in enumerate(ways) if i != wrong]
    model.fitted_unlikely = []
    model.fitted_spin_doubt = model.fitted_ends_doubt = None
    s.update_bindings(np.eye(4)[None], frame(0, depth, [ids == 0]), None, None, observe=True)
    centre = T0[:3, 3]
    for t in range(1, frames + 1):
        M = np.eye(4)
        M[:3, :3] = rot(UP, step_deg * t)
        M[:3, 3] = centre - M[:3, :3] @ centre
        T = M @ T0
        depth, ids = render([(true_v @ T[:3, :3].T + T[:3, 3], f)], seed=t)
        fr = frame(t, depth, [ids == 0])
        motions = s.check_carried_meshes(M[None], fr, None)
        s.update_bindings(motions, fr, None, None, observe=True)
    b = s.bindings[0]
    m = s.fitted_models[0]
    truth_now = [k for k in range(len(b.turns))
                 if ps.same_way(m, (motions[0] @ m.fitted_pose @ b.turns[k])[:3, :3], T[:3, :3])]
    return b, truth_now


def test_one_view_rules_nothing_out():
    b, truth = turning_binding(rot([0, 0, 1], 30), 6, 0.0)
    assert truth and all(b.alive[k] for k in truth)
    assert not any(r and r[0] == "evidence" for r in b.ruled)
    _, _, pairs = b.odds()
    # Votes, not samples: the odds of one view stay those of a few votes.
    assert max(lo for lo, *_ in pairs.values()) < np.log(1e4)


def test_views_rule_the_wrong_ways_round_out_and_never_the_truth():
    b, truth = turning_binding(rot([0, 0, 1], 30), 14, 12.0)
    assert truth and all(b.alive[k] for k in truth)
    assert sum(1 for r in b.ruled if r and r[0] == "evidence") >= 1
    for r in b.ruled:
        if r and r[0] == "evidence":
            assert "votes" in r[1] and "frames" in r[1]


# 3. Every half turn a hypothesis (MODEL_DOUBT_HALF_TURNS).

def test_half_turns_kept_as_hypotheses():
    v, f = TESTER
    true_v = v * TESTER_K
    model = ps.ObjectModel(v, f)
    T0 = resting(true_v, rot([0, 0, 1], 0))
    occluder, of = box_mesh(0.20, 0.10, 0.13)
    O = resting(occluder, rot([0, 0, 1], 0), 0.07, -0.08)
    depth, ids = render([(true_v @ T0[:3, :3].T + T0[:3, 3], f), (occluder @ O[:3, :3].T + O[:3, 3], of)])
    view = ps.observe(ids == 0, depth.astype(np.float64), K)
    fit, reason = ps.fit_model_box(model, view, K)
    assert fit is not None, reason
    model.fitted_id = "test00"
    b = ps.Binding(model)
    B = model.box[:3, :3]
    halves = [P for P in model.box_turns if np.allclose(np.abs(P), np.eye(3))]
    found = 0
    for P in halves:
        R = model.fitted_pose[:3, :3] @ B @ P @ B.T
        found += any(ps.same_way(model, (model.fitted_pose @ T)[:3, :3], R) for T in b.turns)
    # Each half turn that fits at all is one of the binding's ways round.
    assert found >= 2
    assert any(ps.same_way(model, (model.fitted_pose @ T)[:3, :3], T0[:3, :3]) for T in b.turns)


# 4. A bar's length held by the end the view shows (_ModelFit.axial()).

def test_bar_slid_back_onto_its_seen_end():
    v, f = scene_mesh("RI-S1-1", "bar")
    model = ps.ObjectModel(v, f)
    a = model.box[:3, int(np.argmax(model.extent))]
    length, radius = 0.34, 0.0135
    n = 48
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])
    cv = np.vstack([np.column_stack([ring, np.full(n, -length / 2)]), np.column_stack([ring, np.full(n, length / 2)]),
                    [[0, 0, -length / 2], [0, 0, length / 2]]])
    cf = []
    for i in range(n):
        j = (i + 1) % n
        cf += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    cf = np.array(cf)
    axis = np.cross([0, 0, 1.0], a)
    align = rot(axis, np.degrees(np.arccos(np.clip(a[2], -1, 1)))) if np.linalg.norm(axis) > 1e-9 else np.eye(3)
    true_v = cv @ align.T
    T0 = resting(true_v, rot([0, 0, 1], 20))
    along_cam = T0[:3, :3] @ a
    hand, hf = box_mesh(0.09, 0.09, 0.07)
    hand = hand + T0[:3, 3] + 0.17 * along_cam
    depth, ids = render([(true_v @ T0[:3, :3].T + T0[:3, 3], cf), (hand, hf)])
    s = stub_server([model])
    init_fit(s, 0, depth, ids == 0)
    ends = [e["seen"] for e in model.fitted_axial["ends"]]
    assert ends.count(True) == 1                             # one end behind the hand
    for t in range(1, 6):
        motion = np.eye(4)
        motion[:3, 3] = 0.008 * t * along_cam                 # the tracker drifting along the bar
        motions = s.check_carried_meshes(motion[None], frame(t, depth, [ids == 0]), None)
        m = s.fitted_models[0]
        pose = motions[0] @ m.fitted_pose
        P = (m.points @ np.asarray(m.fitted_linear).T) @ pose[:3, :3].T + pose[:3, 3]
        seen_end = ((P - T0[:3, 3]) @ along_cam).min()
        assert abs(seen_end + length / 2) < ps.MODEL_TOL_M * 1.5, (t, seen_end)
        assert m.carried_axial is not None


# 5. A plate's thickness from the support (MODEL_SUPPORT_TOP_PERCENTILE).

def test_plate_thickness_from_the_support():
    v, f = box_mesh(0.07, 0.025, 0.005)                      # the mesh: 5 mm thick
    true_v = v * np.array([2.4, 2.4, 0.6])                    # the part: 3 mm thick
    model = ps.ObjectModel(v, f)
    T0 = resting(true_v, rot([0, 0, 1], 25))
    depth, ids = render([(true_v @ T0[:3, :3].T + T0[:3, 3], f)])
    view = ps.observe(ids == 0, depth.astype(np.float64), K)
    fit, reason = ps.fit_model_box(model, view, K)
    assert fit is not None, reason
    scale = ps.fitted_axis_scale(model)
    k = int(np.argmin(scale * model.extent))
    assert model.fitted_axis_source[k] == "support"
    assert abs(scale[k] * model.extent[k] - 0.003) < 0.0015
    P = (model.points @ np.asarray(model.fitted_linear).T) @ model.fitted_pose[:3, :3].T + model.fitted_pose[:3, 3]
    heights = (P - ORIGIN) @ UP
    assert abs(heights.min()) < 0.0015                       # on the bench, not into it


# 7. Every way round in doubt said of what kind, about which axis.

def test_doubt_kinds_in_the_reply():
    model, true_v, f, T_A, _ = make_tester(rot([0, 0, 1], 30), np.eye(3))
    s = stub_server([model])
    depth, ids = render([(true_v @ T_A[:3, :3].T + T_A[:3, 3], f)])
    init_fit(s, 0, depth, ids == 0)
    B, c = model.box[:3, :3], model.box[:3, 3]
    doubts = []
    for P in (np.diag([-1.0, -1, 1]), np.diag([-1.0, 1, -1]), np.diag([1.0, -1, -1])):
        D = np.eye(4)
        D[:3, :3] = B @ P @ B.T
        D[:3, 3] = model.fitted_linear @ c - D[:3, :3] @ (model.fitted_linear @ c)
        doubts.append(D)
    model.fitted_doubts = doubts
    model.fitted_spin_doubt = model.fitted_ends_doubt = None
    reply = s.bboxes(np.eye(4)[None], frame(0, depth, [ids == 0]))[0]
    kinds = reply["model_doubt_kinds"]
    assert len(kinds) == len(reply["model_doubts"]) == 3
    assert sum(k["kind"] == "ends" for k in kinds) == 2 and len(reply["model_ends_doubts"]) == 2
    assert all(abs(k["deg"] - 180) < 1 for k in kinds)


# 6. A track request answered in time though its frame is not (SERVER_REPLY_SHARE).

def test_track_answered_stale_while_a_frame_takes_long():
    s = object.__new__(ps.PoseServer)
    s.frame_id, s.last_reply = 7, {"ok": True, "frame_id": 6, "poses": [], "lost": []}
    s.init_report = None
    slow = threading.Event()

    def frame_reply(cmd, meta, parts, t0):
        if meta.get("slow"):
            slow.wait(3.0)
        s.frame_id += 1
        s.last_reply = {"ok": True, "frame_id": s.frame_id - 1}
        return s.last_reply

    s.frame_reply = frame_reply
    t0 = time.time()
    reply = s.answer_track({"cmd": "track", "timeout_ms": 1000, "slow": True}, [], t0)
    assert reply["stale"]["working_on"] == 7 and reply["frame_id"] == 6
    assert time.time() - t0 < 0.5 + 0.3
    t1 = time.time()
    again = s.answer_track({"cmd": "track", "timeout_ms": 1000}, [], t1)
    assert "stale" in again and time.time() - t1 < 0.1      # still busy: answered at once, the frame dropped
    slow.set()
    s.track_job["future"].result(timeout=5)
    fresh = s.answer_track({"cmd": "track", "timeout_ms": 1000}, [], time.time())
    assert "stale" not in fresh and fresh["frame_id"] == 8
    plain = s.answer_track({"cmd": "track"}, [], time.time())  # a client that says nothing: as before
    assert "stale" not in plain and plain["frame_id"] == 9
