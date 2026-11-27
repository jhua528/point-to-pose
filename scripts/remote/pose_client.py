"""ROS2 node (runs on the camera machine) that offloads pose tracking to a remote GPU server.

Only needs rclpy + cv_bridge + pyzmq -- point2pose itself is not imported here.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import zlib
from xml.etree import ElementTree

import cv2
import message_filters
import numpy as np
import rclpy
import zmq
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseArray, Pose, PoseStamped
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from yolo_msgs.msg import BoundingBox3D
from pose_stream import LatestFrame

BBOX_TOPIC = "/point2pose/bboxes"
# Where the server cannot locate an object by its mesh for sure -- it fits no
# place, only weakly, or several places about as well (pose_server's
# NeedClick) -- the person is asked to click it: the request goes out on
# CLICK_REQUEST_TOPIC (JSON: {"objects": [names], "why": {name: why},
# "candidates": {name: [click, ...]}, "frame_id", "stamp"}, a click a list of
# [u, v] pixels), with the frame it was asked on, candidates numbered from 0,
# on CLICK_REQUEST_IMAGE_TOPIC. The answer comes on CLICK_TOPIC, one object a
# message: {"object": name, "points": [[u, v], ...]} (labels 1, or "labels"),
# or {"object": name, "candidate": k}. No init is sent until every object asked
# for is answered; then one is, with every object's click -- those the server
# was sure of as it located them -- and nothing is located by mesh.
CLICK_REQUEST_TOPIC = "/point2pose/click_request"
CLICK_REQUEST_IMAGE_TOPIC = "/point2pose/click_request/image"
CLICK_TOPIC = "/point2pose/click"
# Which way round a mesh lies on its part (pose_sever's Binding) goes with its
# pose on /detected/<name>_model as "binding": "single" (nothing in doubt),
# "ambiguous" or "locked", by what ("evidence", "support", "person"), each way
# round still living -- its id, its turn relative to the pose, its probability
# -- and those ruled out, and why. The views settle it as the part turns; a
# doubt need not be settled to act, though, where it changes nothing done:
# with --grasps, each object's grasps (4x4, the gripper in the mesh's frame
# and units) are tried on every way round living, and a grasp that puts the
# gripper within --grasp-tolerance-mm / -deg of where it puts it on the pose
# published -- or turned half round about --grasp-flip-axis of it, a parallel
# gripper's -- is safe to take now ("grasps_safe", their indices). Where the
# doubt matters, the person is asked which way round (--ask-binding: "needed",
# for an object none of whose grasps is safe or that has none given;
# "always"; "never"): on BINDING_REQUEST_TOPIC (JSON: {"object", "fit_id",
# "hypotheses": [{"id", "p"}]}), the frame on BINDING_REQUEST_IMAGE_TOPIC with
# each way round's telling features -- where it has what the others have not
# -- drawn in its colour and number. Tracking goes on meanwhile; the views may
# settle it first, and the request is then withdrawn ({"object", "withdrawn":
# true}). The answer comes on BINDING_TOPIC: {"object": name, "hypothesis":
# id or number}, or {"object": name, "points": [[u, v], ...]} clicked on the
# part's telling feature -- the way round whose features lie nearest them --
# and goes to the server with the next frame (meta["bind"]).
BINDING_REQUEST_TOPIC = "/point2pose/binding_request"
BINDING_REQUEST_IMAGE_TOPIC = "/point2pose/binding_request/image"
BINDING_TOPIC = "/point2pose/binding"
# One asker of the person: a node publishing on BINDING_ASKER_TOPIC -- the
# Agent, which puts its own way-round questions to the person, of the doubts
# its task meets, after looking at the camera's view itself -- asks; while
# one does, this client asks none of its own (but with --ask-binding always),
# the same doubt else asked twice in two ways. It still keeps every way round
# in doubt as asked (binding_asked), unshown, so that the answers that come on
# BINDING_TOPIC are taken. An answer may say whose it is, {"by": "person" /
# "camera" / "robot"}, sent on to the server with it (meta["bind"]); "person"
# where it does not.
BINDING_ASKER_TOPIC = "/point2pose/binding_asker"
# The robot's pose of a part it set down, from whoever knows it (the agent):
# {"object": name, "pose": 4x4, the mesh's pose in the world as
# /detected/<name>_model gives it, "source": e.g. "robot_set_down"} on
# POSE_PRIOR_TOPIC. Sent to the server with the next frame, in the camera's
# frame (meta["priors"]), where it is one more way the part may lie, scored
# with the fit's (pose_sever's MODEL_CHECK_FAILS); what came of it is in the
# model message's "prior".
POSE_PRIOR_TOPIC = "/point2pose/pose_prior"
# Its "source" where the robot holds the part (pose_sever's HOLD_PRIOR_SOURCE):
# turning it in the hand before the camera, it says where the part is, lost to
# the tracker there; sent on as any prior, but the part is not set down -- its
# last-seen box and a click asked of it stay as they were.
HOLD_PRIOR_SOURCE = "robot_hold"
BINDING_COLOURS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 255, 0), (255, 128, 128)]
# Box edges as corner-index pairs; the server orders corners so bit 0/1/2 of the
# index flips x/y/z, hence edges join indices that differ in exactly one bit.
BOX_EDGES = [(i, i | 1 << k) for i in range(8) for k in range(3) if not i >> k & 1]
# A shorter lifetime than the time between replies makes the box flicker.
# Matches perception's staleness, 2 s or 1.5 update intervals: a box outlives
# its pose by no more than the Agent does.
BBOX_LIFETIME_SEC = 2.0
BBOX_LIFETIME_UPDATES = 1.5
# The init locates by its mesh each object sent without points: SAM2's
# proposals, then up to pose_sever.LOCATE_SCREENS rough mesh fits on them
# (1-2 s each) and LOCATE_FITS proper ones (10-12 s), before the fit every
# object gets. On RI-S1-1, the ring and the bar both so, the init took
# 90-130 s against the 60 s allowed, and each retry after a timeout started
# it over; so each such object adds this to --init-timeout-ms.
LOCATE_TIMEOUT_MS = 90000
# The robot's own pixels go with the init, so the server locates no part on
# them (pose_sever.locate_with_mesh()): the arm standing straight up in front
# of the cart is a long black cylinder, and the bar's mesh, stretched, took
# its upper link for the bar (RI-S1-1, 2026-10-24 and -25). Each link's
# visual mesh in the robot description is sampled ROBOT_SPACING_M apart and
# posed by TF; a depth pixel within ROBOT_MARGIN_M of it is the robot, the
# margin taking in the camera's calibration and depth noise.
ROBOT_DESCRIPTION_TOPIC = "/robot_description"
ROBOT_SPACING_M = 0.005
ROBOT_MARGIN_M = 0.015
# A link's samples at most: a larger mesh, the cart's, is sampled sparser.
ROBOT_MAX_SAMPLES = 100000
# The init waits this long after start for the description and the robot's
# transforms, then goes ahead without the robot's pixels: every proposal is
# fitted, as before. Reading the meshes holds the node 4 s, and the first
# frame after it came before any transform: that init went without them,
# and its bar was the arm again.
ROBOT_WAIT_S = 20.0
# A part with a mesh the tracker has lost for RELOCATE_LOST_FRAMES replies in
# a row is located again by its mesh: SAM2/TAPIR does not find a part again
# once it has lost it -- the ring and the bar both, under the gripper putting
# the ring on, stayed lost from then on, the ring on the bar as planned
# (2026-10-28). The server re-seeds no single object, so the whole init is
# sent again, each part tracked now clicked where it is now. Only once the
# robot covers at most RELOCATE_CLEAR_SHARE of where the part was last seen:
# a part lost in the hand would not be found there, and a failed init leaves
# nothing tracked until one succeeds. At most once per RELOCATE_EVERY_S.
RELOCATE_LOST_FRAMES = 10
RELOCATE_CLEAR_SHARE = 0.1
RELOCATE_EVERY_S = 60.0
# A part with a mesh still tracked is clicked where its fitted mesh is seen,
# not located by its mesh: on RELOCATE_CLICKS points of its surface (samples
# RELOCATE_SPACING_M apart), at least RELOCATE_MIN_SEEN of them seen -- the
# image's depth within RELOCATE_DEPTH_M of theirs, off the arm. The rest are
# located by their meshes. Not while both fingertips' visuo-tactile sensors
# touch something (TACTILE_TOPIC, a reading no older than TACTILE_FRESH_S) if
# one of those the hand may hold: the hand (HAND_LINK, between the fingertips)
# within RELOCATE_HAND_M of its box, or of where it was last seen since it was
# lost. A part in the hand is not found there by its mesh: the ring pulled off
# the bar was lost in the hand, and every init sent again failed on it, the
# bar untracked with it (2026-10-29). The bar moved away in RI-S2-2 waited for
# the hand to let go of the ring, tracked in it all along, and the Agent gave
# up (2026-10-29): the ring is clicked in the hand now, where the hand has
# carried it by the time the init goes. Only the sensors' contacts are read.
# An init sent again that failed goes again RELOCATE_RETRY_S on, twice as long
# after each failure in a row up to RELOCATE_RETRY_MAX_S.
TACTILE_TOPIC = "/tactile/state"
TACTILE_FRESH_S = 5.0
HAND_LINK = "gripper_center"
RELOCATE_HAND_M = 0.03
RELOCATE_CLICKS = 3
RELOCATE_SPACING_M = 0.005
RELOCATE_MIN_SEEN = 10
RELOCATE_DEPTH_M = 0.01
RELOCATE_RETRY_S = 15.0
RELOCATE_RETRY_MAX_S = 120.0
# The obstacles the server finds with no click and no mesh (pose_sever's
# OBSTACLE_MIN_PX; --obstacles, how many at most) are named OBSTACLE_PREFIX
# and their number in the init's reply, announced with the objects, and
# published as the objects without a mesh are: /detected/<name>_pose and
# _box, the box upright, and /point2pose/<name>/lost. An init sent again
# finds them again, numbered afresh; one it does not find is lost.
OBSTACLE_PREFIX = "obstacle_"
# An obstacle's box is published upright (upright_box()), and of the ways
# that leaves it -- x either way along its longer flat side, or along the
# other where the two are within OBSTACLE_SIDE_TIE of each other -- the one
# nearest the last published: the RI box, 272 x 269 mm, published from one
# reply to the next turned 180 deg on 12 of 29, the same box.
OBSTACLE_SIDE_TIE = 0.1


def read_obj(path, colours=False):
    """Vertices (float32, metres) and triangles (int32) of an OBJ mesh; with
    `colours`, also each vertex's RGB (uint8), or None unless every v line
    carries one after its position ("v x y z r g b", 0-1 or 0-255).

    Only v and f lines matter; polygons are fanned into triangles, and face
    indices may carry /texture/normal parts or count back from the end.
    """
    vertices, faces, rgb = [], [], []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
                if len(fields) >= 7:
                    rgb.append([float(x) for x in fields[4:7]])
            elif fields[0] == "f":
                ids = [int(x.split("/")[0]) for x in fields[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                faces.extend((ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1))
    v, f = np.asarray(vertices, np.float32), np.asarray(faces, np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError(f"{path}: no usable vertices and faces")
    if not colours:
        return v, f
    c = None
    if len(rgb) == len(v):
        c = np.asarray(rgb, np.float64)
        c = np.clip(np.round(c * 255 if c.max() <= 1.0 else c), 0, 255).astype(np.uint8)
    return v, f, c


def rigid(value):
    """`value` as a 4x4 rigid transform (nested lists), or None if it is not one."""
    T = np.asarray(value if value is not None else np.nan, dtype=float)
    if (T.shape == (4, 4) and np.isfinite(T).all() and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) and np.linalg.det(T[:3, :3]) > 0):
        return T.tolist()
    return None


def upright_box(rotation, extent):
    """The same box with z its most vertical axis (pointing up), x its longer flat side.

    The fit returns the axes in no useful order. A box is symmetric, so they
    can be relabelled freely; this order matches the camera path's length,
    width, height, and keeps a resting box clear of pitch = +/-90 degrees,
    where the rpy perception stores cannot represent the heading.
    """
    R = rotation.as_matrix()
    up = int(np.argmax(np.abs(R[2])))
    flat = [i for i in range(3) if i != up]
    long_i, short_i = sorted(flat, key=lambda i: extent[i], reverse=True)
    z = R[:, up] if R[2, up] >= 0 else -R[:, up]
    x = R[:, long_i]
    return (Rotation.from_matrix(np.column_stack([x, np.cross(z, x), z])),
            extent[[long_i, short_i, up]])


def mesh_box(vertices, linear):
    """(centre, extent) in the mesh's own frame of `vertices` mapped by
    `linear` (its fitted scale): the box of a part with a mesh pose, so the
    box shown and published has the mesh's axes, the ones the Agent plans
    in, rather than axes of its own."""
    scaled = np.asarray(vertices, dtype=np.float64) @ np.asarray(linear, dtype=np.float64).T
    lo, hi = scaled.min(0), scaled.max(0)
    return (lo + hi) / 2, hi - lo


def robot_visuals(urdf):
    """(link, T_link_mesh, scale, uri) of each visual mesh of a URDF's links.

    Only a link's own <visual> children count: the <gazebo> blocks inside
    some links repeat them. Other shapes are left out; none of the cell's
    links has one.
    """
    out = []
    for link in ElementTree.fromstring(urdf).findall("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            T = np.eye(4)
            origin = visual.find("origin")
            if origin is not None:
                rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
                T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                T[:3, 3] = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            scale = np.array([float(x) for x in mesh.get("scale", "1 1 1").split()])
            out.append((link.get("name"), T, scale, mesh.get("filename")))
    return out


def robot_joints(urdf):
    """{child link: (parent link, T_parent_child)} of a URDF's joints, each at
    its zero position: where a joint's link has no TF of its own, its parent's
    places it."""
    out = {}
    for joint in ElementTree.fromstring(urdf).findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        T = np.eye(4)
        origin = joint.find("origin")
        if origin is not None:
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
            T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            T[:3, 3] = [float(x) for x in origin.get("xyz", "0 0 0").split()]
        out[child.get("link")] = (parent.get("link"), T)
    return out


def moving_links(urdf):
    """The links a joint other than a fixed one moves, and those on them:
    the arm and the gripper, not the cart and the arm's base."""
    joints = [(j.find("parent"), j.find("child"), j.get("type")) for j in ElementTree.fromstring(urdf).findall("joint")]
    joints = [(parent.get("link"), child.get("link"), kind) for parent, child, kind in joints
              if parent is not None and child is not None]
    moving = {child for _, child, kind in joints if kind != "fixed"}
    while True:
        more = {child for parent, child, _ in joints if parent in moving} - moving
        if not more:
            return moving
        moving |= more


def mesh_path(uri):
    """A file path for a URDF mesh: file://, package:// or a plain path."""
    if uri.startswith("file://"):
        return uri[len("file://"):]
    if uri.startswith("package://"):
        from ament_index_python.packages import get_package_share_directory
        package, _, rest = uri[len("package://"):].partition("/")
        return os.path.join(get_package_share_directory(package), rest)
    return uri


def surface_samples(vertices, faces, spacing, rng):
    """Points on a mesh's surface, by area, about `spacing` apart."""
    tri = np.asarray(vertices, np.float64)[faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    if area.sum() <= 0:
        return np.zeros((0, 3))
    n = int(min(ROBOT_MAX_SAMPLES, np.ceil(area.sum() / spacing ** 2)))
    face = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    return ((1 - r1)[:, None] * tri[face, 0] + (r1 * (1 - r2))[:, None] * tri[face, 1]
            + (r1 * r2)[:, None] * tri[face, 2])


def robot_pixels(metres, K, points, margin):
    """The pixels whose depth lies within `margin` of `points`, the robot's
    surface in the camera frame: a mask of the depth's shape."""
    mask = np.zeros(metres.shape, bool)
    v, u = np.nonzero(np.isfinite(metres) & (metres > 0))
    if not len(points) or not len(v):
        return mask
    z = metres[v, u]
    P = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    d, _ = cKDTree(points).query(P, distance_upper_bound=margin, workers=-1)
    near = np.isfinite(d)
    mask[v[near], u[near]] = True
    return mask


def encode_mask(mask):
    """A mask as pose_sever.decode_robot_mask() reads it."""
    return {"shape": list(mask.shape),
            "bits": base64.b64encode(zlib.compress(np.packbits(mask).tobytes())).decode("ascii")}


def off_robot(points, labels, robot):
    """(points, labels, dropped): each object's clicks less those on the
    robot's pixels (`robot`, None: none known) -- a click sent on the arm or
    the gripper makes the server's segment of the part the robot's. Only the
    clicks the client is started with are so checked (an upstream VLM's or
    a person's); `dropped`, per object, how many went."""
    if robot is None:
        return points, labels, [0] * len(points)
    h, w = robot.shape
    out_points, out_labels, dropped = [], [], []
    for pts, lbls in zip(points, labels):
        keep = [k for k, (u, v) in enumerate(pts)
                if not robot[min(max(int(round(v)), 0), h - 1), min(max(int(round(u)), 0), w - 1)]]
        out_points.append([pts[k] for k in keep])
        out_labels.append([lbls[k] for k in keep])
        dropped.append(len(pts) - len(keep))
    return out_points, out_labels, dropped


def safe_grasps(grasps, linear, turns, tol_m, tol_deg, flip_axis=None):
    """Indices of `grasps` (4x4, the gripper in the mesh's frame and units)
    that every way round in `turns` (4x4 relative to the pose published, in
    the fitted mesh's frame) puts within `tol_m` and `tol_deg` of where the
    pose published puts them -- or turned half round about `flip_axis` (in
    the gripper's frame) of it."""
    linear = np.asarray(linear, dtype=float)
    flip = None if flip_axis is None else Rotation.from_rotvec(np.pi * np.asarray(flip_axis, float)).as_matrix()
    out = []
    for i, G in enumerate(grasps):
        G = np.asarray(G, dtype=float).copy()
        G[:3, 3] = linear @ G[:3, 3]
        ok = True
        for D in turns:
            H = np.asarray(D, dtype=float) @ G
            turn = G[:3, :3].T @ H[:3, :3]
            angles = [np.linalg.norm(Rotation.from_matrix(turn).as_rotvec())]
            if flip is not None:
                angles.append(np.linalg.norm(Rotation.from_matrix(turn @ flip).as_rotvec()))
            if np.linalg.norm(H[:3, 3] - G[:3, 3]) > tol_m or np.degrees(min(angles)) > tol_deg:
                ok = False
                break
        if ok:
            out.append(i)
    return out


def box_corners(T, extent):
    """The 8 corners of a box of `extent` at pose T, bit 0/1/2 of the index
    flipping x/y/z (BOX_EDGES)."""
    signs = np.array([[(i >> k & 1) * 2 - 1 for k in range(3)] for i in range(8)], dtype=np.float64)
    return (signs * np.asarray(extent, dtype=np.float64) / 2) @ T[:3, :3].T + T[:3, 3]


def box_distance(point, T, extent):
    """How far `point` lies outside a box of `extent` at pose T; 0 inside."""
    local = (np.asarray(point, dtype=np.float64) - T[:3, 3]) @ T[:3, :3]
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(extent, dtype=np.float64) / 2, 0.0)))


def obstacles_by_name(obstacles):
    """A reply's obstacles ({"id", "lost", "bbox"}) by the name each is
    followed under (OBSTACLE_PREFIX and its id), in the reply's order."""
    found = {}
    for item in obstacles if isinstance(obstacles, list) else []:
        try:
            found[f"{OBSTACLE_PREFIX}{int(item['id'])}"] = item
        except (KeyError, TypeError, ValueError):
            continue
    return found


def reply_box(bbox):
    """(pose 4x4, extent) of a reply's box, or None."""
    try:
        T = np.asarray(bbox["pose"], dtype=np.float64)
        extent = np.asarray(bbox["extent"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if T.shape != (4, 4) or extent.shape != (3,) or not (np.isfinite(T).all() and np.isfinite(extent).all()):
        return None
    return T, extent


class PoseClient(Node):
    def __init__(self, args):
        super().__init__("point2pose_client")
        self.bridge = CvBridge()
        self.args = args
        self.K = None
        self.initialized = False
        self._frames = LatestFrame()
        self._frame_received = None
        self.object_names = args.objects
        # name -> (vertices, faces); sent once, with the init request.
        self.models = args.models
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # link -> surface samples in the link's frame; see robot_mask().
        self.robot_links = None
        self.started = time.time()
        self.create_subscription(
            String, ROBOT_DESCRIPTION_TOPIC, self.on_robot_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.named_pubs = {
            name: self.create_publisher(PoseStamped, f"/detected/{name}_pose", 10)
            for name in self.object_names
        }
        self.lost_pubs = {
            name: self.create_publisher(Bool, f"/point2pose/{name}/lost", 10)
            for name in self.object_names
        }
        # Same type the camera path fills: centre, own axes and extents, which
        # is what DetectedObject.footprint() turns into a grasp direction.
        self.box_pubs = {
            name: self.create_publisher(BoundingBox3D, f"/detected/{name}_box", 10)
            for name in self.object_names
        }
        # name -> (anchor->box 4x4, extent); see update_boxes().
        self.box_offsets = {}
        self.model_offsets = {}
        # name -> what was last said of its mesh pose; see say_mesh_changes().
        self.mesh_said = {}
        self.model_pubs = {name: self.create_publisher(String, f"/detected/{name}_model", 10)
                           for name in self.object_names}
        # Each object's observed surface, relative to its published pose: the
        # server's memory of every view, so what is hidden now still counts.
        self.surface_pubs = {name: self.create_publisher(String, f"/detected/{name}_surface", 10)
                             for name in self.object_names}
        self.surfaces = {}
        # Replies in a row each object has been lost in, the pixels it was last
        # seen on (u0, v0, u1, v1), and the clicks of an init sent again; see
        # relocate().
        self.lost_replies = {name: 0 for name in self.object_names}
        self.last_seen_px = {}
        self.relocate_points = None
        self.last_relocate = 0.0
        # The box each was last seen in, and how near the hand has come to it
        # since (relocate_seen()); the parts the init sent again locates by
        # their meshes, and the clicks of those it clicks in the hand, with
        # the hand's pose then: {index: (camera points, hand)}.
        self.last_seen_box, self.hand_near = {}, {}
        # Each part's mesh pose where last seen, and the camera it was seen
        # in: where a pose prior moves its last seen box to (set_down_seen()).
        self.last_seen_model, self.camera_frame = {}, None
        self.relocate_by_mesh, self.relocate_carried = [], {}
        # The parts an init sent again keeps the fits of (relocate()).
        self.relocate_keep = []
        # Inits sent again that failed in a row, and when the next may go.
        self.relocate_failures, self.relocate_retry_at = 0, 0.0
        # (time, both fingertips in contact) of the last tactile reading.
        self.tactile = None
        # The clicks asked for (ask_for_clicks()): {"points", "labels", "asked",
        # "candidates", "shape"}, None while nothing is asked.
        self.awaiting_clicks = None
        # The obstacles found, by name, and their publishers, made as they
        # are first found (OBSTACLE_PREFIX); the box rotation each was last
        # published with (OBSTACLE_SIDE_TIE).
        self.obstacle_names = []
        self.obstacle_pubs = {}
        self.obstacle_last = {}

        self.ctx = zmq.Context()
        self.sock = None
        self._connect()
        # Set by a timeout: the server is still working on that request.
        self.server_busy = False
        self.get_logger().info(f"connected to {args.server}")

        self.create_subscription(CameraInfo, args.info_topic, self.on_info, 10)
        # Image callbacks keep running while the default group waits on the
        # server. Only the newest synchronized pair survives that wait.
        self._image_group = MutuallyExclusiveCallbackGroup()
        rgb_sub = message_filters.Subscriber(self, Image, args.rgb_topic,
                                            qos_profile=QoSProfile(depth=1), callback_group=self._image_group)
        depth_sub = message_filters.Subscriber(self, Image, args.depth_topic,
                                              qos_profile=QoSProfile(depth=1), callback_group=self._image_group)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 2, 0.05)
        sync.registerCallback(self.on_frame)
        self._frame_timer = self.create_timer(0.01, self.process_latest_frame)
        self.create_subscription(String, TACTILE_TOPIC, self.on_tactile, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.click_request_pub = self.create_publisher(String, CLICK_REQUEST_TOPIC, latched)
        self.click_image_pub = self.create_publisher(Image, CLICK_REQUEST_IMAGE_TOPIC, latched)
        self.create_subscription(String, CLICK_TOPIC, self.on_click, 10)
        # Ways round asked about, by name ({"fit_id", "ids", "features"}), and
        # the answers to send with the next frame, by object index; see
        # BINDING_REQUEST_TOPIC.
        self.binding_asked, self.binding_answers = {}, {}
        # Pose priors to send with the next frame, by name: (world pose, source).
        self.pose_priors = {}
        self.create_subscription(String, POSE_PRIOR_TOPIC, self.on_pose_prior, 10)
        self.binding_request_pub = self.create_publisher(String, BINDING_REQUEST_TOPIC, latched)
        self.binding_image_pub = self.create_publisher(Image, BINDING_REQUEST_IMAGE_TOPIC, latched)
        self.create_subscription(String, BINDING_TOPIC, self.on_binding, 10)

        self.pub = self.create_publisher(PoseArray, "/point2pose/poses", 10)
        self.bbox_pub = self.create_publisher(MarkerArray, BBOX_TOPIC, 10)
        # Vision may already be running with no selected objects. Retain this
        # session's names so it can subscribe even if discovery happens later.
        self.objects_pub = self.create_publisher(
            String, "/point2pose/objects",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # Props (obstacles, occluders) another node publishes are announced
        # too, or the Agent, which follows this list, would not know of them.
        self.objects_pub.publish(String(data=",".join([*self.object_names, *args.props])))

    def on_tactile(self, msg):
        try:
            fingers = json.loads(msg.data)["fingers"]
            both = all(bool((fingers.get(side) or {}).get("contact")) for side in ("left", "right"))
        except Exception:
            return
        self.tactile = (time.time(), both)

    def touching(self) -> bool:
        """Whether both fingertips touched something on the last reading, one
        no older than TACTILE_FRESH_S."""
        reading = getattr(self, "tactile", None)
        return bool(reading and reading[1] and time.time() - reading[0] <= TACTILE_FRESH_S)

    def on_info(self, msg):
        if self.K is None:
            self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(f"got intrinsics fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f}")

    def on_robot_description(self, msg):
        if self.robot_links is not None:
            return
        rng = np.random.default_rng(0)
        links, skipped = {}, []
        for link, T, scale, uri in robot_visuals(msg.data):
            if link in self.args.not_robot:
                continue
            try:
                vertices, faces = read_obj(mesh_path(uri))
            except (OSError, ValueError) as exc:
                skipped.append(f"{link} ({os.path.basename(uri)}: {type(exc).__name__})")
                continue
            points = surface_samples(vertices * scale, faces, ROBOT_SPACING_M, rng)
            points = points @ T[:3, :3].T + T[:3, 3]
            links[link] = np.concatenate([links[link], points]) if link in links else points
        self.robot_links = links
        self.robot_joints = robot_joints(msg.data)
        self.robot_moving = moving_links(msg.data)
        self.get_logger().info(
            f"robot mask from {len(links)} links, {sum(map(len, links.values()))} surface points"
            + (f"; no OBJ mesh read for {', '.join(skipped)}" if skipped else ""))

    def robot_mask(self, metres, frame, moving=False):
        """The robot's pixels in this depth image (robot_pixels()), or None
        with no robot description or under half its links in TF yet; with
        `moving`, those of the links that move alone (moving_links())."""
        if not self.robot_links:
            return None
        links = {link: points for link, points in self.robot_links.items()
                 if not moving or link in getattr(self, "robot_moving", ())}
        clouds, missing = [], []
        for link, points in links.items():
            T = self.link_in(frame, link)
            if T is None:
                missing.append(link)
                continue
            clouds.append(points @ T[:3, :3].T + T[:3, 3])
        if not links or 2 * len(clouds) < len(links):
            return None
        if missing:
            self.get_logger().warn(f"robot mask: no transform to {frame} for {', '.join(missing)}",
                                   throttle_duration_sec=30.0)
        return robot_pixels(metres, self.K, np.concatenate(clouds), ROBOT_MARGIN_M)

    def link_in(self, frame, link, depth=0, stamp=None):
        """`link`'s pose in `frame` (4x4) from TF -- at `stamp` (a message's)
        where TF still holds that time, else its latest -- or else from its
        parent's and its URDF joint at zero; None where neither is known. The
        finger pads' joints are in no /joint_states, so they have no TF, and
        the pads -- what touches a part -- were left out of the robot's
        pixels."""
        tf = None
        for when in ([Time.from_msg(stamp)] if stamp is not None else []) + [Time()]:
            try:
                tf = self.tf_buffer.lookup_transform(frame, link, when)
                break
            except Exception:
                continue
        if tf is None:
            joint = getattr(self, "robot_joints", {}).get(link)
            if joint is None or depth > 3:
                return None
            parent = self.link_in(frame, joint[0], depth + 1, stamp)
            return None if parent is None else parent @ joint[1]
        q, t = tf.transform.rotation, tf.transform.translation
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T[:3, 3] = [t.x, t.y, t.z]
        return T

    def _connect(self):
        """A fresh REQ socket: one whose reply timed out cannot send again."""
        if self.sock is not None:
            self.sock.close()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.args.server)

    def server_free(self) -> bool:
        """After a timeout, wait for the server to finish that request first.

        The server answers strictly in order and cannot drop what it is doing,
        so a request sent straight after a timeout queues behind the one that
        timed out and waits for both. For an init that took longer than the
        timeout that is a certain second timeout, then a third: every retry
        rebuilt the pipeline, and the client never recovered. A ping is
        answered only once the server has caught up, and costs it nothing.
        """
        try:
            self.sock.setsockopt(zmq.RCVTIMEO, self.args.init_timeout_ms)
            self.sock.send_json({"cmd": "ping"})
            self.sock.recv_json()
        except zmq.Again:
            self.get_logger().error("server timeout; still busy with the last request")
            self._connect()
            return False
        self.server_busy = False
        self.get_logger().info("server caught up; resuming")
        return True

    def mesh_frames(self, meta):
        """Each object's mesh, if it has one: counts in meta["models"], one
        frame each after rgb and depth (zlib of float32 vertices, int32 faces)."""
        meta["models"], frames = [], []
        for name in self.object_names:
            if name not in self.models:
                meta["models"].append(None)
                continue
            vertices, faces = self.models[name]
            colours = self.args.model_colours.get(name)
            meta["models"].append({"vertices": len(vertices), "faces": len(faces)})
            if self.args.true_size_meshes:
                meta["models"][-1]["known_scale"] = 1.0
            if self.args.mesh_error is not None:
                meta["models"][-1]["mesh_error"] = float(self.args.mesh_error)
            if self.args.mesh_stretch_max is not None:
                meta["models"][-1]["stretch_max"] = float(self.args.mesh_stretch_max)
            if self.args.mesh_unit is not None:
                meta["models"][-1]["unit_m"] = float(self.args.mesh_unit)
            if colours is not None:
                meta["models"][-1]["colours"] = True
            frames.append(zlib.compress(vertices.astype("<f4").tobytes()
                                        + faces.astype("<i4").tobytes()
                                        + (b"" if colours is None else colours.tobytes())))
        return frames

    def request(self, meta, rgb, depth, extra=()):
        # Init builds the pipeline and each object's model, so it takes far
        # longer than a tracking frame, and longer again with every object.
        self.sock.setsockopt(zmq.RCVTIMEO, self.args.init_timeout_ms
                             if meta["cmd"] == "init" else self.args.timeout_ms)
        ok, rgb_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        if not ok:
            raise RuntimeError("rgb encode failed")
        if depth.dtype == np.float32:
            meta["depth_enc"] = "raw_f32"
            meta["depth_shape"] = list(depth.shape)
            depth_buf = zlib.compress(np.ascontiguousarray(depth).tobytes(), 1)
        else:
            meta["depth_enc"] = "png"
            depth_buf = cv2.imencode(".png", depth)[1].tobytes()
        self.sock.send_multipart([json.dumps(meta).encode(), rgb_buf.tobytes(), depth_buf,
                                  *extra])
        return self.sock.recv_json()

    def on_frame(self, rgb_msg, depth_msg):
        self._frames.put(rgb_msg, depth_msg)

    def process_latest_frame(self):
        # A blocking server request exceeds the timer period. Without rearming
        # after it returns, the always-ready timer starves binding subscriptions
        # in this same mutually exclusive callback group.
        self._frame_timer.cancel()
        try:
            latest = self._frames.take(self.args.rate)
            if latest is None:
                return
            rgb, depth, self._frame_received = latest
            self.track_frame(rgb, depth)
        finally:
            self._frame_timer.reset()

    def observation_age(self):
        received = getattr(self, "_frame_received", None)
        return max(0.0, time.monotonic() - received) if received is not None else 0.0

    def track_frame(self, rgb_msg, depth_msg):
        if self.K is None:
            return
        if (not self.initialized and self.robot_links is None
                and time.time() - self.started < ROBOT_WAIT_S):
            return
        now = time.time()
        if self.server_busy:
            # This frame is old by the time the server is free; send the next.
            self.server_free()
            return
        if self.relocation_waits(now):
            return
        if self.awaiting_clicks is not None:
            waiting = [name for name in self.awaiting_clicks["asked"] if self.awaiting_clicks["points"][
                self.object_names.index(name)] is None]
            self.get_logger().info(f"waiting for a click on {', '.join(waiting)} ({CLICK_TOPIC})",
                                   throttle_duration_sec=10.0)
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        depth_factor = self.args.depth_factor
        if depth.dtype == np.float32:
            # float32 metres: send raw bytes so no precision is lost to quantisation.
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_factor = 1.0

        meta = {
            "cmd": "track" if self.initialized else "init",
            "K": self.K.flatten().tolist(),
            "depth_factor": depth_factor,
            "timestamp": rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9,
        }
        extra = ()
        if self.initialized:
            # How long this client waits: the server answers within part of
            # it, stale where the frame is not done (pose_sever's
            # SERVER_REPLY_SHARE), so the socket is not reset under it.
            meta["timeout_ms"] = self.args.timeout_ms
            # The robot on every frame too: the server takes it out of the
            # parts' masks, where a gripper on or by a part would be read as
            # the part (pose_sever's without_robot()).
            robot = self.robot_mask(depth.astype(np.float64) / depth_factor, rgb_msg.header.frame_id)
            if robot is not None:
                meta["robot_mask"] = encode_mask(robot)
            # The robot's poses of parts it set down (POSE_PRIOR_TOPIC), in this camera.
            priors = self.camera_priors(rgb_msg.header.frame_id)
            if priors:
                meta["priors"] = priors
            # The person's ways round (BINDING_TOPIC), until a reply takes them.
            if self.binding_answers:
                meta["bind"] = [{"object": index, "hypothesis": ident, "by": by}
                                for index, (ident, by) in self.binding_answers.items()]
        if not self.initialized:
            meta["points"], meta["labels"] = self.relocate_points or (json.loads(self.args.points),
                                                                       json.loads(self.args.labels))
            if self.relocate_points:
                meta["points"] = self.carried_clicks(meta["points"], rgb_msg.header.frame_id, rgb.shape[:2])
                if self.relocate_keep:
                    meta["keep"] = [self.object_names.index(name) for name in self.relocate_keep]
            robot = self.robot_mask(depth.astype(np.float64) / depth_factor, rgb_msg.header.frame_id)
            if robot is None and self.robot_links and time.time() - self.started < ROBOT_WAIT_S:
                self.get_logger().info("waiting for the robot's transforms", throttle_duration_sec=5.0)
                return
            if not self.relocate_points:
                meta["points"], meta["labels"], dropped = off_robot(meta["points"], meta["labels"], robot)
                for name, n in zip(self.object_names, dropped):
                    if n:
                        self.get_logger().info(f"{name}: {n} of its clicks on the robot left out")
            extra = self.mesh_frames(meta)
            meta["discover"] = self.args.obstacles
            meta["same_scale_meshes"] = self.args.same_scale_meshes
            if getattr(self.args, "baseline", False):
                meta["baseline"] = True         # the unmodified Point2Pose (pose_sever.BASELINE)
            robot_note = "not known"
            if robot is not None:
                meta["robot_mask"] = encode_mask(robot)
                robot_note = f"on {int(robot.sum())} px"
            self.get_logger().info(
                f"initializing {len(self.object_names)} object(s), {len(extra)} with a "
                f"mesh; robot {robot_note}; allowing {self.args.init_timeout_ms / 1000:.0f} s")

        try:
            reply = self.request(meta, rgb, depth, extra)
        except zmq.Again:
            self._invalidate_all()
            self.get_logger().error(f"server timeout on {meta['cmd']}; resetting socket")
            self._connect()
            self.server_busy = True
            self.relocation_failed()
            return

        # How the server located and fitted the objects, line by line, and
        # how long each took; the same lines as its own console.
        for line in reply.get("init_report") or []:
            self.get_logger().info(f"server: {line}")
        if not reply.get("ok") and reply.get("need_click") and meta["cmd"] == "init":
            self._invalidate_all()
            self.ask_for_clicks(reply, rgb, rgb_msg.header)
            return
        if not reply.get("ok"):
            self._invalidate_all()
            self.get_logger().error(f"server error: {reply.get('error')}")
            self.relocation_failed()
            return
        if meta["cmd"] == "init" and getattr(self.args, "baseline", False) and reply.get("baseline") is not True:
            # A server without the baseline mode tracks as the full method: the
            # comparison would be the full method under the baseline's name.
            self.get_logger().fatal("--baseline asked for, but the server does not run as the unmodified "
                                    "Point2Pose (its pose_sever.py predates the baseline mode): stopping")
            raise SystemExit(2)
        stale = reply.get("stale")
        if isinstance(stale, dict) and meta["cmd"] == "track":
            # The server is still on an earlier frame: these are that frame's
            # poses, already published; nothing new to publish, nor to reset.
            self.get_logger().warn(
                f"server still working on frame {stale.get('working_on')} ({stale.get('busy_s')} s); "
                f"no new poses this frame", throttle_duration_sec=2.0)
            return
        if not self.initialized:
            self.initialized = True
            # Ways round asked of an earlier init are no longer the server's,
            # but for the parts that kept their fits (pose_sever's carry_kept_fits()).
            kept = {self.object_names[index] for index in reply.get("kept") or []
                    if isinstance(index, int) and 0 <= index < len(self.object_names)}
            for name in list(self.binding_asked):
                if name not in kept:
                    self.withdraw_binding(name)
            self.binding_answers = {index: answer for index, answer in self.binding_answers.items()
                                    if self.object_names[index] in kept}
            if kept:
                self.get_logger().info(f"{', '.join(sorted(kept))}: kept the fit as last seen")
            self.relocate_points = None
            self.relocate_by_mesh, self.relocate_carried, self.relocate_keep = [], {}, []
            self.relocate_failures = 0
            self.lost_replies = {name: 0 for name in self.object_names}
            self.get_logger().info("pipeline initialized")
            # The clicks the server used: an object sent without points was
            # located by its mesh (pose_sever's locate_unclicked()).
            used = reply.get("points")
            if isinstance(used, list) and len(used) == len(self.object_names):
                for name, sent, click in zip(self.object_names, meta["points"], used):
                    if not sent and name not in kept:
                        self.get_logger().info(f"{name}: located by its mesh at {click}")
            for name, error in zip(self.object_names, reply.get("model_errors") or []):
                if error:
                    self.get_logger().warn(f"{name}: mesh pose unavailable ({error})")

        try:
            poses = np.asarray(reply["poses"], dtype=np.float64)
            lost = list(reply["lost"])
        except (KeyError, TypeError, ValueError) as exc:
            self._invalidate_all()
            self.get_logger().error(f"invalid server pose reply: {exc}")
            return
        if poses.shape != (len(self.object_names), 4, 4) or len(lost) != len(self.object_names):
            self._invalidate_all()
            self.get_logger().error("server returned the wrong number or shape of poses")
            return
        # The hand where it was when the image was taken, as the boxes are:
        # its latest pose, some 2 s on, had carried the tester away from its
        # last box, and the tester lost in the hand was located again by its
        # mesh -- not found there, the person asked to click it (PP-S1-4,
        # 2026-11-09).
        self.camera_frame = rgb_msg.header.frame_id
        self.relocate_seen(lost, reply.get("bboxes"),
                           self.link_in(rgb_msg.header.frame_id, HAND_LINK, stamp=rgb_msg.header.stamp))
        # Apply mesh recognition validity before publishing even the raw array:
        # an anchor tracked after a failed mesh fit is not an object pose.
        self.update_boxes(poses, lost, reply.get("bboxes"))
        if meta.get("bind"):
            self.binding_answers = {}
        if meta.get("priors"):
            self.pose_priors = {}
        age = self.observation_age()
        if meta["cmd"] == "init":
            # An init's obstacles are named now, its frame however old: an
            # init takes 20 s and more, so its reply was always past
            # max_pose_age and dropped below before its obstacles were named,
            # and none was ever announced -- 45 inits of batch 20261112_214052,
            # the person's hand on the bar and PP-S3's box among them. Their
            # boxes are published with the first reply fresh enough.
            lost_markers = MarkerArray()
            self.name_obstacles(reply.get("obstacles"), lost_markers)
            if lost_markers.markers:
                self.bbox_pub.publish(lost_markers)
        if age > self.args.max_pose_age:
            self._invalidate_all()
            self.get_logger().warn(f"discarding pose from an image received {age:.2f}s ago; "
                                   "waiting for the latest frame", throttle_duration_sec=2.0)
            return
        self.ask_binding(reply.get("bboxes"), rgb, rgb_msg.header)
        self.say_mesh_changes(reply.get("model_errors"))
        out = PoseArray()
        out.header = rgb_msg.header
        for index, T in enumerate(poses):
            p = Pose()
            # Keep the index for this object, but make a rejected slot invalid
            # explicitly: some ROS message builds default to identity rotation.
            p.orientation.w = 0.0
            if not lost[index] and np.all(np.isfinite(T)):
                try:
                    q = Rotation.from_matrix(T[:3, :3]).as_quat()
                except ValueError:
                    lost[index] = True
                else:
                    p.position.x, p.position.y, p.position.z = T[:3, 3]
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            else:
                lost[index] = True
            out.poses.append(p)
        self.pub.publish(out)
        self.publish_world_poses(out, lost)
        markers = self.bbox_markers(reply.get("bboxes"), lost, rgb_msg.header, poses)
        markers.markers += self.publish_obstacles(reply.get("obstacles"), rgb_msg.header,
                                                  renamed=meta["cmd"] == "init").markers
        self.bbox_pub.publish(markers)
        self.get_logger().info(
            f"frame {reply['frame_id']} latency={reply['latency_ms']}ms lost={lost} "
            f"observation_age={self.observation_age():.3f}s",
            throttle_duration_sec=2.0,
        )
        if meta["cmd"] == "track":
            self.relocate(lambda: self.robot_mask(depth.astype(np.float64) / depth_factor,
                                                  rgb_msg.header.frame_id, moving=True),
                          reply.get("bboxes"), depth.astype(np.float64) / depth_factor,
                          rgb_msg.header.frame_id)

    def ask_for_clicks(self, reply, rgb, header):
        """The server's need_click (CLICK_REQUEST_TOPIC): the objects it could
        not locate for sure wait for the person's click (on_click()), the
        others keep the clicks it located them at."""
        sure = reply.get("points") or []
        sent, sent_labels = json.loads(self.args.points), json.loads(self.args.labels)
        points, labels = [], []
        for index in range(len(self.object_names)):
            # Where it was located, else as sent (no click: located again).
            click = sure[index] if index < len(sure) and sure[index] else None
            points.append(click if click else sent[index])
            labels.append([1] * len(click) if click else sent_labels[index])
        asked, why, candidates = [], {}, {}
        for request in reply["need_click"]:
            index = int(request["object"])
            if not 0 <= index < len(self.object_names):
                continue
            name = self.object_names[index]
            points[index] = labels[index] = None
            asked.append(name)
            why[name] = str(request.get("why", ""))
            candidates[name] = [c for c in (request.get("candidates") or []) if isinstance(c, list) and c]
        self.awaiting_clicks = {"points": points, "labels": labels, "asked": asked, "candidates": candidates,
                                "shape": rgb.shape[:2]}
        self.click_request_pub.publish(String(data=json.dumps({
            "objects": asked, "why": why, "candidates": candidates,
            "frame_id": header.frame_id, "stamp": header.stamp.sec + header.stamp.nanosec * 1e-9})))
        shown = rgb.copy()
        for name in asked:
            for k, click in enumerate(candidates[name]):
                for u, v in click:
                    cv2.circle(shown, (int(u), int(v)), 4, (0, 255, 255), -1, cv2.LINE_AA)
                u, v = np.asarray(click, dtype=float).mean(0)
                cv2.putText(shown, f"{name} #{k}", (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(shown, f"{name} #{k}", (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
        image = self.bridge.cv2_to_imgmsg(shown, "bgr8")
        image.header = header
        self.click_image_pub.publish(image)
        for name in asked:
            self.get_logger().warn(
                f"{name}: the server is not sure where it is ({why[name]}); click it: publish "
                f"{{\"object\": \"{name}\", \"points\": [[u, v]]}} or {{\"object\": \"{name}\", \"candidate\": k}} "
                f"on {CLICK_TOPIC} ({len(candidates[name])} candidates on {CLICK_REQUEST_IMAGE_TOPIC})")

    def on_click(self, msg):
        """A click asked for (ask_for_clicks()); with every one in, the init is
        sent again with them all."""
        wait = self.awaiting_clicks
        try:
            answer = json.loads(msg.data)
            name = answer["object"]
            if wait is None or name not in wait["asked"]:
                self.get_logger().warn(f"click on {name}: none asked for")
                return
            if "candidate" in answer:
                points = wait["candidates"][name][int(answer["candidate"])]
            else:
                points = [[float(u), float(v)] for u, v in answer["points"]]
            labels = [int(x) for x in answer.get("labels", [1] * len(points))]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.get_logger().error(f"unusable click on {CLICK_TOPIC}: {exc}")
            return
        h, w = wait["shape"]
        if (not points or len(labels) != len(points) or not any(labels)
                or not all(0 <= u < w and 0 <= v < h for u, v in points)):
            self.get_logger().error(f"{name}: a click needs a point on it (label 1), inside the {w} x {h} image")
            return
        index = self.object_names.index(name)
        wait["points"][index], wait["labels"][index] = points, labels
        self.get_logger().info(f"{name}: clicked at {points}")
        if all(p is not None for p in wait["points"]):
            self.relocate_points = (wait["points"], wait["labels"])
            self.relocate_by_mesh, self.relocate_carried = [], {}
            self.awaiting_clicks = None
            self.get_logger().info("every object clicked or located; initializing with those clicks")

    def on_pose_prior(self, msg):
        """A part's pose from whoever set it down (POSE_PRIOR_TOPIC)."""
        try:
            prior = json.loads(msg.data)
            name = prior["object"]
            pose = rigid(prior["pose"])
            if name not in self.object_names or pose is None:
                raise ValueError(f"needs one of {self.object_names} and a rigid 4x4 pose")
        except (ValueError, KeyError, TypeError) as exc:
            self.get_logger().error(f"unusable pose prior on {POSE_PRIOR_TOPIC}: {exc}")
            return
        self.pose_priors[name] = (np.asarray(pose, dtype=float), str(prior.get("source") or "client"))
        self.get_logger().info(f"{name}: pose prior from {self.pose_priors[name][1]}, sent with the next frame")
        if self.pose_priors[name][1] == HOLD_PRIOR_SOURCE:
            return                                  # still in the hand: not set down
        self.set_down_seen(name, self.pose_priors[name][0])
        wait = self.awaiting_clicks
        if wait is not None and name in wait["asked"] and wait["points"][self.object_names.index(name)] is None:
            # A click asked on it while it was in the hand, out of view, waits
            # for nothing now: it is set down, where the init locates it by its
            # mesh. PP-S1-4's tester, lost in the hand, waited for a click the
            # person could not give, and no frame went after it (2026-11-09).
            missing = [n for n in wait["asked"] if wait["points"][self.object_names.index(n)] is None]
            points = [p if p is not None else [] for p in wait["points"]]
            labels = [l if l is not None else [] for l in wait["labels"]]
            self.relocate_points = (points, labels)
            self.relocate_by_mesh, self.relocate_carried = missing, {}
            self.awaiting_clicks = None
            self.get_logger().info(f"{', '.join(missing)}: no click waited for now it is set down; "
                                   "located again by its mesh")

    def set_down_seen(self, name, pose):
        """A part lost, set down at `pose` (world, its mesh's): where it was
        last seen is where it is set down now, its box carried there with its
        mesh, for relocate() to wait on the robot clear of. PP-S3-2's tester,
        lost in the hand up in the air and set down on the bench, waited 60 s
        and more on the arm, gone home, to leave where it was last seen -- in
        the image, where the arm at home is (batch 20261113_keepfit)."""
        box, model = self.last_seen_box.get(name), self.last_seen_model.get(name)
        if not self.lost_replies.get(name, 0) or box is None or model is None:
            return
        cam_world = self.camera_from_world(self.camera_frame, "pose prior's last seen box")
        if cam_world is None:
            return
        T = cam_world @ pose @ np.linalg.inv(model) @ box[0]
        moved = {"pose": T.tolist(), "extent": box[1].tolist()}
        pixels = self.box_pixels(moved)
        if pixels is None:
            return
        self.last_seen_px[name], self.last_seen_box[name] = pixels, (T, box[1])
        self.last_seen_model[name] = cam_world @ pose
        self.hand_near.pop(name, None)
        self.get_logger().info(f"{name}: last seen where it was set down, for locating it again")

    def camera_from_world(self, frame, what):
        """The world in the camera `frame`, 4x4, or None."""
        if not frame:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(frame, "world", Time())
        except Exception as exc:
            self.get_logger().warn(f"no TF {frame} <- world for the {what}: {exc}", throttle_duration_sec=5.0)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        cam_world = np.eye(4)
        cam_world[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        cam_world[:3, 3] = [t.x, t.y, t.z]
        return cam_world

    def camera_priors(self, frame):
        """The pose priors waiting, in the camera `frame`: meta["priors"]."""
        if not self.pose_priors or not frame:
            return []
        cam_world = self.camera_from_world(frame, "pose priors")
        if cam_world is None:
            return []
        return [{"object": self.object_names.index(name), "pose": (cam_world @ pose).tolist(), "source": source}
                for name, (pose, source) in self.pose_priors.items()]

    def ask_binding(self, bboxes, rgb, header):
        """Ask the person which way round, where a mesh's binding is in doubt
        and the doubt matters (--ask-binding); withdraw what the views have
        settled since. See BINDING_REQUEST_TOPIC. While another node asks
        (BINDING_ASKER_TOPIC), every way round in doubt is kept as asked, for
        its answers, and nothing is shown."""
        if not isinstance(bboxes, list) or len(bboxes) != len(self.object_names):
            return
        agent_asks = self.args.ask_binding != "always" and self.agent_asking()
        if agent_asks != getattr(self, "agent_asks", False):
            self.agent_asks = agent_asks
            self.get_logger().info("the Agent asks the person which way round; no request of mine" if agent_asks
                                   else "no Agent asks which way round: I ask, as --ask-binding says")
        for index, (name, bbox) in enumerate(zip(self.object_names, bboxes)):
            binding = (bbox or {}).get("model_binding") if isinstance(bbox, dict) else None
            ambiguous = isinstance(binding, dict) and binding.get("state") == "ambiguous"
            fit = (self.model_offsets.get(name) or (None, None, None, {}))[3]
            grasps = (getattr(self.args, "grasps", None) or {}).get(name)
            matters = (self.args.ask_binding == "always" or agent_asks
                       or (self.args.ask_binding == "needed" and not (grasps and fit.get("grasps_safe"))))
            if not ambiguous or not matters or index in self.binding_answers:
                if name in self.binding_asked and index not in self.binding_answers:
                    self.withdraw_binding(name, "settled" if not ambiguous else "a grasp is safe either way")
                continue
            hypotheses = [h for h in binding.get("hypotheses") or [] if isinstance(h, dict) and h.get("id")]
            ids = [h["id"] for h in hypotheses]
            asked = self.binding_asked.get(name)
            if asked is not None and asked["ids"] == ids:
                continue
            self.binding_asked[name] = {"fit_id": bbox.get("model_fit_id"), "ids": ids,
                                        "features": {h["id"]: h.get("features_px") or [] for h in hypotheses},
                                        "shown": not agent_asks}
            if agent_asks:
                continue
            self.binding_request_pub.publish(String(data=json.dumps({
                "object": name, "fit_id": bbox.get("model_fit_id"),
                "hypotheses": [{"id": h["id"], "number": k, "p": h.get("p")} for k, h in enumerate(hypotheses)],
                "frame_id": header.frame_id, "stamp": header.stamp.sec + header.stamp.nanosec * 1e-9})))
            shown = rgb.copy()
            for k, h in enumerate(hypotheses):
                colour = BINDING_COLOURS[k % len(BINDING_COLOURS)]
                px = np.asarray(h.get("features_px") or [], dtype=float).reshape(-1, 2)
                for u, v in px:
                    cv2.circle(shown, (int(u), int(v)), 3, colour, -1, cv2.LINE_AA)
                if len(px):
                    u, v = px.mean(0)
                    label = f"{name} #{k} p={float(h.get('p') or 0):.2f}"
                    cv2.putText(shown, label, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(shown, label, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                colour, 1, cv2.LINE_AA)
            image = self.bridge.cv2_to_imgmsg(shown, "bgr8")
            image.header = header
            self.binding_image_pub.publish(image)
            self.get_logger().warn(
                f"{name}: which way round it lies is in doubt ({len(ids)} ways, "
                f"{BINDING_REQUEST_IMAGE_TOPIC}); say which: {{\"object\": \"{name}\", \"hypothesis\": k}}, "
                f"or click its telling feature: {{\"object\": \"{name}\", \"points\": [[u, v]]}} on {BINDING_TOPIC}")

    def agent_asking(self) -> bool:
        """Whether a node says on BINDING_ASKER_TOPIC that it asks the person
        which way round; not where this client is no node to ask by."""
        try:
            return self.count_publishers(BINDING_ASKER_TOPIC) > 0
        except AttributeError:                                  # not a node (stood up without rclpy)
            return False

    def withdraw_binding(self, name, why="re-initialized"):
        asked = self.binding_asked.pop(name, None)
        if asked is not None and not asked.get("shown", True):
            return                                   # kept for the Agent's answers, never shown
        self.binding_request_pub.publish(String(data=json.dumps({"object": name, "withdrawn": True, "why": why})))
        self.get_logger().info(f"{name}: way round no longer asked ({why})")

    def on_binding(self, msg):
        """The person's way round (BINDING_TOPIC), to go with the next frame."""
        try:
            answer = json.loads(msg.data)
            name = answer["object"]
            asked = self.binding_asked.get(name)
            if asked is None:
                self.get_logger().warn(f"way round of {name}: none asked for")
                return
            if "hypothesis" in answer:
                ident = answer["hypothesis"]
                if isinstance(ident, int) and not isinstance(ident, bool):
                    ident = asked["ids"][ident]
                if ident not in asked["ids"]:
                    raise ValueError(f"{ident!r} is none of {asked['ids']}")
            else:
                points = np.asarray([[float(u), float(v)] for u, v in answer["points"]], dtype=float)
                if not len(points):
                    raise ValueError("no points")

                def apart(ident):
                    px = np.asarray(asked["features"].get(ident) or [], dtype=float).reshape(-1, 2)
                    return np.inf if not len(px) else float(np.mean(cKDTree(px).query(points)[0]))

                ident = min(asked["ids"], key=apart)
                if not np.isfinite(apart(ident)):
                    raise ValueError("no way round shows features to click")
            by = answer.get("by") or "person"
            if not isinstance(by, str) or not re.fullmatch(r"[a-z+ ]{1,32}", by):
                raise ValueError(f"'by' {by!r} is not a word such as person or camera")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self.get_logger().error(f"unusable answer on {BINDING_TOPIC}: {exc}")
            return
        self.binding_answers[self.object_names.index(name)] = (ident, by)
        self.binding_asked.pop(name, None)
        self.get_logger().info(f"{name}: way round {ident}, by the {by}, sent with the next frame")

    def relocate_seen(self, lost, bboxes, hand=None):
        """Count each object's replies lost in a row, and keep the pixels and
        the box of where each tracked one is seen, and how near the hand --
        `hand`, HAND_LINK's pose -- is to that box, or for one lost has come
        to where it was last seen since (relocate())."""
        if not isinstance(bboxes, list) or len(bboxes) != len(self.object_names):
            bboxes = [None] * len(self.object_names)
        for name, is_lost, bbox in zip(self.object_names, lost, bboxes):
            self.lost_replies[name] = self.lost_replies.get(name, 0) + 1 if is_lost else 0
            pixels = None if is_lost else self.box_pixels(bbox)
            if pixels is not None:
                self.last_seen_px[name] = pixels
                self.last_seen_box[name] = reply_box(bbox)
                self.hand_near.pop(name, None)
                model = rigid(bbox.get("model_pose")) if isinstance(bbox, dict) else None
                if model is not None:
                    self.last_seen_model[name] = np.asarray(model, dtype=np.float64)
                else:
                    self.last_seen_model.pop(name, None)
            box = self.last_seen_box.get(name)
            if hand is not None and box is not None:
                self.hand_near[name] = min(self.hand_near.get(name, np.inf), box_distance(hand[:3, 3], *box))

    def near_hand(self, name) -> bool:
        """Whether the hand is or has come within RELOCATE_HAND_M of where
        `name` was last seen (relocate_seen()): it may hold it."""
        return self.hand_near.get(name, np.inf) <= RELOCATE_HAND_M

    def box_pixels(self, bbox):
        """(u0, v0, u1, v1) of a reply's box in the image, or None."""
        box = reply_box(bbox)
        if box is None or self.K is None:
            return None
        corners = box_corners(*box)
        if np.any(corners[:, 2] <= 0):
            return None
        uv = (corners @ self.K.T)[:, :2] / corners[:, 2:3]
        return (*uv.min(0), *uv.max(0))

    def mesh_clicks(self, name, bbox, metres, robot):
        """Clicks on `name` where its fitted mesh -- the reply's model_pose
        and model_linear -- is seen in `metres` (RELOCATE_DEPTH_M) off
        `robot`: one in each of RELOCATE_CLICKS stretches of what is seen
        along its longest extent in the image, the sample nearest the
        stretch's middle; with the camera points they are on. None without a
        mesh pose, or with under RELOCATE_MIN_SEEN samples seen."""
        try:
            model = np.asarray(bbox["model_pose"], dtype=np.float64)
            linear = np.asarray(bbox.get("model_linear", float(bbox["model_scale"]) * np.eye(3)),
                                dtype=np.float64)
            vertices, faces = self.models[name]
        except (KeyError, TypeError, ValueError):
            return None
        if (model.shape != (4, 4) or linear.shape != (3, 3) or metres is None or self.K is None
                or not (np.isfinite(model).all() and np.isfinite(linear).all())):
            return None
        P = surface_samples(np.asarray(vertices, np.float64) @ linear.T, faces, RELOCATE_SPACING_M,
                            np.random.default_rng(0))
        P = P @ model[:3, :3].T + model[:3, 3]
        P = P[P[:, 2] > 0]
        uv = np.rint((P @ self.K.T)[:, :2] / P[:, 2:3]).astype(int)
        h, w = metres.shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        P, uv = P[inside], uv[inside]
        seen = np.abs(metres[uv[:, 1], uv[:, 0]] - P[:, 2]) <= RELOCATE_DEPTH_M
        if robot is not None and robot.shape == metres.shape:
            seen &= ~robot[uv[:, 1], uv[:, 0]]
        P, uv = P[seen], uv[seen]
        if len(P) < RELOCATE_MIN_SEEN:
            return None
        centred = uv - uv.mean(0)
        order = np.argsort(centred @ np.linalg.svd(centred, full_matrices=False)[2][0])
        picks = [chunk[np.argmin(np.linalg.norm(uv[chunk] - uv[chunk].mean(0), axis=1))]
                 for chunk in np.array_split(order, RELOCATE_CLICKS)]
        return uv[picks].tolist(), P[picks]

    def carried_clicks(self, points, frame, shape):
        """`points` with the clicks of each part clicked in the hand
        (relocate()) where the hand has carried it since, as it is in
        `frame`; one carried out of the image keeps those it had."""
        if not self.relocate_carried:
            return points
        hand = self.link_in(frame, HAND_LINK)
        if hand is None:
            return points
        points = list(points)
        h, w = shape
        for index, (where, then) in self.relocate_carried.items():
            moved = hand @ np.linalg.inv(then)
            P = where @ moved[:3, :3].T + moved[:3, 3]
            if np.any(P[:, 2] <= 0):
                continue
            uv = np.rint((P @ self.K.T)[:, :2] / P[:, 2:3]).astype(int)
            if np.all((uv >= 0) & (uv < [w, h])):
                points[index] = uv.tolist()
        return points

    def relocate(self, arm, bboxes, metres=None, frame=None):
        """Send the init again (RELOCATE_LOST_FRAMES) when a part with a mesh
        has been lost that long and the arm -- `arm()`, the moving links'
        pixels -- is clear of where it was. The cart is not counted: a part
        taken from in front of it leaves the cart in view there, the robot's
        own pixels where the part was (the ring, 2026-10-29). A part with a
        mesh still tracked is clicked where its mesh is seen in `metres`
        (mesh_clicks()), the others located by their meshes; not while both
        fingertips touch something if the hand may hold one of those
        (near_hand())."""
        if time.time() - self.last_relocate < RELOCATE_EVERY_S:
            return
        due = [name for name in self.object_names
               if name in self.models and self.lost_replies.get(name, 0) >= RELOCATE_LOST_FRAMES]
        if self.held_back(due, [name for name in due if self.near_hand(name)]):
            return
        robot = arm() if due else None
        if robot is None:
            return
        for name in due:
            pixels = self.last_seen_px.get(name)
            if pixels is None:
                continue
            h, w = robot.shape
            u0, v0, u1, v1 = (int(np.clip(round(x), 0, limit)) for x, limit in zip(pixels, (w, h, w, h)))
            if u1 > u0 and v1 > v0 and robot[v0:v1, u0:u1].mean() > RELOCATE_CLEAR_SHARE:
                self.get_logger().info(f"{name}: lost for {self.lost_replies[name]} replies; to be located "
                                       "again once the robot is clear of where it was", throttle_duration_sec=30.0)
                return
        points, labels = json.loads(self.args.points), json.loads(self.args.labels)
        if not isinstance(bboxes, list) or len(bboxes) != len(self.object_names):
            bboxes = [None] * len(self.object_names)
        hand = self.link_in(frame, HAND_LINK) if frame else None
        by_mesh, carried = [], {}
        for index, (name, bbox) in enumerate(zip(self.object_names, bboxes)):
            if name in self.models:
                clicks = None if self.lost_replies.get(name, 0) else self.mesh_clicks(name, bbox, metres, robot)
                if clicks is None:
                    points[index], labels[index] = [], []       # by its mesh, wherever it is now
                    by_mesh.append(name)
                    continue
                points[index], labels[index] = clicks[0], [1] * len(clicks[0])
                if hand is not None and self.touching() and self.near_hand(name):
                    carried[index] = (clicks[1], hand)
                continue
            pixels = None if self.lost_replies.get(name, 0) else self.box_pixels(bbox)
            if pixels is not None:
                points[index] = [[round((pixels[0] + pixels[2]) / 2), round((pixels[1] + pixels[3]) / 2)]]
                labels[index] = [1]
        if self.held_back(due, [name for name in by_mesh if self.near_hand(name)]):
            return
        clicked = [name for name in self.models if name not in by_mesh]
        # The others keep their fits as last seen, where the view does not
        # say otherwise (pose_sever's carry_kept_fits()): fitted afresh, a
        # part tracked all along lost its measured size and ways round.
        keep = [name for name in self.models if name not in due]
        self.get_logger().warn(f"{', '.join(due)}: lost for {max(self.lost_replies[n] for n in due)} replies; "
                               "locating every object again"
                               + (f", {', '.join(clicked)} clicked where seen" if clicked else "")
                               + (f", {', '.join(keep)} to keep their fits" if keep else ""))
        self.relocate_points = (points, labels)
        self.relocate_by_mesh, self.relocate_carried, self.relocate_keep = by_mesh, carried, keep
        self.last_relocate = time.time()
        self.initialized = False
        self._invalidate_all()

    def held_back(self, due, held) -> bool:
        """Whether the init is not sent again for `due` now: both fingertips
        touch something and the hand may hold `held`, to be located by their
        meshes."""
        if not (held and self.touching()):
            return False
        self.get_logger().info(f"{', '.join(due)}: lost for {max(self.lost_replies[n] for n in due)} replies; "
                               f"to be located again once the fingertips touch nothing ({', '.join(held)} "
                               "may be in the hand)", throttle_duration_sec=30.0)
        return True

    def relocation_waits(self, now) -> bool:
        """Whether an init sent again (relocate()) is not to go on this frame:
        the last one failed too recently (relocation_failed()), or the
        fingertips touch something and a part it locates by its mesh may be
        in the hand (near_hand())."""
        if self.initialized or self.relocate_points is None:
            return False
        held = [name for name in self.relocate_by_mesh if self.near_hand(name)]
        if held and self.touching():
            self.get_logger().info(f"locating the objects again waits: the fingertips touch something, "
                                   f"{', '.join(held)} may be in the hand", throttle_duration_sec=30.0)
            return True
        return now < self.relocate_retry_at

    def relocation_failed(self):
        """After an init sent again failed: the next waits RELOCATE_RETRY_S,
        doubled for each failure in a row, up to RELOCATE_RETRY_MAX_S."""
        if self.initialized or self.relocate_points is None:
            return
        self.relocate_failures += 1
        wait = min(RELOCATE_RETRY_S * 2 ** (self.relocate_failures - 1), RELOCATE_RETRY_MAX_S)
        self.relocate_retry_at = time.time() + wait
        self.get_logger().warn(f"locating the objects again failed ({self.relocate_failures} in a row); "
                               f"again in {wait:.0f} s")

    def say_mesh_changes(self, errors):
        """Log when a part's mesh pose leaves the replies, comes back, or is
        another fit. The server withholds a fit off the part from that frame,
        and drops and fits again one off it frames running (pose_sever's
        check_carried_meshes()), saying so in its own console: here are when,
        and its reason, model_errors, where it gives one."""
        errors = errors if isinstance(errors, list) else []
        said = getattr(self, "mesh_said", None)
        if said is None:
            said = self.mesh_said = {}
        for index, name in enumerate(self.object_names):
            if name not in getattr(self, "models", {}):
                continue
            native = self.model_offsets.get(name)
            reason = errors[index] if index < len(errors) and errors[index] else None
            now = ("fit", native[3].get("fit_id")) if native else ("none", reason)
            before = said.get(name)
            said[name] = now
            if before is None or now == before:
                continue
            if native is None:
                self.get_logger().warn(
                    f"{name}: no mesh pose in this reply ({reason or 'withheld on this frame'})")
                continue
            fit = native[3]
            what = (f"fit {(fit.get('fit_id') or '?')[:8]} ({fit['quality']}"
                    f"{'' if fit['measured'] else ', size unmeasured'})")
            if before[0] == "none":
                self.get_logger().info(f"{name}: mesh pose back, {what}")
            else:
                self.get_logger().info(f"{name}: mesh pose refitted: {what}, was {(before[1] or '?')[:8]}")

    def update_boxes(self, poses, lost, bboxes):
        """Keep each object's box as an offset from its tracked anchor.

        The server moves the anchor and the box by the same motion, so the
        offset holds from frame to frame. An object whose supplied mesh has not
        fitted is still tracked, by the box of its points: its native-mesh pose
        is published invalid, so the Agent knows it has the box alone. A reply
        without a box leaves none cached for a mesh object.
        """
        if not isinstance(bboxes, list) or len(bboxes) != len(self.object_names):
            bboxes = [None] * len(self.object_names)
        required_models = getattr(self, "models", {})
        for index, (name, T_anchor, bbox, is_lost) in enumerate(zip(self.object_names, poses, bboxes, lost)):
            required = name in required_models
            self.model_offsets.pop(name, None)
            if required or is_lost:
                self.box_offsets.pop(name, None)
            try:
                if is_lost:
                    continue
                T_box = np.asarray(bbox["pose"], dtype=np.float64)
                extent = np.asarray(bbox["extent"], dtype=np.float64)
                if (T_box.shape != (4, 4) or extent.shape != (3,)
                        or not np.isfinite(T_box).all() or not np.isfinite(extent).all()
                        or not np.all(extent > 0)):
                    continue
                if np.linalg.det(T_box[:3, :3]) < 0:
                    T_box[:3, 2] *= -1  # box axes may be reflected
                self.box_offsets[name] = (np.linalg.inv(T_anchor) @ T_box, extent)
                model = np.asarray(bbox["model_pose"], dtype=float)
                scale = float(bbox["model_scale"])
                model_id = bbox["model_id"]
                if (model.shape == (4, 4) and np.isfinite(model).all()
                        and np.allclose(model[3], [0, 0, 0, 1], atol=1e-6)
                        and np.allclose(model[:3, :3].T @ model[:3, :3], np.eye(3), atol=1e-5)
                        and np.isclose(np.linalg.det(model[:3, :3]), 1, atol=1e-5)
                        and np.isfinite(scale) and scale > 0
                        and isinstance(model_id, str) and re.fullmatch(r"[0-9a-f]{64}", model_id)):
                    # Older servers send no fit quality: their fits were fine.
                    linear = np.asarray(bbox.get("model_linear", scale * np.eye(3)), dtype=float)
                    if linear.shape != (3, 3) or not np.isfinite(linear).all():
                        linear = scale * np.eye(3)
                    fit = {"quality": bbox.get("model_quality", "fit"),
                           "measured": bool(bbox.get("model_measured", True)),
                           "linear": linear.tolist()}
                    # Which fit this is, and the fit turned half round about
                    # its length, or end for end, where the view cannot tell
                    # that from it: relative to the mesh pose, so it needs no
                    # carrying to the world.
                    if isinstance(bbox.get("model_fit_id"), str):
                        fit["fit_id"] = bbox["model_fit_id"]
                    for key, field in (("spin_doubt", "model_spin_doubt"), ("ends_doubt", "model_ends_doubt")):
                        doubt = rigid(bbox.get(field))
                        if doubt is not None:
                            fit[key] = doubt
                    # Every way round in doubt: those two, and any other the
                    # view cannot tell from the pose (a quarter turn, say).
                    doubts = [rigid(d) for d in (bbox.get("model_doubts") or [])]
                    if any(d is not None for d in doubts):
                        fit["doubts"] = [d for d in doubts if d is not None]
                    # How far the binding is (BINDING_REQUEST_TOPIC), and the
                    # grasps no way round living changes.
                    if isinstance(bbox.get("model_prior"), dict):
                        fit["prior"] = bbox["model_prior"]
                    binding = bbox.get("model_binding")
                    if isinstance(binding, dict):
                        # All of it: the evidence too -- each way round's
                        # probability, whether it is in doubt, where its
                        # telling features are in the image (this frame's,
                        # the camera's pixels) -- and how much of it there
                        # is, for the Agent to weigh beside its own look.
                        fit["binding"] = {
                            "state": binding.get("state"), "by": binding.get("by"),
                            "confidence": binding.get("confidence"), "frames": binding.get("frames"),
                            "error_rate": binding.get("error_rate"),
                            "hypotheses": [{"id": h.get("id"), "turn": h.get("turn"), "p": h.get("p"),
                                            "in_doubt": h.get("in_doubt"), "features_px": h.get("features_px") or []}
                                           for h in binding.get("hypotheses") or [] if isinstance(h, dict)],
                            "ruled_out": binding.get("ruled_out") or []}
                    grasps = (getattr(self.args, "grasps", None) or {}).get(name)
                    if grasps:
                        turns = [d for d in (rigid(h.get("turn")) for h in (fit.get("binding") or {}).get(
                            "hypotheses", [])[1:]) if d is not None] or fit.get("doubts", [])
                        fit["grasps_safe"] = safe_grasps(grasps, linear, turns,
                                                         self.args.grasp_tolerance_mm / 1000.0,
                                                         self.args.grasp_tolerance_deg, self.args.grasp_flip_axis)
                    # What the mesh itself leaves open, whatever the view: the
                    # axis (in its frame) it is round about, the turn about
                    # which the server holds and nothing tells; and whether its
                    # ends are alike, ends_doubt then always sent.
                    symmetry = bbox.get("model_symmetry")
                    if isinstance(symmetry, dict):
                        axis = np.asarray(symmetry.get("axis") if symmetry.get("axis") is not None else np.nan,
                                          dtype=float)
                        fit["symmetry"] = {
                            "axis": (axis / np.linalg.norm(axis)).tolist()
                            if axis.shape == (3,) and np.isfinite(axis).all() and np.linalg.norm(axis) > 0 else None,
                            "ends": bool(symmetry.get("ends"))}
                    self.model_offsets[name] = (np.linalg.inv(T_anchor) @ model, scale, model_id, fit)
                    if name in required_models:
                        # The box in the mesh's own axes, not the server's; of
                        # the part, where the server left loose fragments out.
                        centre, extent = mesh_box(required_models[name][0], linear)
                        bounds = np.asarray(bbox.get("model_bounds", np.nan), dtype=float)
                        if bounds.shape == (2, 3) and np.isfinite(bounds).all() and np.all(bounds[1] > bounds[0]):
                            centre, extent = bounds.mean(0), bounds[1] - bounds[0]
                        T_box = model.copy()
                        T_box[:3, 3] = model[:3, :3] @ centre + model[:3, 3]
                        self.box_offsets[name] = (np.linalg.inv(T_anchor) @ T_box, extent)
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                pass
            if required and name not in self.box_offsets:
                # Its clicked anchor alone is no pose of the part.
                lost[index] = True
                continue
            self.keep_surface(name, bbox)

    def keep_surface(self, name, bbox):
        """Cache a reply's surface for `name`, to publish with its world pose:
        points (N, 3) relative to the server's model or box frame."""
        try:
            raw = base64.b64decode(bbox["surface"])
            points = np.frombuffer(raw, "<f2").astype(np.float64).reshape(-1, 3)
            views = None
            if bbox.get("surface_views"):
                views = np.frombuffer(base64.b64decode(bbox["surface_views"]), "<f2")
                views = views.astype(np.float64).reshape(-1, 3)
                if views.shape != points.shape or not np.isfinite(views).all():
                    views = None
            frame = bbox["surface_frame"]
            if frame in ("model", "box") and len(points) and np.isfinite(points).all():
                self.surfaces[name] = (frame, points, views)
        except (KeyError, TypeError, ValueError):
            pass

    def _invalidate_object(self, name):
        self.box_offsets.pop(name, None)
        self.model_offsets.pop(name, None)
        self.lost_pubs[name].publish(Bool(data=True))
        self.model_pubs[name].publish(String(data=json.dumps({"valid": False})))

    def _invalidate_all(self):
        markers = MarkerArray()
        for index, name in enumerate(self.object_names):
            self._invalidate_object(name)
            marker = Marker()
            marker.header.frame_id = "world"
            marker.ns, marker.id, marker.action = "point2pose_bbox", index, Marker.DELETE
            markers.markers.append(marker)
        for name in getattr(self, "obstacle_names", []):
            self.obstacle_lost(name, markers)
        self.bbox_pub.publish(markers)

    def publish_world_poses(self, poses, lost):
        """Publish named, valid poses for RViz and Agent in the world frame."""
        if len(poses.poses) != len(self.object_names) or len(lost) != len(self.object_names):
            self._invalidate_all()
            self.get_logger().error(
                "Point2Pose result count does not match --objects; skipping named poses",
                throttle_duration_sec=2.0,
            )
            return
        frame = poses.header.frame_id
        if not frame:
            self._invalidate_all()
            self.get_logger().warn("camera image has no frame_id; skipping named poses",
                                   throttle_duration_sec=5.0)
            return
        if frame == "world":
            translation = np.zeros(3)
            rotation = Rotation.identity()
        else:
            try:
                tf = self.tf_buffer.lookup_transform("world", frame, Time())
            except Exception as exc:
                self._invalidate_all()
                self.get_logger().warn(
                    f"no TF world <- {frame}: {exc}; skipping named poses",
                    throttle_duration_sec=5.0,
                )
                return
            t, q = tf.transform.translation, tf.transform.rotation
            translation = np.array([t.x, t.y, t.z])
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])

        for name, pose, is_lost in zip(self.object_names, poses.poses, lost):
            if is_lost:
                self._invalidate_object(name)
                continue
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            q = pose.orientation
            quat = np.array([q.x, q.y, q.z, q.w])
            if (not np.all(np.isfinite(position)) or not np.all(np.isfinite(quat))
                    or np.linalg.norm(quat) < 1e-6):
                self._invalidate_object(name)
                continue
            world_position = rotation.apply(position) + translation
            world_rotation = rotation * Rotation.from_quat(quat)
            native = self.model_offsets.get(name)
            if native is not None:
                offset, scale, model_id, fit = native
                model_pose = np.eye(4)
                model_pose[:3, 3] = world_rotation.apply(offset[:3, 3]) + world_position
                model_pose[:3, :3] = world_rotation.as_matrix() @ offset[:3, :3]
                self.model_pubs[name].publish(String(data=json.dumps({
                    "frame_id": "world", "pose": model_pose.tolist(),
                    "model_id": model_id, "scale": scale, **fit,
                    "stamp": poses.header.stamp.sec + poses.header.stamp.nanosec * 1e-9,
                    # Age since local image receipt; monotonic time also
                    # works when image stamps use the simulation clock.
                    "observation_age_s": self.observation_age()})))
            else:
                self.model_pubs[name].publish(String(data=json.dumps({"valid": False})))
            box = self.box_offsets.get(name)
            box_frame = None
            if box is not None:
                # The anchor is the clicked surface point with the camera's
                # axes at initialisation; the box is the object's centre and
                # its own axes, so publish that as the object's pose.
                offset, extent = box
                world_position = world_rotation.apply(offset[:3, 3]) + world_position
                box_frame = world_rotation * Rotation.from_matrix(offset[:3, :3])
                # A mesh's box keeps the mesh's axes (update_boxes()); a box
                # of points alone has no axes but its own, put upright.
                world_rotation, extent = ((box_frame, extent) if native is not None
                                          else upright_box(box_frame, extent))
            self.publish_surface(name, native is not None, box_frame, world_rotation)
            stamped = PoseStamped()
            stamped.header.stamp = poses.header.stamp
            stamped.header.frame_id = "world"
            stamped.pose.position.x, stamped.pose.position.y, stamped.pose.position.z = world_position
            (stamped.pose.orientation.x, stamped.pose.orientation.y,
             stamped.pose.orientation.z, stamped.pose.orientation.w) = world_rotation.as_quat()
            if box is not None:
                msg = BoundingBox3D()
                msg.frame_id = "world"
                msg.center = stamped.pose
                msg.size.x, msg.size.y, msg.size.z = extent
                self.box_pubs[name].publish(msg)
            self.named_pubs[name].publish(stamped)
            self.lost_pubs[name].publish(Bool(data=False))
            self.get_logger().info(
                f"world pose {name}: x={world_position[0]:.3f} "
                f"y={world_position[1]:.3f} z={world_position[2]:.3f}",
                throttle_duration_sec=2.0,
            )

    def publish_surface(self, name, has_model, box_frame, upright):
        """The cached surface of `name`, once, relative to the pose published
        for it: the native mesh pose if there is one, else the box, whose axes
        upright_box() reorders (`box_frame` before, `upright` after)."""
        cached = self.surfaces.pop(name, None)
        if cached is None:
            return
        frame, points, views = cached
        if frame == "box":
            if box_frame is None or has_model:
                return
            turn = upright.inv() * box_frame
            points = turn.apply(points)
            views = turn.apply(views) if views is not None else None
        elif not has_model:
            return
        message = {"frame": frame, "points": np.round(points, 4).tolist()}
        if views is not None:
            message["views"] = np.round(views, 3).tolist()
        self.surface_pubs[name].publish(String(data=json.dumps(message)))

    def publish_obstacles(self, obstacles, header, renamed=False):
        """Publish a reply's obstacles (OBSTACLE_PREFIX) and return their
        markers. On an init's reply (`renamed`) the names are those it found,
        announced with the objects; one found before and not now is lost."""
        markers = MarkerArray()
        found = obstacles_by_name(obstacles)
        if renamed:
            self.name_obstacles(obstacles, markers)
        world = np.eye(4)
        if header.frame_id != "world":
            try:
                tf = self.tf_buffer.lookup_transform("world", header.frame_id, Time())
            except Exception:
                world = None
            else:
                q, t = tf.transform.rotation, tf.transform.translation
                world[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                world[:3, 3] = [t.x, t.y, t.z]
        for name in self.obstacle_names:
            item = found.get(name)
            box = None if item is None or item.get("lost") else reply_box(item.get("bbox"))
            if box is None or world is None:
                self.obstacle_lost(name, markers)
                continue
            pose_pub, box_pub, lost_pub = self.obstacle_publishers(name)
            T, extent = box
            Tw = world @ T
            rotation, extent = self.steady_box(name, Rotation.from_matrix(Tw[:3, :3]), extent)
            stamped = PoseStamped()
            stamped.header.stamp, stamped.header.frame_id = header.stamp, "world"
            stamped.pose.position.x, stamped.pose.position.y, stamped.pose.position.z = Tw[:3, 3]
            (stamped.pose.orientation.x, stamped.pose.orientation.y,
             stamped.pose.orientation.z, stamped.pose.orientation.w) = rotation.as_quat()
            msg = BoundingBox3D()
            msg.frame_id = "world"
            msg.center = stamped.pose
            msg.size.x, msg.size.y, msg.size.z = extent
            pose_pub.publish(stamped)
            box_pub.publish(msg)
            lost_pub.publish(Bool(data=False))
            # In the camera's frame, as the parts' boxes are drawn.
            m = Marker()
            m.header, m.ns, m.id = header, "point2pose_obstacle", int(name[len(OBSTACLE_PREFIX):])
            corners = box_corners(T, box[1])
            m.type, m.action = Marker.LINE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.004
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.1, 1.0
            lifetime = max(BBOX_LIFETIME_SEC, BBOX_LIFETIME_UPDATES / self.args.rate)
            m.lifetime.sec, m.lifetime.nanosec = int(lifetime), int(lifetime % 1 * 1e9)
            for a, b in BOX_EDGES:
                m.points.append(Point(x=corners[a][0], y=corners[a][1], z=corners[a][2]))
                m.points.append(Point(x=corners[b][0], y=corners[b][1], z=corners[b][2]))
            markers.markers.append(m)
        return markers

    def steady_box(self, name, rotation, extent):
        """upright_box() of an obstacle's box, the way round nearest the one
        last published for it (OBSTACLE_SIDE_TIE): (rotation, extent)."""
        R, e = upright_box(rotation, extent)
        R = R.as_matrix()
        flip = np.diag([-1.0, -1.0, 1.0])                                          # half a turn about z
        quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # x to the old y
        ways = [(R, e), (R @ flip, e)]
        if e[1] >= (1 - OBSTACLE_SIDE_TIE) * e[0]:
            ways += [(R @ quarter, e[[1, 0, 2]]), (R @ quarter @ flip, e[[1, 0, 2]])]
        last = self.obstacle_last.get(name)
        if last is not None:
            R, e = max(ways, key=lambda way: np.trace(last.T @ way[0]))
        self.obstacle_last[name] = R
        return Rotation.from_matrix(R), e

    def obstacle_publishers(self, name):
        """(pose, box, lost) publishers of an obstacle, made on first use."""
        if name not in self.obstacle_pubs:
            self.obstacle_pubs[name] = (
                self.create_publisher(PoseStamped, f"/detected/{name}_pose", 10),
                self.create_publisher(BoundingBox3D, f"/detected/{name}_box", 10),
                self.create_publisher(Bool, f"/point2pose/{name}/lost", 10))
        return self.obstacle_pubs[name]

    def name_obstacles(self, obstacles, markers):
        """Take an init reply's obstacles as the ones followed: one followed
        before and not found now is lost (its marker's deletion into
        `markers`), and the names are announced with the objects."""
        found = obstacles_by_name(obstacles)
        for name in self.obstacle_names:
            if name not in found:
                self.obstacle_lost(name, markers)
        if list(found) != self.obstacle_names:
            self.obstacle_names = list(found)
            self.objects_pub.publish(String(data=",".join(
                [*self.object_names, *self.args.props, *self.obstacle_names])))
            self.get_logger().info(f"obstacles found: {', '.join(self.obstacle_names) or 'none'}")

    def obstacle_lost(self, name, markers):
        """Say an obstacle is lost, and add the deletion of its marker."""
        self.obstacle_publishers(name)[2].publish(Bool(data=True))
        m = Marker()
        m.header.frame_id = "world"
        m.ns, m.id, m.action = "point2pose_obstacle", int(name[len(OBSTACLE_PREFIX):]), Marker.DELETE
        markers.markers.append(m)

    def bbox_markers(self, bboxes, lost, header, poses=None):
        """One wireframe per object in the camera frame; red while the object is lost.

        A lost object has no marker. An object with a mesh pose is drawn by
        the box published for it, in the mesh's axes (update_boxes()), at its
        tracked anchor `poses`; the others by the server's corners.
        """
        markers = MarkerArray()
        if not isinstance(bboxes, list) or len(bboxes) != len(self.object_names):
            bboxes = [None] * len(self.object_names)
        for obj_id, (name, bbox, is_lost) in enumerate(zip(self.object_names, bboxes, lost)):
            m = Marker()
            m.header = header
            m.ns = "point2pose_bbox"
            m.id = obj_id
            if is_lost:
                m.action = Marker.DELETE
                markers.markers.append(m)
                continue
            try:
                if poses is not None and name in self.model_offsets and name in self.box_offsets:
                    offset, extent = self.box_offsets[name]
                    corners = box_corners(np.asarray(poses[obj_id], dtype=np.float64) @ offset, extent)
                else:
                    corners = np.asarray(bbox["corners"], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                corners = None
            if corners is None or corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
                m.action = Marker.DELETE
                markers.markers.append(m)
                continue
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.005
            m.color.r, m.color.g, m.color.b = (1.0, 0.2, 0.2) if is_lost else (0.2, 1.0, 0.2)
            m.color.a = 1.0
            lifetime = max(BBOX_LIFETIME_SEC, BBOX_LIFETIME_UPDATES / self.args.rate)
            m.lifetime.sec, m.lifetime.nanosec = int(lifetime), int(lifetime % 1 * 1e9)
            for a, b in BOX_EDGES:
                m.points.append(Point(x=corners[a][0], y=corners[a][1], z=corners[a][2]))
                m.points.append(Point(x=corners[b][0], y=corners[b][1], z=corners[b][2]))
            markers.markers.append(m)
        return markers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="tcp://172.26.177.251:5555")
    ap.add_argument("--rgb-topic", default="/camera/color/image_raw")
    ap.add_argument("--depth-topic", default="/camera/depth/image_rect_raw")
    ap.add_argument("--info-topic", default="/camera/color/camera_info")
    ap.add_argument("--points", default="[[[320,240]]]",
                    help="JSON: one [[x,y],...] list per object; [] for an object with a mesh "
                         "has the server locate it by the mesh")
    ap.add_argument("--labels", default="[[1]]", help="JSON: one [1/0,...] list per object")
    ap.add_argument("--objects", required=True,
                    help="comma-separated names, one per --points group (e.g. mug,lid)")
    ap.add_argument("--models", default="{}",
                    help="JSON: {name: path to an OBJ mesh of that object, in metres}. "
                         "An object with a mesh gets the mesh's box, fit on the init "
                         "frame, and with no --points is located by it. Failed pose fits remain "
                         "unrecognised while the server retries on fresh frames; "
                         "there is no point-cloud box fallback")
    ap.add_argument("--baseline", action="store_true",
                    help="the unmodified Point2Pose, for the A2 comparison: one uniform scale per mesh, no "
                         "per-frame depth correction, no ways round in doubt, no click requests, no pose priors")
    ap.add_argument("--same-scale-meshes", action="store_true",
                    help="the meshes of --models come from one reconstruction, so are off the parts' "
                         "true sizes by about one factor: a part's scale is then checked against the "
                         "others', and one too hidden to measure takes theirs. Leave off for meshes "
                         "from different sources")
    ap.add_argument("--true-size-meshes", action="store_true",
                    help="the meshes of --models are the parts' true sizes (CAD models, or "
                         "reconstructions scaled to measure): each is fitted at that size only, "
                         "which also tells it from parts of the same shape and another size")
    ap.add_argument("--mesh-error", type=float, default=None,
                    help="how far the meshes of --models are off their parts' surfaces: the 95th "
                         "percentile, as a share of a mesh's diagonal (0 for CAD models). Every fit's "
                         "tolerance allows for it; unset, the server takes a demonstration "
                         "reconstruction's (MESH_ERROR_P95, 2.25 %%)")
    ap.add_argument("--mesh-stretch-max", type=float, default=None,
                    help="how far the proportions of the meshes of --models may be off their parts', as a "
                         "factor along any box axis (1 for CAD models, whose proportions are exact). Unset, "
                         "the server allows a demonstration reconstruction's (MODEL_STRETCH_MAX, 1.4)")
    ap.add_argument("--mesh-unit", type=float, default=None,
                    help="metres per unit of the meshes of --models: 0.001 for meshes in millimetres. Unset, "
                         "metres. The poses and scales sent back are the meshes' as given, in their units")
    ap.add_argument("--mesh-colours", action="store_true",
                    help="send each mesh's vertex colours (an OBJ's \"v x y z r g b\" lines), which the "
                         "server then also locates its part by: of two parts alike in size and shape, "
                         "the one in the mesh's colours")
    ap.add_argument("--grasps", default=None,
                    help="JSON file: {name: [4x4, ...]}, each the gripper's pose in that object's mesh frame "
                         "and units; each reply then says which no way round in doubt changes "
                         "(BINDING_REQUEST_TOPIC)")
    ap.add_argument("--grasp-tolerance-mm", type=float, default=10.0,
                    help="a way round in doubt moving a grasp no more than this leaves it safe")
    ap.add_argument("--grasp-tolerance-deg", type=float, default=15.0,
                    help="nor turning it more than this")
    ap.add_argument("--grasp-flip-axis", default="none", choices=["x", "y", "z", "none"],
                    help="the gripper's axis it is the same turned half round about (a parallel "
                         "gripper's approach axis), in the grasp's frame")
    ap.add_argument("--ask-binding", default="needed", choices=["needed", "always", "never"],
                    help="when to ask the person which way round a part lies: where its doubt matters -- "
                         "no grasp safe, or none given -- always while in doubt, or never. While a node "
                         f"publishes on {BINDING_ASKER_TOPIC} (the Agent) it asks, and needed asks nothing")
    ap.add_argument("--props", default="",
                    help="comma-separated names announced beside --objects whose poses another "
                         "node publishes: truth_bridge.py --props ... --no-announce")
    ap.add_argument("--obstacles", type=int, default=0,
                    help="how many obstacles at most the server looks for at init, with no click "
                         f"and no mesh, published as {OBSTACLE_PREFIX}<n>; default 0, none: the Agent "
                         "keeps what is in the way from the depth itself (its unnamed voxels), which "
                         "follow a moved box where the server's following lost it and left it where it was "
                         "(PP-S3-2, RI-S3-2, 2026-11-13)")
    ap.add_argument("--not-robot", default="table",
                    help="comma-separated links of the robot description that are not the "
                         "robot, left out of the robot's pixels: the bench, which every part "
                         "on it lies within the margin of")
    ap.add_argument("--depth-factor", type=float, default=1000.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--max-pose-age", type=float, default=3.0,
                    help="discard poses this many seconds after local image receipt (default 3)")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--timeout-ms", type=int, default=15000, help="per tracking frame")
    ap.add_argument("--init-timeout-ms", type=int, default=60000,
                    help="for the first frame, which builds the pipeline and every "
                         "object's model; grows with the number of objects. Each object "
                         "sent without points, located by its mesh, adds "
                         f"{LOCATE_TIMEOUT_MS // 1000} s to it")
    args = ap.parse_args()

    # Each name becomes a topic (/detected/<name>_pose) the Agent subscribes to.
    args.objects = [name.strip() for name in args.objects.split(",")]
    if not np.isfinite(args.rate) or args.rate <= 0 or not np.isfinite(args.max_pose_age) or args.max_pose_age <= 0:
        ap.error("--rate and --max-pose-age must be finite and positive")
    if (len(set(args.objects)) != len(args.objects)
            or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in args.objects)):
        ap.error("--objects must be distinct names such as mug,lid")
    args.props = [name.strip() for name in args.props.split(",") if name.strip()]
    args.not_robot = {name.strip() for name in args.not_robot.split(",") if name.strip()}
    if (set(args.props) & set(args.objects)
            or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in args.props)):
        ap.error("--props must be names such as obstacle_1, none of them in --objects")
    if args.obstacles < 0:
        ap.error("--obstacles must be 0 or more")
    if args.obstacles and any(re.fullmatch(f"{OBSTACLE_PREFIX}[0-9]+", name) for name in [*args.objects, *args.props]):
        ap.error(f"with --obstacles, no --objects or --props may be named {OBSTACLE_PREFIX}<n>: "
                 "the obstacles found are; or --obstacles 0")
    try:
        groups, label_groups = json.loads(args.points), json.loads(args.labels)
    except json.JSONDecodeError as exc:
        ap.error(f"--points/--labels must be JSON: {exc}")
    if (not isinstance(groups, list) or not isinstance(label_groups, list)
            or not len(groups) == len(label_groups) == len(args.objects)):
        ap.error("--objects, --points and --labels need one entry per object")
    try:
        paths = json.loads(args.models)
    except json.JSONDecodeError as exc:
        ap.error(f"--models must be JSON: {exc}")
    if not isinstance(paths, dict) or not set(paths) <= set(args.objects):
        ap.error("--models must map names from --objects to mesh paths")
    # The server locates an object sent with no points by its mesh, and with
    # neither, or with only points off it, has nothing to start it from.
    for name, points, labels in zip(args.objects, groups, label_groups):
        if not isinstance(points, list) or not isinstance(labels, list) or len(points) != len(labels):
            ap.error(f"{name}: --points and --labels need a label for each point")
        if points and not any(labels):
            ap.error(f"{name}: needs a point on it (label 1), or no points and a mesh")
        if not points and name not in paths:
            ap.error(f"{name}: no points and no mesh in --models to locate it by")
    args.init_timeout_ms += LOCATE_TIMEOUT_MS * sum(1 for points in groups if not points)
    try:
        loaded = {name: read_obj(path, colours=True) for name, path in paths.items()}
    except (OSError, ValueError) as exc:
        ap.error(f"--models: {exc}")
    args.models = {name: (v, f) for name, (v, f, _) in loaded.items()}
    args.grasp_flip_axis = None if args.grasp_flip_axis == "none" else np.eye(3)["xyz".index(args.grasp_flip_axis)]
    if args.grasps:
        try:
            raw = json.load(open(args.grasps))
            args.grasps = {name: [np.asarray(G, dtype=float).reshape(4, 4) for G in grasps]
                           for name, grasps in raw.items()}
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            ap.error(f"--grasps: {exc}")
        if not set(args.grasps) <= set(paths):
            ap.error("--grasps: only objects with a mesh in --models have grasps in its frame")
    else:
        args.grasps = {}
    args.model_colours = ({name: c for name, (_, _, c) in loaded.items() if c is not None}
                          if args.mesh_colours else {})
    for name in sorted(set(paths) - set(args.model_colours) if args.mesh_colours else ()):
        print(f"--mesh-colours: {name}'s mesh has no colour on every vertex; sent without", file=sys.stderr)

    rclpy.init()
    node = PoseClient(args)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
