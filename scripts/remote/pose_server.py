"""ZeroMQ inference server exposing ModularPipeline to a remote ROS2 client.

Each object's bounding box is fit here and then carried by the tracked
motion. Without a supplied mesh, the fit uses no known bench
height, no gravity, no object shape -- only the mask and the depth:

  1. The surface the object sits on is fit (RANSAC) to a ring of pixels just
     outside the mask. Mask pixels on that surface are the background the mask
     spilled onto at the object's edge, and are dropped.
  2. What is left gets a minimal-volume box. PCA, the usual shortcut, follows
     how the points are spread; one view sees an L-shaped cloud (a top and a
     side), and PCA tilts the box across it, making it too wide and askew.

A box fit to what one view sees is only as good as the view: a part half
hidden behind a hand gets a box too short, or one tilted across the hidden
side and too large. When the client sends a mesh of the object (one estimated
from a demonstration, say), the box is the mesh's own instead, placed by
fitting its pose to the observations. The mesh supplies the part's shape;
its size is fitted too, since a similar mesh need not be the part's size nor
quite its proportions: one scale, then -- where the view shows the part's
length along them -- a scale along each of the mesh's box axes, fitted with
the pose (MODEL_STRETCH_*). A mesh that already is the part's size keeps
exactly scale 1. Carried by the tracked motion, a fitted mesh's pose is then
corrected on each frame's depth (MODEL_CORRECT_*): a small part with little
texture gives the tracker few points to hold it by.
A failed pose fit stays invalid while subsequent frames are retried in the
background. Point-cloud boxes are used only when no mesh was supplied.

An init may also ask for the obstacles in view (meta["discover"]; see
OBSTACLE_MIN_PX): found with no click and no mesh, followed on every frame by
the depth, and replied apart from the parts ("obstacles").

Which way round a fitted mesh lies, where the init view leaves it in doubt,
is settled while the part is tracked: every way round in doubt rides on the
part's one tracked motion, and each frame's view is evidence for and against
each, until one is left or the person names one (BINDING_CONFIDENCE). Each
reply says how far that is ("model_binding").
"""

import argparse
import base64
import collections
import copy
import gc
import hashlib
import itertools
import json
import os
import sys
import threading
import time
import zlib
import traceback
import types
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import cv2
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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


def decode_robot_mask(meta, shape):
    """The robot's pixels the client sent with an init, or None: meta
    ["robot_mask"] holds the mask's shape and base64 of zlib of its
    np.packbits(). One that cannot be read or is not the depth's size is
    left out, and the proposals on the robot are fitted like any other."""
    spec = meta.get("robot_mask")
    if not spec:
        return None
    try:
        h, w = (int(n) for n in spec["shape"])
        if (h, w) != tuple(shape):
            raise ValueError(f"{h}x{w} against a {shape[0]}x{shape[1]} depth")
        bits = np.frombuffer(zlib.decompress(base64.b64decode(spec["bits"])), np.uint8)
        return np.unpackbits(bits, count=h * w).reshape(h, w).astype(bool)
    except Exception as exc:
        print(f"[server] unusable robot mask ({exc}); proposals on the robot are fitted too",
              flush=True)
        return None


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
# A part rests on its support, so at least this share of its mask's points lie
# on the plane or in front of it, within MODEL_TOL_M (_ModelFit asks the same).
# A plane short of that is an occluder the ring caught: RI-S4-2's ring lies on
# the table behind the box, the box's top filled the ring round its mask, and
# the plane fitted there had the ring 140 mm behind it -- no point in front of
# it, so the ring was taken to lie flat on it, its raw mask fitted (the box's
# edge spilled into it, the table seen through its hole; 35 mm off at the 95th
# percentile), and the box's latch was located as the ring instead. So the
# ring's points on such a plane or in front of it go, and the plane is fitted
# again on what is behind, up to PLANE_OCCLUDER_ROUNDS times.
PLANE_SUPPORT_SHARE = 0.9
PLANE_OCCLUDER_ROUNDS = 2


def backproject(metres, K):
    """Camera-frame point for every pixel, and which pixels have depth."""
    h, w = metres.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = np.isfinite(metres) & (metres > 0)
    z = np.where(valid, metres, 0.0)
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1), valid


def support_plane(mask, cam, valid, rng):
    """(normal, point) of the surface around the mask, normal facing away from
    the camera; None when the ring holds no clear plane. A plane the mask lies
    behind is an occluder, not the support (PLANE_SUPPORT_SHARE)."""
    def grown(radius):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    inner, outer = PLANE_RING_PX
    pts = cam[grown(outer) & ~grown(inner) & valid]
    own = cam[mask & valid]
    for _ in range(PLANE_OCCLUDER_ROUNDS + 1):
        plane = _ring_plane(pts, rng)
        if plane is None:
            return None
        n, p = plane
        if not len(own) or np.mean((own - p) @ n <= MODEL_TOL_M) >= PLANE_SUPPORT_SHARE:
            return plane
        pts = pts[(pts - p) @ n > PLANE_TOL_M]
    return None


def depth_noise(mask, cam, plane):
    """The depth's noise about the part (metres), as the support around it
    shows: the robust spread (scaled median absolute deviation) about
    `plane` of support_plane()'s ring, of its points within PLANE_TOL_M of
    the plane -- so a cluttered ring does not count -- or None with too few."""
    def grown(radius):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    inner, outer = PLANE_RING_PX
    pts = cam[grown(outer) & ~grown(inner) & (cam[..., 2] > 0)]
    n, p = plane
    r = (pts - p) @ n
    r = r[np.abs(r) < PLANE_TOL_M]
    if len(r) < MIN_BOX_POINTS:
        return None
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _ring_plane(pts, rng):
    """support_plane()'s RANSAC on the ring's points: (normal, point), the
    normal facing away from the camera, or None."""
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


def observe(mask, metres, K, rng=None):
    """What one frame shows of one object, or None if too little.

    Returns (cam, mask, keep, plane, note): every pixel's camera-frame point,
    the mask cut to pixels with depth, the mask pixels that are the object
    rather than its support, the support plane (normal, point) or None, and
    how keep was chosen.
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
    return cam, mask, keep, plane, note


def fit_box(mask, metres, K, rng=None, view=None):
    """Box of one object on one frame: (T_cam_box, extent, note), or None.

    T_cam_box's rotation is the box axes and its translation the box centre.
    """
    view = view or observe(mask, metres, K, rng)
    if view is None:
        return None
    cam, mask, keep, _, note = view
    R, centre, extent = min_obb(cam[keep])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, centre
    return T, extent, f"{int(keep.sum())} of {int(mask.sum())} mask px, {note}"


# Model fitting. A mesh point is judged by what the camera sees along its line
# of sight, so an occluded part costs nothing while a wrong pose still does:
#
#   * hidden: the camera sees something nearer (a hand, the gripper, another
#     part). Nothing was seen there, so nothing speaks against it.
#   * open: the camera sees that far or further. It must see the object, in
#     the mask and at the mesh point's depth. Seeing past it -- the bench
#     behind where the mesh says the part is -- counts against the pose.
#     Outside the mask, a surface at the mesh point's depth and off the
#     support does not: it is the part where the mask stops short of an
#     occluder touching it (a gripper finger pressing on it), or that
#     occluder. It counts as hidden. The bare support there still counts
#     against the pose: it is how a thin part's too-big mesh shows.
#   * below the support surface counts against it too. A box tilted over the
#     part's hidden side shows itself that way, or as open points in the air.
#
# The observed points, for their part, must lie on the mesh's visible surface.
MODEL_SAMPLES = 3000
# Mesh points and observed points per candidate pose in the coarse search, and
# its ICP rounds; the best MODEL_KEEP are then refined with every point.
MODEL_COARSE = (1000, 600, 8)
MODEL_FINE_POINTS = 3000
MODEL_FINE_ROUNDS = 30
MODEL_KEEP = 5
# Candidate headings about the support normal, one per this many degrees.
MODEL_YAW_STEP_DEG = 30
# A residual this large is an outlier, and every cost saturates here.
MODEL_TOL_M = 0.006
# Observed depth this far in front of a mesh point hides it.
MODEL_HIDDEN_M = 0.008
# Those millimetres -- MODEL_TOL_M, MODEL_HIDDEN_M, MODEL_COVER_NOISE_M and the
# 95th-percentile distances and floors -- hold for depth noise up to this: the
# acceptance was set on rendered scenes with 2 mm of it, which depth_noise()
# measures at up to 2.3 mm (pick_placement's; ring_insertion's, at 0). Each
# view measures its own on the support around the part, and where the depth is
# noisier they grow with it, so no camera needs calibrating; where it is
# quieter they stay, for they also cover the mesh's own error and the spacing
# of its samples. On BOP's test images (Kinect, Primesense) the noise came out
# 1.3-2.5 mm (median), up to 3.9.
MODEL_NOISE_REF_M = 0.0023
# The first ICP round pairs points this far apart; later rounds narrow it.
MODEL_SEARCH_M = 0.03
# The mask is grown this much before a mesh point counts as outside it.
MODEL_MASK_GROW_PX = 4
# Mask spill can also include a hand in front of the part. Only select a
# depth-connected component when it clearly dominates the foreground.
MODEL_DEPTH_GAP_M = 0.03
MODEL_COMPONENT_MIN_SHARE = 0.7
# A piece of the mask not touching the main one is kept where what lies
# between them in the image is nearer than both (MODEL_BEHIND_SHARE of it, by
# MODEL_HIDDEN_M) -- one part seen round an occluder -- and their facing edges
# could be one part's: apart in depth by no more than MODEL_BEHIND_DEPTH_RATIO
# times their gap across the view, plus MODEL_DEPTH_GAP_M. A bar seen 36 deg
# off end-on with a ring on it showed its far end as a separate 110 px piece,
# dropped, and was fitted 221 mm long of 340. A piece touching the main one,
# a depth jump between them, is still dropped: a hand.
MODEL_BEHIND_MIN_PX = 10
MODEL_BEHIND_SHARE = 0.7
MODEL_BEHIND_DEPTH_RATIO = 3.0
# Share of point-to-point in the ICP metric; the rest is point-to-plane, which
# alone lets a flat face slide along itself.
MODEL_POINT_WEIGHT = 0.1
# The final descent on the cost: step sizes, coarse to fine, and at most this
# many moves at each.
MODEL_POLISH_STEPS_M = (0.004, 0.002, 0.001, 0.0005)
MODEL_POLISH_STEPS_DEG = (4.0, 2.0, 1.0, 0.5)
# With a measured scale, the descent also scales the mesh about its centre by
# these fractions: re-fit from the rigid pose, a scale search alone stops a
# few percent short of the cost's own minimum.
MODEL_POLISH_STEPS_SCALE = (0.02, 0.01, 0.005, 0.0025)
MODEL_POLISH_ROUNDS = 6
# A move must lower the cost this much: less is the depth noise talking.
MODEL_POLISH_MIN_GAIN = 0.002
# The mesh gives the part's shape, not its size: a demonstration's
# reconstruction comes out at whatever size it does -- ring_insertion's bar
# at about twice the simulated bar, pick_placement's tester and sample at
# 0.41 times theirs. One uniform scale is searched over this range, first on
# a grid this coarse (ratio between neighbours), then about its best in steps
# this fine. A best scale at either end of the range means the shape does not
# match, and the fit is refused.
MODEL_SCALE_RANGE = (0.25, 4.0)
# A mesh sent with its scale known (meta["models"][i]["known_scale"]: a CAD
# model, a measured reconstruction) is fitted at that scale alone, neither
# measured nor stretched: the size is the strongest evidence of which proposal
# is the part, and a shape fitted at a free size also fits a patch of a larger
# part or a smaller look-alike.
MODEL_SCALE_COARSE_RATIO = 1.08
MODEL_SCALE_FINE_RATIO = 1.015
# The coarse grid first spans this factor either way of a start's size.
MODEL_SCALE_BRACKET = 2.0
MODEL_SCALE_ROUNDS = 10
# Nor are its proportions exact: the simulated tester is 1.44 times as wide as
# it is deep, its reconstruction 1.09. So the mesh may also be scaled along
# each of its own box axes, by up to this factor either way of its overall
# scale, where that lowers the cost by MODEL_STRETCH_MIN_GAIN. A pose search
# also starts from the proportions of the observed box, when they differ from
# the mesh's by more than MODEL_SCALE_START_RATIO.
MODEL_STRETCH_MAX = 1.4
MODEL_STRETCH_MIN_GAIN = 0.03
MODEL_STRETCH_START_RATIO = 1.15
# The proportions are fitted together with the pose too, by Gauss-Newton over
# both (_ModelFit.stretch): the descent, one axis a step at a time, moves a
# scale only a few percent from where it starts. The simulated bar, 340 by
# 27 mm, has its reconstruction's length (386 mm) scaled 1.27 times its
# width's (39 mm): at one scale it came out 268 mm long on a view of 334 mm,
# the points past its ends 32 mm off it, and was refused; the descent took
# its length from 0.696 to 0.703 of the mesh's. A round changes a scale by at
# most MODEL_STRETCH_STEP. Its pairs narrow to MODEL_STRETCH_RADIUS_M, not to
# MODEL_TOL_M: past that, the points beyond a short mesh's ends no longer
# draw it out, and the bar stopped 6 mm short at each.
MODEL_STRETCH_ROUNDS = 30
MODEL_STRETCH_STEP = 1.15
MODEL_STRETCH_RADIUS_M = 0.012
# Within this share of its bound, a stretched scale is at it.
MODEL_STRETCH_EDGE = 0.99
# Fitted so, a scale only grows -- what the fit at one scale leaves wrong is
# a mesh too short where the view shows more: the bar's length, the ring's
# thickness; the cost leans to a mesh a little small (MODEL_SCALE_MIN_GAIN)
# -- and by MODEL_STRETCH_MAX at most: this fits the mesh's proportions, its
# size is measure_scale()'s. Unbounded, a pose at scale 1 on a patch of the
# tester grew 1.9 times along its length and 3.1 across, 152 mm long where
# the part is 194, and fitted better than the right pose. A scale that ends
# at that bound wants more, as one at the end of MODEL_SCALE_RANGE does: the
# shape does not match, and the stretch is not taken -- the tester's mesh, on
# a sample, pressed both its long sides there, flattened onto it. (The bar
# grew 1.30 times, the ring's thickness 1.15.)
# An axis is stretched only where the view spans it: the observed points
# cover at least this share of the mesh's extent along it, from their
# MODEL_SEEN_PERCENTILE-th to their (100 - it)-th percentile. A bar seen from
# one side shows half its round, and its depth stretched to fit that half:
# 0.62-0.76 of it. Between the 5th and 95th percentiles an upright tester's
# depth, which only its top face spans, came to 0.61 -- 0.88 between the 2nd
# and 98th -- and the proportions its fit had started from, 189 x 62 x 41 mm
# of the part's 194 x 62 x 42, were dropped for one scale: coarse, 48 mm wide.
MODEL_STRETCH_SEEN = 0.82
MODEL_SEEN_PERCENTILE = 2
# A length is measured only where the view shows both its ends: of the mesh's
# camera-facing points in the last MODEL_END_SHARE of it at either end, no
# more than MODEL_END_HIDDEN may be hidden -- something nearer in front, or
# off the image. (An end facing away, with few such points, is not hidden.)
# A hidden end costs nothing, so a mesh shortened into it fits as well: a
# tester half behind a board came out 164 mm long, not 182, and a sample's
# scale 1.69, not 2.44. So the mesh is stretched only along lengths whose
# ends show; and a size whose two longest lengths do not both show is still
# used -- it is the best there is, and a ring's arc gets its size right half
# hidden -- but not counted as measured, so a later, clearer view measures it
# again (MODEL_UPGRADE_INTERVAL_SEC).
MODEL_END_SHARE = 0.08
MODEL_END_HIDDEN = 0.3
# And a length is measured only where the part is seen to stop at its ends:
# MODEL_END_PROBE_M past the end's camera-facing points, no more than
# MODEL_END_HIDDEN of those in the image lie behind something nearer --
# outside the part's own mask, off the support. A mesh fitted short into an
# occluder ends just in front of it, its own end in view: the bar seen nearly
# end-on, a ring on it, came out 221 mm of 340 and measured (ends_open()).
# Past the image's edge is no evidence either way -- the RI bar's end 2 cm in
# from it was measured right -- unless all but a few probes are.
MODEL_END_PROBE_M = 0.03
# A size the view could not measure is still at least what was seen: along
# each of its axes the mesh spans the observed points (2nd to 98th
# percentile), less this for depth noise. A hidden end costs nothing, so half
# behind a board a sample came out 48 mm wide where 60 mm of it showed; its
# scale is raised until it covers what showed. (A measured size stands: a
# reconstruction's probe a little short is no reason to grow the whole part.)
MODEL_COVER_NOISE_M = 0.004
# Nor can a view measure how far a part resting on its support stands off it
# where it sees only its top: a plate's thickness from above, a tester lying
# on its side seen with neither end of its height. The fit keeps the mesh's
# proportion there, and a reconstruction's is not the part's: PP-S4-1's
# sample, 3 mm thick, came out 10.8 mm, 5.8 mm into the bench and 1.4 mm over
# its top; PP-S1-3's tester, 62 mm high on its side, 56 mm, its bottom 6.3 mm
# clear of the bench (2026-11-05: placed 15 mm off where it was planned). So
# along the box axis that stands within MODEL_DOUBT_APART_DEG of the
# support's normal, where the view does not measure it (axes_seen()), a part
# resting there -- its mesh's bottom off the support by no more than the axis
# could be wrong by, MODEL_STRETCH_MAX of it, and the tolerance -- is as high
# as the view sees its top: its observed points' height over the support at
# their 98th percentile, less what depth noise adds there, 2.054 times the
# noise depth_noise() measures (the normal distribution's 98th percentile);
# its bottom on the support. Where the support is not the part's (held in
# the air, on another part), the bottom is far off it and nothing changes.
# A fit refused for going into the support is tried so too: a plate's mesh 4
# times as thick as the part went 9 mm into the bench, and was refused.
MODEL_SUPPORT_TOP_PERCENTILE = 98
MODEL_SUPPORT_TOP_SIGMAS = 2.054
# Where a part lies along its length only its ends tell: a bar's round body
# fits anywhere along it. A hand round one end of RR-S1-2's bar and the ring
# on it hid that end, the length went unmeasured (255 mm of 340), and the bar
# was placed 41-59 mm along its length off the part; RI-S1-1's ring went in
# 40 mm too deep (2026-11-05). So an end counts only where the view shows it:
# not hidden (hidden_ends()) and seen to stop (ends_stop()), or resting on
# the support. Where one end is seen and its mesh's end is further from the
# observed points' end there (the tenth point from it: as few as tell where an
# end is, hidden_ends()) than the tolerance, the mesh is slid along its
# length onto it; and what the ends tell is replied (model_axial): which are
# seen, where the mesh's ends are and how far the part was seen to reach,
# and where none is seen, how far along its length the mesh could slide and
# still span what was seen. A part round about its length -- a bar -- is
# slid so on every frame its carried fit is checked on too, and the reply
# says what that frame's view tells (carried_axial()): neither the tracker
# nor the correction holds a round body along its length, and the live bars
# went 40-59 mm off where the init's view had put them within 3 mm.
# Rotations refined from each start of the pose search, beside the best
# MODEL_KEEP of all: at its own size a mesh can sit snugly on a patch of a
# part twice as big, cheaper than any start at the part's size.
MODEL_KEEP_PER_START = 2
# And about which axis a mesh is round: turned about a box axis through its
# box's centre, by every MODEL_SYMMETRY_STEP_DEG up to a half turn, nine in
# ten of its points (MODEL_SYMMETRY_PERCENTILE) within MODEL_SYMMETRY_SHARE of
# its width across the axis of its surface -- a share, so it is the shape's,
# at any size: in millimetres, a box 31 mm long came out round, its turn thrown
# away -- and 99 in 100 within MODEL_SYMMETRY_TAIL_SHARE, for a feature on one
# side few points fall on, a mug's handle. The fit cannot tell such a turn from
# the pose, and the tracked turn about the axis drifted: 9.1 deg over ten frames
# of one still image, where nothing else moved. About such an axis the
# published turn is held (bboxes()): each reply turns the pose about it as
# little as lies between it and the last one sent, the first with another of
# the mesh's box axes as near the image's up as it goes. A mesh stretched
# unevenly across the axis, by more than MODEL_SYMMETRY_STRETCH, is round about
# it no longer. Neither is a turn about it any doubt (model_spin_doubt).
MODEL_SYMMETRY_STEP_DEG = 20
MODEL_SYMMETRY_PERCENTILE = 90
MODEL_SYMMETRY_SHARE = 0.14
MODEL_SYMMETRY_TAIL_PERCENTILE = 99
MODEL_SYMMETRY_TAIL_SHARE = 0.35
MODEL_SYMMETRY_STRETCH = 1.02
# A part round about an axis is as wide one way across it as the other, this
# ratio at most: rings, bars, cans, 1.01 or under; a block with rounded corners,
# as near itself under a turn but 1.21, shows which way round it lies.
MODEL_SYMMETRY_SIDES = 1.1
# Which way round a part lies. Turned onto its own box -- half round about a
# box axis, or a quarter round about one whose other two sides are within
# MODEL_DOUBT_SIDES of each other -- a mesh not the same so (ObjectModel.box_turns,
# MODEL_DOUBT_SAME_SHARE), nor turned about the axis it is round about, is
# refitted there (settle_turns()). Cheaper by more than MODEL_TURN_COST_SLACK and
# as good a fit, it is taken; within that either way it is sent in doubt
# (model_doubts), for the person to say which. What tells one way round from
# another is often a feature few points fall on -- a probe, a groove, a notch --
# which the view need not show, and tests of the image or the mask's outline
# made for one part's features told others the wrong way round; the cost alone
# decides, and where it cannot, says so. Two ways round it cannot tell apart by
# less than what the two leave unexplained together -- each its share of
# observed points off the mesh and of the mesh's points the view contradicts --
# for that much a mesh off the part can be off either way: the tester's
# reconstruction, its probe off the real one, cost 0.054 more turned end for end
# seen whole and 0.068 less turned the wrong way in PP-S1-3. Nor by less than
# MODEL_TURN_COST_SLACK, the depth's noise: half behind PP-S4-1's occluder,
# 0.4779 against 0.4780.
MODEL_TURN_COST_SLACK = 0.02
# Refitted to within MODEL_DOUBT_APART_DEG of the pose or of one already in
# doubt -- but for a turn that leaves the part as it was -- it is that one.
MODEL_DOUBT_SIDES = 1.15
MODEL_DOUBT_APART_DEG = 30.0
# The same so: turned, its samples no further from its surface -- the
# MODEL_DOUBT_SAME_PERCENTILE-th percentile of their distances, so a feature on
# a thousandth of it counts -- than as many samples drawn afresh from it come out
# on any of MODEL_DOUBT_SAME_DRAWS draws. A ring turned over or a bottle about
# its length is so; a plate's notched end, 1.4 times as far, is not. And so is
# one turn after another that is so: each measured alone, a turn just past the
# floor can be what two just under it make -- RR-S1-1's ring came out the same
# turned over about one diameter and turned half round about its hole, not
# turned over about the diameter square to that one, which those two are, and
# that went out in doubt (2026-11-04). The turns that are so are taken as all
# they make, one after the other (_box_turns()).
MODEL_DOUBT_SAME_PERCENTILE = 99.9
MODEL_DOUBT_SAME_DRAWS = 8
# Each half turn about a box axis -- the three, any two making the third --
# is refitted and polished, its own translation found, whatever its first
# refit costs: on PP-S4-2's tester, 41 % behind the box, the right way round,
# half round about the box's short axis from the fit, cost 0.336 polished
# against the fit's 0.314 -- in doubt by the margin, 0.10 -- but was dropped
# before polishing for costing more than 0.354, and the fit went out 180 deg
# and 23 mm off with no way round in doubt the right one (2026-11-05). A
# quarter turn, where one is tried, still is only where its first refit is
# near. And every way round that fits at all, in doubt or not, is a
# hypothesis of the binding (Binding): beyond the margin it is not sent in
# doubt (model_doubts), but is in model_binding, "in_doubt": false, for the
# views' evidence to bring into doubt and the person to choose.
MODEL_DOUBT_HALF_TURNS = True
# The pose search runs at scale 1, and also at the observed size when that
# is this far from the mesh's: a mesh twice a bar's length fits anywhere
# along it, and scaled about its centre afterwards it leaves the part.
MODEL_SCALE_START_RATIO = 1.3
# The scale is measured only on a view that hides no more than this: a smaller
# mesh slid into the hidden part can fit as well as the right one. On rendered
# scenes up to half hidden, the ring's scale came out within 1%; for the
# pick_placement parts it came out nearer 1 than the truth, and was then kept
# at 1 by MODEL_SCALE_MIN_GAIN, never applied wrongly.
MODEL_SCALE_MAX_HIDDEN = 0.5
# Hidden more, a part takes the other parts' scale where the init gives one
# (fit_model_box()'s scale_hint; see LOCATE_SCALE_AGREE): PP-S4-1's tester,
# 51% behind the box, kept the start it was searched at, 2.94, and came out
# 235 mm long of 182; the sample beside it measured 2.40.
# A measured scale replaces 1 only if it lowers the cost by this much. Too big
# a mesh shows at once, as open points off the part; too small a one costs
# only the few points past its edges, so noise alone can favour a slightly
# smaller one. A mesh that is the part's size so keeps exactly 1, which the
# agent's demonstration grasps require of a native mesh pose.
MODEL_SCALE_MIN_GAIN = 0.03
# A part's shape -- its scale along each of its mesh's box axes -- measured on
# an earlier view (LOCATE_KNOWN_SCALES) is fitted on each later one too, once
# that view's own fit has settled which way round the part lies, about its
# centre, and kept unless the view's own shape costs this much less: a rigid
# part's size is its own, whichever side it shows. Turned over in
# PP-S3-2 (2026-11-02), the tester showed the camera its underside, which the
# reconstruction has only roughly; its fit came out 154-158 mm long of 182,
# measured, both ends seen, its centre 12 mm along its length off the part's,
# and the placement carried that: four runs 14-15 mm off. On that view the
# shape measured upright cost less (0.473 against 0.490; p95 6.0 mm against
# 9.5) and was 1 mm off along its length, but no start of the pose search was
# near it. Not before the ends are settled: the probe under the body, at the
# full length no test saw it, and the colour and the mask turned the part
# end for end, 20 mm off. Nor with MODEL_SCALE_MIN_GAIN's lean to the shape
# held, the lower cost deciding, as between the pose search's starts: first
# seen turned over, the tester held its 154 mm on the upright view after,
# 0.467 against its own 0.441, and would have held it from then on.
MODEL_KNOWN_SHAPE_MIN_GAIN = 0.0
# A part hidden more than MODEL_SCALE_MAX_HIDDEN whose fit passes the coarse
# acceptance neither at its start nor at the other parts' scale is fitted once
# more at the scale, along its box axes, it was located at this init: the
# mesh was fitted to this view then. RR-S1-2's bar, half behind the fist
# holding it, its end showing below it: the reconstruction's 386 x 39 mm,
# thicker for its length than the part, at the ring's 0.80 reached neither
# end, at 1 was too thick (contradiction 0.39); it was located at 0.83 along
# its length and 0.66-0.92 across.
MODEL_LOCATED_FALLBACK = True
# Retry failed localization on fresh frames, with one background fit at a time.
MODEL_RETRY_INTERVAL_SEC = 3.0
# A coarse fit, or one whose size the view hid too much to measure, is fitted
# again on later views this often, and replaced only by a better one -- fine
# over coarse, measured over not. A fine, measured fit is kept: the part's
# size is the part's, and a later view half hidden cannot measure it better.
MODEL_UPGRADE_INTERVAL_SEC = 6.0
# A better one turned MODEL_REFIT_TURN_DEG or more from the fit published --
# half round about its length, or end for end -- is no refinement but the
# other way round, where views a little apart did not agree: the fit is kept,
# and that pose sent in doubt with it (model_spin_doubt, model_ends_doubt)
# for the person to say which. Taken, it would turn the part under the Agent
# between two replies: PP-S1-4's tester came out end for end either way
# round on SAM2's mask and on one cut by its colour, its retries settling it
# on a profile 0.12 against 0.13, and a client running the scene saw its z
# reversed against the replay's (2026-10-30).
# A fine refit over a coarse one may improve size and alignment, but may not
# publish a half turn just because its geometric fit ranks higher. First
# check the refined shape in the currently published direction on this
# frame. Upgrade only if that direction itself passes the fine-fit checks;
# keep the other direction in doubt (preserve_refit_direction()).
MODEL_REFIT_TURN_DEG = 90.0
# Kept, a fit is carried by the tracked motion, and the motion can go wrong
# with the track not lost: a ring lifted face-on in the gripper was tracked
# through its hole, on the camera's ray to it but 147 mm too deep (the far
# background seen through the hole pulled its depth), the mesh with it, and
# the Agent saw the gripper holding nothing. So every frame the carried mesh
# is scored on what the camera sees there, on MODEL_CHECK_POINTS (mesh,
# observed) points, and a frame where the observed points are off it
# (carried_fit_rejection()) is replied without it: held up, the ring's track
# then jumped 217-240 mm off on one frame in three to five, the frames
# between them right, and every one of those frames went out. Off it on
# MODEL_CHECK_FAILS frames running, the fit is no longer the part's where
# the tracker carries it, and it is fitted again at once. The points are
# as many as the fit's acceptance had: at a third of the mesh's, seen ones
# lie sparse on a thin part, and RR-S1-2's bar, fitted at p95 8.7 mm on
# all, came out 17.8 mm on 1000 and 600 -- off the part on the frame it was
# fitted on. All of them cost 2 ms a frame more.
MODEL_CHECK_POINTS = (MODEL_SAMPLES, MODEL_FINE_POINTS)
MODEL_CHECK_FAILS = 2
# The tracked motion is not the part's once its points are gone: the robot
# turning a part over leaves the tracker's points on the side now underneath,
# and on the frames after it set the part down the motion moved a part lying
# still. On PP-S3-1 (2026-11-05, 08:44:49-08:45:40) a refit came back right
# on the frame it was fitted on, was carried to the next by that motion and
# was off the part there (fitness 0.04-0.09, 95th percentile 56-59 mm), was
# turned another way round to fit, withheld, dropped and fitted again, every
# 13 s for 30 s and more, the pose sent 108 deg off between. So the carried
# fit is not all a frame is checked against: off the part where its motion
# carries it, it is tried where it was last seen on the part (a part lying
# still: still_motions), and where the robot says it set it down (a pose
# prior, meta["priors"]), each corrected on the frame as the carried pose is;
# the first of them on the part is kept, the pose held there. Where the
# motion's pose and the held one are further apart than the correction moves
# a pose (MODEL_CORRECT_MAX_M, MODEL_CORRECT_MAX_DEG), or once it is held,
# both are corrected and the cheaper kept -- the held one while it costs no
# more than the motion's
# by MODEL_CORRECT_MIN_GAIN: on the same test, the pose the motion carried,
# corrected, passed the check 8-16 deg off a tester lying still, and was
# taken over the held one on as little. The motion carries it again once the
# held pose is off the part: the robot moved it. And a refit is not sent before
# it is on the part on the frame it is sent with: carried by the motion from
# the frame it was fitted on, where it was fitted, or at the prior, the
# cheapest on the part (adopt_refit()); on none, it is not taken.

# Before it is scored, the carried mesh pose is corrected on the frame's
# depth (correct_carried_pose()): MODEL_CORRECT_ROUNDS ICP rounds from it,
# pairs within MODEL_CORRECT_RADIUS_M, kept where they lower the fit's cost
# by MODEL_CORRECT_MIN_GAIN. A small part with little texture gives the
# tracker few points to hold it: of a still scene's frames, RR-S1-1's ring
# turned by up to 9 degrees, RI-S4-2's by 5.7, where its depth had not
# moved. Around the pose the fit found, the cost is flat to a degree or a
# few millimetres, and where ICP settles in that flat depends on where it
# starts; so the correction is kept from frame to frame, in the mesh's frame
# so it rides along with the tracked motion, and each frame starts from it,
# not from the tracked pose: a still part then stays still. Where that puts
# the mesh more than MODEL_CORRECT_MAX_M / MODEL_CORRECT_MAX_DEG from the
# tracked pose, the tracked pose is corrected too, and kept unless the
# carried correction fits by MODEL_CORRECT_MIN_GAIN better: RR-S1-1's ring's
# tracked pose jumped 14 degrees on one frame in three, its depth not at all.
MODEL_CORRECT_ROUNDS = 5
MODEL_CORRECT_RADIUS_M = 2 * MODEL_TOL_M
MODEL_CORRECT_MIN_GAIN = 0.002
MODEL_CORRECT_MAX_M = 0.02
MODEL_CORRECT_MAX_DEG = 10.0


# The surface seen of each object, kept in its frame at the init frame: every
# SURFACE_EVERY-th frame its mask's points, support removed, are carried back
# there by its tracked motion and binned into voxels. A part's size is
# measured once and kept; so is what was seen of it, so a part half hidden
# now still shows what earlier views saw. Voxels seen on fewer than
# SURFACE_MIN_SEEN frames are left out once there have been that many: depth
# noise, or a hand that passed in front once.
SURFACE_VOXEL_M = 0.003
SURFACE_EVERY = 2
SURFACE_PIXELS = 1500
SURFACE_MAX_VOXELS = 6000
SURFACE_MIN_SEEN = 2
# Replies carry each object's surface this often (in frames), and on the first.
SURFACE_PUBLISH_EVERY = 5

# Which way round a fitted mesh lies on its part -- its binding -- is one fact
# for as long as the part is tracked: the tracker carries the part's motion,
# and every way round rides on that one motion. So the ways round the init
# view could not tell apart (model_doubts) are kept as hypotheses (Binding),
# and every BINDING_EVERY-th frame each is scored on the view until one is
# the part's: no new pose search, only new evidence, gathered while the part
# turns in the hand or the camera moves.
#
# What tells two ways round apart is where they differ: the samples of one
# farther than MODEL_TOL_M from the other's surface -- a probe at one end, a
# handle on one side. Such a sample seen by the camera at its depth on the
# part speaks for its way round; seen through -- the camera sees past it, or
# the bare support there -- against it (sample_evidence()). Seen through
# counts however hidden the feature is on later frames: a frame that looked
# through the place where one way round has its probe rules that out for
# good (the free space, carved). A sample keeps the verdict of most of the
# frames it was seen on, and a frame is looked at only where the part has
# moved against the camera, since the last one looked at, by more than the
# carried correction may move it on its own (MODEL_CORRECT_MAX_M,
# MODEL_CORRECT_MAX_DEG): one still image seen again, its pose re-corrected
# within the cost's flat, re-counted the samples at the edges, and PP-S1-3's
# tester, the right way round left in doubt by the init's view, was ruled
# out after five frames of it.
#
# The evidence is counted in what is independent of the rest, not in
# samples: a feature -- the samples where one way round has what the other
# has not, joined where within MODEL_FRAGMENT_GAP spacings of each other, as
# a mesh's pieces are; under MIN_BOX_POINTS samples, below what the fit
# tells -- gives one vote on each frame looked at, the verdict of most of
# its samples seen there (seen on: one tolerance's area of it or more).
# Counted in samples, a reconstruction a little off the part -- its scale,
# its probe -- had hundreds of them wrong together, and ruled the right way
# round out at odds of 1e6 to 1e21: PP-S3-2's tester, its probe in view, and
# PP-S1-4's (2026-11-05). A vote is wrong as often as a sample on what the
# ways round agree on is seen through (Laplace's rule of succession over
# them: the mesh's error, the tracking's, the mask's). And beside the two
# ways round there is a third: that the mesh does not match the part where
# they differ, every vote then a coin's toss. One way round beats another
# when the odds of the votes for it against the other and that third reach
# BINDING_CONFIDENCE, on votes from SURFACE_MIN_SEEN frames looked at or
# more: one view tells nothing a mesh's error might not, and a vote is worth
# at most twice a toss against the third, so several independent votes are
# needed. A reconstruction whose probe is off the part's has
# its probe seen through the right way round too, and then the third wins.
#
# A way round beaten is dropped; the one left is the binding, published from
# then on without doubts -- turned to it where it was not the one published,
# with a new fit id, so the Agent sees the change. A part at rest on its
# support, no robot pixels on it, also rules out a way round it could not
# rest in (rest_margin()): its samples' centre outside the polygon of those
# within the tolerance of the support, by more than the tolerance, where
# another way round has its centre inside its own. Leaning on something
# else, every way round is outside, and nothing is ruled out. The person can
# settle it too, naming a way round (meta["bind"]) -- or the Agent, on its own
# look at the camera's view, or the robot, of a part it set down: each bind
# says whose it is ("by", "person" where it does not), and so does the
# binding's "by" from then on. Each reply says how far
# the binding is (model_binding), and where each way round in doubt has the
# features that would tell it, up to BINDING_FEATURE_PX image points of them.
BINDING_EVERY = SURFACE_EVERY
BINDING_CONFIDENCE = 0.95
BINDING_FEATURE_PX = 60
# Two ways round turned end for end ("ends", turn_kind()) differ where the
# part's two ends differ -- a notch at one end, a probe -- and that can be
# under what the evidence above counts. PP-S1-1's sample has a notch through
# it at one end, 8.6 x 35.7 mm as fitted, on a plate 1.5 mm thick: 6-7 of
# its 3000 samples lie farther than MODEL_TOL_M from the other way round's
# surface (a feature takes MIN_BOX_POINTS), and in the image the notch is
# 3.9 px deep, inside MODEL_MASK_GROW_PX's band. The ways round cost about
# the same, and the published one had the notch at the wrong end
# (2026-11-07). So every frame looked at also compares the two silhouettes
# (the meshes' triangles, each way round corrected on the frame as for the
# evidence above) with the mask, on the pixels only one of the two covers:
# such a pixel in the mask, at that one's depth, is for it; one off the
# mask, or seen past it, is against it -- unless something nearer hides it,
# or there is no depth there (ends_evidence()). Off the mask is against it
# however near the depth: the bench seen through a notch in a thin plate is
# within the tolerance of the plate's face. Pixels within
# ENDS_SILHOUETTE_EDGE_PX of the mask's edge tell nothing: each way round is
# corrected on its own, and one a pixel off along a long side puts a strip a
# pixel wide all one way -- RI-S1-1's bar, 36 px along its top edge outvoted
# its ends 120 to 47 for the way round with the shorter cone at the tip (the
# view shows the longer), where without the strip the ends split 51 to 44.
# The pixels so told must be ENDS_SILHOUETTE_MIN_SHARE of the two
# silhouettes' area or more, and ENDS_SILHOUETTE_AGREE of them for one way
# round, for a vote, counted with the features' (BINDING_CONFIDENCE). On the
# init's view, a way round the silhouettes vote for over the published one
# is published in its place, the published one kept in doubt: one view
# decides nothing, but of near-equal fits the one the view favours goes
# first. PP-S1-1's sample: 26 to 9 and 24 to 8 px (2.6% and 2.3% of its
# 1371) for the two ways round with the notch where the image has it, the
# published one turned to one of them; its tester and RI-S1-1's bar, under
# ENDS_SILHOUETTE_MIN_SHARE and ENDS_SILHOUETTE_AGREE, kept as fitted.
ENDS_SILHOUETTE_MIN_SHARE = 0.02
ENDS_SILHOUETTE_AGREE = 0.7
ENDS_SILHOUETTE_EDGE_PX = 1
# A feature too thin for the silhouettes still moves where the part ends:
# PP-S1-3's tester lies on its side, its probe 1-2 px thick and about 10 px
# long in the image. The silhouettes split 48 to 23 and 43 to 24 for the
# published way round, its probe at the wrong end (2026-11-07): the probe
# pixels are its mask's edge, and a strip along a long side outvoted the
# rest. Where the part ends along the mesh's long box axis does not: the
# published mesh's probe stuck 22 mm past the part seen at one end, and the
# part seen, its probe, ran 27 mm past the mesh at the other. So each pair of
# ways round turned end for end is also judged by its ends (ends_extent()):
# each way round, at its own corrected pose, how far each of its two ends
# along its long axis lies from where the part seen ends there -- the tenth
# point in from the end, as _ModelFit.axial() measures it -- summed over the
# ends neither of the two has hidden (hidden_ends()); one way round's sum
# less than the other's by ENDS_EXTENT_SHARE of the part's length or more is
# a vote for it, counted as the silhouettes' are, and on the init's view it
# goes first as they do. A share, not a length: the same for a part near or
# far, large or small.
ENDS_EXTENT_SHARE = 0.05
# Where the server keeps an archive (--archive), a binding settled by the
# evidence or the person files the surface seen of the part under it --
# SurfaceMemory's, carried into the mesh's frame, within the mesh's box grown
# by twice MODEL_TOL_M -- under the mesh's id, with its measured shapes
# (LOCATE_KNOWN_SCALES), when the session ends; a voxel's runs are counted.
# On later inits, in later runs, the shapes are known from the start. The
# surface is the part's as seen, in the mesh's frame, for whoever reads the
# archive; it is not evidence for a binding: each frame's tracking error
# spreads it a few millimetres either side of the part's, past MODEL_TOL_M
# at its edges, and as the mesh's samples on a rendered tester turning (the
# points farther than the tolerance from the mesh, seen on two runs) it stood
# out past the part's outline the right way round, was seen through, and
# settled the next run the wrong way.

# Published motions are smoothed: on a still scene the pipeline's pose hops
# among a few near-equal registration solutions, ~1 cm and ~1 deg apart. Each
# frame moves the published pose a fraction of the way to the new one, that
# fraction growing with the step -- SMOOTH_MIN_ALPHA for a step well inside
# SMOOTH_FULL_M / SMOOTH_FULL_DEG, all the way (no lag) for one past them, so
# real motion is followed at once. Only the reply is smoothed; the meshes,
# surfaces and the pipeline itself keep the raw motion. 1.0 turns it off.
SMOOTH_MIN_ALPHA = 0.3
SMOOTH_FULL_M = 0.03
SMOOTH_FULL_DEG = 3.0

# A track request is answered in time even when its frame is not: on
# PP-S1-4 (2026-11-05, from 07:57:52) a frame took longer than the client's
# 15 s, every request after it timed out, and the trial ended 35 s on with no
# reply; RI-S1-2 (2026-11-01) and RI-S4-1 (2026-11-02) stalled so too. What
# took the time is not known: no refit runs on the request's thread (a
# background fit in mesh_fit_executor slowed the request's own checks from 83
# to 151 ms a frame on a rendered tester, 2026-11-05, not to seconds), and
# nothing waits on a lock. So a client that says how long it waits
# (meta["timeout_ms"]) has its track requests worked on another thread, and
# is answered by SERVER_REPLY_SHARE of that whether the frame is done or not:
# not done, with the last frame's reply, marked "stale" -- its frame_id that
# frame's -- and a request that comes while a frame is still worked on is
# answered so at once, its frame dropped. The share leaves the rest of the
# allowance to the transfer and the client's own work. A frame worked on
# longer than the whole allowance has the stack it is in printed, once per
# allowance, for what holds it. A client that says nothing is answered as
# before, when the frame is done.
SERVER_REPLY_SHARE = 0.5

# GPU memory still allocated after a release is held by something an earlier
# session left behind. A fresh process frees all but 8-16 MiB, through inits,
# tracks on the track thread, losses and inits again (PP-S1-1, 2026-11-06).
# The Gazebo trial of 2026-11-06 had 6519 MiB allocated after every release
# on this 8 GB card. Its init then reached 8415 MiB allocated and 10960 MiB
# reserved, and the excess spilled into system memory: the locate took
# 226-276 s (25 s fresh), the init 694 s (55 s fresh), and frames 1.4-3.6 s
# (0.78 s fresh), and the client timed out on the init. The trace found it:
# the autograd graphs of TAPIR's keyframe query features, made on the track
# thread with grad on (no_grad_thread()).
# Over GPU_HELD_WARN_MIB so is said at every release. With --trace-gpu-memory
# the allocator also keeps the Python stack of each live block, and the
# warning lists where the held blocks were allocated (GPU_TRACE_SITES of
# them), how much of the held memory Python objects reach and through what
# (GPU_TRACE_HOLDERS of the paths to it from the server, a module's globals
# or a thread's stack) -- memory Python does not reach is held from C++, such
# as a thread's autocast cache. Every
# GPU_TRACE_EVERY_FRAMES track frames the memory is logged too, and the
# allocation sites are listed again each time the memory grows by another
# GPU_HELD_WARN_MIB over the init's.
GPU_HELD_WARN_MIB = 256
GPU_TRACE_EVERY_FRAMES = 20
GPU_TRACE_SITES = 12
GPU_TRACE_HOLDERS = 12
# How far gpu_python_holders() walks, and how much of each path it names.
GPU_TRACE_VISIT = 3_000_000
GPU_TRACE_PATH = 3
# A frame of the stack is "ours", and names the site, in these.
GPU_TRACE_OURS = (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                  f"{os.sep}sam2{os.sep}", f"{os.sep}tapnet{os.sep}")


def gpu_alloc_sites(limit=GPU_TRACE_SITES):
    """(MiB in live blocks, [(MiB, site)] largest first): torch's snapshot
    of the caching allocator's live blocks, each by where it was allocated --
    the innermost frames of ours (GPU_TRACE_OURS), else of any code."""
    sites = collections.Counter()
    for segment in torch.cuda.memory._snapshot().get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            frames = block.get("frames") or []
            ours = [f for f in frames if any(part in f["filename"] for part in GPU_TRACE_OURS)]
            site = " <- ".join(f"{os.path.basename(f['filename'])}:{f['line']} {f['name']}"
                               for f in (ours or frames)[:4]) or "(allocated before the trace began)"
            sites[site] += block["size"]
    return sum(sites.values()) / 2**20, [(size / 2**20, site) for site, size in sites.most_common(limit)]


def no_grad_thread():
    """A worker thread's start: grad mode off, as on the main thread. Grad
    mode is per thread, and the pipeline's modules turn it off only on the
    thread that builds them (TapirTracker's constructor); on the track
    thread every keyframe's TAPIR query features kept their autograd graph,
    ~720 MiB each, and the 7.9 GB of them outlived every release (2026-11-06,
    found by --trace-gpu-memory)."""
    torch.set_grad_enabled(False)


def gpu_children(obj):
    """(name, child) of what `obj` refers to, for gpu_python_holders()'s
    walk: items, indices, attributes; other referents by their type."""
    if isinstance(obj, (str, bytes, bytearray, int, float, complex, bool, type(None), np.ndarray)):
        return []
    if isinstance(obj, dict):
        return [(f"[{str(k)[:40]!r}]" if isinstance(k, str) else f"[{type(k).__name__}]", v)
                for k, v in list(obj.items())]
    if isinstance(obj, (list, tuple, collections.deque)):
        return [(f"[{i}]", v) for i, v in enumerate(list(obj))]
    if isinstance(obj, (set, frozenset)):
        return [("{}", v) for v in list(obj)]
    if isinstance(obj, types.FrameType):
        return [(f" {obj.f_code.co_name}().{k}", v) for k, v in list(obj.f_locals.items())] + \
               ([(" caller", obj.f_back)] if obj.f_back is not None else [])
    named = []
    try:
        named = [(f".{k}", v) for k, v in list(vars(obj).items())]
    except TypeError:
        pass
    slots = getattr(type(obj), "__slots__", ())
    for slot in (slots,) if isinstance(slots, str) else slots:
        if hasattr(obj, slot):
            named.append((f".{slot}", getattr(obj, slot)))
    if named:
        return named
    return [(f"<{type(r).__name__}>", r) for r in gc.get_referents(obj)]


def gpu_python_holders(server, limit=GPU_TRACE_HOLDERS):
    """(MiB of CUDA storage Python objects reach, MiB of it in tensors made
    with grad on (a grad_fn), [(MiB, path)] largest first, up to `limit`): each
    storage under its shortest path from the
    server, a module's globals or a thread's stack (breadth first, up to
    GPU_TRACE_VISIT objects), cut to its first GPU_TRACE_PATH steps."""
    gc.collect()
    with warnings.catch_warnings():
        # Touching deprecated module attributes (torch.distributed.reduce_op) warns.
        warnings.simplefilter("ignore")
        return _gpu_python_holders(server, limit)


def _gpu_python_holders(server, limit):
    storages, graphed = {}, set()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                ptr = obj.untyped_storage().data_ptr()
                storages.setdefault(ptr, obj.untyped_storage().nbytes())
                if obj.grad_fn is not None:        # made with grad on; parameters are not
                    graphed.add(ptr)
        except Exception:
            continue
    names = {threading.get_ident(): None}
    names.update({t.ident: t.name for t in threading.enumerate()})
    roots = [("server", server)]
    roots += [(f"thread {names.get(ident, ident)}:", frame) for ident, frame in sys._current_frames().items()
              if ident != threading.get_ident()]
    roots += [(name, module) for name, module in list(sys.modules.items()) if module is not None]
    found, seen = {}, {id(storages)}
    queue = collections.deque((obj, (name,)) for name, obj in roots)
    while queue and len(seen) < GPU_TRACE_VISIT:
        obj, path = queue.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, types.ModuleType):
            if len(path) > 1:
                continue                    # walked from sys.modules itself
            seen.add(id(vars(obj)))
        if torch.is_tensor(obj):
            try:
                if obj.is_cuda:
                    found.setdefault(obj.untyped_storage().data_ptr(), path)
            except Exception:
                pass
            continue
        try:
            queue.extend((child, path + (name,)) for name, child in gpu_children(obj) if id(child) not in seen)
        except Exception:
            continue
    held = collections.Counter()
    for ptr, size in storages.items():
        path = found.get(ptr)
        held["".join(path[:GPU_TRACE_PATH + 1]) if path else
             "(not from the server, a module or a thread's stack: through an autograd graph, or a cycle)"] += size
    return (sum(storages.values()) / 2**20, sum(storages[p] for p in graphed) / 2**20,
            [(size / 2**20, path) for path, size in held.most_common(limit)])


class PoseSmoother:
    def __init__(self):
        self.last = None

    def reset(self, motions=None):
        self.last = None if motions is None else np.array(motions, dtype=np.float64)

    def __call__(self, motions):
        motions = np.asarray(motions, dtype=np.float64)
        if (self.last is None or self.last.shape != motions.shape
                or not np.all(np.isfinite(self.last))):
            self.reset(motions)
            return motions.copy()
        out = motions.copy()
        for index, (prev, cur) in enumerate(zip(self.last, motions)):
            if not np.all(np.isfinite(cur)):
                continue
            dR = prev[:3, :3].T @ cur[:3, :3]
            w = cv2.Rodrigues(dR)[0].ravel()
            dt = cur[:3, 3] - prev[:3, 3]
            step = max(np.linalg.norm(dt) / SMOOTH_FULL_M,
                       np.degrees(np.linalg.norm(w)) / SMOOTH_FULL_DEG)
            alpha = float(np.clip(step, SMOOTH_MIN_ALPHA, 1.0))
            out[index, :3, :3] = prev[:3, :3] @ cv2.Rodrigues(alpha * w)[0]
            out[index, :3, 3] = prev[:3, 3] + alpha * dt
        self.last = out
        return out.copy()


class SurfaceMemory:
    """One object's seen surface: a voxel's mean point, on how many frames,
    and the mean direction it was seen from (toward the camera). The
    direction says which side of the part a point is on: the top of a thin
    plate is nearer its reconstruction's bottom than its top."""

    def __init__(self):
        self.voxels = {}
        self.frames = 0

    def add(self, points, frame_id, camera=None):
        """Points of one frame, in the object's init-frame coordinates, seen
        from `camera` there (the camera's origin; None leaves no direction)."""
        if not len(points):
            return
        self.frames += 1
        keys, inverse = np.unique(np.floor(points / SURFACE_VOXEL_M).astype(np.int64),
                                  axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        sums = np.zeros((len(keys), 3))
        np.add.at(sums, inverse, points)
        means = sums / np.bincount(inverse, minlength=len(keys))[:, None]
        if camera is None:
            views = np.zeros_like(means)
        else:
            views = np.asarray(camera, dtype=float) - means
            views /= np.maximum(np.linalg.norm(views, axis=1, keepdims=True), 1e-9)
        for key, mean, view in zip(map(tuple, keys), means, views):
            seen = self.voxels.get(key)
            if seen is None:
                self.voxels[key] = [mean, 1, frame_id, view]
            else:
                seen[0] = (seen[0] * seen[1] + mean) / (seen[1] + 1)
                seen[3] = seen[3] * seen[1] + view
                seen[1] += 1
                seen[2] = frame_id
                seen[3] = seen[3] / max(np.linalg.norm(seen[3]), 1e-9)
        if len(self.voxels) > SURFACE_MAX_VOXELS:
            # The least seen, then the longest unseen, go first.
            order = sorted(self.voxels, key=lambda k: (self.voxels[k][1], self.voxels[k][2]))
            for key in order[:len(self.voxels) - SURFACE_MAX_VOXELS]:
                del self.voxels[key]

    def points(self, views=False):
        """The kept surface, (N, 3) in init-frame coordinates; with `views`,
        also each point's direction toward the camera, (N, 3)."""
        least = min(SURFACE_MIN_SEEN, self.frames)
        kept = [(mean, view) for mean, count, _, view in self.voxels.values() if count >= least]
        points = np.array([mean for mean, _ in kept]).reshape(-1, 3)
        if not views:
            return points
        return points, np.array([view for _, view in kept]).reshape(-1, 3)


def encode_points(points):
    """Points as base64 of little-endian float16 (x, y, z)*: a fraction of a
    millimetre over a part's size, a sixth of JSON's bytes."""
    return base64.b64encode(np.asarray(points, "<f2").tobytes()).decode("ascii")


def fit_rank(model):
    """How settled a fitted model is: (fine, size measured)."""
    return (getattr(model, "fitted_quality", None) == "fit", bool(getattr(model, "fitted_measured", False)))
# Acceptance: share of observed points on the mesh, of open mesh points the
# camera disagrees with, and of mesh points under the support surface. On
# rendered scenes with 2 mm depth noise, up to 65% hidden, the right mesh
# scored fitness 0.84-0.98 and contradiction 0.00-0.02; the other
# demonstration part's mesh mostly scored below 0.8 or above 0.1 -- though
# with most of a part hidden, what is left can fit both.
MODEL_MIN_FITNESS = 0.8
MODEL_MAX_CONTRADICTION = 0.1
MODEL_MAX_PENETRATION = 0.1
# A pose short of those is not thrown away: an imperfect reconstruction can
# fit no better and still say which way the part lies and where it is. Within
# these it is published as a coarse fit, marked so, for the Agent to use as a
# prior and measure the rest; below them the mesh does not fit at all.
MODEL_COARSE_MIN_FITNESS = 0.6
MODEL_COARSE_MAX_CONTRADICTION = 0.25
MODEL_COARSE_MAX_PENETRATION = 0.2
# A candidate must also explain the observation's extent: a good fit to a
# small surface patch must not hide a long tail of distant object points.
MODEL_MAX_P95_DISTANCE_M = 2 * MODEL_TOL_M
MODEL_COARSE_MAX_P95_DISTANCE_M = 4 * MODEL_TOL_M
# And within a share of the part's own width (its box's middle side), down
# to a floor that depth noise alone stays under: millimetres say little about
# a small part. Stretched along its axes, pick_placement's tester mesh lay on
# its 24 mm sample within 8.4 mm at the 95th percentile -- a third of the
# sample's width -- where the right meshes, imperfect as they are, keep
# within a fifth of theirs.
MODEL_MAX_P95_SHARE = 0.3
MODEL_P95_FLOOR_M = 0.0045
MODEL_COARSE_MAX_P95_SHARE = 0.5
MODEL_COARSE_P95_FLOOR_M = 0.006
# A mesh outside this size (largest box side, metres) is in other units.
MODEL_EXTENT_RANGE_M = (0.005, 1.0)
# A reconstruction can leave loose fragments round the part: a sloth's specks,
# the furthest 102 mm off its body, made its box 242 mm long of 198, and no
# proposal was the part's size. Surface samples are joined where within MODEL_FRAGMENT_GAP
# spacings of each other; what that leaves under MODEL_FRAGMENT_SHARE of the
# area goes from the fit and the box (not from the mesh's id). Meshes in many
# pieces that touch, as reconstructions' are, are one part so.
MODEL_FRAGMENT_GAP = 3.0
MODEL_FRAGMENT_SHARE = 0.02


def _twist_spread(points, box):
    """Per box axis, how far the mesh comes from itself turned about it
    through the box's centre: (the MODEL_SYMMETRY_PERCENTILE-th percentile of
    its points' distances from its surface, the
    MODEL_SYMMETRY_TAIL_PERCENTILE-th), the most over the turns
    MODEL_SYMMETRY_STEP_DEG apart up to a half turn; at the mesh's own size."""
    tree = cKDTree(points)
    B, c = box[:3, :3], box[:3, 3]
    spread, tail = np.zeros(3), np.zeros(3)
    for k in range(3):
        for deg in range(MODEL_SYMMETRY_STEP_DEG, 181, MODEL_SYMMETRY_STEP_DEG):
            turned = (points - c) @ _rotation(np.radians(deg) * B[:, k]).T + c
            near, far = np.percentile(tree.query(turned)[0],
                                      [MODEL_SYMMETRY_PERCENTILE, MODEL_SYMMETRY_TAIL_PERCENTILE])
            spread[k], tail[k] = max(spread[k], float(near)), max(tail[k], float(far))
    return spread, tail


def _box_turns(points, box, extent, tri, area):
    """The turns of the 24 onto a box's own axes (in the box's frame), but
    the identity, that take the box onto itself (MODEL_DOUBT_SIDES): (those
    that do not take the mesh (`tri`, of `area`) onto itself, those that do;
    MODEL_DOUBT_SAME_PERCENTILE) -- those that do with every one they make,
    one after another."""
    tree = cKDTree(points)
    B, c = box[:3, :3], box[:3, 3]
    floor = 0.0
    for seed in range(1, MODEL_DOUBT_SAME_DRAWS + 1):
        rng = np.random.default_rng(seed)
        face = rng.choice(len(tri), len(points), p=area / area.sum())
        r1, r2 = np.sqrt(rng.random(len(points))), rng.random(len(points))
        drawn = ((1 - r1)[:, None] * tri[face, 0] + (r1 * (1 - r2))[:, None] * tri[face, 1]
                 + (r1 * r2)[:, None] * tri[face, 2])
        floor = max(floor, float(np.percentile(tree.query(drawn)[0], MODEL_DOUBT_SAME_PERCENTILE)))
    out, same = [], []
    for P in _signed_permutations():
        if np.allclose(P, np.eye(3)):
            continue
        to = np.argmax(np.abs(P), axis=0)
        if np.any(np.maximum(extent[to], extent) > MODEL_DOUBT_SIDES * np.minimum(extent[to], extent)):
            continue
        turned = (points - c) @ (B @ P @ B.T).T + c
        if np.percentile(tree.query(turned)[0], MODEL_DOUBT_SAME_PERCENTILE) > floor:
            out.append(P)
        else:
            same.append(P)
    # What one turn that leaves it as it was makes after another leaves it so.
    grew = True
    while grew and out:
        grew = False
        made = [np.eye(3)] + same
        for k, P in enumerate(out):
            if any(np.allclose(P, A @ C) for A in made for C in made):
                same.append(out.pop(k))
                grew = True
                break
    return out, same


def _without_fragments(v, f):
    """`f` less the faces of loose fragments (MODEL_FRAGMENT_GAP), itself where
    there are none. Its own draw: a mesh with none samples as it would."""
    rng = np.random.default_rng(0)
    tri = v[f]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if area.sum() <= 0:
        return f
    face = rng.choice(len(f), MODEL_SAMPLES, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(MODEL_SAMPLES)), rng.random(MODEL_SAMPLES)
    points = ((1 - r1)[:, None] * tri[face, 0] + (r1 * (1 - r2))[:, None] * tri[face, 1]
              + (r1 * r2)[:, None] * tri[face, 2])
    tree = cKDTree(points)
    pairs = tree.query_pairs(MODEL_FRAGMENT_GAP * np.sqrt(area.sum() / 2 / MODEL_SAMPLES), output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(MODEL_SAMPLES,) * 2)
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    keep = sizes >= MODEL_FRAGMENT_SHARE * MODEL_SAMPLES
    if count == 1 or keep.all():
        return f
    keep[np.argmax(sizes)] = True
    return f[keep[labels[tree.query(tri.mean(1))[1]]]]


class ObjectModel:
    """A mesh reduced to what the fit needs, in the mesh's frame and metres:
    surface samples with their normals, and the mesh's own minimal box."""

    def __init__(self, vertices, faces, rng=None, colours=None):
        rng = rng or np.random.default_rng(0)
        v = np.asarray(vertices, np.float64).reshape(-1, 3)
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        if not len(v) or not len(f) or not np.all(np.isfinite(v)):
            raise ValueError("empty or non-finite mesh")
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError("face index out of range")
        whole = f
        f = _without_fragments(v, f)
        # Only for what the box is said to be: None where the mesh's own vertices are.
        self.kept_vertices = None if len(f) == len(whole) else v[np.unique(f)]
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        if double_area.sum() <= 0:
            raise ValueError("mesh has no area")
        # The faces kept, for the silhouette the mesh casts (mesh_depth()).
        self.mesh_v, self.mesh_f = v, f
        R, centre, extent = min_obb(v[np.unique(f)])
        lo, hi = MODEL_EXTENT_RANGE_M
        if not lo <= extent.max() <= hi:
            raise ValueError(f"mesh is {extent.max():.3g} across; expected metres")
        # Area-weighted samples; the order is random, so any prefix is too.
        face = rng.choice(len(f), MODEL_SAMPLES, p=double_area / double_area.sum())
        r1, r2 = np.sqrt(rng.random(MODEL_SAMPLES)), rng.random(MODEL_SAMPLES)
        a, b, c = tri[face, 0], tri[face, 1], tri[face, 2]
        self.points = ((1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b
                       + (r1 * r2)[:, None] * c)
        # The samples' colours (CIELAB), where the mesh's vertices have them (LOCATE_COLOUR_DELTA).
        self.lab = None
        if colours is not None:
            rgb = np.asarray(colours, np.float64).reshape(-1, 3)[f[face]] / 255.0
            rgb = ((1 - r1)[:, None] * rgb[:, 0] + (r1 * (1 - r2))[:, None] * rgb[:, 1]
                   + (r1 * r2)[:, None] * rgb[:, 2])
            self.lab = cv2.cvtColor(rgb.astype(np.float32)[:, None], cv2.COLOR_RGB2LAB)[:, 0].astype(np.float64)
            self.chroma_tree = cKDTree(self.lab[:, 1:])
        # Only the line a normal lies on is used, never which way it points: a
        # reconstructed mesh's faces are not reliably wound one way.
        self.normals = cross[face] / np.maximum(double_area[face, None], 1e-15)
        self.spacing = float(np.sqrt(double_area.sum() / 2 / MODEL_SAMPLES))
        self.box = np.eye(4)
        self.box[:3, :3], self.box[:3, 3] = R, centre
        self.extent = extent
        # How round the mesh is about each box axis (MODEL_SYMMETRY_STEP_DEG),
        # the turns onto its own box that leave it as it was and those that
        # do not (MODEL_DOUBT_SAME_PERCENTILE), and so whether its ends are alike.
        self.twist_spread, self.twist_tail = _twist_spread(self.points, self.box)
        self.box_turns, self.box_same = _box_turns(self.points, self.box, self.extent, tri, double_area)
        long_k = int(np.argmax(self.extent))
        self.ends_alike = any(P[long_k, long_k] < 0 for P in self.box_same)
        used, inverse = np.unique(whole, return_inverse=True)
        self.model_id = hashlib.sha256(v[used].astype("<f4").tobytes()
                                      + inverse.reshape(-1, 3).astype("<i4").tobytes()).hexdigest()
        self.known_scale = None              # see MODEL_SCALE_RANGE; set by decode_models()
        self.fitted_pose = None
        self.fitted_scale = None
        self.fitted_linear = None
        self.fitted_quality = None
        self.fitted_measured = False
        # The fit turned half round about its length, relative to it, where
        # the view fits that about as well (MODEL_TURN_COST_SLACK); and an id
        # of the fit, new with every fit, for what is said of it to hold to.
        self.fitted_spin_doubt = None
        self.fitted_ends_doubt = None
        self.fitted_doubts = []               # see MODEL_TURN_COST_SLACK
        self.fitted_id = None


def _pixels(P, K):
    """Pixel coordinates of camera-frame points; points behind the camera map
    far outside any image."""
    z = P[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    uv = P[:, :2] / zs[:, None] * K[[0, 1], [0, 1]] + K[:2, 2]
    uv[z <= 1e-6] = -1e9
    return uv


def _seen_from_camera(P, K, spacing, shape):
    """Which mesh points are in the image and its nearest surface there.

    Decided by depth rather than normals, which need not point outwards. The
    image is cut into cells two samples wide, so every cell holds some of the
    near surface; a point more than a cell's width behind the nearest point in
    its cell is on a far face.
    """
    h, w = shape
    z = P[:, 2]
    uv = _pixels(P, K)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    seen = np.zeros(len(P), bool)
    if not inside.any():
        return seen
    cell_m = 2.0 * spacing
    cell_px = max(1.0, cell_m * K[0, 0] / float(np.median(z[inside])))
    cols = int(w / cell_px) + 1
    cell = ((uv[inside, 1] / cell_px).astype(np.int64) * cols
            + (uv[inside, 0] / cell_px).astype(np.int64))
    nearest = np.full(cell.max() + 1, np.inf)
    np.minimum.at(nearest, cell, z[inside])
    seen[inside] = z[inside] <= nearest[cell] + cell_m
    return seen


def _rotation(w):
    return cv2.Rodrigues(np.asarray(w, np.float64).reshape(3, 1))[0]


def _skew(a):
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def round_axis(model, scale):
    """The box axis (index) a mesh at `scale` -- one, or three along its box
    axes -- is round about, or None; see MODEL_SYMMETRY_STEP_DEG."""
    s = np.broadcast_to(np.asarray(scale, np.float64), (3,))
    best = None
    for k in range(3):
        sides = [j for j in range(3) if j != k]
        if s[sides].max() > MODEL_SYMMETRY_STRETCH * s[sides].min():
            continue
        width = model.extent[sides].max()
        if width > MODEL_SYMMETRY_SIDES * model.extent[sides].min():
            continue
        share = model.twist_spread[k] / width
        if (share <= MODEL_SYMMETRY_SHARE and model.twist_tail[k] <= MODEL_SYMMETRY_TAIL_SHARE * width
                and (best is None or share < best[0])):
            best = (share, k)
    return None if best is None else best[1]


def _overall(scale):
    """The overall size of a scale: itself, or the geometric mean of three."""
    return float(np.exp(np.mean(np.log(np.asarray(scale, np.float64)))))


def _describe(scale):
    """A scale for a log line."""
    if np.ndim(scale) == 0:
        return f"{float(scale):.3f}"
    return f"{_overall(scale):.3f} (along the mesh's box axes {' x '.join(f'{v:.3f}' for v in scale)})"


def _stretch_allowed(scale):
    """Whether three per-axis scales keep to MODEL_SCALE_RANGE and to within
    MODEL_STRETCH_MAX of their overall size."""
    f = np.asarray(scale, np.float64)
    lo, hi = MODEL_SCALE_RANGE
    ratio = f / _overall(f)
    return bool(np.all((f >= lo) & (f <= hi))
                and np.all((ratio >= 1 / MODEL_STRETCH_MAX) & (ratio <= MODEL_STRETCH_MAX)))


def _within_stretch(scale):
    """Three per-axis scales brought within MODEL_SCALE_RANGE and within
    MODEL_STRETCH_MAX of their overall size, their order kept; None if that
    does not settle."""
    f = np.asarray(scale, np.float64)
    lo, hi = MODEL_SCALE_RANGE
    limit = MODEL_STRETCH_MAX * (1 - 1e-9)
    for _ in range(30):
        f = np.clip(f, lo, hi)
        if _stretch_allowed(f):
            return f
        overall = _overall(f)
        f = overall * np.clip(f / overall, 1 / limit, limit)
    return None


def _signed_permutations():
    """The 24 rotations that map the coordinate axes onto themselves."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = np.zeros((3, 3))
            P[list(perm), range(3)] = signs
            if np.linalg.det(P) > 0:
                out.append(P)
    return out


def connected_depth_foreground(mask, depth, cam=None, through=None):
    """Remove small mask islands separated by a depth jump from the main part.

    A 2D connected mask can cross an occluder's boundary. Use adjacent valid
    pixels whose depth differs by at most 3 cm as graph edges. If no component
    dominates, retain the evidence for the pose checks instead of choosing one.
    With `cam` (every pixel's camera-frame point), an island the main part
    continues into behind an occluder is kept (MODEL_BEHIND_SHARE).

    `through`, other objects' pixels cut from the mask, joins what it links
    as the mask's own pixels would: RR-S1-2's bar, cut in two by the ring
    round it, had its far piece dropped -- the ring hugs the bar, no
    MODEL_HIDDEN_M nearer -- and was fitted 213 mm long of 330.
    """
    region = mask if through is None else mask | through
    count = int(region.sum())
    indices = np.full(mask.shape, -1, dtype=np.int32)
    indices[region] = np.arange(count)
    sources, targets = [], []
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        linked = region[a] & region[b] & (np.abs(depth[a] - depth[b]) <= MODEL_DEPTH_GAP_M)
        sources.append(indices[a][linked])
        targets.append(indices[b][linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    own = mask[region]                            # the order of `indices`
    sizes = np.bincount(labels[own], minlength=labels.max() + 1 if count else 0)
    if not len(sizes) or sizes.max() < max(MIN_BOX_POINTS, MODEL_COMPONENT_MIN_SHARE * int(mask.sum())):
        return mask
    main = sizes.argmax()
    keep = labels == main
    if cam is not None:
        ys, xs = np.nonzero(region)
        main_px = np.column_stack([xs[keep & own], ys[keep & own]])
        tree = cKDTree(main_px)
        for label in np.flatnonzero(sizes >= MODEL_BEHIND_MIN_PX):
            if label != main and _behind_occluder(np.column_stack([xs, ys])[(labels == label) & own],
                                                  main_px, tree, depth, cam):
                keep |= labels == label
    result = np.zeros_like(mask)
    result[region] = keep
    return result & mask


def _behind_occluder(piece, main_px, tree, depth, cam):
    """Whether a mask piece (pixels, x y) is the main part's continuation
    behind something nearer; see MODEL_BEHIND_SHARE."""
    dist, nearest = tree.query(piece)
    i = int(np.argmin(dist))
    if dist[i] <= 1.5:
        return False                             # touching: a depth jump between them
    a, b = piece[i], main_px[nearest[i]]
    n = int(np.ceil(dist[i]))
    share = np.arange(1, n) / n
    between = np.round(a[None] + (b - a)[None] * share[:, None]).astype(np.int64)
    da, db = depth[a[1], a[0]], depth[b[1], b[0]]
    seen = depth[between[:, 1], between[:, 0]]
    if not len(seen) or np.mean((seen > 0) & (seen < da + (db - da) * share - MODEL_HIDDEN_M)) < MODEL_BEHIND_SHARE:
        return False
    gap = cam[a[1], a[0]] - cam[b[1], b[0]]
    return abs(gap[2]) <= MODEL_BEHIND_DEPTH_RATIO * np.hypot(gap[0], gap[1]) + MODEL_DEPTH_GAP_M


class _ModelFit:
    """Fits one mesh to one object's view; see fit_model_box()."""

    def __init__(self, model, view, K, rng, through=None, scale_hint=None, located=None, image=None,
                 known_shape=None):
        cam, mask, keep, plane, _ = view
        self.lab = (None if image is None or model.lab is None else
                    cv2.cvtColor(np.asarray(image, np.float32) / 255.0, cv2.COLOR_RGB2LAB).astype(np.float64))
        self.scale_hint = scale_hint
        # The part's shape measured on earlier views (MODEL_KNOWN_SHAPE_MIN_GAIN).
        self.known_shape = None if known_shape is None else np.asarray(known_shape, np.float64).reshape(3)
        # The scale, along the mesh's box axes, it was located at this init
        # (MODEL_LOCATED_FALLBACK).
        self.located = located
        self.model, self.K, self.cam = model, K, cam
        self.depth = cam[..., 2]
        kernel = np.ones((2 * MODEL_MASK_GROW_PX + 1,) * 2, np.uint8)
        self.grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        self.mask = mask
        self.plane = None
        if plane is not None:
            n, p = plane
            # A plane the object is behind is an occluder the ring caught, not
            # its support; it would push the mesh out of the right pose.
            if np.mean((cam[mask] - p) @ n <= MODEL_TOL_M) >= PLANE_SUPPORT_SHARE:
                self.plane = (n, p)
        # The depth's noise here, and the tolerances it sets (MODEL_NOISE_REF_M).
        sigma = None if self.plane is None else depth_noise(mask, cam, self.plane)
        self.sigma = sigma
        self.noise = 1.0 if sigma is None else max(1.0, sigma / MODEL_NOISE_REF_M)
        self.tol, self.hidden_m = MODEL_TOL_M * self.noise, MODEL_HIDDEN_M * self.noise
        self.cover_noise = MODEL_COVER_NOISE_M * self.noise
        # A ring's hole and mask spill can include a large patch of table.
        # For a thick enough mesh, use the foreground already isolated by
        # observe(). For very thin parts, that cut selects positive depth
        # noise and biases the fit upwards, so retain the full mask there.
        use_foreground = (self.plane is not None
                          and model.extent.min() > 2 * PLANE_TOL_M)
        obs_mask = keep if use_foreground else mask
        if use_foreground:
            obs_mask = connected_depth_foreground(obs_mask, self.depth, cam, through)
        obs = cam[obs_mask]
        self.observation_note = f"{len(obs)}/{int(mask.sum())} depth points"
        if sigma is not None:
            self.observation_note += (f", depth noise {sigma * 1000:.1f} mm"
                                      + (f" (tolerances x{self.noise:.2f})" if self.noise > 1 else ""))
        if use_foreground and np.any(mask & ~keep):
            self.observation_note += ", support removed"
        if np.any(keep & ~obs_mask):
            self.observation_note += ", isolated depth components removed"
        self.obs = obs[rng.permutation(len(obs))[:MODEL_FINE_POINTS]]
        self.obs_tree = cKDTree(self.obs)
        self._obb_points = cam[obs_mask if use_foreground else keep]
        self._obb = None

    @property
    def obb(self):
        """The observed points' minimal box, (R, centre, extent): where the
        pose search starts (candidates(), start_scales()); scoring a given
        pose (carried_fit_rejection()) needs none, and it costs most of the
        set-up."""
        if self._obb is None:
            self._obb = min_obb(self._obb_points)
        return self._obb

    def linear(self, scale):
        """The mesh-frame linear map of a scale: one number scales the mesh
        uniformly, three scale it along its own box axes (see
        MODEL_STRETCH_MAX)."""
        f = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        B = self.model.box[:3, :3]
        return (B * f) @ B.T

    def cover_seen(self, R, t, terms, scale, note, per_axis=False):
        """The pose, its scale raised, uniformly, until the mesh spans the
        observed points along each of its axes; see MODEL_COVER_NOISE_M.

        With `per_axis`, where the overall size came from elsewhere (the
        other parts' scale), only the axes short of what was seen are raised,
        within MODEL_STRETCH_MAX: raised whole, PP-S4-1's tester, 2.415 by
        the sample and its mesh narrower than the part, came out 214 mm long
        of 182 to be 62 mm wide."""
        extent = np.broadcast_to(np.asarray(scale, np.float64), (3,)) * self.model.extent
        along = self.obs @ (R @ self.model.box[:3, :3])
        span = np.percentile(along, 98, axis=0) - np.percentile(along, 2, axis=0) - self.cover_noise
        factor = float(np.max(span / extent))
        if factor <= MODEL_SCALE_FINE_RATIO:
            return R, t, terms, scale, note
        raised = np.asarray(scale, np.float64) * factor
        raised = float(raised) if np.ndim(raised) == 0 else raised
        if per_axis:
            axes = _within_stretch(np.broadcast_to(np.asarray(scale, np.float64), (3,))
                                   * np.maximum(span / extent, 1.0))
            if axes is not None:
                raised = axes
        if _overall(raised) > MODEL_SCALE_RANGE[1]:
            return R, t, terms, scale, note
        centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        t2 = centre - R @ (self.linear(raised) @ self.model.box[:3, 3])
        moved = self.terms(raised, R, t2, MODEL_SAMPLES, MODEL_FINE_POINTS)
        if moved is None:
            return R, t, terms, scale, note
        R2, t2, moved, _ = self.polish(raised, R, t2, moved)
        for key in ("measured", "scale_note"):
            if key in terms:
                moved[key] = terms[key]
        if per_axis and np.ndim(raised):
            return R2, t2, moved, raised, note + f"; raised to {_describe(raised)} to cover what was seen"
        return R2, t2, moved, raised, note + f"; raised {factor:.2f} times to cover what was seen"

    def known_shape_fit(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality): the pose, its ends and its turn
        about its length settled, at the shape measured on earlier views
        (known_shape), fitted about this one's centre, where it costs no more
        than MODEL_KNOWN_SHAPE_MIN_GAIN over this one and fits as well -- then
        measured, by those views; else this one."""
        known = self.known_shape
        if np.allclose(np.broadcast_to(np.asarray(scale, np.float64), (3,)), known,
                       rtol=MODEL_SCALE_FINE_RATIO - 1):
            return terms, R, t, scale, quality
        mean = self.model.points.mean(0)
        centre = R @ (self.linear(scale) @ mean) + t
        Rk, tk, k_terms = self.icp(known, R, centre - R @ (self.linear(known) @ mean), MODEL_FINE_ROUNDS,
                                   MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
        if k_terms is None:
            return terms, R, t, scale, quality
        Rk, tk, k_terms, _ = self.polish(known, Rk, tk, k_terms)
        k_quality = ("fit" if not model_pose_rejection(k_terms)
                     else "coarse" if not model_pose_rejection(k_terms, coarse=True) else None)
        costs = f"{k_terms['cost']:.3f} against {terms['cost']:.3f} at {_describe(scale)}"
        rank = {"fit": 2, "coarse": 1, None: 0}
        if k_terms["cost"] > terms["cost"] + MODEL_KNOWN_SHAPE_MIN_GAIN or rank[k_quality] < rank[quality]:
            terms["scale_note"] = terms.get("scale_note", "") + (f"; the size measured before, "
                                                                 f"{_describe(known)}, fits worse: {costs}")
            terms["known_shape"] = "beaten"
            return terms, R, t, scale, quality
        k_terms["measured"], k_terms["known_shape"] = True, "kept"
        k_terms["scale_note"] = (terms.get("scale_note", "")
                                 + f"; kept the size measured before, {_describe(known)}: {costs}")
        return k_terms, Rk, tk, known, k_quality

    def axes_seen(self, R, t, scale, ends=True):
        """Per mesh box axis, whether the view shows the part's length along
        it: the observed points span MODEL_STRETCH_SEEN of the mesh's extent
        there, and (`ends`) both ends show (ends_seen()). A mesh stretched
        past what was seen spans less of it, and fails."""
        scale = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        along = self.obs @ (R @ self.model.box[:3, :3])
        span = (np.percentile(along, 100 - MODEL_SEEN_PERCENTILE, axis=0)
                - np.percentile(along, MODEL_SEEN_PERCENTILE, axis=0))
        seen = span >= MODEL_STRETCH_SEEN * scale * self.model.extent
        return seen & self.ends_seen(R, t, scale) if ends else seen

    def ends_seen(self, R, t, scale):
        """Per mesh box axis, whether the view shows both of the mesh's ends
        along it; see MODEL_END_HIDDEN."""
        return ~self.hidden_ends(R, t, scale).any(axis=1)

    def hidden_ends(self, R, t, scale):
        """Per mesh box axis, whether its end at -axis and at +axis is hidden
        (3 x 2); see MODEL_END_HIDDEN."""
        M = self.linear(scale)
        P = self.model.points @ (R @ M).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        hidden = seen.copy()                      # off the image unless shown otherwise
        idx, u, v, d = self.lines_of_sight(P, seen)
        blocked = (d > 0) & (d < P[idx, 2] - self.hidden_m)
        if self.plane is not None:
            # Beside the mask at the mesh's depth, off the support: an occluder
            # touching it, as terms() counts it.
            n, p0 = self.plane
            blocked |= (~self.grown[v, u] & (d > 0) & (np.abs(d - P[idx, 2]) <= self.tol)
                        & ((self.cam[v, u] - p0) @ n < -PLANE_TOL_M))
        hidden[idx] = blocked
        centre = R @ (M @ self.model.box[:3, 3]) + t
        out = np.zeros((3, 2), bool)
        for k in range(3):
            along = (P - centre) @ (R @ self.model.box[:3, k])
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for e, end in enumerate((along <= lo, along >= hi)):
                front = end & seen
                out[k, e] = front.sum() >= 10 and hidden[front].mean() > MODEL_END_HIDDEN
        return out

    def ends_open(self, R, t, scale):
        """Per mesh box axis, whether just past both of the mesh's ends the
        view shows the part stopping there; see MODEL_END_PROBE_M."""
        M = self.linear(scale)
        P = self.model.points @ (R @ M).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        h, w = self.depth.shape
        out = np.ones(3, bool)
        for k in range(3):
            axis = R @ self.model.box[:3, k]
            along = (P - centre) @ axis
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for end, outward in ((along <= lo, -axis), (along >= hi, axis)):
                probe = P[end & seen] + MODEL_END_PROBE_M * outward
                if self.plane is not None:
                    n, p0 = self.plane
                    probe = probe[(probe - p0) @ n <= PLANE_TOL_M]      # not into the support
                if len(probe) < 10:
                    continue
                uv = np.round(_pixels(probe, self.K)).astype(np.int64)
                inside = ((probe[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                          & (uv[:, 1] >= 0) & (uv[:, 1] < h))
                if inside.sum() < 10:
                    out[k] = False                  # nothing past this end in view
                    continue
                u, v = uv[inside, 0], uv[inside, 1]
                d = self.depth[v, u]
                # Not the part itself in front (its own mask), but anything
                # else, however near its edge: seen nearly end-on, the ring
                # hiding the bar's middle was a few pixels across.
                nearer = (d > 0) & (d < probe[inside, 2] - self.hidden_m) & ~self.mask[v, u]
                if self.plane is not None:
                    n, p0 = self.plane
                    nearer &= (self.cam[v, u] - p0) @ n < -PLANE_TOL_M
                if nearer.mean() > MODEL_END_HIDDEN:
                    out[k] = False
        return out

    def ends_stop(self, R, t, scale):
        """Per mesh box axis, per end (-axis, +axis), whether just past it
        the view shows the part stopping there (as ends_open(), end by end;
        False where nothing past it is in view), or the support does: past
        an end resting on it is the support."""
        M = self.linear(scale)
        P = self.model.points @ (R @ M).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        h, w = self.depth.shape
        out = np.zeros((3, 2), bool)
        for k in range(3):
            axis = R @ self.model.box[:3, k]
            along = (P - centre) @ axis
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for e, (end, outward) in enumerate(((along <= lo, -axis), (along >= hi, axis))):
                probe = P[end & seen] + MODEL_END_PROBE_M * outward
                if self.plane is not None:
                    n, p0 = self.plane
                    under = (probe - p0) @ n > PLANE_TOL_M
                    if under.sum() >= 10 and (~under).sum() < 10:
                        out[k, e] = True              # it ends on the support
                        continue
                    probe = probe[~under]
                uv = np.round(_pixels(probe, self.K)).astype(np.int64)
                inside = ((probe[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
                if inside.sum() < 10:
                    continue
                u, v = uv[inside, 0], uv[inside, 1]
                d = self.depth[v, u]
                nearer = (d > 0) & (d < probe[inside, 2] - self.hidden_m) & ~self.mask[v, u]
                if self.plane is not None:
                    n, p0 = self.plane
                    nearer &= (self.cam[v, u] - p0) @ n < -PLANE_TOL_M
                out[k, e] = nearer.mean() <= MODEL_END_HIDDEN
        return out

    def axial(self, terms, R, t, scale, quality):
        """(terms, R, t, quality, report): the pose slid along the mesh's
        long box axis onto its one end the view shows where it is off it by
        more than the tolerance, and the report of model_axial; see the
        comment after MODEL_SUPPORT_TOP_SIGMAS."""
        scale3 = np.array(np.broadcast_to(np.asarray(scale, np.float64), (3,)))
        k = int(np.argmax(scale3 * self.model.extent))
        M = self.linear(scale3)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        axis = R @ self.model.box[:3, k]
        hidden = self.hidden_ends(R, t, scale3)[k]
        stops = self.ends_stop(R, t, scale3)[k]
        seen = [bool(not hidden[e] and stops[e]) for e in range(2)]
        along = (self.model.points @ (R @ M).T + t - centre) @ axis
        obs = (self.obs - centre) @ axis
        mesh_ends = [float(along.min()), float(along.max())]
        # The tenth point from either end: as few as tell where an end is
        # (hidden_ends()); a percentile is inside a small end face.
        ordered = np.sort(obs)
        few = min(10, len(ordered)) - 1
        obs_ends = [float(ordered[few]), float(ordered[-1 - few])]
        slide = 0.0
        if seen[0] != seen[1]:
            e = 0 if seen[0] else 1
            if abs(obs_ends[e] - mesh_ends[e]) > self.tol:
                slide = obs_ends[e] - mesh_ends[e]
        if slide:
            t2 = t + slide * axis
            terms2 = self.terms(scale, R, t2, MODEL_SAMPLES, MODEL_FINE_POINTS)
            quality2 = (None if terms2 is None else "fit" if not model_pose_rejection(terms2)
                        else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
            rank = {"fit": 2, "coarse": 1, None: 0}
            if terms2 is not None and rank[quality2] >= rank[quality]:
                for key in ("measured", "known_shape"):
                    if key in terms:
                        terms2[key] = terms[key]
                terms2["scale_note"] = (terms.get("scale_note", "")
                                        + f"; slid {slide * 1000:+.1f} mm along its length onto the end seen")
                terms, t, quality = terms2, t2, quality2
                mesh_ends = [m + slide for m in mesh_ends]
            else:
                slide = 0.0
        # In the mesh's frame, along its box axis k from the box's centre, at the fit's scale.
        report = {"axis": self.model.box[:3, k].tolist(),
                  "ends": [{"seen": seen[e], "mesh_m": round(mesh_ends[e], 4),
                            "seen_to_m": round(obs_ends[e], 4)} for e in range(2)],
                  "slid_m": round(slide, 4)}
        if not any(seen):
            # Slid either way as far as it still spans what was seen.
            report["slide_range_m"] = [round(min(0.0, obs_ends[1] - mesh_ends[1]), 4),
                                       round(max(0.0, obs_ends[0] - mesh_ends[0]), 4)]
        return terms, R, t, quality, report

    def lines_of_sight(self, P, seen):
        """The seen mesh points' indices and pixels, and the observed depth
        there (0 where there is none)."""
        h, w = self.depth.shape
        uv = np.round(_pixels(P, self.K)).astype(np.int64)
        idx = np.flatnonzero(seen & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                             & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        u, v = uv[idx, 0], uv[idx, 1]
        return idx, u, v, self.depth[v, u]

    def colour_contradiction(self, scale, R, t):
        """Of the pixels the camera sees the mesh's samples on the part at
        at this pose, the share whose colour is none of the mesh's
        (LOCATE_COLOUR_DELTA); 0 without colours on both."""
        if self.lab is None:
            return 0.0
        P = self.model.points @ (R @ self.linear(scale)).T + t
        seen = _seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)
        idx, u, v, d = self.lines_of_sight(P, seen)
        on = (d > 0) & (np.abs(d - P[idx, 2]) <= self.tol) & self.mask[v, u]
        if on.sum() < LOCATE_COLOUR_MIN_POINTS:
            return 0.0
        off = self.model.chroma_tree.query(self.lab[v[on], u[on], 1:])[0]
        return float(np.mean(off > LOCATE_COLOUR_DELTA))

    def place(self, scale, R, mesh_n):
        """Translation putting the mesh's unhidden side on the observed points.

        The side the camera would see, less what is hidden behind something:
        put a part's hidden end into the average and the mesh lands short of
        where the seen end is.
        """
        X = self.model.points[:mesh_n] @ self.linear(scale).T
        spacing = _overall(scale) * self.model.spacing * np.sqrt(MODEL_SAMPLES / mesh_n)
        target = self.obs.mean(0)
        t = target - R @ X.mean(0)
        for _ in range(3):
            P = X @ R.T + t
            idx, _, _, d = self.lines_of_sight(
                P, _seen_from_camera(P, self.K, spacing, self.depth.shape))
            idx = idx[(d <= 0) | (d >= P[idx, 2] - self.hidden_m)]
            if len(idx):
                t = t + target - P[idx].mean(0)
        return t

    def terms(self, scale, R, t, mesh_n, obs_n):
        """Correspondences of one pose and its cost; see the comment above
        MODEL_SAMPLES for what is counted."""
        M = self.linear(scale)
        P = self.model.points[:mesh_n] @ (R @ M).T + t
        # Normals go by the inverse transpose; only their line is used.
        N = self.model.normals[:mesh_n] @ np.linalg.inv(M)
        N = (N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-15)) @ R.T
        spacing = _overall(scale) * self.model.spacing * np.sqrt(MODEL_SAMPLES / mesh_n)
        seen = _seen_from_camera(P, self.K, spacing, self.depth.shape)
        obs = self.obs[:obs_n]
        out = {"pairs": [], "P": P}

        # Observed points onto the mesh's visible surface.
        if seen.sum() < 3:
            return None
        seen_idx = np.flatnonzero(seen)
        dist_a, near = cKDTree(P[seen_idx]).query(obs)
        near = seen_idx[near]
        out["pairs"].append((P[near], obs, N[near], dist_a))

        # Visible mesh points against what the camera sees there.
        idx, u, v, d = self.lines_of_sight(P, seen)
        in_mask = self.grown[v, u]
        open_ = (d > 0) & (d >= P[idx, 2] - self.hidden_m)
        # Outside the mask, off the support and no further than the mesh
        # point: hidden too.
        unmasked = np.zeros(len(idx), bool)
        if self.plane is not None:
            n, p0 = self.plane
            unmasked = (open_ & ~in_mask & (d <= P[idx, 2] + self.tol)
                        & ((self.cam[v, u] - p0) @ n < -PLANE_TOL_M))
        with_depth = max(1, int(np.sum(d > 0)))
        out["hidden"] = float(np.sum(((d > 0) & ~open_) | unmasked)) / with_depth
        out["unmasked"] = float(np.sum(unmasked)) / with_depth
        open_ &= ~unmasked
        idx, u, v, d, in_mask = idx[open_], u[open_], v[open_], d[open_], in_mask[open_]
        # In the mask: paired with the surface seen along the same ray. Outside
        # it: the camera sees past the mesh point, so pull it to the object.
        target = self.cam[v, u]
        if (~in_mask).any():
            target[~in_mask] = self.obs[self.obs_tree.query(P[idx[~in_mask]])[1]]
        dist_b = np.linalg.norm(P[idx] - target, axis=1)
        bad_b = ~in_mask | (np.abs(d - P[idx, 2]) > self.tol)
        out["pairs"].append((P[idx], target, N[idx], dist_b))

        # Mesh points under the support surface.
        below = np.zeros(len(P))
        if self.plane is not None:
            n, p0 = self.plane
            below = np.maximum((P - p0) @ n, 0.0)
        out["below"] = below

        sat = lambda dist: np.minimum(dist / self.tol, 1.0) ** 2
        out["cost"] = (sat(dist_a).mean()
                       + (np.where(bad_b, 1.0, sat(dist_b)).mean() if len(idx) else 0.0)
                       + sat(below).mean())
        out["fitness"] = float(np.mean(dist_a < self.tol))
        out["contradiction"] = float(bad_b.mean()) if len(idx) else 0.0
        out["penetration"] = float(np.mean(below > self.tol))
        out["open_points"] = len(idx)
        out["p95_distance"] = float(np.percentile(dist_a, 95))
        out["width"] = float(np.sort(np.broadcast_to(scale, (3,)) * self.model.extent)[1])
        out["noise"] = self.noise
        return out

    def step(self, R, t, terms, radius, scale=None, free=(), bounds=None):
        """One Gauss-Newton step of the pose, about the mesh's centre; and of
        the log of `scale` along the box axes `free` (indices), about the
        mesh's box centre, within `bounds` (two triples of scales).
        Returns R, t and the scale."""
        P = terms["P"]
        c = P.mean(0)
        free = list(free)
        n_par = 6 + len(free)
        if free:
            scale = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
            axes = (R @ self.model.box[:3, :3])[:, free]
            box_c = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        H, g = np.zeros((n_par, n_par)), np.zeros(n_par)
        lam = MODEL_POINT_WEIGHT
        for p, q, n, dist in terms["pairs"]:
            use = dist < radius
            if not use.any():
                continue
            p, q, n = p[use], q[use], n[use]
            d, r = p - c, p - q
            J = np.zeros((len(p), 3, n_par))
            J[:, 0, 1], J[:, 0, 2], J[:, 1, 0] = d[:, 2], -d[:, 1], -d[:, 2]
            J[:, 1, 2], J[:, 2, 0], J[:, 2, 1] = d[:, 0], d[:, 1], -d[:, 0]
            J[:, :, 3:6] = np.eye(3)
            if free:
                # Stretched along a box axis, a point moves along it by its
                # distance from the box centre along it.
                J[:, :, 6:] = axes[None] * ((p - box_c) @ axes)[:, None, :]
            nJ = np.einsum("ki,kij->kj", n, J)
            k = 1.0 / len(p)
            H += k * (lam * np.einsum("kij,kil->jl", J, J) + (1 - lam) * nJ.T @ nJ)
            g += k * (lam * np.einsum("kij,ki->j", J, r)
                      + (1 - lam) * nJ.T @ np.einsum("ki,ki->k", n, r))
        below = terms["below"] > 0
        if below.any():
            n = self.plane[0]
            a = np.hstack([np.cross(P[below] - c, n), np.tile(n, (below.sum(), 1))]
                          + ([((P[below] - box_c) @ axes) * (n @ axes)] if free else []))
            k = 1.0 / len(P)
            H += k * a.T @ a
            g += k * a.T @ terms["below"][below]
        if not np.any(H):
            return R, t, scale
        xi = -np.linalg.solve(H + (1e-6 * np.trace(H) / n_par + 1e-12) * np.eye(n_par), g)
        E = _rotation(xi[:3])
        R, t = E @ R, E @ (t - c) + c + xi[3:6]
        if free:
            stretched = scale.copy()
            stretched[free] *= np.exp(np.clip(xi[6:], -np.log(MODEL_STRETCH_STEP), np.log(MODEL_STRETCH_STEP)))
            if bounds is not None:
                stretched = np.clip(stretched, *bounds)
            stretched = _within_stretch(stretched)
            if stretched is not None:
                centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
                scale = stretched
                t = centre - R @ (self.linear(scale) @ self.model.box[:3, 3])
        return R, t, scale

    def icp(self, scale, R, t, rounds, mesh_n, obs_n, radius):
        for i in range(rounds):
            terms = self.terms(scale, R, t, mesh_n, obs_n)
            if terms is None:
                return R, t, None
            R, t, _ = self.step(R, t, terms, max(self.tol, radius * 0.75 ** i))
        return R, t, self.terms(scale, R, t, mesh_n, obs_n)

    def stretch(self, scale, R, t, free, rounds=MODEL_STRETCH_ROUNDS, radius=MODEL_SEARCH_M):
        """ICP over the pose and, with it, the scale along the box axes
        `free` (three booleans), which only grows, by MODEL_STRETCH_MAX at
        most: R, t, terms and the three scales. Its pairs are ICP's: a mesh
        too short for what was seen is drawn out by the points past its ends,
        as far as the camera seeing past it lets it."""
        scale = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
        bounds = scale.copy(), scale * MODEL_STRETCH_MAX
        free = np.flatnonzero(free)
        for i in range(rounds):
            terms = self.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
            if terms is None:
                return R, t, None, scale
            R, t, scale = self.step(R, t, terms, max(MODEL_STRETCH_RADIUS_M, radius * 0.85 ** i), scale, free,
                                    bounds)
        return R, t, self.terms(scale, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS), scale

    def polish(self, scale, R, t, terms, free_scale=False, free_axes=False):
        """Descend the cost itself, a step along or about each of the mesh's
        box axes at a time -- and, if `free_scale`, of the mesh's scale, or if
        `free_axes`, of its scale along each box axis. Returns R, t, terms and
        the scale (three numbers once `free_axes`).

        ICP's squared residuals are not the cost: along a direction little
        holds -- a thin part whose near end is hidden slides along its length
        -- the few outliers it drops can carry it millimetres off the pose
        the cost prefers.
        """
        seen = np.ones(3, bool)
        if free_axes:
            scale = np.broadcast_to(np.asarray(scale, np.float64), (3,)).copy()
            seen = self.axes_seen(R, t, scale)
        axes = R @ self.model.box[:3, :3]
        centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        lo, hi = MODEL_SCALE_RANGE
        for step_m, step_rad, step_s in zip(MODEL_POLISH_STEPS_M, np.radians(MODEL_POLISH_STEPS_DEG),
                                            MODEL_POLISH_STEPS_SCALE):
            for _ in range(MODEL_POLISH_ROUNDS):
                best = None
                moves = [(axis, sign, turn) for axis, sign, turn
                         in itertools.product(range(3), (1.0, -1.0), (False, True))]
                if free_scale:
                    moves += [(None, 1.0, False), (None, -1.0, False)]
                if free_axes:
                    moves += [(("stretch", k), sign, False) for k in range(3) if seen[k]
                              for sign in (1.0, -1.0)]
                for axis, sign, turn in moves:
                    sm = scale
                    if axis is None or isinstance(axis, tuple):
                        if axis is None:
                            sm = np.clip(scale * (1 + sign * step_s), lo, hi)
                        else:
                            sm = scale.copy()
                            sm[axis[1]] *= 1 + 2 * sign * step_s
                            if not _stretch_allowed(sm):
                                continue
                        sm = float(sm) if np.ndim(sm) == 0 else sm
                        Rm, tm = R, centre - R @ (self.linear(sm) @ self.model.box[:3, 3])
                    elif turn:
                        E = _rotation(sign * step_rad * axes[:, axis])
                        Rm, tm = E @ R, E @ (t - centre) + centre
                    else:
                        Rm, tm = R, t + sign * step_m * axes[:, axis]
                    moved = self.terms(sm, Rm, tm, MODEL_SAMPLES, MODEL_FINE_POINTS)
                    gain = (best or terms)["cost"] - (moved["cost"] if moved else np.inf)
                    if gain > (0.0 if best else MODEL_POLISH_MIN_GAIN):
                        best, best_pose = moved, (Rm, tm, sm)
                if best is None:
                    break
                terms, (R, t, scale) = best, best_pose
                axes = R @ self.model.box[:3, :3]
                centre = R @ (self.linear(scale) @ self.model.box[:3, 3]) + t
        return R, t, terms, scale

    def candidates(self, on_support=True):
        """Starting rotations: the mesh's box axes laid on the observed box's
        axes every way round, and -- the observed box tilts when part of the
        object is hidden -- with `on_support`, each box axis standing on the
        support surface at a range of headings."""
        box = self.model.box[:3, :3]
        R_obs, _, extent_obs = self.obb
        out = [R_obs @ P @ box.T for P in _signed_permutations()]
        if on_support and self.plane is not None:
            up = -self.plane[0]
            major = R_obs[:, int(np.argmax(extent_obs))]
            x0 = major - (major @ up) * up
            if np.linalg.norm(x0) < 1e-6:
                x0 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
            x0 /= np.linalg.norm(x0)
            y0 = np.cross(up, x0)
            for k in range(3):
                for sign in (1.0, -1.0):
                    a = sign * box[:, k]
                    b1 = box[:, (k + 1) % 3]
                    B_model = np.column_stack([b1, np.cross(a, b1), a])
                    for yaw in np.radians(np.arange(0, 360, MODEL_YAW_STEP_DEG)):
                        e1 = np.cos(yaw) * x0 + np.sin(yaw) * y0
                        B_cam = np.column_stack([e1, np.cross(up, e1), up])
                        out.append(B_cam @ B_model.T)
        return out

    def start_scales(self):
        """Scales to search poses at: 1; the observed size's when the mesh is
        plainly another size, comparing the boxes' middle sides -- the
        smallest side seen is a one-sided view's depth (half a cylinder), the
        largest is cut short by whatever hides or crops the part; and, when
        the observed box's proportions plainly differ from the mesh's, those
        proportions, three numbers along the mesh's box axes matched side for
        side by length. A one-sided view can show the wrong proportions -- a
        bar seen from one side looks flat -- so this is only a start: the
        pose search keeps whichever fits best. Proportions past what may be
        stretched are brought within it, not dropped: the simulated bar seen
        from one side, 334 by 26 by 16 mm, looked flatter than any stretch
        allows, and the starts left all fitted it too short. A known scale
        is the only start."""
        if self.model.known_scale is not None:
            return [self.model.known_scale]
        observed, mesh = np.sort(self.obb[2]), np.sort(self.model.extent)
        s = float(np.clip(observed[1] / mesh[1], *MODEL_SCALE_RANGE))
        out = [1.0]
        if abs(np.log(s)) > np.log(MODEL_SCALE_START_RATIO):
            out.append(s)
        ratio = observed / mesh
        relative = ratio / _overall(ratio)
        if np.max(np.abs(np.log(relative))) > np.log(MODEL_STRETCH_START_RATIO):
            # Its proportions differ, so no one side gives its size: all three do.
            f = np.empty(3)
            f[np.argsort(self.model.extent)] = (np.clip(_overall(ratio), *MODEL_SCALE_RANGE)
                                                * np.clip(relative, 1 / MODEL_STRETCH_MAX, MODEL_STRETCH_MAX))
            f = _within_stretch(f)
            if f is not None:
                out.append(f)
        return out

    def run(self):
        mesh_n, obs_n, rounds = MODEL_COARSE
        coarse = []
        for which, s in enumerate(self.start_scales()):
            for R in self.candidates():
                t = self.place(s, R, mesh_n)
                R, t, terms = self.icp(s, R, t, rounds, mesh_n, obs_n, MODEL_SEARCH_M)
                if terms is not None:
                    coarse.append((terms["cost"], which, R, t, s))
        if not coarse:
            return None
        coarse.sort(key=lambda item: item[0])
        kept = coarse[:MODEL_KEEP]
        for which in {item[1] for item in coarse}:
            kept += [item for item in coarse if item[1] == which][:MODEL_KEEP_PER_START]
        kept = sorted({id(item): item for item in kept}.values(), key=lambda item: item[0])
        refined = []
        for _, _, R, t, s in kept:
            R, t, terms = self.icp(s, R, t, MODEL_FINE_ROUNDS, MODEL_SAMPLES,
                                   MODEL_FINE_POINTS, 2 * self.tol)
            if terms is not None:
                refined.append((terms, R, t, s))
        if not refined:
            return None
        results, done = [], []
        # A minimum combined cost can still violate one of the acceptance
        # gates. Try the other refined rotations before declaring no pose.
        for terms, R, t, s in sorted(refined, key=lambda item: item[0]["cost"]):
            # Starts that refined to the same pose need measuring once.
            if any(np.allclose(np.broadcast_to(s, (3,)), np.broadcast_to(s2, (3,)), rtol=0.02)
                   and np.linalg.norm(t - t2) < self.tol
                   and np.degrees(np.arccos(np.clip((np.trace(R.T @ R2) - 1) / 2, -1, 1))) < 3
                   for R2, t2, s2 in done):
                continue
            done.append((R, t, s))
            start = s
            if np.ndim(s):
                # A start's proportions found the pose. They stand only along
                # the axes the view spans, both ends seen; along the others
                # -- a bar's depth, seen from one side; a length one end of
                # which is hidden, seen short -- the part takes the overall
                # size of the rest, about the same centre. (Not the ends'
                # test: a ring's diameter with an end behind its bar kept,
                # it came out 51 mm of 60.)
                seen = self.axes_seen(R, t, s)
                if not seen.all():
                    centre = R @ (self.linear(s) @ self.model.box[:3, 3]) + t
                    s = np.where(seen, s, _overall(s[seen]) if seen.any() else _overall(s))
                    s = _overall(s) if np.allclose(s, s[0]) else s
                    t = centre - R @ (self.linear(s) @ self.model.box[:3, 3])
                    terms = self.terms(s, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
                    if terms is None:
                        continue
            R, t, terms, _ = self.polish(s, R, t, terms)
            scale, note = s, f"not measured, {terms['hidden']:.0%} hidden"
            terms["measured"] = False
            known = self.model.known_scale is not None
            hinted = False
            if (not known and terms["hidden"] > MODEL_SCALE_MAX_HIDDEN and self.scale_hint is not None
                    and abs(np.log(_overall(s) / self.scale_hint)) > np.log(MODEL_SCALE_FINE_RATIO)):
                shape = s * (self.scale_hint / _overall(s)) if np.ndim(s) else self.scale_hint
                mean = self.model.points.mean(0)
                centre = R @ (self.linear(s) @ mean) + t
                Rh, th, h_terms = self.icp(shape, R, centre - R @ (self.linear(shape) @ mean), MODEL_FINE_ROUNDS,
                                           MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
                if h_terms is not None:
                    Rh, th, h_terms, _ = self.polish(shape, Rh, th, h_terms)
                    if not model_pose_rejection(h_terms, coarse=True):
                        R, t, terms, scale, s = Rh, th, h_terms, shape, shape
                        terms["measured"] = False
                        note, hinted = f"not measured, {terms['hidden']:.0%} hidden", True
            located = False
            if (MODEL_LOCATED_FALLBACK and not known and not hinted and terms["hidden"] > MODEL_SCALE_MAX_HIDDEN
                    and self.located is not None and model_pose_rejection(terms, coarse=True)):
                # Neither its start nor the others' scale: the proportions it
                # was located at (MODEL_LOCATED_FALLBACK).
                shape = np.asarray(self.located, np.float64)
                shape = float(shape) if np.ndim(shape) == 0 else shape
                mean = self.model.points.mean(0)
                centre = R @ (self.linear(s) @ mean) + t
                Rl, tl, l_terms = self.icp(shape, R, centre - R @ (self.linear(shape) @ mean), MODEL_FINE_ROUNDS,
                                           MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * self.tol)
                if l_terms is not None:
                    Rl, tl, l_terms, _ = self.polish(shape, Rl, tl, l_terms)
                    if not model_pose_rejection(l_terms, coarse=True):
                        R, t, terms, scale, s = Rl, tl, l_terms, shape, shape
                        terms["measured"] = False
                        note, located = f"not measured, {terms['hidden']:.0%} hidden", True
            if known:
                terms["measured"], note = True, "known"
            elif terms["hidden"] <= MODEL_SCALE_MAX_HIDDEN:
                measured, shape, Rs, ts, ts_terms, edge = self.measure_scale(R, t, s)
                if ts_terms is None:
                    note = "could not be measured"
                elif edge:
                    note = f"best {measured:.3f} at the edge of {MODEL_SCALE_RANGE[0]}-{MODEL_SCALE_RANGE[1]}"
                elif np.ndim(s) == 0 and abs(np.log(measured)) <= np.log(MODEL_SCALE_FINE_RATIO):
                    # The mesh's own size, as far as this search can tell:
                    # keep exactly 1, at the re-fit's pose if that is better.
                    t1 = ts + Rs @ ((measured - 1.0) * self.model.points.mean(0))
                    t1_terms = self.terms(1.0, Rs, t1, MODEL_SAMPLES, MODEL_FINE_POINTS)
                    if t1_terms is not None:
                        R1, t1, t1_terms, _ = self.polish(1.0, Rs, t1, t1_terms)
                        if s != 1.0 or t1_terms["cost"] < terms["cost"]:
                            R, t, terms, scale = R1, t1, t1_terms, 1.0
                    note = f"measured {measured:.3f}, kept 1"
                elif ts_terms["cost"] < terms["cost"] - MODEL_SCALE_MIN_GAIN:
                    at_start = terms
                    R, t, terms, scale = self.polish(shape, Rs, ts, ts_terms, free_scale=True)
                    note = (f"measured; at {_describe(s)}: fitness {at_start['fitness']:.2f}, "
                            f"contradiction {at_start['contradiction']:.2f}")
                else:
                    note = f"measured {measured:.3f}, too little gain to apply"
                # Then its proportions, on the same clear enough view: by the
                # descent, and drawn out with the pose (stretch(); its pose
                # alone descended after, so that it only grows).
                # An end hidden can be the mesh's own doing: too short, it
                # ends behind what is in front of the part. RR-S1-2's bar,
                # measured 0.71 of the mesh by its width, ended behind the
                # ring near its far end, the bar going on past it, and was
                # raised whole instead (cover_seen()), 31 mm thick of 27. So
                # a length the view spans is drawn out too, and kept if, so
                # drawn, its ends show.
                tries = [self.polish(scale, R, t, terms, free_axes=True)]
                seen = self.axes_seen(R, t, scale)
                spanned = self.axes_seen(R, t, scale, ends=False)
                bound = np.broadcast_to(np.asarray(scale, np.float64), (3,)) * MODEL_STRETCH_MAX
                for free in ([seen, spanned] if (spanned & ~seen).any() else [seen]):
                    if not free.any():
                        continue
                    Rs2, ts2, s_terms, s2 = self.stretch(scale, R, t, free)
                    if s_terms is None or np.any(s2[free] >= MODEL_STRETCH_EDGE * bound[free]):
                        continue
                    if (free & ~seen).any() and not self.ends_seen(Rs2, ts2, s2)[free & ~seen].all():
                        continue
                    tries.append(self.polish(s2, Rs2, ts2, s_terms))
                Rp, tp, p_terms, p_scale = min(tries, key=lambda r: r[2]["cost"])
                if p_terms["cost"] < terms["cost"] - MODEL_STRETCH_MIN_GAIN:
                    R, t, terms, scale = Rp, tp, p_terms, p_scale
                    note += f"; proportions measured, {_describe(scale)}"
                # Measured only if, as fitted, the two longest lengths show their ends.
                longest = np.argsort(np.broadcast_to(scale, (3,)) * self.model.extent)[1:]
                ends = self.ends_seen(R, t, scale)[longest].all()
                stops = self.ends_open(R, t, scale)[longest].all()
                terms["measured"] = ts_terms is not None and not edge and bool(ends) and bool(stops)
                if ts_terms is not None and not edge and not ends:
                    note += "; not counted as measured: an end of the part is hidden"
                elif ts_terms is not None and not edge and not stops:
                    note += "; not counted as measured: the part may go on behind something past an end"
            if hinted:
                note += f"; searched on at the other parts' scale, {self.scale_hint:.3f}, too hidden for its own"
            if located:
                note += f"; searched on at the scale it was located at, {_describe(scale)}, too hidden for its own"
            if not terms["measured"]:
                R, t, terms, scale, note = self.cover_seen(R, t, terms, scale, note, per_axis=hinted or located)
            if np.ndim(start) or start != 1.0:
                note += f"; pose searched at {_describe(start)}"
            terms["scale_note"] = note
            quality = ("fit" if not model_pose_rejection(terms)
                       else "coarse" if not model_pose_rejection(terms, coarse=True) else None)
            if quality == "fit":
                return terms, R, t, scale, quality
            results.append((terms, R, t, scale, quality))
        # No pose fits well: the best coarse one, else the best, refused.
        coarse = [r for r in results if r[4] == "coarse"]
        best = min(coarse or results, key=lambda r: r[0]["cost"])
        return best

    def settle_turns(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality, doubts): the pose, or the mesh turned
        onto its own box (ObjectModel.box_turns) where the view tells that
        apart from it (MODEL_TURN_COST_SLACK) and it fits as well; and the
        turns the view cannot tell from the one returned, relative to it (4x4)."""
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        B, c = self.model.box[:3, :3], self.model.box[:3, 3]
        M = self.linear(scale)
        centre = R @ (M @ c) + t
        round_k = round_axis(self.model, scale)
        same = [np.eye(3)] + list(self.model.box_same)
        near = np.cos(np.radians(MODEL_DOUBT_APART_DEG))

        def alike(Ra, Rb):
            # One pose as the other but for a turn that leaves the part as it was.
            D = B.T @ Ra.T @ Rb @ B
            return ((round_k is not None and D[round_k, round_k] > near)
                    or any((np.trace(S.T @ D) - 1) / 2 > near for S in same))

        rank = {"fit": 2, "coarse": 1, None: 0}

        def margin(way, other):
            # How far apart the view cannot tell two ways round (MODEL_TURN_COST_SLACK).
            left = lambda w: 1 - w[0]["fitness"] + w[0]["contradiction"]
            return max(MODEL_TURN_COST_SLACK, left(way) + left(other))

        ways = [(terms, R, t, quality)]
        for P in self.model.box_turns:
            values, vectors = np.linalg.eig(P)
            axis = np.real(vectors[:, np.argmin(np.abs(values - 1))])
            if round_k is not None and abs(axis[round_k]) > 0.99:
                continue                       # no other way round about it
            to = np.argmax(np.abs(P), axis=0)
            if np.any(np.maximum(extent[to], extent) > MODEL_DOUBT_SIDES * np.minimum(extent[to], extent)):
                continue
            R2 = R @ B @ P @ B.T
            R2, t2, terms2 = self.icp(scale, R2, centre - R2 @ (M @ c), MODEL_FINE_ROUNDS, MODEL_SAMPLES,
                                      MODEL_FINE_POINTS, 2 * self.tol)
            # A half turn is polished whatever its first refit costs (MODEL_DOUBT_HALF_TURNS).
            half = MODEL_DOUBT_HALF_TURNS and np.allclose(np.abs(P), np.eye(3))
            if terms2 is None or (not half and terms2["cost"] > terms["cost"] + 2 * margin(ways[0], ways[0])):
                continue
            R2, t2, terms2, _ = self.polish(scale, R2, t2, terms2)
            quality2 = ("fit" if not model_pose_rejection(terms2)
                        else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
            if quality2 is not None and not any(alike(way[1], R2) for way in ways):
                ways.append((terms2, R2, t2, quality2))
        kept = ways[0]
        for way in sorted(ways[1:], key=lambda way: way[0]["cost"]):
            if rank[way[3]] >= rank[kept[3]] and way[0]["cost"] < kept[0]["cost"] - margin(kept, way):
                kept = way
                break
        note = ""
        if kept is not ways[0]:
            note = f"; turned onto its box another way: cost {kept[0]['cost']:.3f} against {terms['cost']:.3f}"
        doubts = [way for way in sorted(ways, key=lambda way: way[0]["cost"])
                  if way is not kept and rank[way[3]] >= rank[kept[3]]
                  and way[0]["cost"] <= kept[0]["cost"] + margin(kept, way)]
        if doubts:
            costs = ", ".join(f"{way[0]['cost']:.3f}" for way in doubts)
            note += f"; {len(doubts)} other way{'s' if len(doubts) > 1 else ''} round in doubt, costing {costs}"
        # The rest that fit at all: kept for the binding, not in doubt (MODEL_DOUBT_HALF_TURNS).
        rest = [way for way in ways if way is not kept and not any(way is d for d in doubts)]
        terms2, R, t, quality = kept
        if terms2 is not terms:
            terms2["measured"] = terms.get("measured", False)
        terms2["scale_note"] = terms.get("scale_note", "") + note
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        out = []
        for way in doubts:
            other = np.eye(4)
            other[:3, :3], other[:3, 3] = way[1], way[2]
            out.append(np.linalg.inv(pose) @ other)
        self.unlikely = []
        for way in rest:
            other = np.eye(4)
            other[:3, :3], other[:3, 3] = way[1], way[2]
            self.unlikely.append(np.linalg.inv(pose) @ other)
        return terms2, R, t, scale, quality, out

    def rest_on_support(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality, sources): the pose, as high as the
        view sees its top along the box axis nearest the support's normal
        where it rests on the support and the view does not measure that
        axis, its bottom on the support (MODEL_SUPPORT_TOP_PERCENTILE); and
        per box axis, where its size came from: "measured" by the view,
        "support", or "mesh" (the mesh's proportion at the fit's scale)."""
        scale3 = np.array(np.broadcast_to(np.asarray(scale, np.float64), (3,)))
        seen = self.axes_seen(R, t, scale3)
        sources = ["measured" if s else "mesh" for s in seen]
        if self.plane is None:
            return terms, R, t, scale, quality, sources
        n, p0 = self.plane
        up = -n                                  # n points into the support
        axes = R @ self.model.box[:3, :3]
        k = int(np.argmax(np.abs(axes.T @ up)))
        if seen[k] or abs(axes[:, k] @ up) < np.cos(np.radians(MODEL_DOUBT_APART_DEG)):
            return terms, R, t, scale, quality, sources
        M = self.linear(scale3)
        heights = (self.model.points @ (R @ M).T + t - p0) @ up
        extent = scale3[k] * self.model.extent[k]
        bottom = heights.min()
        if abs(bottom) > (1 - 1 / MODEL_STRETCH_MAX) * extent + self.tol:
            return terms, R, t, scale, quality, sources          # not resting there
        top = (np.percentile((self.obs - p0) @ up, MODEL_SUPPORT_TOP_PERCENTILE)
               - MODEL_SUPPORT_TOP_SIGMAS * (self.sigma or 0.0))
        if top <= 0 or abs(heights.max() - top) > extent:
            return terms, R, t, scale, quality, sources          # the top seen is not its own
        lo, hi = MODEL_SCALE_RANGE
        scaled = scale3.copy()
        scaled[k] = float(np.clip(top / self.model.extent[k], lo, hi))
        if abs(scaled[k] / scale3[k] - 1) < MODEL_SCALE_FINE_RATIO - 1:
            sources[k] = "support"
            return terms, R, t, scale, quality, sources
        # Scaled about the box centre, then down onto the support.
        c = self.model.box[:3, 3]
        t2 = R @ (M @ c) + t - R @ (self.linear(scaled) @ c)
        low = ((self.model.points @ (R @ self.linear(scaled)).T + t2 - p0) @ up).min()
        t2 = t2 - low * up
        terms2 = self.terms(scaled, R, t2, MODEL_SAMPLES, MODEL_FINE_POINTS)
        if terms2 is None:
            return terms, R, t, scale, quality, sources
        quality2 = ("fit" if not model_pose_rejection(terms2)
                    else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
        rank = {"fit": 2, "coarse": 1, None: 0}
        if rank[quality2] < rank[quality]:
            return terms, R, t, scale, quality, sources
        for key in ("measured", "known_shape"):
            if key in terms:
                terms2[key] = terms[key]
        terms2["scale_note"] = (terms.get("scale_note", "") + f"; {extent * 1000:.1f} mm along its box axis {k}, "
                                f"unmeasured, set to the {top * 1000:.1f} mm the view sees its top over the support")
        sources[k] = "support"
        return terms2, R, t2, scaled, quality2, sources

    def measure_scale(self, R, t, scale=1.0):
        """The overall scale this rotation's fit, at `scale` and keeping its
        proportions, prefers: (overall scale, scale, R, t, terms, at the
        range's edge). Each scale is re-fit by ICP about the mesh's centre, so
        the part stays where it was seen."""
        mean = self.model.points.mean(0)
        centre = R @ (self.linear(scale) @ mean) + t
        proportions = np.asarray(scale, np.float64) / _overall(scale)
        lo, hi = MODEL_SCALE_RANGE

        def fit(s, R0, mesh_n, obs_n):
            shape = s * proportions if np.ndim(proportions) else s
            Rs, ts, terms = self.icp(shape, R0, centre - R0 @ (self.linear(shape) @ mean),
                                     MODEL_SCALE_ROUNDS, mesh_n, obs_n, 2 * self.tol)
            return (s, Rs, ts, terms) if terms is not None else None

        # A bracket about the start's size, moved on while its best is at
        # its end: the start is near the size, and the range is wide.
        b_lo = max(lo, _overall(scale) / MODEL_SCALE_BRACKET)
        b_hi = min(hi, _overall(scale) * MODEL_SCALE_BRACKET)
        coarse, tried = [], set()
        while True:
            steps = np.log(b_hi / b_lo) / np.log(MODEL_SCALE_COARSE_RATIO)
            grid = np.exp(np.linspace(np.log(b_lo), np.log(b_hi), int(np.ceil(steps)) + 1))
            coarse += [r for r in (fit(s, R, *MODEL_COARSE[:2]) for s in grid
                                   if round(float(np.log(s)), 6) not in tried) if r is not None]
            tried.update(round(float(np.log(s)), 6) for s in grid)
            if not coarse:
                return 1.0, scale, R, t, None, False
            # On only past the edge of everything searched so far: the edge
            # between two brackets is inside, and turning back at it would
            # never end.
            best = min(coarse, key=lambda r: r[3]["cost"])[0]
            searched = np.exp(sorted(tried))
            if best <= searched[0] * MODEL_SCALE_COARSE_RATIO and searched[0] > lo * MODEL_SCALE_FINE_RATIO:
                b_lo, b_hi = max(lo, searched[0] / MODEL_SCALE_BRACKET ** 2), searched[0]
            elif best >= searched[-1] / MODEL_SCALE_COARSE_RATIO and searched[-1] < hi / MODEL_SCALE_FINE_RATIO:
                b_lo, b_hi = searched[-1], min(hi, searched[-1] * MODEL_SCALE_BRACKET ** 2)
            else:
                break
        s0, R0 = min(coarse, key=lambda r: r[3]["cost"])[:2]
        # Re-fit about the best coarse scale with every point, out to the
        # neighbouring grid scales.
        k = int(np.ceil(np.log(MODEL_SCALE_COARSE_RATIO) / np.log(MODEL_SCALE_FINE_RATIO)))
        fine = [r for r in (fit(float(np.clip(s0 * MODEL_SCALE_FINE_RATIO ** i, lo, hi)), R0,
                                MODEL_SAMPLES, MODEL_FINE_POINTS) for i in range(-k, k + 1))
                if r is not None]
        if not fine:
            return 1.0, scale, R, t, None, False
        s, Rs, ts, terms = min(fine, key=lambda r: r[3]["cost"])
        edge = s <= lo * MODEL_SCALE_FINE_RATIO or s >= hi / MODEL_SCALE_FINE_RATIO
        shape = s * proportions if np.ndim(proportions) else s
        return s, shape, Rs, ts, terms, edge


def model_pose_rejection(terms, coarse=False):
    """Why the evidence falls short of a pose -- of a coarse one, if `coarse`
    -- or "" if it does not; not a judgement of the supplied object's identity."""
    fitness, contradiction, penetration, p95, share, floor = (
        (MODEL_COARSE_MIN_FITNESS, MODEL_COARSE_MAX_CONTRADICTION, MODEL_COARSE_MAX_PENETRATION,
         MODEL_COARSE_MAX_P95_DISTANCE_M, MODEL_COARSE_MAX_P95_SHARE, MODEL_COARSE_P95_FLOOR_M) if coarse else
        (MODEL_MIN_FITNESS, MODEL_MAX_CONTRADICTION, MODEL_MAX_PENETRATION, MODEL_MAX_P95_DISTANCE_M,
         MODEL_MAX_P95_SHARE, MODEL_P95_FLOOR_M))
    # The millimetres as the view's depth noise sets them (MODEL_NOISE_REF_M).
    p95, floor = p95 * terms.get("noise", 1.0), floor * terms.get("noise", 1.0)
    if "width" in terms:
        p95 = min(p95, max(floor, share * terms["width"]))
    failed = []
    if terms["fitness"] < fitness:
        failed.append(f"fitness < {fitness:.2f}")
    if terms["contradiction"] > contradiction:
        failed.append(f"contradiction > {contradiction:.2f}")
    if terms["penetration"] > penetration:
        failed.append(f"penetration > {penetration:.2f}")
    if terms["open_points"] < MIN_BOX_POINTS:
        failed.append(f"fewer than {MIN_BOX_POINTS} visible mesh depth samples")
    if terms["p95_distance"] > p95:
        failed.append(f"95th-percentile point distance > {p95 * 1000:.0f} mm")
    return "; ".join(failed)


def fit_model_box(model, view, K, rng=None, through=None, scale_hint=None, located=None, image=None,
                  known_shape=None):
    """Box of one object from its mesh: ((T_cam_box, extent, note), None) if
    the mesh fits the view, (None, reason) if not.

    The mesh is placed by ICP from a spread of starting rotations, each scored
    by what the camera sees along every mesh point's line of sight (see the
    comment above MODEL_SAMPLES). Its box then goes where the mesh went, at
    the mesh's size whatever share of the object was in view. `known_shape`, the part's
    scales along the mesh's box axes measured on earlier views, is kept
    where this view's own costs no less (MODEL_KNOWN_SHAPE_MIN_GAIN);
    model.fitted_known_shape says "kept", or "beaten" by this view's.
    """
    model.fitted_pose = model.fitted_scale = model.fitted_linear = None
    model.fitted_quality, model.fitted_measured = None, False
    model.fitted_spin_doubt = model.fitted_ends_doubt = model.fitted_id = None
    model.fitted_doubts, model.fitted_unlikely = [], []
    model.fitted_known_shape = None
    fit = _ModelFit(model, view, K, rng or np.random.default_rng(0), through, scale_hint, located, image,
                    known_shape)
    result = fit.run()
    if result is None:
        return None, "no pose left any mesh point in view"
    terms, R, t, scale, quality = result
    axis_source = axial = None
    if quality is None:
        # Refused, perhaps only for a height the view could not measure: into
        # the support (MODEL_SUPPORT_TOP_PERCENTILE).
        terms, R, t, scale, quality, axis_source = fit.rest_on_support(terms, R, t, scale, quality)
    doubts = []
    if quality is not None:
        # Relative to the pose, before the shape measured before moves it
        # (MODEL_KNOWN_SHAPE_MIN_GAIN).
        terms, R, t, scale, quality, doubts = fit.settle_turns(terms, R, t, scale, quality)
    if quality is not None and fit.known_shape is not None:
        terms, R, t, scale, quality = fit.known_shape_fit(terms, R, t, scale, quality)
    if quality is not None:
        rescued = axis_source
        terms, R, t, scale, quality, axis_source = fit.rest_on_support(terms, R, t, scale, quality)
        if rescued is not None:
            # Set by the support before, measured by the view only at that size.
            axis_source = ["support" if a == "support" else b for a, b in zip(rescued, axis_source)]
        terms, R, t, quality, axial = fit.axial(terms, R, t, scale, quality)
    stats = (f"fitness {terms['fitness']:.2f}, contradiction {terms['contradiction']:.2f}, "
             f"penetration {terms['penetration']:.2f}, {terms['hidden']:.0%} hidden "
             f"({terms['unmasked']:.0%} outside the mask at the mesh's depth), "
             f"p95 distance {terms['p95_distance'] * 1000:.1f} mm, "
             f"scale {_describe(scale)} ({terms['scale_note']}), {fit.observation_note}")
    if quality is None:
        return None, f"{stats}; {model_pose_rejection(terms, coarse=True)}"
    # The mesh's points are scaled before they are turned and moved: the
    # object is fitted_pose applied to fitted_linear @ the mesh, which for a
    # uniform scale is fitted_scale * the mesh.
    M = fit.linear(scale)
    T = np.eye(4)
    T[:3, :3] = R @ model.box[:3, :3]
    T[:3, 3] = R @ (M @ model.box[:3, 3]) + t
    model.fitted_pose = np.eye(4)
    model.fitted_pose[:3, :3], model.fitted_pose[:3, 3] = R, t
    model.fitted_scale = _overall(scale)
    model.fitted_linear = M
    model.fitted_quality, model.fitted_measured = quality, bool(terms["measured"])
    model.fitted_known_shape = terms.get("known_shape")
    # Along each box axis, where its size came from (MODEL_SUPPORT_TOP_PERCENTILE).
    model.fitted_axis_source = axis_source
    model.fitted_axial = axial                  # see _ModelFit.axial()
    model.fitted_id = uuid.uuid4().hex
    # Every way round in doubt; the first about the long box axis, and the
    # first about another, also as the spin's and the ends' (turn_kind()).
    model.fitted_doubts = doubts
    model.fitted_unlikely = list(getattr(fit, "unlikely", []))
    model.fitted_spin_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "spin"), None)
    model.fitted_ends_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "ends"), None)
    label = ("mesh pose fit" if quality == "fit"
             else f"coarse mesh pose fit ({model_pose_rejection(terms)})")
    return (T, np.broadcast_to(scale, (3,)) * model.extent, f"{label}, {stats}"), None


def retry_mesh_fit(model, mask, metres, K, through=None, scale_hint=None, located=None, image=None,
                   known_shape=None):
    """Worker owns its model state and frame snapshot, never the live pipeline."""
    model = copy.copy(model)
    try:
        view = observe(mask, metres, K)
        if view is None:
            return None, "insufficient current depth", model
        fit, reason = fit_model_box(model, view, K, through=through, scale_hint=scale_hint, located=located,
                                    image=image, known_shape=known_shape)
        return fit, reason, model
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", model


def fitted_axis_scale(model):
    """The fitted mesh's scale along its box axes."""
    linear = getattr(model, "fitted_linear", None)
    linear = linear if linear is not None else model.fitted_scale * np.eye(3)
    B = model.box[:3, :3]
    return np.diag(B.T @ np.asarray(linear, np.float64) @ B)


def refit_turn(model, pose):
    """turn_kind() of `pose` against the fitted mesh's pose where it is
    turned MODEL_REFIT_TURN_DEG or more from it, else None."""
    D = model.fitted_pose[:3, :3].T @ pose[:3, :3]
    if np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))) < MODEL_REFIT_TURN_DEG:
        return None
    return turn_kind(model, D)


def turn_kind(model, D):
    """"spin" for a turn D (relative, in the mesh's frame) about the fitted
    mesh's long box axis, "ends" about another, None about the long axis of a
    mesh round about it (round_axis()), which is no other way round."""
    # D + D^T = 2 cos(a) I + 2 (1 - cos(a)) u u^T: u the top eigenvector.
    axis = np.linalg.eigh(D + D.T)[1][:, -1]
    scale = fitted_axis_scale(model)
    long_k = int(np.argmax(scale * model.extent))
    if abs(axis @ model.box[:3, long_k]) < np.sqrt(0.5):
        return "ends"
    return None if round_axis(model, scale) == long_k else "spin"


def carried_view(model, mask, metres, K, through=None):
    """This frame's view of a fitted part, as a _ModelFit to score or
    correct its carried pose on; None where the view shows too little."""
    view = observe(mask, metres, K)
    if view is None:
        return None
    return _ModelFit(model, view, K, np.random.default_rng(0), through)


def correct_carried_pose(fit, model, pose, scale=None):
    """The carried mesh pose corrected on this frame's view `fit`
    (carried_view()), and its cost there; `pose` itself where the correction
    gains less than MODEL_CORRECT_MIN_GAIN. See MODEL_CORRECT_ROUNDS.
    `scale`: along the mesh's box axes, the fit's where None."""
    scale = fitted_axis_scale(model) if scale is None else scale
    start = fit.terms(scale, pose[:3, :3], pose[:3, 3], *MODEL_CHECK_POINTS)
    if start is None:
        return pose, np.inf
    R, t, terms = fit.icp(scale, pose[:3, :3], pose[:3, 3], MODEL_CORRECT_ROUNDS,
                          *MODEL_CHECK_POINTS, MODEL_CORRECT_RADIUS_M * fit.noise)
    if terms is None or start["cost"] - terms["cost"] < MODEL_CORRECT_MIN_GAIN:
        return pose, float(start["cost"])
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out, float(terms["cost"])


def correction_size(model, pose, corrected):
    """How far `corrected` puts the fitted mesh from `pose`: its box
    centre's move (m) and the turn (deg)."""
    B = model.box[:3, :3]
    centre = (B * fitted_axis_scale(model)) @ B.T @ model.box[:3, 3]
    move = np.linalg.norm((corrected[:3, :3] - pose[:3, :3]) @ centre + corrected[:3, 3] - pose[:3, 3])
    turn = np.degrees(np.linalg.norm(cv2.Rodrigues(pose[:3, :3].T @ corrected[:3, :3])[0]))
    return float(move), float(turn)


def carried_fit_rejection(model, pose, mask, metres, K, through=None, fit=None, scale=None):
    """Why the fitted mesh, at `pose` in this frame's camera (its fit carried
    here by the tracked motion), no longer lies on what the camera sees of
    the part; "" where it does, or where the view shows too little to say.
    `fit`: this frame's carried_view(), where already made; `scale`: along
    the mesh's box axes, the fit's where None.

    Only the observed points' side of the coarse acceptance is asked: their
    share on the mesh and their 95th-percentile distance. A few millimetres
    of tracked depth already put a thin part seen face-on past the coarse
    contradiction (a ring 5 mm deep: over 0.25), and a part held up has for
    its "support" whatever is behind it."""
    if fit is None:
        fit = carried_view(model, mask, metres, K, through)
    if fit is None:
        return ""
    terms = fit.terms(fitted_axis_scale(model) if scale is None else scale, pose[:3, :3], pose[:3, 3],
                      *MODEL_CHECK_POINTS)
    if terms is None:
        return "no mesh point in view"
    failed = []
    if terms["fitness"] < MODEL_COARSE_MIN_FITNESS:
        failed.append(f"fitness {terms['fitness']:.2f} < {MODEL_COARSE_MIN_FITNESS:.2f}")
    p95 = min(MODEL_COARSE_MAX_P95_DISTANCE_M * fit.noise,
              max(MODEL_COARSE_P95_FLOOR_M * fit.noise, MODEL_COARSE_MAX_P95_SHARE * terms["width"]))
    if terms["p95_distance"] > p95:
        failed.append(f"95th-percentile point distance {terms['p95_distance'] * 1000:.0f} mm > {p95 * 1000:.0f} mm")
    return "; ".join(failed)


def sample_evidence(fit, P, spacing):
    """Per point of P (this frame's camera, a way round's samples `spacing`
    apart): (on, through) -- the camera sees the part's surface at it, in
    the mask and at its depth; or sees past it, inside the mask (a hole) or
    clear of it, or the bare support there. Neither where it is hidden, on
    the far side, off the image or a little in front -- the cases as
    _ModelFit.terms() counts them -- nor past it within MODEL_MASK_GROW_PX of
    the mask's edge: a sample on a side seen edge-on lies a pixel either way
    of the outline, and past it there is whatever is behind the part."""
    on, through = np.zeros(len(P), bool), np.zeros(len(P), bool)
    seen = _seen_from_camera(P, fit.K, spacing, fit.depth.shape)
    idx, u, v, d = fit.lines_of_sight(P, seen)
    in_mask = fit.grown[v, u]
    open_ = (d > 0) & (d >= P[idx, 2] - fit.hidden_m)
    if fit.plane is not None:
        n, p0 = fit.plane
        open_ &= ~(~in_mask & (d <= P[idx, 2] + fit.tol) & ((fit.cam[v, u] - p0) @ n < -PLANE_TOL_M))
    on[idx] = open_ & in_mask & (np.abs(d - P[idx, 2]) <= fit.tol)
    through[idx] = open_ & (~in_mask | (fit.mask[v, u] & (d > P[idx, 2] + fit.tol)))
    return on, through


def mesh_depth(model, pose, linear, K, shape):
    """The depth (m) of the fitted mesh's nearest surface at each pixel of an
    image of `shape`, the mesh at `pose` and scaled by `linear` (as
    fitted_linear); inf where it casts no silhouette. Its faces are painted
    far to near, each at its mean depth."""
    h, w = shape
    out = np.full((h, w), np.inf, np.float32)
    P = (model.mesh_v @ np.asarray(linear, np.float64).T) @ pose[:3, :3].T + pose[:3, 3]
    uv = _pixels(P, K)
    f = model.mesh_f
    keep = np.all(P[f, 2] > 1e-3, axis=1)
    near = (uv[f, 0].max(1) >= 0) & (uv[f, 0].min(1) < w) & (uv[f, 1].max(1) >= 0) & (uv[f, 1].min(1) < h)
    f = f[keep & near & np.all(np.abs(uv[f]) < 4 * max(h, w), axis=(1, 2))]
    if not len(f):
        return out
    z = P[f, 2].mean(1)
    order = np.argsort(-z)
    for pts, d in zip(np.round(uv[f[order]] * 16).astype(np.int32), z[order]):
        cv2.fillConvexPoly(out, pts, float(d), lineType=cv2.LINE_8, shift=4)
    return out


def ends_evidence(fit, Zj, Zk):
    """Two ways round's silhouettes against this frame's mask (see
    ENDS_SILHOUETTE_MIN_SHARE): `Zj`, `Zk` their mesh_depth()s. {"for_j",
    "for_k": the pixels for each, "told": both, "area": the silhouettes'
    union, "winner": "j", "k" or None, "why": where None}."""
    Sj, Sk = np.isfinite(Zj), np.isfinite(Zk)
    area = int((Sj | Sk).sum())
    d = fit.depth
    kernel = np.ones((2 * ENDS_SILHOUETTE_EDGE_PX + 1,) * 2, np.uint8)
    mask8 = fit.mask.astype(np.uint8)
    edge = (cv2.dilate(mask8, kernel) > 0) & ~(cv2.erode(mask8, kernel) > 0)
    has = np.isfinite(d) & (d > 0) & ~edge

    def judge(only, Z):
        # Pixels only this way round covers: (for it, against it).
        usable = only & has & ~(d < Z - fit.tol)           # nothing nearer hides it
        on = usable & fit.mask & (d <= Z + fit.tol)
        return int(on.sum()), int((usable & ~on).sum())

    on_j, off_j = judge(Sj & ~Sk, Zj)
    on_k, off_k = judge(Sk & ~Sj, Zk)
    n_j, n_k = on_j + off_k, on_k + off_j
    told = n_j + n_k
    out = {"for_j": n_j, "for_k": n_k, "told": told, "area": area, "winner": None, "why": None}
    if not area or told < ENDS_SILHOUETTE_MIN_SHARE * area:
        out["why"] = f"{told} px told, under {ENDS_SILHOUETTE_MIN_SHARE:.0%} of {area}"
    elif max(n_j, n_k) < ENDS_SILHOUETTE_AGREE * told:
        out["why"] = f"{max(n_j, n_k)} of {told} px for either, under {ENDS_SILHOUETTE_AGREE:.0%}"
    else:
        out["winner"] = "j" if n_j > n_k else "k"
    return out


def ends_extent(fit, model, pose, scale):
    """Along the fitted mesh's long box axis, at `pose` and `scale` (along
    the mesh's box axes): its two ends' points (camera frame), how far each
    lies from where the part seen ends there (m), whether each is hidden
    (_ModelFit.hidden_ends()), and its length (m). See ENDS_EXTENT_SHARE."""
    scale = np.array(np.broadcast_to(np.asarray(scale, np.float64), (3,)))
    R, t = pose[:3, :3], pose[:3, 3]
    k = int(np.argmax(scale * model.extent))
    M = fit.linear(scale)
    centre = R @ (M @ model.box[:3, 3]) + t
    axis = R @ model.box[:3, k]
    along = (model.points @ (R @ M).T + t - centre) @ axis
    mesh_ends = np.array([along.min(), along.max()])
    ordered = np.sort((fit.obs - centre) @ axis)
    few = min(10, len(ordered)) - 1
    seen_ends = np.array([ordered[few], ordered[-1 - few]])
    return {"points": centre + mesh_ends[:, None] * axis, "off": np.abs(mesh_ends - seen_ends),
            "hidden": np.asarray(fit.hidden_ends(R, t, scale)[k], bool), "length": float(np.ptp(mesh_ends))}


def ends_extent_vote(a, b):
    """Two ways round's ends_extent()s, `a` and `b`, against each other:
    {"off_j", "off_k": each one's distance from where the part ends, summed
    over the ends neither hides (m), "share": their difference over the
    part's length, "ends": how many ends told, "winner": "j", "k" or None,
    "why"}. An end of one is the other's nearest end, end for end."""
    off_a = off_b = 0.0
    used = 0
    for e in range(2):
        f = int(np.argmin(np.linalg.norm(b["points"] - a["points"][e], axis=1)))
        if a["hidden"][e] or b["hidden"][f]:
            continue
        off_a, off_b, used = off_a + a["off"][e], off_b + b["off"][f], used + 1
    length = 0.5 * (a["length"] + b["length"])
    share = abs(off_a - off_b) / length if length > 0 else 0.0
    out = {"off_j": round(off_a, 4), "off_k": round(off_b, 4), "share": round(share, 4), "ends": used,
           "winner": None, "why": None}
    if not used:
        out["why"] = "both ends hidden"
    elif share < ENDS_EXTENT_SHARE:
        out["why"] = f"{share:.1%} of its length apart, under {ENDS_EXTENT_SHARE:.0%}"
    else:
        out["winner"] = "j" if off_a < off_b else "k"
    return out


def _ends_flipped(r):
    """A pair's ends' evidence (Binding.vote_ends()) with j and k swapped."""
    swap = {"j": "k", "k": "j"}
    sil, ext = r["silhouette"], r["extent"]
    return {"silhouette": {**sil, "for_j": sil["for_k"], "for_k": sil["for_j"], "winner": swap.get(sil["winner"])},
            "extent": {**ext, "off_j": ext["off_k"], "off_k": ext["off_j"], "winner": swap.get(ext["winner"])}}


def rest_margin(P, plane, tol):
    """How far inside (+, metres) or outside (-) the polygon of its points
    within `tol` of the support `plane` a part at P (its surface samples,
    camera frame) has its centre, both seen along the support's normal; None
    where no point is that near the support."""
    n, p0 = plane
    near = P[(P - p0) @ n >= -tol]
    if not len(near):
        return None
    e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    E = np.column_stack([e1, np.cross(n, e1)])
    q, c = (near - p0) @ E, (P.mean(0) - p0) @ E
    try:
        hull = ConvexHull(q)
    except Exception:
        hull = None                            # a point or a line: nothing inside it
    if hull is not None:
        # Facets as n.x + b <= 0 inside: the least distance in to an edge.
        inside = -(hull.equations[:, :2] @ c + hull.equations[:, 2])
        if inside.min() >= 0:
            return float(inside.min())
        corners = q[hull.vertices]
        edges = list(zip(corners, np.roll(corners, -1, axis=0)))
    else:
        order = np.argsort(q @ (q[-1] - q[0]) if len(q) > 1 else np.zeros(len(q)))
        edges = [(q[order[0]], q[order[-1]])]
    out = np.inf
    for a, b in edges:
        ab = b - a
        s = np.clip((c - a) @ ab / max(ab @ ab, 1e-18), 0.0, 1.0)
        out = min(out, float(np.linalg.norm(c - a - s * ab)))
    return -out


def same_way(model, Ra, Rb):
    """Whether rotations Ra and Rb put the fitted mesh the same way round:
    within MODEL_DOUBT_APART_DEG but for a turn that leaves it as it was
    (ObjectModel.box_same, the axis it is round about); as settle_turns()."""
    B = model.box[:3, :3]
    D = B.T @ Ra.T @ Rb @ B
    near = np.cos(np.radians(MODEL_DOUBT_APART_DEG))
    k = round_axis(model, fitted_axis_scale(model))
    return ((k is not None and D[k, k] > near)
            or any((np.trace(S.T @ D) - 1) / 2 > near for S in [np.eye(3)] + list(model.box_same)))


def model_doubts(model):
    """A fitted mesh's ways round in doubt (relative turns): its fit's, then
    a refit's held back (MODEL_REFIT_TURN_DEG)."""
    doubts = list(getattr(model, "fitted_doubts", None) or [])
    for d in (getattr(model, "fitted_spin_doubt", None), getattr(model, "fitted_ends_doubt", None)):
        if d is not None and not any(np.allclose(d, k) for k in doubts):
            doubts.append(d)
    return doubts


class Binding:
    """Which way round a fitted mesh lies on its part: the published pose and
    the other ways round that fit, relative to it (turns[0] the identity),
    each carried by the part's one motion; the votes of the frames looked at
    so far, pair by pair; which are ruled out, and by what. See
    BINDING_CONFIDENCE."""

    def __init__(self, model, before=None):
        self.model = model
        self.fit_id = model.fitted_id
        self.scale = fitted_axis_scale(model)
        self.spacing = _overall(self.scale) * model.spacing
        # Each way round's scale along the mesh's box axes and its samples so
        # scaled, in its own frame (scale_for()).
        self.turns, self.scales, self.Xs, self.ids, self.alive, self.ruled = [], [], [], [], [], []
        self.n_samples = len(model.points)
        # Per ordered pair (j, k), over the frames looked at: votes for j and
        # for k, the frames that gave one, and the samples they agree on seen
        # through and told (BINDING_CONFIDENCE).
        self.tally = {}
        self.unstable = []
        # Ways round that fit the init's view but beyond the doubt's margin
        # (MODEL_DOUBT_HALF_TURNS): hypotheses, not sent in doubt.
        self.unlikely = []
        self.frames, self.tol, self.by, self.number = 0, MODEL_TOL_M, None, 0
        # The published way round's pose on the last frame looked at.
        self.last_view = None
        self._trees, self._apart = {}, {}
        # The last frame's silhouettes of the ways round turned end for end,
        # {(j, k): ends_evidence()} (ENDS_SILHOUETTE_MIN_SHARE).
        self.last_ends = {}
        self.add(np.eye(4))
        for d in model_doubts(model):
            self.add(d)
        for d in getattr(model, "fitted_unlikely", None) or []:
            k = self.add(d)
            if k is not None:
                self.unlikely[k] = True
        if before is not None:
            self.carry(before)

    def add(self, turn):
        """A way round (a turn relative to the published pose), unless it is
        one held already, alive or ruled out; its index or None."""
        if any(same_way(self.model, held[:3, :3], turn[:3, :3]) for held in self.turns):
            return None
        self.turns.append(np.asarray(turn, np.float64))
        self.scales.append(self.scale_for(self.turns[-1]))
        B = self.model.box[:3, :3]
        self.Xs.append(self.model.points @ ((B * self.scales[-1]) @ B.T).T)
        self.number += 1
        self.ids.append(f"{self.fit_id[:6]}-{self.number}")
        self.alive.append(True)
        self.ruled.append(None)
        self.unstable.append(0)
        self.unlikely.append(False)
        return len(self.turns) - 1

    def carry(self, before):
        """The ways round of an earlier fit's binding and their verdicts, onto
        these where one is the same way round, else as more: the same samples
        of the same mesh, in the init's frame both (before.model.fitted_pose,
        self.model.fitted_pose)."""
        if before.n_samples != self.n_samples:
            return
        mine = [self.model.fitted_pose @ T for T in self.turns]
        onto = {}
        for j, T in enumerate(before.turns):
            theirs = before.model.fitted_pose @ T
            k = next((k for k, m in enumerate(mine) if same_way(self.model, m[:3, :3], theirs[:3, :3])), None)
            if k is None:
                # A way round the new fit did not leave in doubt: kept, living
                # or not, so a settled binding stays settled.
                k = self.add(np.linalg.inv(self.model.fitted_pose) @ theirs)
                mine.append(theirs)
            if k is None:
                continue
            onto[j] = k
            self.unlikely[k] = self.unlikely[k] and before.unlikely[j]
            if not before.alive[j]:
                self.alive[k], self.ruled[k] = False, before.ruled[j]
        for (j, k), counts in before.tally.items():
            if j in onto and k in onto:
                self.tally[(onto[j], onto[k])] = dict(counts)
        self.frames, self.by = before.frames, before.by
        if before.last_view is not None:
            # In this fit's frame: the same part, seen the same.
            self.last_view = before.last_view @ np.linalg.inv(before.model.fitted_pose) @ self.model.fitted_pose

    def scale_for(self, turn):
        """A way round's scale along the mesh's box axes, for its `turn`
        relative to the published one: the published one's, each axis's
        taken to the axis the turn takes it to -- so it fills the box the
        published one was fitted to, as a turn onto its own box does
        (settle_turns()); left as it is where the turn takes no box axis onto
        another. Kept as fitted, a quarter turn of a part fitted a little
        longer one way than the other is out of its box by the difference,
        and the view's evidence is against it for that alone."""
        B = self.model.box[:3, :3]
        P = np.abs(np.round(B.T @ turn[:3, :3] @ B))
        if not (np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)):
            return self.scale
        return P.T @ self.scale

    def placed(self, k):
        """Way round k's samples in the published mesh's frame."""
        T = self.turns[k]
        return self.Xs[k] @ T[:3, :3].T + T[:3, 3]

    def apart(self, j, k):
        """Which of way round j's samples lie farther than MODEL_TOL_M from
        way round k's surface: where j has something k has not."""
        if (j, k) not in self._apart:
            if k not in self._trees:
                self._trees[k] = cKDTree(self.placed(k))
            self._apart[(j, k)] = self._trees[k].query(self.placed(j))[0] > MODEL_TOL_M
        return self._apart[(j, k)]

    def living(self):
        return [k for k, alive in enumerate(self.alive) if alive]

    def observe(self, fit, pose, at_rest):
        """One frame's evidence: `fit` its carried_view(), `pose` the
        published way round's corrected pose there; `at_rest` whether the
        part may be judged at rest on the support (no robot on it). Whether
        the frame was looked at."""
        living = self.living()
        if len(living) < 2:
            return False
        if self.last_view is not None:
            move, turn = correction_size(self.model, self.last_view, pose)
            if move <= MODEL_CORRECT_MAX_M and turn <= MODEL_CORRECT_MAX_DEG:
                return False                   # the view looked at already
        self.last_view = pose
        margins, verdicts, placed, poses = {}, {}, {}, {}
        for k in living:
            posed = pose @ self.turns[k]
            if k:
                # Its own few rounds on this frame's depth, as the published
                # one had (check_carried_meshes()); not kept, so it cannot walk.
                posed, _ = correct_carried_pose(fit, self.model, posed, self.scales[k])
            poses[k] = posed
            P = placed[k] = self.Xs[k] @ posed[:3, :3].T + posed[:3, 3]
            verdicts[k] = sample_evidence(fit, P, self.spacing)
            if at_rest and fit.plane is not None:
                margins[k] = rest_margin(P, fit.plane, fit.tol)
        self.tol = fit.tol
        cast = self.vote(verdicts, placed)
        self.vote_ends(fit, poses, cast)
        for k in living:
            m = margins.get(k)
            if (m is not None and m < -fit.tol
                    and any(margins.get(j) is not None and margins[j] >= 0 for j in living if j != k)):
                self.unstable[k] += 1
        self.frames += 1
        return True

    def feature_groups(self, P_j, P_k):
        """Way round j's features against k, each posed on this frame (P_j,
        P_k): its samples farther than the tolerance from k's, where it has
        what k has not, joined where within MODEL_FRAGMENT_GAP spacings of each
        other, those of MIN_BOX_POINTS samples or more; and where they agree.
        Where they are posed, each corrected on the frame: where they differ
        as turned relative to each other, the corrected one lies on the part
        -- on the tester turning, those samples of the wrong way round came
        out seen on in 13 votes of 36."""
        apart = cKDTree(P_k).query(P_j, distance_upper_bound=self.tol)[0] > self.tol
        idx = np.flatnonzero(apart)
        out = []
        if len(idx) >= MIN_BOX_POINTS:
            pairs = cKDTree(P_j[idx]).query_pairs(MODEL_FRAGMENT_GAP * self.spacing, output_type="ndarray")
            graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(idx), len(idx)))
            count, labels = connected_components(graph, directed=False)
            out = [idx[labels == c] for c in range(count) if np.sum(labels == c) >= MIN_BOX_POINTS]
        return out, ~apart

    def vote(self, verdicts, placed):
        """One frame's votes, {k: (on, through)} of the living ways round,
        each posed on the frame as `placed` ({k: samples}), into the tally of
        each ordered pair (BINDING_CONFIDENCE); the ordered pairs that had a
        vote."""
        voted = set()
        unit = min(1.0, (self.spacing / self.tol) ** 2)
        groups = {(j, k): self.feature_groups(placed[j], placed[k]) for j in placed for k in placed if j != k}
        for j, (on_j, through_j) in verdicts.items():
            for k, (on_k, through_k) in verdicts.items():
                if j == k:
                    continue
                counts = self.tally.setdefault((j, k), dict(votes_j=0, votes_k=0, frames=0, wrong=0, told=0,
                                                            kinds=[0, 0, 0, 0]))
                cast = 0
                # j's features: seen on, for j; seen through, for k. k's: the other way.
                for side, (features, (on, through)) in enumerate(((groups[(j, k)][0], (on_j, through_j)),
                                                                  (groups[(k, j)][0], (through_k, on_k)))):
                    for c in features:
                        n_for, n_against = int(on[c].sum()), int(through[c].sum())
                        if (n_for + n_against) * unit < 1 or n_for == n_against:
                            continue
                        counts["votes_j" if n_for > n_against else "votes_k"] += 1
                        # Which: j's own seen on / through, k's seen through / on.
                        counts["kinds"][2 * side + (n_for < n_against)] += 1
                        cast += 1
                counts["frames"] += bool(cast)
                if cast:
                    voted.add((j, k))
                agree_j, agree_k = groups[(j, k)][1], groups[(k, j)][1]
                counts["wrong"] += int(through_j[agree_j].sum()) + int(through_k[agree_k].sum())
                counts["told"] += int((on_j | through_j)[agree_j].sum()) + int((on_k | through_k)[agree_k].sum())
        return voted

    def vote_ends(self, fit, poses, voted):
        """This frame's votes on each pair of living ways round turned end for
        end, each at its pose here (`poses`): their silhouettes'
        (ENDS_SILHOUETTE_MIN_SHARE) and their ends' (ENDS_EXTENT_SHARE), into
        the tally as the features' votes go; `voted`, the ordered pairs
        vote() counted this frame for already."""
        self.last_ends = {}
        living = sorted(poses)
        pairs = [(j, k) for j in living for k in living if j < k
                 and turn_kind(self.model, (np.linalg.inv(self.turns[j]) @ self.turns[k])[:3, :3]) == "ends"]
        if not pairs:
            return
        B = self.model.box[:3, :3]
        involved = {x for pair in pairs for x in pair}
        depth = {k: mesh_depth(self.model, poses[k], (B * self.scales[k]) @ B.T, fit.K, fit.depth.shape)
                 for k in involved}
        extent = {k: ends_extent(fit, self.model, poses[k], self.scales[k]) for k in involved}
        for j, k in pairs:
            result = self.last_ends[(j, k)] = {"silhouette": ends_evidence(fit, depth[j], depth[k]),
                                               "extent": ends_extent_vote(extent[j], extent[k])}
            cast = False
            for kind, r in result.items():
                if r["winner"] is None:
                    continue
                winner = j if r["winner"] == "j" else k
                for a, b in ((j, k), (k, j)):
                    counts = self.tally.setdefault((a, b), dict(votes_j=0, votes_k=0, frames=0, wrong=0, told=0,
                                                                kinds=[0, 0, 0, 0]))
                    counts["votes_j" if winner == a else "votes_k"] += 1
                    counts.setdefault(kind, [0, 0])[0 if winner == a else 1] += 1
                    if (a, b) not in voted and not cast:
                        counts["frames"] += 1
                cast = True

    def ends_favoured(self):
        """The living way round the last frame's ends voted for over the
        published one, end for end (vote_ends()): more of its two kinds of
        evidence for it than against it; of several, the one with the most,
        then the nearest to where the part ends; None where none."""
        best = None
        for (j, k), r in self.last_ends.items():
            if 0 not in (j, k):
                continue
            other = k if j == 0 else j
            mine = "k" if j == 0 else "j"
            score = sum((e["winner"] == mine) - (e["winner"] not in (None, mine)) for e in r.values())
            ext = r["extent"]
            off = ext["off_k"] if mine == "k" else ext["off_j"]
            if score > 0 and self.alive[other] and (best is None or (score, -off) > best[1]):
                best = (other, (score, -off))
        return None if best is None else best[0]

    def odds(self):
        """(the error rate over every living pair, None) and, per living pair
        (j, k), the evidence for j over k: (the log odds of j against k and
        the mesh not matching there, the votes for j, for k, the frames that
        voted). See BINDING_CONFIDENCE."""
        living = self.living()
        out, all_wrong, all_told = {}, 0, 0
        for j in living:
            for k in living:
                if j == k:
                    continue
                c = self.tally.get((j, k), dict(votes_j=0, votes_k=0, frames=0, wrong=0, told=0))
                all_wrong, all_told = all_wrong + c["wrong"], all_told + c["told"]
                eps = (c["wrong"] + 1) / (c["told"] + 2)
                a, b = c["votes_j"], c["votes_k"]
                for_j = a * np.log(1 - eps) + b * np.log(eps)
                for_k = b * np.log(1 - eps) + a * np.log(eps)
                neither = (a + b) * np.log(0.5)
                out[(j, k)] = (float(for_j - np.logaddexp(for_k, neither)), a, b, c["frames"])
        return (all_wrong + 1) / (all_told + 2), None, out

    def decide(self):
        """Rule out what the evidence and the support rule out; the way round
        to publish (an index), and whether anything changed."""
        living = self.living()
        if len(living) < 2:
            return living[0] if living else 0, False
        changed = False
        for k in living:
            if self.unstable[k] and any(not self.unstable[j] for j in living if j != k):
                self.alive[k], self.ruled[k], changed = False, ("support", "it could not rest on the support so"), True
        living = self.living()
        _, _, pairs = self.odds()
        need = np.log(BINDING_CONFIDENCE / (1 - BINDING_CONFIDENCE))
        out = {}
        for (j, k), (log_odds, votes_j, votes_k, frames) in pairs.items():
            if log_odds >= need and frames >= SURFACE_MIN_SEEN:
                out[k] = ("evidence", f"the views' evidence for {self.ids[j]}: {votes_j} votes to {votes_k} "
                                      f"on {frames} frames, odds {np.exp(log_odds):.3g} to 1")
        if out and len(out) < len(living):
            for k, why in out.items():
                self.alive[k], self.ruled[k] = False, why
            changed = True
        if changed:
            kinds = {self.ruled[k][0] for k in range(len(self.turns)) if not self.alive[k]}
            self.by = " and ".join(sorted(kinds))
        living = self.living()
        for k in living:
            # More votes for it than against it, against the published one: in doubt.
            if self.unlikely[k] and k and (k, 0) in pairs and pairs[(k, 0)][1] > pairs[(k, 0)][2]:
                self.unlikely[k], changed = False, True
        if self.alive[0]:
            return 0, changed
        post = self.posterior()
        return max(living, key=lambda k: post.get(k, 0.0)), changed

    def posterior(self):
        """Each living way round's probability on the evidence so far."""
        living = self.living()
        if len(living) < 2:
            return {k: 1.0 for k in living}
        _, _, pairs = self.odds()
        eps = {pair: (self.tally.get(pair, {}).get("wrong", 0) + 1) / (self.tally.get(pair, {}).get("told", 0) + 2)
               for pair in pairs}
        score = {j: sum((pairs[(j, k)][1] - pairs[(j, k)][2]) * np.log((1 - eps[(j, k)]) / eps[(j, k)])
                        for k in living if k != j) / (len(living) - 1) for j in living}
        top = max(score.values())
        weight = {j: np.exp(s - top) for j, s in score.items()}
        total = sum(weight.values())
        return {j: float(weight[j] / total) for j in living}

    def choose(self, ident, by="person"):
        """Accept a living way round by id. Camera choices keep alternatives
        for later views; person/robot choices rule the alternatives out.
        update_bindings() publishes the accepted camera choice."""
        if ident not in self.ids or not self.alive[self.ids.index(ident)]:
            return False
        keep = self.ids.index(ident)
        if by == "camera":
            # One automatic view is revisable: keep the other hypotheses
            # available to the independent views gathered during motion.
            self.by = by
            return True
        for k in self.living():
            if k != keep:
                self.alive[k], self.ruled[k] = False, (by, f"the {by} chose {ident}")
        self.by = by
        return True

    def rebase(self, k):
        """Make way round k the published one: the turns, re-expressed
        relative to it, k's first; its features' caches dropped."""
        order = [k] + [j for j in range(len(self.turns)) if j != k]
        if self.last_view is not None:
            self.last_view = self.last_view @ self.turns[k]
        base = np.linalg.inv(self.turns[k])
        self.turns = [base @ self.turns[j] for j in order]
        for name in ("scales", "Xs", "ids", "alive", "ruled", "unstable", "unlikely"):
            setattr(self, name, [getattr(self, name)[j] for j in order])
        self.unlikely[0] = False
        self.scale = self.scales[0]
        new = {old: i for i, old in enumerate(order)}
        self.tally = {(new[j], new[k]): c for (j, k), c in self.tally.items()}
        self.last_ends = {(new[j], new[k]) if new[j] < new[k] else (new[k], new[j]):
                          (r if new[j] < new[k] else _ends_flipped(r))
                          for (j, k), r in self.last_ends.items()}
        self._trees, self._apart = {}, {}

    def in_doubt(self):
        """The living ways round sent in doubt (model_doubts): all but the
        published one and those still unlikely (MODEL_DOUBT_HALF_TURNS)."""
        return [k for k in self.living() if k and not self.unlikely[k]]

    def state(self):
        living = self.living()
        if len(living) > 1:
            return "ambiguous"
        return "locked" if len(self.turns) > 1 else "single"

    def features(self, k, pose, K, shape):
        """Up to BINDING_FEATURE_PX pixels of way round k's samples where it
        differs from the other living ones, at `pose` (the published one's,
        this frame's camera)."""
        living = [j for j in self.living() if j != k]
        if not living:
            return []
        apart = np.any([self.apart(k, j) for j in living], axis=0)
        T = pose @ self.turns[k]
        P = self.Xs[k][apart] @ T[:3, :3].T + T[:3, 3]
        uv = _pixels(P, K)
        h, w = shape
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        if len(uv) > BINDING_FEATURE_PX:
            uv = uv[np.linspace(0, len(uv) - 1, BINDING_FEATURE_PX).astype(int)]
        return np.round(uv).astype(int).tolist()

    def reply(self, held, pose, K, shape):
        """model_binding: the state, what settled it, and each living way
        round -- its id, its turn relative to the published pose (`held` the
        turn about the mesh's round axis bboxes() publishes it with), its
        probability, and where its features are in the image."""
        post = self.posterior()
        eps, unit, _ = self.odds() if len(self.living()) > 1 else (None, None, None)
        inv_held = np.linalg.inv(held)
        return {"state": self.state(), "by": self.by if self.state() != "single" else "init",
                "confidence": BINDING_CONFIDENCE, "frames": self.frames,
                "error_rate": None if eps is None else round(float(eps), 4),
                "ruled_out": [{"id": self.ids[k], "by": self.ruled[k][0], "why": self.ruled[k][1]}
                              for k in range(len(self.turns)) if not self.alive[k]],
                "hypotheses": [{"id": self.ids[k], "turn": (inv_held @ self.turns[k] @ held).tolist(),
                                "p": round(post.get(k, 1.0), 4), "in_doubt": not self.unlikely[k],
                                "features_px": self.features(k, pose, K, shape)} for k in self.living()],
                "ends_evidence": [{"ids": [self.ids[j], self.ids[k]],
                                   "votes": {kind: self.tally.get((j, k), {}).get(kind, [0, 0])
                                             for kind in ("silhouette", "extent")},
                                   "last": {"silhouette": {key: r["silhouette"][key]
                                                           for key in ("for_j", "for_k", "told", "area", "why")},
                                            "extent": {key: r["extent"][key]
                                                       for key in ("off_j", "off_k", "share", "ends", "why")}}}
                                  for (j, k), r in self.last_ends.items()]}


# Locating an object by its mesh, for an object sent with a mesh and no click.
# SAM2 proposes masks where its grid of prompt points falls, and the depth
# image's regions (depth_proposals()) are proposed beside them. Those mostly
# on the robot, as the client sends it, are not parts. The mesh's box rules
# most of the rest out by size and proportions, which is cheap; each left is
# fitted roughly (LOCATE_SCREEN), the LOCATE_FITS best of those properly
# (_quick_fit()), and the best fit's visible surface, projected into the
# image, gives the click: points inside the part, as a person would pick
# them. The click comes from the fitted mesh rather than from the proposal,
# because a proposal can be a piece of the part (the label on a can) that
# the mesh fits as well.
#
# Proportions only rule out; they do not rank: a reconstruction's are not
# the part's. RI-S1-1's bar, 340 x 27 mm, is seen 12.9 to 1 against the
# mesh's 386 x 39, 9.8 to 1.
#
# SAM2 did propose RI-S1-1's bar, but its mask took in a row of the ground
# behind it and was ruled out by its size (LOCATE_RATIO_SLACK). With the
# arm's links taken as the robot's, the bar was then not located at all;
# before, the bar's mesh, stretched, had taken the arm's upper link. Held in
# the air, the bar stands 40 cm clear of the cart and more of the ground in
# depth.
#
# A depth region is one where neighbouring pixels differ in depth by
# MODEL_DEPTH_GAP_M at most (as connected_depth_foreground() joins them), or
# by this share of their depth: a surface seen edge-on steps further a row
# the further away it is, and the ground beyond the bench, 17 cm a row at
# 10 m, came out a region per row. A part standing clear of what is behind
# it comes out whole, one resting on the bench is part of the bench's region
# and left to SAM2. Regions smaller than LOCATE_DEPTH_MIN_PX are the depth's
# speckle. A SAM2 mask with LOCATE_DEPTH_SAME_IOU with a region is that
# region, but for its edge's spill onto what is behind -- SAM2's bar took in
# the row of ground under it, 3.2 m further -- and the region is taken in
# its place.
LOCATE_DEPTH_GAP_SHARE = 0.03
LOCATE_DEPTH_MIN_PX = 100
LOCATE_DEPTH_SAME_IOU = 0.8
#
# A proposal's area, in square metres at its depth, against the mesh's
# smallest and largest face at the ends of MODEL_SCALE_RANGE, these factors
# beyond: the bench, a wall, a speck are out.
LOCATE_AREA_SLACK = (0.25, 1.5)
# For a mesh of known scale, the size and area are those at it, this factor
# either way for depth noise and a mask's spill.
LOCATE_KNOWN_SCALE_SLACK = 1.25
# One view of a box shows two of its extents. The proposal's two longest
# must be in the proportions of one such pair of the mesh's, within what the
# fit may stretch it to: each axis MODEL_STRETCH_MAX either way. At 1.6 the
# simulated bar, 337 x 21 mm seen from one side, missed the mesh's 9.8 to 1
# by 1.61. The box is of the proposal's main depth component, as the fit
# takes it (connected_depth_foreground()): left in, the few pixels of SAM2's
# bar mask on the ground 3.2 m behind it made a box 988 x 334 mm.
LOCATE_RATIO_SLACK = MODEL_STRETCH_MAX ** 2
# A proposal further off than that may be the part with another in it: SAM2
# took RR-S1-1's bar with the ring on it as one, 265 x 56 mm, x2.05 off the
# bar's mesh, when a person's hand held the bar's end (2026-10-31); the bar
# fitted nothing else, and the ring was located on the hand. Proposals off by
# no more than this are fitted too, last, where nothing in the slack fits.
LOCATE_MIXED_SLACK = 3.0
# Such a proposal is fitted thinned too (thinned()): less where it is wider
# across than this many times the most of its length -- the ring on the bar,
# 19 px across against the bar's 9 -- where what is left is in the mesh's
# proportions.
LOCATE_THIN_WIDTH = 1.5
# A proposal within this many pixels of the image's edge is cut by it.
LOCATE_EDGE_CUT_PX = 2
# Points of a proposal's surface its box is fit to; min_obb() of every pixel
# of a large proposal costs more than the rest of the screening together.
LOCATE_BOX_POINTS = 2000
# The rough fit: mesh points, observed points and ICP rounds, at the
# observed box's axes only -- about 1 s a proposal against 10 s for the
# proper fit. At most LOCATE_SCREENS proposals, nearest the mesh's
# proportions first, keep the init within the client's allowance.
LOCATE_SCREEN = (500, 300, 4)
LOCATE_SCREENS = 24
LOCATE_FITS = 2
# A mesh sent with its colours (meta["models"][i]["colours"]) is also scored by
# them where it is located: of the pixels its samples are seen on the part at,
# the share whose colour is none of the mesh's -- this far in CIELAB's a*b*
# from every one of its samples' -- counts against the fit as contradicted depth
# does. Lightness is left out: shading and gloss change it, and a texture's
# lightness is another camera's. On YCB-V and LM-O test images the right part's
# pixels came out 4 % none of its colours (median; 30 % at the 90th percentile),
# another part's 34 %. A shape alike in size and form is so told apart by its
# colour, where its colours differ. Under this many pixels, colour says nothing.
LOCATE_COLOUR_DELTA = 15.0
LOCATE_COLOUR_MIN_POINTS = 30
# Scales the best proper fit at one scale is fitted again at; see _quick_fit().
LOCATE_SCALE_STEPS = 7
# A proposal this much on another object -- one located before, a clicked
# one's segment -- is that object; this much on the robot, the robot. Less
# on them, it is taken without their pixels.
LOCATE_TAKEN_SHARE = 0.5
# What is left of a proposal mostly on another object can be one more: SAM2
# took RR-S1-2's ring in with the bar it is on, and the ring was not located
# while the bar's pixels were in it. But what is left is mostly the other
# object's own -- the pixels its fitted mesh missed: RI-S4-1's ring was
# located on 132 px left of the bar's proposal (score 1.00, a mesh shrunk to
# it), PP-S1-3's sample on 335 px left of the tester's, each in place of its
# own proposal. So it is proposed only to an object no round locates
# otherwise, opened by LOCATE_REST_KERNEL so the slivers along the other's
# edge go, and only if LOCATE_REST_SHARE of the proposal is left: RR-S1-2's
# ring, 27% of the depth region it shared with the bar; those two, 4 and 14%.
LOCATE_REST_KERNEL = np.ones((5, 5), np.uint8)
LOCATE_REST_SHARE = 0.2
# Meshes of one reconstruction come out at one scale: the parts' scales agree
# -- 2.4 both of two, 0.85 and 0.75 of two others. Of meshes from anywhere else
# nothing follows, so this holds only where the client says so
# (meta["same_scale_meshes"]). Then a part located at a scale off the others' by
# more than this is located again among fits at theirs: PP-S4-1's tester,
# half behind the box, lost to a patch of the box's handle at 0.46, the
# sample at 2.41; at the sample's, its own proposal fitted at 2.21. One the
# others' scale finds none for keeps where it was. 1.5, not 2: RR-S1-1's ring,
# located on the hand of the person holding the bar, was at 1.23 to the bar's
# 0.75 at times (2026-10-31); parts of one reconstruction agree within 1.15.
LOCATE_SCALE_AGREE = 1.5
# A mesh's scale is its reconstruction's, the same from scene to scene: the
# ring measured 0.870-0.915 in RI-S1-1 to RR-S1-1. So the server keeps each
# mesh's measured shapes -- the scales along its box axes of fine fits whose
# size the view showed, not one an earlier view's gave
# (MODEL_KNOWN_SHAPE_MIN_GAIN), the last LOCATE_KNOWN_SCALES by the mesh's id,
# for as long as it runs; one that fitted its view better than theirs takes
# their place, not a median of the two that no view measured (the tester
# first seen turned over, 154 mm, then upright, 181: 167 mm, 8 mm off on the
# view turned over after). Every fit tries their median, axis by axis
# (fit_model_box()'s known_shape), and a part too hidden to measure its own
# size (MODEL_SCALE_MAX_HIDDEN) takes its overall size (fit_model_box()'s
# scale_hint) before the other parts' scale. Not
# to rule out where a part is located: a part half hidden is located at the
# scale of what shows -- PP-S4-2's tester, behind the box, at 1.83 of its
# 2.49 -- as far off as the box's latch was located as the ring (0.66 of
# 0.87) in RI-S4-2.
LOCATE_KNOWN_SCALES = 10
# A part located on a fit scored under LOCATE_WEAK is looked for again once
# the others are (locate_unclicked()), and moved where it fits better by
# LOCATE_BETTER. Parts seen whole fit 0.94-0.99; RR-S1-1's ring on the hand,
# 0.68, its own pixels 0.99.
LOCATE_WEAK = 0.8
# Where locating by mesh cannot be sure, the person is asked rather than a
# guess made: an object fitting no proposal, fitting its own under
# LOCATE_ASK_BELOW (once looked for again, LOCATE_WEAK), or fitting another place
# within LOCATE_BETTER as well (sharing no more than LOCATE_TAKEN_SHARE of the
# smaller region) is sent back unlocated with the init's reply (NeedClick,
# "need_click"), with where it fits best, for its click. Clicked, it is not
# looked for: its click is its segment. LOCATE_ASK_BELOW is set by the user's
# scenes, not derived: the tester, located right at 0.76-0.80 in PP-S1-3,
# PP-S1-4, PP-S3-1 and PP-S3-2, was asked for under LOCATE_WEAK to no gain
# (2026-11-04); a T-LESS part located wrong at 0.77, which a click had put
# right, is no longer asked for.
LOCATE_ASK_BELOW = 0.75
LOCATE_BETTER = 0.1
# Points per click, kept this far inside the visible surface's edge.
LOCATE_PROMPTS = 3
LOCATE_EDGE_PX = 4
# Where a located part's mesh is seen (mesh_click()'s region) falls short of
# what of it a proposal holds: RR-S1-2's ring, located on the bar, left its
# rim in the bar's proposal, points 3 cm off the bar, and the bar was not
# located (p95 41 mm); taken out grown 2 px, or the mesh sampled six times as
# densely, still not. So a located part is taken out of the proposals the
# others are located in grown LOCATE_REGION_GROW_PX: the bar was located then
# (0.65, 321 mm long), from what was left of its proposal -- unless so the
# proposal is mostly on it, or too little is left of it for what is left to
# be looked at (LOCATE_REST_SHARE), when it is taken out as it is: grown, the
# bar RR-S1-1's ring is on took the ring with it from the proposal they
# share. And out of the others' init masks (fit_boxes()), but
# never out of a located part's own region: grown into it, the bar's region
# cut the ring the bar runs through to two arcs, fitted at scale 3.5; not
# grown at all, the ring's mask kept the bar beside it and was not fitted.
# The masks tracked on later frames are cut so too (without_others()): cut by
# the regions alone, the ring's carried fit was off the part from the first
# frame on (p95 62 mm), the bar beside it in its mask.
LOCATE_REGION_GROW_PX = 4
# SAM2 can take two parts touching for one: on a live RR-S1-2 the ring's
# clicks gave a segment holding the bar, from the ring to the fist and past
# it (1371 px), and the bar's held the ring; the ring's fit failed on it (p95
# 210 mm), and the tracker gave the bar's pixels to the ring, the bar keeping
# 2 points. So a part's segment holding another part's click is the two of
# them: that click goes in as a negative point for it (separate_parts()), to
# the init's segments and to the tracker alike -- the ring's segment then 955
# px, the ring alone, fitted at 0.99, the bar's the bar without the ring.

# Obstacles: what else stands in the view, found with no click and no mesh.
# An init that asks for them (meta["discover"], how many at most) has them
# looked for among the same proposals the parts are located in (SAM2's and
# the depth's), once the parts are: each proposal, largest first, less the
# parts' pixels -- clicked, located -- and the robot's; less its support too
# (observe()), and cut into its depth-connected pieces (MODEL_DEPTH_GAP_M),
# each of OBSTACLE_MIN_PX or more a candidate. A candidate is an obstacle if
# it stands OBSTACLE_MIN_HEIGHT_M or more off its support (or has none: an
# arm in the air), measures OBSTACLE_EXTENT_M across, lies within
# OBSTACLE_MAX_DEPTH_M -- past it is the room -- and is not a surface: no
# more than OBSTACLE_FLAT_SHARE of it within PLANE_TOL_M of one plane where
# it is OBSTACLE_FLAT_M or more across (the bench, a wall; a box seen face
# on is less). One OBSTACLE_OVERLAP on an obstacle kept already is that one.
# Its box is that of its points.
#
# An obstacle is then followed by the depth, not tracked: tracked as a part,
# it took the point tracker's points as one -- RI-S1-1, its two parts beside
# two arms and a hand, went from 1.0 s a frame to 109 s on the sixth, the GPU
# full. On each frame (follow_obstacles()) it is a piece, cut as above, of
# what is near where it was (its last region's bounds, OBSTACLE_FOLLOW_PX
# round): less the robot, the parts' masks and the planes cut from it when it
# was found -- its support, the bench under a box; the camera stands still.
# Of those pieces the one most on its last region -- by intersection over
# union, OBSTACLE_FOLLOW_SHARE or more; moved off it, the share of the piece
# in it grown OBSTACLE_FOLLOW_PX, the piece within OBSTACLE_FOLLOW_SIZE of
# its size -- is where it is now, its box that piece's: a person's arm does
# not move as one body. Not found OBSTACLE_LOST_FRAMES frames running, it is
# lost.
OBSTACLE_MIN_PX = 300
OBSTACLE_MIN_HEIGHT_M = 0.02
OBSTACLE_EXTENT_M = (0.03, 1.5)
OBSTACLE_MAX_DEPTH_M = 2.0
OBSTACLE_FLAT_SHARE = 0.9
OBSTACLE_FLAT_M = 0.5
OBSTACLE_OVERLAP = 0.3
OBSTACLE_FOLLOW_PX = 40
OBSTACLE_FOLLOW_SHARE = 0.3
OBSTACLE_FOLLOW_SIZE = 3.0
OBSTACLE_LOST_FRAMES = 3
# Points a followed obstacle's box is fitted to, on every frame: min_obb() of
# LOCATE_BOX_POINTS took 48 ms an obstacle; a box to keep clear of needs less.
OBSTACLE_BOX_POINTS = 400
# A followed box must not turn where the obstacle does not. min_obb() gives
# its axes in no order and either way round, and of a draw of points another
# box each time: on one image replayed, RI-S4-2's box, 272 x 269 mm, turned
# 89 deg from frame to frame (16 frames of 29), the same box. So the points
# fitted are every n-th of the piece, not a draw -- the same piece, the same
# box -- and its axes are taken in the order and the way round (one of the 24
# _signed_permutations()) nearest the last box's. What turn is left is
# smoothed as the parts' poses are (PoseSmoother): a turn well inside
# OBSTACLE_SMOOTH_FULL_DEG moves the box OBSTACLE_SMOOTH_MIN_ALPHA of the way,
# a larger one all of it. Its centre and size are not smoothed: a box that
# lags an arm is one the arm has left.
OBSTACLE_SMOOTH_MIN_ALPHA = 0.3
OBSTACLE_SMOOTH_FULL_DEG = 20.0
# Two obstacles touching -- an arm and the hand at its end, millimetres apart
# in the depth -- are one piece on some frames and two on others, and the one
# followed first took both: under 2 mm of depth noise the arm's box came out
# 178 mm long, then 295, the hand lost on 26 frames of 30. So a pixel is
# looked at only for the obstacle whose last region is nearest it.


def _quick_fit(model, view, K, screen=False, through=None, image=None):
    """The mesh's pose on one proposal, for telling proposals apart: the
    coarse stage of _ModelFit.run(), its best start refined, and nothing of
    the scale measuring after (fit_boxes() fits the chosen one in full).
    Returns (score, (R, t, scale)) or None: a pose the coarse acceptance
    takes over one it does not, then its share of the observed points on
    the mesh less its share seen through or pressed into what is there.
    With `screen`, the rough fit that ranks proposals (LOCATE_SCREEN): the
    starts at the observed box's axes only, and no refining.

    The observed size is always among the starts. Nothing here measures the
    scale, and a mesh some 25 % off the part's size, within
    MODEL_SCALE_START_RATIO, fails the coarse acceptance at 1: the ring's
    69 mm mesh on the 60 mm simulated ring, standing face-on, overhung its
    rim onto the background behind (contradiction 0.37), and the ring was
    not located on its own mask; at the observed size, 0.14.

    Nor does the observed size always give the part's: half behind a box in
    RI-S4-2 the ring showed half its width, and at 1 it fitted with fitness
    0.99 and contradiction 0.26; RR-S1-2's bar, the ring on it in the same
    mask, came out 0.61 times the mesh's length and 1.09 its width. So the
    best pose at one scale is fitted again at scales about it, up to
    MODEL_SCALE_START_RATIO either way (LOCATE_SCALE_STEPS of them), and
    the best of all is taken. Where none of those passes the coarse
    acceptance, the best is fitted once more with its proportions free, as
    the full fit fits them: the bar's mesh, 386 x 39 mm against the part's
    340 x 27, fits it at no one scale -- at 0.77, contradiction 0.28. A mesh
    of known scale is fitted at it alone.
    `through`, pixels that join the view's pieces (_ModelFit's); `image`,
    the frame (RGB), for a mesh with colours (LOCATE_COLOUR_DELTA)."""
    fit = _ModelFit(model, view, K, np.random.default_rng(0), through=through, image=image)
    mesh_n, obs_n, rounds = LOCATE_SCREEN if screen else MODEL_COARSE
    starts = fit.start_scales()
    observed = float(np.clip(np.sort(fit.obb[2])[1] / np.sort(model.extent)[1], *MODEL_SCALE_RANGE))
    if model.known_scale is None and not any(np.ndim(s) == 0 and np.isclose(s, observed) for s in starts):
        starts.append(observed)
    best = uniform = None
    for s in starts:
        for R in fit.candidates(on_support=not screen):
            t = fit.place(s, R, mesh_n)
            R, t, terms = fit.icp(s, R, t, rounds, mesh_n, obs_n, MODEL_SEARCH_M)
            if terms is not None and (best is None or terms["cost"] < best[0]):
                best = (terms["cost"], R, t, s, terms)
            if terms is not None and np.ndim(s) == 0 and (uniform is None or terms["cost"] < uniform[0]):
                uniform = (terms["cost"], R, t, s, terms)
    if best is None:
        return None

    def scored(R, t, s, terms):
        return ((not model_pose_rejection(terms, coarse=True),
                 terms["fitness"] - terms["contradiction"] - terms["penetration"]
                 - fit.colour_contradiction(s, R, t)), (R, t, s))

    _, R, t, s, terms = best
    if screen:
        return scored(R, t, s, terms)
    R, t, terms = fit.icp(s, R, t, MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * fit.tol)
    found = [scored(R, t, s, terms)] if terms is not None else []
    if uniform is not None and model.known_scale is None:
        _, R, t, s, _ = uniform
        mean = model.points.mean(0)
        centre = R @ (s * mean) + t
        span = np.log(MODEL_SCALE_START_RATIO)
        for ratio in np.exp(np.linspace(-span, span, LOCATE_SCALE_STEPS)):
            si = float(np.clip(s * ratio, *MODEL_SCALE_RANGE))
            Ri, ti, termsi = fit.icp(si, R, centre - R @ (si * mean), MODEL_FINE_ROUNDS, MODEL_SAMPLES,
                                     MODEL_FINE_POINTS, 2 * fit.tol)
            if termsi is not None:
                found.append(scored(Ri, ti, si, termsi))
    if found and not max(found, key=lambda item: item[0])[0][0] and model.known_scale is None:
        _, (R, t, s) = max(found, key=lambda item: item[0])
        terms = fit.terms(s, R, t, MODEL_SAMPLES, MODEL_FINE_POINTS)
        tries = [fit.polish(s, R, t, terms, free_axes=True)] if terms is not None else []
        seen = fit.axes_seen(R, t, s)
        if seen.any():
            Ri, ti, termsi, si = fit.stretch(s, R, t, seen)
            if termsi is not None:
                tries.append(fit.polish(si, Ri, ti, termsi, free_axes=True))
        found += [scored(Ri, ti, si, termsi) for Ri, ti, termsi, si in tries if termsi is not None]
    return max(found, key=lambda item: item[0]) if found else None


def mesh_click(model, R, t, scale, metres, K):
    """The click a mesh at a pose gives: (points, region), the points inside
    the region of the image where the camera sees the mesh's surface, or
    None if it sees none of it."""
    B = model.box[:3, :3]
    M = (B * np.broadcast_to(np.asarray(scale, np.float64), (3,))) @ B.T  # _ModelFit.linear()
    P = model.points @ (R @ M).T + t
    spacing = _overall(scale) * model.spacing
    seen = _seen_from_camera(P, K, spacing, metres.shape)
    uv = np.round(_pixels(P[seen], K)).astype(np.int64)
    h, w = metres.shape
    u, v = np.clip(uv[:, 0], 0, w - 1), np.clip(uv[:, 1], 0, h - 1)
    d = metres[v, u]
    # Not where something nearer hides it: that is the occluder's to show.
    shown = np.isfinite(d) & (d > 0) & (np.abs(d - P[seen, 2]) <= MODEL_HIDDEN_M)
    if shown.sum() < MIN_BOX_POINTS:
        return None
    region = np.zeros((h, w), np.uint8)
    region[v[shown], u[shown]] = 1
    # Samples land a few pixels apart; close the gaps between them.
    gap = spacing * K[0, 0] / float(np.median(P[seen][shown, 2]))
    size = 2 * int(np.ceil(gap)) + 1
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    return region_clicks(region > 0), region > 0


def _grown(region):
    """A located part's region as other parts' proposals lose it (LOCATE_REGION_GROW_PX)."""
    kernel = np.ones((2 * LOCATE_REGION_GROW_PX + 1,) * 2, np.uint8)
    return cv2.dilate(region.astype(np.uint8), kernel) > 0


def region_clicks(region):
    """Up to LOCATE_PROMPTS points [[x, y], ...] inside `region` (bool), as a
    person would click it: the deepest, then each farthest from those picked,
    at least LOCATE_EDGE_PX in from the edge."""
    inside = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
    points = [np.unravel_index(np.argmax(inside), inside.shape)]
    deep = np.argwhere(inside >= LOCATE_EDGE_PX)
    # The rest spread over the part, each farthest from those picked.
    while len(points) < LOCATE_PROMPTS and len(deep):
        far = np.min(np.linalg.norm(deep[:, None] - np.asarray(points)[None], axis=2), axis=1)
        if far.max() < 2 * LOCATE_EDGE_PX:
            break
        points.append(tuple(deep[np.argmax(far)]))
    return [[int(c), int(r)] for r, c in points]


def depth_proposals(metres):
    """The depth image's regions as masks, of LOCATE_DEPTH_MIN_PX or more."""
    valid = np.isfinite(metres) & (metres > 0)
    count = int(valid.sum())
    if not count:
        return []
    z = np.where(valid, metres, 0.0)
    indices = np.full(metres.shape, -1, dtype=np.int32)
    indices[valid] = np.arange(count)
    sources, targets = [], []
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        gap = np.maximum(MODEL_DEPTH_GAP_M, LOCATE_DEPTH_GAP_SHARE * np.maximum(z[a], z[b]))
        linked = valid[a] & valid[b] & (np.abs(z[a] - z[b]) <= gap)
        sources.append(indices[a][linked])
        targets.append(indices[b][linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    regions = np.full(metres.shape, -1, dtype=np.int64)
    regions[valid] = labels
    return [regions == label for label in np.flatnonzero(np.bincount(labels) >= LOCATE_DEPTH_MIN_PX)]


def depth_pieces(mask, depth, min_px):
    """`mask`'s pieces of `min_px` or more, as connected_depth_foreground()
    links pixels (MODEL_DEPTH_GAP_M), largest first."""
    count = int(mask.sum())
    if count < min_px:
        return []
    indices = np.full(mask.shape, -1, dtype=np.int32)
    indices[mask] = np.arange(count)
    sources, targets = [], []
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        linked = mask[a] & mask[b] & (np.abs(depth[a] - depth[b]) <= MODEL_DEPTH_GAP_M)
        sources.append(indices[a][linked])
        targets.append(indices[b][linked])
    sources, targets = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(sources), dtype=np.uint8), (sources, targets)), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    out = []
    for label in np.argsort(-sizes):
        if sizes[label] < min_px:
            break
        piece = np.zeros_like(mask)
        piece[mask] = labels == label
        out.append(piece)
    return out


def others_region(index, regions):
    """The pixels of the regions of the objects other than `index`
    (`regions`, one per object, None where it has none), or None."""
    others = [region for j, region in enumerate(regions or ()) if j != index and region is not None]
    return np.any(others, axis=0) if others else None


def without_others(mask, index, regions):
    """Object `index`'s mask less the regions of the others (`regions`, one
    per object, None where it has none): a pixel one object's mesh is seen
    at is not another's -- the ring hung on a bar, in the bar's segment. With
    a region of its own, the others' are grown, as at init (fit_boxes(),
    LOCATE_REGION_GROW_PX), but not into its own."""
    others = others_region(index, regions)
    if others is None:
        return mask
    own = regions[index] if index < len(regions) else None
    if own is not None:
        others = _grown(others) & ~own
    return mask & ~others


def describe_proposal(mask):
    """Where a proposal is, for the server's log."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return "empty"
    return f"{len(ys)} px, u {xs.min()}-{xs.max()} v {ys.min()}-{ys.max()}"


def at_edge(mask, margin=LOCATE_EDGE_CUT_PX):
    """Whether `mask` reaches the image's edge: what it shows is cut there."""
    return bool(mask[:margin].any() or mask[-margin:].any() or mask[:, :margin].any() or mask[:, -margin:].any())


def thinned(mask):
    """`mask` less the stretches of it, across its length, wider than
    LOCATE_THIN_WIDTH times the most of it -- a bar less the ring on it, or
    the hand at its end -- and less what stands out of the band the rest
    makes. Its length is its pixels' main axis; how wide it is at each pixel
    along it, the spread of its pixels across."""
    v, u = np.nonzero(mask)
    if len(u) < LOCATE_DEPTH_MIN_PX:
        return mask
    pixels = np.column_stack([u, v]).astype(np.float64)
    centred = pixels - pixels.mean(0)
    axes = np.linalg.svd(centred, full_matrices=False)[2]
    along, across = centred @ axes[0], centred @ axes[1]
    bins, which = np.unique(np.floor(along).astype(np.int64), return_inverse=True)
    lo, hi = np.full(len(bins), np.inf), np.full(len(bins), -np.inf)
    np.minimum.at(lo, which, across)
    np.maximum.at(hi, which, across)
    width = hi - lo + 1
    typical = np.median(width)
    narrow = width <= LOCATE_THIN_WIDTH * typical
    if narrow.sum() < 2:
        return mask
    # Its middle line, from the narrow stretches; what stands out of the band
    # along it, in a narrow stretch too -- the ring's rim beside the bar -- goes.
    line = np.polyfit(bins[narrow], ((lo + hi) / 2)[narrow], 1)
    band = np.abs(across - np.polyval(line, along)) <= typical / 2 + 1
    out = np.zeros_like(mask, dtype=bool)
    out[v[narrow[which] & band], u[narrow[which] & band]] = True
    return out


class NeedClick(Exception):
    """An init locating by mesh could not finish for sure (LOCATE_ASK_BELOW):
    `requests`, per object to click, {"object": index, "why": str,
    "candidates": [click, ...]}, its likeliest places best first; `points`,
    every object's click as far as it is sure, [] for those."""

    def __init__(self, requests, points):
        self.requests, self.points = requests, points
        super().__init__("; ".join(f"object {r['object']}: {r['why']}" for r in requests) + "; click them")


def locate_with_mesh(model, proposals, metres, K, clicked=(), taken=None, robot=None, why=None,
                     rest=False, scales=None, grown=None, image=None):
    """Where in the view an object is, by its mesh: (click, region, note,
    (score, overall scale, scale)), the click as mesh_click() gives it and the
    model fitted, or (None, None, reason, None). With `scales`, a range of
    overall scales, a fit outside it is refused, and a rough one ranks last. `clicked` are pixels of other objects, `taken` the
    pixels of other objects -- those already located, those clicked --
    `robot` the robot's pixels: a proposal mostly on them is skipped, one
    partly on them is taken without them -- or, with `rest`, what is left of
    it, if LOCATE_REST_SHARE of it. `why`, a list, is given (index, what
    became of it) for each proposal. `grown`, `taken` with the located parts
    grown (LOCATE_REGION_GROW_PX), is what a proposal loses of them where it
    is not mostly on them so and, with `rest`, enough is left; else `taken`
    itself."""
    why = [] if why is None else why
    f = float(K[0, 0])
    m = np.sort(model.extent)[::-1]
    low, high = MODEL_SCALE_RANGE
    if model.known_scale is not None:
        low, high = model.known_scale / LOCATE_KNOWN_SCALE_SLACK, model.known_scale * LOCATE_KNOWN_SCALE_SLACK
    area_range = (m[1] * m[2] * low ** 2 * LOCATE_AREA_SLACK[0],
                  m[0] * m[1] * high ** 2 * LOCATE_AREA_SLACK[1])
    pairs = np.log([m[0] / m[1], m[0] / m[2], m[1] / m[2]])
    h, w = metres.shape
    clicked = [(min(max(int(round(v)), 0), h - 1), min(max(int(round(u)), 0), w - 1))
               for u, v in clicked]
    valid = np.isfinite(metres) & (metres > 0)
    shortlist, on_robot = [], 0
    # Off the mesh's proportions, but by no more than LOCATE_MIXED_SLACK: tried
    # last, should nothing else fit (_locate_fits()).
    mixed = []
    # A thinned proposal's pixels thinned away, by its index: they join what
    # they link of the rest (connected_depth_foreground()'s `through`). The ring
    # cuts the bar it is on in two; joined by neither depth nor an occluder
    # nearer, RR-S1-1's bar was measured and fitted on one piece, 158 mm of
    # 262, at 0.59 -- and the ring, looked for at that scale, not found
    # (2026-10-31).
    throughs = {}

    fitted = []

    def fit_among(candidates, said_too=""):
        return _locate_fits(model, K, candidates, scales, why, said_too, throughs, image, fitted)

    def measured(region, through=None):
        """(view, its two longest extents, longest first), or (None, why not)."""
        try:
            view = observe(region, metres, K)
        except Exception as exc:
            return None, f"not observed ({exc})"
        if view is None or view[2].sum() < MIN_BOX_POINTS:
            return None, "too few points off its support"
        cam = view[0]
        surface = cam[connected_depth_foreground(view[2], cam[..., 2], cam, through)]
        # Its own draw, so a proposal's box does not depend on those before it.
        surface = surface[np.random.default_rng(0).permutation(len(surface))[:LOCATE_BOX_POINTS]]
        return view, np.sort(min_obb(surface)[2])[::-1]
    for index, mask in enumerate(proposals):
        if any(mask[r, c] for r, c in clicked):
            why.append((index, "on another object's click"))
            continue
        if taken is not None:
            # The bar through a ring's hole, in the ring's proposal, is the bar's.
            left = None
            for lose in ([grown, taken] if grown is not None else [taken]):
                share = (mask & lose).sum() / mask.sum()
                left = mask & ~lose
                if share > LOCATE_TAKEN_SHARE:
                    if rest:
                        left = cv2.morphologyEx(left.astype(np.uint8), cv2.MORPH_OPEN, LOCATE_REST_KERNEL) > 0
                    if not rest or left.sum() < max(LOCATE_DEPTH_MIN_PX, LOCATE_REST_SHARE * proposals[index].sum()):
                        left = None
                        continue
                break
            if left is None:
                why.append((index, f"{share:.0%} on another object"))
                continue
            mask = left
        if robot is not None:
            if (mask & robot).sum() > LOCATE_TAKEN_SHARE * mask.sum():
                on_robot += 1
                why.append((index, f"{(mask & robot).sum() / mask.sum():.0%} on the robot"))
                continue
            # What of the robot it does take in is still not the part.
            mask = mask & ~robot
        z = metres[mask & valid]
        if len(z) < MIN_BOX_POINTS:
            why.append((index, f"{len(z)} depth points"))
            continue
        area = mask.sum() * (np.median(z) / f) ** 2
        if not area_range[0] <= area <= area_range[1]:
            why.append((index, f"{area * 1e4:.3g} cm2, not {area_range[0] * 1e4:.2g}-"
                               f"{area_range[1] * 1e4:.0f}"))
            continue
        view, e = measured(mask)
        if view is None:
            why.append((index, e))
            continue
        if not low * m[2] <= e[0] <= high * m[0]:
            why.append((index, f"{e[0] * 1000:.0f} mm long, not {low * m[2] * 1000:.0f}-"
                               f"{high * m[0] * 1000:.0f}"))
            continue
        miss = float(np.min(np.abs(np.log(e[0] / max(e[1], 1e-4)) - pairs)))
        if miss > np.log(LOCATE_RATIO_SLACK):
            said = f"{e[0] * 1000:.0f} x {e[1] * 1000:.0f} mm, proportions x{np.exp(miss):.2f} off the mesh's"
            if miss <= np.log(LOCATE_MIXED_SLACK):
                # Less what stands much wider than most of its length (thinned()):
                # of its main depth component, not of SAM2's mask, whose edge takes
                # in background a pixel round -- left in, the bench behind made the
                # thinned bar's box 666 x 285 mm.
                cam = view[0]
                whole = connected_depth_foreground(view[2], cam[..., 2], cam)
                thin = thinned(whole)
                cut = whole & ~thin
                thin_view, thin_e = measured(thin, cut) if thin.sum() < mask.sum() else (None, None)
                if thin_view is not None:
                    thin_miss = float(np.min(np.abs(np.log(thin_e[0] / max(thin_e[1], 1e-4)) - pairs)))
                    if thin_miss <= np.log(LOCATE_MIXED_SLACK):
                        said += (f"; thinned to {thin_e[0] * 1000:.0f} x {thin_e[1] * 1000:.0f} mm "
                                 f"({thin.sum()} px)")
                        miss, view = thin_miss, thin_view
                        throughs[index] = cut
                mixed.append((miss, index, view))
            why.append((index, said))
            continue
        shortlist.append((miss, index, view))
    best, tried, screened, chosen = fit_among(shortlist)
    views = {index: view for _, index, view in shortlist}
    cut = chosen is not None and at_edge(views[chosen][1])
    if (best is None or not best[0][0] or cut) and mixed:
        # None fits, or only on what the image's edge cuts, whose size and
        # proportions are not seen: the part may be in a proposal with another
        # (LOCATE_MIXED_SLACK). RR-S1-1's bar, the ring on it in its proposal,
        # was located on the arm of the person holding it, at the image's edge.
        found = fit_among(mixed, " (with another part in it)")
        if found[0] is not None and found[0][0][0] and (best is None or not best[0][0] or found[0][0] > best[0]):
            best, tried, screened, chosen = found[0], tried + found[1], screened + found[2], found[3]
            shortlist = shortlist + mixed
    if not shortlist and not mixed:
        return None, None, (f"none of {len(proposals)} proposals ({on_robot} on the robot) "
                            f"is the mesh's size and shape"), None
    if best is None or not best[0][0]:
        return None, None, (f"the mesh fits none of {tried} likely proposals "
                            f"({on_robot} on the robot)"), None
    (_, score), pose = best
    found = mesh_click(model, *pose, metres, K)
    if found is None:
        return None, None, "the fitted mesh shows no surface in the view", None
    # Other places it fits about as well (LOCATE_TAKEN_SHARE): their clicks.
    rivals = []
    for (accepted, other), other_pose in fitted:
        if other_pose is pose or not accepted or other < score - LOCATE_BETTER:
            continue
        there = mesh_click(model, *other_pose, metres, K)
        if there is None:
            continue
        shared = (there[1] & found[1]).sum() / max(1, min(there[1].sum(), found[1].sum()))
        if shared <= LOCATE_TAKEN_SHARE and all(
                (there[1] & r[1]).sum() / max(1, min(there[1].sum(), r[1].sum())) <= LOCATE_TAKEN_SHARE
                for r in rivals):
            rivals.append((there[0], there[1], other))
    note = (f"score {score:.2f}, scale {_overall(pose[2]):.2f}, best of {tried} fitted of "
            f"{len(screened)} screened of {len(shortlist)} shortlisted of {len(proposals)} proposals, "
            f"{on_robot} on the robot")
    if rivals:
        note += f"; {len(rivals)} other place{'s' if len(rivals) > 1 else ''} fit within {LOCATE_BETTER:.2f}"
    return found[0], found[1], note, (score, _overall(pose[2]), pose[2], [(c, s) for c, _, s in rivals])


def _locate_fits(model, K, candidates, scales, why, said_too="", throughs=None, image=None, fitted=None):
    """(best fit or None, proposals fitted, screened, the best's proposal) of
    `candidates` [(miss, index, view)] (locate_with_mesh()): rough fits rank
    them, the best LOCATE_FITS are fitted properly; `why` is told of each.
    `throughs`, by index, what joins a candidate's pieces (_quick_fit());
    `fitted`, a list, is given each proper fit within `scales`."""
    throughs = throughs or {}
    screened, chosen = [], None
    for rank, (_, index, view) in enumerate(sorted(candidates, key=lambda item: item[0])):
        if rank >= LOCATE_SCREENS:
            why.append((index, "shortlisted, past the screening cap"))
            continue
        try:
            found = _quick_fit(model, view, K, screen=True, through=throughs.get(index), image=image)
        except Exception:
            traceback.print_exc()
            found = None
        if found is None:
            why.append((index, "shortlisted, no rough fit"))
            continue
        within = scales is None or scales[0] <= _overall(found[1][2]) <= scales[1]
        screened.append(((within,) + found[0], index, view))
    screened.sort(key=lambda item: item[0], reverse=True)
    best, tried = None, 0
    for rank, ((_, accepted, rough), index, view) in enumerate(screened):
        said = f"rough fit {rough:.2f}" + ("" if accepted else " refused")
        if rank >= LOCATE_FITS:
            why.append((index, said))
            continue
        tried += 1
        try:
            found = _quick_fit(model, view, K, through=throughs.get(index), image=image)
        except Exception:
            traceback.print_exc()
            found = None
        if found is None:
            why.append((index, f"{said}, no fit"))
            continue
        overall = _overall(found[1][2])
        if scales is not None and not scales[0] <= overall <= scales[1]:
            why.append((index, f"{said}, fit {found[0][1]:.2f} at scale {overall:.2f}, "
                               f"not {scales[0]:.2f}-{scales[1]:.2f}"))
            continue
        why.append((index, f"{said}, fit {found[0][1]:.2f}" + ("" if found[0][0] else " refused") + said_too))
        if fitted is not None:
            fitted.append(found)
        if best is None or found[0] > best[0]:
            best, chosen = found, index
    return best, tried, screened, chosen if best is not None else None


def decode_models(meta, parts, count):
    """One ObjectModel or None per object, from the frames after rgb and depth.

    meta["models"] holds, per object, None or the vertex and face counts of
    one frame: zlib of little-endian float32 vertices then int32 faces, then
    with "colours" an RGB byte triple per vertex (LOCATE_COLOUR_DELTA) -- and,
    where the mesh's scale to the part is known, that ("known_scale"). A
    model that cannot be used returns None. The caller retains which objects
    requested a mesh so a decode failure cannot enable point-cloud fitting.
    """
    specs = meta.get("models") or []
    models = [None] * count
    frames = iter(parts)
    for index, spec in enumerate(specs[:count]):
        if spec is None:
            continue
        buf = next(frames, None)
        try:
            if buf is None:
                raise ValueError("no frame")
            raw = zlib.decompress(buf)
            nv, nf = int(spec["vertices"]), int(spec["faces"])
            coloured = bool(spec.get("colours"))
            if nv <= 0 or nf <= 0 or len(raw) != 12 * (nv + nf) + (3 * nv if coloured else 0):
                raise ValueError("size does not match its counts")
            vertices = np.frombuffer(raw, "<f4", 3 * nv).reshape(nv, 3)
            faces = np.frombuffer(raw, "<i4", 3 * nf, offset=12 * nv).reshape(nf, 3)
            colours = np.frombuffer(raw, np.uint8, 3 * nv, offset=12 * (nv + nf)).reshape(nv, 3) if coloured else None
            known = spec.get("known_scale")
            if known is not None and not (np.isfinite(known) and MODEL_SCALE_RANGE[0] <= known <= MODEL_SCALE_RANGE[1]):
                raise ValueError(f"known scale {known} outside {MODEL_SCALE_RANGE}")
            models[index] = ObjectModel(vertices, faces, colours=colours)
            models[index].known_scale = None if known is None else float(known)
            size = " x ".join(f"{v * 1000:.0f}" for v in models[index].extent)
            print(f"[server] object {index}: mesh {nv} vertices, box {size} mm"
                  + ("" if known is None else f", scale known: {float(known):.3f}")
                  + (", with colours" if coloured else ""), flush=True)
        except Exception as exc:
            print(f"[server] object {index}: unusable mesh ({exc}); recognition rejected",
                  flush=True)
    return models


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
    def __init__(self, cfg_path, bind, archive=None, trace_gpu=False):
        # Where each live GPU block was allocated, kept by torch (GPU_HELD_WARN_MIB).
        self.trace_gpu = bool(trace_gpu) and self.start_gpu_trace()
        self.gpu_after_init = None
        self.gpu_growth_said = 0
        # Where bindings settled file what was seen of their parts; see BINDING_FEATURE_PX.
        self.archive_dir = archive
        self.bindings = []
        self.pending_binds = {}
        self.cfg = OmegaConf.load(cfg_path)
        if self.cfg.pipeline.type != "modular":
            raise ValueError(f"expected modular pipeline, got {self.cfg.pipeline.type}")
        self.cfg.pipeline.params.save_pose = False
        self.cfg.pipeline.params.save_meta_data = False

        self.pipeline = None
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.fitted_models = None
        self.model_required = None
        self.model_errors = None
        self.bbox_warned = False
        self.mesh_fit_executor = None
        self.mesh_fit_job = None
        self.mesh_session = 0
        self.surfaces = []
        self.smoother = PoseSmoother()
        self.init_report = None
        # Each mesh's measured scales, by its id, across inits; see LOCATE_KNOWN_SCALES.
        self.known_scales = {}

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind)
        print(f"[server] listening on {bind}", flush=True)

    def release_pipeline(self):
        """Drop the pipeline and its GPU memory before another is built: built
        first, the new one's models load beside the old ones, and an 8 GB card
        then spills into system memory, where a frame takes tens of seconds."""
        self.pipeline = None
        # A fit still running belongs to the old session; stop waiting on it.
        self.mesh_fit_job = None
        gc.collect()
        # The SAM2 segmenter enters a bf16 autocast at import and never leaves
        # it, so autocast's cache of cast weights is never cleared and keeps
        # every released SAM2's bf16 copy (~430 MiB each) alive.
        torch.clear_autocast_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def start_gpu_trace(self):
        """Have torch's allocator keep the Python stack of each live block
        (GPU_HELD_WARN_MIB): those alive now only, no history to bound."""
        if not torch.cuda.is_available():
            print("[server] --trace-gpu-memory: no CUDA device, nothing to trace", flush=True)
            return False
        torch.cuda.memory._record_memory_history(enabled="state", context="state", stacks="python")
        print("[server] tracing GPU allocations (Python stacks of live blocks)", flush=True)
        return True

    def report_gpu_held(self, when):
        """Say when more than GPU_HELD_WARN_MIB is still allocated `when` (a
        release): what an earlier session left behind; traced, what holds it."""
        if not torch.cuda.is_available():
            return
        held = torch.cuda.memory_allocated() / 2**20
        if held <= GPU_HELD_WARN_MIB:
            return
        print(f"[server] WARNING: {held:.0f} MiB of GPU memory still allocated {when}, held from an earlier "
              "session; a fresh server holds next to none, and past the card's memory every step spills "
              "into system memory and slows"
              + ("" if self.trace_gpu else " -- restart the server, or run it with --trace-gpu-memory "
                 "to see what holds it"), flush=True)
        if self.trace_gpu:
            self.say_gpu_sites(f"held {when}")
            self.say_gpu_holders()

    def say_gpu_sites(self, what):
        try:
            total, sites = gpu_alloc_sites()
            print(f"[server] GPU {what}: {total:.0f} MiB in live blocks, by where allocated:", flush=True)
            for size, site in sites:
                print(f"[server]   {size:8.1f} MiB  {site}", flush=True)
        except Exception:
            traceback.print_exc()

    def say_gpu_holders(self):
        try:
            reached, graphed, holders = gpu_python_holders(self)
            held = torch.cuda.memory_allocated() / 2**20
            print(f"[server] GPU held: {reached:.0f} MiB reached from Python objects ({graphed:.0f} MiB of it in "
                  f"tensors made with grad on, an autograd graph's), {max(held - reached, 0):.0f} "
                  "MiB not (held from C++, such as a thread's autocast cache); threads alive: "
                  + ", ".join(t.name for t in threading.enumerate()) + "; the Python part by what holds it:",
                  flush=True)
            for size, holder in holders:
                print(f"[server]   {size:8.1f} MiB  {holder}", flush=True)
        except Exception:
            traceback.print_exc()

    def trace_gpu_frame(self, frame_id):
        """Traced, every GPU_TRACE_EVERY_FRAMES track frames: the memory, and
        where it was allocated each time it has grown another
        GPU_HELD_WARN_MIB over the init's."""
        if not self.trace_gpu or frame_id % GPU_TRACE_EVERY_FRAMES:
            return
        self.log_gpu_memory(f"on frame {frame_id}")
        if self.gpu_after_init is None:
            return
        steps = int((torch.cuda.memory_allocated() / 2**20 - self.gpu_after_init) // GPU_HELD_WARN_MIB)
        if steps > self.gpu_growth_said:
            self.gpu_growth_said = steps
            self.say_gpu_sites(f"grown {steps * GPU_HELD_WARN_MIB} MiB or more since the init, on frame {frame_id}")

    def log_gpu_memory(self, when):
        if torch.cuda.is_available():
            print(f"[server] GPU memory {when}: "
                  f"allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
                  f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)

    def build_pipeline(self):
        self.archive_session()
        self.release_pipeline()
        self.log_gpu_memory("after release")
        self.report_gpu_held("after release")
        self.pipeline = ModularPipeline(self.cfg)
        self.log_gpu_memory("after build")
        self.bindings, self.pending_binds = [], {}
        self.surfaces = []
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.fitted_models = None
        self.model_required = None
        self.model_errors = None
        self.bbox_warned = False
        self.obstacles = []
        self.smoother.reset()

    def fit_boxes(self, rgb, metres, K, points, labels, models, model_required=None, located=None,
                  robot=None):
        """Each object's box on the init frame, fit to its mask; None where it cannot be.

        A requested mesh gives the box when its pose fits the view, finely or
        coarsely. When it does not, the object is still tracked, by the box of
        its points, and the mesh is retried on later frames; the reply then
        carries no mesh pose for it, so the Agent knows it has only the box.

        Each mask is taken without the regions other objects were located in
        (`located`, {index: region}) and the robot's pixels (`robot`): the
        ring hung on RR-S1-1's bar was in the bar's, and its fit refused (95th
        percentile 15.7 mm off); the obstacle beside the arm and cart took them
        in, and its box came out 730 mm long of 284.
        """
        self.mesh_session = getattr(self, "mesh_session", 0) + 1
        self.models = list(models)
        self.fitted_models = [None] * len(points)
        self.model_required = (list(model_required) if model_required is not None
                               else [model is not None for model in models])
        self.model_errors = [None] * len(points)
        self.model_retry_after = [time.monotonic() + MODEL_RETRY_INTERVAL_SEC] * len(points)
        self.model_check_fails = [0] * len(points)
        self.model_withheld = [False] * len(points)
        self.model_corrections = [None] * len(points)
        self.sent_model_poses = [None] * len(points)          # see held_turn()
        self.bindings = [None] * len(points)                  # see BINDING_CONFIDENCE
        # Where each fit was last on the part, as its motion then, whether it
        # is held there now, and the robot's poses of the parts it set down
        # (see MODEL_CHECK_FAILS).
        self.still_motions = [None] * len(points)
        self.holding = [False] * len(points)
        self.pose_priors, self.last_fitted = {}, {}
        # Each part's hint: the other located parts' scale, where they share
        # one (LOCATE_SCALE_AGREE), for a part too hidden to measure its own
        # (MODEL_SCALE_MAX_HIDDEN).
        # Its own measured scale first (LOCATE_KNOWN_SCALES), and its shape
        # tried on every view (MODEL_KNOWN_SHAPE_MIN_GAIN).
        scales = getattr(self, "locate_scales", {}) if getattr(self, "same_scale", False) else {}
        self.scale_hints = []
        self.shape_hints = [self.known_shape(model) for model in models]
        for index in range(len(points)):
            hint = self.known_scale(models[index])
            others = [s for j, s in scales.items() if j != index]
            if hint is None and others:
                hint = float(np.exp(np.median(np.log(others))))
            self.scale_hints.append(hint)
        try:
            # Same call as a preview, made on the freshly built pipeline before
            # add_user_points(), which is the state a preview sees.
            logits = self.pipeline.preview_user_masks(rgb, points, labels)
            masks = (as_numpy(logits) > 0).reshape(len(points), *metres.shape).copy()
        except Exception:
            self.say("no init masks; mesh objects rejected, no boxes")
            self.model_errors = ["no usable initial mask" if required else None
                                 for required in self.model_required]
            traceback.print_exc()
            return [None] * len(points)
        located = located or {}
        cut = [others_region(index, [located.get(j) for j in range(len(masks))]) for index in range(len(masks))]
        for index in range(len(masks)):
            if cut[index] is not None:
                # Grown as when located (LOCATE_REGION_GROW_PX), but never
                # into the part's own region: the ring the bar runs through.
                if located.get(index) is not None:
                    cut[index] = _grown(cut[index]) & ~located[index]
                masks[index] &= ~cut[index]
            if robot is not None:
                masks[index] &= ~robot
        boxes = []
        for index, mask in enumerate(masks):
            fit = None
            try:
                view = observe(mask, metres, K)
            except Exception:
                traceback.print_exc()
                view = None
            if self.model_required[index] and models[index] is None:
                self.model_errors[index] = "supplied mesh could not be loaded"
                self.say(f"object {index}: supplied mesh could not be loaded; tracking its points' box")
                if view is not None:
                    try:
                        fit = fit_box(mask, metres, K, view=view)
                    except Exception:
                        traceback.print_exc()
            elif self.model_required[index]:
                started = time.time()
                reason = "no usable initial depth observation"
                if view is not None:
                    try:
                        fit, reason = fit_model_box(models[index], view, K, through=cut[index],
                                                    scale_hint=self.scale_hints[index],
                                                    located=getattr(self, "locate_shapes", {}).get(index),
                                                    image=rgb, known_shape=self.shape_hints[index])
                    except Exception as exc:
                        traceback.print_exc()
                        reason = f"{type(exc).__name__}: {exc}"
                self.model_retry_after[index] = time.monotonic() + MODEL_RETRY_INTERVAL_SEC
                if fit is None:
                    self.model_errors[index] = reason
                    self.say(f"object {index}: mesh pose unlocalized in {time.time() - started:.1f} s "
                             f"({reason}); tracking its points' box, retrying the mesh on fresh frames")
                    if view is not None:
                        try:
                            fit = fit_box(mask, metres, K, view=view)
                        except Exception:
                            traceback.print_exc()
                else:
                    self.fitted_models[index] = models[index]
                    self.remember_scale(models[index])
                    self.say(f"object {index}: mesh fit in {time.time() - started:.1f} s")
            elif view is not None:
                try:
                    fit = fit_box(mask, metres, K, view=view)
                except Exception:
                    traceback.print_exc()
            if fit is None:
                self.say(f"object {index}: no box")
                boxes.append(None)
                continue
            T, extent, note = fit
            size = " x ".join(f"{v * 1000:.0f}" for v in extent)
            self.say(f"object {index}: box {size} mm ({note})")
            boxes.append((T, extent))
        return boxes

    def retry_pending_meshes(self, motions, frame, regions=None):
        """Poll one background fit and schedule an unlocalized object's fresh view.

        The pipeline supplies current SAM masks in frame.mask after step().
        No preview with the original click coordinates is used here. The
        worker's result is expressed in its capture camera frame; convert it
        to the tracker's reference before propagating to newer frames.
        """
        now = time.monotonic()
        tracking_lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        job = getattr(self, "mesh_fit_job", None)
        if job is not None:
            future, index, session, capture_inverse, fitted_on = job
            if session == self.mesh_session and tracking_lost[index]:
                # A loss at any point during fitting breaks motion propagation,
                # even if feature tracking recovers before the worker finishes.
                capture_inverse = None
                self.mesh_fit_job = (future, index, session, None, fitted_on)
            if not future.done():
                return
            self.mesh_fit_job = None
            if session == self.mesh_session:
                self.model_retry_after[index] = now + (MODEL_RETRY_INTERVAL_SEC
                                                       if self.fitted_models[index] is None
                                                       else MODEL_UPGRADE_INTERVAL_SEC)
                try:
                    fit, reason, model = future.result()
                    current = self.fitted_models[index]
                    unchecked = ""
                    if fit is not None and (current is None or fit_rank(model) > fit_rank(current)):
                        # Sent only once on the part on this frame (MODEL_CHECK_FAILS).
                        fitted_in_camera = model.fitted_pose
                        here, reason = self.adopt_refit(index, model, frame, regions, motions, capture_inverse,
                                                        fitted_on)
                        if here is None:
                            fit, unchecked = None, reason
                    if fit is not None and (current is None or fit_rank(model) > fit_rank(current)):
                        T_box, extent, note = fit
                        model.fitted_pose = here
                        turn = None if current is None else refit_turn(current, model.fitted_pose)
                        upgrade = turn is not None and fit_rank(model)[0] and not fit_rank(current)[0]
                        if upgrade:
                            upgrade, why = self.preserve_refit_direction(index, current, model, frame,
                                                                         regions, motions)
                            self.say(f"object {index}: refit direction check: {why}")
                        if turn is not None and not upgrade:
                            # The fit stays, the refit in doubt: see MODEL_REFIT_TURN_DEG.
                            setattr(current, f"fitted_{turn}_doubt",
                                    np.linalg.inv(current.fitted_pose) @ model.fitted_pose)
                            print(f"[server] object {index}: refit better but turned "
                                  f"{'half round about its length' if turn == 'spin' else 'end for end'}; "
                                  f"keeping the fit, the refit sent in doubt ({note})", flush=True)
                        else:
                            if upgrade:
                                # The refined shape was checked at a different
                                # pose; rebuild its box in that same frame.
                                B = model.box[:3, :3]
                                T_box = model.fitted_pose.copy()
                                T_box[:3, :3] = model.fitted_pose[:3, :3] @ B
                                T_box[:3, 3] += model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3])
                                extent = fitted_axis_scale(model) * model.extent
                                self.boxes[index] = (T_box, extent)
                            else:
                                to_here = here @ np.linalg.inv(fitted_in_camera)
                                self.boxes[index] = (to_here @ T_box, extent)
                            self.fitted_models[index] = model
                            self.object_state(len(self.models))
                            self.still_motions[index] = np.asarray(motions[index], dtype=np.float64).copy()
                            self.remember_scale(model)
                            self.model_errors[index] = None
                            print(f"[server] object {index}: mesh "
                                  f"{'localized' if current is None else 'fit improved'} on retry"
                                  + (", fine over coarse in the published direction; "
                                     "the refit's other direction kept in doubt" if upgrade else "")
                                  + f" ({note})", flush=True)
                    elif current is not None:
                        print(f"[server] object {index}: "
                              + (f"refit not taken ({unchecked})" if unchecked else "refit no better")
                              + "; keeping the fit", flush=True)
                    else:
                        self.model_errors[index] = reason
                        print(f"[server] object {index}: mesh pose unlocalized ({reason}); will retry", flush=True)
                except Exception as exc:
                    self.model_errors[index] = f"mesh retry failed: {type(exc).__name__}: {exc}"
                    traceback.print_exc()

        eligible = [i for i, model in enumerate(self.models)
                    if self.model_required[i] and model is not None
                    and (self.fitted_models[i] is None or fit_rank(self.fitted_models[i]) < (True, True))
                    and not tracking_lost[i] and now >= self.model_retry_after[i]]
        if not eligible:
            return
        # Oldest due first, so repeated failures do not starve another object.
        index = min(eligible, key=lambda i: (self.fitted_models[i] is not None, self.model_retry_after[i]))
        self.model_retry_after[index] = now + (MODEL_RETRY_INTERVAL_SEC if self.fitted_models[index] is None
                                               else MODEL_UPGRADE_INTERVAL_SEC)
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(self.models), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(self.models), h, w) or not np.isfinite(masks[index]).all():
                raise ValueError("no usable current segmentation mask")
            if not np.isfinite(frame.depth_factor) or frame.depth_factor <= 0:
                raise ValueError("invalid current depth factor")
            mask = self.without_robot(without_others(masks[index] > 0, index, regions)).copy()
            metres = as_numpy(frame.depth) / frame.depth_factor
            if np.count_nonzero(mask & np.isfinite(metres) & (metres > 0)) < MIN_BOX_POINTS:
                raise ValueError("insufficient current mask/depth pixels")
            motion = np.asarray(motions[index], dtype=float)
            if (motion.shape != (4, 4) or not np.isfinite(motion).all()
                    or not np.allclose(motion[3], [0, 0, 0, 1], atol=1e-6)
                    or not np.allclose(motion[:3, :3].T @ motion[:3, :3], np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(motion[:3, :3]), 1, atol=1e-5)):
                raise ValueError("no usable current tracking transform")
            capture_inverse = np.linalg.inv(motion)
            if getattr(self, "mesh_fit_executor", None) is None:
                self.mesh_fit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh_pose",
                                                            initializer=no_grad_thread)
            future = self.mesh_fit_executor.submit(
                retry_mesh_fit, self.models[index], mask, metres, np.array(frame.intrinsics, copy=True),
                others_region(index, regions), self.scale_hints[index],
                getattr(self, "locate_shapes", {}).get(index), np.array(frame.rgb, copy=True),
                self.known_shape(self.models[index]))
            self.mesh_fit_job = (future, index, self.mesh_session, capture_inverse, frame.id)
            print(f"[server] object {index}: retrying mesh pose on frame {frame.id}", flush=True)
        except Exception as exc:
            reason = f"mesh retry waiting: {exc}"
            if self.fitted_models[index] is None:
                if self.model_errors[index] != reason:
                    print(f"[server] object {index}: {reason}", flush=True)
                self.model_errors[index] = reason

    def carried_regions(self, motions, frame):
        """Where the camera sees each fitted mesh in this frame, carried there
        by its tracked motion (mesh_click()'s region), one per object; None
        for one with no fit, lost, or out of view; None for all without a
        pipeline to ask. For without_others()."""
        models = getattr(self, "fitted_models", None)
        if not models:
            return None
        try:
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return None
        out = []
        for index, model in enumerate(models):
            region = None
            if model is not None and model.fitted_pose is not None and index < len(lost) and not lost[index]:
                try:
                    pose = np.asarray(motions[index], dtype=np.float64) @ model.fitted_pose
                    linear = getattr(model, "fitted_linear", None)
                    linear = linear if linear is not None else model.fitted_scale * np.eye(3)
                    B = model.box[:3, :3]
                    found = mesh_click(model, pose[:3, :3], pose[:3, 3],
                                       np.diag(B.T @ np.asarray(linear, np.float64) @ B), metres, K)
                    region = None if found is None else found[1]
                except Exception:
                    region = None
            out.append(region)
        return out

    def check_carried_meshes(self, motions, frame, regions=None):
        """Correct each kept mesh fit, carried into this frame by its tracked
        motion, on this frame's mask and depth (correct_carried_pose()), and
        score it there: off the part, it is withheld from this frame's reply
        (model_withheld); off it MODEL_CHECK_FAILS frames running, it is
        dropped and fitted again now. Returns the motions, each fitted
        object's carrying its corrected mesh pose."""
        motions = np.array(motions, dtype=np.float64)
        models = getattr(self, "fitted_models", None)
        if not models:
            return motions
        withheld = getattr(self, "model_withheld", None)
        if withheld is None or len(withheld) != len(models):
            withheld = self.model_withheld = [False] * len(models)
        withheld[:] = [False] * len(models)
        corrections = getattr(self, "model_corrections", None)
        if corrections is None or len(corrections) != len(models):
            corrections = self.model_corrections = [None] * len(models)
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(models), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(models), h, w):
                return motions
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return motions
        fails = getattr(self, "model_check_fails", None)
        if fails is None or len(fails) != len(models):
            fails = self.model_check_fails = [0] * len(models)
        self.object_state(len(models))
        for index, model in enumerate(models):
            if model is not None and model.fitted_pose is not None and lost[index]:
                self.lost_where_held(index, model, metres, K, frame.id)
            elif hasattr(self, "lost_said"):
                self.lost_said.discard(index)
            if model is None or model.fitted_pose is None or lost[index]:
                fails[index] = 0
                corrections[index] = None
                continue
            try:
                motion = motions[index]
                if motion.shape != (4, 4) or not np.isfinite(motion).all():
                    continue
                carried = motion @ model.fitted_pose
                fit_id = getattr(model, "fitted_id", None)
                kept = corrections[index]
                start = carried @ kept[1] if kept is not None and kept[0] == fit_id else carried
                mask = self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions))
                fit = carried_view(model, mask, metres, K, others_region(index, regions))
                pose, held_here = start, False
                if fit is not None:
                    pose, cost = correct_carried_pose(fit, model, start)
                    move, turn = correction_size(model, carried, pose)
                    if move > MODEL_CORRECT_MAX_M or turn > MODEL_CORRECT_MAX_DEG:
                        tracked, tracked_cost = correct_carried_pose(fit, model, carried)
                        if not cost < tracked_cost - MODEL_CORRECT_MIN_GAIN:
                            pose, cost = tracked, tracked_cost
                    # Where the motion and where it was last seen disagree by
                    # more than the correction moves it, the cheaper (MODEL_CHECK_FAILS).
                    still = self.still_motions[index]
                    if still is not None:
                        last = still @ model.fitted_pose
                        move, turn = correction_size(model, pose, last)
                        if move > MODEL_CORRECT_MAX_M or turn > MODEL_CORRECT_MAX_DEG or self.holding[index]:
                            held, held_cost = correct_carried_pose(fit, model, last)
                            # Held, the motion must fit better to carry it again.
                            better = (held_cost <= cost + MODEL_CORRECT_MIN_GAIN if self.holding[index]
                                      else held_cost < cost - MODEL_CORRECT_MIN_GAIN)
                            if better and not carried_fit_rejection(model, held, mask, metres, K, fit=fit):
                                pose, cost, held_here = held, held_cost, True
                corrections[index] = (fit_id, np.linalg.inv(carried) @ pose)
                why = carried_fit_rejection(model, pose, mask, metres, K, fit=fit) if fit is not None else ""
                if held_here and not why and not self.holding[index]:
                    print(f"[server] object {index}: its tracked motion moved it where it was not on frame "
                          f"{frame.id}; held where it was last seen on the part", flush=True)
                if why:
                    # Off the part where its motion carries it: where it was
                    # last on the part, or set down (MODEL_CHECK_FAILS).
                    found = self.held_or_set_down(index, model, fit, mask, metres, K)
                    if found is not None:
                        label, pose = found
                        if not self.holding[index]:
                            print(f"[server] object {index}: the tracked motion carried its fit off the part "
                                  f"on frame {frame.id} ({why}); held {label}", flush=True)
                        self.holding[index], why = True, ""
                        corrections[index] = (fit_id, np.linalg.inv(carried) @ pose)
                elif held_here:
                    self.holding[index] = True
                elif self.holding[index]:
                    print(f"[server] object {index}: carried by its tracked motion again from frame {frame.id}",
                          flush=True)
                    self.holding[index] = False
                if why:
                    # Off the part: another way round in doubt may lie on it (Binding).
                    turned = self.other_way_on(index, model, fit, pose, mask, metres, K, why)
                    if turned is not None:
                        pose, why = turned, ""
                        carried = motion @ model.fitted_pose
                        corrections[index] = (model.fitted_id, np.linalg.inv(carried) @ pose)
                if not why and fit is not None:
                    pose = self.carried_axial(model, fit, pose, mask, metres, K)
                    corrections[index] = (model.fitted_id, np.linalg.inv(motion @ model.fitted_pose) @ pose)
                if why:
                    corrections[index] = None
                else:
                    motions[index] = pose @ np.linalg.inv(model.fitted_pose)
                    self.still_motions[index] = motions[index].copy()
            except Exception:
                traceback.print_exc()
                continue
            if why and not fails[index]:
                print(f"[server] object {index}: mesh pose withheld from frame {frame.id}: "
                      f"the carried fit is off the part ({why})", flush=True)
            fails[index] = fails[index] + 1 if why else 0
            withheld[index] = bool(why)
            if fails[index] >= MODEL_CHECK_FAILS:
                fails[index] = 0
                if not hasattr(self, "last_fitted"):
                    self.last_fitted = {}
                self.last_fitted[index] = self.fitted_models[index]
                self.fitted_models[index] = None
                self.still_motions[index] = None
                self.model_retry_after[index] = time.monotonic()
                self.model_errors[index] = (f"the carried mesh pose no longer fits the view ({why}); "
                                            "fitting it again")
                print(f"[server] object {index}: {self.model_errors[index]}", flush=True)
        return motions

    def carried_axial(self, model, fit, pose, mask, metres, K):
        """For a part round about its long box axis, the carried pose slid
        along its length onto the one end this frame's view shows, where it
        stays on the part; what the view tells of its ends kept for the reply
        (model.carried_axial). Other parts as they are. See _ModelFit.axial()."""
        scale = fitted_axis_scale(model)
        # Round as the mesh is, whatever the fit stretched it to across its length.
        if round_axis(model, 1.0) != int(np.argmax(scale * model.extent)):
            model.carried_axial = None
            return pose
        R, t = pose[:3, :3], pose[:3, 3]
        terms = fit.terms(scale, R, t, *MODEL_CHECK_POINTS)
        if terms is None:
            return pose
        quality = ("fit" if not model_pose_rejection(terms)
                   else "coarse" if not model_pose_rejection(terms, coarse=True) else None)
        _, _, t2, _, report = fit.axial(terms, R, t, scale, quality)
        slid = pose.copy()
        slid[:3, 3] = t2
        if report["slid_m"] and carried_fit_rejection(model, slid, mask, metres, K, fit=fit):
            report["slid_m"] = 0.0
            slid = pose
        model.carried_axial = report
        return slid

    def lost_where_held(self, index, model, metres, K, frame_id):
        """Diagnostic, once a loss: whether the depth still shows a part the
        tracker lost where it was last on the part -- a part set down that
        the tracker does not find again (PP-S3-1, 2026-11-05: lost=[True]
        after the set-down, to the end). The reply is not changed by it."""
        said = getattr(self, "lost_said", None)
        if said is None:
            said = self.lost_said = set()
        still = self.still_motions[index] if index < len(self.still_motions) else None
        if index in said or still is None:
            return
        said.add(index)
        try:
            pose = still @ model.fitted_pose
            found = mesh_click(model, pose[:3, :3], pose[:3, 3], fitted_axis_scale(model), metres, K)
            if found is None:
                print(f"[server] object {index}: tracking lost on frame {frame_id}; where it was last on the "
                      "part is out of view or hidden", flush=True)
                return
            fit = carried_view(model, found[1], metres, K)
            why = "too little depth there" if fit is None else carried_fit_rejection(model, pose, found[1], metres,
                                                                                       K, fit=fit)
            print(f"[server] object {index}: tracking lost on frame {frame_id}; the depth "
                  + (f"no longer shows it where it was last on the part ({why})" if why
                     else "still shows it where it was last on the part") + " -- diagnostic only", flush=True)
        except Exception:
            traceback.print_exc()

    def object_state(self, count):
        """The per-object state of MODEL_CHECK_FAILS's holds, made where missing."""
        for name, empty in (("still_motions", None), ("holding", False)):
            kept = getattr(self, name, None)
            if kept is None or len(kept) != count:
                setattr(self, name, [empty] * count)
        if getattr(self, "pose_priors", None) is None:
            self.pose_priors = {}

    def frame_view(self, index, frame, regions, model):
        """(fit, mask, metres, K): object `index`'s view on this frame for
        `model` -- its mask less the others' regions and the robot, as
        check_carried_meshes() takes it -- or None where it shows too little."""
        h, w = frame.depth.shape
        masks = as_numpy(getattr(frame, "mask", None))
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3 or masks.shape[1:] != (h, w) or index >= len(masks):
            return None
        mask = self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions))
        metres = as_numpy(frame.depth) / frame.depth_factor
        K = np.asarray(frame.intrinsics, dtype=np.float64)
        fit = carried_view(model, mask, metres, K, others_region(index, regions))
        return None if fit is None else (fit, mask, metres, K)

    def best_on_view(self, model, view, candidates):
        """Of `candidates` ((label, pose in this frame's camera)), each
        corrected on the frame's `view` (frame_view()) as a carried pose is,
        the cheapest on the part: (label, pose, cost); and why each not
        ({label: why})."""
        fit, mask, metres, K = view
        best, why = None, {}
        for label, pose in candidates:
            if pose is None or not np.isfinite(pose).all():
                continue
            posed, cost = correct_carried_pose(fit, model, pose)
            off = carried_fit_rejection(model, posed, mask, metres, K, fit=fit)
            if off:
                why[label] = off
            elif best is None or cost < best[2]:
                best = (label, posed, cost)
        return best, why

    def held_or_set_down(self, index, model, fit, mask, metres, K):
        """(label, pose) where the fit is on the part on this frame (`fit` its
        carried_view()) though its motion carries it off: where it was last on
        it (still_motions), or at the robot's pose prior; None if neither. See
        MODEL_CHECK_FAILS."""
        candidates = []
        still = self.still_motions[index] if index < len(self.still_motions) else None
        if still is not None:
            candidates.append(("where it was last seen on the part", still @ model.fitted_pose))
        prior = (getattr(self, "pose_priors", None) or {}).get(index)
        if prior is not None and prior.get("status") in ("adopted", "agrees"):
            candidates.append((f"where the robot set it down ({prior['source']})", prior["pose"]))
        best, _ = self.best_on_view(model, (fit, mask, metres, K), candidates)
        return None if best is None else best[:2]

    def preserve_refit_direction(self, index, current, model, frame, regions, motions):
        """Validate a finer shape in the old direction before publishing it.

        Both stored poses use the tracker's reference frame. Recheck on the
        current camera frame, including motion, and allow only ordinary
        carried-pose corrections. On failure neither model is changed.
        """
        view = self.frame_view(index, frame, regions, model)
        if view is None:
            return False, "no current view; keeping the published direction"
        fit, _, _, _ = view
        motion = np.asarray(motions[index], dtype=np.float64)
        start = motion @ current.fitted_pose
        posed, _ = correct_carried_pose(fit, model, start)
        move, angle = correction_size(model, start, posed)
        if move > MODEL_CORRECT_MAX_M or angle > MODEL_CORRECT_MAX_DEG:
            return False, "refined shape needs a large correction; keeping the published direction"
        terms = fit.terms(fitted_axis_scale(model), posed[:3, :3], posed[:3, 3], *MODEL_CHECK_POINTS)
        why = "no fit" if terms is None else model_pose_rejection(terms)
        if why:
            return False, f"published direction is not a fine fit ({why}); other direction left in doubt"
        old_pose = model.fitted_pose.copy()
        new_pose = np.linalg.inv(motion) @ posed
        base = np.linalg.inv(new_pose) @ old_pose
        # Re-express every candidate in the new published frame; include the
        # refit's original pose, which was not a doubt relative to itself.
        doubts = [base @ d for d in [np.eye(4)] + model_doubts(model)]
        doubts = [d for d in doubts if not same_way(model, np.eye(3), d[:3, :3])]
        model.fitted_unlikely = [base @ d for d in getattr(model, "fitted_unlikely", [])]
        model.fitted_pose = new_pose
        model.fitted_doubts = doubts
        model.fitted_spin_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "spin"), None)
        model.fitted_ends_doubt = next((d for d in doubts if turn_kind(model, d[:3, :3]) == "ends"), None)
        # Measurement claims and axis diagnostics belonged to the other pose.
        model.fitted_measured = bool(terms.get("measured", False))
        model.fitted_axis_source = None
        model.fitted_axial = None
        return True, "fine fit verified in the published direction; other direction left in doubt"

    def adopt_refit(self, index, model, frame, regions, motions, capture_inverse, fitted_on):
        """The pose to send a background refit (`model`, its fitted_pose in
        the camera of frame `fitted_on`) with on this frame, in the tracker's
        reference, or (None, why): carried here by the tracked motion, where it
        was fitted, or at the robot's pose prior -- the cheapest of them on
        the part here. See MODEL_CHECK_FAILS."""
        view = self.frame_view(index, frame, regions, model)
        motion = np.asarray(motions[index], dtype=np.float64)
        if view is None or motion.shape != (4, 4) or not np.isfinite(motion).all():
            return None, f"too little of it in view on frame {frame.id} to check the refit on"
        T = model.fitted_pose
        candidates = [("where it was fitted", T)]
        if capture_inverse is not None:
            candidates.insert(0, ("carried by its motion", motion @ capture_inverse @ T))
        prior = (getattr(self, "pose_priors", None) or {}).get(index)
        if prior is not None:
            candidates.append((f"where the robot set it down ({prior['source']})", prior["pose"]))
        best, why = self.best_on_view(model, view, candidates)
        if best is None:
            return None, (f"the refit on frame {fitted_on} is off the part on frame {frame.id}: "
                          + "; ".join(f"{label}: {off}" for label, off in why.items()))
        label, pose, _ = best
        print(f"[server] object {index}: the refit on frame {fitted_on} is on the part on frame {frame.id} "
              f"{label}", flush=True)
        return np.linalg.inv(motion) @ pose, ""

    def take_priors(self, meta, frame_id):
        """The robot's poses of parts it set down (meta["priors"]: [{"object":
        index, "pose": 4x4, the mesh's pose as model_pose gives it, in this
        frame's camera, "source": what says so}]), kept to be tried
        (apply_priors()). See MODEL_CHECK_FAILS."""
        self.object_state(len(getattr(self, "models", None) or []))
        for prior in meta.get("priors") or []:
            try:
                index = int(prior["object"])
                pose = np.asarray(prior["pose"], dtype=np.float64).reshape(4, 4)
                if (not 0 <= index < len(self.models) or not np.isfinite(pose).all()
                        or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
                        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                        or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
                    raise ValueError("not an object's rigid pose")
                self.pose_priors[index] = {"pose": pose, "source": str(prior.get("source") or "client"),
                                           "frame": frame_id, "status": "pending", "why": ""}
                print(f"[server] object {index}: pose prior from {self.pose_priors[index]['source']} "
                      f"on frame {frame_id}", flush=True)
            except (KeyError, TypeError, ValueError) as exc:
                print(f"[server] unusable pose prior {prior!r}: {exc}", flush=True)

    def apply_priors(self, motions, frame, regions=None):
        """Try each pending pose prior (take_priors()) on this frame, as one
        more way the part may lie, scored with the fit's: corrected on the
        frame and off the part, it waits for a frame it is on (the gripper
        may still be on it); on it, it is taken where there is no fit or the
        fit is off the part here, agrees where it is the fit's way round,
        and is taken over the fit's -- which stays in doubt (Binding) --
        where the view cannot tell them apart (settle_turns()'s margin),
        refused where the fit's fits better by more. Where it agrees and fits
        this frame better than the carried pose by MODEL_CORRECT_MIN_GAIN, the
        pose is sent there. Returns the motions. See MODEL_CHECK_FAILS."""
        motions = np.array(motions, dtype=np.float64)
        priors = getattr(self, "pose_priors", None) or {}
        for index, prior in priors.items():
            if prior["status"] != "pending":
                continue
            try:
                current = self.fitted_models[index]
                model = current
                if model is None:
                    # Relocating: the mesh at the shape it was last fitted at, else measured.
                    base = getattr(self, "last_fitted", {}).get(index) or self.models[index]
                    if base is None:
                        prior.update(status="refused", why="no mesh for it")
                        continue
                    model = copy.copy(base)
                    if getattr(model, "fitted_linear", None) is None:
                        shape = self.known_shape(model)
                        shape = np.ones(3) * (model.known_scale or 1.0) if shape is None else shape
                        B = model.box[:3, :3]
                        model.fitted_linear, model.fitted_scale = (B * shape) @ B.T, _overall(shape)
                view = self.frame_view(index, frame, regions, model)
                if view is None:
                    prior["why"] = f"too little of it in view on frame {frame.id}"
                    continue
                best, why = self.best_on_view(model, view, [("prior", prior["pose"])])
                if best is None:
                    prior["why"] = f"off the part on frame {frame.id}: {why.get('prior', '')}"
                    continue
                _, pose_p, _ = best
                motion = np.asarray(motions[index], dtype=np.float64)
                if current is None or self.model_withheld[index]:
                    self.take_prior_as_fit(index, model, view, pose_p, motion, prior)
                    motions[index] = motion
                    continue
                pose_c = motion @ current.fitted_pose
                if same_way(current, pose_c[:3, :3], pose_p[:3, :3]):
                    prior.update(status="agrees", why="the fit's way round")
                    scale = fitted_axis_scale(current)
                    tc = view[0].terms(scale, pose_c[:3, :3], pose_c[:3, 3], *MODEL_CHECK_POINTS)
                    tp = view[0].terms(scale, pose_p[:3, :3], pose_p[:3, 3], *MODEL_CHECK_POINTS)
                    if tc is None or tp["cost"] < tc["cost"] - MODEL_CORRECT_MIN_GAIN:
                        motions[index] = pose_p @ np.linalg.inv(current.fitted_pose)
                else:
                    fit = view[0]
                    scale = fitted_axis_scale(current)
                    tc = fit.terms(scale, pose_c[:3, :3], pose_c[:3, 3], *MODEL_CHECK_POINTS)
                    tp = fit.terms(scale, pose_p[:3, :3], pose_p[:3, 3], *MODEL_CHECK_POINTS)
                    left = lambda t: 1 - t["fitness"] + t["contradiction"]
                    margin = max(MODEL_TURN_COST_SLACK, left(tc) + left(tp))
                    costs = f"cost {tp['cost']:.3f} against the fit's {tc['cost']:.3f}, margin {margin:.3f}"
                    if tp["cost"] > tc["cost"] + margin:
                        prior.update(status="refused", why=f"the fit's way round fits the view better: {costs}")
                    else:
                        binding = self.bindings[index]
                        if binding is None or binding.fit_id != current.fitted_id:
                            binding = self.bindings[index] = Binding(current)
                        turn = np.linalg.inv(pose_c) @ pose_p
                        k = binding.add(turn)
                        if k is None:
                            k = next(j for j, T in enumerate(binding.turns)
                                     if same_way(current, T[:3, :3], turn[:3, :3]))
                        if not binding.alive[k]:
                            binding.alive[k], binding.ruled[k] = True, None
                        if k:
                            self.turn_binding(index, current, binding, k)
                        prior.update(status="adopted", why=f"the view cannot tell it from the fit's way round: {costs}")
                print(f"[server] object {index}: pose prior {prior['status']} ({prior['why']})", flush=True)
                fitted = self.fitted_models[index]
                self.still_motions[index] = pose_p @ np.linalg.inv(fitted.fitted_pose) if fitted is not None else None
                if fitted is not None and prior["status"] == "adopted":
                    motions[index] = pose_p @ np.linalg.inv(fitted.fitted_pose)
            except Exception:
                traceback.print_exc()
        return motions

    def take_prior_as_fit(self, index, model, view, pose, motion, prior):
        """A pose prior on the part where the object has no fit on it: the
        mesh fitted there, a new fit (see apply_priors())."""
        fit = view[0]
        model = copy.copy(model)
        scale = fitted_axis_scale(model)
        terms = fit.terms(scale, pose[:3, :3], pose[:3, 3], MODEL_SAMPLES, MODEL_FINE_POINTS)
        model.fitted_pose = np.linalg.inv(motion) @ pose
        model.fitted_quality = "fit" if not model_pose_rejection(terms) else "coarse"
        model.fitted_measured = bool(getattr(model, "fitted_measured", False))
        model.fitted_doubts, model.fitted_spin_doubt, model.fitted_ends_doubt = [], None, None
        model.fitted_id = uuid.uuid4().hex
        B = model.box[:3, :3]
        T = np.eye(4)
        T[:3, :3] = model.fitted_pose[:3, :3] @ B
        T[:3, 3] = model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3]) + model.fitted_pose[:3, 3]
        self.boxes[index] = (T, scale * model.extent)
        self.fitted_models[index] = model
        self.still_motions[index] = np.asarray(motion, dtype=np.float64).copy()
        self.model_errors[index] = None
        self.model_withheld[index] = False
        self.model_check_fails[index] = 0
        self.model_corrections[index] = None
        prior.update(status="adopted", why=f"on the part ({model.fitted_quality}, cost {terms['cost']:.3f}) "
                                           "where the server had no fit on it")

    def update_bindings(self, motions, frame, regions=None, robot=None, observe=True, rank=False):
        """Each fitted mesh's binding (Binding): made for a new fit, carrying
        the evidence of the one before; the ways round a refit held back
        added; the person's choice (meta["bind"]); with `observe`, this
        frame's evidence; and what that rules out applied -- the published
        way round turned to the one left, the doubts those still living.
        `robot`: the robot's pixels, where the part may not be judged at rest.
        `rank`, on the init's view: a way round the ends' silhouettes vote for
        over the published one is published in its place, the other kept in
        doubt (ENDS_SILHOUETTE_MIN_SHARE). See BINDING_CONFIDENCE."""
        models = getattr(self, "fitted_models", None)
        if not models:
            return
        bindings = getattr(self, "bindings", None)
        if bindings is None or len(bindings) != len(models):
            bindings = self.bindings = [None] * len(models)
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(models), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(models), h, w):
                observe = False
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return
        withheld = getattr(self, "model_withheld", None) or [False] * len(models)
        pending = getattr(self, "pending_binds", None) or {}
        for index, model in enumerate(models):
            if model is None or model.fitted_pose is None or getattr(model, "fitted_id", None) is None:
                continue
            try:
                binding = bindings[index]
                if binding is None or binding.fit_id != model.fitted_id:
                    before = (binding if binding is not None and binding.model.model_id == model.model_id
                              else None)
                    binding = bindings[index] = Binding(model, before)
                else:
                    for d in model_doubts(model):
                        binding.add(d)
                chosen = pending.pop(index, None)
                if chosen is not None:
                    ident, by = chosen if isinstance(chosen, tuple) else (chosen, "person")
                    if binding.choose(ident, by):
                        self.say(f"object {index}: way round {ident} chosen by the {by}")
                        if by == "camera" and binding.ids.index(ident):
                            self.turn_binding(index, model, binding, binding.ids.index(ident))
                    else:
                        self.say(f"object {index}: way round {ident} chosen by the {by}, but it is none in doubt; "
                                 "ignored")
                motion = np.asarray(motions[index], dtype=np.float64)
                if (observe and len(binding.living()) > 1 and index < len(lost) and not lost[index]
                        and not withheld[index] and np.isfinite(motion).all()):
                    mask = self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions))
                    fit = carried_view(model, mask, metres, K, others_region(index, regions))
                    if fit is not None:
                        at_rest = robot is None or robot.shape != fit.grown.shape or not (fit.grown & robot).any()
                        if binding.observe(fit, motion @ model.fitted_pose, at_rest):
                            self.say_ends(index, binding, rank)
                if rank and len(binding.living()) > 1:
                    k = binding.ends_favoured()
                    if k is not None:
                        self.say(f"object {index}: the init's view favours {binding.ids[k]} end for end over the "
                                 f"published {binding.ids[0]}; publishing it, {binding.ids[0]} kept in doubt")
                        self.turn_binding(index, model, binding, k)
                living_before = len(binding.living())
                kept, changed = binding.decide()
                if kept:
                    self.turn_binding(index, model, binding, kept)
                living = binding.living()
                model.fitted_doubts = [binding.turns[k] for k in binding.in_doubt()]
                model.fitted_unlikely = [binding.turns[k] for k in living if k and binding.unlikely[k]]
                model.fitted_spin_doubt = next((d for d in model.fitted_doubts
                                                if turn_kind(model, d[:3, :3]) == "spin"), None)
                model.fitted_ends_doubt = next((d for d in model.fitted_doubts
                                                if turn_kind(model, d[:3, :3]) == "ends"), None)
                if changed or chosen is not None:
                    post = binding.posterior()
                    left = ", ".join(f"{binding.ids[k]} {post.get(k, 1.0):.2f}" for k in living)
                    self.say(f"object {index}: binding {binding.state()} by {binding.by} after "
                             f"{binding.frames} frame(s) of evidence; {living_before} way(s) round "
                             f"then {len(living)}: {left}"
                             + "".join(f"; {binding.ids[k]} out: {binding.ruled[k][1]}"
                                       for k in range(len(binding.turns)) if not binding.alive[k]))
            except Exception:
                traceback.print_exc()

    def say_ends(self, index, binding, always=False):
        """The last frame's evidence on the ways round turned end for end
        (Binding.vote_ends()), silhouettes and ends: each pair's, where it
        voted or `always`."""
        for (j, k), both in binding.last_ends.items():
            for kind, r in both.items():
                if r["winner"] is None and not always:
                    continue
                if kind == "silhouette":
                    told = (f"{r['for_j']} px for {binding.ids[j]}, {r['for_k']} for {binding.ids[k]}, of {r['told']} "
                            f"told ({r['told'] / max(r['area'], 1):.1%} of the silhouettes' {r['area']})")
                else:
                    told = (f"{binding.ids[j]} {r['off_j'] * 1000:.1f} mm off where the part ends, {binding.ids[k]} "
                            f"{r['off_k'] * 1000:.1f} mm, on {r['ends']} end(s): {r['share']:.1%} of its length apart")
                verdict = (f"a vote for {binding.ids[j] if r['winner'] == 'j' else binding.ids[k]}" if r["winner"]
                           else f"no vote: {r['why']}")
                self.say(f"object {index}: ends' {kind}s {binding.ids[j]} / {binding.ids[k]}: {told}; {verdict}")

    def other_way_on(self, index, model, fit, pose, mask, metres, K, why):
        """Where the published way round, corrected to `pose`, is off the part
        on this frame (`why`), the way round in doubt that lies on it --
        corrected on `fit` as the published one was, the cheapest of those
        carried_fit_rejection() passes -- published in its place; its pose,
        or None where none does. The other stays in doubt: one frame is one
        view, and one view rules nothing out (BINDING_CONFIDENCE)."""
        bindings = getattr(self, "bindings", None) or []
        binding = bindings[index] if index < len(bindings) else None
        if binding is None or binding.fit_id != model.fitted_id or len(binding.living()) < 2:
            return None
        best = None
        for k in binding.living():
            if not k:
                continue
            posed, cost = correct_carried_pose(fit, model, pose @ binding.turns[k], binding.scales[k])
            if (not carried_fit_rejection(model, posed, mask, metres, K, fit=fit, scale=binding.scales[k])
                    and (best is None or cost < best[2])):
                best = (k, posed, cost)
        if best is None:
            return None
        k, posed, _ = best
        self.say(f"object {index}: the published way round {binding.ids[0]} left the part ({why}) where "
                 f"{binding.ids[k]} lay on it; {binding.ids[k]} published, the other still in doubt")
        self.turn_binding(index, model, binding, k)
        return posed

    def turn_binding(self, index, model, binding, k):
        """Publish way round k of `binding` in place of the published one: the
        fit turned to it, with a new id, its box with it."""
        kind = turn_kind(model, binding.turns[k][:3, :3]) or "about its round axis"
        model.fitted_pose = model.fitted_pose @ binding.turns[k]
        binding.rebase(k)
        B, scale = model.box[:3, :3], binding.scale
        model.fitted_linear = (B * scale) @ B.T
        model.fitted_id = binding.fit_id = uuid.uuid4().hex
        boxes = getattr(self, "boxes", None)
        if boxes and index < len(boxes) and boxes[index] is not None:
            T = np.eye(4)
            T[:3, :3] = model.fitted_pose[:3, :3] @ B
            T[:3, 3] = model.fitted_pose[:3, :3] @ (model.fitted_linear @ model.box[:3, 3]) + model.fitted_pose[:3, 3]
            boxes[index] = (T, scale * model.extent)
        for name in ("model_corrections", "sent_model_poses"):
            kept = getattr(self, name, None)
            if kept and index < len(kept):
                kept[index] = None
        self.say(f"object {index}: published way round turned to {binding.ids[0]} "
                 f"({kind} of the one published before)")

    def load_archive(self, models):
        """Each mesh's archive (see the comment after BINDING_FEATURE_PX),
        where kept: its shapes among the known ones, the first time this run
        sees the mesh."""
        if not getattr(self, "archive_dir", None):
            return
        for index, model in enumerate(models):
            if model is None:
                continue
            path = os.path.join(self.archive_dir, f"{model.model_id}.npz")
            if not os.path.exists(path):
                continue
            try:
                with np.load(path) as data:
                    points = len(data["points"])
                    shapes = np.asarray(data["shapes"], np.float64).reshape(-1, 3).tolist()
                known = getattr(self, "known_scales", None)
                if known is None:
                    known = self.known_scales = {}
                if shapes and model.model_id not in known:
                    known[model.model_id] = shapes[-LOCATE_KNOWN_SCALES:]
                self.say(f"object {index}: archive of {len(shapes)} shapes, {points} surface points")
            except Exception as exc:
                print(f"[server] object {index}: unusable archive {path} ({exc})", flush=True)

    def archive_session(self):
        """File, where an archive is kept, each mesh's measured shapes, and the
        surface seen of its part where its binding was settled by more than
        the init view; see BINDING_FEATURE_PX."""
        if not getattr(self, "archive_dir", None):
            return
        models = getattr(self, "models", None) or []
        fitted = getattr(self, "fitted_models", None) or []
        bindings = getattr(self, "bindings", None) or []
        surfaces = getattr(self, "surfaces", None) or []
        for index, model in enumerate(models):
            if model is None:
                continue
            try:
                path = os.path.join(self.archive_dir, f"{model.model_id}.npz")
                points, runs, shapes = np.zeros((0, 3)), np.zeros(0, int), []
                if os.path.exists(path):
                    with np.load(path) as data:
                        points = np.asarray(data["points"], np.float64).reshape(-1, 3)
                        runs = np.asarray(data["runs"]) if "runs" in data else np.ones(len(points), int)
                        shapes = np.asarray(data["shapes"], np.float64).reshape(-1, 3).tolist()
                shapes = (getattr(self, "known_scales", None) or {}).get(model.model_id, shapes)
                added = 0
                fit = fitted[index] if index < len(fitted) else None
                binding = bindings[index] if index < len(bindings) else None
                if (fit is not None and binding is not None and binding.fit_id == fit.fitted_id
                        and binding.state() == "locked" and fit.fitted_quality == "fit"
                        and index < len(surfaces)):
                    seen = surfaces[index].points()
                    linear = np.asarray(fit.fitted_linear if fit.fitted_linear is not None
                                        else fit.fitted_scale * np.eye(3), np.float64)
                    back = np.linalg.inv(fit.fitted_pose)
                    scaled = seen @ back[:3, :3].T + back[:3, 3]
                    B = fit.box[:3, :3]
                    half = fitted_axis_scale(fit) * fit.extent / 2 + 2 * MODEL_TOL_M
                    inside = np.all(np.abs((scaled - linear @ fit.box[:3, 3]) @ B) <= half, axis=1)
                    new = scaled[inside] @ np.linalg.inv(linear).T
                    added = len(new)
                    voxel = SURFACE_VOXEL_M / _overall(fitted_axis_scale(fit))
                    every = np.vstack([points, new])
                    keys, inverse = np.unique(np.floor(every / voxel).astype(np.int64), axis=0, return_inverse=True)
                    inverse = inverse.reshape(-1)
                    sums = np.zeros((len(keys), 3))
                    np.add.at(sums, inverse, every)
                    # A voxel's runs: the most of the old points in it, and one for this run.
                    counted = np.zeros(len(keys), int)
                    np.maximum.at(counted, inverse[:len(points)], runs.astype(int))
                    counted[np.unique(inverse[len(points):])] += 1
                    order = np.argsort(-counted, kind="stable")[:SURFACE_MAX_VOXELS]
                    points = (sums / np.bincount(inverse, minlength=len(keys))[:, None])[order]
                    runs = counted[order]
                if not added and not shapes:
                    continue
                os.makedirs(self.archive_dir, exist_ok=True)
                partial = path + ".partial.npz"
                np.savez(partial, points=np.asarray(points, np.float32), runs=np.asarray(runs, np.int32),
                         shapes=np.asarray(shapes, np.float64))
                os.replace(partial, path)
                print(f"[server] object {index}: archived {added} surface points "
                      f"({len(points)} in all), {len(shapes)} shapes", flush=True)
            except Exception:
                traceback.print_exc()

    def record_surfaces(self, motions, frame, regions=None):
        """Add this frame's view of each tracked object to its SurfaceMemory."""
        try:
            h, w = frame.depth.shape
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.shape == (len(self.surfaces), 1, h, w):
                masks = masks[:, 0]
            if masks.shape != (len(self.surfaces), h, w):
                return
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            lost = [bool(getattr(obj, "lost", False)) for obj in self.pipeline.objects]
        except Exception:
            return
        rng = np.random.default_rng(frame.id)
        for index, memory in enumerate(self.surfaces):
            try:
                motion = np.asarray(motions[index], dtype=np.float64)
                if lost[index] or motion.shape != (4, 4) or not np.isfinite(motion).all():
                    continue
                view = observe(self.without_robot(without_others(np.asarray(masks[index]) > 0, index, regions)),
                               metres, K, rng)
                if view is None:
                    continue
                cam, _, keep, _, _ = view
                points = cam[keep]
                points = points[rng.permutation(len(points))[:SURFACE_PIXELS]]
                back = np.linalg.inv(motion)
                memory.add(points @ back[:3, :3].T + back[:3, 3], frame.id, camera=back[:3, 3])
            except Exception:
                continue

    def bboxes(self, motions, frame):
        """Per-object boxes; a failure yields None for that box, never a failed reply.

        recognition_lost() rejects mesh objects missing a box while keeping
        the other objects in the reply usable.
        """
        out = []
        for index, motion in enumerate(motions):
            box = self.boxes[index] if self.boxes and index < len(self.boxes) else None
            if box is None:
                out.append(None)
                continue
            try:
                result = box_in_camera(motion, *box, frame.intrinsics, frame.rgb.shape[:2])
                prior = (getattr(self, "pose_priors", None) or {}).get(index)
                if prior is not None:
                    # What came of the robot's pose of it (MODEL_CHECK_FAILS).
                    result["model_prior"] = {k: prior[k] for k in ("source", "status", "why", "frame")}
                models = getattr(self, "fitted_models", None)
                model = models[index] if models and index < len(models) else None
                withheld = getattr(self, "model_withheld", None)
                if withheld and index < len(withheld) and withheld[index]:
                    model = None                 # off the part on this frame; see check_carried_meshes()
                if model is not None and model.fitted_pose is not None:
                    linear = getattr(model, "fitted_linear", None)
                    pose = motion @ model.fitted_pose
                    held, axis = self.held_turn(index, model, pose)
                    result.update(model_pose=(pose @ held).tolist(),
                                  model_scale=model.fitted_scale, model_id=model.model_id,
                                  model_linear=(np.asarray(linear) if linear is not None
                                                else model.fitted_scale * np.eye(3)).tolist(),
                                  model_quality=getattr(model, "fitted_quality", None) or "fit",
                                  model_measured=bool(getattr(model, "fitted_measured", True)),
                                  model_fit_id=getattr(model, "fitted_id", None),
                                  # What the mesh itself leaves open, in its own
                                  # frame: the axis it is round about, and
                                  # whether its ends are alike.
                                  model_symmetry={"axis": axis,
                                                  "ends": bool(getattr(model, "ends_alike", False))})
                    axial = getattr(model, "carried_axial", None) or getattr(model, "fitted_axial", None)
                    if axial is not None:
                        # Which ends of its length the view showed (_ModelFit.axial()):
                        # this frame's for a bar, else the fit's.
                        result["model_axial"] = axial
                    sources = getattr(model, "fitted_axis_source", None)
                    if sources is not None:
                        # Per box axis of the mesh (its frame): the scale along it, the
                        # part's size there, and where that came from (MODEL_SUPPORT_TOP_PERCENTILE).
                        axis_scale = fitted_axis_scale(model)
                        result["model_axes"] = [{"axis": model.box[:3, j].tolist(), "scale": float(axis_scale[j]),
                                                 "extent_m": float(axis_scale[j] * model.extent[j]),
                                                 "source": sources[j]} for j in range(3)]
                    kept = getattr(model, "kept_vertices", None)
                    if kept is not None:
                        # The box of the mesh less its loose fragments (MODEL_FRAGMENT_GAP).
                        scaled = kept @ np.asarray(result["model_linear"], np.float64).T
                        result["model_bounds"] = [scaled.min(0).tolist(), scaled.max(0).tolist()]
                    doubt = getattr(model, "fitted_spin_doubt", None)
                    if doubt is not None:
                        # Relative to the fit, so carried by the motion with it.
                        result["model_spin_doubt"] = (np.linalg.inv(held) @ doubt @ held).tolist()
                    ends = getattr(model, "fitted_ends_doubt", None)
                    if ends is not None:
                        result["model_ends_doubt"] = (np.linalg.inv(held) @ ends @ held).tolist()
                    # Every way round in doubt (MODEL_TURN_COST_SLACK), with a
                    # refit's held back (MODEL_REFIT_TURN_DEG).
                    doubts = list(getattr(model, "fitted_doubts", []))
                    doubts += [d for d in (doubt, ends)
                               if d is not None and not any(np.allclose(d, k) for k in doubts)]
                    if doubts:
                        sent = [np.linalg.inv(held) @ d @ held for d in doubts]
                        result["model_doubts"] = [d.tolist() for d in sent]
                        # Each one's kind (turn_kind(): "spin" about the long box
                        # axis, "ends" about another), the axis it turns about in
                        # the mesh's frame and by how much; and every "ends" one,
                        # where model_ends_doubt has the first.
                        kinds = []
                        for d, d_sent in zip(doubts, sent):
                            axis = np.linalg.eigh(d_sent[:3, :3] + d_sent[:3, :3].T)[1][:, -1]
                            angle = np.degrees(np.arccos(np.clip((np.trace(d_sent[:3, :3]) - 1) / 2, -1, 1)))
                            kinds.append({"kind": turn_kind(model, d[:3, :3]) or "round",
                                          "axis": (axis * np.sign(axis[np.argmax(np.abs(axis))])).tolist(),
                                          "deg": round(float(angle), 1)})
                        result["model_doubt_kinds"] = kinds
                        result["model_ends_doubts"] = [d.tolist() for d, k in zip(sent, kinds) if k["kind"] == "ends"]
                    # How far its binding is, and how to tell its ways round (BINDING_CONFIDENCE).
                    bindings = getattr(self, "bindings", None) or []
                    binding = bindings[index] if index < len(bindings) else None
                    if binding is not None and binding.fit_id == getattr(model, "fitted_id", None):
                        result["model_binding"] = binding.reply(held, pose, np.asarray(frame.intrinsics, np.float64),
                                                                frame.rgb.shape[:2])
                surfaces = getattr(self, "surfaces", None)
                if (surfaces and index < len(surfaces)
                        and (frame.id == 0 or frame.id % SURFACE_PUBLISH_EVERY == 0)):
                    # In the frame of the pose published: the mesh's, else the box's.
                    reference = (model.fitted_pose if model is not None and model.fitted_pose is not None
                                 else box[0])
                    points, views = surfaces[index].points(views=True)
                    if len(points):
                        back = np.linalg.inv(reference)
                        result.update(surface=encode_points(points @ back[:3, :3].T + back[:3, 3]),
                                      surface_views=encode_points(views @ back[:3, :3].T),
                                      surface_frame="model" if reference is not box[0] else "box")
                out.append(result)
            except Exception:
                if not self.bbox_warned:
                    self.bbox_warned = True
                    print("[server] bbox failed; sending None (logged once per session)",
                          flush=True)
                    traceback.print_exc()
                out.append(None)
        return out

    def remember_scale(self, model):
        """Keep a fine fit's shape, its scales along its mesh's box axes, where
        the view measured it: with those measured before, or in their place
        where it fitted the view better than they did; not where theirs was
        kept. See LOCATE_KNOWN_SCALES."""
        how = getattr(model, "fitted_known_shape", None)
        if model is None or model.fitted_quality != "fit" or not model.fitted_measured or how == "kept":
            return
        known = getattr(self, "known_scales", None)
        if known is None:
            known = self.known_scales = {}
        before = [] if how == "beaten" else known.get(model.model_id, [])
        known[model.model_id] = (before + [fitted_axis_scale(model).tolist()])[-LOCATE_KNOWN_SCALES:]

    def known_shape(self, model):
        """The median, axis by axis, of a mesh's measured shapes, or None."""
        shapes = (getattr(self, "known_scales", None) or {}).get(getattr(model, "model_id", None))
        return np.median(np.asarray(shapes, np.float64).reshape(-1, 3), axis=0) if shapes else None

    def known_scale(self, model):
        """The overall size of a mesh's measured shape, or None."""
        shape = self.known_shape(model)
        return None if shape is None else _overall(shape)

    def held_turn(self, index, model, pose):
        """(turn, axis): the turn of the fitted mesh at `pose` about the axis
        it is round about (round_axis()), in its own frame about its box
        centre, that brings it nearest the pose last sent for object `index`
        -- with none sent, another of its box axes as near the image's up as
        it goes (toward the camera where the axis is the up) -- and that axis
        in the mesh's frame; identity and None for a mesh round about none.
        The pose so turned is remembered as sent. See MODEL_SYMMETRY_STEP_DEG."""
        sent = getattr(self, "sent_model_poses", None) or []
        sent += [None] * (index + 1 - len(sent))
        self.sent_model_poses = sent
        turn = np.eye(4)
        scale = fitted_axis_scale(model)
        k = round_axis(model, scale)
        if k is not None:
            B = model.box[:3, :3]
            a = B[:, k]
            R = pose[:3, :3]
            last = sent[index]
            if last is not None:
                D = last[:3, :3].T @ R
                angle = np.arctan2(np.trace(D @ _skew(a)), np.trace(D) - a @ D @ a)
            else:
                e = B[:, min(j for j in range(3) if j != k)]
                e = e * np.sign(e[int(np.argmax(np.abs(e)))])
                toward = np.array([0.0, -1.0, 0.0])
                if abs((R @ a) @ toward) > 0.9:
                    toward = np.array([0.0, 0.0, -1.0])
                angle = np.arctan2((R @ np.cross(a, e)) @ toward, (R @ e) @ toward)
            Q = _rotation(angle * a)
            linear = getattr(model, "fitted_linear", None)
            linear = linear if linear is not None else model.fitted_scale * np.eye(3)
            centre = np.asarray(linear, np.float64) @ model.box[:3, 3]
            turn[:3, :3], turn[:3, 3] = Q, centre - Q @ centre
        sent[index] = pose @ turn
        return turn, None if k is None else model.box[:3, k].tolist()

    def recognition_lost(self, tracking_lost, bboxes):
        """Lost: its tracks are, or it has no box at all. An object whose mesh
        has not fitted yet is not lost: it has the box of its points, and its
        reply says it has no mesh pose."""
        return [bool(lost or bboxes[index] is None) for index, lost in enumerate(tracking_lost)]

    def handle_init(self, meta, parts):
        started = time.time()
        self.init_report = []
        self.build_pipeline()
        rgb = decode_rgb(parts[0])
        depth = decode_depth(parts[1], meta).astype(np.float32)
        K = np.asarray(meta["K"], dtype=np.float64).reshape(3, 3)
        depth_factor = float(meta["depth_factor"])
        models = decode_models(meta, parts[2:], len(meta["points"]))
        self.load_archive(models)
        robot = decode_robot_mask(meta, depth.shape)
        self.same_scale = bool(meta.get("same_scale_meshes"))    # see LOCATE_SCALE_AGREE
        self.init_proposals = None
        meta["points"], meta["labels"], located = self.locate_unclicked(
            rgb, depth / depth_factor, K, meta["points"], meta["labels"], models, robot)
        self.init_points = meta["points"]
        meta["points"], meta["labels"] = self.separate_parts(rgb, meta["points"], meta["labels"])
        # Followed apart from the parts, by the depth (OBSTACLE_FOLLOW_PX).
        self.obstacles = self.discover_obstacles(rgb, depth / depth_factor, K, meta["points"], meta["labels"],
                                                 located, robot, int(meta.get("discover") or 0))
        self.init_proposals = None
        self.anchors = np.stack([
            anchor_from_points(points, labels, depth, K, depth_factor)
            for points, labels in zip(meta["points"], meta["labels"])
        ])
        specs = meta.get("models") or []
        required = [index < len(specs) and specs[index] is not None
                    for index in range(len(meta["points"]))]
        self.boxes = self.fit_boxes(rgb, depth / depth_factor, K, meta["points"], meta["labels"],
                                    models, required, located, robot)
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
        self.surfaces = [SurfaceMemory() for _ in meta["points"]]
        regions = self.carried_regions(motions, frame)
        self.record_surfaces(motions, frame, regions)
        # Where each fit was seen on the part: the init's frame (MODEL_CHECK_FAILS).
        self.object_state(len(motions))
        for index, model in enumerate(self.fitted_models or []):
            if model is not None and model.fitted_pose is not None:
                self.still_motions[index] = np.asarray(motions[index], dtype=np.float64).copy()
        # The init view's own evidence on the ways round it left in doubt.
        self.update_bindings(motions, frame, regions, robot, rank=True)
        self.smoother.reset(motions)
        self.frame_id += 1
        self.say(f"init done in {time.time() - started:.1f} s")
        return motions @ self.anchors, frame, motions

    def say(self, text):
        """A line of the server's log, kept for the init's reply while an init
        runs (init_report): what the client sees of how its objects were
        located and fitted, and how long each took."""
        print(f"[server] {text}", flush=True)
        report = getattr(self, "init_report", None)
        if report is not None:
            report.append(text)

    def proposals(self, rgb, metres):
        """(proposals, sources, (SAM2's, those taken as their depth region, the
        depth's)) of the init frame: SAM2's masks, each with LOCATE_DEPTH_SAME_IOU
        with a depth region that region instead, then the depth's other regions
        (depth_proposals()); (None, None, None) if the segmenter offers none.
        Made once an init: locating and discovering obstacles share them."""
        made = getattr(self, "init_proposals", None)
        if made is not None:
            return made
        sam = self.pipeline.propose_masks(rgb)
        if sam is None:
            return None, None, None
        sam = list(sam)
        regions = depth_proposals(metres)
        proposals, sources, as_region = [], [], 0
        for which, mask in enumerate(sam):
            same = [k for k, region in enumerate(regions)
                    if (mask & region).sum() > LOCATE_DEPTH_SAME_IOU * (mask | region).sum()]
            if same:
                proposals.append(regions.pop(same[0]))
                sources.append(f"SAM2 proposal {which} as its depth region")
                as_region += 1
            else:
                proposals.append(mask)
                sources.append(f"SAM2 proposal {which}")
        proposals += regions
        sources += [f"depth region {k}" for k in range(len(regions))]
        self.init_proposals = (proposals, sources, (len(sam), as_region, len(regions)))
        return self.init_proposals

    def locate_unclicked(self, rgb, metres, K, points, labels, models, robot=None):
        """Each object's click, made from its mesh where none was sent, and
        the region each was located in ({index: region}).

        An object sent with an empty point list and a mesh is located in the
        view by the mesh (locate_with_mesh()); clicked objects are kept as
        they are. No object is located on another's -- a clicked object's
        segment, an object located before -- nor on `robot`, the robot's
        pixels as the client sees them. One not located is tried again once
        others are, without their pixels: RR-S1-2's ring, the bar through its
        hole in its segment, fitted none, and was located once the bar was.
        """
        self.locate_scales, self.locate_shapes = {}, {}
        if all(len(p) for p in points):
            return points, labels, {}
        started = time.time()
        proposals, sources, (sam, as_region, regions) = self.proposals(rgb, metres)
        if proposals is None:
            raise RuntimeError("the segmenter offers no mask proposals; click every object")

        clicked = [p for pts, lbls in zip(points, labels)
                   for p, label in zip(pts, lbls) if label]
        taken = np.zeros(metres.shape, bool)
        # A clicked object's segment is that object's: the obstacle box's
        # round latch, in it, took the hidden ring's place in RI-S4-2.
        by_click = [index for index, pts in enumerate(points) if len(pts)]
        if by_click:
            try:
                logits = self.pipeline.preview_user_masks(rgb, [points[i] for i in by_click],
                                                          [labels[i] for i in by_click])
                taken |= np.any((as_numpy(logits) > 0).reshape(len(by_click), *metres.shape), axis=0)
            except Exception as exc:
                self.say(f"no segments of the clicked objects ({type(exc).__name__}: {exc}); "
                         "proposals on them are fitted too")
        points, labels = list(points), list(labels)
        self.say(f"{sam} SAM2 proposals ({as_region} taken as their depth region), "
                 f"{regions} more from depth; clicked objects on {int(taken.sum())} px; "
                 + (f"robot on {int(robot.sum())} px" if robot is not None else "robot not sent"))
        pending = [index for index, pts in enumerate(points) if not len(pts)]
        for index in pending:
            if models[index] is None:
                raise ValueError(f"object {index}: no click and no usable mesh to locate it by")
        clicked_taken = taken.copy()
        grown = taken.copy()             # the located parts in it grown (LOCATE_REGION_GROW_PX)
        located, notes, fitted, round_, rest = {}, {}, {}, 0, False
        fitted_rivals = {}             # each located part's fit, with the other places it fits (fit[3])
        shapes = {}               # each located part's scale along its box axes
        while pending:
            round_ += 1
            progress = False
            for index in list(pending):
                why = []
                click, region, note, fit = locate_with_mesh(models[index], proposals, metres, K,
                                                            clicked, taken, robot, why, rest, grown=grown,
                                                            image=rgb)
                for which, said in sorted(why):
                    self.say(f"  object {index}{f' (round {round_})' if round_ > 1 else ''}"
                             f"{', what is left of proposals' if rest else ''}, {sources[which]} "
                             f"({describe_proposal(proposals[which])}): {said}")
                notes[index] = note
                if click is None:
                    continue
                taken |= region
                grown |= _grown(region)
                located[index] = region
                fitted[index] = fit[:2] + (rest,)
                shapes[index] = fit[2]
                fitted_rivals[index] = fit
                pending.remove(index)
                progress = True
                points[index], labels[index] = click, [1] * len(click)
                self.say(f"object {index}: located by its mesh at {click} ({note})")
            if not progress:
                if rest or not any((proposal & taken).any() for proposal in proposals):
                    break
                # Last, what is left of the proposals other objects took.
                rest = True
        # Not located: the person is asked (NeedClick), once the rest are.
        unsure = {index: (f"not located by its mesh ({notes[index]})", []) for index in pending}
        rivals = {index: fit[3] for index, fit in fitted_rivals.items()}
        # The one whose scale is furthest off the others' (LOCATE_SCALE_AGREE),
        # the lower scored of two as far off, is located again at theirs. One
        # kept at its own, none fitting at theirs, is the one in doubt: it sets
        # no scale for the others. RR-S1-1's ring, located on the hand at 1.55,
        # was kept so; then the bar, fitted right at 0.59, was located again at
        # the ring's, on the hand at the image's edge (2026-10-31).
        again, kept = set(), set()
        while self.same_scale:
            off = []
            for index, (score, scale, _) in fitted.items():
                others = [fitted[j][1] for j in fitted if j != index and j not in kept]
                if index in again or not others:
                    continue
                at = float(np.exp(np.median(np.log(others))))
                ratio = abs(np.log(scale / at))
                if ratio > np.log(LOCATE_SCALE_AGREE):
                    off.append((round(ratio, 6), -score, index, at))
            if not off:
                break
            _, _, index, at = max(off)
            again.add(index)
            why = []
            others = clicked_taken | np.any([located[j] for j in located if j != index], axis=0)
            others_grown = clicked_taken | np.any([_grown(located[j]) for j in located if j != index], axis=0)
            # What is left of the others' proposals is proposed too: RR-S1-1's
            # ring, with the bar in one proposal, was first located on the hand
            # holding the bar, and is what is left of it once the bar is located.
            click, region, note, fit = locate_with_mesh(
                models[index], proposals, metres, K, clicked, others, robot, why, True,
                (at / LOCATE_SCALE_AGREE, at * LOCATE_SCALE_AGREE), grown=others_grown, image=rgb)
            for which, said in sorted(why):
                self.say(f"  object {index}, again at the others' scale {at:.2f}, {sources[which]} "
                         f"({describe_proposal(proposals[which])}): {said}")
            if click is None:
                self.say(f"object {index}: kept at scale {fitted[index][1]:.2f}, off the others' "
                         f"{at:.2f}; none fits at theirs ({note})")
                kept.add(index)
                continue
            located[index] = region
            fitted[index] = fit[:2] + (fitted[index][2],)
            shapes[index] = fit[2]
            rivals[index] = fit[3]
            points[index], labels[index] = click, [1] * len(click)
            self.say(f"object {index}: located again by its mesh at {click}, at the others' scale "
                     f"({note})")
        # One located weakly is looked for again once the others are, what is
        # left of their proposals proposed too, at their scale if they share
        # one; where it fits
        # better by LOCATE_BETTER, it is there: RR-S1-1's ring, located first on
        # the hand holding the bar (0.68) -- the bar and the ring on it one
        # proposal -- is what is left of that once the bar is located.
        for index in sorted(fitted, key=lambda i: fitted[i][0]):
            score = fitted[index][0]
            others = [fitted[j][1] for j in fitted if j != index and j not in kept]
            if score >= LOCATE_WEAK or not others:
                continue
            at = float(np.exp(np.median(np.log(others))))
            why = []
            taken_now = clicked_taken | np.any([located[j] for j in located if j != index], axis=0)
            grown_now = clicked_taken | np.any([_grown(located[j]) for j in located if j != index], axis=0)
            click, region, note, fit = locate_with_mesh(
                models[index], proposals, metres, K, clicked, taken_now, robot, why, True,
                (at / LOCATE_SCALE_AGREE, at * LOCATE_SCALE_AGREE) if self.same_scale else None, grown=grown_now,
                image=rgb)
            if click is None or fit[0] < score + LOCATE_BETTER:
                continue
            for which, said in sorted(why):
                self.say(f"  object {index}, looked for again ({score:.2f}), {sources[which]} "
                         f"({describe_proposal(proposals[which])}): {said}")
            located[index] = region
            fitted[index] = fit[:2] + (True,)
            shapes[index] = fit[2]
            rivals[index] = fit[3]
            kept.discard(index)             # at the others' scale now
            points[index], labels[index] = click, [1] * len(click)
            self.say(f"object {index}: located again by its mesh at {click}, fitting better than at "
                     f"{score:.2f} ({note})")
        self.locate_scales = {index: fit[1] for index, fit in fitted.items()}
        self.locate_shapes = shapes
        self.say(f"located by mesh in {time.time() - started:.1f} s")
        # Located, but not for sure: weakly, or as well somewhere else.
        for index, (score, scale, _) in fitted.items():
            here = [points[index]]
            if score < LOCATE_ASK_BELOW:
                unsure[index] = (f"fits its place only weakly ({score:.2f})", here)
            elif rivals.get(index):
                said = ", ".join(f"{other:.2f}" for _, other in rivals[index])
                unsure[index] = (f"fits {len(rivals[index]) + 1} places about as well ({score:.2f}; {said})",
                                 here + [click for click, _ in rivals[index]])
            elif index in kept:
                unsure[index] = (f"located at scale {scale:.2f}, off the other parts'", here)
        if unsure:
            for index, (why, _) in sorted(unsure.items()):
                self.say(f"object {index}: unsure -- {why}; the person is asked to click it")
            raise NeedClick([{"object": index, "why": why, "candidates": [np.asarray(c).tolist() for c in found]}
                             for index, (why, found) in sorted(unsure.items())],
                            [[] if index in unsure else np.asarray(p).tolist() for index, p in enumerate(points)])
        return points, labels, located

    def separate_parts(self, rgb, points, labels):
        """Each part's points and labels, the clicks of the other parts its
        segment holds added as negative points; see LOCATE_REGION_GROW_PX."""
        if len(points) < 2:
            return points, labels
        try:
            h, w = rgb.shape[:2]
            masks = (as_numpy(self.pipeline.preview_user_masks(rgb, points, labels)) > 0).reshape(len(points), h, w)
        except Exception as exc:
            self.say(f"parts not checked for one segment holding two ({type(exc).__name__}: {exc})")
            return points, labels
        out_points, out_labels = [], []
        for index in range(len(points)):
            held = [click for other in range(len(points)) if other != index
                    for click, label in zip(points[other], labels[other])
                    if label and masks[index][min(max(int(round(click[1])), 0), h - 1),
                                              min(max(int(round(click[0])), 0), w - 1)]]
            out_points.append(list(points[index]) + [list(click) for click in held])
            out_labels.append(list(labels[index]) + [0] * len(held))
            if held:
                self.say(f"object {index}: its segment holds {len(held)} click(s) of other parts, "
                         "given it as negative points")
        return out_points, out_labels

    def discover_obstacles(self, rgb, metres, K, points, labels, located, robot, count):
        """Up to `count` obstacles in the init frame, each {"region": its pixels,
        "cuts": the planes (normal, point) cut from it, "box": (T_cam_box,
        extent), "missed": 0}, as follow_obstacles() follows them; see
        OBSTACLE_MIN_PX. `points` and `labels`, the parts' clicks, and
        `located`, {index: region} of those located by mesh, are the parts'
        pixels, which no obstacle takes."""
        if count <= 0:
            return []
        started = time.time()
        proposals, sources, _ = self.proposals(rgb, metres)
        if proposals is None:
            self.say("no obstacles looked for: the segmenter offers no mask proposals")
            return []
        cam, valid = backproject(metres, K)
        depth = cam[..., 2]
        taken = np.zeros(metres.shape, bool)
        for region in located.values():
            taken |= region
        try:
            logits = self.pipeline.preview_user_masks(rgb, points, labels)
            taken |= np.any((as_numpy(logits) > 0).reshape(len(points), *metres.shape), axis=0)
        except Exception as exc:
            self.say(f"obstacles: no segments of the parts ({type(exc).__name__}: {exc}); "
                     "only their located regions are left out")
        if robot is not None:
            taken |= robot
        rng = np.random.default_rng(0)
        found, kept, left = [], np.zeros(metres.shape, bool), {}
        for which in sorted(range(len(proposals)), key=lambda k: -int(proposals[k].sum())):
            mask = proposals[which] & ~taken & valid
            if mask.sum() < OBSTACLE_MIN_PX:
                continue
            view = observe(mask, metres, K)
            if view is None:
                continue
            _, _, keep, plane, note = view
            if note.startswith("flat"):
                continue                          # lies flat on its support: no obstacle
            cuts = [plane] if note == "support surface removed" else []
            # A surface within it -- the bench a box stands on, depth-joined
            # to it -- is no obstacle, what stands on it may be: in front of
            # the plane its pixels hold most of, where that is a surface
            # (OBSTACLE_FLAT_M), alone.
            pts = cam[keep]
            surface = _ring_plane(pts[rng.permutation(len(pts))[:LOCATE_BOX_POINTS]], rng)
            if surface is not None:
                n, p = surface
                on = cam[keep & (np.abs((cam - p) @ n) < PLANE_TOL_M)]
                on = on[rng.permutation(len(on))[:LOCATE_BOX_POINTS]] - p
                if (len(on) >= MIN_BOX_POINTS
                        and np.ptp(on @ np.linalg.svd(on, full_matrices=False)[2][0]) >= OBSTACLE_FLAT_M):
                    keep = keep & ((cam - p) @ n < -PLANE_TOL_M)
                    cuts.append(surface)
            for piece in depth_pieces(keep, depth, OBSTACLE_MIN_PX):
                if (piece & kept).sum() > OBSTACLE_OVERLAP * piece.sum():
                    continue
                pts = cam[piece]
                sample = pts[rng.permutation(len(pts))[:LOCATE_BOX_POINTS]]
                R, centre, extent = min_obb(sample)
                size = " x ".join(f"{e * 1000:.0f}" for e in sorted(extent, reverse=True))
                said = f"{sources[which]} ({describe_proposal(piece)}), {size} mm"
                why = None
                if not OBSTACLE_EXTENT_M[0] <= extent.max() <= OBSTACLE_EXTENT_M[1]:
                    why = "not an obstacle's size"
                elif np.median(pts[:, 2]) > OBSTACLE_MAX_DEPTH_M:
                    why = "too far"
                elif plane is not None and -np.min((pts - plane[1]) @ plane[0]) < OBSTACLE_MIN_HEIGHT_M:
                    why = "no higher than its support"
                elif extent.max() >= OBSTACLE_FLAT_M:
                    mid = sample.mean(0)
                    normal = np.linalg.svd(sample - mid, full_matrices=False)[2][2]
                    if np.mean(np.abs((sample - mid) @ normal) < PLANE_TOL_M) >= OBSTACLE_FLAT_SHARE:
                        why = "a surface"
                if why is not None:
                    left[why] = left.get(why, 0) + 1
                    continue
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = R, centre
                found.append({"region": piece, "cuts": cuts, "box": (T, extent), "missed": 0})
                kept |= piece
                self.say(f"obstacle {len(found)}: {said}")
                if len(found) >= count:
                    break
            if len(found) >= count:
                break
        self.say(f"{len(found)} obstacle(s) found in {time.time() - started:.1f} s"
                 + (f"; left out: {', '.join(f'{n} {why}' for why, n in left.items())}" if left else ""))
        return found

    def follow_obstacles(self, frame, regions=None):
        """Each obstacle found at init, found again on this frame by the depth
        near where it was last; see OBSTACLE_FOLLOW_PX. `regions`, the fitted
        meshes' (carried_regions()), are the parts' too."""
        obstacles = getattr(self, "obstacles", None)
        if not obstacles:
            return
        try:
            h, w = frame.depth.shape
            metres = as_numpy(frame.depth) / frame.depth_factor
            K = np.asarray(frame.intrinsics, dtype=np.float64)
            masks = as_numpy(getattr(frame, "mask", None))
            if masks.ndim == 4:
                masks = masks[:, 0]
        except Exception:
            return
        taken = np.zeros((h, w), bool)
        if masks.ndim == 3 and masks.shape[1:] == (h, w):
            taken |= np.any(masks > 0, axis=0)
        for part in [*(regions or []), getattr(self, "track_robot", None)]:
            if part is not None and part.shape == (h, w):
                taken |= part
        cam, valid = backproject(metres, K)
        depth = cam[..., 2]
        grow = np.ones((2 * OBSTACLE_FOLLOW_PX + 1,) * 2, np.uint8)
        # Each pixel is the nearest obstacle's to look at (OBSTACLE_SMOOTH_FULL_DEG).
        owner = None
        if len(obstacles) > 1:
            owner = np.argmin(np.stack([cv2.distanceTransform((~obstacle["region"]).astype(np.uint8), cv2.DIST_L2, 3)
                                        for obstacle in obstacles]), axis=0)
        for number, obstacle in enumerate(obstacles):
            try:
                last = obstacle["region"]
                v, u = np.nonzero(last)
                near = np.zeros((h, w), bool)
                near[max(v.min() - OBSTACLE_FOLLOW_PX, 0):v.max() + OBSTACLE_FOLLOW_PX + 1,
                     max(u.min() - OBSTACLE_FOLLOW_PX, 0):u.max() + OBSTACLE_FOLLOW_PX + 1] = True
                near &= valid & ~taken
                if owner is not None:
                    near &= owner == number
                for n, p in obstacle["cuts"]:
                    near[near] = (cam[near] - p) @ n < -PLANE_TOL_M
                was = cv2.dilate(last.astype(np.uint8), grow) > 0
                pieces = depth_pieces(near, depth, OBSTACLE_MIN_PX)
                # On where it was (intersection over union), else -- moved off
                # it -- within its grown region and as big within
                # OBSTACLE_FOLLOW_SIZE: the bench beside an arm is mostly in
                # that region too, three times its size.
                shares = [((piece & last).sum() / max(piece.sum(), last.sum()), k, piece)
                          for k, piece in enumerate(pieces)]
                if not any(share >= OBSTACLE_FOLLOW_SHARE for share, _, _ in shares):
                    shares = [((piece & was).sum() / piece.sum(), k, piece) for k, piece in enumerate(pieces)
                              if 1 / OBSTACLE_FOLLOW_SIZE <= piece.sum() / last.sum() <= OBSTACLE_FOLLOW_SIZE]
                found = None
                # The most on where it was, of an obstacle's size: joined in
                # the depth to the bench, an arm is not that.
                for share, _, piece in sorted(shares, key=lambda item: -item[0]):
                    if share < OBSTACLE_FOLLOW_SHARE:
                        break
                    pts = cam[piece]
                    R, centre, extent = min_obb(pts[::max(1, len(pts) // OBSTACLE_BOX_POINTS)])
                    if OBSTACLE_EXTENT_M[0] <= extent.max() <= OBSTACLE_EXTENT_M[1]:
                        found = piece
                        break
                if found is None:
                    obstacle["missed"] += 1
                    continue
                piece = found
                # The last box's axes, and so much of the turn (OBSTACLE_SMOOTH_FULL_DEG).
                before = obstacle["box"][0][:3, :3]
                P = max(_signed_permutations(), key=lambda P: np.trace(before.T @ R @ P))
                R, extent = R @ P, np.abs(P).T @ extent
                turn = cv2.Rodrigues(before.T @ R)[0].ravel()
                share = np.clip(np.degrees(np.linalg.norm(turn)) / OBSTACLE_SMOOTH_FULL_DEG,
                                OBSTACLE_SMOOTH_MIN_ALPHA, 1.0)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = before @ _rotation(share * turn), centre
                obstacle.update(region=piece, box=(T, extent), missed=0)
                taken |= piece
            except Exception:
                traceback.print_exc()
                obstacle["missed"] += 1

    def obstacle_replies(self, frame):
        """The obstacles as the reply carries them: {"id", "lost", "bbox"},
        the box as box_in_camera() gives a part's."""
        K = np.asarray(frame.intrinsics, dtype=np.float64)
        out = []
        for number, obstacle in enumerate(getattr(self, "obstacles", None) or [], 1):
            T, extent = obstacle["box"]
            out.append({"id": number, "lost": obstacle["missed"] >= OBSTACLE_LOST_FRAMES,
                        "bbox": box_in_camera(np.eye(4), T, extent, K, frame.depth.shape)})
        return out

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
        # The robot's pixels on this frame, where the client sends them: a
        # gripper on or by a part is in that part's tracked mask.
        self.track_robot = decode_robot_mask(meta, frame.depth.shape)
        regions = self.carried_regions(motions, frame)
        motions = self.check_carried_meshes(motions, frame, regions)
        # The robot's poses of parts it set down (MODEL_CHECK_FAILS).
        self.take_priors(meta, frame.id)
        motions = self.apply_priors(motions, frame, regions)
        # The person's ways round (BINDING_CONFIDENCE): {"object": index, "hypothesis": id,
        # "by": whose -- "person" where a client older than it does not say}.
        for bind in meta.get("bind") or []:
            try:
                self.pending_binds[int(bind["object"])] = (str(bind["hypothesis"]),
                                                           str(bind.get("by") or "person")[:32])
            except (KeyError, TypeError, ValueError, AttributeError):
                print(f"[server] unusable bind {bind!r}", flush=True)
        self.update_bindings(motions, frame, regions, self.track_robot,
                             observe=self.frame_id % BINDING_EVERY == 0)
        self.retry_pending_meshes(motions, frame, regions)
        # A refit taken on this frame has its binding, the old one's ways round
        # carried, before this frame's reply: else that reply has no doubts.
        self.update_bindings(motions, frame, regions, self.track_robot, observe=False)
        self.follow_obstacles(frame, regions)
        if self.frame_id % SURFACE_EVERY == 0:
            self.record_surfaces(motions, frame, regions)
        self.frame_id += 1
        motions = self.smoother(motions)
        return motions @ self.anchors, frame, motions

    def without_robot(self, mask):
        """`mask` less the robot's pixels on this frame (handle_track()), where
        the client sent them. Kept in, the fingers around a held part, or by
        one just let go of, are points off its mesh: the carried check
        withheld PP-S4-1's tester in the hand (p95 146 mm) and fitted it again
        there, coarse and wrong, and the tester just placed in PP-S1-1 was
        withheld as the gripper left (fitness 0.54), refitted 180 deg off."""
        robot = getattr(self, "track_robot", None)
        return mask if robot is None or robot.shape != mask.shape else mask & ~robot

    def handle_preview(self, meta, parts):
        if self.pipeline is None:
            self.build_pipeline()
        rgb = decode_rgb(parts[0])
        logits = self.pipeline.preview_user_masks(rgb, meta["points"], meta["labels"])
        mask = (logits.squeeze(1) > 0).cpu().numpy().astype(np.uint8)
        out = {"mask_shape": list(mask.shape), "mask_sum": mask.sum(-1).sum(-1).tolist()}
        if meta.get("dump_dir"):
            os.makedirs(meta["dump_dir"], exist_ok=True)
            cv2.imwrite(f"{meta['dump_dir']}/rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay[mask[0] > 0] = (0, 0, 255)
            cv2.imwrite(f"{meta['dump_dir']}/mask_overlay.png", overlay)
            out["dumped"] = True
        return out

    def frame_reply(self, cmd, meta, parts, t0):
        """The reply to an init or a track request, its frame worked on."""
        self.track_thread = threading.get_ident()
        if cmd == "init":
            poses, frame, motions = self.handle_init(meta, parts)
            self.log_gpu_memory("after init")
            if torch.cuda.is_available():
                self.gpu_after_init, self.gpu_growth_said = torch.cuda.memory_allocated() / 2**20, 0
        else:
            poses, frame, motions = self.handle_track(meta, parts)
            self.trace_gpu_frame(frame.id)
        lost = [bool(getattr(o, "lost", False)) for o in self.pipeline.objects]
        bboxes = self.bboxes(motions, frame)
        lost = self.recognition_lost(lost, bboxes)
        reply = {
            "ok": True,
            "frame_id": self.frame_id - 1,
            "poses": np.asarray(poses, dtype=np.float64).tolist(),
            "lost": lost,
            "bboxes": bboxes,
            "model_errors": self.model_errors,
            # Found with no click; see OBSTACLE_MIN_PX.
            "obstacles": self.obstacle_replies(frame),
            "latency_ms": round((time.time() - t0) * 1000, 1),
            # The clicks used, those located by mesh among them, and how.
            **({"points": self.init_points, "init_report": self.init_report}
               if cmd == "init" else {}),
        }
        self.last_reply = reply
        return reply

    def settle_track(self):
        """Wait for a frame still worked on (SERVER_REPLY_SHARE) before
        anything else touches the pipeline."""
        job = getattr(self, "track_job", None)
        self.track_job = None
        if job is None:
            return
        try:
            job["future"].result()
        except Exception:
            traceback.print_exc()

    def answer_track(self, meta, parts, t0):
        """A track request's reply: in SERVER_REPLY_SHARE of the client's
        meta["timeout_ms"], stale where its frame is not done by then or an
        earlier one still is; without it, when the frame is done."""
        allowance = meta.get("timeout_ms")
        if allowance is None:
            self.settle_track()
            return self.frame_reply("track", meta, parts, t0)
        allowance = float(allowance) / 1000.0
        job = getattr(self, "track_job", None)
        if job is not None and not job["future"].done():
            return self.stale_reply(job, allowance, t0)
        if job is not None:
            self.track_job = None
            try:
                job["future"].result()
            except Exception as exc:
                print(f"[server] frame {job['frame_id']}, answered stale, failed: {type(exc).__name__}: {exc}",
                      flush=True)
        if getattr(self, "track_executor", None) is None:
            self.track_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="track",
                                                     initializer=no_grad_thread)
        job = {"future": self.track_executor.submit(self.frame_reply, "track", meta, parts, t0),
               "started": time.time(), "frame_id": self.frame_id, "dumped": 0}
        self.track_job = job
        try:
            reply = job["future"].result(timeout=SERVER_REPLY_SHARE * allowance)
        except FutureTimeout:
            return self.stale_reply(job, allowance, t0)
        self.track_job = None
        return reply

    def stale_reply(self, job, allowance, t0):
        """The last frame's reply, marked stale (SERVER_REPLY_SHARE): "stale":
        {"busy_s": how long the frame worked on has taken, "working_on":
        its frame id, "why"}. Past the client's whole allowance, the stack
        the frame is held in is printed, once each allowance."""
        busy = time.time() - job["started"]
        if busy >= allowance * (job["dumped"] + 1):
            job["dumped"] += 1
            stack = sys._current_frames().get(getattr(self, "track_thread", None))
            print(f"[server] frame {job['frame_id']} worked on for {busy:.1f} s, past the client's {allowance:.0f} s; "
                  "where it is:\n" + ("".join(traceback.format_stack(stack)) if stack is not None else "(not found)"),
                  flush=True)
        reply = dict(getattr(self, "last_reply", None) or {"ok": False, "error": "no frame replied to yet"})
        reply["stale"] = {"busy_s": round(busy, 1), "working_on": job["frame_id"],
                          "why": "the server is still working on an earlier frame; this is the last frame's reply"}
        reply["latency_ms"] = round((time.time() - t0) * 1000, 1)
        return reply

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
                if cmd == "track":
                    self.sock.send_json(self.answer_track(meta, msg[1:], t0))
                    continue
                # Nothing else touches the pipeline while a frame is worked on.
                self.settle_track()
                if cmd == "reset":
                    self.archive_session()
                    self.release_pipeline()
                    self.log_gpu_memory("after reset")
                    self.report_gpu_held("after reset")
                    self.sock.send_json({"ok": True})
                    continue
                if cmd == "preview":
                    self.sock.send_json({"ok": True, **self.handle_preview(meta, msg[1:])})
                    continue
                if cmd != "init":
                    raise ValueError(f"unknown cmd: {cmd}")
                self.sock.send_json(self.frame_reply("init", meta, msg[1:], t0))
            except NeedClick as exc:
                # Not a failure: which objects to click, and the clicks of the rest.
                self.sock.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                     "need_click": exc.requests, "points": exc.points,
                                     "init_report": self.init_report})
            except Exception as exc:
                traceback.print_exc()
                self.sock.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                     **({"init_report": self.init_report} if cmd == "init" else {})})
            finally:
                if cmd != "track":
                    self.init_report = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/pipeline/pipeline_test2.yaml")
    ap.add_argument("-b", "--bind", default="tcp://0.0.0.0:5555")
    ap.add_argument("--archive", default=None,
                    help="directory to file each mesh's settled surface and shapes in, and read them from "
                         "(see BINDING_FEATURE_PX); none kept without it")
    ap.add_argument("--trace-gpu-memory", action="store_true",
                    help="keep where each live GPU block was allocated, and say it, with what holds it, when a "
                         f"release leaves more than {GPU_HELD_WARN_MIB} MiB allocated; the memory is logged every "
                         f"{GPU_TRACE_EVERY_FRAMES} track frames too (see GPU_HELD_WARN_MIB)")
    args = ap.parse_args()
    server = PoseServer(args.config, args.bind, archive=args.archive, trace_gpu=args.trace_gpu_memory)
    try:
        server.run()
    finally:
        server.settle_track()
        server.archive_session()
        if server.mesh_fit_executor is not None:
            server.mesh_fit_executor.shutdown(wait=True, cancel_futures=True)
