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
"""

import argparse
import base64
import copy
import gc
import hashlib
import itertools
import json
import time
import zlib
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

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
# Rotations refined from each start of the pose search, beside the best
# MODEL_KEEP of all: at its own size a mesh can sit snugly on a patch of a
# part twice as big, cheaper than any start at the part's size.
MODEL_KEEP_PER_START = 2
# Which end of a long part is which. A long part's ends can differ by little
# but a thin feature -- the tester's probe -- that few points fall on, so
# the cost hardly tells the part from itself turned end for end. Where its
# box is this many times longer than wide, the end-for-end pose is refitted
# too, and taken if it costs at most MODEL_ENDS_COST_SLACK more and the view
# holds it better (MODEL_ENDS_REGION_NOISE); where the view cannot tell, if
# its profile -- how wide the part is along its length, where the camera
# sees it -- matches the observed one better by MODEL_ENDS_MARGIN. A profile
# is robust to what a reconstruction gets wrong: size, proportions, a probe
# a few millimetres off.
MODEL_ENDS_ELONGATION = 1.5
MODEL_ENDS_COST_SLACK = 0.05
# Up to this much more, a turn is still taken where its asymmetric region
# (MODEL_ENDS_REGION_NOISE) and its profile both say it is the part's way
# round: PP-S1-3's tester, lying, its probe to the right, cost 0.070 more
# turned so, its probe 6.9 mm off the mask against 11.7 and its profile 0.08
# against 0.12, and was left the wrong way round.
MODEL_ENDS_CLEAR_COST_SLACK = 0.1
MODEL_ENDS_MARGIN = 0.8
MODEL_PROFILE_BINS = 16
# What tells a long part's ends apart is wherever the two ways round differ:
# the tester's probe, the notch cut into the sample's end, the bar's tip, a
# step. Rather than a test for each kind -- the mask's distance from a
# protruding feature, how far an end's middle is cut back -- two tests for
# any, in turn:
#
#   1. What each fitted pose has and the other has not (difference_support()):
#      its points off the other pose's by more than MODEL_ENDS_REGION_NOISE
#      times the median such distance, the median being how far the two lie
#      apart where they agree. The mask is asked how well it holds them where
#      that pose puts them, a point's support falling off linearly to none
#      MODEL_ENDS_REGION_PX pixels away; one in the support -- a probe turned
#      down into the bench -- holds nothing. A pose's own that is hidden
#      behind something says nothing, and the other's alone then tells:
#      held, that pose is the part's; not held, the hidden one is. A turn
#      held MODEL_ENDS_REGION_MARGIN better than the pose is taken; one held
#      worse says nothing of the other turn (PP-S4-1's tester: one turn held
#      worse, the other, the right one, better). The mask, not the observed
#      points: on the sample's notch they told the wrong way round in
#      PP-S1-3, the plate's edges a pixel apart in the two refits as large
#      as the notch. Taken on the mesh turned in place instead, the tester's
#      lopsided body came out with its probe, and both ways round held it.
#   2. The shape of the ends (end_signature()), for what is too small for
#      that against two refits a pixel or two apart -- the sample's 4 px
#      notch, which SAM2's mask filled in PP-S1-2: the width cut into
#      MODEL_ENDS_BANDS bands, how far each band's end stops short of that
#      end's furthest, on the observed points against the mesh as each pose
#      places it. Each end is measured against itself, so a pose a pixel off
#      does not change it. A turn MODEL_ENDS_SIGNATURE_RATIO as far off or
#      less, and MODEL_ENDS_SIGNATURE_MIN_M nearer, is taken -- the sample's
#      1.0-1.9 mm against 3.3-3.8, the bar's 4.3 against 10.7 -- or kept.
#
# Neither telling, the profile decides, and the other way round is sent in
# doubt (model_ends_doubt), as a spin is: PP-S4-2's tester, its probe behind
# the box one way round and, thin, half in the mask the other. The points
# are MODEL_FEATURE_SAMPLES of the mesh's surface, not the MODEL_SAMPLES the
# fit uses: the tester's probe is under 1% of its area, about 20 of those.
MODEL_FEATURE_SAMPLES = 20000
MODEL_ENDS_REGION_NOISE = 5.0
MODEL_ENDS_REGION_MIN = 30
MODEL_ENDS_REGION_PX = 3.0
MODEL_ENDS_REGION_MARGIN = 0.2
# A region of one pose's that lies this much inside the other's silhouette
# (its convex hull, in the image) is a cut-out of the other's: held by the
# mask on its own, with the other's region hidden, it tells nothing (see
# settle_ends()) --
MODEL_ENDS_INSIDE_SHARE = 0.5
# ... and covers no more than this many pixels: a cut-out a mask fills is
# small -- the sample's notch region 14-24 px in PP-S1-3 -- where what tells
# a part apart is not -- the tester's, 145 px.
MODEL_ENDS_FILL_MAX_PX = 60
# Before both tests, and the profile: which end the part sticks out of. A
# reconstruction gets a thin feature only roughly -- the tester's probe came
# out some 15 mm across from the real one in PP-S1-1, -3 and -4 -- and a
# check of where its points fall, the camera seeing past them or observed
# points near them, turned seven of the ten PP testers the wrong way round
# (2026-10-30). Which end the probe sticks out of it gets right. So where
# the mesh sticks out past one end of its body (the MODEL_ENDS_TIP
# percentile along its length) by MODEL_ENDS_STICK_MIN_M or more, and past
# the other by at most MODEL_ENDS_STICK_SHARE of that -- the tester's 22 mm
# against 0.6, not the sample's 2.9 against 2.1 nor the bar's 10.3 against
# 9.7 -- the observed points past each end are counted, across the section
# wherever they are: within the body's radius and MODEL_ENDS_STICK_MARGIN_M
# of its axis, MODEL_ENDS_STICK_FROM_M past the end out to
# MODEL_ENDS_STICK_REACH times the mesh's own, MODEL_ENDS_STICK_HEIGHT_M or
# more off the support (a row of bench 6 mm above the plane fitted stuck
# out of PP-S4-2's tester), and not the robot's or another part's pixels.
# MODEL_ENDS_STICK_COUNT or more past one end and at most
# MODEL_ENDS_STICK_SHARE as many past the other tell the way round: the
# tester's 19-31 against 0-2 in the eight PP scenes its probe shows in,
# and on four masks of PP-S1-4's the tests after turned it the wrong way
# round on -- one cut by colour, its probe left out, the right way round
# costing 0.184 more (MODEL_ENDS_STICK_COST_SLACK).
MODEL_ENDS_STICK_MIN_M = 0.008
MODEL_ENDS_STICK_SHARE = 0.2
MODEL_ENDS_STICK_MARGIN_M = 0.01
MODEL_ENDS_STICK_FROM_M = 0.005
MODEL_ENDS_STICK_REACH = 2.0
MODEL_ENDS_STICK_HEIGHT_M = 0.01
MODEL_ENDS_STICK_COUNT = 10
MODEL_ENDS_STICK_COST_SLACK = 0.25
# Where the depth does not show it, the colour may: in PP-S4-2 the tester's
# probe sticks out of its visible end two pixels thin -- one row of depth 1 cm
# above the bench, out of SAM2's mask -- and with its other end behind the box
# no test told its ends apart; on the probe's absence from the mask the end
# signature put the blunt end there, the probe behind the box. So where the
# observed points past the ends do not tell (end_stick()), the image is asked
# (end_streak()): every MODEL_ENDS_STREAK_STEP_M along the axis past each end,
# within MODEL_ENDS_STREAK_BAND_PX across it, a pixel MODEL_ENDS_STREAK_CONTRAST
# grey levels off the mean of the two MODEL_ENDS_STREAK_SIDE_PX beside it, and
# those two alike -- a thin line through the background -- not the robot's nor
# another part's and nothing nearer than MODEL_ENDS_STREAK_FRONT_M in front of
# the axis there, counts; the counts tell as end_stick()'s do. Along the
# probe's end the nine other PP scenes counted 19-31, along the other 0-3;
# PP-S4-2, 21 along its visible end, the one its fit had put the blunt end at.
MODEL_ENDS_STREAK_STEP_M = 0.001
MODEL_ENDS_STREAK_BAND_PX = 7
MODEL_ENDS_STREAK_SIDE_PX = 4
MODEL_ENDS_STREAK_CONTRAST = 25
MODEL_ENDS_STREAK_FRONT_M = 0.015
MODEL_ENDS_BANDS = 5
MODEL_ENDS_TIP = 99.0
MODEL_ENDS_SIGNATURE_MIN_M = 0.001
MODEL_ENDS_SIGNATURE_RATIO = 0.65
# None of that tells a mesh's ends apart where the mesh has none to tell: the
# bar's reconstruction comes to a point at both ends, 11.3 and 12.0 mm past
# its body, the bands of its two ends 2.8% of its width apart. The tests then
# answer to the view, not the part: an end hidden in the hand holding it
# "matched" (5.2 mm against 26.8 in RI-S1-1 to RI-S3-2), and in RR-S1-1 the
# other way round came out coarse on its bumps and was dropped -- the bar
# published either way round, never in doubt. So on loading, a long mesh's
# ends are compared with each other: the one sticking out of its body (the
# MODEL_ENDS_TIP percentile along its length) and the other by at most
# MODEL_ENDS_STICK_SHARE of that, or their bands (end_signature()'s) apart by
# MODEL_ENDS_DISTINCT_SHARE of its width or more, and they are told apart --
# the tester's probe, 9.7 mm against 0.3; the sample's notch, 6.4%. Else the
# mesh's ends are alike, the tests are not asked, and the pose goes out with
# the mesh's long box axis to the image's right (MODEL_ENDS_LIKE_UP: up),
# the other way round in doubt (model_ends_doubt), and model_symmetry saying
# that the mesh, not the view, leaves it open.
MODEL_ENDS_DISTINCT_SHARE = 0.045
MODEL_ENDS_LIKE_UP = 0.3
# And about which axis a mesh is round: turned about a box axis through its
# box's centre, by every MODEL_SYMMETRY_STEP_DEG up to a half turn, nine in
# ten of its points (MODEL_SYMMETRY_PERCENTILE) within MODEL_TOL_M of its
# surface, at its fitted size. The fit cannot tell such a turn from the pose,
# and the tracked turn about the axis drifted: RI-S4-1's bar 9.1 deg about its
# length over ten frames of one still image, RI-S3-2's ring 5.2 deg about
# its own axis, where nothing else of either moved. The ring's spread is
# 1.4 mm, the bar's about its length 2.9; the tester's about its length 8.5,
# a near-square section, its faces in view. About such an axis the published
# turn is held (bboxes()): each reply turns the pose about it as little as
# lies between it and the last one sent, the first with another of the
# mesh's box axes as near the image's up as it goes. A mesh stretched
# unevenly across the axis, by more than MODEL_SYMMETRY_STRETCH, is round
# about it no longer. Neither is a turn about it any doubt (model_spin_doubt).
MODEL_SYMMETRY_STEP_DEG = 20
MODEL_SYMMETRY_PERCENTILE = 90
MODEL_SYMMETRY_STRETCH = 1.02
# Which way round a long part is about its length. The tester's two long
# sides differ only by its probe and the groove under it: with its lower
# half behind PP-S4-1's occluder, the demonstration mesh turned half round
# about its length cost 0.4779 and the right way round 0.4780 -- the view
# cannot tell them apart, and no profile does either. So where a part's box
# is MODEL_ENDS_ELONGATION times longer than wide, the pose turned half round
# about its length is refitted too. Cheaper by more than MODEL_SPIN_COST_SLACK,
# it is taken instead; within that either way, the fit is sent in doubt
# (model_spin_doubt: that pose relative to it), for the person to say which.
# Seen whole (PP-S1-1, rendered), the wrong way round cost 0.054 more.
MODEL_SPIN_COST_SLACK = 0.02
# In doubt, the way round with the mesh's own +z up -- away from the support,
# by this much more of the unit z -- is the one sent, the other in doubt.
# The demonstration meshes are as they stood: the plate flat on its +z
# side, the tester upright. A plate is the same either side up to the view,
# and the sample, laid alike in seven scenes, came out upside down in three
# on costs 0.001-0.009 apart. A part on its side, its z level both ways
# round, is left to the cost.
MODEL_SPIN_UP_MARGIN = 0.5
# Where the cost does not tell, the image may. Turned half round about its
# length the tester differs from itself on a band of its surface: its groove,
# sunk where the other way round is flat. Lying groove up in PP-S1-4, the
# groove plain to see, and standing groove to the camera behind PP-S4-1's box,
# it went out groove away, in doubt; the depth told nothing apart -- what each
# way round had of its own was seen past alike in PP-S1-2, its groove down.
# So in doubt, the band of a way round sunk MODEL_SPIN_RECESS_M or more under
# the other, seen and nothing in front of it, MODEL_SPIN_BAND_PX or more of
# it, is looked along (sunk_way()): at every pixel of its length, anywhere
# across it and MODEL_SPIN_LINE_SIDE_PX more, a pixel MODEL_SPIN_LINE_CONTRAST
# grey levels off the mean of the two MODEL_SPIN_LINE_SIDE_PX beside it, those
# two alike -- a line, not an edge. Lined on MODEL_SPIN_LINE_SHARE of it or
# more, that way round is the part's; on MODEL_SPIN_FLAT_SHARE or less, the
# other; else it stays in doubt. The ten PP scenes: the groove's band lined on
# 73-100% where the groove faced the camera, 35% on PP-S1-3's tester half
# turned from it, 0% on PP-S1-2's groove down.
MODEL_SPIN_RECESS_M = 0.002
MODEL_SPIN_BAND_PX = 60
MODEL_SPIN_LINE_SIDE_PX = 3
MODEL_SPIN_LINE_CONTRAST = 12
MODEL_SPIN_LINE_SHARE = 0.3
MODEL_SPIN_FLAT_SHARE = 0.1
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


def _twist_spread(points, box):
    """Per box axis, how far the mesh comes from itself turned about it
    through the box's centre: the MODEL_SYMMETRY_PERCENTILE-th percentile of
    its points' distances from its surface, the most over the turns
    MODEL_SYMMETRY_STEP_DEG apart up to a half turn; at the mesh's own size."""
    tree = cKDTree(points)
    B, c = box[:3, :3], box[:3, 3]
    out = np.zeros(3)
    for k in range(3):
        for deg in range(MODEL_SYMMETRY_STEP_DEG, 181, MODEL_SYMMETRY_STEP_DEG):
            turned = (points - c) @ _rotation(np.radians(deg) * B[:, k]).T + c
            out[k] = max(out[k], float(np.percentile(tree.query(turned)[0], MODEL_SYMMETRY_PERCENTILE)))
    return out


def _ends_alike(points, box, extent):
    """Whether a long mesh's two ends are alike (MODEL_ENDS_DISTINCT_SHARE);
    False for a mesh not MODEL_ENDS_ELONGATION times longer than wide."""
    order = np.argsort(extent)
    if extent[order[2]] < MODEL_ENDS_ELONGATION * extent[order[1]]:
        return False
    B, c = box[:3, :3], box[:3, 3]
    width = extent[order[1]]
    along = (points - c) @ B[:, order[2]]
    ends = np.percentile(along, [100 - MODEL_ENDS_TIP, MODEL_ENDS_TIP])
    past = np.array([ends[0] - along.min(), along.max() - ends[1]])
    if past.max() >= MODEL_ENDS_DISTINCT_SHARE * width and past.min() <= MODEL_ENDS_STICK_SHARE * past.max():
        return False

    def bands(P):
        """end_signature()'s: per end, each band's tip short of the furthest."""
        a = (P - c) @ B[:, order[2]]
        band = np.floor(((P - c) @ B[:, order[1]] + width / 2) / width * MODEL_ENDS_BANDS).astype(int)
        tips = np.full((2, MODEL_ENDS_BANDS), np.nan)
        for e, sign in enumerate((-1.0, 1.0)):
            for b in range(MODEL_ENDS_BANDS):
                inside = sign * a[band == b]
                if len(inside) >= 10:
                    tips[e, b] = np.percentile(inside, MODEL_ENDS_TIP)
            if np.isfinite(tips[e]).any():
                tips[e] -= np.nanmax(tips[e])
        return tips

    own, apart = bands(points), np.inf
    for k in order[:2]:
        # Turned end for end about either shorter axis.
        flip = B @ np.diag([1.0 if j == k else -1.0 for j in range(3)]) @ B.T
        turned = bands((points - c) @ flip.T + c)
        both = np.isfinite(own) & np.isfinite(turned)
        if both.any():
            apart = min(apart, float(np.mean(np.abs(own[both] - turned[both]))))
    return bool(apart < MODEL_ENDS_DISTINCT_SHARE * width)


class ObjectModel:
    """A mesh reduced to what the fit needs, in the mesh's frame and metres:
    surface samples with their normals, and the mesh's own minimal box."""

    def __init__(self, vertices, faces, rng=None):
        rng = rng or np.random.default_rng(0)
        v = np.asarray(vertices, np.float64).reshape(-1, 3)
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        if not len(v) or not len(f) or not np.all(np.isfinite(v)):
            raise ValueError("empty or non-finite mesh")
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError("face index out of range")
        tri = v[f]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
        if double_area.sum() <= 0:
            raise ValueError("mesh has no area")
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
        # Only the line a normal lies on is used, never which way it points: a
        # reconstructed mesh's faces are not reliably wound one way.
        self.normals = cross[face] / np.maximum(double_area[face, None], 1e-15)
        self.spacing = float(np.sqrt(double_area.sum() / 2 / MODEL_SAMPLES))
        face = rng.choice(len(f), MODEL_FEATURE_SAMPLES, p=double_area / double_area.sum())
        r1, r2 = np.sqrt(rng.random(MODEL_FEATURE_SAMPLES)), rng.random(MODEL_FEATURE_SAMPLES)
        a, b, c = tri[face, 0], tri[face, 1], tri[face, 2]
        self.feature_points = ((1 - r1)[:, None] * a + (r1 * (1 - r2))[:, None] * b
                               + (r1 * r2)[:, None] * c)
        self.box = np.eye(4)
        self.box[:3, :3], self.box[:3, 3] = R, centre
        self.extent = extent
        # How round the mesh is about each box axis, and whether its ends
        # are alike; see MODEL_SYMMETRY_STEP_DEG and MODEL_ENDS_DISTINCT_SHARE.
        self.twist_spread = _twist_spread(self.points, self.box)
        self.ends_alike = _ends_alike(self.feature_points, self.box, self.extent)
        used, inverse = np.unique(f, return_inverse=True)
        self.model_id = hashlib.sha256(v[used].astype("<f4").tobytes()
                                      + inverse.reshape(-1, 3).astype("<i4").tobytes()).hexdigest()
        self.fitted_pose = None
        self.fitted_scale = None
        self.fitted_linear = None
        self.fitted_quality = None
        self.fitted_measured = False
        # The fit turned half round about its length, relative to it, where
        # the view fits that about as well (MODEL_SPIN_COST_SLACK); and an id
        # of the fit, new with every fit, for what is said of it to hold to.
        self.fitted_spin_doubt = None
        self.fitted_ends_doubt = None
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
        across = s[[j for j in range(3) if j != k]]
        if across.max() > MODEL_SYMMETRY_STRETCH * across.min():
            continue
        spread = model.twist_spread[k] * across.max()
        if spread <= MODEL_TOL_M and (best is None or spread < best[0]):
            best = (spread, k)
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

    def __init__(self, model, view, K, rng, through=None, scale_hint=None, ignore=None, located=None,
                 image=None):
        cam, mask, keep, plane, _ = view
        # The frame's grey levels, for what the depth does not show (end_streak()).
        self.gray = None if image is None else cv2.cvtColor(np.asarray(image, np.uint8),
                                                            cv2.COLOR_RGB2GRAY).astype(np.float64)
        self.scale_hint = scale_hint
        # The scale, along the mesh's box axes, it was located at this init
        # (MODEL_LOCATED_FALLBACK).
        self.located = located
        # Pixels that say nothing of this part's ends: the robot's, the others'.
        self.ignore = ignore
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
        if use_foreground and np.any(mask & ~keep):
            self.observation_note += ", support removed"
        if np.any(keep & ~obs_mask):
            self.observation_note += ", isolated depth components removed"
        self.obs = obs[rng.permutation(len(obs))[:MODEL_FINE_POINTS]]
        self.obs_tree = cKDTree(self.obs)
        self._obb_points = cam[obs_mask if use_foreground else keep]
        self._obb = None
        self._mask_distance = None
        self._stick_points = None

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
        span = np.percentile(along, 98, axis=0) - np.percentile(along, 2, axis=0) - MODEL_COVER_NOISE_M
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
        blocked = (d > 0) & (d < P[idx, 2] - MODEL_HIDDEN_M)
        if self.plane is not None:
            # Beside the mask at the mesh's depth, off the support: an occluder
            # touching it, as terms() counts it.
            n, p0 = self.plane
            blocked |= (~self.grown[v, u] & (d > 0) & (np.abs(d - P[idx, 2]) <= MODEL_TOL_M)
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
                nearer = (d > 0) & (d < probe[inside, 2] - MODEL_HIDDEN_M) & ~self.mask[v, u]
                if self.plane is not None:
                    n, p0 = self.plane
                    nearer &= (self.cam[v, u] - p0) @ n < -PLANE_TOL_M
                if nearer.mean() > MODEL_END_HIDDEN:
                    out[k] = False
        return out

    def lines_of_sight(self, P, seen):
        """The seen mesh points' indices and pixels, and the observed depth
        there (0 where there is none)."""
        h, w = self.depth.shape
        uv = np.round(_pixels(P, self.K)).astype(np.int64)
        idx = np.flatnonzero(seen & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                             & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        u, v = uv[idx, 0], uv[idx, 1]
        return idx, u, v, self.depth[v, u]

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
            idx = idx[(d <= 0) | (d >= P[idx, 2] - MODEL_HIDDEN_M)]
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
        open_ = (d > 0) & (d >= P[idx, 2] - MODEL_HIDDEN_M)
        # Outside the mask, off the support and no further than the mesh
        # point: hidden too.
        unmasked = np.zeros(len(idx), bool)
        if self.plane is not None:
            n, p0 = self.plane
            unmasked = (open_ & ~in_mask & (d <= P[idx, 2] + MODEL_TOL_M)
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
        bad_b = ~in_mask | (np.abs(d - P[idx, 2]) > MODEL_TOL_M)
        out["pairs"].append((P[idx], target, N[idx], dist_b))

        # Mesh points under the support surface.
        below = np.zeros(len(P))
        if self.plane is not None:
            n, p0 = self.plane
            below = np.maximum((P - p0) @ n, 0.0)
        out["below"] = below

        sat = lambda dist: np.minimum(dist / MODEL_TOL_M, 1.0) ** 2
        out["cost"] = (sat(dist_a).mean()
                       + (np.where(bad_b, 1.0, sat(dist_b)).mean() if len(idx) else 0.0)
                       + sat(below).mean())
        out["fitness"] = float(np.mean(dist_a < MODEL_TOL_M))
        out["contradiction"] = float(bad_b.mean()) if len(idx) else 0.0
        out["penetration"] = float(np.mean(below > MODEL_TOL_M))
        out["open_points"] = len(idx)
        out["p95_distance"] = float(np.percentile(dist_a, 95))
        out["width"] = float(np.sort(np.broadcast_to(scale, (3,)) * self.model.extent)[1])
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
            R, t, _ = self.step(R, t, terms, max(MODEL_TOL_M, radius * 0.75 ** i))
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
        allows, and the starts left all fitted it too short."""
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
                                   MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
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
                   and np.linalg.norm(t - t2) < MODEL_TOL_M
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
            hinted = False
            if (terms["hidden"] > MODEL_SCALE_MAX_HIDDEN and self.scale_hint is not None
                    and abs(np.log(_overall(s) / self.scale_hint)) > np.log(MODEL_SCALE_FINE_RATIO)):
                shape = s * (self.scale_hint / _overall(s)) if np.ndim(s) else self.scale_hint
                mean = self.model.points.mean(0)
                centre = R @ (self.linear(s) @ mean) + t
                Rh, th, h_terms = self.icp(shape, R, centre - R @ (self.linear(shape) @ mean), MODEL_FINE_ROUNDS,
                                           MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
                if h_terms is not None:
                    Rh, th, h_terms, _ = self.polish(shape, Rh, th, h_terms)
                    if not model_pose_rejection(h_terms, coarse=True):
                        R, t, terms, scale, s = Rh, th, h_terms, shape, shape
                        terms["measured"] = False
                        note, hinted = f"not measured, {terms['hidden']:.0%} hidden", True
            located = False
            if (MODEL_LOCATED_FALLBACK and not hinted and terms["hidden"] > MODEL_SCALE_MAX_HIDDEN
                    and self.located is not None and model_pose_rejection(terms, coarse=True)):
                # Neither its start nor the others' scale: the proportions it
                # was located at (MODEL_LOCATED_FALLBACK).
                shape = np.asarray(self.located, np.float64)
                shape = float(shape) if np.ndim(shape) == 0 else shape
                mean = self.model.points.mean(0)
                centre = R @ (self.linear(s) @ mean) + t
                Rl, tl, l_terms = self.icp(shape, R, centre - R @ (self.linear(shape) @ mean), MODEL_FINE_ROUNDS,
                                           MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
                if l_terms is not None:
                    Rl, tl, l_terms, _ = self.polish(shape, Rl, tl, l_terms)
                    if not model_pose_rejection(l_terms, coarse=True):
                        R, t, terms, scale, s = Rl, tl, l_terms, shape, shape
                        terms["measured"] = False
                        note, located = f"not measured, {terms['hidden']:.0%} hidden", True
            if terms["hidden"] <= MODEL_SCALE_MAX_HIDDEN:
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
                return self.settle_ends(terms, R, t, scale, quality)
            results.append((terms, R, t, scale, quality))
        # No pose fits well: the best coarse one, else the best, refused.
        coarse = [r for r in results if r[4] == "coarse"]
        best = min(coarse or results, key=lambda r: r[0]["cost"])
        return self.settle_ends(*best) if best[4] else best

    def settle_ends(self, terms, R, t, scale, quality):
        """The pose, or the same turned end for end where the view holds that
        better; see MODEL_ENDS_ELONGATION, MODEL_ENDS_STICK_MIN_M and
        MODEL_ENDS_REGION_NOISE. Where it cannot tell, terms["ends_doubt"] is
        the other way round's pose."""
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        order = np.argsort(extent)
        if extent[order[2]] < MODEL_ENDS_ELONGATION * extent[order[1]]:
            return terms, R, t, scale, quality
        if getattr(self.model, "ends_alike", False):
            return self.settle_alike_ends(terms, R, t, scale, quality)
        stick = self.end_out(R, t, scale)
        said = lambda past: (f"the part sticks out {past[1]} {past[3]} past the end its mesh does "
                             f"and {past[2]} past the other")
        if stick is not None and stick[0] > 0:
            terms["scale_note"] = terms.get("scale_note", "") + f"; ends kept: {said(stick)}"
            return terms, R, t, scale, quality
        # A half turn about the box centre, about the shortest box axis or
        # the middle one: each turns the part end for end. Only about the
        # middle one, the face the camera sees turns away too: the tester
        # upright in PP-S1-1, its probe up, cost 0.076 more so, and was left
        # upside down; turned about its shortest axis, 0.014 more. Of the
        # turns within MODEL_ENDS_COST_SLACK, the one the view tells apart
        # from the pose most clearly decides, not the cheaper: lying in
        # PP-S1-2, its probe toward the camera, the tester turned about its
        # middle axis, cheaper, could not be told apart, and the profile
        # left it the wrong way round.
        B = self.model.box[:3, :3]
        M = self.linear(scale)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        # stuck: the turns the part sticks out as their meshes do, where it
        # sticks out past the other end of the pose's.
        turns, cheapest, stuck = [], None, []
        for axis in order[:2]:
            flip = B @ np.diag([1.0 if k == axis else -1.0 for k in range(3)]) @ B.T
            R2 = R @ flip
            R2, t2, terms2 = self.icp(scale, R2, centre - R2 @ (M @ self.model.box[:3, 3]),
                                      MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
            if terms2 is None:
                continue
            R2, t2, terms2, _ = self.polish(scale, R2, t2, terms2)
            quality2 = ("fit" if not model_pose_rejection(terms2)
                        else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
            if cheapest is None or terms2["cost"] < cheapest[0]["cost"]:
                cheapest = (terms2, quality2)
            extra = terms2["cost"] - terms["cost"]
            if (stick is not None and stick[0] < 0 and quality2 is not None
                    and not (quality == "fit" and quality2 != "fit") and extra <= MODEL_ENDS_STICK_COST_SLACK):
                past = self.end_out(R2, t2, scale)
                if past is not None and past[0] > 0:
                    stuck.append((R2, t2, terms2, quality2, extra, past))
            if (quality2 is None or (quality == "fit" and quality2 != "fit")
                    or extra > MODEL_ENDS_CLEAR_COST_SLACK):
                continue
            # (gain of the turned pose, cue, kept's support, turned's) of the
            # cue that tells them apart the most.
            # (gain of the turned pose, the kept's support, the turned's).
            told = None
            here_s, there_s = self.difference_support(R, t, R2, t2, scale)
            if here_s is not None and there_s is not None:
                told = (there_s - here_s, here_s, there_s)
            elif here_s is not None or there_s is not None:
                # What one puts in view tells on its own: held, it is the
                # part's; not held, the other, which puts its own behind
                # something, is. Held only where it stands out of the other's
                # outline, as a probe does: inside it, it is a cut-out of the
                # other's a mask may have filled, and said nothing -- the
                # sample was turned the wrong way round on 68-94% held of
                # what the right way has in its notch, that one's own hidden
                # (PP-S1-3, PP-S2-1, PP-S4-2; 2026-10-27).
                shown = there_s if here_s is None else here_s
                filled = self.difference_inside[1 if here_s is None else 0]
                if not (shown >= 0.5 and filled is not None and filled[0] >= MODEL_ENDS_INSIDE_SHARE
                        and filled[1] <= MODEL_ENDS_FILL_MAX_PX):
                    told = ((2 * shown - 1) * (1 if here_s is None else -1), here_s, there_s)
            if extra > MODEL_ENDS_COST_SLACK and not (
                    told is not None and told[0] > MODEL_ENDS_REGION_MARGIN
                    and self.profile_mismatch(R2, t2, scale)
                    < MODEL_ENDS_MARGIN * self.profile_mismatch(R, t, scale)):
                continue
            turns.append((R2, t2, terms2, quality2, told, extra))
        if stuck:
            # Which face of it turns to the camera the cost says.
            R2, t2, terms2, quality2, extra, past = min(stuck, key=lambda turn: turn[2]["cost"])
            why = said(past) + (f", though it costs {extra:.3f} more" if extra > MODEL_ENDS_COST_SLACK else "")
            terms2["scale_note"] = terms.get("scale_note", "") + f"; turned end for end: {why}"
            terms2["measured"] = terms.get("measured", False)
            return terms2, R2, t2, scale, quality2
        if not turns:
            why = "no fit turned end for end"
            if cheapest is not None:
                terms2, quality2 = cheapest
                why = ("turned end for end, " + (f"its fit is {quality2 or 'refused'}" if quality2 != quality
                                                 else f"it costs {terms2['cost'] - terms['cost']:.3f} more"))
            terms["scale_note"] = terms.get("scale_note", "") + f"; ends kept: {why}"
            return terms, R, t, scale, quality
        clear = [turn for turn in turns if turn[4] is not None and abs(turn[4][0]) > MODEL_ENDS_REGION_MARGIN]
        other = None
        if clear:
            # A turn the view prefers to the pose is taken, the most preferred;
            # one it prefers the pose to says nothing of another turn.
            ahead = [turn for turn in clear if turn[4][0] > 0]
            R2, t2, terms2, quality2, told, extra = max(ahead or clear, key=lambda turn: abs(turn[4][0]))
            gain, held, held2 = told
            turn = gain > 0
            show = lambda value: "hidden" if value is None else f"{value:.0%}"
            why = (f"the mask holds what it has and the other way round has not "
                   f"{show(held2 if turn else held)} against {show(held if turn else held2)}")
            if turn and extra > MODEL_ENDS_COST_SLACK:
                why += f", and its profile, though it costs {extra:.3f} more"
        here_sig = None if clear else self.end_signature(R, t, scale)
        signed = [] if here_sig is None else [
            (turn, (here_sig, there_sig)) for turn in turns
            for there_sig in [self.end_signature(turn[0], turn[1], scale)]
            if there_sig is not None and abs(here_sig - there_sig) > MODEL_ENDS_SIGNATURE_MIN_M
            and min(here_sig, there_sig) < MODEL_ENDS_SIGNATURE_RATIO * max(here_sig, there_sig)]
        if signed:
            (R2, t2, terms2, quality2, told, extra), sig = max(signed, key=lambda item: abs(item[1][0] - item[1][1]))
            turn = sig[1] < sig[0]
            why = (f"its ends' shape {1000 * sig[turn]:.1f} mm off the observed against "
                   f"{1000 * sig[not turn]:.1f} mm")
        elif not clear:
            R2, t2, terms2, quality2, told, extra = min(turns, key=lambda turn: turn[2]["cost"])
            here, there = self.profile_mismatch(R, t, scale), self.profile_mismatch(R2, t2, scale)
            turn = there < MODEL_ENDS_MARGIN * here
            why = (f"profile mismatch {there:.2f} against {here:.2f}" if turn
                   else f"profile mismatch {here:.2f} against {there:.2f}")
            why += "; in doubt, the view not telling the two ways round apart"
            other = np.eye(4)
            other[:3, :3], other[:3, 3] = (R, t) if turn else (R2, t2)
        if turn:
            terms2["scale_note"] = terms.get("scale_note", "") + f"; turned end for end: {why}"
            terms2["measured"] = terms.get("measured", False)
            if other is not None:
                terms2["ends_doubt"] = other
            return terms2, R2, t2, scale, quality2
        terms["scale_note"] = terms.get("scale_note", "") + f"; ends kept: {why}"
        if other is not None:
            terms["ends_doubt"] = other
        return terms, R, t, scale, quality

    def settle_alike_ends(self, terms, R, t, scale, quality):
        """settle_ends() for a mesh whose ends are alike (ObjectModel.ends_alike):
        the pose or the same turned end for end, whichever fits better, and of
        two as good, the one with the mesh's long box axis to the image's right
        (MODEL_ENDS_LIKE_UP: up); the other always in doubt. The half turn is
        about the shorter box axis nearer the line of sight, which keeps the
        side the camera sees in view."""
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        order = np.argsort(extent)
        B = self.model.box[:3, :3]
        M = self.linear(scale)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        sight = centre / max(np.linalg.norm(centre), 1e-9)
        k = max(order[:2], key=lambda j: abs((R @ B[:, j]) @ sight))
        R2 = R @ B @ np.diag([1.0 if j == k else -1.0 for j in range(3)]) @ B.T
        t2 = centre - R2 @ (M @ self.model.box[:3, 3])
        rank = {"fit": 2, "coarse": 1, None: 0}
        Rf, tf, terms2 = self.icp(scale, R2, t2, MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS,
                                  2 * MODEL_TOL_M)
        quality2 = None
        if terms2 is not None:
            Rf, tf, terms2, _ = self.polish(scale, Rf, tf, terms2)
            quality2 = ("fit" if not model_pose_rejection(terms2)
                        else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
        if quality2 is not None:
            R2, t2 = Rf, tf                   # else the mesh's own turn, for the doubt alone
        # The long box axis, signed so its largest part in the mesh's frame is +.
        a = B[:, order[2]] * np.sign(B[int(np.argmax(np.abs(B[:, order[2]]))), order[2]])

        def facing(Rx):
            v = Rx @ a
            return v[0] if abs(v[0]) >= MODEL_ENDS_LIKE_UP else -v[1]

        side = "to the image's right" if abs((R @ a)[0]) >= MODEL_ENDS_LIKE_UP else "up"
        turn = quality2 is not None and (rank[quality2] > rank[quality]
                                         or (rank[quality2] == rank[quality] and facing(R2) > facing(R)))
        if quality2 is None or rank[quality2] < rank[quality]:
            why = f"kept, turned end for end its fit is {quality2 or 'refused'}"
        elif rank[quality2] > rank[quality]:
            why = f"turned end for end, its fit {quality2} against {quality}"
        else:
            why = f"{'turned end for end' if turn else 'kept'}, its long axis {side}"
        why = f"; ends alike in the mesh, {why}; the other way round in doubt"
        other = np.eye(4)
        if turn:
            other[:3, :3], other[:3, 3] = R, t
            terms2["scale_note"] = terms.get("scale_note", "") + why
            terms2["measured"] = terms.get("measured", False)
            terms2["ends_doubt"] = other
            return terms2, R2, t2, scale, quality2
        other[:3, :3], other[:3, 3] = R2, t2
        terms["scale_note"] = terms.get("scale_note", "") + why
        terms["ends_doubt"] = other
        return terms, R, t, scale, quality

    def end_out(self, R, t, scale):
        """end_stick(), with what it counted ("points"), or where that does not
        tell, end_streak() ("pixels of colour"); see MODEL_ENDS_STREAK_STEP_M."""
        stick = self.end_stick(R, t, scale)
        if stick is not None and stick[0] == 0:
            streak = self.end_streak(R, t, scale)
            if streak is not None and streak[0] != 0:
                return streak + ("pixels of colour",)
        return None if stick is None else stick + ("points",)

    def end_streak(self, R, t, scale):
        """end_stick() in the image's colour: (sign, line pixels past the end
        the mesh sticks out of, past the other), or None without an image or
        where the mesh sticks out of neither end; see MODEL_ENDS_STREAK_STEP_M."""
        if self.gray is None:
            return None
        M = self.linear(scale)
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        axis = R @ self.model.box[:3, int(np.argmax(extent))]
        centre = R @ (M @ self.model.box[:3, 3]) + t
        P = self.model.feature_points @ (R @ M).T + t
        along = (P - centre) @ axis
        ends = np.percentile(along, [100 - MODEL_ENDS_TIP, MODEL_ENDS_TIP])
        out = np.array([ends[0] - along.min(), along.max() - ends[1]])
        if out.max() < MODEL_ENDS_STICK_MIN_M or out.min() > MODEL_ENDS_STICK_SHARE * out.max():
            return None
        h, w = self.gray.shape
        side, band = MODEL_ENDS_STREAK_SIDE_PX, MODEL_ENDS_STREAK_BAND_PX
        counts = []
        for e, sign in ((0, -1.0), (1, 1.0)):
            n = 0
            for s in np.arange(MODEL_ENDS_STICK_FROM_M, MODEL_ENDS_STICK_REACH * out.max(), MODEL_ENDS_STREAK_STEP_M):
                q = centre + (ends[e] + sign * s) * axis
                uv, ahead = _pixels(np.stack([q, q + sign * 0.01 * axis]), self.K)
                step = ahead - uv
                if q[2] <= 0 or np.linalg.norm(step) < 1e-9:
                    continue
                across = np.array([-step[1], step[0]]) / np.linalg.norm(step)
                u, v = np.round(uv).astype(np.int64)
                if not (0 <= u < w and 0 <= v < h) or (self.ignore is not None and self.ignore[v, u]):
                    continue
                if 0 < self.depth[v, u] < q[2] - MODEL_ENDS_STREAK_FRONT_M:
                    continue                      # something in front: it says nothing
                for off in range(-band, band + 1):
                    c = np.round(uv + off * across).astype(np.int64)
                    a, b = np.round(c + side * across).astype(np.int64), np.round(c - side * across).astype(np.int64)
                    if not all(0 <= x[0] < w and 0 <= x[1] < h for x in (a, b, c)):
                        continue
                    ga, gb = self.gray[a[1], a[0]], self.gray[b[1], b[0]]
                    if (abs(ga - gb) < MODEL_ENDS_STREAK_CONTRAST
                            and abs(self.gray[c[1], c[0]] - (ga + gb) / 2) >= MODEL_ENDS_STREAK_CONTRAST):
                        n += 1
                        break
            counts.append(n)
        mine, other = counts[int(np.argmax(out))], counts[1 - int(np.argmax(out))]
        most = max(mine, other)
        if most < MODEL_ENDS_STICK_COUNT or min(mine, other) > MODEL_ENDS_STICK_SHARE * most:
            return 0, mine, other
        return (1 if mine > other else -1), mine, other

    def end_stick(self, R, t, scale):
        """(sign, points past the end the mesh sticks out of, past the other)
        of the pose: sign 1 where the part sticks out past the end its mesh
        does, -1 past the other, 0 where the view does not tell; None where
        the mesh sticks out of neither end more than the other. See
        MODEL_ENDS_STICK_MIN_M."""
        M = self.linear(scale)
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        axis = R @ self.model.box[:3, int(np.argmax(extent))]
        centre = R @ (M @ self.model.box[:3, 3]) + t
        P = self.model.feature_points @ (R @ M).T + t
        along = (P - centre) @ axis
        radius = np.percentile(np.linalg.norm((P - centre) - np.outer(along, axis), axis=1), MODEL_ENDS_TIP)
        ends = np.percentile(along, [100 - MODEL_ENDS_TIP, MODEL_ENDS_TIP])
        out = np.array([ends[0] - along.min(), along.max() - ends[1]])     # past the end at -axis, +axis
        if out.max() < MODEL_ENDS_STICK_MIN_M or out.min() > MODEL_ENDS_STICK_SHARE * out.max():
            return None
        if self._stick_points is None:
            seen = (self.depth > 0) & np.isfinite(self.cam).all(-1)
            if self.ignore is not None:
                seen &= ~self.ignore
            points = self.cam[seen]
            if self.plane is not None:
                n, p = self.plane
                points = points[(points - p) @ n <= -MODEL_ENDS_STICK_HEIGHT_M]
            self._stick_points = points
        O = self._stick_points
        s = (O - centre) @ axis
        near = np.linalg.norm((O - centre) - np.outer(s, axis), axis=1) <= radius + MODEL_ENDS_STICK_MARGIN_M
        reach = MODEL_ENDS_STICK_REACH * out.max()
        past = (int(np.count_nonzero(near & (s < ends[0] - MODEL_ENDS_STICK_FROM_M) & (s > ends[0] - reach))),
                int(np.count_nonzero(near & (s > ends[1] + MODEL_ENDS_STICK_FROM_M) & (s < ends[1] + reach))))
        mine, other = past[int(np.argmax(out))], past[1 - int(np.argmax(out))]
        most = max(mine, other)
        if most < MODEL_ENDS_STICK_COUNT or min(mine, other) > MODEL_ENDS_STICK_SHARE * most:
            return 0, mine, other
        return (1 if mine > other else -1), mine, other

    def difference_support(self, R, t, R2, t2, scale):
        """(this pose's, the other's): how well the mask holds what each of
        two poses has where the other has not -- its feature points off the
        other's by more than MODEL_ENDS_REGION_NOISE times the median -- where
        that pose puts them, 0-1; None for one with fewer than
        MODEL_ENDS_REGION_MIN of those on the side the camera sees, in view
        and not behind something nearer. See MODEL_ENDS_REGION_NOISE.
        self.difference_inside is left (this pose's, the other's): (the share
        of those pixels inside the other pose's silhouette's convex hull, how
        many pixels they cover) -- a cut-out the other has there, which a mask
        may have filled (MODEL_ENDS_INSIDE_SHARE); None with the support."""
        if self._mask_distance is None:
            self._mask_distance = cv2.distanceTransform(
                (~self.mask).astype(np.uint8), cv2.DIST_L2, 5)
        X = self.model.feature_points
        M = self.linear(scale)
        spacing = _overall(scale) * self.model.spacing * np.sqrt(MODEL_SAMPLES / len(X))
        P, P2 = X @ (R @ M).T + t, X @ (R2 @ M).T + t2
        off, off2 = cKDTree(P2).query(P)[0], cKDTree(P).query(P2)[0]
        tol = max(MODEL_ENDS_REGION_NOISE * np.median(np.concatenate([off, off2])), 2 * spacing)
        out, inside = [], []
        for Q, own, other in ((P, off > tol, P2), (P2, off2 > tol, P)):
            seen = own & _seen_from_camera(Q, self.K, spacing, self.depth.shape)
            idx, u, v, d = self.lines_of_sight(Q, seen)
            # Behind something nearer, a point is hidden -- unless it is in
            # the support: a probe turned down into the bench is the pose's
            # fault, and holds nothing.
            under = np.zeros(len(idx), bool)
            if self.plane is not None:
                n, p0 = self.plane
                under = (Q[idx] - p0) @ n > PLANE_TOL_M
            front = ((d > 0) & (d >= Q[idx, 2] - MODEL_HIDDEN_M)) | under
            if front.sum() < MODEL_ENDS_REGION_MIN:
                out.append(None)
                inside.append(None)
                continue
            held = 1 - np.minimum(self._mask_distance[v[front], u[front]] / MODEL_ENDS_REGION_PX, 1)
            held[under[front]] = 0.0
            out.append(float(np.mean(held)))
            # Where the other pose's outline, cut-outs filled, has them.
            seen_other = _seen_from_camera(other, self.K, spacing, self.depth.shape)
            _, uo, vo, _ = self.lines_of_sight(other, seen_other)
            share = None
            if len(uo) >= 3:
                hull = np.zeros(self.depth.shape, np.uint8)
                cv2.fillConvexPoly(hull, cv2.convexHull(np.column_stack([uo, vo]).astype(np.int32)), 1)
                share = float(hull[v[front], u[front]].mean())
            pixels = len({(a, b) for a, b in zip(u[front].tolist(), v[front].tolist())})
            inside.append(None if share is None else (share, pixels))
        self.difference_inside = tuple(inside)
        return tuple(out)

    def end_signature(self, R, t, scale):
        """How unlike the observed points the mesh is in the shape of the ends
        of its long axis, in metres: the width cut into MODEL_ENDS_BANDS
        bands, each band's end (MODEL_ENDS_TIP percentile along the axis)
        short of that end's furthest band, the observed against the mesh's,
        the mean over the bands both show at both ends; None where none.
        Each end is measured on its own points against itself, so a pose a
        pixel or two off does not change it. See MODEL_ENDS_REGION_NOISE.
        An end hidden (MODEL_END_HIDDEN) is left out: where a hand or the
        image's edge cuts the part, the observed points stop at the cut, and
        that is the cut's shape, not the end's -- the bar's hidden end in the
        hand "matched" one way round, 5.2 mm against 26.8."""
        M = self.linear(scale)
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        order = np.argsort(extent)
        hidden = self.hidden_ends(R, t, scale)[order[2]]
        axis, across = R @ self.model.box[:3, order[2]], R @ self.model.box[:3, order[1]]
        centre = R @ (M @ self.model.box[:3, 3]) + t
        half = extent[order[1]] / 2

        def signature(points):
            along = (points - centre) @ axis
            band = np.floor(((points - centre) @ across + half) / (2 * half) * MODEL_ENDS_BANDS).astype(int)
            out = np.full((2, MODEL_ENDS_BANDS), np.nan)
            for e, sign in enumerate((-1.0, 1.0)):
                for b in range(MODEL_ENDS_BANDS):
                    inside = sign * along[band == b]
                    if len(inside) >= 10:
                        out[e, b] = np.percentile(inside, MODEL_ENDS_TIP)
                out[e] -= np.nanmax(out[e]) if np.isfinite(out[e]).any() else 0.0
            return out

        observed = signature(self.obs)
        mesh = signature(self.model.feature_points @ (R @ M).T + t)
        observed[hidden] = np.nan
        both = np.isfinite(observed) & np.isfinite(mesh)
        if not both.any():
            return None
        return float(np.mean(np.abs(observed[both] - mesh[both])))

    def settle_spin(self, terms, R, t, scale, quality):
        """(terms, R, t, scale, quality, doubt): the pose, or the same turned
        half round about its length where that costs MODEL_SPIN_COST_SLACK
        less; doubt, the other way round as a transform of the scaled mesh
        relative to the pose, where the two cost within that, else None."""
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        order = np.argsort(extent)
        if extent[order[2]] < MODEL_ENDS_ELONGATION * extent[order[1]]:
            return terms, R, t, scale, quality, None
        if round_axis(self.model, scale) == order[2]:
            return terms, R, t, scale, quality, None        # no other way round: MODEL_SYMMETRY_STEP_DEG
        # A half turn about the long box axis, about the box centre.
        B = self.model.box[:3, :3]
        flip = B @ np.diag([1.0 if k == order[2] else -1.0 for k in range(3)]) @ B.T
        M = self.linear(scale)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        R2 = R @ flip
        R2, t2, terms2 = self.icp(scale, R2, centre - R2 @ (M @ self.model.box[:3, 3]),
                                  MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
        if terms2 is None:
            return terms, R, t, scale, quality, None
        R2, t2, terms2, _ = self.polish(scale, R2, t2, terms2)
        quality2 = ("fit" if not model_pose_rejection(terms2)
                    else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
        # Refitted back round to the pose, it is no other way round.
        turned = np.degrees(np.arccos(np.clip((np.trace(R.T @ R2) - 1) / 2, -1, 1)))
        if quality2 is None or (quality == "fit" and quality2 != "fit") or turned < 90:
            return terms, R, t, scale, quality, None
        if terms2["cost"] < terms["cost"] - MODEL_SPIN_COST_SLACK:
            terms2["scale_note"] = (terms.get("scale_note", "") + f"; turned half round about its length: "
                                    f"cost {terms2['cost']:.3f} against {terms['cost']:.3f}")
            terms2["measured"] = terms.get("measured", False)
            return terms2, R2, t2, scale, quality2, None
        if terms2["cost"] > terms["cost"] + MODEL_SPIN_COST_SLACK:
            return terms, R, t, scale, quality, None
        # The cost does not tell; a line along a recess may (MODEL_SPIN_RECESS_M).
        way, seen = self.sunk_way(R, t, R2, t2, scale)
        if way == 1:
            terms2["scale_note"] = terms.get("scale_note", "") + f"; turned half round about its length: {seen}"
            terms2["measured"] = terms.get("measured", False)
            return terms2, R2, t2, scale, quality2, None
        if way == 0:
            terms["scale_note"] = terms.get("scale_note", "") + f"; kept about its length: {seen}"
            return terms, R, t, scale, quality, None
        note = (f"; which way round about its length is in doubt: turned half round costs "
                f"{terms2['cost']:.3f} against {terms['cost']:.3f}")
        if self.plane is not None and (R2[:, 2] - R[:, 2]) @ -self.plane[0] > MODEL_SPIN_UP_MARGIN:
            terms2["scale_note"] = terms.get("scale_note", "") + note + ", taken with the mesh's z up"
            terms2["measured"] = terms.get("measured", False)
            terms, R, t, quality, R2, t2 = terms2, R2, t2, quality2, R, t
        else:
            terms["scale_note"] = terms.get("scale_note", "") + note
        pose, other = np.eye(4), np.eye(4)
        pose[:3, :3], pose[:3, 3] = R, t
        other[:3, :3], other[:3, 3] = R2, t2
        return terms, R, t, scale, quality, np.linalg.inv(pose) @ other

    def sunk_way(self, R, t, R2, t2, scale):
        """(0 for the pose (R, t), 1 for (R2, t2), or None, and why): which of
        two the image shows the part as, by a line along the band where one is
        sunk under the other; see MODEL_SPIN_RECESS_M."""
        if self.gray is None:
            return None, "no image"
        X = self.model.feature_points
        M = self.linear(scale)
        spacing = _overall(scale) * self.model.spacing * np.sqrt(MODEL_SAMPLES / len(X))
        P = [X @ (Ri @ M).T + ti for Ri, ti in ((R, t), (R2, t2))]
        off = [cKDTree(P[1]).query(P[0])[0], cKDTree(P[0]).query(P[1])[0]]
        tol = max(MODEL_ENDS_REGION_NOISE * np.median(np.concatenate(off)), 2 * spacing)
        h, w = self.depth.shape
        nearest, seen = [], []
        for Q in P:
            s = _seen_from_camera(Q, self.K, spacing, (h, w))
            uv = np.round(_pixels(Q[s], self.K)).astype(np.int64)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            z = np.full((h, w), np.inf)
            np.minimum.at(z, (uv[ok, 1], uv[ok, 0]), Q[s][ok, 2])
            nearest.append(z)
            seen.append(s)
        bands = []
        for i in (0, 1):
            Q = P[i][(off[i] > tol) & seen[i]]
            uv = np.round(_pixels(Q, self.K)).astype(np.int64)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            uv, z = uv[ok], Q[ok, 2]
            other = nearest[1 - i][uv[:, 1], uv[:, 0]]
            keep = np.isfinite(other) & (z > other + MODEL_SPIN_RECESS_M)
            keep &= self.depth[uv[:, 1], uv[:, 0]] >= np.minimum(z, other) - MODEL_ENDS_STREAK_FRONT_M
            if self.ignore is not None:
                keep &= ~self.ignore[uv[:, 1], uv[:, 0]]
            bands.append(np.unique(uv[keep], axis=0).astype(np.float64))
        i = int(len(bands[1]) > len(bands[0]))
        band = bands[i]
        if len(band) < MODEL_SPIN_BAND_PX:
            return None, f"no sunk band in view ({len(band)} px)"
        lined, length = self._lined(band)
        share = lined / max(length, 1)
        said = f"the band one way round is sunk lined on {lined} of {length} px"
        if share >= MODEL_SPIN_LINE_SHARE:
            return i, said + ", sunk so"
        if share <= MODEL_SPIN_FLAT_SHARE:
            return 1 - i, said + ", flat there"
        return None, said

    def _lined(self, band):
        """(pixels of `band`'s length with a line across it, its length); see MODEL_SPIN_RECESS_M."""
        c = band.mean(0)
        along = np.linalg.svd(band - c, full_matrices=False)[2][0]
        across = np.array([-along[1], along[0]])
        s, r = (band - c) @ along, (band - c) @ across
        h, w = self.gray.shape
        side, lined, length = MODEL_SPIN_LINE_SIDE_PX, 0, 0
        for t in np.arange(s.min() + 2, s.max() - 2, 1.0):
            length += 1
            for off in range(int(np.floor(r.min())) - side, int(np.ceil(r.max())) + side + 1):
                q = np.round(c + t * along + off * across).astype(np.int64)
                a, b = np.round(q + side * across).astype(np.int64), np.round(q - side * across).astype(np.int64)
                if not all(0 <= x[0] < w and 0 <= x[1] < h for x in (q, a, b)):
                    continue
                ga, gb = self.gray[a[1], a[0]], self.gray[b[1], b[0]]
                if (abs(ga - gb) < MODEL_SPIN_LINE_CONTRAST
                        and abs(self.gray[q[1], q[0]] - (ga + gb) / 2) >= MODEL_SPIN_LINE_CONTRAST):
                    lined += 1
                    break
        return lined, length

    def profile_mismatch(self, R, t, scale):
        """How unlike the observed points the mesh's seen side is in width
        along the mesh's long axis: the mean over the observed stretch of
        |log| of the width ratio, 1 where the mesh has no width to compare."""
        M = self.linear(scale)
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        axis = R @ self.model.box[:3, int(np.argmax(extent))]
        centre = R @ (M @ self.model.box[:3, 3]) + t
        P = self.model.points @ (R @ M).T + t
        P = P[_seen_from_camera(P, self.K, _overall(scale) * self.model.spacing, self.depth.shape)]

        def widths(points, edges):
            along = (points - centre) @ axis
            radial = np.linalg.norm((points - centre) - np.outer(along, axis), axis=1)
            index = np.digitize(along, edges) - 1
            out = np.full(len(edges) - 1, np.nan)
            for b in range(len(out)):
                if np.count_nonzero(index == b) >= 5:
                    out[b] = np.percentile(radial[index == b], 90)
            return out

        along = (self.obs - centre) @ axis
        edges = np.linspace(*np.percentile(along, [1, 99]), MODEL_PROFILE_BINS + 1)
        observed, mesh = widths(self.obs, edges), widths(P, edges)
        both = np.isfinite(observed)
        if not both.any():
            return 1.0
        ratio = np.abs(np.log(np.maximum(observed[both], 1e-4) / np.maximum(mesh[both], 1e-4)))
        return float(np.mean(np.where(np.isfinite(mesh[both]), np.minimum(ratio, 1.0), 1.0)))

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
                                     MODEL_SCALE_ROUNDS, mesh_n, obs_n, 2 * MODEL_TOL_M)
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


def fit_model_box(model, view, K, rng=None, through=None, scale_hint=None, ignore=None, located=None,
                  image=None):
    """Box of one object from its mesh: ((T_cam_box, extent, note), None) if
    the mesh fits the view, (None, reason) if not.

    The mesh is placed by ICP from a spread of starting rotations, each scored
    by what the camera sees along every mesh point's line of sight (see the
    comment above MODEL_SAMPLES). Its box then goes where the mesh went, at
    the mesh's size whatever share of the object was in view. `ignore`, the
    robot's and the other parts' pixels, is no evidence of which end the
    part sticks out of (MODEL_ENDS_STICK_MIN_M).
    """
    model.fitted_pose = model.fitted_scale = model.fitted_linear = None
    model.fitted_quality, model.fitted_measured = None, False
    model.fitted_spin_doubt = model.fitted_ends_doubt = model.fitted_id = None
    fit = _ModelFit(model, view, K, rng or np.random.default_rng(0), through, scale_hint, ignore, located, image)
    result = fit.run()
    if result is None:
        return None, "no pose left any mesh point in view"
    terms, R, t, scale, quality = result
    ends_other = terms.get("ends_doubt")
    doubt = None
    if quality is not None:
        terms, R, t, scale, quality, doubt = fit.settle_spin(terms, R, t, scale, quality)
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
    model.fitted_spin_doubt, model.fitted_id = doubt, uuid.uuid4().hex
    model.fitted_ends_doubt = None if ends_other is None else np.linalg.inv(model.fitted_pose) @ ends_other
    label = ("mesh pose fit" if quality == "fit"
             else f"coarse mesh pose fit ({model_pose_rejection(terms)})")
    return (T, np.broadcast_to(scale, (3,)) * model.extent, f"{label}, {stats}"), None


def retry_mesh_fit(model, mask, metres, K, through=None, scale_hint=None, ignore=None, located=None,
                   image=None):
    """Worker owns its model state and frame snapshot, never the live pipeline."""
    model = copy.copy(model)
    try:
        view = observe(mask, metres, K)
        if view is None:
            return None, "insufficient current depth", model
        fit, reason = fit_model_box(model, view, K, through=through, scale_hint=scale_hint, ignore=ignore,
                                    located=located, image=image)
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
    """"spin" where `pose` is the fitted mesh's pose turned
    MODEL_REFIT_TURN_DEG or more about the mesh's long box axis, "ends"
    where about another, else None -- also for a turn about the long axis
    of a mesh round about it (round_axis()), which is no other way round."""
    D = model.fitted_pose[:3, :3].T @ pose[:3, :3]
    if np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))) < MODEL_REFIT_TURN_DEG:
        return None
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


def correct_carried_pose(fit, model, pose):
    """The carried mesh pose corrected on this frame's view `fit`
    (carried_view()), and its cost there; `pose` itself where the correction
    gains less than MODEL_CORRECT_MIN_GAIN. See MODEL_CORRECT_ROUNDS."""
    scale = fitted_axis_scale(model)
    start = fit.terms(scale, pose[:3, :3], pose[:3, 3], *MODEL_CHECK_POINTS)
    if start is None:
        return pose, np.inf
    R, t, terms = fit.icp(scale, pose[:3, :3], pose[:3, 3], MODEL_CORRECT_ROUNDS,
                          *MODEL_CHECK_POINTS, MODEL_CORRECT_RADIUS_M)
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


def carried_fit_rejection(model, pose, mask, metres, K, through=None, fit=None):
    """Why the fitted mesh, at `pose` in this frame's camera (its fit carried
    here by the tracked motion), no longer lies on what the camera sees of
    the part; "" where it does, or where the view shows too little to say.
    `fit`: this frame's carried_view(), where already made.

    Only the observed points' side of the coarse acceptance is asked: their
    share on the mesh and their 95th-percentile distance. A few millimetres
    of tracked depth already put a thin part seen face-on past the coarse
    contradiction (a ring 5 mm deep: over 0.25), and a part held up has for
    its "support" whatever is behind it."""
    if fit is None:
        fit = carried_view(model, mask, metres, K, through)
    if fit is None:
        return ""
    terms = fit.terms(fitted_axis_scale(model), pose[:3, :3], pose[:3, 3], *MODEL_CHECK_POINTS)
    if terms is None:
        return "no mesh point in view"
    failed = []
    if terms["fitness"] < MODEL_COARSE_MIN_FITNESS:
        failed.append(f"fitness {terms['fitness']:.2f} < {MODEL_COARSE_MIN_FITNESS:.2f}")
    p95 = min(MODEL_COARSE_MAX_P95_DISTANCE_M,
              max(MODEL_COARSE_P95_FLOOR_M, MODEL_COARSE_MAX_P95_SHARE * terms["width"]))
    if terms["p95_distance"] > p95:
        failed.append(f"95th-percentile point distance {terms['p95_distance'] * 1000:.0f} mm > {p95 * 1000:.0f} mm")
    return "; ".join(failed)


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
# The meshes of one init come from one demonstration's reconstruction, at
# its scale: the parts' scales agree -- PP's tester and sample 2.4 both,
# RI's ring 0.85 and bar 0.75. A part located at a scale off the others' by
# more than this is located again among fits at theirs: PP-S4-1's tester,
# half behind the box, lost to a patch of the box's handle at 0.46, the
# sample at 2.41; at the sample's, its own proposal fitted at 2.21. One the
# others' scale finds none for keeps where it was. 1.5, not 2: RR-S1-1's ring,
# located on the hand of the person holding the bar, was at 1.23 to the bar's
# 0.75 at times (2026-10-31); parts of one reconstruction agree within 1.15.
LOCATE_SCALE_AGREE = 1.5
# A mesh's scale is its reconstruction's, the same from scene to scene: the
# ring measured 0.870-0.915 in RI-S1-1 to RR-S1-1. So the server keeps each
# mesh's measured scales -- fine fits whose size the view showed, the last
# LOCATE_KNOWN_SCALES by the mesh's id, for as long as it runs -- and a part
# too hidden to measure its own size (MODEL_SCALE_MAX_HIDDEN) takes their
# median (fit_model_box()'s scale_hint) before the other parts' scale. Not
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


def _quick_fit(model, view, K, screen=False, through=None):
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
    340 x 27, fits it at no one scale -- at 0.77, contradiction 0.28.
    `through`, pixels that join the view's pieces (_ModelFit's)."""
    fit = _ModelFit(model, view, K, np.random.default_rng(0), through=through)
    mesh_n, obs_n, rounds = LOCATE_SCREEN if screen else MODEL_COARSE
    starts = fit.start_scales()
    observed = float(np.clip(np.sort(fit.obb[2])[1] / np.sort(model.extent)[1], *MODEL_SCALE_RANGE))
    if not any(np.ndim(s) == 0 and np.isclose(s, observed) for s in starts):
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
                 terms["fitness"] - terms["contradiction"] - terms["penetration"]), (R, t, s))

    _, R, t, s, terms = best
    if screen:
        return scored(R, t, s, terms)
    R, t, terms = fit.icp(s, R, t, MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
    found = [scored(R, t, s, terms)] if terms is not None else []
    if uniform is not None:
        _, R, t, s, _ = uniform
        mean = model.points.mean(0)
        centre = R @ (s * mean) + t
        span = np.log(MODEL_SCALE_START_RATIO)
        for ratio in np.exp(np.linspace(-span, span, LOCATE_SCALE_STEPS)):
            si = float(np.clip(s * ratio, *MODEL_SCALE_RANGE))
            Ri, ti, termsi = fit.icp(si, R, centre - R @ (si * mean), MODEL_FINE_ROUNDS, MODEL_SAMPLES,
                                     MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
            if termsi is not None:
                found.append(scored(Ri, ti, si, termsi))
    if found and not max(found, key=lambda item: item[0])[0][0]:
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


def locate_with_mesh(model, proposals, metres, K, clicked=(), taken=None, robot=None, why=None,
                     rest=False, scales=None, grown=None):
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

    def fit_among(candidates, said_too=""):
        return _locate_fits(model, K, candidates, scales, why, said_too, throughs)

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
    note = (f"score {score:.2f}, scale {_overall(pose[2]):.2f}, best of {tried} fitted of "
            f"{len(screened)} screened of {len(shortlist)} shortlisted of {len(proposals)} proposals, "
            f"{on_robot} on the robot")
    return found[0], found[1], note, (score, _overall(pose[2]), pose[2])


def _locate_fits(model, K, candidates, scales, why, said_too="", throughs=None):
    """(best fit or None, proposals fitted, screened, the best's proposal) of
    `candidates` [(miss, index, view)] (locate_with_mesh()): rough fits rank
    them, the best LOCATE_FITS are fitted properly; `why` is told of each.
    `throughs`, by index, what joins a candidate's pieces (_quick_fit())."""
    throughs = throughs or {}
    screened, chosen = [], None
    for rank, (_, index, view) in enumerate(sorted(candidates, key=lambda item: item[0])):
        if rank >= LOCATE_SCREENS:
            why.append((index, "shortlisted, past the screening cap"))
            continue
        try:
            found = _quick_fit(model, view, K, screen=True, through=throughs.get(index))
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
            found = _quick_fit(model, view, K, through=throughs.get(index))
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
        if best is None or found[0] > best[0]:
            best, chosen = found, index
    return best, tried, screened, chosen if best is not None else None


def decode_models(meta, parts, count):
    """One ObjectModel or None per object, from the frames after rgb and depth.

    meta["models"] holds, per object, None or the vertex and face counts of
    one frame: zlib of little-endian float32 vertices then int32 faces. A
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
            if nv <= 0 or nf <= 0 or len(raw) != 12 * (nv + nf):
                raise ValueError("size does not match its counts")
            vertices = np.frombuffer(raw, "<f4", 3 * nv).reshape(nv, 3)
            faces = np.frombuffer(raw, "<i4", 3 * nf, offset=12 * nv).reshape(nf, 3)
            models[index] = ObjectModel(vertices, faces)
            size = " x ".join(f"{v * 1000:.0f}" for v in models[index].extent)
            print(f"[server] object {index}: mesh {nv} vertices, box {size} mm", flush=True)
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

    def log_gpu_memory(self, when):
        if torch.cuda.is_available():
            print(f"[server] GPU memory {when}: "
                  f"allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
                  f"reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB", flush=True)

    def build_pipeline(self):
        self.release_pipeline()
        self.log_gpu_memory("after release")
        self.pipeline = ModularPipeline(self.cfg)
        self.log_gpu_memory("after build")
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
        # Each part's hint: the other located parts' scale, for a part too
        # hidden to measure its own (MODEL_SCALE_MAX_HIDDEN).
        # Its own measured scale first (LOCATE_KNOWN_SCALES).
        scales = getattr(self, "locate_scales", {})
        self.scale_hints = []
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
                        ignore = np.delete(masks, index, axis=0).any(0)
                        for part in (cut[index], robot):
                            if part is not None:
                                ignore |= part
                        fit, reason = fit_model_box(models[index], view, K, through=cut[index],
                                                    scale_hint=self.scale_hints[index], ignore=ignore,
                                                    located=getattr(self, "locate_shapes", {}).get(index),
                                                    image=rgb)
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
            future, index, session, capture_inverse = job
            if session == self.mesh_session and tracking_lost[index]:
                # A loss at any point during fitting breaks motion propagation,
                # even if feature tracking recovers before the worker finishes.
                capture_inverse = None
                self.mesh_fit_job = (future, index, session, None)
            if not future.done():
                return
            self.mesh_fit_job = None
            if session == self.mesh_session:
                self.model_retry_after[index] = now + (MODEL_RETRY_INTERVAL_SEC
                                                       if self.fitted_models[index] is None
                                                       else MODEL_UPGRADE_INTERVAL_SEC)
                try:
                    fit, reason, model = future.result()
                    if capture_inverse is None:
                        fit, reason = None, "tracking lost during mesh localization; waiting for recovery"
                    current = self.fitted_models[index]
                    if fit is not None and (current is None or fit_rank(model) > fit_rank(current)):
                        T_box, extent, note = fit
                        model.fitted_pose = capture_inverse @ model.fitted_pose
                        turn = None if current is None else refit_turn(current, model.fitted_pose)
                        if turn is not None:
                            # The fit stays, the refit in doubt: see MODEL_REFIT_TURN_DEG.
                            setattr(current, f"fitted_{turn}_doubt",
                                    np.linalg.inv(current.fitted_pose) @ model.fitted_pose)
                            print(f"[server] object {index}: refit better but turned "
                                  f"{'half round about its length' if turn == 'spin' else 'end for end'}; "
                                  f"keeping the fit, the refit sent in doubt ({note})", flush=True)
                        else:
                            self.boxes[index] = (capture_inverse @ T_box, extent)
                            self.fitted_models[index] = model
                            self.remember_scale(model)
                            self.model_errors[index] = None
                            print(f"[server] object {index}: mesh "
                                  f"{'localized' if current is None else 'fit improved'} on retry ({note})",
                                  flush=True)
                    elif current is not None:
                        print(f"[server] object {index}: refit no better; keeping the fit", flush=True)
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
            ignore = np.delete(masks > 0, index, axis=0).any(0)
            robot = getattr(self, "track_robot", None)
            for part in (others_region(index, regions), robot if robot is not None and robot.shape == (h, w) else None):
                if part is not None:
                    ignore |= part
            if getattr(self, "mesh_fit_executor", None) is None:
                self.mesh_fit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh_pose")
            future = self.mesh_fit_executor.submit(
                retry_mesh_fit, self.models[index], mask, metres, np.array(frame.intrinsics, copy=True),
                others_region(index, regions), self.scale_hints[index], ignore,
                getattr(self, "locate_shapes", {}).get(index), np.array(frame.rgb, copy=True))
            self.mesh_fit_job = (future, index, self.mesh_session, capture_inverse)
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
        for index, model in enumerate(models):
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
                pose = start
                if fit is not None:
                    pose, cost = correct_carried_pose(fit, model, start)
                    move, turn = correction_size(model, carried, pose)
                    if move > MODEL_CORRECT_MAX_M or turn > MODEL_CORRECT_MAX_DEG:
                        tracked, tracked_cost = correct_carried_pose(fit, model, carried)
                        if not cost < tracked_cost - MODEL_CORRECT_MIN_GAIN:
                            pose = tracked
                corrections[index] = (fit_id, np.linalg.inv(carried) @ pose)
                why = carried_fit_rejection(model, pose, mask, metres, K, fit=fit) if fit is not None else ""
                if why:
                    corrections[index] = None
                else:
                    motions[index] = pose @ np.linalg.inv(model.fitted_pose)
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
                self.fitted_models[index] = None
                self.model_retry_after[index] = time.monotonic()
                self.model_errors[index] = (f"the carried mesh pose no longer fits the view ({why}); "
                                            "fitting it again")
                print(f"[server] object {index}: {self.model_errors[index]}", flush=True)
        return motions

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
                    doubt = getattr(model, "fitted_spin_doubt", None)
                    if doubt is not None:
                        # Relative to the fit, so carried by the motion with it.
                        result["model_spin_doubt"] = (np.linalg.inv(held) @ doubt @ held).tolist()
                    ends = getattr(model, "fitted_ends_doubt", None)
                    if ends is not None:
                        result["model_ends_doubt"] = (np.linalg.inv(held) @ ends @ held).tolist()
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
        """Keep a fine fit's scale where the view measured it; see LOCATE_KNOWN_SCALES."""
        if model is None or model.fitted_quality != "fit" or not model.fitted_measured:
            return
        known = getattr(self, "known_scales", None)
        if known is None:
            known = self.known_scales = {}
        known[model.model_id] = (known.get(model.model_id, []) + [float(model.fitted_scale)])[-LOCATE_KNOWN_SCALES:]

    def known_scale(self, model):
        """The median of a mesh's measured scales, or None."""
        scales = (getattr(self, "known_scales", None) or {}).get(getattr(model, "model_id", None))
        return float(np.median(scales)) if scales else None

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
        robot = decode_robot_mask(meta, depth.shape)
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
        self.record_surfaces(motions, frame, self.carried_regions(motions, frame))
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
        shapes = {}               # each located part's scale along its box axes
        while pending:
            round_ += 1
            progress = False
            for index in list(pending):
                why = []
                click, region, note, fit = locate_with_mesh(models[index], proposals, metres, K,
                                                            clicked, taken, robot, why, rest, grown=grown)
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
                pending.remove(index)
                progress = True
                points[index], labels[index] = click, [1] * len(click)
                self.say(f"object {index}: located by its mesh at {click} ({note})")
            if not progress:
                if rest or not any((proposal & taken).any() for proposal in proposals):
                    break
                # Last, what is left of the proposals other objects took.
                rest = True
        if pending:
            index = pending[0]
            raise ValueError(f"object {index}: not located by its mesh ({notes[index]}); click it")
        # The one whose scale is furthest off the others' (LOCATE_SCALE_AGREE),
        # the lower scored of two as far off, is located again at theirs. One
        # kept at its own, none fitting at theirs, is the one in doubt: it sets
        # no scale for the others. RR-S1-1's ring, located on the hand at 1.55,
        # was kept so; then the bar, fitted right at 0.59, was located again at
        # the ring's, on the hand at the image's edge (2026-10-31).
        again, kept = set(), set()
        while True:
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
                (at / LOCATE_SCALE_AGREE, at * LOCATE_SCALE_AGREE), grown=others_grown)
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
            points[index], labels[index] = click, [1] * len(click)
            self.say(f"object {index}: located again by its mesh at {click}, at the others' scale "
                     f"({note})")
        # One located weakly is looked for again once the others are, what is
        # left of their proposals proposed too, at their scale; where it fits
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
                (at / LOCATE_SCALE_AGREE, at * LOCATE_SCALE_AGREE), grown=grown_now)
            if click is None or fit[0] < score + LOCATE_BETTER:
                continue
            for which, said in sorted(why):
                self.say(f"  object {index}, looked for again ({score:.2f}), {sources[which]} "
                         f"({describe_proposal(proposals[which])}): {said}")
            located[index] = region
            fitted[index] = fit[:2] + (True,)
            shapes[index] = fit[2]
            kept.discard(index)             # at the others' scale now
            points[index], labels[index] = click, [1] * len(click)
            self.say(f"object {index}: located again by its mesh at {click}, fitting better than at "
                     f"{score:.2f} ({note})")
        self.locate_scales = {index: fit[1] for index, fit in fitted.items()}
        self.locate_shapes = shapes
        self.say(f"located by mesh in {time.time() - started:.1f} s")
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
        self.retry_pending_meshes(motions, frame, regions)
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
                    self.release_pipeline()
                    self.log_gpu_memory("after reset")
                    self.sock.send_json({"ok": True})
                    continue
                if cmd == "preview":
                    self.sock.send_json({"ok": True, **self.handle_preview(meta, msg[1:])})
                    continue

                if cmd == "init":
                    poses, frame, motions = self.handle_init(meta, msg[1:])
                    self.log_gpu_memory("after init")
                elif cmd == "track":
                    poses, frame, motions = self.handle_track(meta, msg[1:])
                else:
                    raise ValueError(f"unknown cmd: {cmd}")

                lost = [bool(getattr(o, "lost", False)) for o in self.pipeline.objects]
                bboxes = self.bboxes(motions, frame)
                lost = self.recognition_lost(lost, bboxes)
                self.sock.send_json(
                    {
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
                )
            except Exception as exc:
                traceback.print_exc()
                self.sock.send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                     **({"init_report": self.init_report} if cmd == "init" else {})})
            finally:
                self.init_report = None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/pipeline/pipeline_test2.yaml")
    ap.add_argument("-b", "--bind", default="tcp://0.0.0.0:5555")
    args = ap.parse_args()
    server = PoseServer(args.config, args.bind)
    try:
        server.run()
    finally:
        if server.mesh_fit_executor is not None:
            server.mesh_fit_executor.shutdown(wait=True, cancel_futures=True)
