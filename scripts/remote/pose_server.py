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
fitting its pose to the observations. The mesh supplies the part's shape, so
the box keeps its proportions; one uniform scale is fitted too, since a
similar mesh need not be the part's size. A mesh that already is keeps
exactly scale 1.
A failed pose fit stays invalid while subsequent frames are retried in the
background. Point-cloud boxes are used only when no mesh was supplied.
"""

import argparse
import base64
import copy
import hashlib
import itertools
import json
import time
import zlib
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
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
# An axis is stretched only where the view spans it: the observed points
# cover at least this share of the mesh's extent along it. A bar seen from
# one side shows half its round, and its depth stretched to fit that half.
MODEL_STRETCH_SEEN = 0.7
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
# too, and kept if it costs at most MODEL_ENDS_COST_SLACK more and its
# profile -- how wide the part is along its length, where the camera sees
# it -- matches the observed one better by MODEL_ENDS_MARGIN. A profile is
# robust to what a reconstruction gets wrong: size, proportions, a probe a
# few millimetres off.
MODEL_ENDS_ELONGATION = 1.5
MODEL_ENDS_COST_SLACK = 0.05
MODEL_ENDS_MARGIN = 0.8
MODEL_PROFILE_BINS = 16
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
# A measured scale replaces 1 only if it lowers the cost by this much. Too big
# a mesh shows at once, as open points off the part; too small a one costs
# only the few points past its edges, so noise alone can favour a slightly
# smaller one. A mesh that is the part's size so keeps exactly 1, which the
# agent's demonstration grasps require of a native mesh pose.
MODEL_SCALE_MIN_GAIN = 0.03
# Retry failed localization on fresh frames, with one background fit at a time.
MODEL_RETRY_INTERVAL_SEC = 3.0
# A coarse fit, or one whose size the view hid too much to measure, is fitted
# again on later views this often, and replaced only by a better one -- fine
# over coarse, measured over not. A fine, measured fit is kept: the part's
# size is the part's, and a later view half hidden cannot measure it better.
MODEL_UPGRADE_INTERVAL_SEC = 6.0


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
        self.box = np.eye(4)
        self.box[:3, :3], self.box[:3, 3] = R, centre
        self.extent = extent
        used, inverse = np.unique(f, return_inverse=True)
        self.model_id = hashlib.sha256(v[used].astype("<f4").tobytes()
                                      + inverse.reshape(-1, 3).astype("<i4").tobytes()).hexdigest()
        self.fitted_pose = None
        self.fitted_scale = None
        self.fitted_linear = None
        self.fitted_quality = None
        self.fitted_measured = False


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


def connected_depth_foreground(mask, depth):
    """Remove small mask islands separated by a depth jump from the main part.

    A 2D connected mask can cross an occluder's boundary. Use adjacent valid
    pixels whose depth differs by at most 3 cm as graph edges. If no component
    dominates, retain the evidence for the pose checks instead of choosing one.
    """
    count = int(mask.sum())
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
    if not len(sizes) or sizes.max() < max(MIN_BOX_POINTS, MODEL_COMPONENT_MIN_SHARE * count):
        return mask
    result = np.zeros_like(mask)
    result[mask] = labels == sizes.argmax()
    return result


class _ModelFit:
    """Fits one mesh to one object's view; see fit_model_box()."""

    def __init__(self, model, view, K, rng):
        cam, mask, keep, plane, _ = view
        self.model, self.K, self.cam = model, K, cam
        self.depth = cam[..., 2]
        kernel = np.ones((2 * MODEL_MASK_GROW_PX + 1,) * 2, np.uint8)
        self.grown = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        self.plane = None
        if plane is not None:
            n, p = plane
            # A plane the object is behind is an occluder the ring caught, not
            # its support; it would push the mesh out of the right pose.
            if np.mean((cam[mask] - p) @ n <= MODEL_TOL_M) >= 0.9:
                self.plane = (n, p)
        # A ring's hole and mask spill can include a large patch of table.
        # For a thick enough mesh, use the foreground already isolated by
        # observe(). For very thin parts, that cut selects positive depth
        # noise and biases the fit upwards, so retain the full mask there.
        use_foreground = (self.plane is not None
                          and model.extent.min() > 2 * PLANE_TOL_M)
        obs_mask = keep if use_foreground else mask
        if use_foreground:
            obs_mask = connected_depth_foreground(obs_mask, self.depth)
        obs = cam[obs_mask]
        self.observation_note = f"{len(obs)}/{int(mask.sum())} depth points"
        if use_foreground and np.any(mask & ~keep):
            self.observation_note += ", support removed"
        if np.any(keep & ~obs_mask):
            self.observation_note += ", isolated depth components removed"
        self.obs = obs[rng.permutation(len(obs))[:MODEL_FINE_POINTS]]
        self.obs_tree = cKDTree(self.obs)
        self.obb = min_obb(cam[obs_mask if use_foreground else keep])

    def linear(self, scale):
        """The mesh-frame linear map of a scale: one number scales the mesh
        uniformly, three scale it along its own box axes (see
        MODEL_STRETCH_MAX)."""
        f = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        B = self.model.box[:3, :3]
        return (B * f) @ B.T

    def cover_seen(self, R, t, terms, scale, note):
        """The pose, its scale raised, uniformly, until the mesh spans the
        observed points along each of its axes; see MODEL_COVER_NOISE_M."""
        extent = np.broadcast_to(np.asarray(scale, np.float64), (3,)) * self.model.extent
        along = self.obs @ (R @ self.model.box[:3, :3])
        span = np.percentile(along, 98, axis=0) - np.percentile(along, 2, axis=0) - MODEL_COVER_NOISE_M
        factor = float(np.max(span / extent))
        if factor <= MODEL_SCALE_FINE_RATIO:
            return R, t, terms, scale, note
        raised = np.asarray(scale, np.float64) * factor
        raised = float(raised) if np.ndim(raised) == 0 else raised
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
        return R2, t2, moved, raised, note + f"; raised {factor:.2f} times to cover what was seen"

    def axes_seen(self, R, t, scale):
        """Per mesh box axis, whether the view shows the part's length along
        it: the observed points span MODEL_STRETCH_SEEN of the mesh's extent
        there, and both ends show (ends_seen()). A mesh stretched past what
        was seen spans less of it, and fails."""
        scale = np.broadcast_to(np.asarray(scale, np.float64), (3,))
        along = self.obs @ (R @ self.model.box[:3, :3])
        span = np.percentile(along, 95, axis=0) - np.percentile(along, 5, axis=0)
        return (span >= MODEL_STRETCH_SEEN * scale * self.model.extent) & self.ends_seen(R, t, scale)

    def ends_seen(self, R, t, scale):
        """Per mesh box axis, whether the view shows both of the mesh's ends
        along it; see MODEL_END_HIDDEN."""
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
        out = np.ones(3, bool)
        for k in range(3):
            along = (P - centre) @ (R @ self.model.box[:3, k])
            lo, hi = np.quantile(along, [MODEL_END_SHARE, 1 - MODEL_END_SHARE])
            for end in (along <= lo, along >= hi):
                front = end & seen
                if front.sum() >= 10 and hidden[front].mean() > MODEL_END_HIDDEN:
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

    def step(self, R, t, terms, radius):
        """One Gauss-Newton step of the pose, about the mesh's centre."""
        P = terms["P"]
        c = P.mean(0)
        H, g = np.zeros((6, 6)), np.zeros(6)
        lam = MODEL_POINT_WEIGHT
        for p, q, n, dist in terms["pairs"]:
            use = dist < radius
            if not use.any():
                continue
            p, q, n = p[use], q[use], n[use]
            d, r = p - c, p - q
            J = np.zeros((len(p), 3, 6))
            J[:, 0, 1], J[:, 0, 2], J[:, 1, 0] = d[:, 2], -d[:, 1], -d[:, 2]
            J[:, 1, 2], J[:, 2, 0], J[:, 2, 1] = d[:, 0], d[:, 1], -d[:, 0]
            J[:, :, 3:] = np.eye(3)
            nJ = np.einsum("ki,kij->kj", n, J)
            k = 1.0 / len(p)
            H += k * (lam * np.einsum("kij,kil->jl", J, J) + (1 - lam) * nJ.T @ nJ)
            g += k * (lam * np.einsum("kij,ki->j", J, r)
                      + (1 - lam) * nJ.T @ np.einsum("ki,ki->k", n, r))
        below = terms["below"] > 0
        if below.any():
            n = self.plane[0]
            a = np.hstack([np.cross(P[below] - c, n), np.tile(n, (below.sum(), 1))])
            k = 1.0 / len(P)
            H += k * a.T @ a
            g += k * a.T @ terms["below"][below]
        if not np.any(H):
            return R, t
        xi = -np.linalg.solve(H + (1e-6 * np.trace(H) / 6 + 1e-12) * np.eye(6), g)
        E = _rotation(xi[:3])
        return E @ R, E @ (t - c) + c + xi[3:]

    def icp(self, scale, R, t, rounds, mesh_n, obs_n, radius):
        for i in range(rounds):
            terms = self.terms(scale, R, t, mesh_n, obs_n)
            if terms is None:
                return R, t, None
            R, t = self.step(R, t, terms, max(MODEL_TOL_M, radius * 0.75 ** i))
        return R, t, self.terms(scale, R, t, mesh_n, obs_n)

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

    def candidates(self):
        """Starting rotations: the mesh's box axes laid on the observed box's
        axes every way round, and -- the observed box tilts when part of the
        object is hidden -- each box axis standing on the support surface at
        a range of headings."""
        box = self.model.box[:3, :3]
        R_obs, _, extent_obs = self.obb
        out = [R_obs @ P @ box.T for P in _signed_permutations()]
        if self.plane is not None:
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
        pose search keeps whichever fits best."""
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
            if _stretch_allowed(f):
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
                # -- a bar's depth, seen from one side -- the part takes the
                # overall size of the rest, about the same centre.
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
                # Then its proportions, on the same clear enough view.
                Rp, tp, p_terms, p_scale = self.polish(scale, R, t, terms, free_axes=True)
                if p_terms["cost"] < terms["cost"] - MODEL_STRETCH_MIN_GAIN:
                    R, t, terms, scale = Rp, tp, p_terms, p_scale
                    note += f"; proportions measured, {_describe(scale)}"
                # Measured only if, as fitted, the two longest lengths show their ends.
                longest = np.argsort(np.broadcast_to(scale, (3,)) * self.model.extent)[1:]
                ends = self.ends_seen(R, t, scale)[longest].all()
                terms["measured"] = ts_terms is not None and not edge and bool(ends)
                if ts_terms is not None and not edge and not ends:
                    note += "; not counted as measured: an end of the part is hidden"
            if not terms["measured"]:
                R, t, terms, scale, note = self.cover_seen(R, t, terms, scale, note)
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
        """The pose, or the same turned end for end where that matches the
        observed profile better; see MODEL_ENDS_ELONGATION."""
        extent = np.broadcast_to(scale, (3,)) * self.model.extent
        order = np.argsort(extent)
        if extent[order[2]] < MODEL_ENDS_ELONGATION * extent[order[1]]:
            return terms, R, t, scale, quality
        # A half turn about the middle box axis, about the box centre.
        B = self.model.box[:3, :3]
        flip = B @ np.diag([1.0 if k == order[1] else -1.0 for k in range(3)]) @ B.T
        M = self.linear(scale)
        centre = R @ (M @ self.model.box[:3, 3]) + t
        R2 = R @ flip
        R2, t2, terms2 = self.icp(scale, R2, centre - R2 @ (M @ self.model.box[:3, 3]),
                                  MODEL_FINE_ROUNDS, MODEL_SAMPLES, MODEL_FINE_POINTS, 2 * MODEL_TOL_M)
        if terms2 is None:
            return terms, R, t, scale, quality
        R2, t2, terms2, _ = self.polish(scale, R2, t2, terms2)
        quality2 = ("fit" if not model_pose_rejection(terms2)
                    else "coarse" if not model_pose_rejection(terms2, coarse=True) else None)
        if (quality2 is None or (quality == "fit" and quality2 != "fit")
                or terms2["cost"] > terms["cost"] + MODEL_ENDS_COST_SLACK):
            return terms, R, t, scale, quality
        here, there = self.profile_mismatch(R, t, scale), self.profile_mismatch(R2, t2, scale)
        if there < MODEL_ENDS_MARGIN * here:
            terms2["scale_note"] = (terms.get("scale_note", "") + f"; turned end for end: profile "
                                    f"mismatch {there:.2f} against {here:.2f}")
            terms2["measured"] = terms.get("measured", False)
            return terms2, R2, t2, scale, quality2
        terms["scale_note"] = (terms.get("scale_note", "")
                               + f"; ends kept: profile mismatch {here:.2f} against {there:.2f}")
        return terms, R, t, scale, quality

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


def fit_model_box(model, view, K, rng=None):
    """Box of one object from its mesh: ((T_cam_box, extent, note), None) if
    the mesh fits the view, (None, reason) if not.

    The mesh is placed by ICP from a spread of starting rotations, each scored
    by what the camera sees along every mesh point's line of sight (see the
    comment above MODEL_SAMPLES). Its box then goes where the mesh went, at
    the mesh's size whatever share of the object was in view.
    """
    model.fitted_pose = model.fitted_scale = model.fitted_linear = None
    model.fitted_quality, model.fitted_measured = None, False
    fit = _ModelFit(model, view, K, rng or np.random.default_rng(0))
    result = fit.run()
    if result is None:
        return None, "no pose left any mesh point in view"
    terms, R, t, scale, quality = result
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
    label = ("mesh pose fit" if quality == "fit"
             else f"coarse mesh pose fit ({model_pose_rejection(terms)})")
    return (T, np.broadcast_to(scale, (3,)) * model.extent, f"{label}, {stats}"), None


def retry_mesh_fit(model, mask, metres, K):
    """Worker owns its model state and frame snapshot, never the live pipeline."""
    model = copy.copy(model)
    try:
        view = observe(mask, metres, K)
        if view is None:
            return None, "insufficient current depth", model
        fit, reason = fit_model_box(model, view, K)
        return fit, reason, model
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", model


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

        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(bind)
        print(f"[server] listening on {bind}", flush=True)

    def build_pipeline(self):
        self.pipeline = ModularPipeline(self.cfg)
        self.surfaces = []
        self.frame_id = 0
        self.anchors = None
        self.boxes = None
        self.fitted_models = None
        self.model_required = None
        self.model_errors = None
        self.bbox_warned = False

    def fit_boxes(self, rgb, metres, K, points, labels, models, model_required=None):
        """Each object's box on the init frame, fit to its mask; None where it cannot be.

        A requested mesh gives the box when its pose fits the view, finely or
        coarsely. When it does not, the object is still tracked, by the box of
        its points, and the mesh is retried on later frames; the reply then
        carries no mesh pose for it, so the Agent knows it has only the box.
        """
        self.mesh_session = getattr(self, "mesh_session", 0) + 1
        self.models = list(models)
        self.fitted_models = [None] * len(points)
        self.model_required = (list(model_required) if model_required is not None
                               else [model is not None for model in models])
        self.model_errors = [None] * len(points)
        self.model_retry_after = [time.monotonic() + MODEL_RETRY_INTERVAL_SEC] * len(points)
        try:
            # Same call as a preview, made on the freshly built pipeline before
            # add_user_points(), which is the state a preview sees.
            logits = self.pipeline.preview_user_masks(rgb, points, labels)
            masks = (as_numpy(logits) > 0).reshape(len(points), *metres.shape)
        except Exception:
            print("[server] no init masks; mesh objects rejected, no boxes", flush=True)
            self.model_errors = ["no usable initial mask" if required else None
                                 for required in self.model_required]
            traceback.print_exc()
            return [None] * len(points)
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
                print(f"[server] object {index}: supplied mesh could not be loaded; "
                      "tracking its points' box", flush=True)
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
                        fit, reason = fit_model_box(models[index], view, K)
                    except Exception as exc:
                        traceback.print_exc()
                        reason = f"{type(exc).__name__}: {exc}"
                self.model_retry_after[index] = time.monotonic() + MODEL_RETRY_INTERVAL_SEC
                if fit is None:
                    self.model_errors[index] = reason
                    print(f"[server] object {index}: mesh pose unlocalized ({reason}); "
                          "tracking its points' box, retrying the mesh on fresh frames", flush=True)
                    if view is not None:
                        try:
                            fit = fit_box(mask, metres, K, view=view)
                        except Exception:
                            traceback.print_exc()
                else:
                    self.fitted_models[index] = models[index]
                    print(f"[server] object {index}: mesh fit in "
                          f"{time.time() - started:.1f} s", flush=True)
            elif view is not None:
                try:
                    fit = fit_box(mask, metres, K, view=view)
                except Exception:
                    traceback.print_exc()
            if fit is None:
                print(f"[server] object {index}: no box", flush=True)
                boxes.append(None)
                continue
            T, extent, note = fit
            size = " x ".join(f"{v * 1000:.0f}" for v in extent)
            print(f"[server] object {index}: box {size} mm ({note})", flush=True)
            boxes.append((T, extent))
        return boxes

    def retry_pending_meshes(self, motions, frame):
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
                        self.boxes[index] = (capture_inverse @ T_box, extent)
                        self.fitted_models[index] = model
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
            mask = (masks[index] > 0).copy()
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
                self.mesh_fit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh_pose")
            future = self.mesh_fit_executor.submit(
                retry_mesh_fit, self.models[index], mask, metres, np.array(frame.intrinsics, copy=True))
            self.mesh_fit_job = (future, index, self.mesh_session, capture_inverse)
            print(f"[server] object {index}: retrying mesh pose on frame {frame.id}", flush=True)
        except Exception as exc:
            reason = f"mesh retry waiting: {exc}"
            if self.fitted_models[index] is None:
                if self.model_errors[index] != reason:
                    print(f"[server] object {index}: {reason}", flush=True)
                self.model_errors[index] = reason

    def record_surfaces(self, motions, frame):
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
                view = observe(np.asarray(masks[index]) > 0, metres, K, rng)
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
                if model is not None and model.fitted_pose is not None:
                    linear = getattr(model, "fitted_linear", None)
                    result.update(model_pose=(motion @ model.fitted_pose).tolist(),
                                  model_scale=model.fitted_scale, model_id=model.model_id,
                                  model_linear=(np.asarray(linear) if linear is not None
                                                else model.fitted_scale * np.eye(3)).tolist(),
                                  model_quality=getattr(model, "fitted_quality", None) or "fit",
                                  model_measured=bool(getattr(model, "fitted_measured", True)))
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

    def recognition_lost(self, tracking_lost, bboxes):
        """Lost: its tracks are, or it has no box at all. An object whose mesh
        has not fitted yet is not lost: it has the box of its points, and its
        reply says it has no mesh pose."""
        return [bool(lost or bboxes[index] is None) for index, lost in enumerate(tracking_lost)]

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
        models = decode_models(meta, parts[2:], len(meta["points"]))
        specs = meta.get("models") or []
        required = [index < len(specs) and specs[index] is not None
                    for index in range(len(meta["points"]))]
        self.boxes = self.fit_boxes(rgb, depth / depth_factor, K, meta["points"], meta["labels"],
                                    models, required)
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
        self.record_surfaces(motions, frame)
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
        self.retry_pending_meshes(motions, frame)
        if self.frame_id % SURFACE_EVERY == 0:
            self.record_surfaces(motions, frame)
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
                lost = self.recognition_lost(lost, bboxes)
                self.sock.send_json(
                    {
                        "ok": True,
                        "frame_id": self.frame_id - 1,
                        "poses": np.asarray(poses, dtype=np.float64).tolist(),
                        "lost": lost,
                        "bboxes": bboxes,
                        "model_errors": self.model_errors,
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
    server = PoseServer(args.config, args.bind)
    try:
        server.run()
    finally:
        if server.mesh_fit_executor is not None:
            server.mesh_fit_executor.shutdown(wait=True, cancel_futures=True)
